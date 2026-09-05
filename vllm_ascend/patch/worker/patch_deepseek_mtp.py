import contextvars
import re
from collections.abc import Callable, Generator, Iterable

import torch
import torch.nn as nn
import vllm
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm.config import VllmConfig
from vllm.model_executor.model_loader import weight_utils as vllm_weight_utils
from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP, DeepSeekMultiTokenPredictorLayer
from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader

MTP_ROT_WEIGHT_NAME = "rot.weight"

_BACKBONE_LAYER_RE = re.compile(r"^(?:model\.)?layers\.(\d+)\.")

_EXTRA_LAYER_FILTER: contextvars.ContextVar[Callable[[str], bool] | None] = (
    contextvars.ContextVar("vllm_ascend_glm_extra_layer_filter", default=None)
)


def _make_extra_layer_predicate(num_hidden_layers: int) -> Callable[[str], bool]:
    def should_skip_layer(weight_name: str) -> bool:
        match = _BACKBONE_LAYER_RE.match(weight_name)
        return match is not None and int(match.group(1)) >= num_hidden_layers

    return should_skip_layer


def _filter_extra_checkpoint_layers(
    weights: Iterable[tuple[str, torch.Tensor]],
    num_hidden_layers: int,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Drop checkpoint layer weights beyond the layers configured in the model.

    A reduced GLM-5.2 config (e.g. ``num_hidden_layers=3`` for single-node,
    no-PP debugging) still points at a checkpoint containing all 78 backbone
    layers plus MTP/nextn layers. Those weights have no destination module and
    would otherwise fail inside ``DeepseekV2Model.load_weights``, so filter
    them out before ``AutoWeightsLoader`` dispatches them.
    """
    should_skip_layer = _make_extra_layer_predicate(num_hidden_layers)
    for name, weight in weights:
        if not should_skip_layer(name):
            yield name, weight


_ORIGINAL_SHOULD_SKIP_WEIGHT = vllm_weight_utils.should_skip_weight


def _patched_should_skip_weight(
    weight_name: str,
    local_expert_ids: set[int] | None,
) -> bool:
    """Extend the safetensors pre-decode skip hook with the layer slice filter.

    ``safetensors_weights_iterator`` calls this helper *before*
    ``f.get_tensor``, so skipping extra checkpoint layers here also avoids
    reading their tensor bodies from disk. The extra filter only applies while
    ``_EXTRA_LAYER_FILTER`` is set by the reduced GLM-5.2 model loader.
    """
    if _ORIGINAL_SHOULD_SKIP_WEIGHT(weight_name, local_expert_ids):
        return True
    should_skip_layer = _EXTRA_LAYER_FILTER.get()
    return should_skip_layer is not None and should_skip_layer(weight_name)


vllm_weight_utils.should_skip_weight = _patched_should_skip_weight


def get_spec_layer_idx_from_weight_name(config: DeepseekV2Config | DeepseekV3Config, weight_name: str) -> int | None:
    if hasattr(config, "num_nextn_predict_layers") and config.num_nextn_predict_layers > 0:
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if (
                weight_name.startswith(f"model.layers.{layer_idx + i}.")
                or weight_name.startswith(MTP_ROT_WEIGHT_NAME)
                or weight_name.startswith(f"layers.{layer_idx + i}.")
            ):
                return layer_idx + i
    return None


class AscendDeepSeekMultiTokenPredictorLayer(DeepSeekMultiTokenPredictorLayer):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__(vllm_config, prefix)
        quant_description = getattr(vllm_config.quant_config, "quant_description", None)
        self.is_rot_used = quant_description.get("is_rot_used", False) if quant_description is not None else False
        self.target_model_type = vllm_config.speculative_config.target_model_config.hf_text_config.model_type
        if self.is_rot_used and self.target_model_type == "glm_moe_dsa":
            self.rot = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        # masking inputs at position 0, as not needed by MTP
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        inputs_embeds = self.enorm(inputs_embeds)
        if self.is_rot_used and self.target_model_type == "glm_moe_dsa":
            previous_hidden_states = self.rot(previous_hidden_states)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states = self.eh_proj(torch.cat([inputs_embeds, previous_hidden_states], dim=-1))

        hidden_states, residual = self.mtp_block(positions=positions, hidden_states=hidden_states, residual=None)
        hidden_states = residual + hidden_states  # pre-final-norm (logits hidden)
        # Recycle the post-final-norm hidden into the next draft step.
        # compute_logits applies shared_head (== final norm) to the pre-norm
        # element, so logits and the recycle each get exactly one final-norm.
        # Matches SGLang's deepseek_nextn.
        return hidden_states, self.shared_head(hidden_states)


class AscendDeepSeekMTP(DeepSeekMTP):
    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        if name != MTP_ROT_WEIGHT_NAME:
            return super()._rewrite_spec_layer_name(spec_layer, name)
        else:
            return f"model.layers.{spec_layer}.rot.weight"


class AscendGlmMoeDsaForCausalLM(GlmMoeDsaForCausalLM):
    def load_weights(self, weights):
        num_hidden_layers = getattr(self.config, "num_hidden_layers", None)
        if num_hidden_layers is None:
            return AutoWeightsLoader(self, skip_prefixes=[MTP_ROT_WEIGHT_NAME]).load_weights(weights)

        weights = _filter_extra_checkpoint_layers(weights, num_hidden_layers)
        token = _EXTRA_LAYER_FILTER.set(_make_extra_layer_predicate(num_hidden_layers))
        try:
            loader = AutoWeightsLoader(self, skip_prefixes=[MTP_ROT_WEIGHT_NAME])
            return loader.load_weights(weights)
        finally:
            _EXTRA_LAYER_FILTER.reset(token)


vllm.model_executor.models.deepseek_v2.get_spec_layer_idx_from_weight_name = get_spec_layer_idx_from_weight_name
vllm.model_executor.models.deepseek_mtp.get_spec_layer_idx_from_weight_name = get_spec_layer_idx_from_weight_name
vllm.model_executor.models.deepseek_mtp.DeepSeekMultiTokenPredictorLayer = AscendDeepSeekMultiTokenPredictorLayer
vllm.model_executor.models.deepseek_mtp.DeepSeekMTP = AscendDeepSeekMTP
vllm.model_executor.models.deepseek_v2.GlmMoeDsaForCausalLM = AscendGlmMoeDsaForCausalLM
