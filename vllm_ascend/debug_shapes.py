"""
Shape debug helper for DeepSeek V4 inference on Ascend NPU.

Usage:
    export VLLM_DEBUG_SHAPES=1          # enable shape logging
    export VLLM_DEBUG_SHAPES_STEPS=3    # only log first N steps (default 3)
    export VLLM_DEBUG_SHAPES_LAYERS=2   # only log first N layers (default all)
    export VLLM_DEBUG_SHAPES_WEIGHTS=1  # also log weight shapes during loading

Then run with --enforce-eager for full coverage.

Effectiveness tags:
    [EAGER] - always works (pure Python, no graph capture)
    [WARMUP] - only during NPUGraph warmup (not replayed in production)
    [DEAD] - inside torch.ops._C_ascend.* C++ custom op, NEVER fires from Python
    [GRAPH] - inside ACLGraph capture scope, fires in warmup + eager, not in replay
"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger("vllm_ascend.debug_shapes")

# ── env-controlled switches ──────────────────────────────────────────
_DEBUG_SHAPES: bool = os.environ.get("VLLM_DEBUG_SHAPES", "0") == "1"
_DEBUG_WEIGHTS: bool = os.environ.get("VLLM_DEBUG_SHAPES_WEIGHTS", "0") == "1"
_MAX_STEPS: int = int(os.environ.get("VLLM_DEBUG_SHAPES_STEPS", "3"))
_MAX_LAYERS: int = int(os.environ.get("VLLM_DEBUG_SHAPES_LAYERS", "-1"))  # -1 = all
_MAX_WEIGHT_PRINTS: int = int(os.environ.get("VLLM_DEBUG_SHAPES_MAX_WEIGHTS", "500"))

_step_counter: int = 0


def _active(layer_idx: int | None = None) -> bool:
    """Check whether shape logging is active for this layer."""
    if not _DEBUG_SHAPES:
        return False
    if _MAX_LAYERS >= 0 and layer_idx is not None and layer_idx >= _MAX_LAYERS:
        return False
    return True


def tensor_shape(x) -> str:
    """Safe shape string: 'list(shape)' or 'scalar' or 'None'."""
    if x is None:
        return "None"
    if isinstance(x, torch.Tensor):
        sh = list(x.shape)
        dt = str(x.dtype).replace("torch.", "")
        return f"{sh} {dt}"
    if isinstance(x, (list, tuple)):
        shapes = [tensor_shape(t) for t in x]
        return "[" + ", ".join(shapes[:6]) + (", ..." if len(shapes) > 6 else "") + "]"
    if isinstance(x, dict):
        items = [f"{k}={tensor_shape(v)}" for k, v in list(x.items())[:6]]
        return "{" + ", ".join(items) + "}"
    return str(type(x).__name__)


def log_shape(tag: str, x, layer_idx: int | None = None,
              effectiveness: str = "[EAGER]") -> None:
    """Log a single tensor shape.

    Args:
        tag: human-readable label, e.g. "hidden_states after embed"
        x: tensor or nested structure
        layer_idx: layer index (None = model-level)
        effectiveness: one of [EAGER], [WARMUP], [DEAD], [GRAPH]
    """
    if not _active(layer_idx):
        return
    if _step_counter >= _MAX_STEPS:
        _active.__dict__["_logged_exceeded"] = True
        return

    layer_str = f" L{layer_idx}" if layer_idx is not None else ""
    logger.info("[SHAPE][%s][step=%d]%s %s: %s",
                effectiveness, _step_counter, layer_str, tag, tensor_shape(x))


def log_weight(name: str, weight: torch.Tensor, layer_idx: int | None = None) -> None:
    """Log a weight tensor shape during model loading."""
    if not _DEBUG_WEIGHTS:
        return
    # limit total weight prints
    cnt = getattr(log_weight, "_cnt", 0)
    if cnt >= _MAX_WEIGHT_PRINTS:
        return
    log_weight._cnt = cnt + 1
    layer_str = f" L{layer_idx}" if layer_idx is not None else ""
    logger.info("[WEIGHT][%d]%s %s: %s %s",
                cnt, layer_str, name, list(weight.shape), str(weight.dtype).replace("torch.", ""))


def log_weights_summary(weights: dict[str, torch.Tensor]) -> None:
    """Log a summary of all weight shapes grouped by prefix."""
    if not _DEBUG_WEIGHTS:
        return
    # Group by top-level prefix
    groups: dict[str, int] = {}
    total_params = 0
    for name, w in weights.items():
        prefix = name.split(".")[0] if "." in name else name
        groups[prefix] = groups.get(prefix, 0) + w.numel()
        total_params += w.numel()
    logger.info("[WEIGHT_SUMMARY] total params: %d (%.2f B)", total_params, total_params / 1e9)
    for prefix, n in sorted(groups.items(), key=lambda x: -x[1]):
        logger.info("[WEIGHT_SUMMARY]   %s: %d (%.2f M)", prefix, n, n / 1e6)


def step_begin() -> None:
    """Call at the beginning of each inference step to advance the counter."""
    global _step_counter
    _step_counter += 1
    if _DEBUG_SHAPES and _step_counter <= _MAX_STEPS:
        logger.info("[SHAPE] ======== STEP %d ========", _step_counter - 1)


def step_end() -> None:
    """Call at the end of each inference step."""
    pass


# ── convenience: pre-built tag strings ────────────────────────────────


class Tags:
    """Pre-built tag strings with effectiveness markers for common checkpoints."""

    # ── Model-level (deepseek_v4.py DeepseekV4Model.forward) ──
    EMBED_OUTPUT = ("embed_output", "[EAGER]")              # after VocabParallelEmbedding
    HC_EXPAND = ("after_hc_expand", "[EAGER]")              # after unsqueeze+repeat
    LAYER_IN = ("layer_in", "[EAGER]")                      # per-layer entry
    LAYER_OUT = ("layer_out", "[EAGER]")                    # per-layer exit
    HC_HEAD_IN = ("hc_head_in", "[EAGER]")                  # before HC head fusion
    HC_HEAD_OUT = ("hc_head_out", "[EAGER]")                # after HC head
    FINAL_NORM_OUT = ("final_norm_out", "[EAGER]")          # after final RMSNorm

    # ── DecoderLayer-level (deepseek_v4.py DecoderLayer.forward) ──
    HC_PRE_ATTN_OUT = ("hc_pre_attn_out", "[EAGER]")       # after hc_pre (attn path)
    INPUT_LAYERNORM_OUT = ("input_layernorm_out", "[EAGER]") # after input_layernorm
    ATTN_OUT = ("attn_out", "[EAGER]")                      # after self_attn
    HC_POST_ATTN_OUT = ("hc_post_attn_out", "[EAGER]")      # after hc_post (attn)
    HC_PRE_FFN_OUT = ("hc_pre_ffn_out", "[EAGER]")          # after hc_pre (ffn path)
    POST_ATTN_LAYERNORM_OUT = ("post_attn_layernorm_out", "[EAGER]")  # after post_attention_layernorm
    MLP_OUT = ("mlp_out", "[EAGER]")                         # after self.mlp (MoE)
    HC_POST_FFN_OUT = ("hc_post_ffn_out", "[EAGER]")        # after hc_post (ffn)

    # ── Attention-level (deepseek_v4.py DeepseekV4Attention.forward → dsa_v1.py) ──
    ATTN_ENTRY = ("attn_entry", "[EAGER]")                   # attention forward entry
    ATTN_EXIT = ("attn_exit", "[EAGER]")                     # attention forward exit
    ATTN_ALLGATHER_OUT = ("attn_allgather_out", "[WARMUP]")  # after FC1 AllGather (inside graph)
    Q_A_OUT = ("q_a_out", "[WARMUP]")                        # wq_a output
    QR_OUT = ("qr_out", "[WARMUP]")                          # q_norm output
    Q_OUT = ("q_out", "[WARMUP]")                            # wq_b output (Q final)
    KV_OUT = ("kv_out", "[WARMUP]")                          # wkv output
    KV_NORM_OUT = ("kv_norm_out", "[WARMUP]")                # kv_norm output
    O_PROJ_IN = ("o_proj_in", "[WARMUP]")                    # attention output → O-proj
    O_PROJ_OUT = ("o_proj_out", "[WARMUP]")                  # after wo_b

    # ── MoE-level (deepseek_v4.py DeepseekV4MoE.forward) ──
    MOE_ENTRY = ("moe_entry", "[EAGER]")                     # MoE forward entry
    MOE_EXIT = ("moe_exit", "[EAGER]")                       # MoE forward exit
    ROUTER_LOGITS = ("router_logits", "[EAGER]")             # after gate linear
    TOPK_WEIGHTS = ("topk_weights", "[EAGER]")               # after select_experts
    TOPK_IDS = ("topk_ids", "[EAGER]")                       # after select_experts
    SHARED_OUT = ("shared_out", "[EAGER]")                   # shared expert output
    MOE_FUSION_OUT = ("moe_fusion_out", "[EAGER]")           # after routed + shared fusion
    MOE_ALLGATHER_OUT = ("moe_allgather_out", "[EAGER]")     # after TP all_gather (SP)

    # ── Custom Op boundary (before/after C++ ops, what you CAN see) ──
    HC_PRE_CUSTOM_IN = ("hc_pre_npu_in", "[EAGER]")          # input to npu_hc_pre_v2
    HC_PRE_CUSTOM = ("hc_pre_npu", "[DEAD]")                 # inside npu_hc_pre_v2 ← NEVER
    HC_POST_CUSTOM_IN = ("hc_post_npu_in", "[EAGER]")        # input to npu_hc_post
    HC_POST_CUSTOM = ("hc_post_npu", "[DEAD]")               # inside npu_hc_post ← NEVER
    COMPRESSOR = ("compressor", "[DEAD]")                    # inside Compressor custom op ← NEVER
    LIGHTNING_INDEXER = ("lightning_indexer", "[DEAD]")      # inside lightning indexer ← NEVER
    MOE_DISPATCH = ("moe_dispatch", "[DEAD]")                # inside npu_moe_distribute_dispatch ← NEVER
    MOE_COMBINE = ("moe_combine", "[DEAD]")                  # inside npu_moe_distribute_combine ← NEVER
    MOE_GMM1 = ("moe_gmm1", "[DEAD]")                        # inside npu_grouped_matmul gate/up ← NEVER
    MOE_SWIGLU = ("moe_swiglu", "[DEAD]")                    # inside npu_swiglu ← NEVER
    MOE_GMM2 = ("moe_gmm2", "[DEAD]")                        # inside npu_grouped_matmul down ← NEVER

    # ── MTP-level (deepseek_v4_mtp.py) ──
    MTP_ENTRY = ("mtp_entry", "[EAGER]")                     # MTP forward entry
    MTP_EXIT = ("mtp_exit", "[EAGER]")                       # MTP forward exit
    MTP_TARGET_HS = ("mtp_target_hs", "[EAGER]")             # target hidden states input
    MTP_E_PROJ_OUT = ("mtp_e_proj_out", "[EAGER]")           # e_proj output
    MTP_H_PROJ_OUT = ("mtp_h_proj_out", "[EAGER]")           # h_proj output
    MTP_FUSED_HS = ("mtp_fused_hs", "[EAGER]")               # after e_proj + h_proj fusion
    MTP_BLOCK_OUT = ("mtp_block_out", "[EAGER]")             # after mtp_block (decoder layer)
    MTP_LOGITS = ("mtp_logits", "[EAGER]")                   # MTP logits output
