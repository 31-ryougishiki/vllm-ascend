"""Model-boundary zigzag CP helpers for DSA prefill.

These helpers mirror SGLang's NPU legacy DSA-CP flow at the model boundary:

* the full natural-order embedding/positions tensor is sharded once into the
  rank-local ``[prev_block, next_block]`` layout;
* all FlashComm collectives keep the same local order because they are
  rank-concatenating all-gather / reduce-scatter pairs (token order inside a
  GEMM/norm/MoE token-wise pass is irrelevant);
* the local output is gathered once at the model boundary and reranged back
  to natural token order for logits.

The metadata used here is built by ``AscendSFAMetadataBuilder`` and stored in
``DSACPContext``.
"""

from __future__ import annotations

import torch
from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather

from vllm_ascend.ascend_forward_context import _EXTRA_CTX


def is_zigzag_cp_active() -> bool:
    try:
        return bool(_EXTRA_CTX.zigzag_cp_active)
    except Exception:
        return False


def get_zigzag_cp_context():
    try:
        return _EXTRA_CTX.zigzag_cp_context
    except Exception:
        return None


def zigzag_shard_tensor(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Natural-order full tensor -> rank-local ``[prev_block, next_block]``."""
    ctx = get_zigzag_cp_context()
    assert ctx is not None and ctx.zigzag_index is not None
    index = ctx.zigzag_index
    if x.ndim > 1 and dim == 0:
        return x[index].contiguous()
    if dim == 0:
        return x[index].contiguous()
    raise NotImplementedError("zigzag shard currently only supports dim=0")


def zigzag_shard_positions(positions: torch.Tensor) -> torch.Tensor:
    ctx = get_zigzag_cp_context()
    assert ctx is not None and ctx.zigzag_index is not None
    return positions[ctx.zigzag_index].contiguous()


def zigzag_gather_tensor(x: torch.Tensor) -> torch.Tensor:
    """Rank-local ``[prev,next]`` tensor -> full natural-order tensor.

    ``tensor_model_parallel_all_gather`` concatenates rank-local tensors in rank
    order: ``[r0_prev, r0_next, r1_prev, r1_next, ...]``. ``inv_gather_index``
    reranges that concatenation back to ``[token0, token1, ..., token_{T-1}]``.
    """
    ctx = get_zigzag_cp_context()
    assert ctx is not None and ctx.inv_gather_index is not None
    if get_tensor_model_parallel_world_size() == 1:
        gathered = x
    else:
        gathered = tensor_model_parallel_all_gather(x, 0)
    return gathered[ctx.inv_gather_index].contiguous()


def zigzag_gather_hidden_states_list(hidden_states_list):
    return [zigzag_gather_tensor(h) for h in hidden_states_list]


def zigzag_gather_hidden_states_and_aux(hidden_states):
    if isinstance(hidden_states, tuple):
        return (
            zigzag_gather_tensor(hidden_states[0]),
            zigzag_gather_hidden_states_list(hidden_states[1]),
        )
    return zigzag_gather_tensor(hidden_states)
