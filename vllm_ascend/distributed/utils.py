import os

import torch
import torch.distributed as dist
from vllm.distributed import get_dcp_group
from vllm.distributed.parallel_state import GroupCoordinator

from vllm_ascend.layers.cp_zigzag import fixed_order_rank_sum


def get_decode_context_model_parallel_world_size() -> int:
    """Return DCP world size (v0.21.0 helper removed on vLLM main)."""
    return get_dcp_group().world_size


def get_decode_context_model_parallel_rank() -> int:
    """Return DCP rank within group (v0.21.0 helper removed on vLLM main)."""
    return get_dcp_group().rank_in_group


def _allreduce_slice_reduce_scatter(tensor: torch.Tensor, group: GroupCoordinator) -> torch.Tensor:
    """Owner-independent reduce-scatter: all-reduce, then slice our chunk.

    An AllReduce sums every row across the whole TP group.  Its result is the
    concatenation of the chunks that a ReduceScatter would have delivered to
    the different owners, so each rank can simply slice its own chunk.  As the
    reduction order is a property of the collective (rank order), not of the
    token-row owner, the same per-token partials produce the same bits in B
    and C even though zigzag moves a token between owners.
    """
    world_size = int(group.world_size)
    rows = int(tensor.shape[0])
    chunk = rows // world_size
    summed = tensor.contiguous().clone()
    dist.all_reduce(summed, group=group.device_group)
    rank = int(group.rank_in_group)
    return summed[rank * chunk : (rank + 1) * chunk].contiguous()


def _plain_reduce_scatter(tensor: torch.Tensor, group: GroupCoordinator) -> torch.Tensor:
    """Original reduce_scatter implementation (owner-dependent rounding).

    Selected by ``VLLM_ASCEND_CP_BALANCE_REDUCE_MODE=reducescatter``.  It
    exists only to restore the exact pre-cp_balance baseline while debugging.
    """
    world_size = int(group.world_size)
    rows = int(tensor.shape[0])
    chunk = rows // world_size
    trailing = tuple(tensor.shape[1:])
    output = torch.empty(
        (chunk, *trailing), dtype=tensor.dtype, device=tensor.device
    )
    dist.reduce_scatter_tensor(
        output, tensor.contiguous(), group=group.device_group
    )
    return output


def _all_to_all_fixed_order_reduce_scatter(tensor: torch.Tensor, group: GroupCoordinator) -> torch.Tensor:
    """Reduce-scatter with a source-rank sum order, using all_to_all_single.

    Kept as an A/B alternative for ``VLLM_ASCEND_CP_BALANCE_REDUCE_MODE=alltoall``.
    It has lower communication volume than all-reduce but uses a less common
    HCCL collective and therefore needs extra validation on each target SoC.
    """
    world_size = int(group.world_size)
    rows = int(tensor.shape[0])
    chunk = rows // world_size
    trailing = tuple(tensor.shape[1:])
    send = tensor.reshape(world_size, chunk, -1).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(
        recv.view(-1),
        send.view(-1),
        group=group.device_group,
    )
    result = fixed_order_rank_sum([recv[source_rank] for source_rank in range(world_size)])
    return result.reshape(chunk, *trailing).contiguous()


def fixed_order_reduce_scatter(tensor: torch.Tensor, group: GroupCoordinator) -> torch.Tensor:
    """Reduce-scatter whose rounding does not depend on chunk ownership.

    ``dist.reduce_scatter_tensor`` (and the HCCL kernel behind it) may
    accumulate a chunk differently depending on which rank receives that
    chunk.  cp_balance deliberately moves tokens between chunk owners, so that
    implementation detail turns the same per-token partials into different
    bf16 values in B and C.

    Two owner-independent implementations are provided and selected by
    ``VLLM_ASCEND_CP_BALANCE_REDUCE_MODE``:

    ``allreduce`` (default)
        Sum the complete row set in rank order and slice the local chunk.  This
        is the most robust mode on HCCL and is the current correctness target.
    ``alltoall``
        Exchange chunks with ``all_to_all_single`` and sum the received
        per-source chunks in rank order ``0..world_size-1``.  Lower
        communication volume, kept for performance A/B testing.

    Every rank must arrange ``tensor`` in the same row order and own exactly
    ``tensor.shape[0] / group.world_size`` of those rows.  Callers fall back to
    the original collective when that precondition does not hold.
    """
    world_size = int(group.world_size)
    if world_size <= 1:
        return tensor
    rows = int(tensor.shape[0])
    if rows == 0:
        return tensor
    if rows % world_size != 0:
        raise ValueError(
            f"fixed_order_reduce_scatter needs rows divisible by {world_size}, got {rows}"
        )

    mode = os.getenv("VLLM_ASCEND_CP_BALANCE_REDUCE_MODE", "allreduce").strip().lower()
    if mode in ("allreduce", "all_reduce", "ar"):
        return _allreduce_slice_reduce_scatter(tensor, group)
    if mode in ("alltoall", "all_to_all", "a2a"):
        return _all_to_all_fixed_order_reduce_scatter(tensor, group)
    if mode in ("reducescatter", "reduce_scatter", "rs"):
        return _plain_reduce_scatter(tensor, group)
    raise ValueError(
        "VLLM_ASCEND_CP_BALANCE_REDUCE_MODE must be one of "
        f"'allreduce' / 'alltoall' / 'reducescatter', got {mode!r}"
    )


def all_gather_async(
    input: torch.Tensor, group: GroupCoordinator, output: torch.Tensor | None = None, async_op: bool = True
):
    if group.world_size == 1:
        return input, None
    if output is None:
        input_size = input.size()
        output_size = (input_size[0] * group.world_size,) + input_size[1:]
        output = torch.empty(output_size, dtype=input.dtype, device=input.device)
    return output, dist.all_gather_into_tensor(output, input, group=group.device_group, async_op=async_op)


def split_tensor_along_first_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
):
    """Split a tensor along its first dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                in memory.

    Returns:
        A list of Tensors
    """
    from vllm.distributed.utils import divide

    # Get the size and dimension.
    first_dim_size = divide(tensor.size()[0], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, first_dim_size, dim=0)
    # NOTE: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list
