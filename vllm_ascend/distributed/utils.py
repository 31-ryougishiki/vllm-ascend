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


def fixed_order_reduce_scatter(tensor: torch.Tensor, group: GroupCoordinator) -> torch.Tensor:
    """Reduce-scatter with a source-rank order that does not depend on the owner.

    ``dist.reduce_scatter_tensor`` (and the HCCL kernel behind it) may
    accumulate a chunk differently depending on which rank receives that
    chunk.  cp_balance deliberately moves tokens between chunk owners, so that
    implementation detail turns the same per-token partials into different
    bf16 values in B and C.

    This helper keeps the same O(N) communication volume but exchanges whole
    chunks with ``all_to_all_single`` and then sums the received per-source
    chunks in rank order ``0..world_size-1``.  Token identity, not chunk
    ownership, now determines the accumulation order.

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
