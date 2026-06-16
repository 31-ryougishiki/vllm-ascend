#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Cross-group communication for split attn/moe layer computation.

This module provides point-to-point communication between attn group and moe group
in the layer-split distributed inference scenario.
"""

import logging
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Global state for cross-group communication
_CROSS_GROUP_INITIALIZED = False
_CROSS_ATTN_RANKS = None
_CROSS_MOE_RANKS = None
_CROSS_P2P_GROUPS = {}  # {local_rank: torch.distributed.ProcessGroup}


def init_cross_group(split_tp_size: int, split_ep_size: int) -> None:
    """Initialize cross-group communication for attn/moe split.

    Creates the mapping between attn ranks and moe ranks for point-to-point
    communication.

    Args:
        split_tp_size: Number of GPUs in attn group
        split_ep_size: Number of GPUs in moe group
    """
    global _CROSS_GROUP_INITIALIZED, _CROSS_ATTN_RANKS, _CROSS_MOE_RANKS

    if _CROSS_GROUP_INITIALIZED:
        return

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Attn ranks: [0, split_tp_size-1]
    # Moe ranks: [split_tp_size, world_size-1]
    _CROSS_ATTN_RANKS = list(range(split_tp_size))
    _CROSS_MOE_RANKS = list(range(split_tp_size, world_size))

    # Initialize P2P groups for dedicated point-to-point communication
    init_p2p_groups(split_tp_size, split_ep_size)

    _CROSS_GROUP_INITIALIZED = True


def init_p2p_groups(split_tp_size: int, split_ep_size: int) -> None:
    """Initialize point-to-point communication groups for attn/moe split.

    Creates dedicated 2-rank groups for each attn-moe pair:
    - group 0: [0, split_tp_size] (e.g., [0, 2])
    - group 1: [1, split_tp_size+1] (e.g., [1, 3])

    Args:
        split_tp_size: Number of GPUs in attn group
        split_ep_size: Number of GPUs in moe group

    Note:
        torch.distributed.new_group requires ALL processes to participate.
        All ranks (0 to world_size-1) must call this function.
    """
    global _CROSS_P2P_GROUPS

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    backend = dist.get_backend()

    # For each pair, create a new group containing only these 2 ranks
    for local_rank in range(split_tp_size):
        attn_rank = local_rank
        moe_rank = split_tp_size + local_rank

        # Skip if moe_rank exceeds world_size
        if moe_rank >= world_size:
            break

        p2p_ranks = [attn_rank, moe_rank]

        # new_group requires ALL processes to participate
        group = dist.new_group(p2p_ranks, backend=backend)
        _CROSS_P2P_GROUPS[local_rank] = group


def is_cross_group_initialized() -> bool:
    """Check if cross-group is initialized."""
    return _CROSS_GROUP_INITIALIZED


def _get_peer_rank(local_rank: int, is_attn: bool) -> int:
    """Get the peer rank in the other group.

    Args:
        local_rank: Local rank within the group
        is_attn: True if current rank is in attn group

    Returns:
        Global rank of the peer in the other group
    """
    global _CROSS_ATTN_RANKS, _CROSS_MOE_RANKS

    if not _CROSS_GROUP_INITIALIZED:
        raise RuntimeError("Cross-group not initialized. Call init_cross_group first.")

    if is_attn:
        # Attn rank i communicates with moe rank i
        return _CROSS_MOE_RANKS[local_rank]
    else:
        # Moe rank i communicates with attn rank i
        return _CROSS_ATTN_RANKS[local_rank]


def send_to_moe(hidden_states: torch.Tensor) -> None:
    """Send hidden states from attn group to moe group.

    Directly uses the hidden_states tensor without copying.

    Args:
        hidden_states: Tensor to send (will be sent directly)
    """
    global _CROSS_ATTN_RANKS, _CROSS_P2P_GROUPS

    rank = dist.get_rank()
    if rank not in _CROSS_ATTN_RANKS:
        return  # Not in attn group, skip

    local_rank = rank  # In split mode, rank == local_rank within attn group
    dst_rank = _get_peer_rank(local_rank, is_attn=True)
    group = _CROSS_P2P_GROUPS.get(local_rank)

    # Send hidden_states directly - moe side will receive into its own tensor
    dist.send(hidden_states.contiguous(), dst=dst_rank, group=group)


def recv_from_attn(tensor: torch.Tensor) -> torch.Tensor:
    """Receive hidden states from attn group in moe group.

    Receives directly into the provided tensor.

    Args:
        tensor: Tensor to receive into.

    Returns:
        Received tensor (the same tensor passed in)
    """
    global _CROSS_MOE_RANKS, _CROSS_P2P_GROUPS

    rank = dist.get_rank()
    if rank not in _CROSS_MOE_RANKS:
        raise RuntimeError(f"rank {rank} is not in moe group: {_CROSS_MOE_RANKS}")

    local_rank = rank - _CROSS_MOE_RANKS[0]
    src_rank = _get_peer_rank(local_rank, is_attn=False)
    group = _CROSS_P2P_GROUPS.get(local_rank)

    dist.recv(tensor, src=src_rank, group=group)
    return tensor


def send_to_attn(hidden_states: torch.Tensor) -> None:
    """Send hidden states from moe group to attn group.

    Directly uses the hidden_states tensor without copying.

    Args:
        hidden_states: Tensor to send (will be sent directly)
    """
    global _CROSS_MOE_RANKS, _CROSS_P2P_GROUPS

    rank = dist.get_rank()
    if rank not in _CROSS_MOE_RANKS:
        return  # Not in moe group, skip

    local_rank = rank - _CROSS_MOE_RANKS[0]
    dst_rank = _get_peer_rank(local_rank, is_attn=False)
    group = _CROSS_P2P_GROUPS.get(local_rank)

    # Send hidden_states directly - attn side will receive into its own tensor
    dist.send(hidden_states.contiguous(), dst=dst_rank, group=group)


def recv_from_moe(tensor: torch.Tensor) -> torch.Tensor:
    """Receive hidden states from moe group in attn group.

    Receives directly into the provided tensor.

    Args:
        tensor: Tensor to receive into.

    Returns:
        Received tensor (the same tensor passed in)
    """
    global _CROSS_ATTN_RANKS, _CROSS_P2P_GROUPS

    rank = dist.get_rank()
    if rank not in _CROSS_ATTN_RANKS:
        raise RuntimeError(f"rank {rank} is not in attn group: {_CROSS_ATTN_RANKS}")

    local_rank = rank  # In split mode, rank == local_rank within attn group
    src_rank = _get_peer_rank(local_rank, is_attn=True)
    group = _CROSS_P2P_GROUPS.get(local_rank)

    dist.recv(tensor, src=src_rank, group=group)
    return tensor