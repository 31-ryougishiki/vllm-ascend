# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Model-level zigzag CP balance planning for the DSA-CP metadata paths.

The SFA metadata builder and the indexer metadata builder both shard the
scheduled tokens over the TP group, and both have to agree on the exact token
order of one prefill batch: the model boundary, the KV/indexer cache writes and
the attention kernels all consume the rank-local rows.  This module is the
single place that turns a batch into that order, so the two builders cannot
drift apart.

Zigzag replaces the contiguous ``[local_start, local_end_with_pad)`` slice with
a head/tail pairing per sequence: every sequence is cut into ``2 * cp_size``
blocks and rank ``r`` owns block ``r`` and block ``2 * cp_size - 1 - r``.  The
per-rank row count stays exactly ``num_tokens_pad / cp_size`` (see
:func:`vllm_ascend.layers.cp_zigzag.build_zigzag_plan`), so every collective
keeps its shape while the causal attention work is balanced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm_ascend.layers.cp_zigzag import ZigzagPlan, build_zigzag_plan, zigzag_ineligible_reason


@dataclass
class ZigzagCPPlan:
    """Device tensors of one zigzag batch, shared by every DSA-CP consumer."""

    # Global natural positions of this rank's local rows, in [prev, next] order.
    zigzag_index: torch.Tensor
    # Rank-concatenating all-gather order ([r0_prev, r0_next, r1_prev, ...]).
    zigzag_gather_index: torch.Tensor
    # ``inv_gather_index[p]`` is the gathered row holding natural token ``p``.
    inv_gather_index: torch.Tensor
    # Merged single-call metadata: prev/next are exposed as 2 * B batches.
    actual_seq_lengths_query_zigzag: torch.Tensor
    actual_seq_lengths_key_zigzag: torch.Tensor
    # block_table rows duplicated into the same [all prevs, all nexts] order.
    block_table_zigzag: torch.Tensor
    # Padded slot mapping permuted by zigzag_gather_index: the row order the
    # rank-concatenating all-gather produces, i.e. what the KV cache writer and
    # the indexer cache writer have to scatter with.
    slot_mapping_cp_gathered: torch.Tensor
    # Local token count; also the number of rows every rank owns.
    local_tokens: int
    # CPU-side lengths, used for logging and by the debug branch report.
    q_len_prev_list: tuple[int, ...]
    q_len_next_list: tuple[int, ...]
    kv_len_prev_list: tuple[int, ...]
    kv_len_next_list: tuple[int, ...]
    plan: ZigzagPlan


def _cumsum(values: list[int]) -> list[int]:
    result: list[int] = []
    total = 0
    for value in values:
        total += int(value)
        result.append(total)
    return result


def resolve_seq_lens_cpu(common_attn_metadata: Any, num_reqs: int) -> torch.Tensor:
    """Host-side ``seq_lens`` for the first ``num_reqs`` requests.

    ``seq_lens_cpu`` is published by the runner; falling back to a ``.to("cpu")``
    copy would add a device-to-host sync to the metadata hot path of every
    prefill batch.
    """
    seq_lens_cpu = getattr(common_attn_metadata, "seq_lens_cpu", None)
    if seq_lens_cpu is None:
        return common_attn_metadata.seq_lens[:num_reqs].to("cpu")
    return seq_lens_cpu[:num_reqs]


def collect_batch_lengths(
    common_attn_metadata: Any,
    num_reqs: int,
    seq_lens_cpu: torch.Tensor | None = None,
) -> tuple[list[int], list[int], list[bool] | None, list[int]]:
    """Return per-real-request ``(query_lens, prefix_lens, is_prefilling, row_ids)``.

    Padded request slots carry zero scheduled tokens and are not guaranteed to
    sit at the tail of the tensors, so the row ids of the real requests are
    returned together with their lengths.
    """
    if seq_lens_cpu is None:
        seq_lens_cpu = resolve_seq_lens_cpu(common_attn_metadata, num_reqs)
    query_start_loc_cpu = getattr(common_attn_metadata, "query_start_loc_cpu", None)
    if query_start_loc_cpu is None or num_reqs <= 0:
        return [], [], None, []

    num_rows = int(query_start_loc_cpu.shape[0]) - 1
    raw_query_lens = [
        int(query_start_loc_cpu[i + 1]) - int(query_start_loc_cpu[i]) for i in range(min(num_reqs, num_rows))
    ]
    real_req_indices = [i for i, query_len in enumerate(raw_query_lens) if query_len > 0]
    if not real_req_indices:
        return [], [], None, []

    seq_lens_list = seq_lens_cpu.tolist()
    longest = max(real_req_indices)
    # Draft / dummy metadata may carry per-request fields shorter than num_reqs:
    # skip the extraction instead of indexing out of range.  The eligibility
    # gate then reports no_query_lens and the batch keeps the continuous path.
    if longest >= len(seq_lens_list):
        return [], [], None, []

    is_prefilling_cpu: list[bool] | None = None
    is_prefilling_tensor = getattr(common_attn_metadata, "is_prefilling_cpu", None)
    if is_prefilling_tensor is not None:
        if longest >= len(is_prefilling_tensor):
            return [], [], None, []
        is_prefilling_list = is_prefilling_tensor.tolist()
        is_prefilling_cpu = [bool(is_prefilling_list[i]) for i in real_req_indices]

    query_lens_cpu = [raw_query_lens[i] for i in real_req_indices]
    prefix_lens_cpu = [int(seq_lens_list[i]) - raw_query_lens[i] for i in real_req_indices]
    return query_lens_cpu, prefix_lens_cpu, is_prefilling_cpu, real_req_indices


def zigzag_gate_reason(
    common_attn_metadata: Any,
    num_tokens_pad: int,
    cp_size: int,
    *,
    query_lens_cpu: list[int],
    prefix_lens_cpu: list[int],
    is_prefilling_cpu: list[bool] | None,
    num_actual_tokens: int,
    draft_index: int | None,
    v2_model_runner: bool,
    dp_size: int,
    dcp_replicated: bool,
    full_o_proj: bool,
) -> str | None:
    """Name the first gate that keeps this batch on continuous DSA-CP."""
    return zigzag_ineligible_reason(
        getattr(common_attn_metadata, "attn_state", None),
        num_tokens_pad,
        cp_size,
        query_lens_cpu,
        prefix_lens_cpu,
        is_prefilling_cpu,
        num_actual_tokens,
        # Only an actual draft-step metadata build must avoid zigzag: a
        # speculative config alone (deepseek_mtp with enforce_eager) still runs
        # a pure-prefill main forward whose output is gathered back to natural
        # order before the MTP proposer consumes it.
        speculative=draft_index is not None,
        v2_model_runner=v2_model_runner,
        dp_size=dp_size,
        dcp_replicated=dcp_replicated,
        full_o_proj=full_o_proj,
    )


def build_zigzag_cp_plan(
    *,
    common_attn_metadata: Any,
    num_tokens_pad: int,
    num_actual_tokens: int,
    num_reqs: int,
    cp_size: int,
    cp_rank: int,
    device: torch.device,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    query_lens_cpu: list[int],
    prefix_lens_cpu: list[int],
    real_req_indices: list[int],
) -> ZigzagCPPlan:
    """Build every device tensor of one zigzag batch.

    ``slot_mapping`` (natural order, already padded with -1 to
    ``num_tokens_pad``) and ``block_table`` (one row per request) are the two
    inputs that must be permuted; both are returned through the plan.
    """
    plan = build_zigzag_plan(
        query_lens_cpu,
        prefix_lens_cpu,
        cp_size,
        cp_rank,
        num_tokens_pad,
        num_actual_tokens,
    )
    if plan.total_q_prev_tokens + plan.total_q_next_tokens != plan.local_tokens:
        raise AssertionError("zigzag prev/next split does not cover the local token count")

    # Match the width of attn_metadata.block_table / seq_lens: padded request
    # slots carry zero tokens in every kernel-facing tensor.
    pad_reqs = max(int(num_reqs) - plan.num_reqs, 0)
    q_len_prev_list = list(plan.q_len_prev_list) + [0] * pad_reqs
    q_len_next_list = list(plan.q_len_next_list) + [0] * pad_reqs
    kv_len_prev_list = list(plan.kv_len_prev_list) + [0] * pad_reqs
    kv_len_next_list = list(plan.kv_len_next_list) + [0] * pad_reqs

    def _int64(values) -> torch.Tensor:
        return torch.tensor(list(values), dtype=torch.int64, device=device)

    # The merged operators see prev/next as two batches per request: query
    # boundaries are cumulative prefixes over [all prevs, all nexts] while KV
    # lengths stay raw per-batch values.
    prev_cum = _cumsum(q_len_prev_list)
    next_cum = _cumsum(q_len_next_list)
    prev_total = prev_cum[-1] if prev_cum else 0
    q_len_zigzag_list = prev_cum + [prev_total + value for value in next_cum]
    kv_len_zigzag_list = kv_len_prev_list + kv_len_next_list

    # block_table rows must follow the same [all prevs, all nexts] order as the
    # local Q tensor.  Re-select the real request rows by the ids the plan was
    # built from, then repeat them.
    real_row_index = torch.tensor(real_req_indices, dtype=torch.long, device=block_table.device)
    ordered_block_table = block_table.index_select(0, real_row_index)
    if pad_reqs > 0:
        ordered_block_table = torch.cat([ordered_block_table, torch.zeros_like(block_table[:pad_reqs])], dim=0)

    zigzag_gather_index = _int64(plan.zigzag_gather_index)
    return ZigzagCPPlan(
        zigzag_index=_int64(plan.zigzag_index),
        zigzag_gather_index=zigzag_gather_index,
        inv_gather_index=_int64(plan.inv_gather_index),
        actual_seq_lengths_query_zigzag=torch.tensor(q_len_zigzag_list, dtype=torch.int32, device=device),
        actual_seq_lengths_key_zigzag=torch.tensor(kv_len_zigzag_list, dtype=torch.int32, device=device),
        block_table_zigzag=torch.cat([ordered_block_table, ordered_block_table], dim=0),
        slot_mapping_cp_gathered=slot_mapping[zigzag_gather_index],
        local_tokens=plan.local_tokens,
        q_len_prev_list=tuple(q_len_prev_list),
        q_len_next_list=tuple(q_len_next_list),
        kv_len_prev_list=tuple(kv_len_prev_list),
        kv_len_next_list=tuple(kv_len_next_list),
        plan=plan,
    )
