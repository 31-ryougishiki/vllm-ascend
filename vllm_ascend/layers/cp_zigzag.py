"""Zigzag row plan for DSA-CP prefill (cp_balance).

The model stream stays full, natural-ordered and replicated on every TP rank —
exactly the layout the contiguous DSA-CP slicing runs with.  Zigzag only
changes *which rows of that stream each rank computes inside the attention*:

* each sequence is cut into ``2 * cp_size`` blocks and rank ``r`` owns block
  ``r`` plus block ``2 * cp_size - 1 - r``, so the causal attention work is
  balanced instead of growing with the rank index;
* the rank-local rows are selected from the padded natural-order stream with
  :func:`build_zigzag_plan`'s ``zigzag_index`` and the attention output rejoins
  the model stream through :func:`zigzag_gather_tensor` (rank-concatenating
  all-gather + ``inv_gather_index``).

Because the model stream stays replicated, every MLP/MoE collective outside
the attention keeps its upstream, per-token identical behaviour.

The metadata plan built by :func:`build_zigzag_plan` is stored in
``DSACPContext`` by ``AscendSFAMetadataBuilder``.  :func:`zigzag_ineligible_reason`
is the single eligibility predicate; the metadata builder is its only caller
and the one that fixes the SP padding alignment the plan relies on.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather
from vllm.logger import init_logger

from vllm_ascend import envs as ascend_envs

logger = init_logger(__name__)


# Attention states in which every scheduled token belongs to a prompt.  Mixed
# prefill/decode and speculative states are intentionally excluded.
_PURE_PREFILL_ATTENTION_STATES = {
    "PrefillNoCache",
    "PrefillCacheHit",
    "ChunkedPrefill",
}


@dataclass(frozen=True)
class ZigzagPlan:
    """CPU-side zigzag token plan for one prefill batch.

    ``num_tokens`` is the SP-padded total token count; the trailing padding is
    appended to the last sequence, exactly like SGLang's
    ``prepare_context_parallel_metadata``.
    """

    num_tokens: int
    num_reqs: int
    cp_size: int
    # Global token positions owned by this rank in [prev, next] order.
    zigzag_index: tuple[int, ...]
    # Rank-concatenating all-gather order: [r0_prev, r0_next, r1_prev, ...].
    zigzag_gather_index: tuple[int, ...]
    # inv_gather_index[p] is the row of the gathered tensor holding token p.
    inv_gather_index: tuple[int, ...]
    # Split point between prev and next tokens in the rank-local tensor.
    total_q_prev_tokens: int
    total_q_next_tokens: int
    q_len_prev_list: tuple[int, ...]
    q_len_next_list: tuple[int, ...]
    kv_len_prev_list: tuple[int, ...]
    kv_len_next_list: tuple[int, ...]

    @property
    def local_tokens(self) -> int:
        return self.num_tokens // self.cp_size


class _MaxFlowEdge:
    __slots__ = ("to", "cap", "rev")

    def __init__(self, to: int, cap: int, rev: int) -> None:
        self.to = to
        self.cap = cap
        self.rev = rev


class _MaxFlow:
    """Small Dinic max-flow used for the CPU-side zigzag metadata plan.

    The graph has ``cp_size + distinct_remainders + 2`` nodes and capacities
    are tiny (<= 2 * batch size on aggregate edges), so this stays well below
    a millisecond in practice and never launches any device work.
    """

    def __init__(self, num_nodes: int) -> None:
        self.num_nodes = num_nodes
        self.graph: list[list[_MaxFlowEdge]] = [[] for _ in range(num_nodes)]

    def add_edge(self, src: int, dst: int, capacity: int) -> None:
        self.graph[src].append(_MaxFlowEdge(dst, capacity, len(self.graph[dst])))
        self.graph[dst].append(_MaxFlowEdge(src, 0, len(self.graph[src]) - 1))

    def max_flow(self, source: int, sink: int, limit: int) -> int:
        flow = 0
        while flow < limit:
            level = [-1] * self.num_nodes
            level[source] = 0
            queue: deque[int] = deque([source])
            while queue:
                node = queue.popleft()
                for edge in self.graph[node]:
                    if edge.cap > 0 and level[edge.to] < 0:
                        level[edge.to] = level[node] + 1
                        queue.append(edge.to)
            if level[sink] < 0:
                break

            it = [0] * self.num_nodes

            def dfs(node: int, available: int) -> int:
                if node == sink:
                    return available
                for idx in range(it[node], len(self.graph[node])):
                    it[node] = idx
                    edge = self.graph[node][idx]
                    if edge.cap > 0 and level[edge.to] == level[node] + 1:
                        pushed = dfs(edge.to, min(available, edge.cap))
                        if pushed:
                            edge.cap -= pushed
                            self.graph[edge.to][edge.rev].cap += pushed
                            return pushed
                return 0

            while flow < limit:
                pushed = dfs(source, min(limit - flow, 10**9))
                if not pushed:
                    break
                flow += pushed
        return flow


def _decompose_remainder_rows(
    remainder: int, row_count: int, cp_size: int, column_caps: Sequence[int]
) -> list[list[int]] | None:
    """Decompose aggregated column capacities back into ``row_count`` rows.

    Every row must sum to ``remainder`` and every column receives at most 2
    units from one row (the two blocks of the zigzag pair).
    """
    caps = list(column_caps)
    rows: list[list[int]] = []
    for _ in range(row_count):
        row = [0] * cp_size
        left = remainder
        while left > 0:
            order = sorted(range(cp_size), key=lambda rank: (-caps[rank], rank))
            progressed = False
            for rank in order:
                if left <= 0:
                    break
                take = min(left, caps[rank], 2 - row[rank])
                if take > 0:
                    row[rank] += take
                    caps[rank] -= take
                    left -= take
                    progressed = True
            if not progressed:
                return None
        rows.append(row)
    if any(cap != 0 for cap in caps):
        return None
    return rows


def _allocate_remainder_extras(
    remainders: Sequence[int], cp_size: int
) -> list[list[int]]:
    """Allocate each sequence's ``remainder`` extra blocks across CP pairs.

    Every sequence is split into ``2 * cp_size`` blocks of ``base`` or
    ``base + 1`` tokens (SGLang's canonical shape).  The extra blocks are
    assigned to the ``cp_size`` head/tail pairs so every rank owns exactly the
    same number of local tokens, which keeps the FlashComm collectives
    equal-shaped while preserving the near-uniform per-rank attention workload
    of the SGLang zigzag layout.

    A tiny aggregate max-flow produces pair capacities and the rows are then
    decomposed deterministically.  When the padded batch length is a multiple
    of ``cp_size`` the total number of extra blocks is divisible by
    ``cp_size``; if an individual sequence makes the aggregate flow
    infeasible, the caller falls back to an exact per-row flow.
    """
    total_extra = sum(remainders)
    if total_extra % cp_size != 0:
        raise AssertionError(
            "zigzag remainder extras must be divisible by cp_size: "
            f"{total_extra=}, {cp_size=}"
        )
    per_rank_extra = total_extra // cp_size
    if total_extra == 0:
        return [[0] * cp_size for _ in remainders]

    groups = list(Counter(int(x) for x in remainders).items())
    group_count = len(groups)

    # Aggregate flow graph: source -> remainder groups -> CP pair columns.
    source = 0
    group_base = 1
    column_base = 1 + group_count
    sink = 1 + group_count + cp_size
    max_flow = _MaxFlow(sink + 1)
    for group_idx, (remainder, count) in enumerate(groups):
        max_flow.add_edge(source, group_base + group_idx, remainder * count)
        for rank in range(cp_size):
            max_flow.add_edge(group_base + group_idx, column_base + rank, 2 * count)
    for rank in range(cp_size):
        max_flow.add_edge(column_base + rank, sink, per_rank_extra)

    required_flow = total_extra
    if max_flow.max_flow(source, sink, required_flow) != required_flow:
        return _individual_remainder_extras(remainders, cp_size)

    column_caps_by_remainder: dict[int, list[int]] = {}
    for group_idx, (remainder, _count) in enumerate(groups):
        caps = [0] * cp_size
        for edge in max_flow.graph[group_base + group_idx]:
            if column_base <= edge.to < column_base + cp_size:
                used = max_flow.graph[edge.to][edge.rev].cap
                caps[edge.to - column_base] = used
        column_caps_by_remainder[remainder] = caps

    rows_by_remainder: dict[int, list[list[int]]] = {}
    for remainder, count in groups:
        rows = _decompose_remainder_rows(
            remainder, count, cp_size, column_caps_by_remainder[remainder]
        )
        if rows is None:
            return _individual_remainder_extras(remainders, cp_size)
        rows_by_remainder[remainder] = rows

    # Restore the original sequence order; sequences with equal remainders are
    # interchangeable because only the remainder shape affects the extras.
    row_iters = {remainder: iter(rows_by_remainder[remainder]) for remainder, _ in groups}
    return [next(row_iters[int(remainder)]) for remainder in remainders]


def _individual_remainder_extras(
    remainders: Sequence[int], cp_size: int
) -> list[list[int]]:
    """Exact per-row max-flow fallback for remainder-extra assignment."""
    row_count = len(remainders)
    source = 0
    row_base = 1
    column_base = 1 + row_count
    sink = 1 + row_count + cp_size
    total_extra = sum(remainders)
    per_rank_extra = total_extra // cp_size

    max_flow = _MaxFlow(sink + 1)
    for row_idx, remainder in enumerate(remainders):
        if remainder:
            max_flow.add_edge(source, row_base + row_idx, remainder)
            for rank in range(cp_size):
                max_flow.add_edge(row_base + row_idx, column_base + rank, 2)
    for rank in range(cp_size):
        max_flow.add_edge(column_base + rank, sink, per_rank_extra)

    if max_flow.max_flow(source, sink, total_extra) != total_extra:
        raise AssertionError(
            "unable to balance zigzag remainder extras for the prefill batch"
        )

    rows: list[list[int]] = []
    for row_idx, _remainder in enumerate(remainders):
        row = [0] * cp_size
        for edge in max_flow.graph[row_base + row_idx]:
            if column_base <= edge.to < column_base + cp_size:
                row[edge.to - column_base] = max_flow.graph[edge.to][edge.rev].cap
        rows.append(row)
    return rows


def _balanced_zigzag_blocks(
    query_lens: Sequence[int], cp_size: int, num_tokens_pad: int
) -> tuple[list[int], list[list[int]]]:
    """Split every sequence into ``2 * cp_size`` balanced blocks.

    Block lengths follow SGLang's ``base`` / ``base + 1`` remainder shape; the
    remainder extras are placed so every rank-local token count is exactly
    ``num_tokens_pad / cp_size``.
    """
    segment_num = 2 * cp_size
    per_rank_tokens = num_tokens_pad // cp_size

    effective_query_lens = [int(x) for x in query_lens]
    padding = num_tokens_pad - sum(effective_query_lens)
    if padding < 0:
        raise ValueError(
            "zigzag padded token count must cover all scheduled tokens: "
            f"{num_tokens_pad=} < sum(query_lens)={sum(effective_query_lens)}"
        )
    if padding > 0:
        # Padding rows live at the tail of the natural stream, i.e. they
        # extend the last request's query length.
        effective_query_lens[-1] += padding

    bases = [query_len // segment_num for query_len in effective_query_lens]
    remainders = [query_len % segment_num for query_len in effective_query_lens]
    pair_extras = _allocate_remainder_extras(remainders, cp_size)

    block_sizes: list[list[int]] = []
    for seq_idx, base in enumerate(bases):
        blocks = [base] * segment_num
        extras = pair_extras[seq_idx]
        for rank, extra_count in enumerate(extras):
            if extra_count >= 1:
                blocks[rank] += 1
            if extra_count == 2:
                blocks[segment_num - 1 - rank] += 1
        expected = base * segment_num + remainders[seq_idx]
        if sum(blocks) != expected:
            raise AssertionError(
                f"internal zigzag block error for sequence {seq_idx}: "
                f"{blocks=} sum {sum(blocks)} != {expected}"
            )
        block_sizes.append(blocks)

    # Rank-local token counts must be equal for equal-shape collectives.
    for rank in range(cp_size):
        rank_tokens = sum(
            blocks[rank] + blocks[segment_num - 1 - rank]
            for blocks in block_sizes
        )
        if rank_tokens != per_rank_tokens:
            raise AssertionError(
                f"zigzag rank {rank} owns {rank_tokens} tokens, expected "
                f"{per_rank_tokens}"
            )

    return effective_query_lens, block_sizes


def build_zigzag_plan(
    query_lens: Sequence[int],
    prefix_lens: Sequence[int],
    cp_size: int,
    cp_rank: int,
    num_tokens_pad: int,
    num_actual_tokens: int | None = None,
) -> ZigzagPlan:
    """Build the CPU-side token plan for one DSA-CP zigzag prefill.

    ``query_lens`` are per-request scheduled (extend) lengths and
    ``prefix_lens`` are the per-request radix-cache / already-computed prefix
    lengths.  The natural token stream is sequence-concatenated; any global
    SP padding is appended to the tail of the stream.
    """
    query_lens = tuple(int(x) for x in query_lens)
    prefix_lens = tuple(int(x) for x in prefix_lens)
    if not query_lens:
        raise ValueError("zigzag plan requires at least one request")
    if len(prefix_lens) != len(query_lens):
        raise ValueError(
            f"prefix_lens must match query_lens, got {len(prefix_lens)} "
            f"prefixes for {len(query_lens)} queries"
        )
    if cp_size <= 1:
        raise ValueError("zigzag plan requires cp_size > 1")
    segment_num = 2 * cp_size
    # Balanced local rows only need ``num_tokens_pad % cp_size == 0``.  The
    # extra head/tail blocks are distributed by ``_allocate_remainder_extras``,
    # so a batch whose real length is aligned to ``cp_size`` (exactly what the
    # continuous-slice path already pads to) can stay on the zigzag path
    # without changing the collective shape between B and C.  Requiring
    # ``2 * cp_size`` here forced an extra 32-token pad that made the two
    # layouts feed different M to every FlashComm GEMM / SFA call.
    if num_tokens_pad % cp_size != 0:
        raise ValueError(
            f"zigzag padded length must be a multiple of cp_size = "
            f"{cp_size}, got {num_tokens_pad}"
        )
    if num_actual_tokens is None:
        num_actual_tokens = sum(query_lens)
    if not 0 <= num_actual_tokens <= num_tokens_pad:
        raise ValueError(
            f"num_actual_tokens {num_actual_tokens} outside [0, "
            f"{num_tokens_pad}]"
        )
    if num_actual_tokens != sum(query_lens):
        raise ValueError(
            f"num_actual_tokens {num_actual_tokens} != sum(query_lens) "
            f"{sum(query_lens)}"
        )

    effective_query_lens, block_sizes = _balanced_zigzag_blocks(
        query_lens, cp_size, num_tokens_pad
    )

    # Cumulative natural-stream starts for every block.
    block_starts: list[list[int]] = []
    offset = 0
    for blocks in block_sizes:
        starts: list[int] = []
        for block_len in blocks:
            starts.append(offset)
            offset += block_len
        block_starts.append(starts)
    if offset != num_tokens_pad:
        raise AssertionError(
            f"block starts end at {offset}, expected padded length {num_tokens_pad}"
        )

    # Rank-local order: all sequences' prev blocks first, then all sequences'
    # next blocks.
    zigzag_index: list[int] = []
    for seq_idx in range(len(query_lens)):
        start = block_starts[seq_idx][cp_rank]
        zigzag_index.extend(range(start, start + block_sizes[seq_idx][cp_rank]))
    next_block = segment_num - 1 - cp_rank
    for seq_idx in range(len(query_lens)):
        start = block_starts[seq_idx][next_block]
        zigzag_index.extend(range(start, start + block_sizes[seq_idx][next_block]))
    if len(zigzag_index) != num_tokens_pad // cp_size:
        raise AssertionError(
            f"zigzag_index has {len(zigzag_index)} tokens, expected "
            f"{num_tokens_pad // cp_size}"
        )

    # Rank-concatenating all-gather order.
    gather_positions: list[int] = []
    for rank in range(cp_size):
        for seq_idx in range(len(query_lens)):
            start = block_starts[seq_idx][rank]
            gather_positions.extend(
                range(start, start + block_sizes[seq_idx][rank])
            )
        tail_block = segment_num - 1 - rank
        for seq_idx in range(len(query_lens)):
            start = block_starts[seq_idx][tail_block]
            gather_positions.extend(
                range(start, start + block_sizes[seq_idx][tail_block])
            )

    inv_positions = [0] * num_tokens_pad
    for row, global_pos in enumerate(gather_positions):
        inv_positions[global_pos] = row

    q_len_prev_list = [block_sizes[s][cp_rank] for s in range(len(query_lens))]
    q_len_next_list = [block_sizes[s][next_block] for s in range(len(query_lens))]
    total_q_prev_tokens = sum(q_len_prev_list)
    total_q_next_tokens = sum(q_len_next_list)
    if total_q_prev_tokens + total_q_next_tokens != num_tokens_pad // cp_size:
        raise AssertionError(
            "prev/next token split does not cover the rank-local tensor"
        )

    kv_len_prev_list: list[int] = []
    kv_len_next_list: list[int] = []
    for seq_idx, blocks in enumerate(block_sizes):
        # Rank r's prev block sees blocks [0 .. r]; its next block sees
        # blocks [0 .. 2*cp-1-r].  Radix-cache prefix tokens are already in
        # the KV cache, so their length is added to both causal lengths.
        kv_len_prev_list.append(
            prefix_lens[seq_idx] + sum(blocks[: cp_rank + 1])
        )
        kv_len_next_list.append(
            prefix_lens[seq_idx] + sum(blocks[: segment_num - cp_rank])
        )

    return ZigzagPlan(
        num_tokens=num_tokens_pad,
        num_reqs=len(query_lens),
        cp_size=cp_size,
        zigzag_index=tuple(zigzag_index),
        zigzag_gather_index=tuple(gather_positions),
        inv_gather_index=tuple(inv_positions),
        total_q_prev_tokens=total_q_prev_tokens,
        total_q_next_tokens=total_q_next_tokens,
        q_len_prev_list=tuple(q_len_prev_list),
        q_len_next_list=tuple(q_len_next_list),
        kv_len_prev_list=tuple(kv_len_prev_list),
        kv_len_next_list=tuple(kv_len_next_list),
    )


def zigzag_ineligible_reason(
    attn_state: Any,
    num_tokens_pad: int,
    cp_size: int,
    query_lens: Sequence[int] | None,
    prefix_lens: Sequence[int] | None = None,
    is_prefilling: Sequence[bool] | None = None,
    num_actual_tokens: int | None = None,
    *,
    speculative: bool = False,
    v2_model_runner: bool = False,
    dp_size: int = 1,
    dcp_replicated: bool = False,
    full_o_proj: bool = True,
) -> str | None:
    """Name the first gate that keeps this batch on continuous DSA-CP.

    None means the batch may use the zigzag layout.  The gate order is the
    one of can_enable_zigzag_for_batch; the name is only reported by the
    VLLM_ASCEND_CP_BALANCE_DEBUG branch log.
    """
    if not ascend_envs.VLLM_ASCEND_CP_BALANCE:
        return "flag_off"
    if cp_size <= 1:
        return "cp_size<=1"
    if speculative:
        return "draft"
    if v2_model_runner:
        return "v2_model_runner"
    if dp_size > 1:
        return "dp>1"
    if dcp_replicated:
        return "dcp_replicated"
    if not full_o_proj:
        return "o_proj_not_full"
    state_name = getattr(attn_state, "name", attn_state)
    if state_name not in _PURE_PREFILL_ATTENTION_STATES:
        return f"state={state_name}"
    if query_lens is None:
        return "no_query_lens"
    query_lens = tuple(int(x) for x in query_lens)
    if not query_lens:
        return "empty_batch"
    if prefix_lens is None:
        prefix_lens = (0,) * len(query_lens)
    prefix_lens = tuple(int(x) for x in prefix_lens)
    if len(prefix_lens) != len(query_lens):
        return "prefix_len_mismatch"
    if any(prefix < 0 for prefix in prefix_lens):
        return "negative_prefix"
    if any(query_len < 2 * cp_size for query_len in query_lens):
        return f"query_len<{2 * cp_size}"
    if num_actual_tokens is None:
        num_actual_tokens = sum(query_lens)
    if num_actual_tokens > num_tokens_pad:
        return "actual>pad"
    if num_actual_tokens < ascend_envs.VLLM_ASCEND_CP_BALANCE_MIN_TOKENS:
        return f"actual<min({ascend_envs.VLLM_ASCEND_CP_BALANCE_MIN_TOKENS})"
    # cp_size alignment is enough: every rank still owns exactly
    # num_tokens_pad / cp_size rows because the remainder extras are
    # distributed per rank.  Keeping the same alignment as the continuous
    # path is what prevents an M-shape difference between B and C.
    if num_tokens_pad % cp_size != 0:
        return "pad%cp_size!=0"
    if is_prefilling is None:
        # PrefillNoCache is pure prefill by construction.  ChunkedPrefill /
        # PrefillCacheHit can also contain decode rows, so require an explicit
        # per-request is_prefilling mask there instead of guessing.
        if state_name != "PrefillNoCache":
            return f"is_prefilling_missing({state_name})"
        return None
    is_prefilling = tuple(bool(x) for x in is_prefilling)
    if len(is_prefilling) != len(query_lens):
        return "is_prefilling_len_mismatch"
    if not all(is_prefilling):
        return "not_all_prefilling"
    return None


def zigzag_gather_tensor(
    x: torch.Tensor,
    inv_gather_index: torch.Tensor,
    num_tokens: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Write this rank's ``[prev,next]`` rows into ``out`` in natural order.

    ``tensor_model_parallel_all_gather`` concatenates rank-local tensors in rank
    order: ``[r0_prev, r0_next, r1_prev, r1_next, ...]`` and
    ``inv_gather_index`` (the layer's own plan, ``DSACPContext.inv_gather_index``)
    maps natural position -> row of that concatenation.  Instead of building the
    reranged tensor and copying it into ``out`` (two extra full-size passes), the
    permutation is fused into the single write into ``out`` with
    ``index_select(..., out=)``; the copy fallback keeps the same result on
    backends without the out variant.

    ``num_tokens`` is the replicated stream row count (``out.shape[0]``): the
    trailing SP padding rows are dropped exactly like the contiguous-slice path
    does.
    """
    rows = int(inv_gather_index.shape[0])
    input_rows = int(x.shape[0])
    if input_rows == 0 or rows % input_rows != 0:
        raise RuntimeError(
            f"zigzag gather expects num_tokens_pad ({rows}) to be a multiple of "
            f"the rank-local rows ({input_rows})"
        )
    if int(out.shape[0]) != num_tokens or num_tokens > rows:
        raise RuntimeError(
            f"zigzag gather expects an output buffer of {min(num_tokens, rows)} rows "
            f"for num_tokens={num_tokens} of {rows} padded rows, got {tuple(out.shape)}"
        )
    if get_tensor_model_parallel_world_size() == 1:
        gathered = x
    else:
        gathered = tensor_model_parallel_all_gather(x.contiguous(), 0)
    if int(gathered.shape[0]) != rows:
        raise RuntimeError(
            f"zigzag gather got {int(gathered.shape[0])} gathered rows, expected {rows}"
        )
    index = inv_gather_index[:num_tokens]
    try:
        torch.index_select(gathered, 0, index, out=out)
    except (RuntimeError, NotImplementedError):
        logger.warning_once(
            "zigzag gather: index_select(out=) unavailable on this device; "
            "falling back to gather+copy"
        )
        out.copy_(gathered[index])
    return out
