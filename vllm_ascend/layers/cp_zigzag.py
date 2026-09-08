"""Model-boundary zigzag CP helpers for DSA prefill.

These helpers mirror SGLang's NPU legacy DSA-CP flow at the model boundary:

* the full natural-order embedding/positions tensor is sharded once into the
  rank-local ``[prev_blocks_of_all_seqs, next_blocks_of_all_seqs]`` layout;
* all FlashComm collectives keep the same local order because they are
  rank-concatenating all-gather / reduce-scatter pairs (token order inside a
  GEMM/norm/MoE token-wise pass is irrelevant);
* the local output is gathered once at the model boundary and reranged back
  to natural token order for logits.

The metadata plan built by :func:`build_zigzag_plan` is stored in
``DSACPContext`` by ``AscendSFAMetadataBuilder``.  The same eligibility
predicate :func:`can_enable_zigzag_for_batch` is shared by
``NPUModelRunner._pad_for_sequence_parallelism`` and the metadata builder so
the SP padding amount and the attention token layout can never diverge.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather

from vllm_ascend import envs as ascend_envs

# Attention states in which every scheduled token belongs to a prompt.  Mixed
# prefill/decode and speculative states are intentionally excluded.
_PURE_PREFILL_ATTENTION_STATES = {
    "PrefillNoCache",
    "PrefillCacheHit",
    "ChunkedPrefill",
}


def is_zigzag_cp_active() -> bool:
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX

    try:
        return bool(_EXTRA_CTX.zigzag_cp_active)
    except Exception:
        return False


def get_zigzag_cp_context():
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX

    try:
        return _EXTRA_CTX.zigzag_cp_context
    except Exception:
        return None


def zigzag_reorder_moe_aux(x: torch.Tensor, ctx=None) -> torch.Tensor:
    """Reorder a full padded MoE-side tensor to rank-concatenating zigzag order.

    MoE prepare all-gathers rank-local zigzag tensors into
    ``[r0_prev, r0_next, r1_prev, r1_next, ...]``.  ``input_ids`` (hash
    routing) and ``mc2_mask`` must therefore use the exact same full padded
    ``zigzag_gather_index``; using the same helper for both guarantees they can
    never drift apart.  Padding rows are reordered as well and are masked
    downstream (``slot_mapping == -1`` / ``mc2_mask == False``).
    """
    if ctx is None:
        ctx = get_zigzag_cp_context()
    assert ctx is not None
    gather_index = getattr(ctx, "zigzag_gather_index", None)
    if gather_index is None:
        return x
    if x.shape[0] != gather_index.shape[0]:
        raise RuntimeError(
            f"zigzag MoE reorder expects {gather_index.shape[0]} rows, got "
            f"{x.shape[0]}"
        )
    return x[gather_index]


@dataclass(frozen=True)
class ZigzagPlan:
    """CPU-side zigzag token plan for one prefill batch.

    ``num_tokens`` is the SP-padded total token count.  ``query_lens`` are the
    real per-request scheduled token counts; when ``num_tokens > sum(query_lens)``
    the padding is appended to the last sequence, exactly like SGLang's
    ``prepare_context_parallel_metadata``.
    """

    num_tokens: int
    num_actual_tokens: int
    num_reqs: int
    cp_size: int
    cp_rank: int
    prefix_offsets: tuple[int, ...]
    query_lens: tuple[int, ...]
    effective_query_lens: tuple[int, ...]
    # ``block_sizes[s][i]`` is sequence ``s``'s i-th block length; blocks are
    # contiguous in the natural (sequence-concatenated, tail-padded) stream.
    block_sizes: tuple[tuple[int, ...], ...]
    split_list: tuple[int, ...]
    # Global token positions owned by this rank in [prev, next] order.
    zigzag_index: tuple[int, ...]
    # Rank-concatenating all-gather order: [r0_prev, r0_next, r1_prev, ...].
    zigzag_gather_index: tuple[int, ...]
    # inv_gather_index[p] is the row of the gathered tensor holding token p.
    inv_gather_index: tuple[int, ...]
    # Real (non-padding) rows in gather order and their gather positions.
    actual_gather_index: tuple[int, ...]
    actual_rows: tuple[int, ...]
    # Split point between prev and next tokens in the rank-local tensor.
    total_q_prev_tokens: int
    total_q_next_tokens: int
    q_len_prev_list: tuple[int, ...]
    q_len_next_list: tuple[int, ...]
    kv_len_prev_list: tuple[int, ...]
    kv_len_next_list: tuple[int, ...]
    # SGLang-style block-level rerange metadata.
    cp_reverse_index: tuple[int, ...]
    reverse_split_len: tuple[int, ...]

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
    decomposed deterministically.  If the aggregate flow were ever
    infeasible (the total number of extra blocks is always divisible by
    ``cp_size`` for a ``2 * cp_size`` aligned batch), the caller falls back
    to an exact per-row flow.
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
    if num_tokens_pad % segment_num != 0:
        raise ValueError(
            f"zigzag padded length must be a multiple of 2 * cp_size = "
            f"{segment_num}, got {num_tokens_pad}"
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

    # Real rows in gather order and their positions in the gathered tensor.
    actual_gather_positions = [
        pos for pos in gather_positions if pos < num_actual_tokens
    ]
    actual_gather_rows = [
        row for row, pos in enumerate(gather_positions) if pos < num_actual_tokens
    ]

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

    # SGLang-compatible block-level rerange permutation and split lengths.
    cp_reverse_index: list[int] = []
    for batch_id in range(len(query_lens)):
        cp_reverse_index.extend(
            range(batch_id, segment_num * len(query_lens), 2 * len(query_lens))
        )
        cp_reverse_index.extend(
            range(
                (segment_num - 1) * len(query_lens) + batch_id,
                0,
                -2 * len(query_lens),
            )
        )
    reverse_split_len: list[int] = []
    for rank in range(cp_size):
        for seq_idx in range(len(query_lens)):
            reverse_split_len.append(block_sizes[seq_idx][rank])
        for seq_idx in range(len(query_lens)):
            reverse_split_len.append(block_sizes[seq_idx][segment_num - 1 - rank])

    split_list = tuple(
        block_len for blocks in block_sizes for block_len in blocks
    )

    return ZigzagPlan(
        num_tokens=num_tokens_pad,
        num_actual_tokens=num_actual_tokens,
        num_reqs=len(query_lens),
        cp_size=cp_size,
        cp_rank=cp_rank,
        prefix_offsets=prefix_lens,
        query_lens=query_lens,
        effective_query_lens=tuple(effective_query_lens),
        block_sizes=tuple(tuple(blocks) for blocks in block_sizes),
        split_list=split_list,
        zigzag_index=tuple(zigzag_index),
        zigzag_gather_index=tuple(gather_positions),
        inv_gather_index=tuple(inv_positions),
        actual_gather_index=tuple(actual_gather_positions),
        actual_rows=tuple(actual_gather_rows),
        total_q_prev_tokens=total_q_prev_tokens,
        total_q_next_tokens=total_q_next_tokens,
        q_len_prev_list=tuple(q_len_prev_list),
        q_len_next_list=tuple(q_len_next_list),
        kv_len_prev_list=tuple(kv_len_prev_list),
        kv_len_next_list=tuple(kv_len_next_list),
        cp_reverse_index=tuple(cp_reverse_index),
        reverse_split_len=tuple(reverse_split_len),
    )


def can_enable_zigzag_for_batch(
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
) -> bool:
    """The single source of truth for whether a batch may use zigzag CP.

    Both ``NPUModelRunner._pad_for_sequence_parallelism`` and
    ``AscendSFAMetadataBuilder`` must call this predicate so the SP padding
    alignment (``tp_size`` vs ``2 * tp_size``) and the attention layout can
    never disagree.

    SGLang guards every sequence with ``extend_len >= 2 * cp_size``; we keep
    that guard for multi-request batches as well.
    """
    if not ascend_envs.VLLM_ASCEND_CP_BALANCE:
        return False
    if cp_size <= 1:
        return False
    if speculative or v2_model_runner or dp_size > 1:
        # The model-boundary fallback paths cannot carry the zigzag layout.
        return False
    state_name = getattr(attn_state, "name", attn_state)
    if state_name not in _PURE_PREFILL_ATTENTION_STATES:
        return False
    if query_lens is None:
        return False

    query_lens = tuple(int(x) for x in query_lens)
    if not query_lens:
        return False
    if prefix_lens is None:
        prefix_lens = (0,) * len(query_lens)
    prefix_lens = tuple(int(x) for x in prefix_lens)
    if len(prefix_lens) != len(query_lens):
        return False
    if any(prefix < 0 for prefix in prefix_lens):
        return False
    if any(query_len < 2 * cp_size for query_len in query_lens):
        return False

    if num_actual_tokens is None:
        num_actual_tokens = sum(query_lens)
    if num_actual_tokens > num_tokens_pad:
        return False
    if num_actual_tokens < ascend_envs.VLLM_ASCEND_CP_BALANCE_MIN_TOKENS:
        return False
    if num_tokens_pad % (2 * cp_size) != 0:
        return False

    if is_prefilling is None:
        # PrefillNoCache is pure prefill by construction.  ChunkedPrefill /
        # PrefillCacheHit can also contain decode rows, so require an explicit
        # per-request is_prefilling mask there instead of guessing.
        if state_name != "PrefillNoCache":
            return False
    else:
        is_prefilling = tuple(bool(x) for x in is_prefilling)
        if len(is_prefilling) != len(query_lens):
            return False
        if not all(is_prefilling):
            return False
    return True


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
