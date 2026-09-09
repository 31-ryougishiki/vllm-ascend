# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.attention.sfa_v1 import (
    AscendAttentionState,
    AscendSFAImpl,
    _build_zigzag_meta,
    _can_zigzag,
    _supports_npu_advanced_index,
)
from vllm_ascend.attention.sfa_v1 import ascend_envs
from vllm_ascend.ascend_forward_context import (
    _disable_zigzag_metadata_for_fallback,
)
from vllm_ascend.layers import cp_zigzag as zigzag_cp


def test_fp8_dtypes_are_not_advanced_indexed_on_npu():
    # aclnnIndex does not implement float8; the zigzag KV/indexer writers
    # detect these dtypes and fall back to full padded slot reordering.
    if hasattr(torch, "float8_e4m3fn"):
        assert not _supports_npu_advanced_index(torch.float8_e4m3fn)
    if hasattr(torch, "float8_e5m2"):
        assert not _supports_npu_advanced_index(torch.float8_e5m2)
    assert _supports_npu_advanced_index(torch.bfloat16)
    assert _supports_npu_advanced_index(torch.float16)
    assert _supports_npu_advanced_index(torch.int8)


def test_can_zigzag_supports_multi_request_pure_prefill():
    states = (
        AscendAttentionState.ChunkedPrefill,
        AscendAttentionState.PrefillNoCache,
        AscendAttentionState.PrefillCacheHit,
    )
    with (
        patch.object(ascend_envs, "VLLM_ASCEND_CP_BALANCE", True),
        patch.object(ascend_envs, "VLLM_ASCEND_CP_BALANCE_MIN_TOKENS", 16),
    ):
        for state in states:
            assert _can_zigzag(
                state,
                16,
                16,
                2,
                [16],
                [0],
                [True],
                num_actual_tokens=16,
            )
            # Multiple requests are now supported.
            assert _can_zigzag(
                state,
                16,
                16,
                2,
                [8, 8],
                [0, 0],
                [True, True],
                num_actual_tokens=16,
            )
        # Short single request below the global threshold.
        assert not _can_zigzag(
            states[0], 8, 8, 2, [8], [0], num_actual_tokens=8
        )
        # Metadata padded-length mismatch is rejected.
        assert not _can_zigzag(
            states[0], 16, 32, 2, [16], [0], num_actual_tokens=16
        )
        # One sequence shorter than 2 * cp_size falls back, mirroring SGLang.
        assert not _can_zigzag(
            states[0], 16, 16, 2, [12, 4], [0, 0], num_actual_tokens=16
        )
        # Decode / mixed batches never enable zigzag.
        assert not _can_zigzag(
            AscendAttentionState.DecodeOnly, 16, 16, 2, [16], [0]
        )
        assert not _can_zigzag(
            AscendAttentionState.SpecDecoding, 16, 16, 2, [16], [0]
        )
        # Runner-side disable flags also disable metadata-side zigzag.
        assert not _can_zigzag(
            states[0], 16, 16, 2, [16], [0], speculative=True
        )
        assert not _can_zigzag(
            states[0], 16, 16, 2, [16], [0], v2_model_runner=True
        )
        assert not _can_zigzag(
            states[0], 16, 16, 2, [16], [0], dp_size=2
        )
        assert not _can_zigzag(
            states[0], 16, 16, 2, [16], [0], dcp_replicated=True
        )
        # Rank-local zigzag rows require the SFA prefill full-o_proj path;
        # KV-consumer-only o_proj layouts must not enable zigzag.
        assert not _can_zigzag(
            states[0], 16, 16, 2, [16], [0], full_o_proj=False
        )
        # A batch containing a decode request is rejected by is_prefilling.
        assert not _can_zigzag(
            states[0],
            16,
            16,
            2,
            [8, 8],
            [0, 0],
            is_prefilling=[True, False],
            num_actual_tokens=16,
        )

    with patch.object(ascend_envs, "VLLM_ASCEND_CP_BALANCE", False):
        assert not _can_zigzag(
            states[0], 16, 16, 2, [16], [0], num_actual_tokens=16
        )


def test_can_zigzag_supports_prefix_hit_and_non_divisible_prompt():
    states = (
        AscendAttentionState.ChunkedPrefill,
        AscendAttentionState.PrefillCacheHit,
    )
    with (
        patch.object(ascend_envs, "VLLM_ASCEND_CP_BALANCE", True),
        patch.object(ascend_envs, "VLLM_ASCEND_CP_BALANCE_MIN_TOKENS", 8),
    ):
        # Radix-cache prefixes are accepted; actual query tokens are still
        # zigzag-balanced and prefix lengths are carried separately.
        assert _can_zigzag(
            states[0],
            12,
            12,
            2,
            [6, 6],
            [4, 0],
            [True, True],
            num_actual_tokens=12,
        )
        assert _can_zigzag(
            states[1],
            12,
            12,
            2,
            [6, 6],
            [4, 0],
            [True, True],
            num_actual_tokens=12,
        )
        # actual 18 tokens, SP-padded to 20 = 2 * cp_size * 5.
        assert _can_zigzag(
            states[0], 20, 20, 2, [18], [0], [True], num_actual_tokens=18
        )
        # Padded length itself must stay 2 * cp_size aligned.
        assert not _can_zigzag(
            states[0], 18, 18, 2, [18], [0], num_actual_tokens=18
        )
        # Negative prefix is invalid metadata.
        assert not _can_zigzag(
            states[0], 12, 12, 2, [12], [-1], num_actual_tokens=12
        )


def test_build_zigzag_meta_matches_sglang_zigzag_layout():
    rank0 = _build_zigzag_meta(16, 2, 0, [16], [0], torch.device("cpu"))
    rank1 = _build_zigzag_meta(16, 2, 1, [16], [0], torch.device("cpu"))

    torch.testing.assert_close(rank0["zigzag_index"], torch.tensor([0, 1, 2, 3, 12, 13, 14, 15]))
    torch.testing.assert_close(rank1["zigzag_index"], torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]))
    assert rank0["q_half"] == 4
    assert rank0["total_q_prev_tokens"] == 4
    assert rank0["total_q_next_tokens"] == 4
    torch.testing.assert_close(rank0["q_len_prev"], torch.tensor([4], dtype=torch.int32))
    torch.testing.assert_close(rank0["kv_len_prev"], torch.tensor([4], dtype=torch.int32))
    torch.testing.assert_close(rank0["kv_len_next"], torch.tensor([16], dtype=torch.int32))
    torch.testing.assert_close(rank1["kv_len_prev"], torch.tensor([8], dtype=torch.int32))
    torch.testing.assert_close(rank1["kv_len_next"], torch.tensor([12], dtype=torch.int32))
    torch.testing.assert_close(
        rank0["split_list"], torch.tensor([4, 4, 4, 4])
    )
    # Single-call merged metadata: one prev batch + one next batch.
    torch.testing.assert_close(
        rank0["actual_seq_lengths_query_zigzag"],
        torch.tensor([4, 8], dtype=torch.int32),
    )
    torch.testing.assert_close(
        rank0["actual_seq_lengths_key_zigzag"],
        torch.tensor([4, 16], dtype=torch.int32),
    )


def test_build_zigzag_meta_multi_request_balances_rank_local_tokens():
    rank0 = _build_zigzag_meta(12, 2, 0, [8, 4], [0, 0], torch.device("cpu"))
    rank1 = _build_zigzag_meta(12, 2, 1, [8, 4], [0, 0], torch.device("cpu"))

    # Canonical SGLang-shaped blocks: seq0 [2,2,2,2], seq1 [1,1,1,1].
    # Rank-local order is [all prev blocks, all next blocks].
    torch.testing.assert_close(
        rank0["zigzag_index"], torch.tensor([0, 1, 8, 6, 7, 11])
    )
    torch.testing.assert_close(
        rank1["zigzag_index"], torch.tensor([2, 3, 9, 4, 5, 10])
    )
    torch.testing.assert_close(
        rank0["split_list"], torch.tensor([2, 2, 2, 2, 1, 1, 1, 1])
    )

    # Every rank owns exactly num_tokens_pad / cp_size = 6 tokens, which keeps
    # all_gather / reduce_scatter equal-shaped.
    assert rank0["total_q_prev_tokens"] == 3
    assert rank0["total_q_next_tokens"] == 3
    assert rank1["total_q_prev_tokens"] == 3
    assert rank1["total_q_next_tokens"] == 3

    # Query lengths are cumulative per-request offsets for the kernels.
    torch.testing.assert_close(
        rank0["q_len_prev"], torch.tensor([2, 3], dtype=torch.int32)
    )
    torch.testing.assert_close(
        rank0["q_len_next"], torch.tensor([2, 3], dtype=torch.int32)
    )
    torch.testing.assert_close(
        rank0["actual_seq_q_prev_list"], torch.tensor([2, 1])
    )
    torch.testing.assert_close(
        rank0["actual_seq_q_next_list"], torch.tensor([2, 1])
    )

    # KV visibility follows prefix + cumulative block lengths.
    torch.testing.assert_close(
        rank0["kv_len_prev"], torch.tensor([2, 1], dtype=torch.int32)
    )
    torch.testing.assert_close(
        rank0["kv_len_next"], torch.tensor([8, 4], dtype=torch.int32)
    )
    torch.testing.assert_close(
        rank1["kv_len_prev"], torch.tensor([4, 2], dtype=torch.int32)
    )
    torch.testing.assert_close(
        rank1["kv_len_next"], torch.tensor([6, 3], dtype=torch.int32)
    )

    # Single-call metadata describes [prev0, prev1, next0, next1] as four
    # batches with cumulative query boundaries and raw KV lengths.
    torch.testing.assert_close(
        rank0["actual_seq_lengths_query_zigzag"],
        torch.tensor([2, 3, 5, 6], dtype=torch.int32),
    )
    torch.testing.assert_close(
        rank0["actual_seq_lengths_key_zigzag"],
        torch.tensor([2, 1, 8, 4], dtype=torch.int32),
    )

    # Rank-concatenating gather order and its inverse rerange.
    gather_index = torch.tensor(
        [0, 1, 8, 6, 7, 11, 2, 3, 9, 4, 5, 10]
    )
    torch.testing.assert_close(rank0["zigzag_gather_index"], gather_index)
    torch.testing.assert_close(
        gather_index[rank0["inv_gather_index"]], torch.arange(12)
    )


def test_build_zigzag_meta_balances_uneven_remainders():
    # Uneven query lengths whose SGLang-style first-rem block assignment
    # would leave the rank-local token counts unequal.  The plan must still
    # produce equal local shapes (required by all_gather) while keeping every
    # block within [base, base + 1] of its sequence.
    query_lens = [37, 19, 17, 22, 14, 34, 31, 10]
    cp_size = 4
    segment_num = 2 * cp_size
    num_tokens_pad = 184  # sum(query_lens), already 2 * cp aligned
    metas = [
        _build_zigzag_meta(
            num_tokens_pad,
            cp_size,
            rank,
            query_lens,
            [0] * len(query_lens),
            torch.device("cpu"),
        )
        for rank in range(cp_size)
    ]

    for rank, meta in enumerate(metas):
        assert meta["total_q_prev_tokens"] + meta["total_q_next_tokens"] == 46
        assert meta["zigzag_index"].shape[0] == 46

    # Natural-order block sizes remain canonical: per sequence each block is
    # either floor(L/2c) or floor(L/2c) + 1 and blocks sum to L.
    split_list = metas[0]["split_list"]
    blocks_per_seq = split_list.view(len(query_lens), segment_num)
    for seq_idx, query_len in enumerate(query_lens):
        blocks = blocks_per_seq[seq_idx]
        assert int(blocks.sum()) == query_len
        assert int(blocks.max()) - int(blocks.min()) <= 1


def test_build_zigzag_meta_supports_radix_cache_prefixes():
    rank0 = _build_zigzag_meta(
        8, 2, 0, [4, 4], [8, 16], torch.device("cpu")
    )
    rank1 = _build_zigzag_meta(
        8, 2, 1, [4, 4], [8, 16], torch.device("cpu")
    )

    torch.testing.assert_close(rank0["prefix_offsets"], torch.tensor([8, 16]))
    # Canonical blocks [1,1,1,1] for each request: rank0 owns b0/b3,
    # rank1 owns b1/b2 of each request, in [all prev, all next] order.
    torch.testing.assert_close(
        rank0["zigzag_index"], torch.tensor([0, 4, 3, 7])
    )
    torch.testing.assert_close(
        rank1["zigzag_index"], torch.tensor([1, 5, 2, 6])
    )
    # Prefix length is added to both causal KV lengths.
    torch.testing.assert_close(
        rank0["kv_len_prev"], torch.tensor([9, 17], dtype=torch.int32)
    )
    torch.testing.assert_close(
        rank0["kv_len_next"], torch.tensor([12, 20], dtype=torch.int32)
    )
    torch.testing.assert_close(
        rank1["kv_len_prev"], torch.tensor([10, 18], dtype=torch.int32)
    )
    torch.testing.assert_close(
        rank1["kv_len_next"], torch.tensor([11, 19], dtype=torch.int32)
    )
    torch.testing.assert_close(
        rank0["q_len_prev"], torch.tensor([1, 2], dtype=torch.int32)
    )
    torch.testing.assert_close(
        rank0["q_len_next"], torch.tensor([1, 2], dtype=torch.int32)
    )
    # Merged single-call view of the same prefix batch.
    torch.testing.assert_close(
        rank0["actual_seq_lengths_query_zigzag"],
        torch.tensor([1, 2, 3, 4], dtype=torch.int32),
    )
    torch.testing.assert_close(
        rank0["actual_seq_lengths_key_zigzag"],
        torch.tensor([9, 17, 12, 20], dtype=torch.int32),
    )


def test_zigzag_gather_index_reranges_to_natural_order():
    meta = _build_zigzag_meta(16, 2, 0, [16], [0], torch.device("cpu"))
    gather_index = meta["zigzag_gather_index"]
    # rank-concatenating order: r0 [b0|b3], r1 [b1|b2]
    torch.testing.assert_close(
        gather_index,
        torch.tensor([0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11]),
    )
    torch.testing.assert_close(
        gather_index[meta["inv_gather_index"]], torch.arange(16)
    )
    # Without padding the actual index aliases the padded gather index.
    assert meta["zigzag_actual_gather_index"] is gather_index
    assert meta["zigzag_actual_rows"] is None


def test_build_zigzag_meta_filters_padding_rows():
    meta = _build_zigzag_meta(20, 2, 0, [18], [0], torch.device("cpu"), 18)

    # rank0: block0 (tokens 0..4) + block3 (tokens 15..19, last two padded).
    torch.testing.assert_close(
        meta["zigzag_index"],
        torch.tensor([0, 1, 2, 3, 4, 15, 16, 17, 18, 19]),
    )
    assert meta["q_half"] == 5
    assert meta["total_q_prev_tokens"] == 5
    assert meta["total_q_next_tokens"] == 5
    # Padded gather order: r0 [b0|b3], r1 [b1|b2].
    gather_index = meta["zigzag_gather_index"]
    torch.testing.assert_close(
        gather_index,
        torch.tensor(
            [0, 1, 2, 3, 4, 15, 16, 17, 18, 19, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        ),
    )
    torch.testing.assert_close(
        gather_index[meta["inv_gather_index"]], torch.arange(20)
    )

    # Only rows holding real tokens are exposed for KV/indexer writes.
    expected_actual_rows = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    )
    expected_actual_gather_index = torch.tensor(
        [0, 1, 2, 3, 4, 15, 16, 17, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    )
    torch.testing.assert_close(meta["zigzag_actual_rows"], expected_actual_rows)
    torch.testing.assert_close(
        meta["zigzag_actual_gather_index"], expected_actual_gather_index
    )
    torch.testing.assert_close(
        gather_index[meta["zigzag_actual_rows"]],
        meta["zigzag_actual_gather_index"],
    )


def test_zigzag_moe_aux_reorder_uses_full_padded_gather_index():
    meta = _build_zigzag_meta(16, 2, 0, [16], [0], torch.device("cpu"))
    ctx = SimpleNamespace(
        zigzag_gather_index=meta["zigzag_gather_index"],
    )
    input_ids = torch.arange(16, dtype=torch.int64)
    mc2_mask = torch.arange(16) % 2 == 0

    reordered_ids = zigzag_cp.zigzag_reorder_moe_aux(input_ids, ctx)
    reordered_mask = zigzag_cp.zigzag_reorder_moe_aux(mc2_mask, ctx)

    torch.testing.assert_close(
        reordered_ids, input_ids[meta["zigzag_gather_index"]]
    )
    torch.testing.assert_close(
        reordered_mask, mc2_mask[meta["zigzag_gather_index"]]
    )
    # input_ids and mc2_mask use the exact same full padded index.
    assert reordered_ids.shape == reordered_mask.shape
    torch.testing.assert_close(
        reordered_ids[reordered_mask],
        input_ids[meta["zigzag_gather_index"]][
            mc2_mask[meta["zigzag_gather_index"]]
        ],
    )


def _make_impl() -> AscendSFAImpl:
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.q_lora_rank = 2
    impl.n_head = 2
    impl.head_dim = 4
    impl.enable_sparse_li_c8 = False
    impl.wk_weights_proj = MagicMock()
    return impl


def _zigzag_ctx():
    return SimpleNamespace(
        zigzag_index=torch.tensor([0, 1, 6, 7]),
        inv_gather_index=torch.tensor([0, 4, 1, 5, 2, 6, 3, 7]),
    )


def test_model_boundary_shard_matches_local_zigzag_order():
    x = torch.arange(8, dtype=torch.float32).view(8, 1)
    ctx = _zigzag_ctx()
    with patch.object(zigzag_cp, "get_zigzag_cp_context", return_value=ctx):
        sharded = zigzag_cp.zigzag_shard_tensor(x)
    torch.testing.assert_close(sharded[:, 0], torch.tensor([0.0, 1.0, 6.0, 7.0]))


def test_indexer_zigzag_uses_single_call_with_merged_batches():
    impl = _make_impl()
    x = torch.zeros(6, 6)
    kw = torch.zeros(6, 6)
    impl.wk_weights_proj.return_value = (kw, None)
    q_c = torch.zeros(6, 2)
    block_table_zigzag = torch.arange(24).view(4, 6)
    attn_metadata = SimpleNamespace(
        dsa_cp_context=SimpleNamespace(
            zigzag_index=torch.tensor([0, 1, 2, 6, 7, 8]),
            total_q_prev_tokens=3,
            total_q_next_tokens=3,
            actual_seq_lengths_query_zigzag=torch.tensor(
                [2, 3, 5, 6], dtype=torch.int32
            ),
            actual_seq_lengths_key_zigzag=torch.tensor(
                [2, 5, 4, 6], dtype=torch.int32
            ),
            block_table_zigzag=block_table_zigzag,
        )
    )

    q_li = torch.arange(48, dtype=torch.float32).view(6, 2, 4)
    call_shapes = []
    call_block_tables = []
    fake_returns = []

    def fake_device(*args, **kwargs):
        q_li_h = args[1]
        call_shapes.append(tuple(q_li_h.shape))
        call_block_tables.append(kwargs.get("block_table"))
        result = torch.full((q_li_h.shape[0], 1), float(len(call_shapes)))
        fake_returns.append(result)
        return result

    with (
        patch.object(impl, "_indexer_qk_proj", return_value=(q_li, None, None)),
        patch(
            "vllm_ascend.attention.sfa_v1.DeviceOperator.indexer_select_post_process",
            side_effect=fake_device,
        ),
        patch("vllm_ascend.attention.sfa_v1.record_attention_compute_start"),
    ):
        result = impl.indexer_select_post_process(
            x=x,
            q_c=q_c,
            kv_cache=(),
            attn_metadata=attn_metadata,
            cos=torch.zeros(6, 2),
            sin=torch.zeros(6, 2),
            actual_seq_lengths_query=attn_metadata.dsa_cp_context.actual_seq_lengths_query_zigzag,
            actual_seq_lengths_key=attn_metadata.dsa_cp_context.actual_seq_lengths_key_zigzag,
            block_table=attn_metadata.dsa_cp_context.block_table_zigzag,
        )

    assert call_shapes == [(6, 2, 4)]
    assert len(call_block_tables) == 1
    assert call_block_tables[0] is block_table_zigzag
    torch.testing.assert_close(result, fake_returns[0])


def test_sfa_zigzag_single_call_passes_merged_block_table():
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    block_table_zigzag = torch.arange(12).view(4, 3)
    attn_metadata = SimpleNamespace(
        dsa_cp_context=SimpleNamespace(
            zigzag_index=torch.tensor([0, 1, 2, 3, 4, 5]),
            actual_seq_lengths_query_zigzag=torch.tensor(
                [2, 4, 6], dtype=torch.int32
            ),
            actual_seq_lengths_key_zigzag=torch.tensor(
                [2, 4, 6], dtype=torch.int32
            ),
            block_table_zigzag=block_table_zigzag,
        )
    )
    ctx = attn_metadata.dsa_cp_context
    fake_output = torch.zeros(6, 1, 4)
    captured = {}

    def fake_device(*args, **kwargs):
        captured.update(kwargs)
        return fake_output

    with patch(
        "vllm_ascend.attention.sfa_v1.DeviceOperator.execute_sparse_flash_attention_process",
        side_effect=fake_device,
    ):
        result = impl._execute_sparse_flash_attention_process(
            ql_nope=torch.zeros(6, 1, 4),
            q_pe=torch.zeros(6, 1, 2),
            kv_cache=(),
            topk_indices=torch.zeros(6, 1, 4, dtype=torch.int32),
            attn_metadata=attn_metadata,
            actual_seq_lengths_query=ctx.actual_seq_lengths_query_zigzag,
            actual_seq_lengths_key=ctx.actual_seq_lengths_key_zigzag,
            block_table=ctx.block_table_zigzag,
        )

    assert captured["block_table"] is block_table_zigzag
    assert result is fake_output


def test_disable_zigzag_metadata_restores_continuous_fallback():
    fallback_slot_mapping = torch.tensor([10, 11, 12, 13])
    fallback_cos = torch.full((4, 2), 1.0)
    fallback_sin = torch.full((4, 2), 2.0)
    ctx = SimpleNamespace(
        zigzag_index=torch.tensor([0, 1, 6, 7]),
        zigzag_gather_index=torch.tensor([0, 1, 2, 3, 12, 13, 14, 15]),
        inv_gather_index=torch.tensor([0, 4, 1, 5, 2, 6, 3, 7]),
        zigzag_actual_gather_index=torch.tensor([0, 1, 6, 7]),
        zigzag_actual_rows=torch.tensor([0, 1, 2, 3]),
        q_half=2,
        total_q_prev_tokens=2,
        total_q_next_tokens=2,
        split_list=torch.tensor([2, 2, 2, 2]),
        cp_reverse_index=torch.tensor([0, 2, 3, 1]),
        reverse_split_len=torch.tensor([2, 2, 2, 2]),
        prefix_offsets=torch.tensor([0]),
        q_len_prev=torch.tensor([2], dtype=torch.int32),
        q_len_next=torch.tensor([2], dtype=torch.int32),
        kv_len_prev=torch.tensor([2], dtype=torch.int32),
        kv_len_next=torch.tensor([4], dtype=torch.int32),
        actual_seq_q_prev_list=torch.tensor([2]),
        actual_seq_q_next_list=torch.tensor([2]),
        kv_len_prev_list=torch.tensor([2]),
        kv_len_next_list=torch.tensor([4]),
        actual_seq_lengths_query_zigzag=torch.tensor([2, 4], dtype=torch.int32),
        actual_seq_lengths_key_zigzag=torch.tensor([2, 4], dtype=torch.int32),
        block_table_zigzag=torch.tensor([[0, 1], [0, 1]]),
        slot_mapping_cp=torch.tensor([0, 1, 6, 7]),
        fallback_slot_mapping_cp=fallback_slot_mapping,
        fallback_cos=fallback_cos,
        fallback_sin=fallback_sin,
    )
    meta = SimpleNamespace(
        cos=torch.full((4, 2), 9.0),
        sin=torch.full((4, 2), 8.0),
        dsa_cp_context=ctx,
    )

    _disable_zigzag_metadata_for_fallback({"layer0": meta})

    assert meta.cos is fallback_cos
    assert meta.sin is fallback_sin
    assert ctx.slot_mapping_cp is fallback_slot_mapping
    assert ctx.zigzag_index is None
    assert ctx.zigzag_gather_index is None
    assert ctx.inv_gather_index is None
    assert ctx.zigzag_actual_gather_index is None
    assert ctx.zigzag_actual_rows is None
    assert ctx.q_half == 0
    assert ctx.total_q_prev_tokens == 0
    assert ctx.total_q_next_tokens == 0
    assert ctx.split_list is None
    assert ctx.cp_reverse_index is None
    assert ctx.reverse_split_len is None
    assert ctx.prefix_offsets is None
    assert ctx.q_len_prev is None
    assert ctx.q_len_next is None
    assert ctx.kv_len_prev is None
    assert ctx.kv_len_next is None
    assert ctx.actual_seq_q_prev_list is None
    assert ctx.actual_seq_q_next_list is None
    assert ctx.kv_len_prev_list is None
    assert ctx.kv_len_next_list is None
    assert ctx.actual_seq_lengths_query_zigzag is None
    assert ctx.actual_seq_lengths_key_zigzag is None
    assert ctx.block_table_zigzag is None
    assert ctx.fallback_slot_mapping_cp is None
    assert ctx.fallback_cos is None
    assert ctx.fallback_sin is None


def test_disable_zigzag_metadata_skips_non_zigzag_metadata():
    cos = torch.full((4, 2), 3.0)
    sin = torch.full((4, 2), 4.0)
    meta = SimpleNamespace(
        cos=cos,
        sin=sin,
        dsa_cp_context=SimpleNamespace(
            zigzag_index=None,
            slot_mapping_cp=torch.tensor([4, 5, 6, 7]),
        ),
    )

    _disable_zigzag_metadata_for_fallback(meta)

    assert meta.cos is cos
    assert meta.sin is sin
    assert torch.equal(meta.dsa_cp_context.slot_mapping_cp, torch.tensor([4, 5, 6, 7]))


def test_disable_zigzag_metadata_raises_without_fallback():
    meta = SimpleNamespace(
        cos=torch.full((4, 2), 3.0),
        sin=torch.full((4, 2), 4.0),
        dsa_cp_context=SimpleNamespace(
            zigzag_index=torch.tensor([0, 1, 6, 7]),
            slot_mapping_cp=torch.tensor([0, 1, 6, 7]),
            fallback_slot_mapping_cp=None,
            fallback_cos=None,
            fallback_sin=None,
        ),
    )

    with pytest.raises(RuntimeError, match="fallback tensors"):
        _disable_zigzag_metadata_for_fallback(meta)
