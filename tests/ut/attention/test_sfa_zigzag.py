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
)
from vllm_ascend.attention.sfa_v1 import ascend_envs
from vllm_ascend.ascend_forward_context import (
    _disable_zigzag_metadata_for_fallback,
)
from vllm_ascend.layers import cp_zigzag as zigzag_cp


def test_can_zigzag_requires_env_threshold_and_single_full_prefill():
    states = (
        AscendAttentionState.ChunkedPrefill,
        AscendAttentionState.PrefillNoCache,
    )
    with (
        patch.object(ascend_envs, "VLLM_ASCEND_CP_BALANCE", True),
        patch.object(ascend_envs, "VLLM_ASCEND_CP_BALANCE_MIN_TOKENS", 16),
    ):
        for state in states:
            assert _can_zigzag(state, 16, 16, 2, 1, 16)
        assert not _can_zigzag(states[0], 8, 8, 2, 1, 8)
        assert not _can_zigzag(states[0], 16, 32, 2, 1, 16)
        assert not _can_zigzag(states[0], 16, 16, 2, 2, 16)
        assert not _can_zigzag(AscendAttentionState.DecodeOnly, 16, 16, 2, 1, 16)

    with patch.object(ascend_envs, "VLLM_ASCEND_CP_BALANCE", False):
        assert not _can_zigzag(states[0], 16, 16, 2, 1, 16)


def test_can_zigzag_supports_non_divisible_prompt_with_sp_padding():
    states = (
        AscendAttentionState.ChunkedPrefill,
        AscendAttentionState.PrefillNoCache,
    )
    with (
        patch.object(ascend_envs, "VLLM_ASCEND_CP_BALANCE", True),
        patch.object(ascend_envs, "VLLM_ASCEND_CP_BALANCE_MIN_TOKENS", 16),
    ):
        # actual 18 tokens, SP-padded to 20 = 2 * cp_size * 5.
        assert _can_zigzag(states[0], 20, 20, 2, 1, 18, 18)
        # Cached prefix / chunked later chunk: seq_len != actual query length.
        assert not _can_zigzag(states[0], 20, 20, 2, 1, 16, 18)
        # Inconsistent metadata.
        assert not _can_zigzag(states[0], 20, 20, 2, 1, 18, 20)
        # Padded length itself must stay 2 * cp_size aligned.
        assert not _can_zigzag(states[0], 18, 18, 2, 1, 18, 18)


def test_build_zigzag_meta_matches_sglang_zigzag_layout():
    rank0 = _build_zigzag_meta(16, 2, 0, torch.device("cpu"))
    rank1 = _build_zigzag_meta(16, 2, 1, torch.device("cpu"))

    torch.testing.assert_close(rank0["zigzag_index"], torch.tensor([0, 1, 2, 3, 12, 13, 14, 15]))
    torch.testing.assert_close(rank1["zigzag_index"], torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]))
    assert rank0["q_half"] == 4
    torch.testing.assert_close(rank0["q_len_prev"], torch.tensor([4], dtype=torch.int32))
    torch.testing.assert_close(rank0["kv_len_prev"], torch.tensor([4], dtype=torch.int32))
    torch.testing.assert_close(rank0["kv_len_next"], torch.tensor([16], dtype=torch.int32))
    torch.testing.assert_close(rank1["kv_len_prev"], torch.tensor([8], dtype=torch.int32))
    torch.testing.assert_close(rank1["kv_len_next"], torch.tensor([12], dtype=torch.int32))


def test_zigzag_gather_index_reranges_to_natural_order():
    meta = _build_zigzag_meta(16, 2, 0, torch.device("cpu"))
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
    meta = _build_zigzag_meta(20, 2, 0, torch.device("cpu"), num_actual_tokens=18)

    # rank0: block0 (tokens 0..4) + block3 (tokens 15..19, last two padded).
    torch.testing.assert_close(
        meta["zigzag_index"],
        torch.tensor([0, 1, 2, 3, 4, 15, 16, 17, 18, 19]),
    )
    assert meta["q_half"] == 5
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


def test_indexer_zigzag_splits_halves_and_concats():
    impl = _make_impl()
    x = torch.zeros(4, 6)
    kw = torch.zeros(4, 6)
    impl.wk_weights_proj.return_value = (kw, None)
    q_c = torch.zeros(4, 2)
    attn_metadata = SimpleNamespace(
        dsa_cp_context=SimpleNamespace(
            zigzag_index=torch.tensor([0, 1, 6, 7]),
            q_half=2,
            q_len_prev=torch.tensor([2], dtype=torch.int32),
            q_len_next=torch.tensor([2], dtype=torch.int32),
            kv_len_prev=torch.tensor([2], dtype=torch.int32),
            kv_len_next=torch.tensor([4], dtype=torch.int32),
        )
    )

    q_li = torch.arange(32, dtype=torch.float32).view(4, 2, 4)
    call_shapes = []
    fake_returns = []

    def fake_device(*args, **kwargs):
        q_li_h = args[1]
        call_shapes.append(tuple(q_li_h.shape))
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
        result = impl._indexer_select_post_process_zigzag(
            x=x,
            q_c=q_c,
            kv_cache=(),
            attn_metadata=attn_metadata,
            cos=torch.zeros(4, 2),
            sin=torch.zeros(4, 2),
        )

    assert call_shapes == [(2, 2, 4), (2, 2, 4)]
    torch.testing.assert_close(result, torch.cat(fake_returns, dim=0))


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
        q_len_prev=torch.tensor([2], dtype=torch.int32),
        q_len_next=torch.tensor([2], dtype=torch.int32),
        kv_len_prev=torch.tensor([2], dtype=torch.int32),
        kv_len_next=torch.tensor([4], dtype=torch.int32),
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
    assert ctx.q_len_prev is None
    assert ctx.q_len_next is None
    assert ctx.kv_len_prev is None
    assert ctx.kv_len_next is None
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
