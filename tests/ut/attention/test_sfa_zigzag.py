# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.attention.sfa_v1 import (
    AscendAttentionState,
    AscendSFAImpl,
    _build_zigzag_meta,
    _can_zigzag,
)
from vllm_ascend.attention.sfa_v1 import ascend_envs


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


def _make_impl(cp_size: int, rank: int) -> AscendSFAImpl:
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.q_lora_rank = 2
    impl.n_head = 2
    impl.head_dim = 4
    impl.enable_sparse_li_c8 = False
    return impl


def _check_p2p_op(op, kind, peer):
    assert op.op is kind
    assert op.peer == peer


def test_start_zigzag_q_exchange_posts_receives_before_sends():
    impl = _make_impl(4, 0)
    q_c_packed = torch.arange(8, dtype=torch.float32).view(4, 2)

    group = MagicMock()
    batch_mock = MagicMock()
    with (
        patch(
            "vllm_ascend.attention.sfa_v1.get_tp_group",
            return_value=SimpleNamespace(
                world_size=4, rank_in_group=0, device_group=group, ranks=[0, 1, 2, 3]
            ),
        ),
        patch("torch.distributed.batch_isend_irecv", batch_mock),
    ):
        recv, reqs = impl._start_zigzag_q_exchange(q_c_packed)

    assert recv.shape == q_c_packed.shape
    # Desired blocks: 0 (local copy from slot 0) and 7 (remote from rank 3).
    torch.testing.assert_close(recv[:2], q_c_packed[:2])
    assert reqs is not None
    ops = batch_mock.call_args.args[0]
    assert len(ops) == 2
    _check_p2p_op(ops[0], torch.distributed.irecv, 3)
    _check_p2p_op(ops[1], torch.distributed.isend, 1)


def test_restore_zigzag_output_layout_for_rank0_cp4():
    impl = _make_impl(4, 0)
    output = torch.arange(4, dtype=torch.float32).view(4, 1)
    attn_metadata = SimpleNamespace(
        dsa_cp_context=SimpleNamespace(zigzag_index=torch.tensor([0, 1, 6, 7]), q_half=2)
    )

    group = MagicMock()
    batch_mock = MagicMock()
    with (
        patch(
            "vllm_ascend.attention.sfa_v1.get_tp_group",
            return_value=SimpleNamespace(
                world_size=4, rank_in_group=0, device_group=group, ranks=[0, 1, 2, 3]
            ),
        ),
        patch("torch.distributed.batch_isend_irecv", batch_mock),
    ):
        impl._restore_zigzag_output(output, attn_metadata)

    ops = batch_mock.call_args.args[0]
    assert len(ops) == 2
    _check_p2p_op(ops[0], torch.distributed.irecv, 1)
    _check_p2p_op(ops[1], torch.distributed.isend, 3)


def test_indexer_zigzag_splits_halves_and_concats():
    impl = _make_impl(4, 0)
    q_c = torch.zeros(4, 2)
    weights = torch.zeros(4, 3)
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
    fake_q_li = q_li
    call_shapes = []
    fake_returns = []

    def fake_device(*args, **kwargs):
        q_li_h = args[1]
        call_shapes.append(tuple(q_li_h.shape))
        result = torch.full((q_li_h.shape[0], 1), float(len(call_shapes)))
        fake_returns.append(result)
        return result

    with (
        patch.object(impl, "_indexer_qk_proj", return_value=(fake_q_li, None, None)),
        patch("vllm_ascend.attention.sfa_v1.DeviceOperator.indexer_select_post_process", side_effect=fake_device),
        patch("vllm_ascend.attention.sfa_v1.record_attention_compute_start"),
    ):
        result = impl._indexer_select_post_process_zigzag(
            q_c=q_c,
            weights=weights,
            kv_cache=(),
            attn_metadata=attn_metadata,
            cos=torch.zeros(4, 2),
            sin=torch.zeros(4, 2),
        )

    assert call_shapes == [(2, 2, 4), (2, 2, 4)]
    torch.testing.assert_close(result, torch.cat(fake_returns, dim=0))
