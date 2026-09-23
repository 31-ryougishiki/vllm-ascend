# SPDX-License-Identifier: Apache-2.0
"""DSA-CP zigzag exit write: rank-local rows -> replicated natural-order stream.

``zigzag_gather_tensor`` fuses the plan's inverse permutation into the single
write into the stream buffer (``index_select(..., out=)``) instead of building
the reranged tensor and copying it into the buffer.  Both forms must produce the
same bytes; this test pins the fused write against the explicit gather+copy
reference, using the real ``build_zigzag_plan`` permutation on CPU.

The all-gather itself is not exercised here: the TP world size is mocked to 1,
which is exactly the branch the helper takes when there is nothing to gather.
"""

from unittest import mock

import torch

from vllm_ascend.layers.cp_zigzag import build_zigzag_plan, zigzag_gather_tensor


def _plan(cp_size: int, cp_rank: int, num_tokens_pad: int, num_actual_tokens: int):
    return build_zigzag_plan(
        (num_actual_tokens,), (0,), cp_size, cp_rank, num_tokens_pad, num_actual_tokens
    )


def _gather_order(cp_size: int, num_tokens_pad: int, num_actual_tokens: int) -> list:
    """Natural positions in rank-concatenating order (what the all-gather produces)."""
    return [pos for rank in range(cp_size) for pos in _plan(cp_size, rank, num_tokens_pad, num_actual_tokens).zigzag_index]


def test_exit_write_is_the_plan_permutation() -> None:
    cp_size, hidden = 4, 6
    num_actual_tokens, num_tokens_pad = 29, 32
    inv = torch.tensor(_plan(cp_size, 0, num_tokens_pad, num_actual_tokens).inv_gather_index, dtype=torch.int64)
    gather_order = _gather_order(cp_size, num_tokens_pad, num_actual_tokens)

    # The all-gather result, laid out in rank-concatenating order.
    gathered = torch.arange(num_tokens_pad * hidden, dtype=torch.float32).view(num_tokens_pad, hidden)
    # Same values addressed by natural position: this is what the stream must hold.
    natural = torch.empty_like(gathered)
    natural[torch.tensor(gather_order, dtype=torch.int64)] = gathered

    stream = torch.full((num_tokens_pad, hidden), -1.0)
    with mock.patch('vllm_ascend.layers.cp_zigzag.get_tensor_model_parallel_world_size', return_value=1):
        out = zigzag_gather_tensor(gathered, inv, num_tokens_pad, out=stream)

    assert out is stream
    assert torch.equal(stream, natural)
    # Explicit reference form (gather + copy) must agree.
    assert torch.equal(stream, gathered[inv].contiguous())


def test_exit_write_drops_the_trailing_padding_rows() -> None:
    cp_size, hidden = 8, 4
    num_actual_tokens, num_tokens_pad = 61, 64
    inv = torch.tensor(_plan(cp_size, 0, num_tokens_pad, num_actual_tokens).inv_gather_index, dtype=torch.int64)
    gather_order = _gather_order(cp_size, num_tokens_pad, num_actual_tokens)
    gathered = torch.arange(num_tokens_pad * hidden, dtype=torch.float32).view(num_tokens_pad, hidden)
    natural = torch.empty_like(gathered)
    natural[torch.tensor(gather_order, dtype=torch.int64)] = gathered

    stream = torch.full((num_actual_tokens, hidden), -1.0)
    with mock.patch('vllm_ascend.layers.cp_zigzag.get_tensor_model_parallel_world_size', return_value=1):
        zigzag_gather_tensor(gathered, inv, num_actual_tokens, out=stream)

    assert torch.equal(stream, natural[:num_actual_tokens])


def test_exit_write_rejects_a_mismatched_row_count() -> None:
    inv = torch.arange(8, dtype=torch.int64)
    local_rows = torch.zeros(8, 2)
    with mock.patch('vllm_ascend.layers.cp_zigzag.get_tensor_model_parallel_world_size', return_value=1):
        try:
            zigzag_gather_tensor(local_rows, inv, 5, out=torch.zeros(4, 2))
        except RuntimeError as exc:
            assert "num_tokens" in str(exc)
        else:
            raise AssertionError("expected a row-count mismatch to raise")