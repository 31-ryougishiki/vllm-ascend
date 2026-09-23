# SPDX-License-Identifier: Apache-2.0
"""Layout invariants of the DSA-CP zigzag plan (`vllm_ascend.layers.cp_zigzag`).

The whole cp_balance fix rests on three properties of `build_zigzag_plan`:

* every rank owns exactly ``num_tokens_pad / cp_size`` rows of the padded
  natural-order stream;
* concatenating the ranks' ``zigzag_index`` in rank order reproduces
  ``zigzag_gather_index`` (this is what a TP all-gather of the rank-local rows
  produces);
* ``inv_gather_index`` is the inverse of that concatenation, so the attention
  can restore the replicated natural-order stream at the o_proj exit.

They are pure CPU properties of the plan, so they are asserted here instead of
only on a machine with NPUs.
"""

import pytest

from vllm_ascend.layers.cp_zigzag import build_zigzag_plan


def _plans(query_lens, prefix_lens, cp_size, num_tokens_pad, num_actual_tokens):
    return [
        build_zigzag_plan(
            tuple(query_lens),
            tuple(prefix_lens),
            cp_size,
            cp_rank,
            num_tokens_pad,
            num_actual_tokens,
        )
        for cp_rank in range(cp_size)
    ]


def _assert_invariants(plans, cp_size, num_tokens_pad):
    local_rows = num_tokens_pad // cp_size
    for cp_rank, plan in enumerate(plans):
        assert len(plan.zigzag_index) == local_rows, (cp_rank, len(plan.zigzag_index), local_rows)

    gathered = [pos for plan in plans for pos in plan.zigzag_index]
    assert tuple(gathered) == plans[0].zigzag_gather_index
    assert sorted(gathered) == list(range(num_tokens_pad)), "each padded position is owned exactly once"

    plan = plans[0]
    for row, global_pos in enumerate(plan.inv_gather_index):
        assert plan.zigzag_gather_index[global_pos] == row, (global_pos, row)
    # What the attention does at the o_proj exit: all_gather + inverse permutation.
    assert [gathered[global_pos] for global_pos in plan.inv_gather_index] == list(range(num_tokens_pad))


@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize(
    ("query_lens", "prefix_lens", "num_tokens_pad", "num_actual_tokens"),
    [
        ([2777], [0], 2784, 2777),
        ([2780], [0], 2784, 2780),
        # Small last request with a large pad share: real rows are owned after
        # pad rows on every rank, which must not disturb the invariants.
        ([256, 2], [0, 0], 384, 258),
        ([1200, 1577], [0, 0], 2784, 2777),
        ([1008, 1008, 1008], [512, 0, 0], 3024, 3024),
    ],
)
def test_zigzag_plan_owns_every_padded_row_once(cp_size, query_lens, prefix_lens, num_tokens_pad, num_actual_tokens):
    assert num_tokens_pad % cp_size == 0
    plans = _plans(query_lens, prefix_lens, cp_size, num_tokens_pad, num_actual_tokens)
    _assert_invariants(plans, cp_size, num_tokens_pad)


def test_single_request_padding_is_owned_by_the_tail_block():
    """A single request's graph/tp padding must sit after its real tokens.

    The pads are appended to the last request, and for one request they land at
    the end of the rank that owns its tail block, so no real row can have a pad
    row inside its own causal window.
    """
    num_actual_tokens, num_tokens_pad, cp_size = 2777, 2784, 8
    plans = _plans([num_actual_tokens], [0], cp_size, num_tokens_pad, num_actual_tokens)
    pad_positions = set(range(num_actual_tokens, num_tokens_pad))
    owners = [cp_rank for cp_rank, plan in enumerate(plans) if pad_positions.intersection(plan.zigzag_index)]
    assert len(owners) == 1, owners
    owner_rows = plans[owners[0]].zigzag_index
    assert set(owner_rows[-len(pad_positions) :]) == pad_positions
    assert all(pos < num_actual_tokens for pos in owner_rows[: -len(pad_positions)])
