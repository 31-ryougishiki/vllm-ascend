# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
"""CPU code-path checks for the cp_balance precision fix.

The remote ``tools/cp_balance_compare`` A/B runs are expensive (one model load
per configuration) and cannot see the actual defect directly: every local
operator is row-wise and the first observed difference is a *collective*
(``tensor_model_parallel_reduce_scatter``) whose rounding depends on which rank
owns the token's chunk.  This script checks the whole code path offline:

1. zigzag plan invariants and the new cp_size alignment (B and C get the same
   padded M),
2. the reduce ownership permutation B vs C and a bf16 model that shows why an
   owner-ordered reduction is *not* token-order invariant while
   ``fixed_order_rank_sum`` is,
3. source wiring: the fixed-order helper is actually imported and used by the
   DSA-CP row-parallel fallback.

Run directly (no NPU, a few milliseconds)::

    python tools/cp_balance_compare/selftest_cp_logic.py
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent


def _load_cp_zigzag():
    """Load the repo source without importing the installed vllm package."""
    source = REPO_ROOT / "vllm_ascend/layers/cp_zigzag.py"
    if not source.is_file():
        raise RuntimeError(f"cannot find {source}")

    if "vllm" not in sys.modules:
        vllm = types.ModuleType("vllm")
        sys.modules["vllm"] = vllm
    if "vllm.distributed" not in sys.modules:
        distributed = types.ModuleType("vllm.distributed")
        distributed.get_tensor_model_parallel_world_size = lambda: 1
        distributed.tensor_model_parallel_all_gather = lambda x, dim=0: x
        sys.modules["vllm.distributed"] = distributed
    if "vllm_ascend" not in sys.modules:
        ascend = types.ModuleType("vllm_ascend")
        sys.modules["vllm_ascend"] = ascend
    if "vllm_ascend.envs" not in sys.modules:
        envs = types.ModuleType("vllm_ascend.envs")
        envs.VLLM_ASCEND_CP_BALANCE = True
        envs.VLLM_ASCEND_CP_BALANCE_MIN_TOKENS = 0
        sys.modules["vllm_ascend.envs"] = envs

    name = "_cp_balance_selftest_cp_zigzag"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _plan_invariants(cp_zigzag, cp_size: int, query_lens: list[int]) -> tuple[int, int]:
    actual = sum(query_lens)
    num_tokens_pad = ((actual + cp_size - 1) // cp_size) * cp_size
    local = num_tokens_pad // cp_size
    plans = [
        cp_zigzag.build_zigzag_plan(
            query_lens,
            [0] * len(query_lens),
            cp_size,
            rank,
            num_tokens_pad,
            actual,
        )
        for rank in range(cp_size)
    ]

    all_local: list[int] = []
    for rank, plan in enumerate(plans):
        if len(plan.zigzag_index) != local:
            raise AssertionError(
                f"rank {rank} owns {len(plan.zigzag_index)} rows, expected {local}"
            )
        all_local.extend(plan.zigzag_index)
    if sorted(all_local) != list(range(num_tokens_pad)):
        raise AssertionError("rank-local zigzag rows do not cover the padded batch")
    if tuple(all_local) != tuple(plans[0].zigzag_gather_index):
        raise AssertionError("gather index is not the rank-concatenating local order")
    gather = plans[0].zigzag_gather_index
    inv = plans[0].inv_gather_index
    if [gather[i] for i in inv] != list(range(num_tokens_pad)):
        raise AssertionError("inv_gather_index is not the inverse of the gather order")
    if num_tokens_pad % cp_size != 0:
        raise AssertionError("plan padding is not cp_size aligned")
    return actual, num_tokens_pad


def _partial_value(source_rank: int, token: int, torch):
    """Order-sensitive bf16-ish partial for one token and source rank."""
    code = (token * 131 + source_rank * 37) % 97
    sign = 1.0 if source_rank % 2 == 0 else -1.0
    value = sign * (1.0 + (code % 11) * 0.125) + ((code % 7) - 3) * 1e-3
    return torch.tensor(value, dtype=torch.bfloat16)


def _sum_in_order(source_ranks, token: int, torch):
    result = _partial_value(source_ranks[0], token, torch).clone()
    for source_rank in source_ranks[1:]:
        result = result + _partial_value(source_rank, token, torch)
    return result


def _reduce_checks(cp_zigzag, cp_size: int, query_lens: list[int]) -> tuple[int, int, int]:
    import torch

    actual, num_tokens_pad = _plan_invariants(cp_zigzag, cp_size, query_lens)
    local = num_tokens_pad // cp_size
    plans = [
        cp_zigzag.build_zigzag_plan(
            query_lens, [0] * len(query_lens), cp_size, rank,
            num_tokens_pad, actual,
        )
        for rank in range(cp_size)
    ]

    # B: rank-concatenating all-gather order is natural [0..N).
    b_gather = list(range(num_tokens_pad))
    # C: zigzag rank order from the plan.
    c_gather = list(plans[0].zigzag_gather_index)
    b_owner_of = {token: pos // local for pos, token in enumerate(b_gather)}
    c_owner_of = {token: pos // local for pos, token in enumerate(c_gather)}

    # Exercise the real fixed-order helper, not just the scalar model above.
    token_zero_parts = [
        _partial_value(source, 0, torch).reshape(1) for source in range(cp_size)
    ]
    actual_helper = cp_zigzag.fixed_order_rank_sum(token_zero_parts).item()
    expected_helper = _sum_in_order(list(range(cp_size)), 0, torch).item()
    if actual_helper != expected_helper:
        raise AssertionError(
            f"fixed_order_rank_sum mismatch: {actual_helper} != {expected_helper}"
        )

    fixed_diff = 0
    owner_ring_diff = 0
    max_magnitude = 0.0
    for token in range(actual):
        owner_b = b_owner_of[token]
        owner_c = c_owner_of[token]
        # ``ReduceScatter`` algorithms that start the accumulation at the
        # output owner; the B/C owner permutation alone changes the sequence.
        order_b = [(owner_b + step) % cp_size for step in range(cp_size)]
        order_c = [(owner_c + step) % cp_size for step in range(cp_size)]
        ring_b = _sum_in_order(order_b, token, torch)
        ring_c = _sum_in_order(order_c, token, torch)
        if ring_b != ring_c:
            owner_ring_diff += 1
            magnitude = abs(float(ring_b.float()) - float(ring_c.float()))
            max_magnitude = max(max_magnitude, magnitude)

        source_order = list(range(cp_size))
        fixed_b = _sum_in_order(source_order, token, torch)
        fixed_c = _sum_in_order(source_order, token, torch)
        if fixed_b != fixed_c:
            fixed_diff += 1

    return owner_ring_diff, fixed_diff, int(max_magnitude * 1e6)


def _source_wiring_checks() -> None:
    linear_op = (REPO_ROOT / "vllm_ascend/ops/linear_op.py").read_text(encoding="utf-8")
    cp_zigzag = (REPO_ROOT / "vllm_ascend/layers/cp_zigzag.py").read_text(encoding="utf-8")
    model_runner = (REPO_ROOT / "vllm_ascend/worker/model_runner_v1.py").read_text(encoding="utf-8")
    sfa_v1 = (REPO_ROOT / "vllm_ascend/attention/sfa_v1.py").read_text(encoding="utf-8")
    distributed_utils = (REPO_ROOT / "vllm_ascend/distributed/utils.py").read_text(encoding="utf-8")
    register_ops = (REPO_ROOT / "vllm_ascend/ops/register_custom_ops.py").read_text(encoding="utf-8")

    for needle, where in (
        ("fixed_order_reduce_scatter", linear_op),
        ("all_to_all_single", linear_op),
        ("if dsa_cp:", linear_op),
        ("def fixed_order_rank_sum", cp_zigzag),
        ("num_tokens_pad % cp_size", cp_zigzag),
        ("def fixed_order_reduce_scatter", distributed_utils),
        ("_fixed_order_dsa_cp_reduce_scatter", register_ops),
    ):
        if needle not in where:
            raise AssertionError(f"source wiring missing: {needle!r}")
    for forbidden, where in (
        ("round_up(num_actual_tokens, 2 * tp_size)", model_runner),
        ("round_up(num_scheduled_tokens, align_size * 2)", model_runner),
        ("SP-padded to 2 * cp_size", sfa_v1),
        ("tensor_model_parallel_reduce_scatter(x, 0)\n\n    else:", register_ops),
    ):
        if forbidden in where:
            raise AssertionError(f"stale 2 * cp_size assumption remains: {forbidden!r}")


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        cp_zigzag = _load_cp_zigzag()
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[logic] skip: cannot load cp_zigzag ({exc})", file=sys.stderr)
        return 0

    cases = [
        (8, [2048]),
        (8, [2049]),
        (8, [4096]),
        (16, [2049]),
        (4, [37, 19, 17, 22, 14, 34, 31, 10]),
    ]
    rc = 0
    for cp_size, query_lens in cases:
        try:
            owner_ring_diff, fixed_diff, magnitude = _reduce_checks(
                cp_zigzag, cp_size, query_lens
            )
        except Exception as exc:  # pragma: no cover - diagnostics only
            print(f"[logic] FAIL cp={cp_size} lenses={query_lens}: {exc}")
            rc = 1
            continue
        status = "PASS" if owner_ring_diff > 0 and fixed_diff == 0 else "FAIL"
        if status == "FAIL":
            rc = 1
        print(
            f"[logic] {status} cp={cp_size} lenses={query_lens}: "
            f"owner-ordered diffs={owner_ring_diff} "
            f"fixed-order diffs={fixed_diff} max|d|={magnitude}e-6"
        )

    try:
        _source_wiring_checks()
        print("[logic] PASS source wiring: cp_size alignment + fixed-order DSA-CP reduce")
    except Exception as exc:
        print(f"[logic] FAIL source wiring: {exc}")
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
