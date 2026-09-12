#!/usr/bin/env python3
"""Re-run one quantized GEMM with the two row orders seen in a cp_balance round.

Why this exists (HANDOVER §0.1): layer 0's ``down_proj`` receives a **bit-identical**
``(fp8, e8m0 scale)`` row per token in both layouts, yet produces different bf16
output; and a repeated C round gives ``C2−C == 0``, so the effect is a *deterministic*
function of the row order, not run-to-run noise.  This script takes the exact tensors
of one round and asks the op directly: does a token's result depend on its position
in the row set?

Two modes:

* data only (runs anywhere, no NPU) -- shows each side's row count, where the
  positions came from, and the token -> row permutation between the layouts::

      python tools/cp_balance_compare/repro_row_order.py --dir /root/cp_probe/<ts> --layer 0

* re-run the op on the NPU host -- needs the ``w_*.pt`` dump of the same layer
  (``qin:<layer>`` also writes the layer's own weight/weight_scale for rank 0)::

      python tools/cp_balance_compare/repro_row_order.py --dir /root/cp_probe/<ts> \\
          --layer 0 --kind dnq --rank 0 --run-op

  A non-empty ``max|d|`` between the two orders, for rows that carry the *same*
  token, is the minimal reproduction: same values, same weight, only the row order
  differs.  It is what a kernel/vendor report needs, and it needs no model load.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - the target host always has torch
    print("this script needs torch", file=sys.stderr)
    raise

NAME_RE = re.compile(
    r"(?P<kind>[a-z_]+)_cpbal(?P<cpbal>\d+)_layer(?P<layer>-?\d+)"
    r"_rank(?P<rank>\d+)_pid(?P<pid>\d+)_(?P<ts>\d+)\.pt$"
)


def _latest(dump_dir: str, kind: str, layer: int, cpbal: int, rank: int):
    """Newest dump of ``kind`` for one (layer, cpbal, rank), or ``None``."""
    pattern = os.path.join(dump_dir, f"{kind}_cpbal{cpbal}_layer{layer}_rank{rank}_pid*_*.pt")
    best = None
    for path in glob.glob(pattern):
        match = NAME_RE.search(os.path.basename(path))
        if match is None:
            continue
        stamp = int(match.group("ts"))
        if best is None or stamp > best[0]:
            best = (stamp, path)
    if best is None:
        return None, None
    try:
        payload = torch.load(best[1], map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - a truncated dump must not crash the analysis
        print(f"[repro] cannot read {best[1]}: {exc}", file=sys.stderr)
        return None, None
    return best[1], payload


def _describe(tag: str, path: str, payload: dict) -> None:
    quant = payload.get("q")
    scale = payload.get("s")
    print(
        f"[repro] {tag}: {os.path.basename(path)}\n"
        f"         op={payload.get('op')} fused={payload.get('fused')} "
        f"rows={payload.get('rows')} positions_from={payload.get('positions_from')} "
        f"in_dtype={payload.get('in_dtype')}\n"
        f"         q={tuple(quant.shape) if torch.is_tensor(quant) else None} "
        f"{quant.dtype if torch.is_tensor(quant) else ''} | "
        f"s={tuple(scale.shape) if torch.is_tensor(scale) else None} "
        f"{scale.dtype if torch.is_tensor(scale) else ''}"
    )


def _row_map(payload: dict) -> dict[int, int]:
    positions = payload.get("positions")
    if not torch.is_tensor(positions):
        return {}
    return {int(pos): row for row, pos in enumerate(positions.tolist()) if int(pos) >= 0}


def _run_op(args, q, scale, weight, weight_scale):
    import torch_npu

    from vllm_ascend.device.mxfp_compat import FLOAT8_E8M0FNU_DTYPE

    return torch_npu.npu_quant_matmul(
        q,
        weight,
        weight_scale,
        scale_dtype=FLOAT8_E8M0FNU_DTYPE,
        pertoken_scale=scale,
        pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
        bias=None,
        output_dtype=torch.bfloat16,
        group_sizes=[1, 1, args.group_size],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dir", required=True, help="dump directory of one round")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--kind", default="dnq", choices=("dnq", "guq", "gugu_out", "dndn_in"))
    parser.add_argument("--rank", type=int, default=0, help="TP rank whose weight shard to use")
    parser.add_argument("--run-op", action="store_true", help="re-run the GEMM (needs the w dump)")
    parser.add_argument("--group-size", type=int, default=32, help="MX group size (W8A8_MXFP8: 32)")
    parser.add_argument("--token", type=int, default=None, help="report this token's row in both layouts")
    args = parser.parse_args(argv)

    if not Path(args.dir).is_dir():
        print(f"[repro] no such directory: {args.dir}", file=sys.stderr)
        return 2

    sides = {}
    for cpbal in (0, 1):
        path, payload = _latest(args.dir, args.kind, args.layer, cpbal, args.rank)
        if payload is None:
            print(f"[repro] no {args.kind} dump for layer={args.layer} rank={args.rank} cpbal={cpbal}",
                  file=sys.stderr)
            return 2
        _describe("B" if cpbal == 0 else "C", path, payload)
        sides[cpbal] = payload

    rows_b, rows_c = _row_map(sides[0]), _row_map(sides[1])
    common = sorted(set(rows_b) & set(rows_c))
    moved = [tok for tok in common if rows_b[tok] != rows_c[tok]]
    print(
        f"[repro] compared tokens={len(common)} | 行号发生变化={len(moved)}"
        f"（B 与 C 的 token→行号映射不同，正是本脚本要复现的东西）"
    )
    for token in moved[:5]:
        print(f"[repro]   token {token}: row_B={rows_b[token]} row_C={rows_c[token]}")
    if args.token is not None:
        print(
            f"[repro] token {args.token}: row_B={rows_b.get(args.token)} row_C={rows_c.get(args.token)}"
        )

    if not args.run_op:
        print("[repro] 只看数据到此为止；加 --run-op 在 NPU 上重跑这个 GEMM（需要 w dump）")
        return 0

    wpath, wpayload = _latest(args.dir, "w", args.layer, 1, args.rank)
    if wpayload is None:
        wpath, wpayload = _latest(args.dir, "w", args.layer, 0, args.rank)
    if wpayload is None:
        print(
            "[repro] 找不到 w dump（该层的 weight/weight_scale）。"
            "请用含 qin:<层> 的 spec 再跑一轮（rank 0 会多写一个 w_*.pt），"
            "或在离线复现里改用同形状的随机权重。",
            file=sys.stderr,
        )
        return 2
    weight = wpayload.get("weight")
    weight_scale = wpayload.get("weight_scale")
    print(
        f"[repro] weight: {os.path.basename(wpath)} {tuple(weight.shape)} {weight.dtype} | "
        f"scale={None if weight_scale is None else tuple(weight_scale.shape)}"
    )

    try:
        out_b = _run_op(args, sides[0]["q"], sides[0]["s"], weight, weight_scale)
        out_c = _run_op(args, sides[1]["q"], sides[1]["s"], weight, weight_scale)
    except Exception as exc:  # noqa: BLE001 - report, never mask an unsupported call
        print(f"[repro] 重跑 GEMM 失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("[repro] 把这一行连同上面的 shape/ dtype 一起贴回来", file=sys.stderr)
        return 2

    tokens = common
    idx_b = torch.tensor([rows_b[t] for t in tokens], dtype=torch.long)
    idx_c = torch.tensor([rows_c[t] for t in tokens], dtype=torch.long)
    left = out_b.index_select(0, idx_b).float()
    right = out_c.index_select(0, idx_c).float()
    delta = (left - right).abs()
    per_token = delta.max(dim=1).values
    differing = int((per_token > 0).sum())
    print(
        f"[repro] 输出比较（逐 token，同一 weight、同一 token 值，只有行序不同）: "
        f"differing={differing}/{len(tokens)} max|d|={float(per_token.max()):.3e}"
    )
    if differing:
        where = torch.nonzero(per_token > 0).flatten()[:5].tolist()
        for index in where:
            token = tokens[index]
            print(
                f"[repro]   token {token}: row_B={rows_b[token]} row_C={rows_c[token]} "
                f"max|d|={float(per_token[index]):.3e}"
            )
        print("[repro] RESULT: 行序相关（同一个 token 的行换位置后结果就变）⇒ 最小复现成立")
    else:
        print("[repro] RESULT: 该 op 在这个调用形状下与行序无关 ⇒ 差异不在 matmul，往归约/后续算子查")
    return 1 if differing else 0


if __name__ == "__main__":
    sys.exit(main())
