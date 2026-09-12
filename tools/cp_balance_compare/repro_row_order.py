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


ACL_FORMAT_FRACTAL_NZ = 29


def _e8m0_dtype():
    """``torch_npu.float8_e8m0fnu`` if this build has it, else ``None``.

    Read straight off ``torch_npu`` on purpose: importing ``vllm_ascend`` here would
    pull in the platform plugin (noisy) and, on the paths that touch
    ``get_ascend_config()``, demand an ascend-config singleton that an offline
    script has no way to initialize (that is exactly how the first attempt failed).
    """
    import torch_npu

    return getattr(torch_npu, "float8_e8m0fnu", None)


def _random_weight(args, q):
    """A same-shaped random fp8 weight, so the op can be re-run without a w dump.

    Order sensitivity is a property of the *call shape* (tiling / accumulation), not
    of the weight values, so a random weight of the right shape answers "does this
    op care about the row order".  The layout still has to be the one the kernel
    expects -- post-loading the layer keeps ``weight`` transposed to ``[K, N]`` and
    NZ-cast, and the scale packed as ``[K/group/2, N, 2]`` -- so the same cast is
    applied here (``npu_format_cast``, what ``maybe_trans_nz`` wraps).
    """
    import torch_npu

    device = q.device
    k = int(q.shape[1])
    group = max(1, args.group_size)
    if k % group or (k // group) % 2:
        raise ValueError(f"K={k} 不能被 group={group} 整除两次，MX scale 无法打包")
    weight = torch.randn(k, args.n, device=device).to(torch.float8_e4m3fn)
    if args.nz:
        weight = torch_npu.npu_format_cast(
            weight, ACL_FORMAT_FRACTAL_NZ, customize_dtype=torch.float8_e4m3fn
        )
    scale = torch.full((k // group // 2, args.n, 2), 127, dtype=torch.uint8, device=device)
    print(
        f"[repro] 随机权重（同形状，nz={args.nz}）: weight={tuple(weight.shape)} {weight.dtype} | "
        f"scale={tuple(scale.shape)} {scale.dtype}"
    )
    return weight, scale


def _run_op(args, q, scale, weight, weight_scale):
    import torch_npu

    return torch_npu.npu_quant_matmul(
        q,
        weight,
        weight_scale,
        scale_dtype=_e8m0_dtype(),
        pertoken_scale=scale,
        pertoken_scale_dtype=_e8m0_dtype(),
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
    parser.add_argument(
        "--random-weight",
        action="store_true",
        help="with --run-op and no w dump: build a same-shaped random fp8 weight instead",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=768,
        help="output dim of the GEMM for --random-weight (down_proj: hidden_size / TP)",
    )
    parser.add_argument(
        "--no-nz",
        dest="nz",
        action="store_false",
        help="with --random-weight: skip the NZ format cast (use when the site runs "
        "with VLLM_ASCEND_ENABLE_NZ=0)",
    )
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
    # ``positions`` are KV-cache slots, not token indices: a request whose block
    # table starts at block 1 has slot = token + 128 (HANDOVER §6.4).  Both layouts
    # share the request's block table, so a common base turns them into natural
    # token indices and makes the printed numbers mean what they say.
    base = min(common) if common else 0
    print(
        f"[repro] compared tokens={len(common)} (slot base={base}，下面按 slot-base 打印 token 序号) "
        f"| 行号发生变化={len(moved)}（B 与 C 的 token→行号映射不同，正是本脚本要复现的东西）"
    )
    for token in moved[:5]:
        print(
            f"[repro]   token {token - base} (slot {token}): "
            f"row_B={rows_b[token]} row_C={rows_c[token]}"
        )
    if args.token is not None:
        slot = args.token + base
        print(
            f"[repro] token {args.token} (slot {slot}): "
            f"row_B={rows_b.get(slot)} row_C={rows_c.get(slot)}"
        )

    if not args.run_op:
        print(
            "[repro] 只看数据到此为止；加 --run-op 在 NPU 上重跑这个 GEMM"
            "（优先用同层 w dump；没有就 --random-weight + --n <输出维>）"
        )
        return 0

    wpath, wpayload = _latest(args.dir, "w", args.layer, 1, args.rank)
    if wpayload is None:
        wpath, wpayload = _latest(args.dir, "w", args.layer, 0, args.rank)
    weight_scale = None
    if wpayload is not None:
        weight = wpayload.get("weight")
        weight_scale = wpayload.get("weight_scale")
        print(
            f"[repro] weight: {os.path.basename(wpath)} {tuple(weight.shape)} {weight.dtype} | "
            f"scale={None if weight_scale is None else tuple(weight_scale.shape)}"
        )
    elif args.random_weight:
        # No w dump (it is only written from bf1fa116a on): a same-shaped random
        # weight still answers "does this op care about the row order", and it needs
        # no extra round.
        try:
            weight, weight_scale = _random_weight(args, sides[1]["q"])
        except Exception as exc:  # noqa: BLE001
            print(f"[repro] 随机权重构造失败: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    else:
        print(
            "[repro] 找不到 w dump（该层的 weight/weight_scale）。"
            "用含 qin:<层> 的 spec 再跑一轮（rank 0 会多写一个 w_*.pt），"
            "或加 --random-weight --n <down_proj 输出维> 用同形状随机权重直接试。",
            file=sys.stderr,
        )
        return 2

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
    # Sanity: a wrong weight layout can make the kernel return zeros/NaN, and an
    # all-equal garbage output would look like "order independent" -- the opposite
    # of the truth.  Refuse to draw a conclusion from a degenerate result.
    if not torch.isfinite(left).all() or float(left.abs().max()) == 0.0:
        print(
            "[repro] 警告: 输出为 NaN/全零 ⇒ 权重布局或调用形状不被该内核接受，"
            "本次结论不可信；试 --no-nz，或改用真实 w dump（带 qin:<层> 跑一轮）",
            file=sys.stderr,
        )
        return 2
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
                f"[repro]   token {token - base} (slot {token}): "
                f"row_B={rows_b[token]} row_C={rows_c[token]} max|d|={float(per_token[index]):.3e}"
            )
        print(
            "[repro] RESULT: 该 op 与行序相关（同一个 token 的行换个位置结果就变）⇒ 最小复现成立；"
            "接下来在模型里避免这种行序（或报给算子侧）"
        )
    else:
        print(
            "[repro] RESULT: 该 op 在这个调用形状下与行序无关 ⇒ 差异不在 matmul，"
            "往紧随其后的 tensor_model_parallel_reduce_scatter / LSE 归约查"
        )
    return 1 if differing else 0


if __name__ == "__main__":
    sys.exit(main())
