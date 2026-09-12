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

  ``--run-op`` runs on ``--device`` (default ``npu:0``), *not* on ``q.device``: the
  dumps are loaded with ``map_location="cpu"``, so anything derived from the
  payload is a CPU tensor and the NPU ops reject it (see ``_resolve_device``).

  ``--all-ranks`` repeats the comparison for every TP rank that dumped this layer
  and prints a summary table.  That matters because the activation dumps are per
  rank while ``w`` (the weight) is written by rank 0 only: ranks other than 0 are
  reproduced with a same-shaped random weight, and the line says which source was
  used.  The reduction that follows the GEMM is a multi-rank op, so a rank sweep
  is also the cheapest way to see whether the GEMM can be excluded everywhere.
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


def _ranks_in_dir(dump_dir: str, kind: str, layer: int) -> list[int]:
    """TP ranks whose ``kind`` dumps exist for this layer (either layout).

    ``w`` (the weight) is written by rank 0 only, but the activation dumps are
    written by every rank of the TP group, so the reproduction can be run for all
    of them -- which matters here because the reduction that follows the GEMM is
    a multi-rank op that a single-rank rerun cannot exercise.
    """
    ranks: set[int] = set()
    for cpbal in (0, 1):
        pattern = os.path.join(dump_dir, f"{kind}_cpbal{cpbal}_layer{layer}_rank*_pid*_*.pt")
        for path in glob.glob(pattern):
            match = NAME_RE.search(os.path.basename(path))
            if match is not None:
                ranks.add(int(match.group("rank")))
    return sorted(ranks)


ACL_FORMAT_FRACTAL_NZ = 29


def _resolve_device(args) -> torch.device:
    """The device the re-run GEMM must execute on.

    It cannot be inferred from the payload: :func:`_latest` loads every dump with
    ``map_location="cpu"`` (the checker half of the tool is CPU-only), so
    ``q.device`` is *always* CPU.  Building the random weight there and calling
    the op with it is what failed on the site with
    ``npu::npu_format_cast ... arguments from the 'CPU' backend`` -- an offline
    tool defect that reads like a model problem, so it gets an explicit knob and
    a printed line instead of a device guess.
    """
    import torch_npu  # noqa: F401 - importing it registers the PrivateUse1 backend

    return torch.device(args.device)


def _to_device(value, device: torch.device):
    """Move a tensor (or ``None``) to ``device``; anything else is passed through."""
    return value.to(device) if torch.is_tensor(value) else value


def _e8m0_dtype():
    """``torch_npu.float8_e8m0fnu`` if this build has it, else ``None``.

    Read straight off ``torch_npu`` on purpose: importing ``vllm_ascend`` here would
    pull in the platform plugin (noisy) and, on the paths that touch
    ``get_ascend_config()``, demand an ascend-config singleton that an offline
    script has no way to initialize (that is exactly how the first attempt failed).
    """
    import torch_npu

    return getattr(torch_npu, "float8_e8m0fnu", None)


def _random_weight(args, q, device: torch.device):
    """A same-shaped random fp8 weight, so the op can be re-run without a w dump.

    Order sensitivity is a property of the *call shape* (tiling / accumulation), not
    of the weight values, so a random weight of the right shape answers "does this
    op care about the row order".  The layout still has to be the one the kernel
    expects -- post-loading the layer keeps ``weight`` transposed to ``[K, N]`` and
    NZ-cast, and the scale packed as ``[K/group/2, N, 2]`` -- so the same cast is
    applied here (``npu_format_cast``, what ``maybe_trans_nz`` wraps).

    ``device`` comes from ``--device``; ``q`` only supplies the K dimension (it was
    loaded on the CPU, so its device would make the cast fail).
    """
    import torch_npu

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
        f"[repro] 随机权重（同形状，nz={args.nz}, device={device}）: "
        f"weight={tuple(weight.shape)} {weight.dtype} | "
        f"scale={tuple(scale.shape)} {scale.dtype}"
    )
    return weight, scale


def _run_op(args, q, scale, weight, weight_scale, device: torch.device):
    """Run the layer's own quantized GEMM once, on ``device``, with these inputs.

    Every input is moved to ``device`` first: the payload was loaded on the CPU
    (``map_location="cpu"``), and ``npu_quant_matmul`` refuses CPU arguments.  The
    result comes back to the CPU so the comparison below cannot depend on a device
    sync.
    """
    import torch_npu

    output = torch_npu.npu_quant_matmul(
        _to_device(q, device),
        _to_device(weight, device),
        _to_device(weight_scale, device),
        scale_dtype=_e8m0_dtype(),
        pertoken_scale=_to_device(scale, device),
        pertoken_scale_dtype=_e8m0_dtype(),
        bias=None,
        output_dtype=torch.bfloat16,
        group_sizes=[1, 1, args.group_size],
    )
    return output.to("cpu") if torch.is_tensor(output) else output


def _reproduce_rank(args, device: torch.device, rank: int, quiet: bool = False) -> tuple[int, dict]:
    """Row-order reproduction for **one** TP rank: ``(exit code, summary)``.

    ``quiet`` keeps the per-rank file/permutation detail out of an ``--all-ranks``
    run; the summary line and the verdict are printed either way.
    """
    sides = {}
    for cpbal in (0, 1):
        path, payload = _latest(args.dir, args.kind, args.layer, cpbal, rank)
        if payload is None:
            print(
                f"[repro] rank {rank}: no {args.kind} dump for layer={args.layer} cpbal={cpbal}",
                file=sys.stderr,
            )
            return 2, {"rank": rank, "status": "no-dump"}
        if not quiet:
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
        f"[repro] rank {rank}: compared tokens={len(common)} "
        f"(slot base={base}，下面按 slot-base 打印 token 序号) "
        f"| 行号发生变化={len(moved)}（B 与 C 的 token→行号映射不同，正是本脚本要复现的东西）"
    )
    if not quiet:
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

    summary = {
        "rank": rank,
        "status": "data-only",
        "compared": len(common),
        "moved": len(moved),
        "positions_from": sides[1].get("positions_from"),
    }
    if not args.run_op:
        if not quiet:
            print(
                "[repro] 只看数据到此为止；加 --run-op 在 NPU 上重跑这个 GEMM"
                "（优先用同层 w dump；没有就 --random-weight + --n <输出维>）"
            )
        return 0, summary

    # Per rank: that rank's own weight shard when its ``w`` dump exists (rank 0
    # only, by design), otherwise the same-shaped random weight.
    wpath, wpayload = _latest(args.dir, "w", args.layer, 1, rank)
    if wpayload is None:
        wpath, wpayload = _latest(args.dir, "w", args.layer, 0, rank)
    weight_scale = None
    source = "w"
    if wpayload is not None:
        weight = wpayload.get("weight")
        weight_scale = wpayload.get("weight_scale")
        weight_format = str(wpayload.get("weight_format", "as-is"))
        print(
            f"[repro] weight(rank {rank}): {os.path.basename(wpath)} {tuple(weight.shape)} "
            f"{weight.dtype} | scale={None if weight_scale is None else tuple(weight_scale.shape)} "
            f"| format={weight_format}"
        )
        if args.nz and weight_format not in ("as-is", "NZ"):
            # The in-model dump had to undo the load-time NZ cast to get the weight
            # off the NPU ("copy_ do not support internal format"); put it back so the
            # op still sees the layout the kernel consumed.  Same call the layer's
            # process_weights_after_loading made (see vllm_ascend.utils.maybe_trans_nz).
            import torch_npu

            weight = torch_npu.npu_format_cast(
                weight.to(device), ACL_FORMAT_FRACTAL_NZ, customize_dtype=weight.dtype
            )
            print(
                f"[repro] weight(rank {rank}): {weight_format} → NZ 已重放"
                f"（customize_dtype={weight.dtype}）"
            )
    elif args.random_weight:
        # No w dump (it is only written for rank 0): a same-shaped random weight
        # still answers "does this op care about the row order", and it needs no
        # extra round.
        source = "random"
        try:
            weight, weight_scale = _random_weight(args, sides[1]["q"], device)
        except Exception as exc:  # noqa: BLE001
            print(f"[repro] rank {rank}: 随机权重构造失败: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2, {"rank": rank, "status": "weight-failed"}
    else:
        print(
            "[repro] 找不到 w dump（该层的 weight/weight_scale）。"
            "用含 qin:<层> 的 spec 再跑一轮（rank 0 会多写一个 w_*.pt），"
            "或加 --random-weight --n <down_proj 输出维> 用同形状随机权重直接试。",
            file=sys.stderr,
        )
        return 2, {"rank": rank, "status": "no-weight"}
    summary["weight"] = source
    if source == "w":
        summary["weight_format"] = weight_format

    try:
        out_b = _run_op(args, sides[0]["q"], sides[0]["s"], weight, weight_scale, device)
        out_c = _run_op(args, sides[1]["q"], sides[1]["s"], weight, weight_scale, device)
    except Exception as exc:  # noqa: BLE001 - report, never mask an unsupported call
        print(f"[repro] rank {rank}: 重跑 GEMM 失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            f"[repro] 把这一行连同上面的 shape/ dtype 一起贴回来（device={device}）",
            file=sys.stderr,
        )
        return 2, {"rank": rank, "status": "op-failed"}

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
            f"[repro] rank {rank}: 警告: 输出为 NaN/全零 ⇒ 权重布局或调用形状不被该内核接受，"
            "本次结论不可信；试 --no-nz，或改用真实 w dump（带 qin:<层> 跑一轮）",
            file=sys.stderr,
        )
        return 2, {"rank": rank, "status": "degenerate"}
    delta = (left - right).abs()
    per_token = delta.max(dim=1).values
    differing = int((per_token > 0).sum())
    summary.update(
        {
            "status": "differs" if differing else "identical",
            "differing": differing,
            "max_abs": float(per_token.max()) if per_token.numel() else 0.0,
        }
    )
    print(
        f"[repro] rank {rank}: 输出比较（逐 token，同一 weight、同一 token 值，只有行序不同）: "
        f"differing={differing}/{len(tokens)} max|d|={float(per_token.max()):.3e} w={source}"
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
            f"[repro] RESULT(rank {rank}): 该 op 与行序相关（同一个 token 的行换个位置结果就变）"
            "⇒ 最小复现成立；接下来在模型里避免这种行序（或报给算子侧）"
        )
    else:
        print(
            f"[repro] RESULT(rank {rank}): 该 op 在这个调用形状下与行序无关 "
            "⇒ 差异不在 matmul，往紧随其后的 tensor_model_parallel_reduce_scatter / LSE 归约查"
        )
    return (1 if differing else 0), summary


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
    parser.add_argument(
        "--device",
        default="npu:0",
        help="device the re-run GEMM runs on (default npu:0). The dumps are loaded on the "
        "CPU, so the payload cannot supply it; use e.g. npu:3 when chip 0 is busy.",
    )
    parser.add_argument("--token", type=int, default=None, help="report this token's row in both layouts")
    parser.add_argument(
        "--all-ranks",
        action="store_true",
        help="run the comparison for every TP rank that has dumps in --dir and print a summary "
        "table (the activation dumps are per rank; only rank 0 also has a w dump)",
    )
    args = parser.parse_args(argv)

    if not Path(args.dir).is_dir():
        print(f"[repro] no such directory: {args.dir}", file=sys.stderr)
        return 2

    if args.all_ranks:
        ranks = _ranks_in_dir(args.dir, args.kind, args.layer)
        if not ranks:
            print(
                f"[repro] {args.dir} 里没有 {args.kind} layer={args.layer} 的 dump（两个布局都没有）",
                file=sys.stderr,
            )
            return 2
    else:
        ranks = [args.rank]
    quiet = args.all_ranks and len(ranks) > 1

    device = None
    if args.run_op:
        try:
            device = _resolve_device(args)
        except Exception as exc:  # noqa: BLE001 - a missing backend must not look like a result
            print(f"[repro] 无法使用设备 {args.device}: {type(exc).__name__}: {exc}", file=sys.stderr)
            print(
                "[repro] 该步骤必须在装了 torch_npu 的 NPU 机器上跑；换设备用 --device npu:<id>",
                file=sys.stderr,
            )
            return 2
        print(f"[repro] device: {device}（dump 是 CPU 加载的，权重与算子输入都会先搬到这个设备）")

    codes: list[int] = []
    summaries: list[dict] = []
    for rank in ranks:
        code, summary = _reproduce_rank(args, device, rank, quiet=quiet)
        codes.append(code)
        summaries.append(summary)

    if quiet or len(ranks) > 1:
        print()
        print(f"[repro] === 全 rank 汇总（{len(ranks)} 个 rank；权重来源随 rank 而异）===")
        for summary in summaries:
            if "differing" in summary:
                print(
                    f"[repro]   rank {summary['rank']:>2}: {summary['status']:<9} "
                    f"differing={summary['differing']}/{summary['compared']} "
                    f"max|d|={summary['max_abs']:.3e} w={summary.get('weight', '?')}"
                    f"{'' if 'weight_format' not in summary else '->' + summary['weight_format']} "
                    f"positions_from={summary.get('positions_from')}"
                )
            else:
                print(f"[repro]   rank {summary['rank']:>2}: {summary['status']}")
        if args.run_op:
            ok = [s for s in summaries if s.get("status") == "identical"]
            moved = [s for s in summaries if s.get("status") == "differs"]
            if moved:
                print(
                    f"[repro] RESULT(all-ranks): {len(moved)}/{len(summaries)} rank 上该 op 与行序相关 "
                    f"(ranks {[s['rank'] for s in moved]}) ⇒ 最小复现成立（在这些 rank 上）"
                )
            elif len(ok) == len(summaries):
                print(
                    f"[repro] RESULT(all-ranks): {len(ok)}/{len(summaries)} rank 在该调用形状下与行序无关 "
                    "⇒ matmul 侧已排除（真实权重仅 rank 0 有，其余 rank 用随机权重）；"
                    "差异只能出在紧随其后的 tensor_model_parallel_reduce_scatter / LSE 归约"
                )
    return max(codes) if codes else 0


if __name__ == "__main__":
    sys.exit(main())
