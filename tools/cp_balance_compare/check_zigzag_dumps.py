#!/usr/bin/env python3
"""Analyse the zigzag diagnostic dumps produced by sfa_v1.py.

用法（在跑完 A/B 之后，CPU 侧执行，不需要 NPU）::

    # 1) 核对 top-k：期望每行的有效值集合 == {0..valid-1}
    python tools/cp_balance_compare/check_zigzag_dumps.py --kind topk --dir /dev/shm/cp_balance_dump

    # 2) 逐 token 比较 B(CP_BALANCE=0) 与 C(CP_BALANCE=1) 的 KV cache
    python tools/cp_balance_compare/check_zigzag_dumps.py --kind kv --dir /dev/shm/cp_balance_dump

`topk` mode answers P1 of the precision triage: for prompts shorter than
``sparse_count`` the LightningIndexer is expected to emit either the identity
range ``[0..p]`` (A5/arch35: the ``validS2Len < topkCount_`` shortcut) or, on
arch22/A3, the same *set* in score-descending order.  A row whose valid set is
incomplete means the indexer told the SFA to attend to fewer positions than the
causal window requires.

`kv` mode answers P2: the dump stores the packed KV cache rows in natural token
order, so row ``p`` of a cpbal0 dump and row ``p`` of a cpbal1 dump are the same
token and can be compared byte for byte.  The first differing layer/token is the
first place where the two layouts disagree.

For a **multi-layer sweep** (``VLLM_ASCEND_CP_BALANCE_DUMP=kv:all``) use
``--summary-only``: it prints one compact table row per layer
(``ranks_diff / rows_differ / first_token / max|int8|``), names the first layer
whose KV already differs, and skips the per-pair detail.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - the target host always has torch
    print("this script needs torch (run it on the vLLM host)", file=sys.stderr)
    raise

NAME_RE = re.compile(
    r"(?P<kind>topk|kv)_cpbal(?P<cpbal>\d+)_layer(?P<layer>-?\d+)_rank(?P<rank>\d+)_pid(?P<pid>\d+)_(?P<ts>\d+)\.pt$"
)


def _load(path: str, warn: bool = True) -> dict | None:
    """Load one dump, or ``None`` when the file is unreadable.

    A round whose dump directory ran out of space leaves truncated files
    ("failed finding central directory"); one bad file must not abort the whole
    analysis, otherwise a partially complete sweep is worth nothing.
    """
    payload = None
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            if warn:
                print(f"[dump] skipping unreadable dump {os.path.basename(path)}: {exc}", file=sys.stderr)
            return None
    if not isinstance(payload, dict):
        if warn:
            print(f"[dump] skipping {os.path.basename(path)}: unexpected payload {type(payload)}", file=sys.stderr)
        return None
    return payload


def _as_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return []


def _iter_dumps(dump_dir: str, kind: str):
    pattern = os.path.join(dump_dir, f"{kind}_*.pt")
    for path in sorted(glob.glob(pattern)):
        match = NAME_RE.search(os.path.basename(path))
        if match is None or match.group("kind") != kind:
            continue
        yield path, match.groupdict()


def _latest_per_server(files):
    """Keep the newest dump per (layer, rank, cp_balance).

    Several rounds share one dump directory (the file name carries pid and
    timestamp), so an older round must not be mistaken for the current one.
    """
    latest: dict[tuple[int, int, str], tuple[str, dict]] = {}
    counts: dict[tuple[int, int, str], int] = defaultdict(int)
    for path, meta in files:
        key = (int(meta["layer"]), int(meta["rank"]), str(meta["cpbal"]))
        counts[key] += 1
        current = latest.get(key)
        if current is None or int(meta["ts"]) > int(current[1]["ts"]):
            latest[key] = (path, meta)
    return latest, counts


def _expected_windows(payload: dict) -> tuple[list[tuple[int, int, int]], str]:
    """Return [(row_prefix, q_len, kv_len)] per virtual batch, plus the shape name.

    Both call shapes use the same per-batch convention: query lengths are prefix
    sums (TND) and KV lengths are raw per-batch values (PA_BSND).  Therefore
    ``per_row_valid(row) = (kv_len - q_len) + (row - row_prefix) + 1`` = the
    token's absolute position + 1.
    """
    zigzag = bool(payload.get("zigzag_active"))
    if zigzag:
        q_prefix = _as_list(payload.get("actual_seq_lengths_query_zigzag"))
        kv_raw = _as_list(payload.get("actual_seq_lengths_key_zigzag"))
        name = "merged-2B"
    else:
        q_prefix = _as_list(payload.get("actual_seq_lengths_query"))
        kv_raw = _as_list(payload.get("actual_seq_lengths_key"))
        name = "continuous"
    if not q_prefix:
        return [], name
    windows: list[tuple[int, int, int]] = []
    prev = 0
    for q_end, kv_len in zip(q_prefix, kv_raw):
        windows.append((prev, max(0, int(q_end) - prev), int(kv_len)))
        prev = int(q_end)
    return windows, name


def _row_windows(payload: dict) -> tuple[list[int], list[int], str]:
    """Expand per-batch windows to per-row (absolute position + 1, kv_len)."""
    windows, name = _expected_windows(payload)
    positions: list[int] = []
    kv_lens: list[int] = []
    for row_prefix, q_len, kv_len in windows:
        delta = kv_len - q_len
        for row in range(q_len):
            positions.append(delta + row + 1)
            kv_lens.append(kv_len)
    return positions, kv_lens, name


def check_topk(args) -> int:
    files = list(_iter_dumps(args.dir, "topk"))
    if not files:
        print(f"[topk] no dump found under {args.dir}", file=sys.stderr)
        return 2
    latest, counts = _latest_per_server(files)
    failures = 0
    ignored = 0
    for key in sorted(latest):
        path, meta = latest[key]
        if counts[key] > 1:
            ignored += counts[key] - 1
            if not args.summary_only:
                print(f"[topk] {counts[key] - 1} older dump(s) for layer={key[0]} rank={key[1]} "
                      f"cpbal{key[2]} ignored (keeping the newest)")
        payload = _load(path)
        if payload is None:
            continue
        topk = payload.get("topk_indices")
        if not isinstance(topk, torch.Tensor):
            print(f"[topk] {os.path.basename(path)}: no topk_indices tensor, skipped")
            continue
        flat = topk.reshape(topk.shape[0], -1)
        rows, width = flat.shape
        positions, kv_lens, shape = _row_windows(payload)
        if not args.summary_only:
            print(
                f"\n[topk] {os.path.basename(path)}\n"
                f"       cp_balance={meta['cpbal']} layer={meta['layer']} rank={meta['rank']} "
                f"call={shape} rows={rows} width={width} windows={len(positions)}"
            )
        if not positions:
            if not args.summary_only:
                print("       no query metadata in dump; cannot verify rows")
            continue
        if len(positions) != rows and not args.summary_only:
            print(f"       WARNING: {len(positions)} expected rows vs {rows} dumped rows")
        checked = identity = set_only = bad = 0
        first_bad = None
        for row in range(min(rows, len(positions))):
            valid = min(int(positions[row]), width)
            entries = [int(x) for x in flat[row, :valid].tolist()]
            tail = [int(x) for x in flat[row, valid : min(width, valid + 8)].tolist()]
            checked += 1
            expected = list(range(valid))
            if entries == expected:
                identity += 1
                continue
            if sorted(entries) == expected:
                set_only += 1
                continue
            bad += 1
            if first_bad is None:
                missing = sorted(set(expected) - set(entries))
                extra = sorted(set(entries) - set(expected))
                first_bad = (row, valid, missing[:8], extra[:8], tail[:4])
        if args.summary_only:
            print(
                f"[topk] layer={meta['layer']} rank={meta['rank']} cpbal={meta['cpbal']} "
                f"call={shape} rows={rows} width={width} identity={identity} "
                f"set-only={set_only} mismatched={bad}"
            )
        else:
            print(f"       rows checked={checked} identity={identity} set-only={set_only} mismatched={bad}")
        if bad:
            failures += 1
            row, valid, missing, extra, tail = first_bad
            print(
                f"       first mismatch: row={row} valid={valid} "
                f"missing={missing} extra={extra} tail_after_valid={tail}"
            )
        elif not args.summary_only:
            print("       OK: every row's valid prefix is exactly the causal window")
    if ignored and args.summary_only:
        print(f"[topk] {ignored} older dump(s) ignored (newest per (layer, rank, cp_bal) wins)")
    print()
    if failures:
        print(f"[topk] RESULT: {failures} dump(s) with mismatched rows -> P1 violated (indexer/plumbing)")
        return 1
    print("[topk] RESULT: P1 holds on all dumps (top-k == causal window; indexer selection is not the cause)")
    return 0


def _kv_rows(payload: dict) -> np.ndarray | None:
    """Packed KV rows as a 2-D uint8 array (row = token), or ``None``.

    Vectorised on purpose: a full-layer sweep is 78 layers x ranks pairs, and the
    old per-row ``bytes(tensor.tolist())`` loop made the analysis take minutes.
    """
    kv = payload.get("kv_nat")
    if not isinstance(kv, torch.Tensor):
        return None
    return kv.contiguous().view(torch.uint8).numpy()


def _row_diff_mask(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Boolean per-row "these two rows differ" mask over the common row count."""
    n = min(left.shape[0], right.shape[0])
    return (left[:n] != right[:n]).any(axis=1)


def _max_nope_diff(left: np.ndarray, right: np.ndarray, diff_rows: list[int]) -> int:
    """Max ``|int8|`` difference over the packed nope part of the differing rows.

    The first 512 bytes of a packed row are the quantized nope part, so the
    magnitude is readable directly in quantization steps.
    """
    if not diff_rows:
        return 0
    rows = np.asarray(diff_rows, dtype=np.int64)
    left_part = left[rows, :512].astype(np.int16)
    right_part = right[rows, :512].astype(np.int16)
    return int(np.abs(left_part - right_part).max())


def _fp_diff(left: dict, right: dict) -> tuple[int | None, int | None, float, float]:
    """Row-wise comparison of the pre-quantization KV (``kv_fp_nat``).

    Returns ``(differing_rows, first_row, max_abs_delta, max_rel_delta)``;
    ``(None, None, 0.0, 0.0)`` when the dump has no FP copy.  This is the
    sensitive comparison: the packed cache is int8/fp8 quantized, so two
    different FP values can share a byte and a real divergence would hide.
    """
    left_fp = left.get("kv_fp_nat")
    right_fp = right.get("kv_fp_nat")
    if not (isinstance(left_fp, torch.Tensor) and isinstance(right_fp, torch.Tensor)):
        return None, None, 0.0, 0.0
    if left_fp.shape != right_fp.shape:
        return None, None, 0.0, 0.0
    lf = left_fp.numpy().astype(np.float32)
    rf = right_fp.numpy().astype(np.float32)
    delta = np.abs(lf - rf)
    per_row = delta.reshape(delta.shape[0], -1).max(axis=1)
    changed = per_row > 0
    rows = int(changed.sum())
    if not rows:
        return 0, None, 0.0, 0.0
    first = int(np.argmax(changed))
    max_abs = float(delta.max())
    scale = float(np.abs(lf).max()) or 1.0
    return rows, first, max_abs, max_abs / scale


def _summary_verdict(per_layer: dict[int, dict]) -> tuple[int | None, str]:
    """First layer that differs, judged on the FP copy when it exists."""
    use_fp = any(stats["fp_rows"] for stats in per_layer.values())
    for layer in sorted(per_layer):
        stats = per_layer[layer]
        if not stats["compared"]:
            continue
        if use_fp:
            if any(rows for rows in stats["fp_rows"]):
                return layer, "fp"
        elif stats["differ"]:
            return layer, "int8"
    return None, "fp" if use_fp else "int8"


def _print_kv_summary(per_layer: dict[int, dict], ignored: int) -> None:
    """One compact table per comparison for a whole sweep, plus the verdict."""
    print()
    if ignored:
        print(f"[kv] {ignored} older dump(s) ignored (newest per (layer, rank, cp_bal) wins)")

    print(f"[kv/int8] {'layer':>5}  {'ranks_diff':>10}  {'rows_differ':>13}  {'first_token':>11}  {'max|d|':>7}")
    for layer in sorted(per_layer):
        stats = per_layer[layer]
        if not stats["compared"]:
            continue
        if not stats["rows"]:
            print(f"[kv/int8] {layer:>5}  n/a (no kv_nat in dump -- int8 copy skipped on this site)")
            continue
        rows = stats["rows"]
        span = f"{min(rows)}-{max(rows)}" if rows else "-"
        first = "-" if stats["first"] is None else str(stats["first"])
        suffix = f"  [{stats['no_int8']} rank(s) without kv_nat]" if stats["no_int8"] else ""
        print(f"[kv/int8] {layer:>5}  {stats['differ']:>4}/{stats['compared']:<5}  {span:>13}  "
              f"{first:>11}  {stats['max']:>7}{suffix}")

    have_fp = any(stats["fp_rows"] for stats in per_layer.values())
    if have_fp:
        print(f"[kv/fp  ] {'layer':>5}  {'ranks_diff':>10}  {'rows_differ':>13}  {'first_token':>11}  "
              f"{'max|d|':>10}  {'rel':>9}")
        for layer in sorted(per_layer):
            stats = per_layer[layer]
            if not stats["compared"]:
                continue
            rows = stats["fp_rows"]
            span = f"{min(rows)}-{max(rows)}" if rows else "-"
            first = "-" if stats["fp_first"] is None else str(stats["fp_first"])
            print(f"[kv/fp  ] {layer:>5}  {stats['fp_differ']:>4}/{stats['compared']:<5}  {span:>13}  "
                  f"{first:>11}  {stats['fp_max_abs']:>10.3e}  {stats['fp_max_rel']:>9.2e}")

    first_layer, which = _summary_verdict(per_layer)
    if first_layer is None:
        print(f"[kv] FIRST DIVERGENCE ({which}): none -- every compared (layer, rank) is bit-identical")
        return
    stats = per_layer[first_layer]
    if which == "fp":
        detail = (
            f"{stats['fp_differ']}/{stats['compared']} ranks, first token {stats['fp_first']}, "
            f"max|d|={stats['fp_max_abs']:.3e} (rel {stats['fp_max_rel']:.2e})"
        )
    else:
        detail = (
            f"{stats['differ']}/{stats['compared']} ranks, first token {stats['first']}, "
            f"max|int8| {stats['max']}"
        )
    print(f"[kv] FIRST DIVERGENCE ({which}): layer {first_layer} is the first layer whose KV already differs ({detail})")
    if first_layer == 0:
        print(
            "[kv] -> layer 0 的 KV 直接来自 embedding（与排布无关），所以分歧不是从上一层传进来的，"
            "而是写 KV / rope / 布局本身在这一层就不一样"
        )
    else:
        print(
            f"[kv] -> 该层 KV 是「该层输入隐状态」的投影，所以分歧是在上一层（layer {first_layer - 1}）"
            "的输出里进入的；下一步按这一层去定位是哪一步先不等"
        )


def check_kv(args) -> int:
    files = list(_iter_dumps(args.dir, "kv"))
    if not files:
        print(f"[kv] no dump found under {args.dir}", file=sys.stderr)
        return 2
    latest, counts = _latest_per_server(files)
    by_layer: dict[tuple[int, int], dict[str, tuple[str, dict]]] = defaultdict(dict)
    ignored = 0
    # Loading every dump is the slow part of a full sweep (GBs); report progress
    # on stderr so a redirected run does not look like a hang.
    total = len(latest)
    for index, key in enumerate(sorted(latest), 1):
        layer, rank, cpbal = key
        path, meta = latest[key]
        if counts[key] > 1:
            ignored += counts[key] - 1
            if not args.summary_only:
                print(f"[kv] {counts[key] - 1} older dump(s) for layer={layer} rank={rank} "
                      f"cpbal{cpbal} ignored (keeping the newest)")
        payload = _load(path, warn=not args.summary_only)
        if payload is not None:
            by_layer[(layer, rank)][cpbal] = (path, payload)
        if index % 64 == 0 or index == total:
            print(f"[kv] loaded {index}/{total} dumps", file=sys.stderr)
    per_layer: dict[int, dict] = defaultdict(
        lambda: {
            "compared": 0, "differ": 0, "rows": [], "first": None, "max": 0, "no_int8": 0,
            "fp_differ": 0, "fp_rows": [], "fp_first": None, "fp_max_abs": 0.0, "fp_max_rel": 0.0,
        }
    )
    failures = 0
    last_reported_layer = None
    for (layer, rank), variants in sorted(by_layer.items()):
        if len(variants) < 2:
            if not args.summary_only:
                print(
                    f"[kv] layer={layer} rank={rank}: only cp_balance="
                    f"{sorted(variants)} dumped, need both 0 (B) and 1 (C) to compare"
                )
            continue
        keys = sorted(variants)
        left_key, right_key = keys[0], keys[-1]
        left_path, left = variants[left_key]
        right_path, right = variants[right_key]
        # The packed (int8/fp8) copy is best effort: a site whose NPU
        # index_select fails produces FP-only dumps, which are still decisive.
        int8_available = isinstance(left.get("kv_nat"), torch.Tensor) and isinstance(
            right.get("kv_nat"), torch.Tensor
        )
        if int8_available:
            left_rows = _kv_rows(left)
            right_rows = _kv_rows(right)
            mask = _row_diff_mask(left_rows, right_rows)
            n = int(mask.size)
            diff_rows = np.nonzero(mask)[0].tolist()
        else:
            left_rows = right_rows = np.zeros((0, 0), dtype=np.uint8)
            n = 0
            diff_rows = []
        stats = per_layer[layer]
        stats["compared"] += 1
        if int8_available:
            stats["rows"].append(len(diff_rows))
        else:
            stats["no_int8"] += 1
        fp_rows, fp_first, fp_max_abs, fp_max_rel = _fp_diff(left, right)
        if fp_rows:
            stats["fp_differ"] += 1
            stats["fp_rows"].append(fp_rows)
            stats["fp_first"] = fp_first if stats["fp_first"] is None else min(stats["fp_first"], fp_first)
            stats["fp_max_abs"] = max(stats["fp_max_abs"], fp_max_abs)
            stats["fp_max_rel"] = max(stats["fp_max_rel"], fp_max_rel)
        elif fp_rows == 0:
            stats["fp_rows"].append(0)
        if not args.summary_only:
            print(
                f"\n[kv] layer={layer} rank={rank} cpbal{left_key} vs cpbal{right_key}: "
                + (
                    f"rows {left_rows.shape[0]} vs {right_rows.shape[0]}, differing rows={len(diff_rows)}"
                    if int8_available
                    else "no kv_nat in dump (int8 copy skipped on this site)"
                )
                + (
                    f" | fp rows={fp_rows} first={fp_first} max|d|={fp_max_abs:.3e} rel={fp_max_rel:.2e}"
                    if fp_rows is not None
                    else " | (no kv_fp_nat in dump)"
                )
            )
        if int8_available and left_rows.shape[0] != right_rows.shape[0] and not args.summary_only:
            print("     WARNING: row counts differ (num_actual_tokens/shape mismatch)")
        if diff_rows:
            failures += 1
            stats["differ"] += 1
            first = diff_rows[0]
            stats["first"] = first if stats["first"] is None else min(stats["first"], first)
            max_diff = _max_nope_diff(left_rows, right_rows, diff_rows)
            stats["max"] = max(stats["max"], max_diff)
            if not args.summary_only:
                width = int(left_rows.shape[1])
                byte_diff = int((left_rows[first] != right_rows[first]).sum())
                print(
                    f"     first differing token={first} bytes_differing={byte_diff}/{width} "
                    f"max|int8(nope) diff|={float(_max_nope_diff(left_rows, right_rows, [first])):.1f}"
                    f"  all-differing-rows max|int8|={max_diff}"
                )
                print(f"     B file: {os.path.basename(left_path)}")
                print(f"     C file: {os.path.basename(right_path)}")
        elif not args.summary_only:
            print("     OK: identical packed KV for every token")
        if args.summary_only and layer != last_reported_layer:
            last_reported_layer = layer
            print(f"[kv] compared layer {layer}", file=sys.stderr)
    _print_kv_summary(per_layer, ignored)
    print()
    fp_layers = sorted(layer for layer, stats in per_layer.items() if any(stats["fp_rows"]))
    if failures or fp_layers:
        shown = ", ".join(str(layer) for layer in fp_layers[:8])
        if len(fp_layers) > 8:
            shown += f", ... (+{len(fp_layers) - 8} more)"
        print(f"[kv] RESULT: {failures} (layer, rank) pair(s) differ in the packed int8 copy; "
              f"the FP copy differs on {len(fp_layers)} layer(s): {shown or '-'} -> P2 violated "
              "(a token's KV content is not layout invariant)")
        return 1
    print("[kv] RESULT: P2 holds on the dumped layers (per-token KV identical across layouts)")
    return 0


def main(argv=None) -> int:
    try:
        # Keep the Chinese summary in the docstring from crashing on hosts with
        # a legacy (non-UTF-8) console encoding.
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="/dev/shm/cp_balance_dump", help="dump directory")
    parser.add_argument("--kind", choices=("topk", "kv", "both"), default="both")
    parser.add_argument("--list", action="store_true", help="only list the dumps found")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="print only the compact tables (kv: per-layer rows/ranks/first token/max|int8|; "
        "topk: one line per dump) plus the RESULT lines -- use this for a multi-layer sweep",
    )
    args = parser.parse_args(argv)

    if args.list or not os.path.isdir(args.dir):
        for path in sorted(glob.glob(os.path.join(args.dir, "*.pt"))):
            print(os.path.basename(path))
        if not os.path.isdir(args.dir):
            print(f"dump directory does not exist: {args.dir}", file=sys.stderr)
            return 2
        return 0

    rc = 0
    if args.kind in ("topk", "both"):
        rc |= check_topk(args)
    if args.kind in ("kv", "both"):
        rc |= check_kv(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
