#!/usr/bin/env python3
"""Analyse the zigzag diagnostic dumps produced by sfa_v1.py.

用法（在跑完 A/B 之后，CPU 侧执行，不需要 NPU）::

    # 1) 核对 top-k：期望每行的有效值集合 == {0..valid-1}
    python tools/cp_balance_compare/check_zigzag_dumps.py --kind topk --dir /dev/shm/cp_balance_dump

    # 2) 逐 token 比较 B(CP_BALANCE=0) 与 C(CP_BALANCE=1) 的 KV cache
    python tools/cp_balance_compare/check_zigzag_dumps.py --kind kv --dir /dev/shm/cp_balance_dump

    # 3) 逐 token 比较 attention 的输入/输出（B vs C，按 token 位置对齐）
    python tools/cp_balance_compare/check_zigzag_dumps.py --kind act --dir /root/cp_probe --summary-only

`act` mode answers "which step diverges first" at full precision: for every
traced layer the dump stores the attention *input* (the previous layer's output)
and the attention *output* (this layer's attention contribution), each keyed by
**global token position** -- never by row, because under zigzag a rank owns
``[prev_block, next_block]`` while the continuous slice owns
``[local_start, local_end)``.  ``in`` equal + ``out`` different pins the
divergence to attention; ``out`` of layer L equal + ``in`` of layer L+1
different pins it to the MoE/MLP in between.  Unlike the KV copy this is not
quantized, so a single-ULP difference is visible.

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

Each dump also carries ``kv_fp_nat``: the same packed rows taken from
``fused_kv_no_split`` *before* the cache scatter.  That is what makes the analysis
work on sites where the NPU cache readback (``index_select``) fails -- but it is
the **same packed numbers** (fp8 e4m3 + e8m0 scales on a sparse-C8 site), so it
does **not** remove quantization: sub-fp8 differences stay invisible and the first
difference it reports is only an upper bound in depth.

For a **multi-layer sweep** (``VLLM_ASCEND_CP_BALANCE_DUMP=kv:all``) use
``--summary-only``: it prints one compact table row per layer
(``ranks_diff / rows_differ / first_token / max|d| / rel``), names the first layer
whose KV already differs, and skips the per-pair detail (progress goes to stderr).
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
    r"(?P<kind>topk|kv|actin|actout|mlpin|mlpout|guq|guout|dnin|dnq)_cpbal(?P<cpbal>\d+)_layer(?P<layer>-?\d+)"
    r"_rank(?P<rank>\d+)_pid(?P<pid>\d+)_(?P<ts>\d+)\.pt$"
)

# ``act`` is one logical kind with several ops in the file name (attention in/out,
# the MLP boundary of the same layers, and the quantized activation each MLP GEMM
# consumes); the other kinds map to themselves so ``--kind kv``/``--kind act``
# glob exactly their own files.
_KIND_GLOBS = {
    "topk": ("topk_*.pt",),
    "kv": ("kv_*.pt",),
    "act": (
        "actin_*.pt",
        "actout_*.pt",
        "mlpin_*.pt",
        "guq_*.pt",
        "guout_*.pt",
        "dnin_*.pt",
        "dnq_*.pt",
        "mlpout_*.pt",
    ),
}
# Row order inside one layer, i.e. the order the numbers flow: the layer input
# and its attention output, then the MLP pipeline -- its input, the activation
# quantization the gate_up GEMM consumes, that GEMM's output, the down_proj
# input, the quantization the down_proj GEMM consumes, and finally the MLP
# output.  The earliest differing row is where the divergence is born; the pair
# ``(qin, GEMM out)`` is what separates "the quantization changed" from "the
# GEMM kernel changed".
_ACT_OP_ORDER = {
    "in": 0,
    "out": 1,
    "mlp_in": 2,
    "gu_q": 3,
    "gu_out": 4,
    "dn_in": 5,
    "dn_q": 6,
    "mlp_out": 7,
}
_FILE_KIND_OP = {
    "actin": "in",
    "actout": "out",
    "mlpin": "mlp_in",
    "guq": "gu_q",
    "guout": "gu_out",
    "dnin": "dn_in",
    "dnq": "dn_q",
    "mlpout": "mlp_out",
}


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
    """Yield ``(path, meta)`` for every dump of ``kind``.

    ``act`` is one logical kind stored under two file kinds (``actin`` /
    ``actout``), so it matches both; every other kind matches itself.
    """
    for pattern in _KIND_GLOBS.get(kind, (f"{kind}_*.pt",)):
        for path in sorted(glob.glob(os.path.join(dump_dir, pattern))):
            match = NAME_RE.search(os.path.basename(path))
            if match is None:
                continue
            file_kind = match.group("kind")
            if file_kind == kind or (kind == "act" and file_kind in _FILE_KIND_OP):
                yield path, match.groupdict()


def _latest_per_server(files, fields: tuple[str, ...] = ("layer", "rank", "cpbal")):
    """Keep the newest dump per key.

    Several rounds share one dump directory (the file name carries pid and
    timestamp), so an older round must not be mistaken for the current one.  The
    key is ``(layer, rank, cp_balance)`` by default; the activation trace also
    needs the op (the file-name kind carries it) because one layer has both an
    ``in`` and an ``out`` sample.
    """
    latest: dict[tuple, tuple[str, dict]] = {}
    counts: dict[tuple, int] = defaultdict(int)
    for path, meta in files:
        key = tuple(meta[field] if field != "layer" else int(meta["layer"]) for field in fields)
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


def _topk_positions(payload: dict, rows: int) -> np.ndarray | None:
    """Global token position of every row of a top-k dump, or ``None``.

    Old dumps have no ``positions`` field (written before the activation trace
    existed); without it the rows of B and C cannot be matched up (the same row
    index is a different token in the two layouts), so the cross-layout
    comparison is skipped instead of compared by row.
    """
    positions = payload.get("positions")
    if not isinstance(positions, torch.Tensor):
        return None
    pos = positions.detach().to("cpu").to(torch.int64).numpy()[:rows]
    return pos if pos.size else None


def _topk_global_table(paths: list[str]) -> tuple[dict[int, np.ndarray], int] | None:
    """Pool every rank of one layout into ``{token position: index list}``.

    Ranks must be pooled, not paired rank-by-rank: under zigzag rank ``r`` owns
    ``[block r, block 15-r]`` while the continuous layout gives it
    ``[256r, 256r+256)``, so rank 1's token sets do not even intersect.  Only the
    whole layout covers the sequence.
    """
    table: dict[int, np.ndarray] = {}
    width = 0
    for path in paths:
        payload = _load(path)
        if payload is None:
            continue
        topk = payload.get("topk_indices")
        if not isinstance(topk, torch.Tensor) or topk.ndim < 2:
            continue
        rows = int(topk.shape[0])
        positions = _topk_positions(payload, rows)
        if positions is None:
            continue
        flat = topk.reshape(rows, -1)
        width = max(width, int(flat.shape[1]))
        array = flat.numpy()
        limit = int(payload.get("num_actual_tokens") or 0)
        for index, position in enumerate(positions):
            if position < 0 or (limit > 0 and position >= limit):
                continue
            table[int(position)] = array[index]
    if not table:
        return None
    return table, width


def _compare_topk_layouts(layer: int, left: tuple, right: tuple) -> dict:
    """Compare the two layouts' index lists token by token."""
    table_l, width_l = left
    table_r, width_r = right
    width = min(width_l, width_r)
    common = sorted(set(table_l) & set(table_r))
    # Positions are cache slots (see `_act_table`): normalise by the common base
    # so the reported token indices are natural-stream indices.
    base = common[0] if common else 0
    set_diff = order_diff = 0
    first_set = first_order = None
    sample = None
    for position in common:
        row_l = table_l[position][:width]
        row_r = table_r[position][:width]
        if np.array_equal(row_l, row_r):
            continue
        # Only non-negative entries are real KV positions.
        list_l = [int(x) for x in row_l[row_l >= 0]]
        list_r = [int(x) for x in row_r[row_r >= 0]]
        if sorted(list_l) != sorted(list_r):
            set_diff += 1
            if first_set is None:
                first_set = position - base
                if sample is None:
                    sample = (position - base, sorted(set(list_l) - set(list_r))[:8],
                              sorted(set(list_r) - set(list_l))[:8])
        else:
            order_diff += 1
            if first_order is None:
                first_order = position - base
                if sample is None:
                    sample = (position - base, list_l[:6], list_r[:6])
    return {
        "layer": layer, "compared": len(common), "set_diff": set_diff, "order_diff": order_diff,
        "first_set": first_set, "first_order": first_order, "sample": sample,
        "covered_l": len(table_l), "covered_r": len(table_r),
    }


def _topk_cross_config(args, latest: dict) -> int:
    """Second pass of ``--kind topk``: B vs C index lists, keyed by token.

    Separate from the per-dump causal-window check on purpose: that one is about
    the indexer's *contract*, this one is about layout invariance, and they fail
    for different reasons.
    """
    by_layer: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for key in sorted(latest):
        layer, _rank, cpbal = key
        by_layer[layer][cpbal].append(latest[key][0])
    rows: list[dict] = []
    for layer, variants in sorted(by_layer.items()):
        if len(variants) < 2:
            continue
        left = _topk_global_table(variants[sorted(variants)[0]])
        right = _topk_global_table(variants[sorted(variants)[-1]])
        if left is None or right is None:
            continue
        rows.append(_compare_topk_layouts(layer, left, right))
    if not rows:
        print("[topk/cross] no dump carries per-row positions -> cross-layout order comparison skipped")
        print("[topk/cross] (set VLLM_ASCEND_CP_BALANCE_DUMP=topk:<layer> and re-run the round to get them)")
        return 0
    print()
    print(f"[topk/cross] {'layer':>5}  {'compared':>8}  {'covered_B':>9}  {'covered_C':>9}  "
          f"{'set_diff':>8}  {'order_diff':>10}  {'first_set':>9}  {'first_order':>11}")
    set_diffs = order_diffs = 0
    for stats in rows:
        set_diffs += stats["set_diff"]
        order_diffs += stats["order_diff"]
        print(f"[topk/cross] {stats['layer']:>5}  {stats['compared']:>8}  {stats['covered_l']:>9}  "
              f"{stats['covered_r']:>9}  {stats['set_diff']:>8}  {stats['order_diff']:>10}  "
              f"{'-' if stats['first_set'] is None else stats['first_set']:>9}  "
              f"{'-' if stats['first_order'] is None else stats['first_order']:>11}")
    if rows and rows[0]["sample"] is not None:
        position, left_part, right_part = rows[0]["sample"]
        print(f"[topk/cross] first differing token={position}: B={left_part} C={right_part}")
    if set_diffs:
        print(f"[topk/cross] RESULT: {set_diffs} token(s) selected a DIFFERENT SET of positions in B and C "
              "-> the indexer itself is layout dependent (a real bug, not just ordering)")
        return 1
    if order_diffs:
        print(f"[topk/cross] RESULT: every set is identical, but {order_diffs} token(s) present the same set "
              "in a different ORDER. The SFA kernel consumes the list in the given order, so this alone can "
              "produce different roundings -> prime suspect for the numeric divergence")
        return 0
    print("[topk/cross] RESULT: identical sets AND identical order for every compared token "
          "-> the indexer output is layout invariant")
    return 0


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
    cross = _topk_cross_config(args, latest)
    print()
    if failures or cross:
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


def _as_float32(tensor: torch.Tensor) -> np.ndarray | None:
    """float32 numpy view of a dump tensor, for *any* dtype (fp8 included).

    numpy has no float8 dtype, so the conversion must go through torch
    (``t.numpy()`` raises "Got unsupported ScalarType Float8_e4m3fn").  A
    sparse-C8 site packs the KV as fp8 (e4m3) + e8m0 scales, which is exactly
    what ``kv_fp_nat``/``kv_nat`` hold there.
    """
    try:
        return tensor.detach().to("cpu").float().numpy()
    except Exception as exc:
        print(f"[dump] cannot convert {getattr(tensor, 'dtype', '?')} to float32: {exc}", file=sys.stderr)
        return None


def _fp_diff(left: dict, right: dict) -> dict | None:
    """Compare the source-side packed KV copy (``kv_fp_nat``) two ways.

    The stored copy is *quantized* (fp8 e4m3 on a sparse-C8 site), so two
    different pre-quantization values can land on the same stored number -- and,
    conversely, a difference far below one quantization step usually leaves the
    stored *values* identical while still flipping the **sign of a quantized
    zero**.  Both signals matter and they mean different things, so they are
    reported separately:

    * ``rows_value`` / ``first_value`` / ``max_abs``: the stored numbers differ.
      This is a divergence at (or above) the quantization step -- a real content
      difference.
    * ``rows_bytes`` / ``first_bytes`` / ``byte_only``: the stored *bytes* differ.
      When ``rows_value`` is 0 this can only be a sign flip of a quantized zero
      (e4m3 encodes every finite value once, except ``+0``/``-0``): a
      **sub-quantization** divergence, i.e. proof that the pre-quantization
      values differ while agreeing to within one step.  That is the sensitive
      detector the value comparison alone cannot provide.

    Returns ``None`` when a dump has no comparable FP copy.
    """
    left_fp = left.get("kv_fp_nat")
    right_fp = right.get("kv_fp_nat")
    if not (isinstance(left_fp, torch.Tensor) and isinstance(right_fp, torch.Tensor)):
        return None
    if left_fp.shape != right_fp.shape or left_fp.ndim != 2:
        return None
    lf = _as_float32(left_fp)
    rf = _as_float32(right_fp)
    if lf is None or rf is None:
        return None
    rows = int(lf.shape[0])
    elements = int(lf.shape[1]) if lf.ndim == 2 else 0
    if rows == 0 or elements == 0:
        return None
    raw_l = left_fp.detach().to("cpu").contiguous().view(torch.uint8).numpy()
    raw_r = right_fp.detach().to("cpu").contiguous().view(torch.uint8).numpy()
    itemsize = max(1, raw_l.shape[1] // elements)
    # One byte is one element for fp8, but fp16/bf16/fp32 rows are wider: fold
    # the byte mask back to one entry per element so it can be combined with the
    # value mask.
    byte_diff = (raw_l != raw_r).reshape(rows, elements, itemsize).any(axis=2)
    # NaN == NaN must not count as a value difference (a NaN on one side only
    # still does: the two layouts then really hold different numbers).
    both_nan = np.isnan(lf) & np.isnan(rf)
    value_diff = (lf != rf) & ~both_nan
    per_row_bytes = byte_diff.any(axis=1)
    per_row_value = value_diff.any(axis=1)
    delta = np.abs(lf - rf)
    delta[both_nan] = 0.0
    # A NaN on one side only stays NaN in `delta`; ``delta.max()`` would then be
    # NaN, and ``max(0.0, nan)`` returns 0.0 in plain Python -- which is how a
    # real divergence once got reported as "max|d| = 0.000e+00".  Keep the
    # magnitude meaningful: the finite maximum when there is one, ``inf`` when
    # the only differences are NaN-vs-number, and count those pairs separately.
    finite_diff = np.isfinite(delta) & value_diff
    unilateral_nan = int((np.isnan(lf) ^ np.isnan(rf)).sum())
    if finite_diff.any():
        max_abs = float(delta[finite_diff].max())
    elif per_row_value.any():
        max_abs = float("inf")
    else:
        max_abs = 0.0
    stats = {
        "rows_bytes": int(per_row_bytes.sum()),
        "rows_value": int(per_row_value.sum()),
        "first_bytes": int(np.argmax(per_row_bytes)) if per_row_bytes.any() else None,
        "first_value": int(np.argmax(per_row_value)) if per_row_value.any() else None,
        "max_abs": max_abs,
        "max_rel": 0.0,
        "unilateral_nan": unilateral_nan,
        # elements that differ as bytes but not as values: sign of zero (or a
        # NaN payload) -- the fingerprint of a sub-quantization difference.
        "byte_only": int((byte_diff & ~value_diff).sum()),
    }
    if stats["max_abs"]:
        scale = float(np.nanmax(np.abs(lf))) or 1.0
        stats["max_rel"] = stats["max_abs"] / scale
    return stats

def _summary_verdict(per_layer: dict[int, dict]) -> tuple[int | None, str]:
    """First layer whose KV differs, judged on the FP copy when it exists.

    A value-level difference wins over a byte-only (sub-quantization) one: the
    verdict reports the earliest layer with a *content* difference when there is
    one, and otherwise the earliest layer whose stored bytes differ at all.
    """
    use_fp = any(stats["fp_present"] for stats in per_layer.values())
    for layer in sorted(per_layer):
        stats = per_layer[layer]
        if not stats["compared"]:
            continue
        if use_fp:
            if any(rows for rows in stats["fp_value_rows"]):
                return layer, "fp/value"
        elif stats["differ"]:
            return layer, "int8"
    if use_fp:
        for layer in sorted(per_layer):
            if any(rows for rows in per_layer[layer]["fp_byte_rows"]):
                return layer, "fp/bytes"
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

    have_fp = any(stats["fp_present"] for stats in per_layer.values())
    if have_fp:
        print(f"[kv/fp  ] {'layer':>5}  {'val_ranks':>9}  {'rows_val':>10}  {'first_val':>9}  "
              f"{'rows_byte':>10}  {'first_byte':>10}  {'max|d|':>10}  {'rel':>9}  {'byte_only':>9}")
        for layer in sorted(per_layer):
            stats = per_layer[layer]
            if not stats["compared"]:
                continue
            val_rows = stats["fp_value_rows"]
            byte_rows = stats["fp_byte_rows"]
            val_span = f"{min(val_rows)}-{max(val_rows)}" if val_rows else "-"
            byte_span = f"{min(byte_rows)}-{max(byte_rows)}" if byte_rows else "-"
            first_val = "-" if stats["fp_first_value"] is None else str(stats["fp_first_value"])
            first_byte = "-" if stats["fp_first_byte"] is None else str(stats["fp_first_byte"])
            suffix = f"  [{stats['fp_nan_only']} NaN-only pair(s)]" if stats["fp_nan_only"] else ""
            print(f"[kv/fp  ] {layer:>5}  {stats['fp_differ']:>4}/{stats['compared']:<4}  {val_span:>10}  "
                  f"{first_val:>9}  {byte_span:>10}  {first_byte:>10}  "
                  f"{stats['fp_max_abs']:>10.3e}  {stats['fp_max_rel']:>9.2e}  {stats['fp_byte_only']:>9}{suffix}")

    first_layer, which = _summary_verdict(per_layer)
    if first_layer is None:
        print(f"[kv] FIRST DIVERGENCE ({which}): none -- every compared (layer, rank) is bit-identical")
        return
    stats = per_layer[first_layer]
    if which == "fp/value":
        detail = (
            f"{stats['fp_differ']}/{stats['compared']} ranks, first token {stats['fp_first_value']}, "
            f"max|d|={stats['fp_max_abs']:.3e} (rel {stats['fp_max_rel']:.2e})"
        )
        print(f"[kv] FIRST DIVERGENCE (fp/value): layer {first_layer} is the first layer whose KV "
              f"*values* already differ ({detail}) -> a real content difference (>= 1 quantization step)")
    elif which == "fp/bytes":
        detail = (
            f"{stats['fp_differ']}/{stats['compared']} ranks, first token {stats['fp_first_byte']}, "
            f"rows_byte={min(stats['fp_byte_rows'])}-{max(stats['fp_byte_rows'])}, "
            f"byte_only elements={stats['fp_byte_only']}"
        )
        print(f"[kv] FIRST DIVERGENCE (fp/bytes): layer {first_layer} is the first layer whose stored KV "
              f"*bytes* differ ({detail})")
        print(
            "[kv] -> 但 rows_val 全 0、max|d|=0：所有层的 KV **数值完全相同**，差异只出现在量化零点"
            "的符号位（e4m3 里每个有限值只有一个编码，±0 除外）→ 这是**亚量化（sub-quantization）**"
            "分歧：量化前的值确实不同，但小于一个 fp8 量化步。"
        )
    else:
        detail = (
            f"{stats['differ']}/{stats['compared']} ranks, first token {stats['first']}, "
            f"max|int8| {stats['max']}"
        )
        print(f"[kv] FIRST DIVERGENCE (int8): layer {first_layer} is the first layer whose KV already "
              f"differs ({detail})")
    if first_layer == 0:
        print(
            "[kv] -> layer 0 的 KV 直接来自 embedding（与排布无关），所以分歧不是从上一层传进来的，"
            "而是写 KV / rope / 布局本身在这一层就不一样"
        )
    else:
        print(
            f"[kv] -> 该层 KV 是「该层输入隐状态」的投影，所以分歧是在上一层（layer {first_layer - 1}）"
            "的输出里进入的；下一步用 probe（op 级、全精度）定位是哪一步先不等"
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
            "fp_present": 0, "fp_differ": 0, "fp_byte_rows": [], "fp_value_rows": [],
            "fp_first_byte": None, "fp_first_value": None,
            "fp_max_abs": 0.0, "fp_max_rel": 0.0, "fp_byte_only": 0, "fp_nan_only": 0,
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
        fp = _fp_diff(left, right)
        if fp is not None:
            stats["fp_present"] += 1
            stats["fp_byte_rows"].append(fp["rows_bytes"])
            stats["fp_value_rows"].append(fp["rows_value"])
            if fp["first_bytes"] is not None:
                stats["fp_first_byte"] = (
                    fp["first_bytes"] if stats["fp_first_byte"] is None
                    else min(stats["fp_first_byte"], fp["first_bytes"])
                )
            if fp["rows_value"]:
                stats["fp_differ"] += 1
                stats["fp_first_value"] = (
                    fp["first_value"] if stats["fp_first_value"] is None
                    else min(stats["fp_first_value"], fp["first_value"])
                )
                stats["fp_max_abs"] = max(stats["fp_max_abs"], fp["max_abs"])
                stats["fp_max_rel"] = max(stats["fp_max_rel"], fp["max_rel"])
            stats["fp_byte_only"] = max(stats["fp_byte_only"], fp["byte_only"])
            stats["fp_nan_only"] = max(stats["fp_nan_only"], fp["unilateral_nan"])
        if not args.summary_only:
            print(
                f"\n[kv] layer={layer} rank={rank} cpbal{left_key} vs cpbal{right_key}: "
                + (
                    f"rows {left_rows.shape[0]} vs {right_rows.shape[0]}, differing rows={len(diff_rows)}"
                    if int8_available
                    else "no kv_nat in dump (int8 copy skipped on this site)"
                )
                + (
                    f" | fp rows_value={fp['rows_value']} first_value={fp['first_value']} "
                    f"max|d|={fp['max_abs']:.3e} rel={fp['max_rel']:.2e} | "
                    f"rows_bytes={fp['rows_bytes']} first_byte={fp['first_bytes']} "
                    f"byte_only={fp['byte_only']} nan_only={fp['unilateral_nan']}"
                    if fp is not None
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
    value_layers = sorted(layer for layer, stats in per_layer.items() if any(stats["fp_value_rows"]))
    byte_layers = sorted(layer for layer, stats in per_layer.items() if any(stats["fp_byte_rows"]))
    if failures or value_layers:
        shown = ", ".join(str(layer) for layer in (value_layers or byte_layers)[:8])
        if len(value_layers or byte_layers) > 8:
            shown += f", ... (+{len(value_layers or byte_layers) - 8} more)"
        print(f"[kv] RESULT: {failures} (layer, rank) pair(s) differ in the packed int8 copy; "
              f"the FP copy differs *in value* on {len(value_layers)} layer(s): {shown or '-'} -> P2 violated "
              "(a token's KV content is not layout invariant)")
        return 1
    if byte_layers:
        print(f"[kv] RESULT: no value-level difference anywhere; the stored bytes differ on "
              f"{len(byte_layers)} layer(s) ({byte_layers[0]}..{byte_layers[-1]}) and only as flipped "
              "signs of quantized zeros -> the KV *content* agrees within one quantization step "
              "(sub-quantization divergence), i.e. no layout/content error, but the two layouts are "
              "not bit-identical")
        return 0
    print("[kv] RESULT: P2 holds on the dumped layers (per-token KV bit-identical across layouts)")
    return 0


def _act_rows(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    """``(positions, raw_bytes, float32, itemsize)`` of one activation dump.

    Two payload shapes are understood:

    * ``act`` -- a bf16/fp16/fp32 activation (the ``act``/``mlp`` trace);
    * ``q``   -- the **quantized** activation a GEMM consumes (fp8 e4m3 / int8),
      optionally with its ``s`` scale (e8m0).  Its bytes and the scale's bytes are
      concatenated row-wise into one comparison, so a change in either the codes
      or the scale counts as a difference; ``float32`` comes from the codes only
      and is used for the magnitude column.

    ``raw_bytes`` is the exact stored representation (viewed as bytes), so the
    comparison is bit-exact.
    """
    positions = payload.get("positions")
    if not isinstance(positions, torch.Tensor):
        return None
    act = payload.get("act")
    quant = payload.get("q")
    source = act if isinstance(act, torch.Tensor) else quant
    if not isinstance(source, torch.Tensor) or source.ndim != 2:
        return None
    rows = min(int(source.shape[0]), int(positions.numel()))
    if rows == 0:
        return None
    pos = positions[:rows].detach().to("cpu").to(torch.int64).numpy()
    kept = source.detach()[:rows].to("cpu").contiguous()
    raw = kept.view(torch.uint8).numpy()
    width = int(kept.shape[1])
    itemsize = max(1, raw.shape[1] // width) if width else 1
    if not isinstance(act, torch.Tensor):
        scale = payload.get("s")
        if (
            isinstance(scale, torch.Tensor)
            and scale.ndim == 2
            and int(scale.shape[0]) >= rows
        ):
            scale_bytes = (
                scale.detach()[:rows].to("cpu").contiguous().view(torch.uint8).numpy()
            )
            raw = np.concatenate([raw, scale_bytes], axis=1)
    return pos, raw, kept.float().numpy(), itemsize


def _act_table(paths: list[str]):
    """Concatenate one layout's ranks into a position-indexed table.

    Ranks hold disjoint token sets, so concatenating them rebuilds the sequence;
    padding rows (slot ``-1``) are dropped because their content is meaningless
    in both layouts.

    ``positions`` are **KV-cache slots**, not token indices: a request's block
    table need not start at block 0 (a 2048-token request whose blocks start at
    block 1 occupies slots 128..2175).  They are therefore returned raw and
    normalised by the caller, and must never be filtered against a *token* count
    such as ``num_actual_tokens`` -- doing that silently drops the tail of every
    request whose base is non-zero (observed: 1920 of 2048 tokens compared).
    """
    pos_parts: list[np.ndarray] = []
    raw_parts: list[np.ndarray] = []
    fp_parts: list[np.ndarray] = []
    itemsize = 1
    for path in paths:
        payload = _load(path)
        if payload is None:
            continue
        rows = _act_rows(payload)
        if rows is None:
            continue
        pos, raw, fp, itemsize = rows
        keep = pos >= 0
        if not keep.any():
            continue
        pos_parts.append(pos[keep])
        raw_parts.append(raw[keep])
        fp_parts.append(fp[keep])
    if not pos_parts:
        return None
    pos = np.concatenate(pos_parts)
    raw = np.concatenate(raw_parts, axis=0)
    fp = np.concatenate(fp_parts, axis=0)
    order = np.argsort(pos, kind="stable")
    return pos[order], raw[order], fp[order], itemsize


def _print_act_summary(
    rows: list[dict], ignored: int, block_size: int = 0
) -> tuple[int | None, str | None, dict | None]:
    """Compact per-(layer, op) table plus the first divergence, in layer order."""
    print()
    if ignored:
        print(f"[act] {ignored} older dump(s) ignored (newest per (layer, op, rank, cp_bal) wins)")
    print(f"[act] {'layer':>5}  {'op':>3}  {'compared':>8}  {'differ':>7}  {'first_pos':>9}  "
          f"{'max|d|':>10}  {'rel':>9}  {'block':>7}")
    first: tuple[int, str, dict] | None = None
    for stats in sorted(rows, key=lambda s: (s["layer"], _ACT_OP_ORDER.get(s["op"], 9))):
        span = "-" if stats["first_pos"] is None else str(stats["first_pos"])
        block = "-"
        if stats["first_pos"] is not None and block_size > 0:
            block = str(stats["first_pos"] // block_size)
        detail = (
            f"{stats['max_abs']:>10.3e}  {stats['max_rel']:>9.2e}"
            if stats["differ"]
            else f"{'-':>10}  {'-':>9}"
        )
        print(f"[act] {stats['layer']:>5}  {stats['op']:>3}  {stats['compared']:>8}  "
              f"{stats['differ']:>7}  {span:>9}  {detail}  {block:>7}")
        if stats["differ"] and first is None:
            first = (stats["layer"], stats["op"], stats)
    if first is None:
        print("[act] FIRST DIVERGENCE (act): none -- every compared token is bit-identical "
              "in both layouts on the traced layers")
        return None, None, None
    layer, op, stats = first
    print(
        f"[act] FIRST DIVERGENCE (act): layer {layer} op={op} at token {stats['first_pos']} "
        f"({stats['differ']}/{stats['compared']} tokens differ, max|d|={stats['max_abs']:.3e}, "
        f"rel={stats['max_rel']:.2e})"
    )
    return layer, op, stats


def check_act(args) -> int:
    """Compare the per-layer activation trace of B and C token by token.

    The dumps are keyed by **global token position** (never by row: under zigzag
    a rank holds ``[prev_block, next_block]``, under the continuous slice it
    holds ``[local_start, local_end)``).  ``in`` is the layer input (the previous
    layer's output), ``out`` is this layer's attention output, so one traced
    layer decides whether attention or the MoE/MLP in between introduced the
    difference:

    * ``in`` equal, ``out`` different -> attention (indexer / SFA / o_proj);
    * ``out`` of layer L equal, ``in`` of layer L+1 different -> MoE/MLP.

    Inside one MLP the ``qin`` rows (``gu_q``/``dn_q``) carry the **quantized**
    activation each GEMM consumes, so the verdict can separate "the activation
    quantization changed" from "the GEMM kernel changed" -- without them a
    ``gu_out``/``mlp_out`` difference has two indistinguishable explanations.
    """
    files = list(_iter_dumps(args.dir, "act"))
    if not files:
        print(f"[act] no dump found under {args.dir}", file=sys.stderr)
        return 2
    latest, counts = _latest_per_server(files, fields=("kind", "layer", "rank", "cpbal"))
    groups: dict[tuple[int, str], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    ignored = 0
    for key in sorted(latest, key=lambda k: (int(k[1]), k[0], int(k[2]), int(k[3]))):
        file_kind, layer, rank, cpbal = key
        op = _FILE_KIND_OP.get(file_kind, "in")
        if counts[key] > 1:
            ignored += counts[key] - 1
        groups[(layer, op)][cpbal].append(latest[key][0])
    if not groups:
        print(f"[act] no usable dump under {args.dir}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    incomplete: list[str] = []
    total = len(groups)
    for index, (layer, op) in enumerate(sorted(groups, key=lambda k: (k[0], _ACT_OP_ORDER.get(k[1], 9))), 1):
        variants = groups[(layer, op)]
        if len(variants) < 2:
            # Never drop a traced step silently: a one-sided op usually means the
            # other layout's dump was skipped (e.g. no token positions available),
            # and an unmentioned gap looks exactly like "this step is identical".
            note = (f"layer={layer} op={op}: only cp_balance={sorted(variants)} dumped, "
                    "need both 0 (B) and 1 (C) to compare")
            incomplete.append(note)
            if not args.summary_only:
                print(f"[act] {note}")
            continue
        left = _act_table(variants["0"])
        right = _act_table(variants["1"])
        if left is None or right is None:
            print(f"[act] layer={layer} op={op}: no usable rows on one side", file=sys.stderr)
            continue
        pos_l, raw_l, fp_l, itemsize = left
        pos_r, raw_r, fp_r, _ = right
        index_l = {int(p): i for i, p in enumerate(pos_l)}
        index_r = {int(p): i for i, p in enumerate(pos_r)}
        common = sorted(set(index_l) & set(index_r))
        # Positions are cache slots; both layouts share the request's block table,
        # so subtracting the common base turns them into natural-stream token
        # indices (equal to the prompt token index for a single request) and makes
        # the printed numbers comparable with the KV dump's row indices.
        base = min(common) if common else 0
        stats = {
            "layer": layer, "op": op, "compared": len(common), "differ": 0,
            "first_pos": None, "max_abs": 0.0, "max_rel": 0.0, "block": None,
            "cols": None, "ncols": None, "width": int(raw_l.shape[1]) // max(1, itemsize),
        }
        if common:
            rows_l = np.fromiter((index_l[p] for p in common), dtype=np.int64, count=len(common))
            rows_r = np.fromiter((index_r[p] for p in common), dtype=np.int64, count=len(common))
            mask = (raw_l[rows_l] != raw_r[rows_r]).any(axis=1)
            stats["differ"] = int(mask.sum())
            if stats["differ"]:
                where = np.nonzero(mask)[0]
                first_at = int(where[0])
                stats["first_pos"] = common[first_at] - base
                delta = np.abs(fp_l[rows_l[where]] - fp_r[rows_r[where]])
                stats["max_abs"] = float(delta.max())
                scale = float(np.abs(fp_l[rows_l[where]]).max()) or 1.0
                stats["max_rel"] = stats["max_abs"] / scale
                row_l = raw_l[rows_l[first_at]]
                row_r = raw_r[rows_r[first_at]]
                cols = np.nonzero(row_l != row_r)[0]
                stats["ncols"] = int(cols.size)
                stats["cols"] = [int(c) // max(1, itemsize) for c in cols[:8]]
        rows.append(stats)
        if not args.summary_only and stats["differ"]:
            first_token = common[int(np.nonzero(mask)[0][0])] - base
            print(f"\n[act] layer={layer} op={op}: first differing token={first_token} "
                  f"({stats['ncols']}/{stats['width']} elements differ on that row), "
                  f"differing tokens={stats['differ']}/{stats['compared']} "
                  f"(tokens {first_token}..{common[-1] - base}), "
                  f"max|d|={stats['max_abs']:.3e} rel={stats['max_rel']:.2e}, "
                  f"first columns={stats['cols']}")
        if index % 8 == 0 or index == total:
            print(f"[act] compared {index}/{total} (layer, op) groups", file=sys.stderr)

    layer, op, stats = _print_act_summary(rows, ignored, getattr(args, "block_size", 0))
    if incomplete:
        print()
        for note in incomplete:
            print(f"[act] INCOMPLETE {note}")
    print()
    if layer is None:
        print("[act] RESULT: traced activations are bit-identical across layouts")
        return 0
    # Sibling rows of the same layer: the verdict can then say whether the step
    # *before* the earliest divergence was already clean, which is exactly what
    # separates the two root causes inside one GEMM.  ``None`` means the op was
    # not traced at all -- that is not the same as "clean", and the verdict says
    # so instead of guessing.
    same_layer = {item["op"]: item for item in rows if item["layer"] == layer}

    def _clean(op_name: str) -> bool | None:
        item = same_layer.get(op_name)
        return None if item is None else not bool(item["differ"])

    quant_clean = _clean("gu_q")
    gu_out_clean = _clean("gu_out")
    dn_q_clean = _clean("dn_q")
    if op == "in":
        if layer == 0:
            print("[act] -> layer 0 的输入是 embedding（与排布无关）却在两种排布下不同："
                  "说明差异不是从层内来的，而是 token→rank 的映射/位置（positions）本身不一致")
        else:
            print(f"[act] -> layer {layer} 的 attention 输入（= layer {layer - 1} 的输出）先不等，"
                  f"而 attention 输出尚未确认；差异是在 layer {layer - 1} 的 MoE/MLP（或其间 norm/残差）里进入的。"
                  f"下一步：对比该 layer 的 attention 输出（op=out）以确认")
    elif op == "mlp_in":
        print(f"[act] -> layer {layer} 的 **MLP 输入**先不等，而同一层的 attention 输出相同"
              "（见同层 op=out 那一行）⇒ 分歧产生在「attention 输出 → MLP 输入」之间："
              "pre-MLP 的 norm / 残差 / 跨 rank 归约（不是 MLP 本身，也不是 attention）")
    elif op == "gu_q":
        print(f"[act] -> layer {layer} 的 **gate_up 量化输入**（喂给 npu_quant_matmul 的 fp8 + e8m0 scale）先不等，"
              f"而同层 MLP 的 bf16 输入（op=mlp_in）逐字节相同 ⇒ 根因在**激活量化这一步**："
              f"同一批 bf16 行在不同排布下量化出不同的 fp8/scale（MX 量化并非逐 token 独立，"
              f"或融合的 norm+quant 内核按 tile 共享状态）。"
              f"下一步：查该层量化/融合内核的入参形状与 tile，不必再往 GEMM 里找")
    elif op == "gu_out":
        if quant_clean is False:
            print(f"[act] -> layer {layer} 的 **gate_up_proj 输出**先不等，且同层 **gu_q 也不等** ⇒ "
                  f"根因在**量化**：同一个 bf16 输入量化出不同的 fp8/scale（见 op=gu_q 那一行）")
        elif quant_clean is True:
            print(f"[act] -> layer {layer} 的 **gate_up_proj 输出**先不等，而它的两个输入都逐字节相同"
                  f"（bf16 见 op=mlp_in，量化后的 fp8/scale 见 op=gu_q）⇒ 根因在 **npu_quant_matmul 这个 GEMM 内核本身**："
                  f"逐行数学与权重都相同、输入逐位相同却给出不同结果 ⇒ 查内核 tiling/workspace 与确定性"
                  f"（同一配置重跑一次即可判定），而不是 token 排布的数学")
        else:
            print(f"[act] -> layer {layer} 的 **gate_up_proj 输出**先不等，而 MLP 输入逐字节相同 ⇒ "
                  f"分歧产生在这一个 GEMM 内部：**激活量化（A-quant）** 或 GEMM 内核的 tiling/累加顺序"
                  f"（本层未打点 qin，无法区分；加 `qin:{layer}` 后一轮即可定死）")
    elif op == "dn_in":
        print(f"[act] -> layer {layer} 的 **down_proj 输入（bf16，silu 之后）**先不等，"
              f"而 gate_up_proj 输出相同 ⇒ 分歧产生在中间的**激活函数（silu）**这一步")
    elif op == "dn_q":
        if gu_out_clean is False:
            print(f"[act] -> layer {layer} 的 **down_proj 量化输入**先不等，但同层 gate_up 输出也不等 ⇒ "
                  f"先看更早的 op=gu_q / op=gu_out 两行，分歧不在这一步")
        else:
            print(f"[act] -> layer {layer} 的 **down_proj 量化输入**（silu 之后的 fp8 + scale）先不等，"
                  f"而 gate_up_proj 输出相同 ⇒ 根因在 **silu 之后的量化**这一步")
    elif op == "mlp_out":
        if dn_q_clean is False:
            print(f"[act] -> layer {layer} 的 **MLP/MoE 输出**先不等，但同层 down_proj 的量化输入也不等 ⇒ "
                  f"根因在 down_proj 之前（见 op=dn_q / op=dn_in 两行）")
        elif dn_q_clean is True:
            print(f"[act] -> layer {layer} 的 **MLP/MoE 输出**先不等，而它的输入与 down_proj 的量化输入都相同"
                  f"（见同层 op=mlp_in / op=dn_q）⇒ 根因在 **down_proj 这个 GEMM（含跨 rank 归约）**："
                  f"输入逐位相同却给出不同输出 ⇒ 查内核 tiling/workspace 与确定性")
        else:
            print(f"[act] -> layer {layer} 的 **MLP/MoE 输出**先不等，而它的输入相同（见同层 mlp_in 那一行）"
                  "⇒ 分歧产生在这一层的 MLP/MoE **内部**：激活量化 / GEMM 分组与内核 tiling / 专家计算"
                  f"（本层未打点 qin，无法区分；加 `qin:{layer}` 后一轮即可定死）")
    else:
        print(f"[act] -> layer {layer} 的 attention 输出先不等（输入见 op=in 那一行）："
              "差异在这一层的 attention 内部产生（indexer 选点顺序 / SFA 归约顺序 / o_proj），"
              "不是上一层传进来的")
    print("[act] RESULT: traced activations differ between B (cp_balance=0) and C (cp_balance=1)")
    return 1


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
    parser.add_argument("--kind", choices=("topk", "kv", "act", "both", "all"), default="both")
    parser.add_argument("--list", action="store_true", help="only list the dumps found")
    parser.add_argument(
        "--block-size",
        type=int,
        default=0,
        help="zigzag block size = seq_len / (2 * cp_size) (TP=8, 2048 tokens -> 128); "
        "only used to label which block the first divergence falls into (0 = don't label)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="print only the compact tables (kv: per-layer rows/ranks/first token/max|int8|; "
        "act: per-layer op/compared/differ/first token/magnitude; topk: one line per dump) "
        "plus the RESULT lines -- use this for a multi-layer sweep",
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
    if args.kind in ("topk", "both", "all"):
        rc |= check_topk(args)
    if args.kind in ("kv", "both", "all"):
        rc |= check_kv(args)
    if args.kind in ("act", "all"):
        rc |= check_act(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
