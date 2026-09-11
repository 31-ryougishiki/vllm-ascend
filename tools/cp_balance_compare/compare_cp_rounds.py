#!/usr/bin/env python3
"""Compare several CP_BALANCE A/B rounds side by side (CPU only, no NPU).

读取每轮的 ``summary.json``，把关键指标并排打印，回答"改了这个变量之后，
``C−B`` 是否回到噪声级"。

用法::

    python tools/cp_balance_compare/compare_cp_rounds.py \
        before=/dev/shm/cp_ab/r_probe \
        after=/dev/shm/cp_ab_sweep/r_sweep

每轮目录由 ``run_cp_diag.sh`` 产出（含 ``summary.json``）。
判定口径与 driver 一致：``p99|d|C-B <= threshold`` 视为回到噪声级
（``threshold`` 由 driver 写成 ``max(--delta-threshold, 5×噪声 p99)``）。
第一轮以外的轮次额外打印相对第一轮的 p99 变化，便于一眼看出是否"回噪"。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


def _load_round(spec: str) -> tuple[str, dict[str, Any]]:
    label, _, path = spec.partition("=")
    if not path:
        raise SystemExit(f"expected LABEL=DIR, got {spec!r}")
    summary_path = path if path.endswith(".json") else os.path.join(path, "summary.json")
    if not os.path.isfile(summary_path):
        raise SystemExit(f"{label}: no summary.json at {summary_path}")
    with open(summary_path, encoding="utf-8") as handle:
        return label, json.load(handle)


def _metrics(summary: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for case in summary.get("cases", []):
        for metric in case.get("metrics", []):
            rows[(str(metric.get("case", case.get("case"))), int(metric.get("req", 0)))] = metric
    return rows


def _fmt(value: Any, spec: str = "") -> str:
    if value is None or value == "":
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return format(number, spec) if spec else f"{number:g}"


def _verdict(metric: dict[str, Any]) -> str:
    p99 = metric.get("p99|d|C-B")
    threshold = metric.get("threshold")
    if p99 is None:
        return "?"
    try:
        p99 = float(p99)
        threshold = float(threshold) if threshold is not None else 0.05
    except (TypeError, ValueError):
        return "?"
    return "DIFF" if p99 > threshold else "OK  "


def _runtime(summary: dict[str, Any], config: str) -> str:
    entry = (summary.get("runtime") or {}).get(config) or {}
    if not entry:
        return "n/a"
    return f"metadata={bool(entry.get('metadata'))} forward={bool(entry.get('forward'))}"


def _label(key: tuple[str, int], multi: set[str]) -> str:
    case, req = key
    return f"{case}#{req}" if case in multi else case


def main(argv: list[str] | None = None) -> int:
    try:
        # The Chinese verdict lines must not crash on a legacy console encoding.
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rounds", nargs="+", metavar="LABEL=DIR", help="one entry per round; the first is the reference")
    parser.add_argument("--base-config", default="B", help="reference config inside a round (default B)")
    parser.add_argument("--target-config", default="C", help="config under test inside a round (default C)")
    args = parser.parse_args(argv)

    if len(args.rounds) < 1:
        parser.error("need at least one LABEL=DIR")

    rounds = [_load_round(spec) for spec in args.rounds]
    tables = {label: _metrics(summary) for label, summary in rounds}
    keys = sorted({key for table in tables.values() for key in table})
    req_counts: dict[str, int] = {}
    for key in keys:
        req_counts[key[0]] = max(req_counts.get(key[0], 0), key[1])
    multi = {case for case, count in req_counts.items() if count > 0}

    for label, summary in rounds:
        skipped = summary.get("skipped") or []
        print(f"\n=== round {label} ===")
        print(f"    runtime: {args.base_config}{{{_runtime(summary, args.base_config)}}} "
              f"{args.target_config}{{{_runtime(summary, args.target_config)}}}")
        if skipped:
            print(f"    WARNING: skipped configs: {skipped} (results incomplete)")
        noise = summary.get("noise") or {}
        print(f"    noise floor (B2-B): p99={_fmt(noise.get('p99'), '.3g')} max={_fmt(noise.get('max'), '.3g')}")
        print(f"    {'case':<20} {'len':>6}  {'top1%':>7}  {'p99|d|':>9}  {'max|d|':>8}  "
              f"{'first_div':>9}  {'block@first_div':<24} {'verdict':<7}")
        for key in keys:
            metric = tables[label].get(key)
            name = _label(key, multi)
            if metric is None:
                print(f"    {name:<20} {'(missing)':>6}")
                continue
            print(f"    {name:<20} {_fmt(metric.get('length')):>6}  "
                  f"{_fmt(metric.get('top1%C~B'), '.2f'):>7}  "
                  f"{_fmt(metric.get('p99|d|C-B'), '.3g'):>9}  "
                  f"{_fmt(metric.get('max|d|C-B'), '.3g'):>8}  "
                  f"{_fmt(metric.get('first_div_C-B')):>9}  "
                  f"{str(metric.get('block@first_div_C-B') or '-'):<24} {_verdict(metric):<7}")

    if len(rounds) > 1:
        reference_label = rounds[0][0]
        print(f"\n=== p99|d| C-B per round (reference: {reference_label}) ===")
        header = f"    {'case':<20} " + "  ".join(f"{label:>12}" for label, _ in rounds) + "   change vs ref"
        print(header)
        for key in keys:
            cells: list[str] = []
            values: list[float | None] = []
            for label, _ in rounds:
                value = tables[label].get(key, {}).get("p99|d|C-B")
                try:
                    values.append(float(value) if value is not None else None)
                except (TypeError, ValueError):
                    values.append(None)
                cells.append(f"{_fmt(value, '.3g'):>12}")
            change = "-"
            if values[0] is not None and values[-1] is not None:
                if values[0] == 0:
                    change = "ref=0"
                else:
                    change = f"{(values[-1] - values[0]) / values[0] * 100:+.1f}%"
            print(f"    {_label(key, multi):<20} " + "  ".join(cells) + f"   {change}")

    # 结论提示：只看"最新一轮是否回到噪声级"
    last_label, _ = rounds[-1]
    last_table = tables[last_label]
    if not last_table:
        print(f"\n[verdict] round {last_label}: no comparable case")
        return 2
    shared = [key for key in keys if key in last_table]
    missing = [key for key in keys if key not in last_table]
    diffs = [key for key in shared if _verdict(last_table[key]).strip() == "DIFF"]
    print()
    if missing:
        print(f"[verdict] round {last_label}: {len(missing)} case(s) missing vs the reference round: "
              f"{', '.join(_label(key, multi) for key in missing)}")
    if diffs:
        print(f"[verdict] round {last_label}: {len(diffs)} case(s) still above threshold: "
              f"{', '.join(_label(key, multi) for key in diffs)}")
        print("[verdict] -> 该轮配置不足以解释异常，继续按决策树查 T2(P1)/T3(P2)")
        return 1
    print(f"[verdict] round {last_label}: C-B back to the noise floor for every comparable case")
    print("[verdict] -> 差异只由该轮改动的那个变量造成（T1 成立则根因 = 合并 2B 调用）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
