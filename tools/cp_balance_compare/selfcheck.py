#!/usr/bin/env python3
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
"""One-shot self check for a new site or a repaired environment (no NPU needed).

Typical use, from the vllm-ascend repository root::

    python tools/cp_balance_compare/selfcheck.py

The script runs two groups of checks and prints, for every one of them, the
result it expects, so a single log is enough to decide whether an expensive
A/B round can start:

1. **site**   -- repository / interpreter / dependencies / ``vllm_ascend``
                 import source / NPU / port / disk / leftovers;
2. **driver** -- ``selftest_mock.py`` against the local mock vLLM server
                 (no NPU, no model load).

Two things it deliberately does **not** duplicate, because the tools that own
them already run them on every round (and a copy here only rots):

* the configuration gate -- ``python tools/cp_balance_compare/ab_cp_compare.py --preflight``
  (``--config-check strict`` is also applied to every real round);
* the round command preview -- ``bash tools/cp_balance_compare/run_cp_diag.sh <mode> --dry-run``.

After the round, the same script collects the evidence to send back::

    python tools/cp_balance_compare/selfcheck.py --collect

Exit codes: ``0`` = ready to run a round, ``1`` = at least one FAIL,
``2`` = the script could not continue (bad repository layout).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_DUMP_DIR = "/dev/shm/cp_balance_dump"
DEFAULT_OUT_ROOT = "/dev/shm/cp_ab"

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"

# Files whose content is the version: the site copy is often synced by hand (no
# git on the box), so a content digest is the only reliable "which code is this".
KEY_FILES = (
    "tools/cp_balance_compare/ab_cp_compare.py",
    "tools/cp_balance_compare/check_zigzag_dumps.py",
    "tools/cp_balance_compare/run_cp_diag.sh",
    "tools/cp_balance_compare/selftest_mock.py",
    "tools/cp_balance_compare/repro_row_order.py",
    "tools/cp_balance_compare/launcher_glm52_w4a4c8_mxfp4.sh",
    # Site profiles: switching nodes is an env var, so the *file* has to be part
    # of the version too (a stale copy on the box silently runs the wrong node).
    "tools/cp_balance_compare/sites.sh",
    "vllm_ascend/attention/sfa_v1.py",
    "vllm_ascend/worker/model_runner_v1.py",
    "vllm_ascend/layers/cp_zigzag.py",
    "vllm_ascend/envs.py",
    "vllm_ascend/ascend_forward_context.py",
)
# Behaviour markers: a digest says "different", a marker says "which fix is here".
MARKERS = (
    ("dump kind qin", "vllm_ascend/attention/sfa_v1.py", '"qin"'),
    ("采样点位置键共用", "vllm_ascend/worker/model_runner_v1.py", "_cp_balance_dump_positions"),
    ("MLP dump kind 显式", "vllm_ascend/worker/model_runner_v1.py", '"mlpin"'),
    ("权重打点 w", "vllm_ascend/worker/model_runner_v1.py", "def _dump_weights"),
    ("旧名兼容 gugu_out", "tools/cp_balance_compare/check_zigzag_dumps.py", "gugu_out"),
    ("父目录/子目录提示", "tools/cp_balance_compare/check_zigzag_dumps.py", "_newer_subdir"),
    ("probe 每轮独立目录", "tools/cp_balance_compare/run_cp_diag.sh", "cp_probe/$(date"),
    ("probe2 模式", "tools/cp_balance_compare/run_cp_diag.sh", "probe2"),
    ("未定义名静态检查", "tools/cp_balance_compare/selftest_mock.py", "loads undefined name"),
    # The reproducer loads dumps on the CPU, so --run-op MUST place its tensors
    # explicitly; without this marker the site cannot tell "the CPU-backend fix is
    # here" from "the file is merely different" and re-runs a round that dies on
    # the same device error.
    ("repro 运行设备显式", "tools/cp_balance_compare/repro_row_order.py", "_resolve_device"),
    # The w dump used to crash the whole forward on an NZ-format weight
    # ("copy_ do not support internal format"); the fix is the ND fallback plus the
    # weight_format field the reproducer reads back.
    ("w dump NZ 兜底", "vllm_ascend/worker/model_runner_v1.py", "_weight_to_cpu"),
)

class Report:
    """Check results, mirrored to stdout and (optionally) to a log file."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._handle = path.open("w", encoding="utf-8") if path is not None else None
        self.results: list[tuple[str, str, str]] = []

    def say(self, text: str = "") -> None:
        print(text)
        if self._handle is not None:
            self._handle.write(text + "\n")

    def section(self, title: str) -> None:
        self.say("")
        self.say(f"===== {title} =====")

    def item(self, status: str, name: str, detail: str = "", expect: str = "") -> None:
        self.results.append((status, name, detail))
        head = f"[{status}] {name}"
        self.say(f"{head}: {detail}" if detail else head)
        if expect:
            self.say(f"         期望结果: {expect}")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def count(self, status: str) -> int:
        return sum(1 for item in self.results if item[0] == status)


def _run(cmd: list[str], cwd: Path | None = None, timeout: float = 120.0):
    """Run a command, returning None instead of raising on any failure."""
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _failure_excerpt(lines: list[str], span: int = 8) -> list[str]:
    """The assertion message of a failed test run, or the tail if there is none."""
    for index, line in enumerate(lines):
        if "AssertionError" in line or "Error:" in line:
            return [item for item in lines[index:index + span] if item.strip()]
    return lines[-span:]


# --------------------------------------------------------------------------- #
# 1. site checks
# --------------------------------------------------------------------------- #


def check_interpreter(report: Report) -> None:
    version = sys.version.split()[0]
    report.item(
        PASS if sys.version_info >= (3, 9) else FAIL,
        "Python 版本",
        f"{version} ({sys.executable})",
        ">= 3.9（driver 用了 3.9+ 语法）",
    )
    for module in ("numpy", "requests"):
        try:
            imported = importlib.import_module(module)
            report.item(PASS, f"依赖 {module}", getattr(imported, "__version__", "ok"))
        except ImportError as exc:
            report.item(FAIL, f"依赖 {module}", f"导入失败: {exc}", "driver 需要 numpy + requests")
    try:
        importlib.import_module("torch")
        report.item(PASS, "依赖 torch", "ok")
    except ImportError:
        report.item(
            WARN,
            "依赖 torch",
            "当前解释器里没有 torch",
            "只在 CPU 侧判读 dump 时需要（check_zigzag_dumps.py），driver 本身不需要",
        )


def fingerprint_lines(repo_root: Path) -> tuple[list[str], str]:
    """``(lines, digest)`` for a git-free version fingerprint.

    The offline site is synced by hand and often cannot run ``git``, so "which code
    is this box running" has to come from the files themselves: a sha256 per key
    file plus a few behaviour markers (a digest only says *different*, a marker says
    *which fix is present*).  The combined ``[fp]`` line is what gets pasted back.

    Digests are taken over the **LF-normalised** bytes: the dev checkout is on
    Windows (CRLF in the working copy) while the site is Linux, and a raw-byte hash
    made the two report different fingerprints for identical content.
    """
    import hashlib

    lines: list[str] = []
    combined = hashlib.sha256()
    for relative in KEY_FILES:
        path = repo_root / relative
        if not path.is_file():
            lines.append(f"[file] {relative:<52} MISSING")
            combined.update(f"{relative}:MISSING".encode())
            continue
        digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:12]
        combined.update(f"{relative}:{digest}".encode())
        lines.append(f"[file] {relative:<52} {digest}")
    for label, relative, needle in MARKERS:
        path = repo_root / relative
        present = path.is_file() and needle in path.read_text(encoding="utf-8", errors="replace")
        lines.append(f"[mark] {label:<28} {'OK' if present else 'MISSING'}")
        combined.update(f"{label}:{present}".encode())
    selftest = repo_root / "tools/cp_balance_compare/selftest_mock.py"
    count = 0
    if selftest.is_file():
        count = sum(
            1 for line in selftest.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.startswith("def test_")
        )
    lines.append(f"[mark] {'selftest 用例数':<28} {count}")
    combined.update(f"tests:{count}".encode())
    digest = combined.hexdigest()[:16]
    lines.append(f"[fp] {digest}    <- 远端把这行贴回来即可对齐版本（不需要 git）")
    return lines, digest


def check_repo_state(report: Report) -> None:
    if shutil.which("git") is None:
        report.item(WARN, "git 版本基线", "找不到 git", "能报出 HEAD 与工作区改动，便于 patch 对齐")
        return
    head = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    date = _run(["git", "-C", str(REPO_ROOT), "log", "-1", "--format=%cd"])
    dirty = _run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])
    if head is None or head.returncode != 0:
        report.item(WARN, "git 版本基线", "git 命令失败", "能报出 HEAD 与工作区改动")
        return
    changed = [line for line in (dirty.stdout if dirty else "").splitlines() if line.strip()]
    report.say(f"[{INFO}] git HEAD: {head.stdout.strip()}  ({_first_line(date.stdout if date else '')})")
    report.item(
        WARN if changed else PASS,
        "工作区改动",
        f"{len(changed)} 个文件被修改" if changed else "干净",
        "干净最好；若有本地调试改动请一并贴出来（我给的 patch 要对齐现场版本）",
    )
    for line in changed[:20]:
        report.say(f"         {line}")


def check_import_source(report: Report) -> None:
    """The server imports whatever PYTHONPATH points at -- it must be this tree."""
    found: dict[str, str] = {}
    for module in ("vllm", "vllm_ascend"):
        try:
            imported = importlib.import_module(module)
            found[module] = str(getattr(imported, "__file__", "?"))
        except Exception as exc:  # noqa: BLE001 - any import failure is informative here
            report.item(
                WARN,
                f"import {module}",
                f"失败: {type(exc).__name__}: {exc}",
                "driver 进程本身能 import；server 侧由 launcher 的 PYTHONPATH 决定",
            )
            continue
        report.item(PASS, f"import {module}", found[module])
    source = found.get("vllm_ascend")
    if source:
        under_repo = str(REPO_ROOT) in source
        report.item(
            PASS if under_repo else WARN,
            "vllm_ascend 来源",
            f"{source} {'属于' if under_repo else '不属于'} {REPO_ROOT}",
            f"应当是 {REPO_ROOT}/vllm_ascend/...（否则改的代码不会被加载）",
        )


def _scan_processes() -> list[str]:
    hits: list[str] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return hits
    for entry in proc.glob("[0-9]*"):
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        text = raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        if not text or "selfcheck" in text:
            continue
        if "vllm serve" in text or "EngineCore" in text or "vllm.entrypoints" in text:
            hits.append(f"{entry.name}: {text[:160]}")
    return hits


def check_host(report: Report, base_port: int, dump_dir: str, out_root: str) -> None:
    if shutil.which("npu-smi"):
        info = _run(["npu-smi", "info"], timeout=60.0)
        if info is not None and info.returncode == 0:
            lines = info.stdout.splitlines()
            # Each *physical* chip has exactly one "<id> Ascend910..." Name row.
            chips = sum(1 for line in lines if "Ascend910" in line)
            want = int(os.environ.get("TP_SIZE", "8"))
            report.say("[INFO] npu-smi info (前 16 行):")
            for line in lines[:16]:
                report.say(f"         {line}")
            report.item(
                PASS if chips >= want else WARN,
                "NPU 可见性",
                f"npu-smi 报告 {chips} 个 Ascend910（需要 >= TP_SIZE={want}）",
                f">= {want} 个 Ascend910 可见（launcher 默认 TP={want}；不够就减 TP_SIZE "
                "或调整 ASCEND_RT_VISIBLE_DEVICES）",
            )
        else:
            report.item(WARN, "NPU 可见性", "npu-smi info 执行失败", "8 张卡空闲")
    else:
        report.item(WARN, "NPU 可见性", "找不到 npu-smi", "在 NPU 机器上应能看到 8 张卡")

    processes = _scan_processes()
    report.item(
        WARN if processes else PASS,
        "残留 server 进程",
        f"{len(processes)} 个匹配 vllm/EngineCore 的进程" if processes else "没有",
        "没有残留；有的话先 kill 掉再跑（否则占卡/占端口）",
    )
    for line in processes[:10]:
        report.say(f"         {line}")

    sock = socket.socket()
    sock.settimeout(1.0)
    try:
        busy = sock.connect_ex(("127.0.0.1", base_port)) == 0
    finally:
        sock.close()
    report.item(
        WARN if busy else PASS,
        f"端口 {base_port}",
        "已被占用" if busy else "空闲",
        "driver 会在这个端口反复起停 server，必须空闲",
    )

    shm = Path("/dev/shm")
    if shm.is_dir():
        usage = shutil.disk_usage(str(shm))
        free_gib = usage.free / float(1 << 30)
        report.item(
            PASS if free_gib >= 5.0 else WARN,
            "/dev/shm 可用空间",
            f"{free_gib:.1f} GiB (总 {usage.total / float(1 << 30):.1f} GiB)",
            ">= 5 GiB（dump + 每轮 PNG/CSV 都写在这里）",
        )
    else:
        report.item(
            FAIL if sys.platform == "linux" else WARN,
            "/dev/shm",
            "目录不存在",
            "必须存在：dump 目录与默认输出目录都在 /dev/shm 下"
            + ("" if sys.platform == "linux" else "（当前主机不是 Linux，本地预览时忽略此项）"),
        )

    dumps = sorted(Path(dump_dir).glob("*.pt")) if Path(dump_dir).is_dir() else []
    report.item(
        INFO,
        "已有 dump",
        f"{dump_dir}: {len(dumps)} 个 .pt",
        "本次是首跑，应当是 0；非 0 说明目录里有旧数据，判读时只取最新一份",
    )
    rounds = sorted(Path(out_root).glob("*/summary.json")) if Path(out_root).is_dir() else []
    report.item(
        INFO,
        "已有轮次",
        f"{out_root}: {len(rounds)} 个 summary.json",
        "本次是首跑，应当是 0",
    )
    for path in rounds[:10]:
        report.say(f"         {path}")


# --------------------------------------------------------------------------- #
# 2. driver self test
# --------------------------------------------------------------------------- #


def run_selftest(report: Report, timeout: float = 900.0) -> None:
    script = HERE / "selftest_mock.py"
    base = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path(tempfile.gettempdir())
    scratch = base / f"cp_ab_selfcheck_{os.getpid()}"
    try:
        scratch.mkdir(parents=True, exist_ok=True)
    except OSError:
        scratch = Path(tempfile.mkdtemp(prefix="cp_ab_selfcheck_"))
    env = dict(os.environ)
    env["CP_AB_SELFTEST_DIR"] = str(scratch)
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(scratch),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        report.item(FAIL, "driver 自测", f"无法执行: {exc}", "末行打印 SELFTEST OK")
        return
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    output = (proc.stdout or "") + (proc.stderr or "")
    lines = output.splitlines()
    ok = proc.returncode == 0 and "SELFTEST OK" in output
    report.say("[INFO] selftest_mock.py 输出尾部:")
    for line in lines[-25:]:
        report.say(f"         {line}")
    skipped = [line for line in lines if "[skip]" in line]
    for line in skipped:
        report.say(f"         (跳过项) {line}")
    if not ok:
        report.say("[INFO] 失败断言:")
        for line in _failure_excerpt(lines):
            report.say(f"         {line}")
    note = f"rc={proc.returncode}, {sum(1 for l in lines if l.startswith('[ok]'))} 项通过"
    if skipped:
        note += f", {len(skipped)} 项跳过（非 Linux 上跳过依赖 bash 的 launcher 用例，属预期）"
    if not ok:
        failing = sorted(
            {line.strip().rsplit(" in ", 1)[1].split("(")[0] for line in lines if " in test_" in line}
        )
        if failing:
            note += f", 失败用例: {', '.join(failing)}"
    report.item(
        PASS if ok else FAIL,
        "driver 自测",
        note,
        "末行是 SELFTEST OK；有 [ok] 失败或 rc!=0 都要先修",
    )


# --------------------------------------------------------------------------- #
# evidence collection (after the round)
# --------------------------------------------------------------------------- #


def _round_metrics(summary: dict[str, Any], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in summary.get("cases", []):
        for metric in case.get("metrics", []):
            rows.append({key: metric.get(key) for key in keys})
    return rows


def collect(report: Report, dump_dir: str, out_root: str, base_port: int) -> int:
    report.section("证据收集")
    if shutil.which("git"):
        head = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
        dirty = _run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])
        report.say(f"=== HEAD ===\n{(head.stdout if head else '').strip()}")
        report.say(f"=== git status ===\n{(dirty.stdout if dirty else '').strip() or '(clean)'}")
    try:
        import vllm  # noqa: F401
        import vllm_ascend  # noqa: F401

        report.say(f"=== versions ===\nvllm_ascend: {vllm_ascend.__file__}")
    except Exception as exc:  # noqa: BLE001
        report.say(f"=== versions ===\n(import failed: {exc})")
    report.say(f"=== python ===\n{sys.version.splitlines()[0]} ({sys.executable})")

    dumps = sorted(Path(dump_dir).glob("*.pt")) if Path(dump_dir).is_dir() else []
    report.say(f"=== dumps ({dump_dir}, {len(dumps)}) ===")
    for path in dumps:
        report.say(f"  {path.name}  {path.stat().st_size} bytes")

    rounds = sorted(Path(out_root).glob("*/summary.json")) if Path(out_root).is_dir() else []
    keys = (
        "case", "req", "length", "threshold",
        "p99|d|C-B", "max|d|C-B", "first_div_C-B", "block@first_div_C-B",
        "top1%C~B", "top5_ovl%C~B", "gen_top1%C~B",
        "p99|d|B2-B", "first_div_B2-B",
    )
    report.say(f"=== rounds ({out_root}, {len(rounds)}) ===")
    for path in rounds:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            report.say(f"  {path}: unreadable ({exc})")
            continue
        report.say(f"  ## {path.parent.name}: noise={summary.get('noise')} "
                   f"runtime={summary.get('runtime')} skipped={summary.get('skipped')}")
        for row in _round_metrics(summary, keys):
            report.say(f"     {row}")

    report.say("=== logs ===")
    logs = sorted(Path(out_root).glob("*/logs/server_*.log")) if Path(out_root).is_dir() else []
    markers = (
        "[cp-ab]", "[cp-ab-cfg]", "[CP_BALANCE]", "Error", "Traceback",
        "[noise]", "[tokens]", "[runtime]", "[check]",
    )
    for path in logs:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        report.say(f"  ## {path}")
        for number, line in enumerate(text.splitlines(), 1):
            if any(marker in line for marker in markers):
                report.say(f"     {number}: {line[:300]}")

    report.say("=== next ===")
    report.say(f"  1) bash tools/cp_balance_compare/run_cp_diag.sh baseline   # base_port={base_port}")
    report.say("  2) bash tools/cp_balance_compare/run_cp_diag.sh check")
    report.say(
        "  3) python tools/cp_balance_compare/compare_cp_rounds.py "
        f"baseline={Path(out_root) / 'r1_baseline'}"
    )
    report.item(INFO, "证据已收集", f"{report.path}", "把整个日志文件发回来即可")
    return 0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="vllm-ascend repository root")
    parser.add_argument("--base-port", type=int, default=8034)
    parser.add_argument("--dump-dir", default=DEFAULT_DUMP_DIR)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report", default="", help="log file; default /dev/shm/cp_balance_selfcheck_<ts>.log")
    parser.add_argument("--skip-selftest", action="store_true", help="skip the mock-server self test")
    parser.add_argument("--collect", action="store_true", help="collect the evidence of finished rounds")
    parser.add_argument(
        "--fingerprint",
        action="store_true",
        help="print the git-free version fingerprint (file digests + fix markers) and exit",
    )
    return parser.parse_args(argv)


def _report_path(requested: str) -> Path | None:
    if requested:
        return Path(requested)
    base = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path(tempfile.gettempdir())
    if not os.access(str(base), os.W_OK):
        return None
    return base / f"cp_balance_selfcheck_{time.strftime('%Y%m%d_%H%M%S')}.log"


def main(argv: list[str] | None = None) -> int:
    try:
        # The marker labels are Chinese: a legacy console (Windows GBK) mangles
        # them or raises.  `--fingerprint` output gets pasted back verbatim, so it
        # has to be readable; UTF-8 streams are left untouched.
        for stream in (sys.stdout, sys.stderr):
            if "utf" in (getattr(stream, "encoding", "") or "").lower():
                stream.reconfigure(errors="replace")
            else:
                stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    if args.fingerprint:
        for line in fingerprint_lines(repo_root)[0]:
            print(line)
        return 0
    report = Report(_report_path(args.report))
    if report.path is not None:
        print(f"[selfcheck] 报告写入 {report.path}")

    report.say(f"cp_balance selfcheck @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.say(f"repo_root = {repo_root}")
    report.say(f"base_port = {args.base_port}  dump_dir = {args.dump_dir}  out_root = {args.out_root}")
    # Version, git-free: the offline box is synced by hand and may not have git.
    for line in fingerprint_lines(repo_root)[0]:
        report.say(line)

    if not (HERE / "ab_cp_compare.py").is_file():
        report.item(FAIL, "工具目录", f"{HERE} 里没有 ab_cp_compare.py", "从仓库根目录运行本脚本")
        report.close()
        return 2

    if args.collect:
        rc = collect(report, args.dump_dir, args.out_root, args.base_port)
        report.close()
        return rc

    report.section("1. 现场体检")
    check_interpreter(report)
    check_repo_state(report)
    check_import_source(report)
    check_host(report, args.base_port, args.dump_dir, args.out_root)

    if args.skip_selftest:
        report.item(WARN, "driver 自测", "--skip-selftest", "建议至少跑一次")
    else:
        report.section("2. driver 自测（mock server，无 NPU）")
        run_selftest(report)

    report.section("总结")
    report.say(f"PASS={report.count(PASS)}  WARN={report.count(WARN)}  FAIL={report.count(FAIL)}")
    for status, name, detail in report.results:
        if status in (FAIL, WARN):
            report.say(f"  [{status}] {name}: {detail}")
    if report.count(FAIL):
        report.say("")
        report.say("[verdict] 有 FAIL，先修完再跑轮次（跑一轮 = 2~3 次模型加载）。")
        rc = 1
    else:
        report.say("")
        report.say("[verdict] READY：可以跑轮次")
        # 配置门与命令预览不在本脚本里重复实现：驱动与 run_cp_diag 自己就带，
        # 而且每轮都会跑（自检里那份会随指纹/spec 变化而腐化）。
        report.say("  配置门（判据：末行 [preflight] all configs OK）:")
        report.say("    python tools/cp_balance_compare/ab_cp_compare.py --preflight")
        report.say("  命令预览（判据：打印 dir=/root/cp_probe 且 spec 含 qin:）:")
        report.say("    bash tools/cp_balance_compare/run_cp_diag.sh probe --dry-run")
        rc = 0
    if report.path is not None:
        report.say(f"[selfcheck] 请把 {report.path} 发回来")
    report.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
