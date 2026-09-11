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
"""Standalone single-config launch + inference, for hand debugging.

它把 **Base 配置**（默认 ``B``：DSA-CP on + ``VLLM_ASCEND_CP_BALANCE=0``）的 server
单独拉起来，等 ``/health`` 就绪后用与 A/B driver **完全相同**的请求体（同一个
``completion_payload``）发一次推理，然后默认把 server 留着供手工调试。

一条命令，在 vllm-ascend 仓库根目录执行::

    python tools/cp_balance_compare/run_single.py

    python tools/cp_balance_compare/run_single.py --config C           # 换成 zigzag
    python tools/cp_balance_compare/run_single.py --prompt-lens 2048  # 只跑一个长度
    python tools/cp_balance_compare/run_single.py --no-keep           # 推理完就停 server
    python tools/cp_balance_compare/run_single.py --url http://127.0.0.1:8034   # 只发请求
    python tools/cp_balance_compare/run_single.py --env VLLM_ASCEND_CP_BALANCE_DUMP=topk:all,kv:all

期望结果（脚本开头也会打印一遍）:

* 模型拉起：屏幕出现 ``[<配置>] ...`` 实时日志，随后 ``[ready] <配置>: /health -> 200``；
* 配置门：``[check] <配置> fingerprint OK: {...}``（没有这行就说明 launcher 没应用 env 覆盖）；
* 推理：``[http] <- 200`` 且每个 case 打印一行
  ``[result] <case>.<req> n_prompt=... n_out=... tok_lp[1]=... top1[1]=...``；
* 失败：非 200 时打印 **HTTP 状态和响应正文**（真正的报错通常在那里），异常时打印完整
  traceback + server 日志尾部；请求体/响应体都落在 ``<out>/payload_*.json`` /
  ``<out>/response_*.txt``，可用 curl 原样复现；
* 默认保持 server 运行（``[keep] ...`` + curl 复现命令），Ctrl-C 停止并释放 NPU。

退出码：``0`` 推理成功，``1`` 拉起或推理失败，``2`` 参数/环境问题。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_LAUNCHER = "bash tools/cp_balance_compare/launcher_glm52_w4a4c8_mxfp4.sh {port}"
DEFAULT_OUT = "/dev/shm/cp_single"


def load_driver():
    """Import ``ab_cp_compare.py`` as a module (same trick as selfcheck.py)."""
    spec = importlib.util.spec_from_file_location("cp_balance_single_driver", HERE / "ab_cp_compare.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cp_balance_single_driver"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args(argv: list[str] | None = None, cp_default: int = 8) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        default="B",
        help="B = Base（DSA-CP on + CP_BALANCE=0，默认）；C = zigzag（CP_BALANCE=1）",
    )
    parser.add_argument("--port", type=int, default=8034)
    parser.add_argument("--launcher", default=os.environ.get("LAUNCHER", DEFAULT_LAUNCHER))
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--prompt-lens", default="2048,2049,4096", help="与 A/B 轮一致，可用 2048 缩小")
    parser.add_argument("--min-tokens", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument(
        "--cp-size",
        type=int,
        default=cp_default,
        help="zigzag 的 cp_size == launcher 的 TP_SIZE（默认取 $CP_SIZE/$TP_SIZE，否则 8）",
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--url", default="", help="只发请求，不拉起 server")
    parser.add_argument("--env", action="append", default=[], metavar="K=V", help="额外的 server env 覆盖")
    parser.add_argument("--startup-timeout", type=float, default=3600.0)
    parser.add_argument("--http-timeout", type=float, default=900.0)
    parser.add_argument(
        "--no-keep",
        dest="keep",
        action="store_false",
        default=True,
        help="推理结束后停掉 server（默认保留，方便手工继续调）",
    )
    return parser.parse_args(argv)


def driver_args(driver: Any, cfg: argparse.Namespace) -> argparse.Namespace:
    """The driver's own argument set: env injection can never drift from an A/B round."""
    argv = [
        "--launcher", cfg.launcher,
        "--configs", cfg.config,
        "--base-port", str(cfg.port),
        "--repo-root", cfg.repo_root,
        "--out", cfg.out,
        "--prompt-lens", cfg.prompt_lens,
        "--cp-balance-min-tokens", str(cfg.min_tokens),
        "--cp-size", str(cfg.cp_size),
        "--topk", str(cfg.topk),
        "--config-check", "warn",
        "--zigzag-check", "off",
        "--on-zigzag-miss", "continue",
        "--http-timeout", str(cfg.http_timeout),
        "--startup-timeout", str(cfg.startup_timeout),
        "--no-plot",
        "--no-csv",
    ]
    for item in cfg.env:
        argv += ["--env", item]
    return driver.parse_args(argv)


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def describe(case_id: str, req: int, entry: dict[str, Any]) -> None:
    """One line per request so a success is obvious at a glance."""
    tok_lp = entry.get("tok_lp") or []
    top1 = entry.get("top1") or []
    first_lp = next((value for value in tok_lp[1:] if value is not None), None)
    first_top1 = next((value for value in top1[1:] if value), None)
    print(
        f"[result] {case_id}.{req} n_prompt={entry.get('n_prompt')} n_out={entry.get('n_out')} "
        f"tok_lp[1]={_fmt(first_lp)} top1[1]={first_top1 or '-'}"
    )


def send_once(
    driver: Any,
    url: str,
    dargs: argparse.Namespace,
    prompts: list[Any],
    tag: str,
    out_dir: Path,
):
    """POST once with the driver's exact payload; save both bodies for replay.

    Returns the parsed entries, or ``None`` when the server answered non-200 --
    that response body *is* the error message, so it gets printed and kept.
    """
    payload, sent_ids = driver.completion_payload(dargs, prompts)
    payload_path = out_dir / f"payload_{tag}.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[http] POST {url}/v1/completions (body -> {payload_path})")
    resp = driver.HTTP_SESSION.post(url + "/v1/completions", json=payload, timeout=dargs.http_timeout)
    body_path = out_dir / f"response_{tag}.txt"
    body_path.write_text(resp.text, encoding="utf-8")
    print(f"[http] <- {resp.status_code} ({len(resp.text)} bytes -> {body_path})")
    if resp.status_code != 200:
        print("[http] 响应正文（报错信息通常在这里）:")
        print(resp.text[:4000])
        return None

    data = resp.json()
    choices = sorted(data["choices"], key=lambda item: item.get("index", 0))
    if len(choices) != len(prompts):
        raise RuntimeError(f"asked {len(prompts)} prompts, got {len(choices)} choices")
    entries = []
    for choice, prompt, hint in zip(choices, prompts, sent_ids):
        entry = driver.parse_choice(choice, data, dargs.topk)
        entry["prompt"] = prompt
        driver.verify_response(entry, hint)
        entries.append(entry)
    return entries


def _banner(driver: Any, name: str, cfg: argparse.Namespace, dargs: argparse.Namespace, out_dir: Path) -> None:
    env = driver.resolved_config(name, dargs)
    print("=" * 78)
    print(f"[single] config={name} port={cfg.port} out={out_dir}")
    print(
        f"[single] server env: CP_BALANCE={env['VLLM_ASCEND_CP_BALANCE']} "
        f"MIN_TOKENS={env['VLLM_ASCEND_CP_BALANCE_MIN_TOKENS']} "
        f"EMBED_LOCAL={env['VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL']} "
        f"SPEC={env['VLLM_ASCEND_SPEC_CONFIG']!r} KV={env['VLLM_ASCEND_KV_TRANSFER_CONFIG']!r}"
    )
    print(
        f"[single] 期望结果: [ready] {name}: /health -> 200；[check] {name} fingerprint OK；"
        "[http] <- 200；每个 case 一行 [result]"
    )
    print("=" * 78)


def _keep_alive(url: str, model: str, payload_path: Path) -> None:
    print(f"[keep] server 保持运行: {url} (model={model})")
    print(
        f"[keep] 手工复现: curl -s {url}/v1/completions "
        f"-H 'Content-Type: application/json' -d @{payload_path}"
    )
    print("[keep] Ctrl-C 停止 server 并释放 NPU")
    while True:
        time.sleep(5)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass
    driver = load_driver()
    cfg = parse_args(argv, driver.cp_size_default())
    name = cfg.config.strip().upper()
    if name not in driver.SUPPORTED_CONFIGS:
        print(f"[single] --config {name} 不支持；可选 {list(driver.SUPPORTED_CONFIGS)}（B = Base, C = zigzag）")
        return 2

    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    dargs = driver_args(driver, cfg)
    _banner(driver, name, cfg, dargs, out_dir)

    cases = driver.build_cases(dargs)
    if not cases:
        print("[single] --prompt-lens 没有产生任何 case")
        return 2
    log_path = log_dir / f"server_{name}.log"
    proc = log = reader = None
    try:
        if cfg.url:
            url = cfg.url.rstrip("/")
            print(f"[single] --url 模式：只发请求，不拉起 server（{url}）")
        else:
            proc, port, log, log_path, reader = driver.launch_server(dargs, name, log_dir)
            url = f"http://127.0.0.1:{port}"
            if not driver.wait_ready(
                url, dargs.startup_timeout, proc, label=name, log_every=dargs.wait_log_every
            ):
                state = "exited" if proc is not None and proc.poll() is not None else "not ready"
                print(f"[fail] server {state}；日志尾部:\n{driver.tail(log_path)}")
                return 1
            driver.verify_config(name, dargs, log_path)

        print(f"[single] cases: {[case[0] for case in cases]}")
        failed = ""
        for case_id, prompts in cases:
            try:
                entries = send_once(driver, url, dargs, prompts, case_id, out_dir)
            except Exception:  # noqa: BLE001 - a debug script must show everything
                print(f"[fail] {case_id}: 请求/解析异常:\n{traceback.format_exc()}")
                failed = case_id
                break
            if entries is None:
                failed = case_id
                break
            for req, entry in enumerate(entries):
                describe(case_id, req, entry)

        if failed:
            print(f"[fail] case {failed} 未通过；server 日志尾部:\n{driver.tail(log_path)}")
        else:
            print("[pass] 所有 case 推理成功（请求体与 A/B driver 完全一致）")
        if cfg.keep and proc is not None:
            _keep_alive(url, dargs.model, out_dir / f"payload_{cases[0][0]}.json")
        return 1 if failed else 0
    except KeyboardInterrupt:
        print("\n[single] 收到中断，停止 server")
        return 130
    finally:
        if proc is not None:
            print(f"[single] 停止 server（原始日志: {log_path}）")
        driver.stop_server(proc, log, reader, 0.0)


if __name__ == "__main__":
    sys.exit(main())
