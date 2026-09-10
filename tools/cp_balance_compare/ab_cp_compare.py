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
"""A/B/C prefill precision comparison for DSA-CP cp_balance.

Configurations (same weights, same TP size and the same additional_config
except ``enable_dsa_cp``):

* ``A``  : DSA-CP off                            -> golden non-CP anchor (optional)
* ``B``  : DSA-CP on  + ``VLLM_ASCEND_CP_BALANCE=0`` -> DSA-CP continuous baseline
* ``C``  : DSA-CP on  + ``VLLM_ASCEND_CP_BALANCE=1`` -> zigzag cp_balance
* ``<ref>2`` : optional repeat of the first config -> noise floor (``--repeat-a``)

The decisive comparison is ``C-B`` (same DSA-CP path, only the token layout
differs).  ``A`` only separates "DSA-CP's own error" (``B-A``) from cp_balance;
when B (``CP_BALANCE=0``) is already the trusted baseline, run
``--configs B,C --repeat-a`` and use ``B2-B`` as the noise floor.

For every prompt the driver calls ``/v1/completions`` with ``echo=true`` and
``logprobs=K`` and records, for every token position:

* ``token_logprobs[i]`` : logprob of the true prompt token at position ``i``.
  The target token is identical in every config, so the difference is a clean
  per-position precision signal.
* ``top_logprobs[i]``   : top-K distribution -> top-1 token id / top-5 set,
  used for the top-1 agreement rate and the top-5 overlap rate.

The token count is taken from the response (``prompt_token_ids`` or the
``token_logprobs`` length), so both token-id prompts and plain text prompts
work.  Servers are started sequentially because one TP=N server occupies the
whole node; with several nodes, start them yourself and use ``--urls``.

Examples
--------
Run everything on one node (uses ``launcher_template.sh`` by default)::

    MODEL_PATH=/path/to/weights python tools/cp_balance_compare/ab_cp_compare.py \
        --out /dev/shm/cp_ab --prompt-lens 2048,2049,4096 --repeat-a

Three servers on three nodes::

    python tools/cp_balance_compare/ab_cp_compare.py --out /dev/shm/cp_ab \
        --urls A=http://n1:12800,B=http://n2:12800,C=http://n3:12800

Real prompts from a JSONL file (see ``load_prompts_file``)::

    python tools/cp_balance_compare/ab_cp_compare.py --out /dev/shm/cp_ab \
        --prompts-file prompts.jsonl

Self test without an NPU::

    python tools/cp_balance_compare/selftest_mock.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("ab_cp_compare needs `requests`; run `pip install requests numpy`")

# Never route localhost traffic through the shell's HTTP proxy.  The launcher
# unsets proxies for the server, but the driver runs in the caller's shell where
# http_proxy/https_proxy may be set; requests would then send /health through a
# proxy and the driver would look "stuck" forever.
HTTP_SESSION = requests.Session()
HTTP_SESSION.trust_env = False


FINGERPRINT_PREFIX = "[cp-ab]"
CFG_JSON_PREFIX = "[cp-ab-cfg]"
ZIGZAG_MARKERS = {
    "metadata": "[CP_BALANCE] metadata zigzag=1",
    "forward": "[CP_BALANCE] forward zigzag_active=1",
}

# Minimal, self-consistent additional_config for a single PD-less prefill node.
# PD-only knobs must not be set here: `recompute_scheduler_enable=true` is
# rejected by vllm_ascend.platform when kv_role is not 'kv_consumer', which is
# exactly the case when the comparison runs with --no-kv-connector.
# Use --extra-additional-config to add or override keys per run.
BASE_ADDITIONAL_CONFIG: dict[str, Any] = {
    "enable_cpu_binding": "True",
    "multistream_overlap_shared_expert": "True",
    "enable_sparse_sfa_c8": True,
    "enable_sparse_li_c8": True,
    "enable_dsa_cp": True,
}

# name -> (enable_dsa_cp, VLLM_ASCEND_CP_BALANCE)
CONFIGS: dict[str, tuple[bool, str]] = {
    "A": (False, "0"),
    "B": (True, "0"),
    "C": (True, "1"),
}

Prompt = str | list[int]
PromptCase = tuple[str, list[Prompt]]


# --------------------------------------------------------------------------- #
# configuration / fingerprint
# --------------------------------------------------------------------------- #


def additional_config(enable_dsa_cp: bool, extra_json: str = "") -> str:
    cfg = dict(BASE_ADDITIONAL_CONFIG)
    cfg["enable_dsa_cp"] = enable_dsa_cp
    if extra_json:
        try:
            extra = json.loads(extra_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--extra-additional-config is not valid JSON: {exc}") from exc
        if not isinstance(extra, dict):
            raise SystemExit("--extra-additional-config must be a JSON object")
        cfg.update(extra)
    return json.dumps(cfg)


def config_spec(name: str) -> tuple[bool, str]:
    """(enable_dsa_cp, VLLM_ASCEND_CP_BALANCE) for a config, incl. ``A2``/``B2``."""
    base = name[:-1] if name.endswith("2") and name[:-1] in CONFIGS else name
    return CONFIGS[base]


def resolved_config(name: str, args: argparse.Namespace) -> dict[str, str]:
    """Effective env for one config, including ``--env`` overrides."""
    enable_dsa_cp, cp_balance = config_spec(name)
    resolved = {
        "VLLM_ASCEND_CP_BALANCE": cp_balance,
        "VLLM_ASCEND_ADDITIONAL_CONFIG": additional_config(enable_dsa_cp, args.extra_additional_config),
        "VLLM_ASCEND_SPEC_CONFIG": args.spec_config,
        "VLLM_ASCEND_KV_TRANSFER_CONFIG": "" if args.no_kv_connector else args.kv_transfer_config,
        "VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL": args.embed_local,
        # Lets the driver prove from the server log that C really entered the
        # zigzag path instead of silently falling back to continuous slices.
        "VLLM_ASCEND_CP_BALANCE_DEBUG_LOG": "1",
    }
    for key, value in args.env_override:
        resolved[key] = value
    return resolved


def expected_additional_config(name: str, args: argparse.Namespace) -> dict[str, Any]:
    resolved = resolved_config(name, args)
    try:
        return json.loads(resolved["VLLM_ASCEND_ADDITIONAL_CONFIG"])
    except json.JSONDecodeError:
        return {"__unparsable__": resolved["VLLM_ASCEND_ADDITIONAL_CONFIG"]}


def config_env(name: str, args: argparse.Namespace) -> dict[str, str]:
    return resolved_config(name, args)


def expected_fingerprint(name: str, args: argparse.Namespace) -> dict[str, int]:
    resolved = resolved_config(name, args)
    enable_dsa_cp = True
    try:
        enable_dsa_cp = bool(json.loads(resolved["VLLM_ASCEND_ADDITIONAL_CONFIG"]).get("enable_dsa_cp", True))
    except (TypeError, ValueError):
        enable_dsa_cp = '"enable_dsa_cp"' in resolved["VLLM_ASCEND_ADDITIONAL_CONFIG"]
    return {
        "CP_BALANCE": int(resolved["VLLM_ASCEND_CP_BALANCE"]),
        "DSA_CP": 1 if enable_dsa_cp else 0,
        "EMBED_LOCAL": int(resolved["VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL"]),
        "SPEC": 1 if resolved["VLLM_ASCEND_SPEC_CONFIG"] else 0,
        "KV": 1 if resolved["VLLM_ASCEND_KV_TRANSFER_CONFIG"] else 0,
    }


def parse_fingerprint(line: str) -> dict[str, int]:
    pos = line.find(FINGERPRINT_PREFIX)
    if pos < 0:
        return {}
    result: dict[str, int] = {}
    for token in line[pos + len(FINGERPRINT_PREFIX):].strip().split():
        key, sep, value = token.partition("=")
        if not sep:
            continue
        try:
            result[key.upper()] = int(value)
        except ValueError:
            continue
    return result


def last_fingerprint(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if FINGERPRINT_PREFIX in line:
            return line
    return None


def additional_config_from_log(log_path: Path) -> dict[str, Any] | None:
    """Parse the launcher's effective additional_config from ``[cp-ab-cfg]``."""
    if not log_path.exists():
        return None
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if CFG_JSON_PREFIX in line:
            payload = line.split(CFG_JSON_PREFIX, 1)[1].strip()
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, dict) else None
    return None


def verify_config(name: str, args: argparse.Namespace, log_path: Path) -> bool:
    """Make sure the launcher really applied this config's env overrides.

    A launcher that silently ignores ``VLLM_ASCEND_ADDITIONAL_CONFIG`` makes
    A/B/C all run the same configuration, which would look like a perfect
    precision match.  The launcher therefore prints a ``[cp-ab]`` fingerprint
    line and a ``[cp-ab-cfg]`` line with the effective additional_config; both
    are compared here.
    """
    if args.config_check == "off":
        return True
    expected = expected_fingerprint(name, args)
    line = None
    for _ in range(10):
        line = last_fingerprint(log_path)
        if line is not None:
            break
        time.sleep(0.5)
    if line is None:
        message = (
            f"server_{name}.log has no {FINGERPRINT_PREFIX} fingerprint; cannot "
            "verify that the launcher applied the env overrides"
        )
        if args.config_check == "strict":
            raise RuntimeError(message)
        print(f"[warn] {message}")
        return False
    actual = parse_fingerprint(line)
    diff = {key: (expected.get(key), actual.get(key)) for key in expected if expected.get(key) != actual.get(key)}
    if diff:
        message = f"{name} config mismatch: expected {expected}, launcher reported {actual}, diff {diff}"
        if args.config_check == "strict":
            raise RuntimeError(message)
        print(f"[warn] {message}")
        return False

    expected_cfg = expected_additional_config(name, args)
    actual_cfg = additional_config_from_log(log_path)
    if actual_cfg is None:
        print(
            f"[warn] {name}: launcher did not log {CFG_JSON_PREFIX}; "
            "additional_config content was not verified (only the 5 flags were)"
        )
    elif actual_cfg != expected_cfg:
        message = (
            f"{name} additional_config mismatch:\n"
            f"  expected: {json.dumps(expected_cfg, ensure_ascii=False)}\n"
            f"  launcher: {json.dumps(actual_cfg, ensure_ascii=False)}"
        )
        if args.config_check == "strict":
            raise RuntimeError(message)
        print(f"[warn] {message}")
        return False

    print(f"[check] {name} fingerprint OK: {actual}")
    return True


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #


def gen_ids(rng: np.random.Generator, length: int, vocab: int) -> list[int]:
    # Deterministic synthetic ids; token-id prompts keep the comparison
    # independent of tokenizer / chat-template details.
    return rng.integers(1, vocab, size=length, dtype=np.int64).tolist()


def parse_lens(spec: str) -> list[int]:
    return [int(x) for x in spec.replace(" ", "").split(",") if x]


def load_prompts_file(path: str) -> list[PromptCase]:
    """Load prompts from JSONL or a JSON array.

    Each record is ``{"case": "<optional>", "prompt": <prompt>}`` where
    ``<prompt>`` is a string, a list of token ids, or a list of such prompts
    (all entries of the same ``case`` are sent together in one request so they
    share one prefill batch).
    """
    text = Path(path).read_text(encoding="utf-8")
    records: list[dict[str, Any]]
    if text.lstrip().startswith("["):
        records = json.loads(text)
    else:
        records = []
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                records.append(json.loads(line))

    cases: dict[str, list[Prompt]] = {}
    for index, record in enumerate(records):
        if "prompt" not in record:
            raise ValueError(f"{path}: record {index + 1} has no 'prompt' field")
        case = record.get("case") or f"line{index + 1}"
        prompt = record["prompt"]
        entries = cases.setdefault(case, [])
        if isinstance(prompt, list) and prompt and all(isinstance(item, list) for item in prompt):
            entries.extend(prompt)
        else:
            entries.append(prompt)
    return list(cases.items())


def build_cases(args: argparse.Namespace) -> list[PromptCase]:
    if args.prompts_file:
        return load_prompts_file(args.prompts_file)
    rng = np.random.default_rng(args.seed)
    cases: list[PromptCase] = []
    for length in parse_lens(args.prompt_lens):
        cases.append((f"single_L{length}", [gen_ids(rng, length, args.vocab)]))
    if args.multi_lens:
        for group in args.multi_lens.split(";"):
            if not group.strip():
                continue
            lens = parse_lens(group)
            cases.append(("multi_" + "_".join(str(x) for x in lens), [gen_ids(rng, x, args.vocab) for x in lens]))
    return cases


# --------------------------------------------------------------------------- #
# server lifecycle
# --------------------------------------------------------------------------- #


def launch_server(args: argparse.Namespace, name: str, log_dir: Path):
    port = args.base_port
    env = os.environ.copy()
    env.update(config_env(name, args))
    cmd = args.launcher.format(port=port)
    log_path = log_dir / f"server_{name}.log"
    print(f"[launch] {name}: {cmd}")
    print(f"[launch] {name}: CP_BALANCE={env.get('VLLM_ASCEND_CP_BALANCE')}")
    if args.dry_run:
        return None, port, None, log_path
    log = open(log_path, "w")
    try:
        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=args.repo_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:
        log.close()
        raise
    return proc, port, log, log_path


def wait_ready(
    url: str,
    timeout: float,
    proc=None,
    label: str = "",
    log_every: float = 30.0,
) -> bool:
    """Ready == ``GET <url>/health`` returns 200.

    Polls every 3s until ``timeout``; if the launcher process exits early the
    wait stops immediately (bad config / HBM not released).  A heartbeat is
    printed every ``log_every`` seconds so a long model load does not look like
    a hang.
    """
    start = time.time()
    deadline = start + timeout
    last_log = start
    last_error = "connecting"
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            # The launcher died (bad config, HBM not released, ...): fail fast.
            return False
        try:
            if HTTP_SESSION.get(url + "/health", timeout=3).status_code == 200:
                print(f"[ready] {label}: /health -> 200 (elapsed={time.time() - start:.0f}s)")
                return True
        except requests.RequestException as exc:
            last_error = type(exc).__name__
        now = time.time()
        if now - last_log >= log_every:
            print(
                f"[wait] {label}: /health not ready yet "
                f"(elapsed={now - start:.0f}s/{timeout:.0f}s, last={last_error})"
            )
            last_log = now
        time.sleep(3)
    return False


def stop_server(proc, log, restart_wait: float = 30.0) -> None:
    try:
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=30)
    finally:
        if log is not None:
            log.close()
        if proc is not None:
            # Let HCCL / device memory settle before the next launch; 8-card
            # HCCL teardown can take longer than the default 30s on loaded nodes.
            time.sleep(restart_wait)


def tail(path: Path, lines: int = 40) -> str:
    try:
        content = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


def zigzag_state_from_log(log_path: Path) -> dict[str, bool]:
    """Whether the server log shows the zigzag metadata / forward marker."""
    try:
        text = log_path.read_text(errors="replace") if log_path.exists() else ""
    except OSError:
        text = ""
    return {key: marker in text for key, marker in ZIGZAG_MARKERS.items()}


def check_zigzag_activation(
    name: str, args: argparse.Namespace, state: dict[str, bool], log_path: Path
) -> str | None:
    """Return a failure message when the strict zigzag check fails."""
    print(f"[runtime] {name}: metadata_zigzag={state['metadata']} forward_zigzag={state['forward']}")
    if args.zigzag_check == "off":
        return None
    enable_dsa_cp, cp_balance = config_spec(name)
    expects_zigzag = enable_dsa_cp and cp_balance == "1"
    if expects_zigzag:
        if state["forward"]:
            return None
        message = (
            f"{name} did not log '{ZIGZAG_MARKERS['forward']}'; it may have fallen back to the "
            f"continuous-slice path (see {log_path})"
        )
        if args.zigzag_check == "strict":
            print(f"[error] {message}")
            return message
        print(f"[warn] {message}")
    elif state["forward"]:
        print(f"[warn] {name} unexpectedly logged zigzag_active=1")
    return None


# --------------------------------------------------------------------------- #
# query + parse
# --------------------------------------------------------------------------- #


def _lp_value(value: Any) -> float:
    if isinstance(value, dict):
        return float(value.get("logprob", math.nan))
    return float(value)


def _rank_topk(entry: dict[Any, Any], top_k: int) -> tuple[list[int], list[float]]:
    ranked = sorted(((int(key), _lp_value(value)) for key, value in entry.items()), key=lambda kv: kv[1], reverse=True)
    return [token for token, _ in ranked[:top_k]], [logprob for _, logprob in ranked[:top_k]]


def parse_choice(choice: dict[str, Any], data: dict[str, Any], top_k: int) -> dict[str, Any]:
    """Turn one completion choice into per-output-position arrays.

    ``tok_lp[i]`` is the logprob of the token at output position ``i`` and
    ``n_prompt`` is the number of prompt tokens; with ``echo`` the response
    carries ``n_prompt + 1`` output positions (the last one is the first
    generated token).  When ``logprobs`` is missing the vLLM
    ``prompt_logprobs`` extension is used as a fallback.
    """
    prompt_ids = choice.get("prompt_token_ids") or data.get("prompt_token_ids")
    prompt_ids = list(prompt_ids) if prompt_ids else None

    logprobs = choice.get("logprobs") or {}
    token_lps = logprobs.get("token_logprobs")
    top_lps = logprobs.get("top_logprobs")

    if top_lps:
        n_out = len(top_lps)
        n_prompt = len(prompt_ids) if prompt_ids is not None else max(0, n_out - 1)
        token_lps = list(token_lps) if token_lps is not None else [None] * n_out
    else:
        pl = choice.get("prompt_logprobs") or data.get("prompt_logprobs")
        if not pl:
            raise RuntimeError("server returned no logprobs: pass logprobs/echo or prompt_logprobs")
        n_prompt = len(prompt_ids) if prompt_ids is not None else len(pl)
        n_out = len(pl)
        token_lps = [None] * n_out
        top_lps = list(pl)
        for index, entry in enumerate(pl):
            if index == 0 or not entry or prompt_ids is None or index >= len(prompt_ids):
                continue
            value = entry.get(str(prompt_ids[index]))
            if value is not None:
                token_lps[index] = _lp_value(value)

    if n_out < n_prompt:
        raise RuntimeError(
            f"logprobs cover {n_out} positions but the prompt has {n_prompt} tokens; is echo enabled?"
        )

    top1: list[int | None] = []
    top1_lp: list[float | None] = []
    top5: list[list[int]] = []
    for entry in top_lps:
        if not entry:
            top1.append(None)
            top1_lp.append(None)
            top5.append([])
            continue
        tokens, values = _rank_topk(entry, 5)
        top1.append(tokens[0])
        top1_lp.append(values[0])
        top5.append(tokens)

    return {
        "prompt_token_ids": prompt_ids,
        "n_prompt": n_prompt,
        "n_out": n_out,
        "tok_lp": [None if value is None else float(value) for value in token_lps],
        "top1": top1,
        "top1_lp": top1_lp,
        "top5": top5,
    }


def query_batch(url: str, args: argparse.Namespace, prompts: list[Prompt]) -> list[dict[str, Any]]:
    prompt_field: Any = prompts[0] if len(prompts) == 1 else prompts
    payload = {
        "model": args.model,
        "prompt": prompt_field,
        "max_tokens": 1,
        "temperature": 0.0,
        "seed": 0,
        "echo": True,
        "logprobs": args.topk,
        "prompt_logprobs": args.topk,
        "add_special_tokens": False,
    }
    last_err: Exception | None = None
    for attempt in range(args.http_retries):
        try:
            resp = HTTP_SESSION.post(url + "/v1/completions", json=payload, timeout=args.http_timeout)
            resp.raise_for_status()
            data = resp.json()
            choices = sorted(data["choices"], key=lambda item: item.get("index", 0))
            if len(choices) != len(prompts):
                raise RuntimeError(f"asked {len(prompts)} prompts, got {len(choices)} choices")
            results = []
            for choice, prompt in zip(choices, prompts):
                entry = parse_choice(choice, data, args.topk)
                entry["prompt"] = prompt
                results.append(entry)
            return results
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"request failed after {args.http_retries} tries: {last_err}")


def query_case(url: str, args: argparse.Namespace, case: PromptCase) -> list[dict[str, Any]]:
    case_id, prompts = case
    if len(prompts) == 1:
        return query_batch(url, args, prompts)
    if args.multi_mode == "batch":
        # One EngineRequest with several prompts: they are scheduled in the
        # same prefill batch, which is what the multi-request zigzag path
        # needs.  (Concurrent requests only "most likely" land in one batch.)
        return query_batch(url, args, prompts)
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        return list(pool.map(lambda prompt: query_batch(url, args, [prompt])[0], prompts))


def query_all(url: str, args: argparse.Namespace, cases: list[PromptCase]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        case_id = case[0]
        try:
            out[case_id] = query_case(url, args, case)
            print(f"    {case_id}: {len(case[1])} prompt(s) OK")
        except Exception as exc:  # noqa: BLE001
            print(f"    {case_id}: FAILED ({exc})")
            out[case_id] = [{"error": str(exc)} for _ in case[1]]
    return out


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #


def build_pairs(names: list[str]) -> list[tuple[str, str]]:
    """Ordered ``(newer, older)`` pairs so deltas read as e.g. ``C-B``.

    For ``[A, B, C]`` this is exactly ``(B,A), (C,B), (C,A)``; for ``[B, C]``
    it is just ``(C,B)``; a repeated config ``B2`` adds ``(B2,B)``.
    """
    base = [name for name in names if not (name.endswith("2") and name[:-1] in names)]
    pairs: list[tuple[str, str]] = []
    for index in range(1, len(base)):
        pairs.append((base[index], base[index - 1]))
    for index in range(2, len(base)):
        pairs.append((base[index], base[0]))
    for name in names:
        if name.endswith("2") and name[:-1] in base:
            pairs.append((name, name[:-1]))
    return pairs


def as_float(values) -> np.ndarray:
    return np.array([math.nan if v is None else float(v) for v in values], dtype=np.float64)


def as_int(values) -> np.ndarray:
    return np.array([-1 if v is None else int(v) for v in values], dtype=np.int64)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = values.astype(np.float64)
    if window <= 1 or values.size == 0:
        return values
    window = min(window, values.size)
    kernel = np.ones(window, dtype=np.float64) / window
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def first_sustained(delta: np.ndarray, threshold: float, run: int) -> int | None:
    """First index where ``|delta|`` stays above ``threshold`` for ``run`` steps."""
    if delta.size < run:
        return None
    streak = 0
    for index, flag in enumerate(np.abs(delta) > threshold):
        streak = streak + 1 if flag else 0
        if streak >= run:
            return index - run + 1
    return None


def zigzag_segments(length: int, cp_size: int) -> list[tuple[int, int, int, str]]:
    """``[(start, end, rank, 'prev'|'next')]`` for a single-request zigzag plan."""
    try:
        from vllm_ascend.layers.cp_zigzag import build_zigzag_plan
    except Exception:  # pragma: no cover - only importable on the dev node
        return []
    segment_num = 2 * cp_size
    num_tokens_pad = ((length + segment_num - 1) // segment_num) * segment_num
    try:
        plan = build_zigzag_plan([length], [0], cp_size, 0, num_tokens_pad, length)
    except Exception:  # pragma: no cover
        return []
    blocks = plan.block_sizes[0]
    out: list[tuple[int, int, int, str]] = []
    start = 0
    for block, size in enumerate(blocks):
        end = start + size
        if block < cp_size:
            rank, kind = block, "prev"
        else:
            rank, kind = segment_num - 1 - block, "next"
        out.append((start, end, rank, kind))
        start = end
    return out


def segment_of(segments: list[tuple[int, int, int, str]], pos: int) -> str:
    for start, end, rank, kind in segments:
        if start <= pos < end:
            return f"rank{rank}/{kind}(b{start}-{end})"
    return "-"


def _case_arrays(results: dict, case_id: str, names: list[str], req: int):
    """Return per-config arrays clamped to a common prompt length."""
    entries = {name: results[name][case_id][req] for name in names}
    lengths = [int(entry["n_prompt"]) for entry in entries.values()]
    id_sets = [tuple(entry.get("prompt_token_ids") or ()) for entry in entries.values()]
    if len(set(id_sets)) > 1 and any(id_sets):
        print(f"    [warn] {case_id}.{req}: prompt_token_ids differ between configs")
    length = min(lengths)
    if length < 2:
        return None
    tok_lp = {name: as_float(entry["tok_lp"])[1:length] for name, entry in entries.items()}
    top1 = {name: as_int(entry["top1"])[1:length] for name, entry in entries.items()}
    top5 = {name: entry["top5"][1:length] for name, entry in entries.items()}
    gen_pos = {
        name: (int(entry["n_prompt"]) if int(entry["n_out"]) > int(entry["n_prompt"]) else None)
        for name, entry in entries.items()
    }
    return length, tok_lp, top1, top5, gen_pos


def _pair_metrics(pairs, tok_lp, top1, top5, threshold: float, run: int, length: int, segments):
    metrics: dict[str, Any] = {}
    for x, y in pairs:
        key = f"{x}-{y}"
        delta = tok_lp[x] - tok_lp[y]
        metrics[f"p99|d|{key}"] = float(np.nanpercentile(np.abs(delta), 99))
        metrics[f"max|d|{key}"] = float(np.nanmax(np.abs(delta)))
        first = first_sustained(delta, threshold, run)
        metrics[f"first_div_{key}"] = "" if first is None else int(first) + 1
        if first is not None and segments:
            metrics[f"block@first_div_{key}"] = segment_of(segments, int(first) + 1)
        else:
            metrics[f"block@first_div_{key}"] = "-"
        agree_key = f"{x}~{y}"
        valid = (top1[x] >= 0) & (top1[y] >= 0)
        if valid.any():
            metrics[f"top1%{agree_key}"] = 100.0 * float(np.mean((top1[x] == top1[y])[valid]))
        else:
            metrics[f"top1%{agree_key}"] = float("nan")
        overlap = np.array(
            [
                top1[x][k] in top5[y][k] and top1[y][k] in top5[x][k]
                for k in range(len(tok_lp[x]))
            ],
            dtype=np.float64,
        )
        if valid.any():
            metrics[f"top5_ovl%{agree_key}"] = 100.0 * float(np.mean(overlap[valid]))
        else:
            metrics[f"top5_ovl%{agree_key}"] = float("nan")
    return metrics


def _generated_token_metrics(results, case_id, req, pairs, gen_pos):
    metrics: dict[str, Any] = {}
    for x, y in pairs:
        key = f"{x}~{y}"
        pos_x, pos_y = gen_pos[x], gen_pos[y]
        if pos_x is None or pos_y is None:
            metrics[f"gen_top1%{key}"] = None
            metrics[f"gen_top5_ovl%{key}"] = None
            continue
        top1_x = as_int(results[x][case_id][req]["top1"])
        top1_y = as_int(results[y][case_id][req]["top1"])
        if top1_x.size <= pos_x or top1_y.size <= pos_y:
            metrics[f"gen_top1%{key}"] = None
            metrics[f"gen_top5_ovl%{key}"] = None
            continue
        token_x, token_y = int(top1_x[pos_x]), int(top1_y[pos_y])
        if token_x < 0 or token_y < 0:
            metrics[f"gen_top1%{key}"] = None
            metrics[f"gen_top5_ovl%{key}"] = None
            continue
        metrics[f"gen_top1%{key}"] = int(token_x == token_y)
        top5_x = results[x][case_id][req]["top5"][pos_x]
        top5_y = results[y][case_id][req]["top5"][pos_y]
        metrics[f"gen_top5_ovl%{key}"] = int(token_x in top5_y and token_y in top5_x)
    return metrics


def compare_case(results, case_id, names, args, noise, out_dir) -> dict[str, Any]:
    ref_prompts = results[names[0]].get(case_id) or []
    if not ref_prompts or any("error" in entry for entry in ref_prompts):
        return {"case": case_id, "status": "missing results"}
    pairs = build_pairs(names)
    case_metrics = []
    for req in range(len(ref_prompts)):
        if any(req >= len(results[name].get(case_id, [])) or "error" in results[name][case_id][req] for name in names):
            print(f"    {case_id}.{req}: skipped (missing/failed config)")
            continue
        extracted = _case_arrays(results, case_id, names, req)
        if extracted is None:
            continue
        length, tok_lp, top1, top5, gen_pos = extracted
        threshold = max(args.delta_threshold, 5.0 * float((noise or {}).get("p99", 0.0)))
        segments = (
            zigzag_segments(length, args.cp_size) if args.annotate_plan and case_id.startswith("single_") else []
        )
        metrics = {"case": case_id, "req": req, "length": length, "threshold": threshold}
        metrics.update(_pair_metrics(pairs, tok_lp, top1, top5, threshold, args.run_len, length, segments))
        metrics.update(_generated_token_metrics(results, case_id, req, pairs, gen_pos))
        case_metrics.append(metrics)

        if args.plot:
            plot_case(out_dir, case_id, req, tok_lp, top1, metrics, segments, args)
        if args.csv:
            write_csv(out_dir, case_id, req, tok_lp, top1, pairs, metrics)
    return {"case": case_id, "metrics": case_metrics}


def plot_case(out_dir, case_id, req, tok_lp, top1, metrics, segments, args) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        print("    [warn] matplotlib not installed; skipping plot")
        return

    positions = np.arange(1, metrics["length"])
    stride = max(1, positions.size // args.plot_points)
    xs = positions[::stride]
    names = list(tok_lp)
    pairs = build_pairs(names)

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for name, values in tok_lp.items():
        axes[0].plot(xs, values[::stride], label=name, lw=1.0)
    axes[0].set_ylabel("logprob of true token")
    axes[0].legend(loc="lower right")
    axes[0].set_title(f"{case_id} req{req} len={metrics['length']}")

    for x, y in pairs:
        delta = tok_lp[x] - tok_lp[y]
        axes[1].plot(xs, np.abs(delta[::stride]), label=f"{x}-{y}", lw=0.9)
    axes[1].axhline(metrics["threshold"], color="k", ls="--", lw=0.8, label=f"thr={metrics['threshold']:.4f}")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("|logprob delta|")
    axes[1].legend(loc="upper right")

    for x, y in pairs:
        agree = (top1[x] == top1[y]).astype(np.float64)
        axes[2].plot(positions, rolling_mean(agree, args.agree_window), label=f"{x}~{y}", lw=0.9)
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_ylabel(f"top-1 agreement (win={args.agree_window})")
    axes[2].set_xlabel("prompt token position")
    axes[2].legend(loc="lower right")

    if segments:
        for start, _end, _rank, _kind in segments:
            for ax in axes:
                ax.axvline(start, color="grey", ls=":", lw=0.4, alpha=0.6)

    fig.tight_layout()
    path = out_dir / f"{case_id}_req{req}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"    plot -> {path}")


def write_csv(out_dir, case_id, req, tok_lp, top1, pairs, metrics) -> None:
    names = list(tok_lp)
    path = out_dir / f"{case_id}_req{req}.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = (
            ["pos"]
            + [f"logprob_{n}" for n in names]
            + [f"top1_{n}" for n in names]
            + [f"d_{x}-{y}" for x, y in pairs]
            + [f"agree_{x}~{y}" for x, y in pairs]
        )
        writer.writerow(header)
        positions = np.arange(1, metrics["length"])
        for k, pos in enumerate(positions):
            writer.writerow(
                [int(pos)]
                + [float(tok_lp[n][k]) for n in names]
                + [int(top1[n][k]) for n in names]
                + [float(tok_lp[x][k] - tok_lp[y][k]) for x, y in pairs]
                + [int(top1[x][k] == top1[y][k]) for x, y in pairs]
            )


# --------------------------------------------------------------------------- #
# query orchestration / main
# --------------------------------------------------------------------------- #


def parse_urls(spec: str) -> dict[str, str]:
    urls = {}
    for item in spec.split(","):
        name, _, url = item.partition("=")
        urls[name.strip()] = url.strip().rstrip("/")
    return urls


def _launcher_dry_run(args: argparse.Namespace, name: str, timeout: float = 60.0):
    """Run the launcher with DRY_RUN=1 and return (rc, stdout, stderr)."""
    env = os.environ.copy()
    env.update(config_env(name, args))
    env["DRY_RUN"] = "1"
    cmd = args.launcher.format(port=args.base_port)
    proc = subprocess.Popen(
        ["bash", "-lc", cmd],
        cwd=args.repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        # The launcher ignores DRY_RUN and started a real server: kill the group.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            out, err = proc.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, err = proc.communicate(timeout=20)
        return None, out, err


def preflight(args: argparse.Namespace, names: list[str]) -> int:
    """Validate launcher / env / fingerprint for every config without a model load."""
    failed = []
    for name in names:
        rc, out, err = _launcher_dry_run(args, name)
        line = next((item for item in out.splitlines() if FINGERPRINT_PREFIX in item), None)
        actual = parse_fingerprint(line) if line else {}
        expected = expected_fingerprint(name, args)
        cfg_line = next((item for item in out.splitlines() if CFG_JSON_PREFIX in item), None)
        cfg_note = ""
        cfg_failed = False
        if cfg_line:
            try:
                actual_cfg = json.loads(cfg_line.split(CFG_JSON_PREFIX, 1)[1].strip())
            except json.JSONDecodeError:
                actual_cfg = None
            expected_cfg = expected_additional_config(name, args)
            if actual_cfg != expected_cfg:
                cfg_failed = True
                cfg_note = (
                    f"additional_config mismatch: expected={json.dumps(expected_cfg, ensure_ascii=False)} "
                    f"launcher={json.dumps(actual_cfg, ensure_ascii=False)}"
                )
        else:
            cfg_note = f"(no {CFG_JSON_PREFIX} line: additional_config not verified)"
        if rc is None:
            status = "FAILED (launcher ignored DRY_RUN; add DRY_RUN support)"
        elif rc != 0:
            last_err = err.strip().splitlines()[-1] if err.strip() else ""
            status = f"FAILED (rc={rc}) {last_err}"
        elif actual != expected:
            status = f"FAILED fingerprint expected={expected} actual={actual}"
        elif cfg_failed:
            status = f"FAILED {cfg_note}"
        else:
            status = "OK"
        if status != "OK":
            failed.append(name)
        dry_line = next((item for item in out.splitlines() if "dry-run" in item), "")
        suffix = f"{dry_line} {cfg_note}".strip()
        print(f"[preflight] {name}: {status} {suffix}".rstrip())
    if failed:
        print(f"[preflight] FAILED configs: {failed}")
        return 1
    print("[preflight] all configs OK (env overrides + fingerprint + vllm/model path reachable)")
    return 0


def print_metrics(metrics: dict[str, Any]) -> None:
    top1_pairs = sorted(key[len("top1%"):] for key in metrics if key.startswith("top1%"))
    delta_pairs = sorted(key[len("p99|d|"):] for key in metrics if key.startswith("p99|d|"))
    top1 = " ".join(f"{pair}={metrics['top1%' + pair]:.2f}" for pair in top1_pairs)
    deltas = " ".join(f"{pair}={metrics['p99|d|' + pair]:.2e}" for pair in delta_pairs)
    focus = "C-B" if "C-B" in delta_pairs else (delta_pairs[-1] if delta_pairs else "")
    first = (metrics.get(f"first_div_{focus}") or "-") if focus else "-"
    block = metrics.get(f"block@first_div_{focus}", "-") if focus else "-"
    gen_pairs = sorted(key[len("gen_top1%"):] for key in metrics if key.startswith("gen_top1%"))
    gen = " ".join(f"{pair}={metrics['gen_top1%' + pair]}" for pair in gen_pairs)
    print(
        f"[{metrics['case']}.{metrics['req']}] len={metrics['length']} "
        f"top1% {top1} | p99|d| {deltas} | "
        f"first_div {focus}={first} @ {block} "
        f"gen_top1% {gen}"
    )


def compute_noise(results: dict, names: list[str]) -> dict[str, float]:
    """Noise floor from any repeated config (``X2`` vs ``X``)."""
    repeats = [(name, name[:-1]) for name in names if name.endswith("2") and name[:-1] in names]
    if not repeats:
        return {}
    repeat_name, base_name = repeats[0]
    diffs = []
    for case_id, prompts in results[base_name].items():
        for req, entry in enumerate(prompts):
            if "error" in entry or req >= len(results[repeat_name].get(case_id, [])):
                continue
            x = as_float(entry["tok_lp"])
            y = as_float(results[repeat_name][case_id][req]["tok_lp"])
            size = min(x.size, y.size)
            diffs.append(np.abs(x[1:size] - y[1:size]))
    if not diffs:
        return {}
    delta = np.concatenate(diffs)
    noise = {"p99": float(np.nanpercentile(delta, 99)), "max": float(np.nanmax(delta))}
    print(f"[noise] {repeat_name} vs {base_name}: p99={noise['p99']:.3e} max={noise['max']:.3e}")
    return noise


def collect_sequential(args, names, cases, out_dir, log_dir):
    results = {}
    runtimes: dict[str, dict[str, bool]] = {}
    failures: list[str] = []
    skipped: list[str] = []
    for index, name in enumerate(names):
        proc, port, log, log_path = launch_server(args, name, log_dir)
        if args.dry_run:
            if log is not None:
                log.close()
            continue
        url = f"http://127.0.0.1:{port}"
        try:
            print(
                f"[wait] {name}: polling {url}/health "
                f"(startup timeout={args.startup_timeout:.0f}s); the model load can take minutes"
            )
            if not wait_ready(
                url,
                args.startup_timeout,
                proc,
                label=name,
                log_every=args.wait_log_every,
            ):
                state = "exited" if proc is not None and proc.poll() is not None else "not ready"
                print(f"[error] {name} server {state}; log tail:\n{tail(log_path)}")
                raise SystemExit(1)
            try:
                verify_config(name, args, log_path)
            except RuntimeError as exc:
                print(f"[error] {exc}")
                raise SystemExit(1)
            print(f"[query] {name} -> {url}")
            results[name] = query_all(url, args, cases)
            (out_dir / f"results_{name}.json").write_text(json.dumps(results[name], ensure_ascii=False))
            runtimes[name] = zigzag_state_from_log(log_path)
            failure = check_zigzag_activation(name, args, runtimes[name], log_path)
            if failure is not None:
                failures.append(failure)
                if args.on_zigzag_miss == "skip" and index + 1 < len(names):
                    skipped = names[index + 1:]
                    print(
                        f"[warn] skipping remaining configs {skipped}: {name} did not enter zigzag, "
                        "so the rest cannot answer the precision question "
                        "(pass --on-zigzag-miss continue to force the full run)"
                    )
                    break
        finally:
            stop_server(proc, log, args.restart_wait)
    return results, runtimes, failures, skipped


def collect_from_urls(args, names, cases):
    urls = parse_urls(args.urls)
    missing = [name for name in names if name not in urls]
    if missing:
        raise SystemExit(f"--urls missing configs: {missing}")
    results = {}
    for name in names:
        print(f"[query] {name} -> {urls[name]}")
        results[name] = query_all(urls[name], args, cases)
    return results


def run(args: argparse.Namespace) -> int:
    requested = [item.strip().upper() for item in args.configs.split(",") if item.strip()]
    unknown = [item for item in requested if item not in CONFIGS]
    if unknown:
        raise SystemExit(f"--configs has unknown entries {unknown}; choose from {sorted(CONFIGS)}")
    if len(requested) < 2:
        raise SystemExit("--configs needs at least two configs, e.g. 'B,C' or 'A,B,C'")
    names = list(requested)
    if args.repeat_a:
        # Repeat the reference (first) config as <ref>2 to measure the noise floor.
        names.append(f"{requested[0]}2")
    if args.preflight:
        return preflight(args, names)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    cases = build_cases(args)
    print(f"[cases] {[case[0] for case in cases]}")

    if args.urls:
        if args.config_check != "off":
            print("[warn] --urls mode cannot read server logs; skipping the config fingerprint check")
        if args.zigzag_check != "off":
            print("[warn] --urls mode cannot read server logs; skipping the zigzag activation check")
        results = collect_from_urls(args, names, cases)
        runtimes: dict[str, dict[str, bool]] = {}
        failures: list[str] = []
        skipped: list[str] = []
    else:
        results, runtimes, failures, skipped = collect_sequential(args, names, cases, out_dir, log_dir)
    if args.dry_run:
        return 0

    # Compare only the configs that were actually collected, so an early stop
    # (zigzag miss) still produces the A/B/C deltas instead of an empty summary.
    collected = [name for name in names if name in results]
    print(f"[compare] configs: {collected}" + (f" (skipped {skipped})" if skipped else ""))
    (out_dir / "results_all.json").write_text(json.dumps(results, ensure_ascii=False))
    noise = compute_noise(results, collected)
    summaries = []
    for case in cases:
        summary = compare_case(results, case[0], collected, args, noise, out_dir)
        summaries.append(summary)
        for metrics in summary.get("metrics", []):
            print_metrics(metrics)
    summary_payload = {"noise": noise, "runtime": runtimes, "skipped": skipped, "cases": summaries}
    (out_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    print(f"[done] results in {out_dir}")
    if failures:
        for failure in failures:
            print(f"[error] {failure}")
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default="/dev/shm/cp_ab", help="output directory")
    parser.add_argument("--repo-root", default=".", help="working directory for the launcher")
    parser.add_argument(
        "--launcher",
        default="bash tools/cp_balance_compare/launcher_template.sh {port}",
        help="server launcher command template; {port} is substituted",
    )
    parser.add_argument("--urls", default="", help="A=http://..,B=http://..,C=http://.. (skip launching)")
    parser.add_argument(
        "--configs",
        default="A,B,C",
        help="comma-separated configs to run; the first one is the reference. "
        "Use 'B,C' to compare the DSA-CP baseline directly against cp_balance.",
    )
    parser.add_argument("--base-port", type=int, default=12800)
    parser.add_argument("--model", default="glm")
    parser.add_argument("--prompt-lens", default="2048,2049,4096", help="comma-separated synthetic prompt lengths")
    parser.add_argument("--multi-lens", default="", help="multi-request groups separated by ';'")
    parser.add_argument("--prompts-file", default="", help="JSONL/JSON prompts; replaces --prompt-lens")
    parser.add_argument("--vocab", type=int, default=100000, help="synthetic token id upper bound")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--cp-size", type=int, default=8)
    parser.add_argument("--spec-config", default="", help="VLLM_ASCEND_SPEC_CONFIG; empty disables MTP")
    parser.add_argument(
        "--kv-transfer-config",
        default="",
        help="VLLM_ASCEND_KV_TRANSFER_CONFIG JSON; empty disables the PD connector",
    )
    parser.add_argument("--no-kv-connector", action="store_true", help="force-disable the PD connector")
    parser.add_argument("--embed-local", default="0", help="VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL for all configs")
    parser.add_argument("--multi-mode", choices=("batch", "concurrent"), default="batch")
    parser.add_argument(
        "--extra-additional-config",
        default="",
        help="JSON object merged into the base additional_config for every config "
        "(e.g. '{\"recompute_scheduler_enable\": false}')",
    )
    parser.add_argument("--env", action="append", default=[], metavar="K=V", help="extra env override")
    parser.add_argument(
        "--repeat-a",
        action="store_true",
        help="also run the first/reference config again as <ref>2 for the noise floor",
    )
    parser.add_argument("--config-check", choices=("off", "warn", "strict"), default="warn")
    parser.add_argument(
        "--zigzag-check",
        choices=("off", "warn", "strict"),
        default="warn",
        help="check the C server log for the zigzag activation marker",
    )
    parser.add_argument(
        "--on-zigzag-miss",
        choices=("skip", "continue"),
        default="skip",
        help="when strict zigzag check fails: stop launching remaining configs, "
        "still write results/summary and exit non-zero (skip), or run everything (continue)",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="only validate launcher/env/fingerprint with DRY_RUN=1 (no model load)",
    )
    parser.add_argument(
        "--restart-wait",
        type=float,
        default=30.0,
        help="seconds to wait after killing a server before launching the next one",
    )
    parser.add_argument("--startup-timeout", type=float, default=3600)
    parser.add_argument(
        "--wait-log-every",
        type=float,
        default=30.0,
        help="seconds between '/health not ready' heartbeat lines while the server loads",
    )
    parser.add_argument("--http-timeout", type=float, default=900)
    parser.add_argument("--http-retries", type=int, default=3)
    parser.add_argument("--delta-threshold", type=float, default=0.05)
    parser.add_argument("--run-len", type=int, default=8)
    parser.add_argument("--agree-window", type=int, default=65)
    parser.add_argument("--plot-points", type=int, default=3000)
    parser.add_argument("--annotate-plan", dest="annotate_plan", action="store_true", default=True)
    parser.add_argument("--no-annotate-plan", dest="annotate_plan", action="store_false")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.add_argument("--no-csv", dest="csv", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.env_override = [item.split("=", 1) for item in args.env]
    return args


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
