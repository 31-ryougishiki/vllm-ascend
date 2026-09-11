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
"""Self test for ``ab_cp_compare.py`` using ``mock_vllm_server.py``.

No NPU is required: the mock server speaks just enough of the vLLM OpenAI
protocol (``/health`` + ``/v1/completions`` with ``echo``/``logprobs``) to run
the driver end to end.  Run it directly or through pytest::

    python tools/cp_balance_compare/selftest_mock.py
    pytest tools/cp_balance_compare/selftest_mock.py -q
"""

from __future__ import annotations

import importlib.util
import io
import json
import shutil
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


driver = _load("cp_balance_ab_cp_compare", "ab_cp_compare.py")
mock = _load("cp_balance_mock_vllm_server", "mock_vllm_server.py")
single = _load("cp_balance_run_single", "run_single.py")
try:
    checker = _load("cp_balance_check_zigzag_dumps", "check_zigzag_dumps.py")
except ImportError:  # pragma: no cover - only the dump checker needs torch
    checker = None


def _temp_dir(prefix: str) -> Path:
    """Create a scratch dir that also works in sandboxed dev environments."""
    import os

    base = Path(os.environ.get("CP_AB_SELFTEST_DIR") or os.getcwd())
    for attempt in range(1000):
        path = base / f"{prefix}{os.getpid()}_{attempt}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"cannot create a scratch dir under {base}")


def test_wait_ready_stops_when_launcher_exited() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=30)
    assert driver.wait_ready("http://127.0.0.1:1", timeout=5, proc=proc, label="X", log_every=1) is False


def test_wait_ready_ignores_proxy_env() -> None:
    import os

    servers, urls = mock.start_mock_servers({"B": 0.0})
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy")
    saved = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = "http://127.0.0.1:9"  # an unreachable proxy
        # The driver session must bypass proxies for localhost.
        assert driver.wait_ready(urls["B"], timeout=10, proc=None, label="B", log_every=100)
    finally:
        mock.stop_mock_servers(servers)
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_parse_choice_token_ids_prompt() -> None:
    choice = mock.make_choice(0, [11, 22, 33, 44], top_k=3, offset=0.0, echo=True, key_style="ids")
    entry = driver.parse_choice(choice, {"choices": [choice]}, 3)
    assert entry["n_prompt"] == 4
    assert entry["n_out"] == 5
    assert entry["tok_lp"][0] is None
    assert entry["top1"][0] is None
    assert entry["top1"][1] == "token_id:22"
    # last output position is the first generated token
    assert entry["top1"][4] == "token_id:45"


def test_parse_choice_all_key_styles_agree() -> None:
    """Text / placeholder / numeric top_logprobs keys must parse identically.

    The mock encodes candidate token ids *inside* the key, so the identities
    themselves differ between styles; what must be format independent is the
    structure: same lengths, same logprob values, one comparable identity per
    candidate (cross-format identity mapping is covered by
    ``test_parse_choice_text_keyed_top_logprobs``).
    """
    prompt = [11, 22, 33, 44]
    baseline = None
    for style in ("text", "ids", "numeric"):
        choice = mock.make_choice(0, prompt, top_k=3, offset=0.0, echo=True, key_style=style)
        entry = driver.parse_choice(choice, {"choices": [choice]}, 3)
        assert entry["n_prompt"] == 4 and entry["n_out"] == 5
        assert entry["top1"][0] is None
        assert entry["top1"][1] is not None and entry["top1"][4] is not None
        assert len(entry["top5"][1]) == 3
        assert all(isinstance(token, str) and token for token in entry["top5"][1])
        if baseline is None:
            baseline = entry
            continue
        assert entry["tok_lp"] == baseline["tok_lp"]
        assert entry["n_prompt"] == baseline["n_prompt"]
        assert entry["top1_lp"] == baseline["top1_lp"]


def test_token_key_normalisation() -> None:
    assert driver.token_key(123) == "token_id:123"
    assert driver.token_key("123") == "token_id:123"
    assert driver.token_key("token_id:123") == "token_id:123"
    assert driver.token_key("Redux") == "tok:Redux"
    assert driver.token_key(",arr") == "tok:,arr"


def test_parse_choice_text_prompt_uses_response_prompt_token_ids() -> None:
    text = "hello ascend cp balance"
    choice = mock.make_choice(0, text, top_k=3, offset=0.0, echo=True)
    entry = driver.parse_choice(choice, {"choices": [choice]}, 3)
    assert entry["prompt_token_ids"] == mock.fake_tokenize(text)
    assert entry["n_prompt"] == len(mock.fake_tokenize(text))
    assert entry["n_out"] == entry["n_prompt"] + 1


def test_parse_choice_prompt_logprobs_fallback() -> None:
    choice = mock.make_choice(0, [5, 6, 7, 8], top_k=3, offset=0.0, echo=False, key_style="ids")
    assert "logprobs" not in choice
    entry = driver.parse_choice(choice, {"choices": [choice]}, 3)
    assert entry["n_prompt"] == 4
    assert entry["n_out"] == 4
    assert entry["top1"][1] == "token_id:6"
    assert entry["tok_lp"][1] is not None


def test_fingerprint_verification() -> None:
    out = Path(_temp_dir("cp_ab_fp_"))
    try:
        strict = driver.parse_args(["--out", str(out), "--config-check", "strict"])
        warn = driver.parse_args(["--out", str(out), "--config-check", "warn"])
        # The "no fingerprint" branch below must not wait out the real budget.
        strict.fingerprint_wait_s = 0.0
        warn.fingerprint_wait_s = 0.0
        log = out / "server_C.log"

        expected = driver.expected_fingerprint("C", strict)
        fingerprint = "[cp-ab] " + " ".join(f"{key}={value}" for key, value in expected.items())
        # Realistic ordering: the shipped launchers print the vendored set_env
        # note (same "[cp-ab]" prefix) before the fingerprint, and the effective
        # additional_config after it.
        log.write_text(
            "[cp-ab] vendor env: /tmp/fake_vendor/set_env.bash\n"
            f"{fingerprint}\n"
            f"[cp-ab-cfg] {json.dumps(driver.expected_additional_config('C', strict))}\n",
            encoding="utf-8",
        )
        def verify(args_) -> tuple[object, str]:
            """``(result_or_exception, captured_output)`` for one verify_config call.

            The negative cases below print on purpose (``[warn] C config mismatch``);
            letting that text reach the console makes it indistinguishable from a
            *real* round's configuration failure -- which is exactly how it was
            misread once.  Capture it and assert on it instead.
            """
            buffer = io.StringIO()
            try:
                with redirect_stdout(buffer):
                    result = driver.verify_config("C", args_, log)
            except Exception as exc:  # noqa: BLE001
                return exc, buffer.getvalue()
            return result, buffer.getvalue()

        result, text = verify(strict)
        assert result is True, (result, text)
        assert "fingerprint OK" in text, text

        log.write_text("[cp-ab] CP_BALANCE=0 DSA_CP=1 EMBED_LOCAL=0 SPEC=0 KV=0\n")
        result, text = verify(strict)
        assert isinstance(result, RuntimeError), (result, text)  # strict aborts a round
        result, text = verify(warn)
        assert result is False and "config mismatch" in text, (result, text)

        log.write_text("no fingerprint in this log\n")
        result, text = verify(strict)
        assert isinstance(result, RuntimeError), (result, text)
        result, text = verify(warn)
        assert result is False and "has no [cp-ab] fingerprint" in text, (result, text)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_fingerprint_line_ignores_other_cp_ab_lines() -> None:
    """``[cp-ab]`` is reused for notes; only the CP_BALANCE line is the fingerprint.

    The shipped launchers echo ``[cp-ab] vendor env: ...`` before the
    fingerprint, and the lookup used to take the first ``[cp-ab]`` match, so
    ``preflight`` reported every config as unverified (``actual={}``) even when
    the launcher had in fact applied the env overrides.
    """
    text = "\n".join(
        [
            "[cp-ab] vendor env: /vllm-workspace/vllm-ascend/.../set_env.bash",
            "[cp-ab] CP_BALANCE=1 DSA_CP=1 MIN_TOKENS=2048 EMBED_LOCAL=0 SPEC=0 KV=0",
            '[cp-ab-cfg] {"enable_dsa_cp": true}',
        ]
    )
    assert driver.is_fingerprint_line("[cp-ab] CP_BALANCE=1 DSA_CP=1")
    assert not driver.is_fingerprint_line("[cp-ab] vendor env: /x/set_env.bash")
    assert not driver.is_fingerprint_line("[cp-ab-cfg] {}")
    assert driver.find_fingerprint("no fingerprint here") is None

    line = driver.find_fingerprint(text)
    assert line is not None and "CP_BALANCE=1" in line
    assert driver.parse_fingerprint(line) == {
        "CP_BALANCE": 1,
        "DSA_CP": 1,
        "MIN_TOKENS": 2048,
        "EMBED_LOCAL": 0,
        "SPEC": 0,
        "KV": 0,
    }

    # The log-side lookup used by verify_config must survive the same ordering.
    out = _temp_dir("cp_ab_fpline_")
    try:
        log = out / "server_C.log"
        log.write_text(text + "\n", encoding="utf-8")
        from_log = driver.last_fingerprint(log)
        assert from_log is not None
        assert driver.parse_fingerprint(from_log) == driver.parse_fingerprint(line)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_stream_log_mirrors_to_screen_and_file() -> None:
    """A live model load must be followable while the raw log stays on disk.

    ``--no-stream-log`` only drops the echo; the file write is what
    ``verify_config``/``tail()`` read, so it must always happen.
    """
    out = _temp_dir("cp_ab_stream_")
    try:
        log_path = out / "server_B.log"
        payload = "line one\n\nline two\n"
        with open(log_path, "w", encoding="utf-8") as handle:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                driver._stream_log(io.StringIO(payload), handle, "[B]")
        # Raw lines, byte for byte, including the blank one.
        assert log_path.read_text(encoding="utf-8") == payload
        printed = buffer.getvalue()
        assert "[B] line one" in printed
        assert "[B] line two" in printed
        assert "[B] \n" not in printed  # blank lines are not echoed

        # prefix=None: file only (what --no-stream-log does).
        with open(log_path, "w", encoding="utf-8") as handle:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                driver._stream_log(io.StringIO(payload), handle, None)
        assert log_path.read_text(encoding="utf-8") == payload
        assert buffer.getvalue() == ""
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_launch_server_streams_log_to_screen() -> None:
    """A real launcher's output must reach both the log file and the screen."""
    if not _bash_usable():
        print("[skip] bash not usable on this host")
        return
    out = _temp_dir("cp_ab_launch_")
    try:
        script = out / "noisy_launcher.sh"
        script.write_text(
            "#!/usr/bin/env bash\n"
            "echo '[cp-ab] CP_BALANCE=0 DSA_CP=1 MIN_TOKENS=2048 EMBED_LOCAL=0 SPEC=0 KV=0'\n"
            "echo 'loading weights 1/2'\n"
            "echo 'loading weights 2/2'\n",
            encoding="utf-8",
        )
        args = driver.parse_args(
            ["--launcher", f"bash {script.as_posix()} {{port}}", "--repo-root", str(out)]
        )
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            proc, _port, log, log_path, reader = driver.launch_server(args, "C", out)
            proc.wait(timeout=120)
            driver.stop_server(proc, log, reader, 0.0)
        raw = log_path.read_text(encoding="utf-8")
        assert "loading weights 2/2" in raw
        assert driver.find_fingerprint(raw) is not None, raw
        printed = buffer.getvalue()
        assert "[launch] C" in printed
        assert "[C] loading weights 1/2" in printed
        assert "[C] loading weights 2/2" in printed
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_run_single_send_once_against_mock() -> None:
    """The debug script must send and parse exactly what an A/B round does."""
    servers, urls = mock.start_mock_servers({"B": 0.0})
    out = _temp_dir("cp_ab_single_")
    try:
        args = driver.parse_args(["--out", str(out), "--topk", "3"])
        prompts = [[11, 22, 33, 44]]
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            entries = single.send_once(driver, urls["B"], args, prompts, "single_L4", out)
        assert entries is not None and len(entries) == 1, buffer.getvalue()
        assert entries[0]["n_prompt"] == 4
        printed = buffer.getvalue()
        assert "[http] <- 200" in printed

        # Both bodies are kept so a failing request can be replayed with curl.
        payload = json.loads((out / "payload_single_L4.json").read_text(encoding="utf-8"))
        assert payload["prompt"] == [11, 22, 33, 44]
        assert payload["logprobs"] == 3 and payload["prompt_logprobs"] == 3
        assert payload["max_tokens"] == 1 and payload["echo"] is True
        assert payload["return_token_ids"] is True
        assert payload["add_special_tokens"] is False
        assert (out / "response_single_L4.txt").exists()
    finally:
        mock.stop_mock_servers(servers)
        shutil.rmtree(out, ignore_errors=True)


def test_run_single_url_mode_against_mock() -> None:
    """End to end without an NPU: --url mode launches nothing and still reports."""
    servers, urls = mock.start_mock_servers({"B": 0.0})
    out = _temp_dir("cp_ab_single_main_")
    try:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = single.main(
                ["--url", urls["B"], "--out", str(out), "--prompt-lens", "4", "--topk", "3", "--no-keep"]
            )
        printed = buffer.getvalue()
        assert rc == 0, printed
        assert "[http] <- 200" in printed
        assert "[result] single_L4.0 n_prompt=4" in printed
        assert "[pass] 所有 case 推理成功" in printed
    finally:
        mock.stop_mock_servers(servers)
        shutil.rmtree(out, ignore_errors=True)


def test_run_single_rejects_unknown_config() -> None:
    """'A' (DSA-CP off) is not part of this tool: the debug script must say so."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        rc = single.main(["--config", "A", "--url", "http://127.0.0.1:1"])
    assert rc == 2
    assert "不支持" in buffer.getvalue()


def test_check_zigzag_kv_summary_names_first_diverging_layer() -> None:
    """A multi-layer sweep must print one table row per layer and name the first one.

    The sweep is the expensive round's payload (``DUMP_SPEC=kv:all``), so the
    summary has to be readable: per-layer ranks/rows/first-token/magnitude plus a
    single decisive "FIRST DIVERGENCE" line.
    """
    if checker is None:
        print("[skip] torch not available (check_zigzag_dumps needs it)")
        return
    import argparse

    import torch

    out = _temp_dir("cp_ab_kvsummary_")
    try:
        torch.manual_seed(1234)  # the fixture (and its printed magnitudes) must be reproducible
        base = torch.randint(-120, 120, (256, 64), dtype=torch.int8)
        base_fp = torch.randn(256, 64, dtype=torch.float16)

        def dump(layer: int, rank: int, cpbal: int, packed, fp, ts: int) -> None:
            torch.save(
                {"kv_nat": packed.clone().view(torch.uint8), "kv_fp_nat": fp.clone()},
                out / f"kv_cpbal{cpbal}_layer{layer}_rank{rank}_pid{100 + rank}_{ts}.pt",
            )

        # layer 0: the packed (int8) copy is identical but the FP copy is not --
        # exactly the blind spot that quantized dumps cannot see.
        fp0 = base_fp.clone()
        fp0[64:, :8] = (fp0[64:, :8].float() + 0.01).to(torch.float16)
        # layer 1: both copies differ from token 64 on.
        packed1 = base.clone()
        packed1[64:, :32] = (packed1[64:, :32].to(torch.int16) + 2).to(torch.int8)
        fp1 = base_fp.clone()
        fp1[64:, :8] = (fp1[64:, :8].float() + 1.0).to(torch.float16)

        for rank in range(2):
            dump(0, rank, 0, base, base_fp, 1000)
            dump(1, rank, 0, base, base_fp, 1000)
            dump(0, rank, 1, base, fp0, 1001)
            dump(1, rank, 1, packed1, fp1, 1001)

        args = argparse.Namespace(dir=str(out), summary_only=True)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = checker.check_kv(args)
        text = buffer.getvalue()
        assert rc == 1, text
        assert "[kv/fp  ]" in text, text
        # The FP copy must win: it sees layer 0, which the int8 copy reports as clean.
        assert "FIRST DIVERGENCE (fp/value): layer 0" in text, text
        assert "first token 64" in text, text
        assert "[kv/int8]     0     0/2" in text, text  # the blind spot the FP copy closes
        # The layer-0 FP row must report the magnitude in real units (~1e-2),
        # not the int8 step count the packed copy would show.
        fp_row = next(line for line in text.splitlines() if line.startswith("[kv/fp  ]     0"))
        assert "e-02" in fp_row, fp_row

        # Identical layouts (both copies) must report no divergence at all.
        for rank in range(2):
            dump(0, rank, 1, base, base_fp, 1002)
            dump(1, rank, 1, base, base_fp, 1002)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = checker.check_kv(args)
        text = buffer.getvalue()
        assert rc == 0, text
        assert "FIRST DIVERGENCE (fp): none" in text, text
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_check_zigzag_kv_summary_fp_only_dumps() -> None:
    """FP-only dumps (a site whose NPU index_select fails) must still judge P2.

    The packed copy is best effort; if it is missing the int8 table must say so
    instead of reporting "identical", and the verdict must come from the FP copy.
    """
    if checker is None:
        print("[skip] torch not available (check_zigzag_dumps needs it)")
        return
    import argparse

    import torch

    out = _temp_dir("cp_ab_fponly_")
    try:
        torch.manual_seed(7)
        base = torch.randn(64, 8, dtype=torch.float16)
        shifted = base.clone()
        shifted[32:, :4] = (shifted[32:, :4].float() + 0.01).to(torch.float16)
        for rank in range(2):
            torch.save({"kv_fp_nat": base.clone()}, out / f"kv_cpbal0_layer0_rank{rank}_pid{rank}_1000.pt")
            torch.save({"kv_fp_nat": shifted.clone()}, out / f"kv_cpbal1_layer0_rank{rank}_pid{rank}_1001.pt")
            torch.save({"kv_fp_nat": base.clone()}, out / f"kv_cpbal0_layer1_rank{rank}_pid{rank}_1000.pt")
            torch.save({"kv_fp_nat": base.clone()}, out / f"kv_cpbal1_layer1_rank{rank}_pid{rank}_1001.pt")
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = checker.check_kv(argparse.Namespace(dir=str(out), summary_only=True))
        text = buffer.getvalue()
        assert rc == 1, text
        assert "n/a (no kv_nat in dump" in text, text
        assert "FIRST DIVERGENCE (fp/value): layer 0" in text, text
        assert "first token 32" in text, text
        assert "layer 0 的 KV 直接来自 embedding" in text, text
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_check_zigzag_kv_skips_truncated_dumps() -> None:
    """A truncated dump (the round ran out of disk) must not abort the analysis.

    A full-layer sweep writes GBs; when the dump directory fills up, ``torch.save``
    leaves partial files ("failed finding central directory").  The remaining
    layers are still worth reading, so one bad file is skipped with a warning.
    """
    if checker is None:
        print("[skip] torch not available (check_zigzag_dumps needs it)")
        return
    import argparse

    import torch

    out = _temp_dir("cp_ab_truncated_")
    try:
        torch.manual_seed(11)
        base = torch.randn(32, 8, dtype=torch.float16)
        for rank in (0, 1):
            torch.save({"kv_fp_nat": base.clone()}, out / f"kv_cpbal0_layer0_rank{rank}_pid{rank}_1000.pt")
            if rank == 0:
                (out / f"kv_cpbal1_layer0_rank{rank}_pid{rank}_1001.pt").write_bytes(b"PK\x03\x04truncated")
            else:
                shifted = base.clone()
                shifted[16:, :4] = (shifted[16:, :4].float() + 0.01).to(torch.float16)
                torch.save({"kv_fp_nat": shifted.clone()}, out / f"kv_cpbal1_layer0_rank{rank}_pid{rank}_1001.pt")
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = checker.check_kv(argparse.Namespace(dir=str(out), summary_only=True))
        text = buffer.getvalue()
        # The good rank is still compared (1/1), the truncated one is dropped.
        assert rc == 1, text
        assert "1/1" in text, text
        assert "FIRST DIVERGENCE (fp/value): layer 0" in text, text
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_cp_balance_modules_import_locally_used_stdlib() -> None:
    """A stdlib name used at module level without an import is a NameError at start-up.

    This exact bug cost a round: ``_dump_dir()`` called ``os.getenv`` at import
    time while ``sfa_v1.py`` only imported ``os`` inside one function.  Neither
    py_compile nor importing the tool modules can see it, and the only symptom is
    a server that dies while loading the model -- so check it statically here.
    """
    import ast

    stdlib = {"os", "sys", "time", "math", "json", "shutil", "glob", "re", "threading"}
    repo = HERE.parent.parent
    problems: list[str] = []
    # ``model_runner_v1.py`` hosts the MLP/quant dump installers, i.e. the code
    # most likely to grow a function-local import we forget to declare.
    for relative in (
        "vllm_ascend/attention/sfa_v1.py",
        "vllm_ascend/envs.py",
        "vllm_ascend/worker/model_runner_v1.py",
    ):
        path = repo / relative
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Only *top-level* imports count as module scope: a function-local
        # `import os` must not make `os` look available to another function.
        module_level: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                module_level.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module_level.update(alias.asname or alias.name for alias in node.names)

        def imported_in(scope: ast.AST) -> set[str]:
            names: set[str] = set()
            for node in ast.walk(scope):
                if isinstance(node, ast.Import):
                    names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    names.update(alias.asname or alias.name for alias in node.names)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    names.add(node.id)
            return names

        for func in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            local = {arg.arg for arg in (*func.args.args, *func.args.kwonlyargs)} | imported_in(func)
            for node in ast.walk(func):
                if (
                    isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)
                    and node.id in stdlib
                    and node.id not in local
                    and node.id not in module_level
                ):
                    problems.append(f"{relative}:{node.lineno}: {func.name}() uses {node.id} without an import")
    assert not problems, "\n".join(problems)


def test_check_zigzag_kv_handles_fp8_dumps() -> None:
    """Sparse-C8 sites pack the KV as fp8; the comparison must not choke on it.

    ``tensor.numpy()`` raises "Got unsupported ScalarType Float8_e4m3fn", which
    crashed a whole 1248-dump sweep at the first comparison.  The conversion has
    to go through torch (``.float()``).
    """
    if checker is None:
        print("[skip] torch not available (check_zigzag_dumps needs it)")
        return
    import argparse

    import torch

    if not hasattr(torch, "float8_e4m3fn"):
        print("[skip] this torch has no float8_e4m3fn")
        return

    out = _temp_dir("cp_ab_fp8_")
    try:
        torch.manual_seed(13)
        base = torch.randn(64, 32).to(torch.float8_e4m3fn)
        shifted = base.clone().float()
        shifted[32:, :8] += 0.5
        shifted = shifted.to(torch.float8_e4m3fn)
        for rank in range(2):
            torch.save({"kv_fp_nat": base.clone()}, out / f"kv_cpbal0_layer0_rank{rank}_pid{rank}_1000.pt")
            torch.save({"kv_fp_nat": shifted.clone()}, out / f"kv_cpbal1_layer0_rank{rank}_pid{rank}_1001.pt")
        buffer, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(errors):
            rc = checker.check_kv(argparse.Namespace(dir=str(out), summary_only=True))
        text = buffer.getvalue()
        assert rc == 1, text + errors.getvalue()
        assert "FIRST DIVERGENCE (fp/value): layer 0" in text, text
        assert "first token 32" in text, text
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_check_zigzag_act_joins_by_position() -> None:
    """The activation trace must be compared by token position, never by row.

    With ``cp_size=2`` and 8 tokens the blocks are ``[0,1] [2,3] [4,5] [6,7]``:
    the continuous layout gives rank 0 the range ``[0,4)``, while zigzag gives it
    ``[0,1] + [6,7]``.  The same row index is therefore a different token, so a
    row-wise comparison would report a difference in ``in`` (which is identical)
    and would mis-place the first divergence.  Joining by position must report
    ``in`` as clean and the first ``out`` difference at token 2 -- the first
    token whose causal window crosses a block boundary.
    """
    if checker is None:
        print("[skip] torch not available (check_zigzag_dumps needs it)")
        return
    import argparse

    import torch

    out = _temp_dir("cp_ab_act_")
    try:
        torch.manual_seed(21)
        base = torch.randn(8, 4).to(torch.bfloat16)
        shifted = base.float().clone()
        shifted[2:] += 0.5  # from token 2 on: the first token past the first block
        shifted = shifted.to(torch.bfloat16)

        def save(kind: str, cpbal: int, rank: int, positions: list[int], data, ts: int = 0) -> None:
            # position -1 models a padding row (slot -1): it must be dropped.
            torch.save(
                {
                    "kind": "act",
                    "op": kind[3:],
                    "positions": torch.tensor(positions, dtype=torch.int32),
                    "act": data[[max(p, 0) for p in positions]],
                    "num_actual_tokens": 8,
                },
                out / f"{kind}_cpbal{cpbal}_layer0_rank{rank}_pid{100 + rank}_{(1000 + cpbal) if not ts else ts}.pt",
            )

        for rank in range(2):
            positions = list(range(4 * rank, 4 * rank + 4))  # continuous slice
            save("actin", 0, rank, positions, base)
            save("actout", 0, rank, positions, base)
        padding = {0: [0, 1, 6, 7, -1], 1: [2, 3, 4, 5, -1]}  # zigzag + one pad row
        for rank, positions in padding.items():
            save("actin", 1, rank, positions, base)
            save("actout", 1, rank, positions, shifted)

        buffer, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(errors):
            rc = checker.check_act(argparse.Namespace(dir=str(out), summary_only=True))
        text = buffer.getvalue()
        assert rc == 1, text + errors.getvalue()
        # `in` is bit-identical once compared by position: proof that the join is
        # by token and not by row.
        assert "[act]     0   in         8        0" in text, text
        assert "[act]     0  out         8        6" in text, text
        assert "FIRST DIVERGENCE (act): layer 0 op=out at token 2" in text, text
        assert "attention 内部产生" in text, text

        # Identical layouts (both ops) must report no divergence at all.
        for rank, positions in padding.items():
            save("actout", 1, rank, positions, base, ts=1002)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = checker.check_act(argparse.Namespace(dir=str(out), summary_only=True))
        text = buffer.getvalue()
        assert rc == 0, text
        assert "FIRST DIVERGENCE (act): none" in text, text
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_check_zigzag_act_pins_moe_divergence() -> None:
    """Equal attention output + different next-layer input means the MoE/MLP did it.

    Layer 0 is clean in both ops; layer 1's *input* differs while its attention
    output does not.  The verdict must name layer 1 / op=in (not op=out) and point
    at the MoE/MLP of layer 0, otherwise the trace cannot bisect attention vs MLP.
    """
    if checker is None:
        print("[skip] torch not available (check_zigzag_dumps needs it)")
        return
    import argparse

    import torch

    out = _temp_dir("cp_ab_act_moe_")
    try:
        torch.manual_seed(22)
        base = torch.randn(8, 4).to(torch.bfloat16)
        shifted = base.float().clone()
        shifted[3:] += 0.5  # the MoE/MLP of layer 0 diverges from token 3 on
        shifted = shifted.to(torch.bfloat16)
        ranks = {0: [0, 1, 6, 7], 1: [2, 3, 4, 5]}
        for layer in (0, 1):
            for rank, positions in ranks.items():
                for cpbal in (0, 1):
                    data_in = base if (layer == 0 or cpbal == 0) else shifted
                    for op in ("in", "out"):
                        torch.save(
                            {
                                "kind": "act",
                                "op": op,
                                "positions": torch.tensor(positions, dtype=torch.int32),
                                "act": (data_in if op == "in" else base)[positions],
                                "num_actual_tokens": 8,
                            },
                            out / f"act{op}_cpbal{cpbal}_layer{layer}_rank{rank}_pid{100 + rank}_{1000 + cpbal}.pt",
                        )
        buffer, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(errors):
            rc = checker.check_act(argparse.Namespace(dir=str(out), summary_only=True))
        text = buffer.getvalue()
        assert rc == 1, text + errors.getvalue()
        assert "[act]     0  out         8        0" in text, text
        assert "[act]     1   in         8        5" in text, text
        assert "FIRST DIVERGENCE (act): layer 1 op=in at token 3" in text, text
        assert "MoE/MLP" in text, text
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_check_zigzag_kv_separates_bytes_from_values() -> None:
    """A flipped *sign of zero* is a byte difference, not a value difference.

    e4m3 stores every finite number exactly once except ``+0``/``-0``, so two
    dumps can differ in bytes on thousands of rows while every stored *value* is
    identical.  That is not a content error -- it is the fingerprint of a
    difference *below* one quantization step (the pre-quantization values differ
    and one of them flipped the sign of a value that rounded to zero).  The
    checker must report it as sub-quantization (rc=0, ``rows_value == 0``) and
    never as a value divergence (rc=1); the TP=8 KV sweep produced exactly this
    signature (``rows_differ`` in the thousands with ``max|d| = 0.000e+00``).
    """
    if checker is None:
        print("[skip] torch not available (check_zigzag_dumps needs it)")
        return
    import argparse

    import torch

    if not hasattr(torch, "float8_e4m3fn"):
        print("[skip] this torch has no float8_e4m3fn")
        return

    out = _temp_dir("cp_ab_bytesonly_")
    try:
        torch.manual_seed(11)
        rows = 256
        base = (torch.randn(rows, 8) * 2).to(torch.float8_e4m3fn)
        base_raw = base.view(torch.uint8).clone()
        # Column 0 is a hard +0 from token 128 on in B ...
        base_raw[128:, 0] = 0x00
        left = base_raw.view(torch.float8_e4m3fn)
        # ... and the same *value* (-0.0 == 0.0) stored as -0 in C.
        right_raw = base_raw.clone()
        right_raw[128:, 0] = 0x80
        right = right_raw.view(torch.float8_e4m3fn)
        assert torch.equal(left.float(), right.float()), "fixture must differ only in bytes"
        for rank in range(2):
            torch.save({"kv_fp_nat": left.clone()}, out / f"kv_cpbal0_layer0_rank{rank}_pid{rank}_1000.pt")
            torch.save({"kv_fp_nat": right.clone()}, out / f"kv_cpbal1_layer0_rank{rank}_pid{rank}_1001.pt")
        buffer, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(errors):
            rc = checker.check_kv(argparse.Namespace(dir=str(out), summary_only=True))
        text = buffer.getvalue()
        assert rc == 0, text + errors.getvalue()
        fp_row = next(line for line in text.splitlines() if line.startswith("[kv/fp  ]     0"))
        # layer, val_ranks, rows_val, first_val, rows_byte, first_byte, max|d|, rel, byte_only
        assert fp_row.split() == [
            "[kv/fp", "]", "0", "0/2", "0-0", "-", "128-128", "128", "0.000e+00", "0.00e+00", "128",
        ], fp_row
        assert "FIRST DIVERGENCE (fp/bytes): layer 0" in text, text
        assert "sub-quantization" in text, text
        assert "no value-level difference" in text, text
        assert "P2 violated" not in text, text
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_check_zigzag_kv_reports_nan_difference_instead_of_zero() -> None:
    """A NaN must never be summarised as ``max|d| = 0.000e+00``.

    The packed KV row is not a pure fp8 array: on a sparse-C8 site the last
    bytes are a bf16 rope payload and fp32 scales transported through an
    fp8-typed tensor, so reading the whole row as fp8 yields NaNs (0x7F/0xFF
    patterns).  ``delta.max()`` is then NaN and plain ``max(0.0, nan)`` returns
    ``0.0`` -- that is exactly how a real divergence in this project was
    reported as ``max|d| = 0.000e+00``.  The magnitude must stay finite and the
    one-sided NaN must be visible.
    """
    if checker is None:
        print("[skip] torch not available (check_zigzag_dumps needs it)")
        return
    import argparse

    import torch

    out = _temp_dir("cp_ab_nan_")
    try:
        torch.manual_seed(31)
        left = torch.randn(8, 4).to(torch.bfloat16)
        right = left.clone()
        right[4:, 0] = float("nan")
        for rank in range(2):
            torch.save({"kv_fp_nat": left.clone()}, out / f"kv_cpbal0_layer0_rank{rank}_pid{rank}_1000.pt")
            torch.save({"kv_fp_nat": right.clone()}, out / f"kv_cpbal1_layer0_rank{rank}_pid{rank}_1001.pt")
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = checker.check_kv(argparse.Namespace(dir=str(out), summary_only=True))
        text = buffer.getvalue()
        assert rc == 1, text
        fp_row = next(line for line in text.splitlines() if line.startswith("[kv/fp  ]     0"))
        assert "0.000e+00" not in fp_row, fp_row  # never hide a NaN difference as zero
        assert "NaN-only pair(s)" in fp_row, fp_row
        assert "rows_val" in text and "4-4" in fp_row, fp_row
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_check_zigzag_topk_cross_layout_uses_positions() -> None:
    """The B/C top-k comparison must match rows by token, and flag order-only changes.

    Same trap as the activation trace: under zigzag rank 0 owns ``[0,1]+[6,7]``
    while the continuous layout gives it ``[0..3]``, so a row-wise comparison
    would report nonsense.  With ``positions`` present the checker must compare
    per token and distinguish "different set" (a real indexer bug) from "same
    set, different order" (the SFA kernel consumes the list as given, so this is
    a prime suspect for a numeric difference).
    """
    if checker is None:
        print("[skip] torch not available (check_zigzag_dumps needs it)")
        return
    import argparse

    import torch

    out = _temp_dir("cp_ab_topkcross_")
    try:
        width = 8

        def save(cpbal: int, rank: int, positions: list[int], rows: list[list[int]],
                 q_prefix: list[int], kv_raw: list[int], ts: int = 0) -> None:
            topk = torch.tensor(
                [row + [-1] * (width - len(row)) for row in rows], dtype=torch.int32
            )
            torch.save(
                {
                    "kind": "topk",
                    "topk_indices": topk,
                    "positions": torch.tensor(positions, dtype=torch.int32),
                    "actual_seq_lengths_query": torch.tensor(q_prefix, dtype=torch.int32),
                    "actual_seq_lengths_key": torch.tensor(kv_raw, dtype=torch.int32),
                },
                out / f"topk_cpbal{cpbal}_layer0_rank{rank}_pid{100 + rank}_{(1000 + cpbal) if not ts else ts}.pt",
            )

        # B (continuous): rank 0 -> tokens 0..3 (one batch, kv=4), rank 1 -> 4..7
        # (one batch, kv=8).  Every list is the full ascending causal window, so
        # the per-dump window check passes as well.
        save(0, 0, [0, 1, 2, 3],
             [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]], [4], [4])
        save(0, 1, [4, 5, 6, 7],
             [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6],
              [0, 1, 2, 3, 4, 5, 6, 7]], [4], [8])
        # C (zigzag): rank 0 -> tokens [0,1] + [6,7] (two batches), rank 1 ->
        # [2,3] + [4,5].  Same sets everywhere; token 7 is presented reversed.
        save(1, 0, [0, 1, 6, 7],
             [[0], [0, 1], [0, 1, 2, 3, 4, 5, 6], [7, 6, 5, 4, 3, 2, 1, 0]], [2, 4], [2, 8])
        save(1, 1, [2, 3, 4, 5],
             [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]], [2, 4], [4, 6])

        buffer, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(errors):
            rc = checker.check_topk(argparse.Namespace(dir=str(out), summary_only=True))
        text = buffer.getvalue()
        assert rc == 0, text + errors.getvalue()
        assert "[topk/cross]" in text, text
        cross_row = next(line for line in text.splitlines() if line.startswith("[topk/cross]     0"))
        # layer, compared, covered_B, covered_C, set_diff, order_diff, first_set, first_order
        expected = ["[topk/cross]", "0", "8", "8", "8", "0", "1", "-", "7"]
        assert cross_row.split() == expected, cross_row
        assert "same set in a different ORDER" in text, text

        # A different set must be reported as a real mismatch (rc=1): token 7
        # now selects a set that is not B's causal window either.
        save(1, 0, [0, 1, 6, 7],
             [[0], [0, 1], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6, 0]], [2, 4], [2, 8], ts=1002)
        save(1, 1, [2, 3, 4, 5],
             [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]], [2, 4], [4, 6], ts=1002)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = checker.check_topk(argparse.Namespace(dir=str(out), summary_only=True))
        text = buffer.getvalue()
        assert rc == 1, text
        assert "selected a DIFFERENT SET" in text, text
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_check_zigzag_act_positions_are_slots_not_token_indices() -> None:
    """A non-zero block base must not shrink (or shift) the comparison.

    ``positions`` are KV-cache slots: a 2048-token request whose block table
    starts at block 1 occupies slots 128..2175.  The checker used to drop rows
    with ``slot >= num_actual_tokens`` -- comparing 1920 of 2048 tokens and
    mislabelling the first divergence by the base.  It must instead keep every
    ``slot >= 0`` row and report the natural-stream token index.
    """
    if checker is None:
        print("[skip] torch not available (check_zigzag_dumps needs it)")
        return
    import argparse

    import torch

    out = _temp_dir("cp_ab_slotbase_")
    try:
        torch.manual_seed(41)
        base_t = torch.randn(8, 4).to(torch.bfloat16)
        shifted = base_t.clone()
        shifted[3:] = (shifted[3:].float() + 0.5).to(torch.bfloat16)
        slots = [128, 129, 130, 131, 132, 133, 134, 135]  # blocks start at block 1

        def save(kind: str, cpbal: int, data) -> None:
            torch.save(
                {
                    "kind": "act",
                    "op": kind[3:],
                    "positions": torch.tensor(slots, dtype=torch.int64),
                    "act": data.clone(),
                    "num_actual_tokens": 8,  # a *token* count, smaller than the slots
                },
                out / f"{kind}_cpbal{cpbal}_layer0_rank0_pid100_{1000 + cpbal}.pt",
            )

        for cpbal, data in ((0, base_t), (1, shifted)):
            save("actin", cpbal, base_t)   # inputs identical
            save("actout", cpbal, data)    # outputs differ from token 3 on
        buffer, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(errors):
            rc = checker.check_act(argparse.Namespace(dir=str(out), summary_only=True, block_size=4))
        text = buffer.getvalue()
        assert rc == 1, text + errors.getvalue()
        fp_row = next(line for line in text.splitlines() if line.startswith("[act]     0  out"))
        # [act] <layer> <op> <compared> <differ> <first_pos> <max|d|> <rel> <block>
        cols = fp_row.split()
        assert cols[3:6] == ["8", "5", "3"], fp_row   # all 8 tokens compared, first at token 3
        assert cols[8] == "0", fp_row                 # token 3 is in block 0 (block size 4)
        assert "[act]     0   in         8        0" in text, text
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_check_zigzag_act_covers_mlp_boundary() -> None:
    """`--kind act` must also report the MLP boundary (mlpin/mlpout files).

    The MLP dump is what bisects one layer into three steps: attention out
    (identical) -> MLP in -> MLP out -> next layer's attention in.  With the MLP
    output already different while its input is identical, the verdict must name
    ``op=mlp_out`` and point inside the MLP/MoE -- not at attention.
    """
    if checker is None:
        print("[skip] torch not available (check_zigzag_dumps needs it)")
        return
    import argparse

    import torch

    out = _temp_dir("cp_ab_mlp_")
    try:
        torch.manual_seed(51)
        base = torch.randn(8, 4).to(torch.bfloat16)
        shifted = base.clone()
        shifted[4:] = (shifted[4:].float() + 0.25).to(torch.bfloat16)
        positions = list(range(8))
        for kind, data in (
            ("actin", base), ("actout", base),          # attention: identical
            ("mlpin", base),                            # MLP input: identical
            ("guout", base),                            # gate_up output: identical
            ("dnin", base),                             # down_proj input: identical
            ("mlpout", shifted),                        # MLP output: differs from token 4
        ):
            for cpbal, payload in ((0, base), (1, data)):
                torch.save(
                    {
                        "kind": "mlp" if kind.startswith(("mlp", "gu", "dn")) else "act",
                        "positions": torch.tensor(positions, dtype=torch.int32),
                        "act": payload.clone(),
                        "num_actual_tokens": 8,
                    },
                    out / f"{kind}_cpbal{cpbal}_layer0_rank0_pid100_{1000 + cpbal}.pt",
                )
        buffer, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(errors):
            rc = checker.check_act(argparse.Namespace(dir=str(out), summary_only=True, block_size=4))
        text = buffer.getvalue()
        assert rc == 1, text + errors.getvalue()
        ops = [line.split()[2] for line in text.splitlines() if line.startswith("[act]     0 ")]
        assert ops == ["in", "out", "mlp_in", "gu_out", "dn_in", "mlp_out"], ops
        assert "FIRST DIVERGENCE (act): layer 0 op=mlp_out at token 4" in text, text
        assert "MLP/MoE **内部**" in text, text

        # A one-sided op must be reported, never silently omitted: dropping the
        # C dump of mlp_out makes the step look "identical" in the table.
        for cpbal in (1,):
            path = out / f"mlpout_cpbal{cpbal}_layer0_rank0_pid100_{1000 + cpbal}.pt"
            path.unlink()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            checker.check_act(argparse.Namespace(dir=str(out), summary_only=True, block_size=4))
        text = buffer.getvalue()
        assert "[act] INCOMPLETE layer=0 op=mlp_out: only cp_balance=['0'] dumped" in text, text
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_check_zigzag_act_qin_separates_quant_from_gemm() -> None:
    """`gu_q`/`dn_q` must separate "the quantization changed" from "the GEMM changed".

    With W8A8_MXFP8 weights the activation quantization is per row (32-element MX
    groups), so a bit-identical bf16 input *must* quantize identically.  If it does
    not, the root cause is the quantization step; if it does while the GEMM output
    still differs, the root cause is the GEMM kernel itself.  The verdict must say
    which one, from the same table -- without that split, ``gu_out`` alone cannot
    tell the two apart.
    """
    if checker is None:
        print("[skip] torch not available (check_zigzag_dumps needs it)")
        return
    import argparse

    import torch

    out = _temp_dir("cp_ab_qin_")
    try:
        torch.manual_seed(59)
        base = torch.rand(8, 4).add(0.5).to(torch.bfloat16)
        q_same = torch.rand(8, 4).add(0.5).to(torch.float8_e4m3fn)
        s_same = torch.full((8, 1), 127, dtype=torch.uint8)
        q_other = q_same.clone()
        q_other[4:] = (q_other[4:].float() + 0.5).to(torch.float8_e4m3fn)
        s_other = s_same.clone()
        s_other[4:] = 128
        shifted = base.clone()
        shifted[4:] = (shifted[4:].float() + 0.25).to(torch.bfloat16)
        positions = torch.tensor(list(range(8)), dtype=torch.int32)

        def save(kind: str, cpbal: int, payload: dict) -> None:
            payload = dict(payload)
            payload["positions"] = positions
            torch.save(
                payload,
                out / f"{kind}_cpbal{cpbal}_layer0_rank0_pid100_{1000 + cpbal}.pt",
            )

        def qin(q, s) -> dict:
            return {"kind": "qin", "op": "gu_q", "q": q.clone(), "s": s.clone()}

        def run() -> str:
            buffer, errors = io.StringIO(), io.StringIO()
            with redirect_stdout(buffer), redirect_stderr(errors):
                rc = checker.check_act(
                    argparse.Namespace(dir=str(out), summary_only=True, block_size=4)
                )
            text = buffer.getvalue()
            assert rc == 1, text + errors.getvalue()
            return text

        # 1) bf16 input AND the fp8/scale pair are identical; only the GEMM output
        #    differs -> the GEMM kernel is the root cause.
        for cpbal in (0, 1):
            save("mlpin", cpbal, {"kind": "mlp", "act": base.clone()})
            save("guq", cpbal, qin(q_same, s_same))
        for cpbal, data in ((0, base), (1, shifted)):
            save("guout", cpbal, {"kind": "mlp", "act": data.clone()})
        text = run()
        assert "[act] FIRST DIVERGENCE (act): layer 0 op=gu_out at token 4" in text, text
        assert "GEMM 内核本身" in text, text
        # Only mlpin/guq/guout were written: the verdict must say the pairing is
        # incomplete instead of letting a missing row pass as "compared".
        assert "区分不完整" in text, text

        # 2) the quantization itself differs from token 4 on -> that row wins.
        for cpbal, (q, s) in ((0, (q_same, s_same)), (1, (q_other, s_other))):
            save("guq", cpbal, qin(q, s))
        text = run()
        assert "[act] FIRST DIVERGENCE (act): layer 0 op=gu_q at token 4" in text, text
        assert "量化这一步" in text, text

        # 3) a dispatch difference (one layout fused, the other quantizing inside
        #    apply) must be called out, not silently averaged into the table.
        for cpbal, (q, s) in ((0, (q_same, s_same)), (1, (q_other, s_other))):
            payload = qin(q, s)
            payload["fused"] = cpbal == 1
            payload["rows"] = 8
            save("guq", cpbal, payload)
        text = run()
        assert "[act] PROVENANCE" in text and "fused=False(B) vs True(C)" in text, text

        # 4) a payload whose width differs between layouts (here: the scale is
        #    present on one side only) must be reported, never crash numpy and
        #    never be read as "identical".
        for cpbal, payload in ((0, qin(q_same, s_same)), (1, {"kind": "qin", "op": "gu_q", "q": q_same.clone()})):
            save("guq", cpbal, payload)
        for kind in ("mlpin", "guout"):
            for cpbal in (0, 1):
                (out / f"{kind}_cpbal{cpbal}_layer0_rank0_pid100_{1000 + cpbal}.pt").unlink(missing_ok=True)
        buffer, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(errors):
            rc = checker.check_act(
                argparse.Namespace(dir=str(out), summary_only=True, block_size=4)
            )
        text, errs = buffer.getvalue(), errors.getvalue()
        assert "payload width differs" in errs, errs
        assert rc == 2, f"rc={rc}\n{text}{errs}"
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_parse_dump_spec_accepts_the_round_spec() -> None:
    """The shipped DUMP spec parser must accept exactly the probe round's spec.

    ``_parse_dump_spec`` lives in ``vllm_ascend``, which cannot be imported without
    vLLM/DSA, so the function is extracted from the source and executed here with a
    stub logger.  A typo or an unregistered kind in ``DUMP_SPEC`` costs a full round
    (no dump files appear, and the round only fails at the very end), and nothing
    else on the CPU side would notice.
    """
    import ast
    from typing import Any as _Any

    path = HERE.parent.parent / "vllm_ascend/attention/sfa_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    func = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_parse_dump_spec"),
        None,
    )
    assert func is not None, "sfa_v1._parse_dump_spec is gone; the dump spec story changed"
    kinds = next(
        (
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", None) == "_ZIGZAG_DUMP_KINDS"
        ),
        None,
    )
    assert kinds, "sfa_v1._ZIGZAG_DUMP_KINDS is gone"

    class _StubLogger:
        def warning(self, *args, **kwargs):  # noqa: ANN002, ANN003 - stub
            pass

    namespace: dict = {
        "Any": _Any,
        "logger": _StubLogger(),
        "_ZIGZAG_DUMP_KINDS": kinds,
        "_ZIGZAG_ALL_LAYERS": set(range(1 << 20)),
    }
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    parse = namespace["_parse_dump_spec"]

    assert parse("act:0,1,mlp:0,1,qin:0,topk:0,kv:0,1") == {
        "act": {0, 1},
        "mlp": {0, 1},
        "qin": {0},
        "topk": {0},
        "kv": {0, 1},
    }
    # Bare numbers continue the previous kind; an unknown kind is dropped (not crash).
    assert parse("kv:0,6") == {"kv": {0, 6}}
    assert parse("nope:0") == {}
    assert parse("") == {}
    assert parse("qin:all") == {"qin": set(range(1 << 20))}


def test_load_prompts_file() -> None:
    out = Path(_temp_dir("cp_ab_pf_"))
    try:
        jsonl = out / "prompts.jsonl"
        jsonl.write_text(
            "\n".join(
                [
                    '{"case": "t", "prompt": "hello"}',
                    '{"prompt": [1, 2, 3]}',
                    '{"case": "batch", "prompt": [[1, 2], [3, 4]]}',
                    '{"case": "batch", "prompt": [5, 6]}',
                ]
            )
            + "\n"
        )
        cases = dict(driver.load_prompts_file(str(jsonl)))
        assert cases["t"] == ["hello"]
        assert cases["line2"] == [[1, 2, 3]]
        assert cases["batch"] == [[1, 2], [3, 4], [5, 6]]

        json_array = out / "prompts.json"
        json_array.write_text(json.dumps([{"case": "a", "prompt": "x"}]))
        assert driver.load_prompts_file(str(json_array)) == [("a", ["x"])]
    finally:
        shutil.rmtree(out, ignore_errors=True)


_BASH_USABLE: bool | None = None


def _bash_usable() -> bool:
    """True only when bash can really run commands here (cached)."""
    global _BASH_USABLE
    if _BASH_USABLE is not None:
        return _BASH_USABLE
    if shutil.which("bash") is None:
        _BASH_USABLE = False
        return False
    try:
        probe = subprocess.run(["bash", "-c", "exit 0"], capture_output=True, timeout=20)
        _BASH_USABLE = probe.returncode == 0
    except (OSError, subprocess.SubprocessError):
        # Sandboxed dev hosts can block subprocess/named-pipe creation.
        _BASH_USABLE = False
    return _BASH_USABLE


def test_compare_case_with_subset_of_configs() -> None:
    """A zigzag miss stops after C; B/C-only comparison must still work."""

    def entry(n_prompt: int) -> dict:
        tok_lp = [None] + [-1.0] * (n_prompt - 1) + [-1.0]
        top1 = [None] + ["token_id:10"] * (n_prompt - 1) + ["token_id:11"]
        return {
            "prompt_token_ids": list(range(n_prompt)),
            "n_prompt": n_prompt,
            "n_out": n_prompt + 1,
            "tok_lp": tok_lp,
            "top1": top1,
            "top1_lp": tok_lp,
            "top5": [[] for _ in range(n_prompt + 1)],
        }

    results = {"B": {"c": [entry(16)]}, "C": {"c": [entry(16)]}}
    args = driver.parse_args(["--out", "/tmp/cp_ab_subset", "--no-plot", "--no-annotate-plan"])
    out = _temp_dir("cp_ab_subset_")
    try:
        summary = driver.compare_case(results, "c", ["B", "C"], args, {}, out)
        metrics = summary["metrics"][0]
        assert "p99|d|C-B" in metrics
        assert "first_div_C-B" in metrics
        assert "top1%C~B" in metrics
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_verify_config_checks_additional_config() -> None:
    out = _temp_dir("cp_ab_cfg_")
    try:
        strict = driver.parse_args(["--out", str(out), "--config-check", "strict"])
        log = out / "server_C.log"
        flags = driver.expected_fingerprint("C", strict)
        flag_line = "[cp-ab] " + " ".join(f"{key}={value}" for key, value in flags.items()) + "\n"

        def verify() -> tuple[object, str]:
            """Same output capture as test_fingerprint_verification (see there)."""
            buffer = io.StringIO()
            try:
                with redirect_stdout(buffer):
                    result = driver.verify_config("C", strict, log)
            except Exception as exc:  # noqa: BLE001
                return exc, buffer.getvalue()
            return result, buffer.getvalue()

        cfg_ok = json.dumps(driver.expected_additional_config("C", strict), ensure_ascii=False)
        log.write_text(flag_line + f"[cp-ab-cfg] {cfg_ok}\n")
        result, text = verify()
        assert result is True, (result, text)

        log.write_text(flag_line + '[cp-ab-cfg] {"enable_dsa_cp": false}\n')
        result, text = verify()
        assert isinstance(result, RuntimeError) and "additional_config mismatch" in str(result), (result, text)

        # Old launchers without the cfg line stay compatible: flags only (and the
        # compatibility warning is asserted here instead of leaking to the log).
        log.write_text(flag_line)
        result, text = verify()
        assert result is True, (result, text)
        assert "did not log [cp-ab-cfg]" in text, text
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_base_additional_config_has_no_pd_only_knobs() -> None:
    assert "recompute_scheduler_enable" not in driver.BASE_ADDITIONAL_CONFIG
    args = driver.parse_args(
        ["--out", "/tmp/cp_ab_cfg_extra", "--extra-additional-config", '{"recompute_scheduler_enable": true}']
    )
    merged = json.loads(driver.resolved_config("C", args)["VLLM_ASCEND_ADDITIONAL_CONFIG"])
    assert merged["recompute_scheduler_enable"] is True
    assert merged["enable_dsa_cp"] is True


def test_build_pairs_for_subsets() -> None:
    assert driver.build_pairs(["A", "B", "C"]) == [("B", "A"), ("C", "B"), ("C", "A")]
    assert driver.build_pairs(["B", "C"]) == [("C", "B")]
    assert driver.build_pairs(["B", "C", "B2"]) == [("C", "B"), ("B2", "B")]
    assert driver.build_pairs(["A", "B", "C", "A2"]) == [
        ("B", "A"),
        ("C", "B"),
        ("C", "A"),
        ("A2", "A"),
    ]


def test_end_to_end_with_b_c_configs() -> None:
    """Baseline = B: --configs B,C --repeat-a uses B2 as the noise floor."""
    out = _temp_dir("cp_ab_bc_")
    servers, urls = mock.start_mock_servers({"B": 1e-6, "C": 0.2})
    try:
        prompts = out / "prompts.jsonl"
        prompts.write_text(json.dumps({"case": "ids", "prompt": [11, 22, 33]}) + "\n")
        results_dir = out / "results"
        argv = [
            "--out",
            str(results_dir),
            "--urls",
            f"B={urls['B']},C={urls['C']},B2={urls['B']}",
            "--configs",
            "B,C",
            "--repeat-a",
            "--prompts-file",
            str(prompts),
            "--no-plot",
            "--topk",
            "3",
        ]
        captured = io.StringIO()
        with redirect_stdout(captured):
            rc = driver.run(driver.parse_args(argv))
        assert rc == 0, captured.getvalue()
        summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["noise"]["p99"] == 0.0
        metrics = summary["cases"][0]["metrics"][0]
        assert "p99|d|C-B" in metrics
        assert "top1%C~B" in metrics
        assert abs(metrics["p99|d|C-B"] - 0.2) < 1e-3
        assert "top1%B~A" not in metrics
    finally:
        mock.stop_mock_servers(servers)
        shutil.rmtree(out, ignore_errors=True)


def test_zigzag_state_from_log() -> None:
    out = _temp_dir("cp_ab_zigzag_log_")
    try:
        log = out / "server_C.log"
        log.write_text("no marker here\n")
        assert driver.zigzag_state_from_log(log) == {"metadata": False, "forward": False}
        log.write_text(
            "[CP_BALANCE] metadata zigzag=1 rank=0 cp_size=8 num_tokens_pad=2064\n"
            "[CP_BALANCE] forward zigzag_active=1 num_tokens=2064 local_tokens=258\n"
        )
        assert driver.zigzag_state_from_log(log) == {"metadata": True, "forward": True}
        missing = driver.zigzag_state_from_log(out / "missing.log")
        assert missing == {"metadata": False, "forward": False}
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_preflight_logic_with_stubbed_launcher() -> None:
    """--preflight compares the launcher fingerprint against the expected config."""

    def make_dry_run(flip: bool):
        def _dry_run(args, name, timeout=60.0):
            fingerprint = dict(driver.expected_fingerprint(name, args))
            if flip:
                fingerprint["CP_BALANCE"] = 1 - fingerprint["CP_BALANCE"]
            line = "[cp-ab] " + " ".join(f"{key}={value}" for key, value in fingerprint.items())
            return 0, line + "\n[cp-ab][dry-run] OK\n", ""

        return _dry_run

    out = _temp_dir("cp_ab_preflight_logic_")
    try:
        argv = ["--out", str(out / "results"), "--preflight"]
        original = driver._launcher_dry_run
        try:
            driver._launcher_dry_run = make_dry_run(flip=False)
            captured = io.StringIO()
            with redirect_stdout(captured):
                rc = driver.run(driver.parse_args(argv))
            assert rc == 0, captured.getvalue()

            driver._launcher_dry_run = make_dry_run(flip=True)
            captured = io.StringIO()
            with redirect_stdout(captured):
                rc = driver.run(driver.parse_args(argv))
            assert rc == 1, captured.getvalue()
        finally:
            driver._launcher_dry_run = original
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_preflight_with_fake_launcher() -> None:
    if not _bash_usable():
        print("[skip] bash not usable on this host")
        return
    import shlex

    out = _temp_dir("cp_ab_preflight_")
    try:
        launcher = out / "fake_launcher.sh"
        launcher.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "dsa=1",
                    "if printf '%s' \"$VLLM_ASCEND_ADDITIONAL_CONFIG\" | "
                    "grep -q '\"enable_dsa_cp\": false'; then dsa=0; fi",
                    "spec=0",
                    "if [ -n \"$VLLM_ASCEND_SPEC_CONFIG\" ]; then spec=1; fi",
                    "kv=0",
                    "if [ -n \"$VLLM_ASCEND_KV_TRANSFER_CONFIG\" ]; then kv=1; fi",
                    # Same "[cp-ab]" prefix as the fingerprint, printed first:
                    # the regression that made preflight report actual={}.
                    "echo \"[cp-ab] vendor env: /tmp/fake_vendor/set_env.bash\"",
                    "echo \"[cp-ab] CP_BALANCE=${VLLM_ASCEND_CP_BALANCE} DSA_CP=${dsa} "
                    "MIN_TOKENS=${VLLM_ASCEND_CP_BALANCE_MIN_TOKENS} "
                    "EMBED_LOCAL=${VLLM_ASCEND_CP_BALANCE_EMBED_LOCAL} SPEC=${spec} KV=${kv}\"",
                    "echo \"[cp-ab-cfg] ${VLLM_ASCEND_ADDITIONAL_CONFIG}\"",
                    "if [ -n \"${DRY_RUN:-}\" ]; then echo '[cp-ab][dry-run] OK'; exit 0; fi",
                    "echo 'should not reach here' >&2; exit 9",
                ]
            )
            + "\n"
        )
        args = driver.parse_args(
            [
                "--out",
                str(out / "results"),
                "--launcher",
                f"bash {shlex.quote(launcher.as_posix())} {{port}}",
                "--preflight",
            ]
        )
        captured = io.StringIO()
        with redirect_stdout(captured):
            rc = driver.run(args)
        text = captured.getvalue()
        assert rc == 0, text
        assert text.count("[preflight]") >= 3, text
        assert "FAILED" not in text, text
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_shipped_launchers_print_complete_fingerprint() -> None:
    """The real launchers' fingerprint line must match the driver's expectation.

    ``--config-check strict`` compares the expected dict key by key, so a
    launcher that forgot ``MIN_TOKENS`` would abort every run.  This runs the
    shipped scripts with ``DRY_RUN=1`` (no model load) and checks the
    ``[cp-ab]`` line against ``expected_fingerprint`` for every config.
    """
    def fingerprint_line(launcher: str) -> str:
        """The literal echo line the launcher prints (kept in sync by hand)."""
        text = (HERE / launcher).read_text(encoding="utf-8")
        return next(line for line in text.splitlines() if f"{driver.FINGERPRINT_PREFIX} CP_BALANCE=" in line)

    # 1) The line itself must carry every field the driver compares, otherwise
    #    `verify_config` reports `diff {'MIN_TOKENS': (2048, None)}` no matter
    #    what env the launcher receives.
    args = driver.parse_args(["--no-kv-connector"])
    for launcher in ("launcher_template.sh", "launcher_glm52_w4a4c8_mxfp4.sh"):
        line = fingerprint_line(launcher)
        for field in sorted(driver.expected_fingerprint("C", args)):
            assert f"{field}=" in line, f"{launcher}: {field} missing from the fingerprint line: {line}"
        # Every `KEY=` token must be one the driver actually compares.
        printed = {token.partition("=")[0] for token in line.split() if token.partition("=")[1]}
        assert printed >= set(driver.expected_fingerprint("C", args)), (
            f"{launcher}: fingerprint prints {sorted(printed)}, driver expects "
            f"{sorted(driver.expected_fingerprint('C', args))}"
        )

    if not _bash_usable():
        print("[skip] bash not usable on this host")
        return
    import os
    import shlex
    import subprocess

    launchers = ["launcher_template.sh", "launcher_glm52_w4a4c8_mxfp4.sh"]
    out = _temp_dir("cp_ab_launcher_")
    try:
        # Every knob is set explicitly so the check does not depend on the
        # launcher's own defaults.
        overrides = _launcher_dry_run_env(out)
        for launcher in launchers:
            for config in ("B", "C", "B2"):
                args = driver.parse_args(["--no-kv-connector"])
                env = os.environ.copy()
                env.update(driver.config_env(config, args))
                env.update(overrides)
                env["DRY_RUN"] = "1"
                cmd = f"bash {shlex.quote((HERE / launcher).as_posix())} {args.base_port}"
                try:
                    proc = subprocess.run(
                        ["bash", "-lc", cmd],
                        cwd=str(HERE),
                        env=env,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=120,
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    print(f"[skip] {launcher}: cannot execute bash ({exc})")
                    return
                text = proc.stdout + proc.stderr
                if "vllm not found in PATH" in text:
                    # Windows/msys cannot give the fake `vllm` an exec bit; the
                    # fingerprint-format half of this check already ran above.
                    print("[skip] fake vllm is not executable on this host")
                    return
                assert proc.returncode == 0, f"{launcher} {config}: rc={proc.returncode}\n{text}"
                line = driver.find_fingerprint(text)
                assert line is not None, f"{launcher} {config}: no fingerprint line\n{text}"
                assert driver.parse_fingerprint(line) == driver.expected_fingerprint(config, args), (
                    f"{launcher} {config}: fingerprint mismatch\n  got      {driver.parse_fingerprint(line)}\n"
                    f"  expected {driver.expected_fingerprint(config, args)}"
                )
    finally:
        shutil.rmtree(out, ignore_errors=True)


def _launcher_dry_run_env(out: Path) -> dict[str, str]:
    """A model dir plus a fake ``vllm`` on PATH, so a shipped launcher's DRY_RUN passes.

    Every knob is set explicitly so the check never depends on the launcher's
    own defaults (and no site path or vendor env is needed).
    """
    import os

    fake_model = out / "model"
    fake_model.mkdir(exist_ok=True)
    fake_bin = out / "bin"
    fake_bin.mkdir(exist_ok=True)
    fake_vllm = fake_bin / "vllm"
    fake_vllm.write_text("#!/usr/bin/env bash\nexit 0\n")
    os.chmod(fake_vllm, 0o755)
    return {
        "MODEL_PATH": str(fake_model),
        # The site launcher's DRY_RUN validates the repo path too; point it at an
        # existing dir so the check is host independent.
        "VLLM_ASCEND_REPO": str(out),
        "PRE_LAUNCH_SCRIPT": "",
        "VENDOR_SET_ENV": "",
        "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
    }


def test_launcher_skip_source_keeps_fingerprint_complete() -> None:
    """CP_AB_SKIP_SOURCE=1 (caller already sourced the site env) must not break the gate.

    Skipping the two sources only removes work; the fingerprint line the driver
    compares must still be printed, otherwise every round would fail
    ``--config-check strict`` on a machine that uses the fast path.
    """
    if not _bash_usable():
        print("[skip] bash not usable on this host")
        return
    import os
    import shlex
    import subprocess

    out = _temp_dir("cp_ab_skip_source_")
    try:
        args = driver.parse_args(["--no-kv-connector"])
        env = os.environ.copy()
        env.update(driver.config_env("C", args))
        env.update(_launcher_dry_run_env(out))
        env["DRY_RUN"] = "1"
        env["CP_AB_SKIP_SOURCE"] = "1"
        cmd = f"bash {shlex.quote((HERE / 'launcher_glm52_w4a4c8_mxfp4.sh').as_posix())} {args.base_port}"
        proc = subprocess.run(
            ["bash", "-lc", cmd],
            cwd=str(HERE),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        text = proc.stdout + proc.stderr
        if "vllm not found in PATH" in text:
            print("[skip] fake vllm is not executable on this host")
            return
        assert proc.returncode == 0, f"rc={proc.returncode}\n{text}"
        assert "CP_AB_SKIP_SOURCE=1" in text, text
        line = driver.find_fingerprint(text)
        assert line is not None, text
        assert driver.parse_fingerprint(line) == driver.expected_fingerprint("C", args), text
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_verify_response_rejects_uncomparable_replies() -> None:
    """A 200 response is not enough: the reply must be position-comparable."""
    prompt = [11, 22, 33, 44]
    choice = mock.make_choice(0, prompt, top_k=3, offset=0.0, echo=True)
    good = driver.parse_choice(choice, {"choices": [choice]}, 3)
    driver.verify_response(good, prompt)  # must not raise

    # Same length, different token at position 3: every position still has a
    # logprob, so only the sent-vs-echoed id check can catch it.
    swapped_choice = mock.make_choice(0, prompt, top_k=3, offset=0.0, echo=True, echo_ids=[11, 22, 33, 99])
    entry = driver.parse_choice(swapped_choice, {"choices": [swapped_choice]}, 3)
    assert entry["prompt_token_ids"] == [11, 22, 33, 99], entry["prompt_token_ids"]
    try:
        driver.verify_response(entry, prompt)
    except RuntimeError as exc:
        assert "different token ids" in str(exc), exc
    else:
        raise AssertionError("a same-length token rewrite must be rejected")

    # The server echoed one token fewer than was sent.
    truncated_choice = mock.make_choice(0, prompt, top_k=3, offset=0.0, echo=True, echo_ids=[11, 22, 33])
    entry = driver.parse_choice(truncated_choice, {"choices": [truncated_choice]}, 3)
    try:
        driver.verify_response(entry, prompt)
    except RuntimeError as exc:
        assert "echoed" in str(exc), exc
    else:
        raise AssertionError("a truncated echo must be rejected")

    # The server prepended BOS even though add_special_tokens=False.
    rewritten = mock.make_choice(0, prompt, top_k=3, offset=0.0, echo=True, break_mode="rewrite")
    entry = driver.parse_choice(rewritten, {"choices": [rewritten]}, 3)
    try:
        driver.verify_response(entry, prompt)
    except RuntimeError as exc:
        assert "echoed" in str(exc), exc
    else:
        raise AssertionError("a rewritten prompt must be rejected")

    # A server that returns no logprobs at all.
    silent = mock.make_choice(0, prompt, top_k=3, offset=0.0, echo=True, break_mode="no_logprobs")
    entry = driver.parse_choice(silent, {"choices": [silent]}, 3)
    try:
        driver.verify_response(entry, prompt)
    except RuntimeError as exc:
        assert "logprob" in str(exc), exc
    else:
        raise AssertionError("missing logprobs must be rejected")

    # Text prompts have no local token ids: the check trusts the response.
    text_choice = mock.make_choice(0, "hello world", top_k=3, offset=0.0, echo=True)
    entry = driver.parse_choice(text_choice, {"choices": [text_choice]}, 3)
    driver.verify_response(entry, None)


def test_verify_response_used_by_query_batch() -> None:
    """query_batch must fail (and retry) instead of returning bad data."""
    servers, urls = mock.start_mock_servers({"bad": 0.0})
    try:
        servers[0].break_mode = "truncate"
        args = driver.parse_args(["--topk", "3", "--http-retries", "1", "--http-retry-backoff", "0"])
        try:
            driver.query_batch(urls["bad"], args, [[11, 22, 33]])
        except RuntimeError as exc:
            assert "request failed after 1 tries" in str(exc), exc
            assert "echoed" in str(exc), exc
        else:
            raise AssertionError("query_batch must reject a truncated echo")
    finally:
        mock.stop_mock_servers(servers)


def test_batch_prompt_echoes_per_choice() -> None:
    """A batch request must echo each choice's own tokens, not the flattened list."""
    servers, urls = mock.start_mock_servers({"ok": 0.0})
    try:
        args = driver.parse_args(["--topk", "3", "--http-retries", "1", "--http-retry-backoff", "0"])
        prompts = [[1, 2, 3, 4], [9, 10, 11, 12]]
        entries = driver.query_batch(urls["ok"], args, prompts)
        assert [entry["prompt_token_ids"] for entry in entries] == prompts
        assert [entry["n_prompt"] for entry in entries] == [4, 4]
    finally:
        mock.stop_mock_servers(servers)


def test_report_token_budget_warns_below_min_tokens() -> None:
    entries = [{"n_prompt": 8}, {"n_prompt": 9}]
    captured = io.StringIO()
    with redirect_stdout(captured):
        driver.report_token_budget("short", entries, min_tokens=2048)
    text = captured.getvalue()
    assert "MIN_TOKENS=2048" in text, text
    assert "zigzag stays off" in text, text

    captured = io.StringIO()
    with redirect_stdout(captured):
        driver.report_token_budget("long", [{"n_prompt": 4096}], min_tokens=2048)
    assert captured.getvalue() == ""


def test_parse_choice_text_keyed_top_logprobs() -> None:
    """Real vLLM defaults to decoded-text keys; the driver must cope with them.

    This is the shape that used to crash the driver with
    ``invalid literal for int() with base 10: ',arr'``.
    """
    entry = {
        "prompt_token_ids": [11, 22, 33],
        "logprobs": {
            "token_logprobs": [None, -0.5, -0.6, -0.7],
            "tokens": ["", "a", ",", "b"],
            "top_logprobs": [
                None,
                {"a": -0.5, ",arr": -1.2, "Redux": -2.0},
                {",": -0.6, "opoulos": -1.5, "and": -2.1},
                {"b": -0.7, "c": -1.7},
            ],
        },
    }
    parsed = driver.parse_choice(entry, {"choices": [entry]}, 3)
    assert parsed["n_prompt"] == 3
    assert parsed["tok_lp"] == [None, -0.5, -0.6, -0.7]
    # Text keys are kept as opaque identities: no int() conversion anywhere.
    assert parsed["top1"][1] == "tok:a"
    assert parsed["top1"][2] == "tok:,"
    assert parsed["top5"][2][0] == "tok:,"
    assert "tok:Redux" in parsed["top5"][1]
    assert parsed["top1"][3] == "tok:b"

    # A config pair with identical text-keyed replies must agree 100%.
    metrics = driver._pair_metrics([("C", "B")], {"B": driver.as_float(parsed["tok_lp"]),
                                                  "C": driver.as_float(parsed["tok_lp"])},
                                   {"B": driver.as_token(parsed["top1"]), "C": driver.as_token(parsed["top1"])},
                                   {"B": parsed["top5"], "C": parsed["top5"]},
                                   0.05, 8, 3, [])
    assert metrics["top1%C~B"] == 100.0

    # Different text tokens at one position -> disagreement there.
    other = driver.parse_choice({**entry, "logprobs": {**entry["logprobs"],
                              "top_logprobs": [None, {"a": -0.5}, {"zzz": -0.6}, {"b": -0.7}]}},
                                {"choices": [entry]}, 3)
    metrics = driver._pair_metrics([("C", "B")], {"B": driver.as_float(parsed["tok_lp"]),
                                                  "C": driver.as_float(parsed["tok_lp"])},
                                   {"B": driver.as_token(other["top1"]), "C": driver.as_token(parsed["top1"])},
                                   {"B": other["top5"], "C": parsed["top5"]},
                                   0.05, 8, 3, [])
    assert 0.0 < metrics["top1%C~B"] < 100.0


def test_prompt_logprobs_fallback_text_keys() -> None:
    """prompt_logprobs fallback must not crash on either key format."""
    ids = [11, 22, 33]
    # Numeric-string keys (ids/placeholders): the true token is found.
    choice = {
        "prompt_token_ids": ids,
        "prompt_logprobs": [None, {"22": -0.5}, {"33": -0.6}],
    }
    parsed = driver.parse_choice(choice, {"choices": [choice]}, 3)
    assert parsed["tok_lp"] == [None, -0.5, -0.6]

    # Decoded-text keys that do NOT contain the prompt token: the old code
    # silently left tok_lp[i] = None (looked like lost alignment), the new code
    # says so explicitly.
    choice = {
        "prompt_token_ids": ids,
        "prompt_logprobs": [None, {"a": -0.5}, {"b": -0.6}],
    }
    try:
        driver.parse_choice(choice, {"choices": [choice]}, 3)
    except RuntimeError as exc:
        assert "keyed by decoded token text" in str(exc), exc
    else:
        raise AssertionError("text-keyed prompt_logprobs must fail loudly")


def test_configs_are_cp_balance_only() -> None:
    """Only B/C (cp_balance off/on) are selectable; A is gone."""
    supported = driver.parse_args(["--configs", "B,C"])
    assert supported.configs == "B,C"

    try:
        driver.run(driver.parse_args(["--out", "/tmp/cp_ab_a", "--configs", "A,B,C"]))
    except SystemExit as exc:
        assert "not supported" in str(exc)
    else:
        raise AssertionError("--configs A,B,C must be rejected")

    try:
        driver.run(driver.parse_args(["--out", "/tmp/cp_ab_x", "--configs", "B,X"]))
    except SystemExit as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("--configs B,X must be rejected")


def test_end_to_end_against_mock() -> None:
    out = Path(_temp_dir("cp_ab_e2e_"))
    # B and B2 are separate servers with the same offset, so the noise floor is
    # a real cross-run floor instead of a trivial zero.
    servers, urls = mock.start_mock_servers({"B": 1e-6, "C": 0.2, "B2": 1e-6})
    try:
        text = "hello ascend cp balance"
        prompts = out / "prompts.jsonl"
        prompts.write_text(
            "\n".join(
                [
                    json.dumps({"case": "text", "prompt": text}),
                    json.dumps({"case": "ids", "prompt": [11, 22, 33, 44, 55, 66, 77, 88]}),
                    json.dumps(
                        {
                            "case": "batch",
                            "prompt": [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
                        }
                    ),
                ]
            )
            + "\n"
        )
        results_dir = out / "results"
        argv = [
            "--out",
            str(results_dir),
            "--urls",
            f"B={urls['B']},C={urls['C']},B2={urls['B2']}",
            "--prompts-file",
            str(prompts),
            "--repeat-a",
            "--no-plot",
            "--topk",
            "3",
        ]
        args = driver.parse_args(argv)
        captured = io.StringIO()
        with redirect_stdout(captured):
            rc = driver.run(args)
        assert rc == 0
        printed = captured.getvalue()
        assert "first_div C-B" in printed

        summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
        metrics_by_case = {case["case"]: case["metrics"] for case in summary["cases"]}
        assert summary["noise"]["p99"] == 0.0
        assert summary["cases"][0]["metrics"][0]["length"] == len(mock.fake_tokenize(text))

        text_metrics = metrics_by_case["text"][0]
        assert text_metrics["first_div_C-B"] == 1
        assert abs(text_metrics["p99|d|C-B"] - 0.2) < 1e-3
        assert text_metrics["top1%C~B"] == 100.0

        assert len(metrics_by_case["batch"]) == 2
        assert metrics_by_case["ids"][0]["top1%C~B"] == 100.0

        all_results = json.loads((results_dir / "results_all.json").read_text(encoding="utf-8"))
        reference = all_results["B"]["text"][0]
        assert reference["prompt_token_ids"] == mock.fake_tokenize(text)
        assert reference["n_prompt"] == len(mock.fake_tokenize(text))
        assert all_results["C"]["text"][0]["prompt_token_ids"] == reference["prompt_token_ids"]
    finally:
        mock.stop_mock_servers(servers)
        shutil.rmtree(out, ignore_errors=True)


def main() -> int:
    """Run every ``test_*`` in this module; print per-test time and the slowest.

    The CPU self test is the only regression gate on a box without an NPU, so it
    has to stay cheap enough to run after every change; the timing line makes a
    newly expensive test visible instead of felt as "the selftest is slow".
    """
    import time

    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    timings: list[tuple[float, str]] = []
    started = time.perf_counter()
    for test in tests:
        begin = time.perf_counter()
        test()
        elapsed = time.perf_counter() - begin
        timings.append((elapsed, test.__name__))
        print(f"[ok] {test.__name__}")
    total = time.perf_counter() - started
    slowest = ", ".join(f"{name}={secs:.1f}s" for secs, name in sorted(timings, reverse=True)[:5])
    print(f"[time] {len(tests)} tests in {total:.1f}s; slowest: {slowest}")
    print("SELFTEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

