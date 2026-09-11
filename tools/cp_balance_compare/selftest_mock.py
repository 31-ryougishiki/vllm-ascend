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
from contextlib import redirect_stdout
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


def _expect_runtime_error(func) -> None:
    try:
        func()
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


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
        assert driver.verify_config("C", strict, log)

        log.write_text("[cp-ab] CP_BALANCE=0 DSA_CP=1 EMBED_LOCAL=0 SPEC=0 KV=0\n")
        _expect_runtime_error(lambda: driver.verify_config("C", strict, log))
        assert driver.verify_config("C", warn, log) is False

        log.write_text("no fingerprint in this log\n")
        _expect_runtime_error(lambda: driver.verify_config("C", strict, log))
        assert driver.verify_config("C", warn, log) is False
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

        cfg_ok = json.dumps(driver.expected_additional_config("C", strict), ensure_ascii=False)
        log.write_text(flag_line + f"[cp-ab-cfg] {cfg_ok}\n")
        assert driver.verify_config("C", strict, log)

        log.write_text(flag_line + '[cp-ab-cfg] {"enable_dsa_cp": false}\n')
        _expect_runtime_error(lambda: driver.verify_config("C", strict, log))

        # Old launchers without the cfg line stay compatible: flags only.
        log.write_text(flag_line)
        assert driver.verify_config("C", strict, log)
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
        fake_model = out / "model"
        fake_model.mkdir()
        fake_bin = out / "bin"
        fake_bin.mkdir()
        fake_vllm = fake_bin / "vllm"
        fake_vllm.write_text("#!/usr/bin/env bash\nexit 0\n")
        os.chmod(fake_vllm, 0o755)

        # Every knob is set explicitly so the check does not depend on the
        # launcher's own defaults.
        overrides = {
            "MODEL_PATH": str(fake_model),
            "PRE_LAUNCH_SCRIPT": "",
            "VENDOR_SET_ENV": "",
            "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
            "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
        }
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
                        timeout=120,
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    print(f"[skip] {launcher}: cannot execute bash ({exc})")
                    return
                text = proc.stdout + proc.stderr
                assert proc.returncode == 0, f"{launcher} {config}: rc={proc.returncode}\n{text}"
                line = driver.find_fingerprint(text)
                assert line is not None, f"{launcher} {config}: no fingerprint line\n{text}"
                assert driver.parse_fingerprint(line) == driver.expected_fingerprint(config, args), (
                    f"{launcher} {config}: fingerprint mismatch\n  got      {driver.parse_fingerprint(line)}\n"
                    f"  expected {driver.expected_fingerprint(config, args)}"
                )
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
        args = driver.parse_args(["--topk", "3", "--http-retries", "1"])
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
        args = driver.parse_args(["--topk", "3", "--http-retries", "1"])
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
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"[ok] {test.__name__}")
    print("SELFTEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

