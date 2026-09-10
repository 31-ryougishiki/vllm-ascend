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


def test_parse_choice_token_ids_prompt() -> None:
    choice = mock.make_choice(0, [11, 22, 33, 44], top_k=3, offset=0.0, echo=True)
    entry = driver.parse_choice(choice, {"choices": [choice]}, 3)
    assert entry["n_prompt"] == 4
    assert entry["n_out"] == 5
    assert entry["tok_lp"][0] is None
    assert entry["top1"][0] is None
    assert entry["top1"][1] == 22
    # last output position is the first generated token
    assert entry["top1"][4] == 45


def test_parse_choice_text_prompt_uses_response_prompt_token_ids() -> None:
    text = "hello ascend cp balance"
    choice = mock.make_choice(0, text, top_k=3, offset=0.0, echo=True)
    entry = driver.parse_choice(choice, {"choices": [choice]}, 3)
    assert entry["prompt_token_ids"] == mock.fake_tokenize(text)
    assert entry["n_prompt"] == len(mock.fake_tokenize(text))
    assert entry["n_out"] == entry["n_prompt"] + 1


def test_parse_choice_prompt_logprobs_fallback() -> None:
    choice = mock.make_choice(0, [5, 6, 7, 8], top_k=3, offset=0.0, echo=False)
    assert "logprobs" not in choice
    entry = driver.parse_choice(choice, {"choices": [choice]}, 3)
    assert entry["n_prompt"] == 4
    assert entry["n_out"] == 4
    assert entry["top1"][1] == 6
    assert entry["tok_lp"][1] is not None


def test_fingerprint_verification() -> None:
    out = Path(_temp_dir("cp_ab_fp_"))
    try:
        strict = driver.parse_args(["--out", str(out), "--config-check", "strict"])
        warn = driver.parse_args(["--out", str(out), "--config-check", "warn"])
        log = out / "server_C.log"

        expected = driver.expected_fingerprint("C", strict)
        log.write_text("[cp-ab] " + " ".join(f"{key}={value}" for key, value in expected.items()) + "\n")
        assert driver.verify_config("C", strict, log)

        log.write_text("[cp-ab] CP_BALANCE=0 DSA_CP=1 EMBED_LOCAL=0 SPEC=0 KV=0\n")
        _expect_runtime_error(lambda: driver.verify_config("C", strict, log))
        assert driver.verify_config("C", warn, log) is False

        log.write_text("no fingerprint in this log\n")
        _expect_runtime_error(lambda: driver.verify_config("C", strict, log))
        assert driver.verify_config("C", warn, log) is False
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


def _bash_usable() -> bool:
    if shutil.which("bash") is None:
        return False
    try:
        probe = subprocess.run(["bash", "-c", "exit 0"], capture_output=True, timeout=20)
        return probe.returncode == 0
    except (OSError, subprocess.SubprocessError):
        # Sandboxed dev hosts can block subprocess/named-pipe creation.
        return False


def test_compare_case_with_subset_of_configs() -> None:
    """A zigzag miss stops after C; B/C-only comparison must still work."""

    def entry(n_prompt: int) -> dict:
        tok_lp = [None] + [-1.0] * (n_prompt - 1) + [-1.0]
        top1 = [None] + [10] * (n_prompt - 1) + [11]
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
                    "echo \"[cp-ab] CP_BALANCE=${VLLM_ASCEND_CP_BALANCE} DSA_CP=${dsa} "
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


def test_end_to_end_against_mock() -> None:
    out = Path(_temp_dir("cp_ab_e2e_"))
    servers, urls = mock.start_mock_servers({"A": 0.0, "B": 1e-6, "C": 0.2})
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
            f"A={urls['A']},B={urls['B']},C={urls['C']},A2={urls['A']}",
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

        text_metrics = metrics_by_case["text"][0]
        assert text_metrics["first_div_C-B"] == 1
        assert abs(text_metrics["p99|d|C-B"] - 0.2) < 1e-3
        assert text_metrics["top1%C~B"] == 100.0

        assert len(metrics_by_case["batch"]) == 2
        assert metrics_by_case["ids"][0]["top1%C~B"] == 100.0

        all_results = json.loads((results_dir / "results_all.json").read_text(encoding="utf-8"))
        reference = all_results["A"]["text"][0]
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

