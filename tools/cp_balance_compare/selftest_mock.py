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

