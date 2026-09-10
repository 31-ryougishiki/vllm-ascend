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
"""Tiny mock of the vLLM ``/v1/completions`` endpoint.

Used by ``selftest_mock.py`` to validate ``ab_cp_compare.py`` without an NPU,
and usable manually to exercise the driver end to end::

    python tools/cp_balance_compare/mock_vllm_server.py --ports 18001,18002,18003 \\
        --offsets 0,1e-6,0.2

Every served response is deterministic: the logprob of a token is
``-0.5 - 0.001 * (token % 100) + offset`` so different ports behave like
different configurations.  Both text prompts (tokenised with a stable fake
tokenizer) and token-id prompts are accepted, and ``prompt_token_ids`` is
always returned, mirroring vLLM.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

FAKE_VOCAB = 1000


def fake_tokenize(text: str) -> list[int]:
    return [1 + (ord(char) % (FAKE_VOCAB - 1)) for char in text]


def normalize_prompts(prompt: Any) -> list[Any]:
    if isinstance(prompt, str):
        return [prompt]
    if isinstance(prompt, list):
        if not prompt:
            return []
        if all(isinstance(item, str) for item in prompt):
            return list(prompt)
        if all(isinstance(item, int) for item in prompt):
            return [prompt]
        if all(isinstance(item, list) for item in prompt):
            return list(prompt)
    raise ValueError(f"unsupported prompt payload: {type(prompt)}")


def ids_for(prompt: Any) -> list[int]:
    if isinstance(prompt, str):
        return fake_tokenize(prompt)
    return [int(token) for token in prompt]


def make_choice(index: int, prompt: Any, top_k: int, offset: float, echo: bool) -> dict[str, Any]:
    ids = ids_for(prompt)
    num_prompt = len(ids)
    token_logprobs: list[float | None] = [None]
    top_logprobs: list[dict[str, float] | None] = [None]
    prompt_logprobs: list[dict[str, float] | None] = [None]
    for position in range(1, num_prompt + 1):
        token = ids[position] if position < num_prompt else (ids[-1] + 1 if ids else 1)
        logprob = -0.5 - 0.001 * (token % 100) + offset
        candidates = {str(token + delta): logprob - 0.1 * delta for delta in range(top_k)}
        token_logprobs.append(logprob)
        top_logprobs.append(candidates)
        if position < num_prompt:
            prompt_logprobs.append(candidates)
    choice: dict[str, Any] = {
        "index": index,
        "text": "",
        "finish_reason": "length",
        "prompt_token_ids": ids,
        "prompt_logprobs": prompt_logprobs,
    }
    if echo:
        choice["logprobs"] = {
            "token_logprobs": token_logprobs,
            "tokens": [],
            "top_logprobs": top_logprobs,
            "text_offset": [],
        }
    return choice


class _Handler(BaseHTTPRequestHandler):
    server_version = "cp-ab-mock"

    def log_message(self, *args):  # silence request logging
        return

    def _send(self, code: int, payload: Any = None) -> None:
        body = b"" if payload is None else json.dumps(payload).encode()
        self.send_response(code)
        if payload is not None:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - http.server API
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        else:
            self._send(404)

    def do_POST(self):  # noqa: N802 - http.server API
        if self.path != "/v1/completions":
            self._send(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length) or b"{}")
        prompts = normalize_prompts(request.get("prompt"))
        top_k = int(request.get("logprobs") or request.get("prompt_logprobs") or 3)
        top_k = max(1, top_k)
        offset = float(getattr(self.server, "offset", 0.0))
        echo = bool(getattr(self.server, "echo_mode", True))
        choices = [make_choice(i, prompt, top_k, offset, echo) for i, prompt in enumerate(prompts)]
        self._send(200, {"choices": choices})


def start_mock_servers(
    offsets: dict[str, float], base_port: int = 0
) -> tuple[list[ThreadingHTTPServer], dict[str, str]]:
    """Start one mock server per config name; returns (servers, name -> url).

    ``base_port=0`` binds an ephemeral port, which keeps the self test free of
    port collisions.
    """
    servers: list[ThreadingHTTPServer] = []
    urls: dict[str, str] = {}
    for index, (name, offset) in enumerate(offsets.items()):
        server = ThreadingHTTPServer(("127.0.0.1", base_port + index), _Handler)
        server.offset = offset  # type: ignore[attr-defined]
        server.echo_mode = True  # type: ignore[attr-defined]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        port = server.server_address[1]
        urls[name] = f"http://127.0.0.1:{port}"
    return servers, urls


def stop_mock_servers(servers: list[ThreadingHTTPServer]) -> None:
    for server in servers:
        server.shutdown()
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ports", default="18001,18002,18003")
    parser.add_argument("--offsets", default="0,1e-6,0.2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ports = [int(item) for item in args.ports.split(",")]
    offsets = [float(item) for item in args.offsets.split(",")]
    if len(ports) != len(offsets):
        raise SystemExit("--ports and --offsets must have the same length")
    # start_mock_servers assigns base_port + index; honour explicit ports by
    # translating them into a base only when they are contiguous.
    servers, urls = start_mock_servers(
        {f"port{port}": offset for port, offset in zip(ports, offsets)}, base_port=ports[0]
    )
    print(json.dumps(urls, indent=2))
    print("mock servers running; Ctrl-C to stop")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        stop_mock_servers(servers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
