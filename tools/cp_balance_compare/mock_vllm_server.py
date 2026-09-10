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

    python tools/cp_balance_compare/mock_vllm_server.py --ports 18034,18035 \\
        --offsets 1e-6,0.2

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
    """Return one entry per completion choice, like the vLLM completions API."""
    if isinstance(prompt, str):
        return [prompt]
    if isinstance(prompt, list):
        if not prompt:
            return []
        if all(isinstance(item, str) for item in prompt):
            return list(prompt)  # batch of text prompts -> one choice each
        if all(isinstance(item, int) for item in prompt):
            return [prompt]  # a single token-id prompt
        if all(isinstance(item, list) for item in prompt):
            return [list(item) for item in prompt]  # batch of token-id prompts
    raise ValueError(f"unsupported prompt payload: {type(prompt)}")


def ids_for(prompt: Any) -> list[int]:
    if isinstance(prompt, str):
        return fake_tokenize(prompt)
    return [int(token) for token in prompt]


def make_choice(
    index: int,
    prompt: Any,
    top_k: int,
    offset: float,
    echo: bool,
    break_mode: str = "",
    echo_ids: list[int] | None = None,
    key_style: str = "text",
) -> dict[str, Any]:
    """Build one choice.

    ``key_style`` mirrors the server's ``top_logprobs`` key format:
    ``"text"`` (vLLM default: decoded token text, e.g. ``"Redux"``),
    ``"ids"`` (``return_tokens_as_token_ids=True`` -> ``token_id:N``), or
    ``"numeric"`` (plain ids).
    """
    ids = ids_for(prompt)
    if not ids:
        raise ValueError("empty prompt")
    scored = list(ids)
    if break_mode == "rewrite":
        # Simulate a server that prepends BOS even though add_special_tokens=False.
        scored = [0] + scored
    num_scored = len(scored)  # logprob array length == echoed prompt length
    prompt_ids = list(scored)
    if break_mode == "truncate":
        # Echo one id fewer than the request carried, as if the last token had
        # been dropped, while still scoring the full prompt length.
        prompt_ids = prompt_ids[:-1] or prompt_ids
    if echo_ids is not None:
        # Serve ``echo_ids`` as the echoed prompt while still scoring
        # ``num_scored`` positions: what a server that rewrote the token
        # sequence (swapped ids) looks like on the wire.
        prompt_ids = [int(token) for token in echo_ids]
        if not prompt_ids:
            raise ValueError("echo_ids must not be empty")

    def key_for(step: int, candidate: int) -> str:
        if key_style == "ids":
            return f"token_id:{candidate}"
        if key_style == "numeric":
            return str(candidate)
        return f"tok{step}_{candidate - 1}"

    token_logprobs: list[float | None] = [None]
    top_logprobs: list[dict[str, float] | None] = [None]
    prompt_logprobs: list[dict[str, float] | None] = [None]
    for position in range(1, num_scored + 1):
        # Position `position` scores the prompt token at that index; the last
        # entry stands for the first generated token (last prompt id + 1).
        token = scored[position] if position < num_scored else (scored[-1] + 1)
        logprob = -0.5 - 0.001 * (token % 100) + offset
        candidates = {
            key_for(position, token + delta): logprob - 0.1 * delta for delta in range(top_k)
        }
        token_logprobs.append(logprob)
        top_logprobs.append(candidates)
        if position < num_scored:
            prompt_logprobs.append(candidates)
    if break_mode == "no_logprobs":
        # Simulate a server that forgot to return logprobs (only echo + ids).
        token_logprobs = [None] * (num_scored + 1)
        top_logprobs = [None] * (num_scored + 1)
        prompt_logprobs = [None] * (num_scored + 1)
    choice: dict[str, Any] = {
        "index": index,
        "text": "",
        "finish_reason": "length",
        "prompt_token_ids": prompt_ids,
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
        break_mode = str(getattr(self.server, "break_mode", ""))
        key_style = str(getattr(self.server, "key_style", "text"))
        choices = [
            make_choice(index, prompt, top_k, offset, echo, break_mode, None, key_style)
            for index, prompt in enumerate(prompts)
        ]
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
        # base_port=0 means "pick an ephemeral port per server"; otherwise use
        # consecutive ports.  Do not use `base_port + index` when base_port is 0,
        # that would try to bind port 1, 2, ... on the host.
        bind_port = 0 if base_port == 0 else base_port + index
        server = ThreadingHTTPServer(("127.0.0.1", bind_port), _Handler)
        server.offset = offset  # type: ignore[attr-defined]
        server.echo_mode = True  # type: ignore[attr-defined]
        server.break_mode = ""  # type: ignore[attr-defined]
        # vLLM's default is decoded-text keys; the self test flips this to
        # "ids"/"numeric" to cover the placeholder and plain-id formats too.
        server.key_style = "text"  # type: ignore[attr-defined]
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
    parser.add_argument("--ports", default="18034,18035")
    parser.add_argument("--offsets", default="1e-6,0.2")
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
