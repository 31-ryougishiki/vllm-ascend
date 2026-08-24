# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-overhead NPU-event timing for DeepSeek V4 DSA attention.

This module exists to compare ``additional_config.enable_dsa_cp=true`` with
``false`` without disturbing the measurement too much:

* Timing is disabled unless ``VLLM_ASCEND_DSA_TIMING=1``.
* Every timed phase records two ``torch.npu.Event`` markers on the compute
  stream.  Recording is non-blocking and does not synchronize the host.
* A synchronization is only performed once per ``window`` model forwards
  (default 10).  After that single synchronization all recorded intervals are
  queried and accumulated.  This keeps the per-layer overhead to two event
  records, which is far cheaper than calling ``torch.npu.synchronize()`` per
  layer (43+ layers) or per attention phase.
* Results are written per process as raw and summary CSV files so the
  orchestrator can aggregate across DP/TP ranks.
"""

from __future__ import annotations

import atexit
import csv
import os
import re
from dataclasses import dataclass

import torch
from vllm.logger import init_logger

from vllm_ascend import envs
from vllm_ascend.utils import enable_dsa_cp

logger = init_logger(__name__)

# Phases instrumented in both AscendDSAImpl and AscendDSACPImpl.
PHASE_PROLOG = "prolog"
PHASE_INDEXER = "indexer"
PHASE_COMPRESSOR = "compressor"
PHASE_ATTENTION = "attention"
PHASE_OUTPUT = "output"

_PHASES = (PHASE_PROLOG, PHASE_INDEXER, PHASE_COMPRESSOR, PHASE_ATTENTION, PHASE_OUTPUT)


def scenario_for_ratio(compress_ratio: int) -> str:
    """Map DeepSeek V4 layer kind to a stable scenario name."""
    ratio = max(int(compress_ratio), 0)
    if ratio <= 1:
        return "full"
    if ratio == 4:
        return "indexer_c4"
    if ratio == 128:
        return "compressed_c128"
    return f"c{ratio}"


@dataclass
class _PhaseSample:
    layer_name: str
    phase: str
    scenario: str
    start_event: torch.npu.Event
    end_event: torch.npu.Event
    num_tokens: int = 0
    attn_state: str = ""
    started: bool = False
    ended: bool = False


def _safe_tag() -> str:
    tag = envs.VLLM_ASCEND_DSA_TIMING_TAG or ""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", tag)


class DsaAttentionTimer:
    """Singleton-ish collector shared by every DSV4 attention layer.

    Each process (one NPU rank) has one collector.  Layers reuse a pool of
    event pairs; one slot per (layer, phase) per window step.
    """

    def __init__(self) -> None:
        self._enabled: bool | None = None
        self._window: int | None = None
        self._log_interval: int | None = None
        self._output_dir: str | None = None
        self._tag: str | None = None
        self._active = False
        self._step_id = 0
        self._flushed_steps = 0
        self._samples: dict[tuple[str, str], list[_PhaseSample]] = {}
        self._used_keys: set[tuple[str, str]] = set()
        self._accum: dict[tuple[str, str], dict[str, float]] = {}
        self._raw_path: str | None = None
        self._summary_path: str | None = None

    @property
    def enabled(self) -> bool:
        if self._enabled is None:
            self._enabled = bool(int(getattr(envs, "VLLM_ASCEND_DSA_TIMING", 0)))
        return self._enabled

    @property
    def window(self) -> int:
        if self._window is None:
            self._window = max(
                1, int(getattr(envs, "VLLM_ASCEND_DSA_TIMING_WINDOW", 10))
            )
        return self._window

    @property
    def log_interval(self) -> int:
        if self._log_interval is None:
            self._log_interval = max(
                self.window,
                int(getattr(envs, "VLLM_ASCEND_DSA_TIMING_LOG_INTERVAL", 50)),
            )
        return self._log_interval

    @property
    def output_dir(self) -> str:
        if self._output_dir is None:
            self._output_dir = str(
                getattr(envs, "VLLM_ASCEND_DSA_TIMING_OUTPUT_DIR", "./dsa_timing")
            )
        return self._output_dir

    @property
    def tag(self) -> str:
        if self._tag is None:
            self._tag = _safe_tag()
        return self._tag

    def _paths(self) -> tuple[str, str]:
        if self._raw_path is None:
            os.makedirs(self.output_dir, exist_ok=True)
            suffix = self.tag or str(os.getpid())
            self._raw_path = os.path.join(
                self.output_dir, f"dsa_timing_raw_{suffix}.csv"
            )
            self._summary_path = os.path.join(
                self.output_dir, f"dsa_timing_summary_{suffix}.csv"
            )
        return self._raw_path, self._summary_path

    @staticmethod
    def _new_event() -> torch.npu.Event:
        return torch.npu.Event(enable_timing=True)

    def _slot(self, layer_name: str, phase: str) -> _PhaseSample:
        key = (layer_name, phase)
        slots = self._samples.get(key)
        if slots is None:
            slots = [
                self._make_sample(layer_name, phase) for _ in range(self.window)
            ]
            self._samples[key] = slots
        idx = self._step_id % self.window
        return slots[idx]

    def _make_sample(self, layer_name: str, phase: str) -> _PhaseSample:
        # Events are allocated lazily; the module may be imported on CPU-only
        # build/test machines where torch.npu.Event is unavailable.
        return _PhaseSample(
            layer_name=layer_name,
            phase=phase,
            scenario=scenario_for_ratio(0),
            start_event=self._new_event(),
            end_event=self._new_event(),
        )

    def begin_step(self) -> None:
        if not self.enabled:
            return
        self._step_id += 1
        self._active = True

    def mark_start(
        self,
        layer_name: str,
        phase: str,
        scenario: str,
        num_tokens: int,
        attn_state=None,
    ) -> None:
        if not self.enabled or not self._active:
            return
        if ".mtp." in layer_name:
            # MTP draft attention is out of scope for the DSV4 main-model
            # c0/c4/c128 comparison.
            return
        sample = self._slot(layer_name, phase)
        if sample.started and not sample.ended:
            # A previous window cycle should have been flushed before reuse.
            return
        sample.scenario = scenario
        sample.num_tokens = max(0, int(num_tokens))
        sample.attn_state = getattr(attn_state, "name", str(attn_state))
        sample.start_event.record()
        sample.started = True
        sample.ended = False
        self._used_keys.add((layer_name, phase))

    def mark_end(self, layer_name: str, phase: str) -> None:
        if not self.enabled or not self._active:
            return
        sample = self._slot(layer_name, phase)
        if not sample.started:
            return
        sample.end_event.record()
        sample.ended = True

    def end_step(self) -> None:
        if not self.enabled or not self._active:
            return
        self._active = False
        if self._step_id % self.window != 0:
            return
        if not self._used_keys:
            self._flushed_steps += self.window
            return
        self._flush()

    def _flush(self) -> None:
        # One device synchronization for a whole window of model forwards.
        torch.npu.synchronize()

        raw_rows: list[dict[str, object]] = []
        for key in list(self._used_keys):
            for sample in self._samples[key]:
                if not (sample.started and sample.ended):
                    continue
                elapsed_ms = sample.end_event.elapsed_time(sample.start_event)
                ms_per_token = (
                    elapsed_ms / sample.num_tokens
                    if sample.num_tokens > 0
                    else 0.0
                )
                raw_rows.append(
                    {
                        "step_window": self._step_id // self.window,
                        "layer_name": sample.layer_name,
                        "scenario": sample.scenario,
                        "phase": sample.phase,
                        "num_tokens": sample.num_tokens,
                        "attn_state": sample.attn_state,
                        "elapsed_ms": round(float(elapsed_ms), 6),
                        "ms_per_token": round(ms_per_token, 9),
                    }
                )
                acc_key = (sample.scenario, sample.phase)
                acc = self._accum.setdefault(
                    acc_key, {"count": 0.0, "sum_ms": 0.0, "tokens": 0.0}
                )
                acc["count"] += 1.0
                acc["sum_ms"] += float(elapsed_ms)
                acc["tokens"] += float(sample.num_tokens)
            for sample in self._samples[key]:
                sample.started = False
                sample.ended = False
        self._used_keys.clear()

        if raw_rows:
            self._write_raw_rows(raw_rows)
        self._write_summary()
        self._flushed_steps += self.window
        if self._flushed_steps % self.log_interval == 0:
            logger.info(
                "DSA timing summary (steps=%s): %s",
                self._flushed_steps,
                {
                    f"{scenario}/{phase}": {
                        "avg_ms": round(acc["sum_ms"] / acc["count"], 3)
                        if acc["count"]
                        else 0.0,
                        "ms_per_token": round(acc["sum_ms"] / acc["tokens"], 6)
                        if acc["tokens"]
                        else 0.0,
                    }
                    for (scenario, phase), acc in sorted(self._accum.items())
                },
            )

    def _write_raw_rows(self, rows: list[dict[str, object]]) -> None:
        raw_path, _ = self._paths()
        fieldnames = [
            "step_window",
            "layer_name",
            "scenario",
            "phase",
            "num_tokens",
            "attn_state",
            "elapsed_ms",
            "ms_per_token",
        ]
        write_header = not os.path.exists(raw_path)
        with open(raw_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def _write_summary(self) -> None:
        _, summary_path = self._paths()
        fieldnames = [
            "dsa_cp",
            "scenario",
            "phase",
            "count",
            "sum_ms",
            "avg_ms",
            "total_tokens",
            "ms_per_token",
        ]
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for (scenario, phase), acc in sorted(self._accum.items()):
                writer.writerow(
                    {
                        "dsa_cp": bool(enable_dsa_cp()),
                        "scenario": scenario,
                        "phase": phase,
                        "count": int(acc["count"]),
                        "sum_ms": round(acc["sum_ms"], 6),
                        "avg_ms": round(acc["sum_ms"] / acc["count"], 6)
                        if acc["count"]
                        else 0.0,
                        "total_tokens": int(acc["tokens"]),
                        "ms_per_token": round(acc["sum_ms"] / acc["tokens"], 9)
                        if acc["tokens"]
                        else 0.0,
                    }
                )

    def flush(self) -> None:
        if not self.enabled:
            return
        self._active = False
        if not self._used_keys:
            return
        try:
            self._flush()
        except Exception:  # noqa: BLE001 - best effort at interpreter shutdown
            logger.exception("Failed to flush DSA timing samples")


dsa_timer = DsaAttentionTimer()
atexit.register(dsa_timer.flush)
