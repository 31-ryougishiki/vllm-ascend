# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Per-step MoE/Attention timing utility using torch.npu.Event for accurate
NPU stream timing (not CPU wall-clock).

Set VLLM_ASCEND_DISABLE_MOE_TIMER=1 to disable this timer entirely,
eliminating all performance overhead from timing operations.
"""
import time
from typing import Dict, List, Optional

from vllm.logger import init_logger

logger = init_logger(__name__)

# Layers to time (1-indexed, e.g. time layer 4 and layer 10)
TIME_LAYERS: List[int] = [24]

# Accumulated timing records: list of {layer, step, seg, dt_ms}
_records: List[Dict] = []
# Pending npu.Event observations: list of (segment, layer, start_evt, end_evt)
_pending_events: List = []
_step_counter: int = 0
_num_tokens: int = 0
_step_t0: float = 0.0
_prev_dump_time: float = 0.0
_current_layer: Optional[int] = None

# Lazily evaluated disable flag
_disabled: Optional[bool] = None

# NPU event timing state
_npu_event_start: object = None
_npu_event_enabled: bool = True


def _is_disabled() -> bool:
    global _disabled
    if _disabled is None:
        from vllm_ascend import envs as ascend_envs
        _disabled = ascend_envs.VLLM_ASCEND_DISABLE_MOE_TIMER
    return _disabled


def step_begin(num_tokens: int = 0):
    """Call once per scheduling step / model forward."""
    if _is_disabled():
        return
    global _step_counter, _records, _num_tokens, _step_t0
    _step_counter += 1
    _num_tokens = num_tokens
    _step_t0 = time.perf_counter()
    _records.clear()


def layer_begin(layer_idx: int):
    if _is_disabled():
        return
    global _current_layer
    _current_layer = layer_idx


def should_time() -> bool:
    if _is_disabled():
        return False
    return _current_layer in TIME_LAYERS


def tick(name: str = ""):
    """Start timing segment. Records an npu.Event on the current stream."""
    if _is_disabled():
        return
    global _npu_event_start, _npu_event_enabled
    if not should_time() and name:
        # For tick_always callers proxied through tick()
        if _current_layer is not None and _current_layer not in TIME_LAYERS:
            return
    try:
        import torch
        _npu_event_start = torch.npu.Event(enable_timing=True)
        _npu_event_start.record()
    except Exception:
        _npu_event_enabled = False
        _npu_event_start = None


def tick_always(name: str):
    """Start timing, bypassing should_time(). For embed, lm_head etc."""
    if _is_disabled():
        return
    global _npu_event_start, _npu_event_enabled
    try:
        import torch
        _npu_event_start = torch.npu.Event(enable_timing=True)
        _npu_event_start.record()
    except Exception:
        _npu_event_enabled = False
        _npu_event_start = None


def tock(segment: str):
    """End timing. Records end event and stores pending observation."""
    if _is_disabled():
        return
    if not should_time():
        return
    _record_event(segment)


def tock_always(segment: str):
    """End timing bypassing should_time()."""
    if _is_disabled():
        return
    _record_event(segment)


def _record_event(segment: str):
    global _npu_event_start, _npu_event_enabled, _pending_events
    if not _npu_event_enabled or _npu_event_start is None:
        return
    try:
        import torch
        end_evt = torch.npu.Event(enable_timing=True)
        end_evt.record()
        _pending_events.append(
            (segment, _current_layer, _npu_event_start, end_evt))
        _npu_event_start = None
    except Exception:
        _npu_event_enabled = False


def save() -> float:
    return 0.0


def restore(t0: float):
    pass


def get_records() -> List[Dict]:
    return _records


def dump():
    """Synchronize pending events and print step timing summary."""
    if _is_disabled():
        return
    global _prev_dump_time, _pending_events

    _flush_events()

    if not _records:
        return

    now = time.perf_counter()
    step_total = (now - _step_t0) * 1000
    gap = (now - _prev_dump_time) * 1000 if _prev_dump_time > 0 else 0
    _prev_dump_time = now

    logger.info("=== Step %d  num_tokens=%d  total=%.3f ms  "
                "gap_from_prev_dump=%.3f ms  Timing (ms) ===",
                _step_counter, _num_tokens, step_total, gap)
    for r in _records:
        _ly = r["layer"] if r["layer"] is not None else -1
        logger.info("  layer=%3d  %-30s  %8.3f ms",
                    _ly, r["seg"], r["dt_ms"])


def _flush_events():
    """Synchronize all pending npu.Event pairs and convert to records."""
    global _pending_events, _records
    if not _pending_events:
        return
    try:
        for seg, layer, start_evt, end_evt in _pending_events:
            start_evt.synchronize()
            end_evt.synchronize()
            dt_ms = start_evt.elapsed_time(end_evt)
            _records.append({
                "step": _step_counter,
                "layer": layer,
                "seg": seg,
                "dt_ms": round(dt_ms, 3),
            })
    except Exception:
        pass
    _pending_events.clear()
