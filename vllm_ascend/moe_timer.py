# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Per-step MoE/Attention timing utility using mspti for NPU-level profiling.

Set VLLM_ASCEND_DISABLE_MOE_TIMER=1 to disable this timer entirely,
eliminating all performance overhead from timing operations.

Set VLLM_ASCEND_MSPTI_TIMER=1 to enable mspti-backed range markers
instead of CPU wall-clock timing.  MstxMonitor captures NPU execution
time for both compute and communication within each range.
"""
import os
import time
from typing import Dict, List, Optional

from vllm.logger import init_logger

logger = init_logger(__name__)

# Layers to time (1-indexed, e.g. time layer 4 and layer 10)
TIME_LAYERS: List[int] = [24]

# Accumulated timing records: list of {layer, step, seg, dt_ms}
_records: List[Dict] = []
_step_counter: int = 0
_num_tokens: int = 0
_step_t0: float = 0.0
_prev_dump_time: float = 0.0
_current_layer: Optional[int] = None

# mspti state
_t0: float = 0.0
_range_ids: Dict[str, object] = {}
_segment_layers: Dict[str, int] = {}
_mspti_enabled: bool = False
_mstx_range_start = None
_mstx_range_end = None
_mstx_monitor = None

# Lazily evaluated flags
_disabled: Optional[bool] = None
_mspti_checked: Optional[bool] = None


def _is_disabled() -> bool:
    global _disabled
    if _disabled is None:
        from vllm_ascend import envs as ascend_envs
        _disabled = ascend_envs.VLLM_ASCEND_DISABLE_MOE_TIMER
    return _disabled


def _init_mspti():
    """Lazily initialize mspti profiling on first use."""
    global _mspti_checked, _mspti_enabled
    global _mstx_range_start, _mstx_range_end, _mstx_monitor

    if _mspti_checked is not None:
        return
    _mspti_checked = True

    _mspti_enabled = os.environ.get("VLLM_ASCEND_MSPTI_TIMER", "0") == "1"
    if not _mspti_enabled:
        return

    try:
        import torch_npu
        from mspti import MstxMonitor

        _mstx_range_start = torch_npu.npu.mstx.range_start
        _mstx_range_end = torch_npu.npu.mstx.range_end

        _mstx_monitor = MstxMonitor()
        _mstx_monitor.start(
            mark_cb=None,
            range_cb=_on_range_data,
        )

        logger.info("moe_timer: mspti profiling enabled (MstxMonitor)")
    except Exception as e:
        logger.warning("moe_timer: mspti init failed: %s, falling back to CPU timer", e)
        _mspti_enabled = False
        _mstx_monitor = None


def _on_range_data(data):
    """MstxMonitor range callback: record range marker durations.
    Only records markers created by our tick()/tock() calls (those with
    clean names).  Auto-generated HCCL markers (JSON blobs) and unnamed
    markers are filtered out."""
    name = data.name
    if not name or name.startswith("{"):
        return
    duration_ms = (data.end - data.start) / 1_000_000.0
    layer = _segment_layers.get(name, _current_layer)
    _records.append({
        "step": _step_counter,
        "layer": layer,
        "seg": name,
        "dt_ms": round(duration_ms, 3),
    })


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
    _init_mspti()
    if _is_disabled():
        return
    global _current_layer
    _current_layer = layer_idx


def should_time() -> bool:
    if _is_disabled():
        return False
    return _current_layer in TIME_LAYERS


def tick(name: str = ""):
    """Start timing segment.  Only records when should_time() is True."""
    if _is_disabled():
        return None
    _init_mspti()
    global _t0, _range_ids
    _t0 = time.perf_counter()
    if not should_time():
        return None
    _start_mspti_range(name)
    return None


def tick_always(name: str):
    """Start timing segment, bypassing should_time(). For embed, lm_head etc."""
    if _is_disabled():
        return None
    _init_mspti()
    global _t0
    _t0 = time.perf_counter()
    _start_mspti_range(name)
    return None


def _start_mspti_range(name: str):
    global _range_ids, _segment_layers
    if _mspti_enabled and name and _mstx_range_start is not None:
        try:
            rid = _mstx_range_start(name)
            _range_ids[name] = rid
            _segment_layers[name] = _current_layer if _current_layer is not None else -1
        except Exception:
            pass


def tock(segment: str):
    """End timing segment and record.  In mspti mode the name given to
    tick() is used as the mstx range name; *segment* is used for the
    CPU-timer fallback label."""
    if _is_disabled():
        return
    if not should_time():
        return
    dt = (time.perf_counter() - _t0) * 1000  # ms
    if _mspti_enabled:
        _end_mspti_range(segment)
    else:
        _records.append({
            "step": _step_counter,
            "layer": _current_layer,
            "seg": segment,
            "dt_ms": round(dt, 3),
        })


def tock_always(segment: str):
    """End timing and record, bypassing should_time()."""
    if _is_disabled():
        return
    dt = (time.perf_counter() - _t0) * 1000  # ms
    if _mspti_enabled:
        _end_mspti_range(segment)
    else:
        _records.append({
            "step": _step_counter,
            "layer": _current_layer,
            "seg": segment,
            "dt_ms": round(dt, 3),
        })


def _end_mspti_range(name: str):
    """End the mspti range marker started by tick(name)."""
    global _range_ids
    rid = _range_ids.pop(name, None)
    if rid is not None and _mstx_range_end is not None:
        try:
            _mstx_range_end(rid)
        except Exception:
            pass


def save() -> float:
    if _is_disabled():
        return 0.0
    return _t0


def restore(t0: float):
    if _is_disabled():
        return
    global _t0
    _t0 = t0


def get_records() -> List[Dict]:
    return _records


def dump():
    """Print current step timing summary."""
    if _is_disabled():
        return
    global _prev_dump_time
    now = time.perf_counter()
    step_total = (now - _step_t0) * 1000
    gap = (now - _prev_dump_time) * 1000 if _prev_dump_time > 0 else 0
    _prev_dump_time = now

    if _mspti_enabled and _mstx_monitor is not None:
        try:
            _mstx_monitor.flush_all()
        except Exception:
            pass

    if not _records:
        return

    logger.info("=== Step %d  num_tokens=%d  total=%.3f ms  "
                "gap_from_prev_dump=%.3f ms  Timing (ms) ===",
                _step_counter, _num_tokens, step_total, gap)
    for r in _records:
        _ly = r["layer"] if r["layer"] is not None else -1
        logger.info("  layer=%3d  %-30s  %8.3f ms",
                    _ly, r["seg"], r["dt_ms"])
