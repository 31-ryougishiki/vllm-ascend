# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Per-step MoE/Attention timing utility.  Times 2 specific layers each step.
"""
import time
from typing import Dict, List, Optional

from vllm.logger import init_logger

logger = init_logger(__name__)

# Layers to time (1-indexed, e.g. time layer 4 and layer 10)
TIME_LAYERS: List[int] = [24]

# Accumulated timing records: list of {layer, step, seg, dt_ms}
_records: List[Dict] = []
_step_counter: int = 0
_current_layer: Optional[int] = None
_t0: float = 0.0


def step_begin():
    """Call once per scheduling step / model forward."""
    global _step_counter, _records
    _step_counter += 1
    _records.clear()


def layer_begin(layer_idx: int):
    global _current_layer
    _current_layer = layer_idx


def should_time() -> bool:
    return _current_layer in TIME_LAYERS


def tick():
    """Start timing segment."""
    global _t0
    _t0 = time.perf_counter()


def tock(segment: str):
    """End timing segment and record."""
    if not should_time():
        return
    dt = (time.perf_counter() - _t0) * 1000  # ms
    _records.append({
        "step": _step_counter,
        "layer": _current_layer,
        "seg": segment,
        "dt_ms": round(dt, 3),
    })


def get_records() -> List[Dict]:
    return _records


def dump():
    """Print current step timing summary."""
    if not _records:
        return
    logger.info("=== Step %d Timing (ms) ===", _step_counter)
    for r in _records:
        logger.info("  layer=%3d  %-30s  %8.3f ms",
                    r["layer"], r["seg"], r["dt_ms"])
