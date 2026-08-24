# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import tempfile
from unittest.mock import MagicMock, patch


def test_scenario_for_ratio():
    from vllm_ascend.attention.dsa_timing import scenario_for_ratio

    assert scenario_for_ratio(0) == "full"
    assert scenario_for_ratio(1) == "full"
    assert scenario_for_ratio(4) == "indexer_c4"
    assert scenario_for_ratio(128) == "compressed_c128"
    assert scenario_for_ratio(32) == "c32"


def test_windowed_flush_syncs_once():
    from vllm_ascend.attention import dsa_timing

    class FakeEvent:
        def __init__(self, *args, **kwargs):
            self.recorded = False

        def record(self):
            self.recorded = True

        @staticmethod
        def elapsed_time(other):
            return 5.0

    timer = dsa_timing.dsa_timer
    timer._samples.clear()
    timer._used_keys.clear()
    timer._accum.clear()
    timer._raw_path = None
    timer._summary_path = None
    timer._step_id = 0
    timer._flushed_steps = 0
    timer._enabled = None
    timer._window = None
    timer._log_interval = None
    timer._output_dir = None

    if not hasattr(dsa_timing.torch, "npu"):
        dsa_timing.torch.npu = MagicMock()

    sync_calls = []
    with tempfile.TemporaryDirectory() as out_dir, patch.dict(
        os.environ,
        {
            "VLLM_ASCEND_DSA_TIMING": "1",
            "VLLM_ASCEND_DSA_TIMING_WINDOW": "2",
            "VLLM_ASCEND_DSA_TIMING_LOG_INTERVAL": "100",
            "VLLM_ASCEND_DSA_TIMING_OUTPUT_DIR": out_dir,
            "VLLM_ASCEND_DSA_TIMING_TAG": "ut",
        },
    ), patch.object(
        dsa_timing.torch.npu, "synchronize", side_effect=lambda: sync_calls.append(1)
    ), patch.object(
        dsa_timing.DsaAttentionTimer,
        "_new_event",
        staticmethod(lambda: FakeEvent()),
    ):
        layer = "model.layers.0.self_attn"
        timer.begin_step()
        timer.mark_start(layer, "attention", "full", 100, None)
        timer.mark_end(layer, "attention")
        timer.end_step()
        assert sync_calls == [], "no sync should happen before window is full"

        timer.begin_step()
        timer.mark_start(layer, "attention", "full", 100, None)
        timer.mark_end(layer, "attention")
        timer.end_step()
        assert sync_calls == [1], "exactly one sync should happen per window"

        summary_path = timer._summary_path
        assert summary_path is not None
        assert summary_path.endswith("dsa_timing_summary_ut.csv")

    # Reset the singleton so other tests see a clean state.
    timer._samples.clear()
    timer._used_keys.clear()
    timer._accum.clear()
    timer._raw_path = None
    timer._summary_path = None
    timer._step_id = 0
    timer._flushed_steps = 0
    timer._enabled = None
    timer._window = None
    timer._log_interval = None
    timer._output_dir = None
