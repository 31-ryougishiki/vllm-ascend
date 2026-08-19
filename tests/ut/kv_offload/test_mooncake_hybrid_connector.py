import sys
import threading
import time
import types
import unittest
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock, patch

import torch

fake_engine = types.ModuleType("mooncake.engine")
fake_engine.TransferEngine = MagicMock()  # type: ignore[attr-defined]
sys.modules["mooncake.engine"] = fake_engine

from vllm.v1.request import RequestStatus  # noqa: E402

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import (  # noqa: E402
    MAX_REQUESTS_PER_PEER_HANDLER,
    KVCacheRecvingThread,
    MooncakeConnectorScheduler,
    get_dp_side_channel_port_offset,
)


class MockRequest:
    def __init__(
        self,
        request_id,
        prompt_token_ids,
        kv_transfer_params,
        status,
        num_prompt_tokens=None,
    ):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        if num_prompt_tokens is None:
            num_prompt_tokens = len(prompt_token_ids) if prompt_token_ids is not None else 0
        self.num_prompt_tokens = num_prompt_tokens
        self.kv_transfer_params = kv_transfer_params
        self.status = status
        self.output_token_ids = [101]


class TestHybridKVCacheRecvingThreadDispatch(unittest.TestCase):
    def _make_thread(self):
        thread = object.__new__(KVCacheRecvingThread)
        thread.executor = ThreadPoolExecutor(max_workers=2)
        thread.peer_request_queues = defaultdict(deque)
        thread.active_peer_request_handlers = set()
        thread.peer_request_queues_lock = threading.Lock()
        thread.request_task_counts = defaultdict(int)
        thread.finished_request_markers = set()
        thread.request_task_counts_lock = threading.Lock()
        return thread

    def test_executor_workers_bind_kv_cache_device_before_handling_requests(self):
        expected_device = torch.device("npu:5")
        kv_cache = MagicMock(device=expected_device)
        model_config = types.SimpleNamespace(
            is_deepseek_mla=False,
            hf_config=types.SimpleNamespace(compress_ratios=[1]),
            hf_text_config=types.SimpleNamespace(num_hidden_layers=1),
        )
        vllm_config = types.SimpleNamespace(
            model_config=model_config,
            cache_config=types.SimpleNamespace(block_size=16),
        )
        kv_cache_config = types.SimpleNamespace(kv_cache_groups=[])
        worker_events: defaultdict[int, list[tuple[str, int | str]]] = defaultdict(list)
        events_lock = threading.Lock()
        both_workers_started = threading.Event()
        release_workers = threading.Event()

        def record_set_device(device):
            device_index = device if isinstance(device, int) else torch.device(device).index
            with events_lock:
                worker_events[threading.get_ident()].append(("set_device", device_index))

        with (
            patch("torch.npu.set_device", side_effect=record_set_device),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector.is_vl_model",
                return_value=False,
            ),
        ):
            thread = KVCacheRecvingThread(
                tp_rank=1,
                tp_size=2,
                _prefill_pp_size=1,
                engine=MagicMock(),
                local_engine_id="local_engine",
                local_handshake_port=5555,
                side_channel_port=30000,
                local_kv_caches_base_addr=[0x1000],
                block_len_per_addr=[1024],
                block_stride_per_addr=[1024],
                addr_group_idx=[0],
                mamba_ssm_size=(0, 0),
                use_hybrid=False,
                has_mamba=False,
                hma_group_size=1,
                ready_event=threading.Event(),
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
                kv_caches={"layer.0": (kv_cache, kv_cache)},
            )

            def handle_request(req_meta: dict[str, Any]):
                with events_lock:
                    worker_events[threading.get_ident()].append(("handle", req_meta["request_id"]))
                    handled_worker_count = sum(
                        any(event == "handle" for event, _ in events) for events in worker_events.values()
                    )
                    if handled_worker_count == 2:
                        both_workers_started.set()
                release_workers.wait()

            thread._handle_request = handle_request  # type: ignore[method-assign]
            try:
                for index in range(2):
                    thread._submit_request(
                        {
                            "request_id": f"req-{index}",
                            "remote_host": f"host-{index}",
                            "remote_handshake_port": 6000 + index,
                            "all_task_done": True,
                        }
                    )
                self.assertTrue(both_workers_started.wait(timeout=5.0), "executor did not start two workers")
            finally:
                release_workers.set()
                thread.executor.shutdown(wait=True, cancel_futures=True)

        handled_worker_events = [events for events in worker_events.values() if any(e == "handle" for e, _ in events)]
        self.assertEqual(len(handled_worker_events), 2)
        for events in handled_worker_events:
            self.assertEqual(events[0], ("set_device", expected_device.index))
            self.assertEqual(events[1][0], "handle")

    def test_submit_request_serializes_same_peer_fifo(self):
        thread = self._make_thread()
        release_first_request = threading.Event()
        first_request_started = threading.Event()
        other_peer_started = threading.Event()
        handled_requests: list[str] = []
        active_by_peer: defaultdict[tuple[str, int], int] = defaultdict(int)
        max_active_by_peer: defaultdict[tuple[str, int], int] = defaultdict(int)
        state_lock = threading.Lock()

        def handle_request(req_meta: dict[str, Any]):
            peer_key = (req_meta["remote_host"], req_meta["remote_handshake_port"])
            with state_lock:
                active_by_peer[peer_key] += 1
                max_active_by_peer[peer_key] = max(max_active_by_peer[peer_key], active_by_peer[peer_key])
                handled_requests.append(req_meta["request_id"])

            if req_meta["request_id"] == "same-peer-1":
                first_request_started.set()
                self.assertTrue(release_first_request.wait(timeout=2.0))
            elif req_meta["request_id"] == "other-peer-1":
                other_peer_started.set()

            time.sleep(0.01)
            with state_lock:
                active_by_peer[peer_key] -= 1

        thread._handle_request = handle_request  # type: ignore[method-assign]
        same_peer_1 = {
            "request_id": "same-peer-1",
            "remote_host": "host-a",
            "remote_handshake_port": 6000,
            "all_task_done": False,
        }
        same_peer_2 = {
            "request_id": "same-peer-2",
            "remote_host": "host-a",
            "remote_handshake_port": 6000,
            "all_task_done": True,
        }
        other_peer = {
            "request_id": "other-peer-1",
            "remote_host": "host-b",
            "remote_handshake_port": 6001,
            "all_task_done": True,
        }

        try:
            thread._submit_request(same_peer_1)
            self.assertTrue(first_request_started.wait(timeout=1.0))
            thread._submit_request(same_peer_2)
            thread._submit_request(other_peer)

            self.assertTrue(other_peer_started.wait(timeout=1.0))
            time.sleep(0.05)
            self.assertNotIn("same-peer-2", handled_requests)
        finally:
            release_first_request.set()
            thread.executor.shutdown(wait=True, cancel_futures=True)

        self.assertLess(handled_requests.index("same-peer-1"), handled_requests.index("same-peer-2"))
        self.assertEqual(max_active_by_peer[("host-a", 6000)], 1)
        self.assertEqual(max_active_by_peer[("host-b", 6001)], 1)

    def test_peer_handler_yields_after_batch_limit(self):
        thread = self._make_thread()
        peer_key = ("host-a", 6000)
        requests = [
            {
                "request_id": f"req-{idx}",
                "remote_host": peer_key[0],
                "remote_handshake_port": peer_key[1],
            }
            for idx in range(MAX_REQUESTS_PER_PEER_HANDLER + 1)
        ]
        handled_requests: list[str] = []
        thread.peer_request_queues[peer_key].extend(requests)
        thread.active_peer_request_handlers.add(peer_key)
        thread.executor = MagicMock()

        def handle_request(req_meta: dict[str, Any]):
            handled_requests.append(req_meta["request_id"])

        thread._handle_request = handle_request  # type: ignore[method-assign]

        thread._handle_peer_requests(peer_key)

        self.assertEqual(handled_requests, [f"req-{idx}" for idx in range(MAX_REQUESTS_PER_PEER_HANDLER)])
        self.assertEqual(
            [req["request_id"] for req in thread.peer_request_queues[peer_key]],
            [f"req-{MAX_REQUESTS_PER_PEER_HANDLER}"],
        )
        self.assertIn(peer_key, thread.active_peer_request_handlers)
        thread.executor.submit.assert_called_once_with(thread._handle_peer_requests, peer_key)


class TestMooncakeHybridConnectorScheduler(unittest.TestCase):
    def _make_scheduler(self):
        scheduler = object.__new__(MooncakeConnectorScheduler)
        scheduler.use_hybrid = True
        scheduler.use_compress = True
        scheduler.num_swa_blocks = [0, 2]
        scheduler.group_block_size = [128, 128]
        scheduler.group_compress_ratio = [4, 1]
        scheduler._reqs_need_send = {}
        scheduler.block_size = 128
        scheduler.engine_id = "engine"
        scheduler.side_channel_host = "127.0.0.1"
        scheduler.side_channel_port = 12345
        scheduler.tp_size = 1
        scheduler.multi_nodes_meta_mapping = {}
        return scheduler

    def test_compute_transfer_block_ids_trims_swa_groups(self):
        scheduler = self._make_scheduler()
        block_ids = (list(range(10)), [100, 101, 102, 103])

        transfer_block_ids = scheduler._compute_transfer_block_ids(block_ids, prompt_len=129)

        self.assertEqual(transfer_block_ids, ([0], [100, 101]))

    def test_request_finished_trims_before_swa_clip(self):
        scheduler = self._make_scheduler()
        request = MockRequest(
            "req1",
            prompt_token_ids=list(range(129)),
            kv_transfer_params={"do_remote_decode": True},
            status=RequestStatus.FINISHED_LENGTH_CAPPED,
        )
        block_ids = (list(range(10)), [100, 101, 102, 103])

        delay_free, params = scheduler.request_finished_all_groups(request, block_ids)

        self.assertTrue(delay_free)
        self.assertIsNotNone(params)
        self.assertEqual(params["remote_block_ids"], ([0], [100, 101]))
        self.assertEqual(params["num_prompt_blocks"], 2)
        self.assertIn("req1", scheduler._reqs_need_send)

    def test_request_finished_uses_num_prompt_tokens(self):
        scheduler = self._make_scheduler()
        request = MockRequest(
            "req1",
            prompt_token_ids=None,
            kv_transfer_params={"do_remote_decode": True},
            status=RequestStatus.FINISHED_LENGTH_CAPPED,
            num_prompt_tokens=129,
        )
        block_ids = (list(range(10)), [100, 101, 102, 103])

        delay_free, params = scheduler.request_finished_all_groups(request, block_ids)

        self.assertTrue(delay_free)
        self.assertIsNotNone(params)
        self.assertEqual(params["remote_block_ids"], ([0], [100, 101]))
        self.assertEqual(params["num_prompt_blocks"], 2)


class TestHeterogeneousMooncakeHybridPorts(unittest.TestCase):
    def _make_parallel_config(
        self, *, hetero: bool, dp_rank: int, tp_size: int
    ) -> types.SimpleNamespace:
        if hetero:
            offsets = [0, 3, 7, 11]

            def get_rank_offset_for_dp(rank: int) -> int:
                return offsets[rank]

            return types.SimpleNamespace(
                is_heterogeneous_tp=True,
                data_parallel_rank=dp_rank,
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=1,
                get_rank_offset_for_dp=get_rank_offset_for_dp,
            )
        return types.SimpleNamespace(
            is_heterogeneous_tp=False,
            data_parallel_rank=dp_rank,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=1,
        )

    def test_heterogeneous_dp_port_offset_is_cumulative(self):
        self.assertEqual(
            get_dp_side_channel_port_offset(
                self._make_parallel_config(hetero=True, dp_rank=0, tp_size=3)
            ),
            0,
        )
        self.assertEqual(
            get_dp_side_channel_port_offset(
                self._make_parallel_config(hetero=True, dp_rank=2, tp_size=4)
            ),
            7,
        )

    def test_homogeneous_dp_port_offset_uses_uniform_stride(self):
        self.assertEqual(
            get_dp_side_channel_port_offset(
                self._make_parallel_config(hetero=False, dp_rank=3, tp_size=4)
            ),
            12,
        )

    def test_scheduler_handshake_mapping_uses_published_handshake_port(self):
        kv_port = 36000
        scheduler = object.__new__(MooncakeConnectorScheduler)
        scheduler.vllm_config = types.SimpleNamespace(
            kv_transfer_config=types.SimpleNamespace(kv_port=kv_port)
        )
        scheduler.multi_nodes_meta_mapping = {}

        # Heterogeneous DP0 publishes global offsets 0, 1, 2 while the local
        # worker keys are still 0, 1, 2.  DP1 would publish 3..6 instead.
        metadata = {
            local_rank: types.SimpleNamespace(
                local_ip=f"10.0.0.{local_rank}",
                engine_id=f"engine-{local_rank}",
                handshake_port=kv_port + local_rank,
            )
            for local_rank in range(3)
        }
        scheduler.set_xfer_handshake_metadata(metadata)

        self.assertEqual(
            scheduler.multi_nodes_meta_mapping["2"],
            {
                "host": "10.0.0.2",
                "engine_id": "engine-2",
                "handshake_port": kv_port + 2,
            },
        )

    def test_remote_host_lookup_falls_back_to_dp_local_offset(self):
        worker = types.SimpleNamespace(
            vllm_config=types.SimpleNamespace(
                kv_transfer_config=types.SimpleNamespace(kv_port=36000)
            )
        )
        # Old-style mapping keyed by DP-local rank.
        mapping = {"2": {"host": "10.0.0.2", "engine_id": "engine-2"}}
        self.assertEqual(
            KVCacheRecvingThread._get_remote_host_info_by_port(
                worker,
                base_port=36003,
                remote_handshake_port=36005,
                remote_host="10.0.0.9",
                remote_engine_id="engine-9",
                remote_multi_nodes_meta_mapping=mapping,
            ),
            ("10.0.0.2", "engine-2"),
        )

    def test_remote_host_lookup_matches_absolute_handshake_port(self):
        # Producer kv_port (36000) and consumer kv_port (36200) differ, and
        # the producer DP1 worker publishes global offset 5.
        worker = types.SimpleNamespace(
            vllm_config=types.SimpleNamespace(
                kv_transfer_config=types.SimpleNamespace(kv_port=36200)
            )
        )
        mapping = {
            "5": {
                "host": "10.0.1.5",
                "engine_id": "engine-5",
                "handshake_port": 36005,
            }
        }
        self.assertEqual(
            KVCacheRecvingThread._get_remote_host_info_by_port(
                worker,
                base_port=36003,
                remote_handshake_port=36005,
                remote_host="10.0.1.9",
                remote_engine_id="engine-9",
                remote_multi_nodes_meta_mapping=mapping,
            ),
            ("10.0.1.5", "engine-5"),
        )
