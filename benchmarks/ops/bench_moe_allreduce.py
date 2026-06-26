# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Standalone benchmark for all-reduce on moe_ar tensors dumped during inference.

Reproduces the all-reduce communication that occurs inside the ``moe_ar`` timer
in vllm-ascend when using ALLGATHER MoE mode (the only path where
``tensor_model_parallel_all_reduce`` is called instead of being a no-op).

Usage::

    # Benchmark tp=2 (needs rank0 and rank1 dump files in the same directory)
    python benchmarks/ops/bench_moe_allreduce.py \\
        --tensor-path /tmp/moe_ar_dumps/moe_ar_input_rank0_step0_bf16_256x7168.pt \\
        --world-size 2

    # Benchmark tp=4
    python benchmarks/ops/bench_moe_allreduce.py \\
        --tensor-path /tmp/moe_ar_dumps/moe_ar_input_rank0_step0_bf16_256x7168.pt \\
        --world-size 4 \\
        --warmup-iterations 20 \\
        --iterations 100
"""

import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu


def benchmark_allreduce(
    rank: int,
    world_size: int,
    tensor_path: str,
    warmup_iters: int,
    iters: int,
    port: int,
) -> None:
    """Worker function executed on each NPU rank.

    Each rank loads its own dump file (inferred by substituting the rank
    number in the base filename), moves the tensor to NPU, and runs
    ``dist.all_reduce`` in a warmup + measurement loop.

    Parameters
    ----------
    rank :
        Local rank of this worker (0 .. world_size-1).
    world_size :
        Total number of ranks (tp size).
    tensor_path :
        Path to the **rank-0** dump file.  Other ranks derive their path by
        replacing ``_rank0_`` with ``_rank{rank}_``.
    warmup_iters :
        Number of warmup iterations (not measured).
    iters :
        Number of timed iterations.
    port :
        TCP port for the ``init_process_group`` rendezvous.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
    )
    torch_npu.npu.set_device(rank)

    # Derive this rank's dump path from the rank-0 path.
    base_dir = os.path.dirname(tensor_path) or "."
    base_name = os.path.basename(tensor_path)
    rank_name = base_name.replace("_rank0_", f"_rank{rank}_")
    rank_path = os.path.join(base_dir, rank_name)

    if not os.path.exists(rank_path):
        raise FileNotFoundError(
            f"Rank {rank} dump file not found: {rank_path}.  "
            f"Expected file derived from --tensor-path by substituting "
            f"'_rank0_' → '_rank{rank}_'."
        )

    tensor = torch.load(rank_path, weights_only=True, map_location="cpu")
    tensor = tensor.npu(rank)

    def _allreduce_op() -> None:
        dist.all_reduce(tensor)

    # Warmup ----------------------------------------------------------------
    for _ in range(warmup_iters):
        _allreduce_op()
    torch_npu.npu.synchronize(rank)

    # Measurement -----------------------------------------------------------
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    times = np.zeros(iters)

    for i in range(iters):
        start.record()
        _allreduce_op()
        end.record()
        torch_npu.npu.synchronize(rank)
        times[i] = start.elapsed_time(end)  # ms

    dist.destroy_process_group()

    if rank == 0:
        data_size_bytes = tensor.element_size() * tensor.numel()
        # Ring / bidirectional all-reduce bandwidth formula
        effective_bytes = data_size_bytes * 2 * (world_size - 1) / world_size
        avg_ms = np.mean(times)
        min_ms = np.min(times)
        med_ms = np.median(times)

        print("--- All-Reduce Benchmark Results ---")
        print(f"World size (tp)  : {world_size}")
        print(f"Tensor shape     : {tuple(tensor.shape)}")
        print(f"Tensor dtype     : {tensor.dtype}")
        print(f"Data size (MB)   : {data_size_bytes / 1e6:.2f}")
        print(f"Warmup iters     : {warmup_iters}")
        print(f"Measured iters   : {iters}")
        print(f"Min   time       : {min_ms:.3f} ms")
        print(f"Avg   time       : {avg_ms:.3f} ms")
        print(f"Median time      : {med_ms:.3f} ms")
        print(f"Avg bandwidth    : "
              f"{effective_bytes / 1e6 / (avg_ms / 1e3):.2f} GB/s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark all-reduce for moe_ar tensors dumped during "
                    "vllm-ascend inference.",
    )
    parser.add_argument(
        "--tensor-path",
        type=str,
        required=True,
        help="Path to the rank-0 dump file "
             "(e.g. moe_ar_input_rank0_step0_bf16_256x7168.pt).",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        choices=[2, 4],
        help="Tensor-parallel (tp) world size: 2 or 4 (default: 2).",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=20,
        help="Number of warmup iterations (default: 20).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of measured iterations (default: 100).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=29500,
        help="TCP port for process-group rendezvous (default: 29500).",
    )
    args = parser.parse_args()

    mp.spawn(
        benchmark_allreduce,
        args=(
            args.world_size,
            args.tensor_path,
            args.warmup_iterations,
            args.iterations,
            args.port,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
