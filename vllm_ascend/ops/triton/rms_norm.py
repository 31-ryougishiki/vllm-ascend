import time

import torch
from vllm.triton_utils import tl, triton

# DIAG: shapes for which the triton kernel was already launched.  A new key
# (esp. TOTAL_BATCH change from a new num_tokens) forces a triton JIT
# recompile on first use — one of the suspects for the occasional spike.
_seen_shapes: set = set()


def _diag_launch_log(is_new, key, setup_ms, alloc_ms, launch_ms):
    from vllm.logger import init_logger
    _lg = init_logger("vllm_ascend.ops.triton.rms_norm")
    _lg.warning(
        "DIAG triton_q_rms new=%s key=(total_batch=%s,dim=%s,BLOCK_M=%s,vc=%s) "
        "setup=%.1fms alloc=%.1fms launch=%.1fms",
        is_new, key[0], key[1], key[2], key[3],
        setup_ms, alloc_ms, launch_ms,
    )


@triton.jit
def triton_rms_kernel(
    hidden_state_ptr,
    hidden_state_stride_bs,
    norm_output_ptr,
    variance_epsilon,
    TOTAL_BATCH: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    core_id = tl.program_id(0)
    core_num = tl.num_programs(0)
    batch_per_core = tl.cdiv(TOTAL_BATCH, core_num)
    start_batch = core_id * batch_per_core
    end_batch = tl.minimum(start_batch + batch_per_core, TOTAL_BATCH)
    offset_d = tl.arange(0, DIM)

    for row_start in tl.range(start_batch, end_batch, BLOCK_M):
        offset_row = row_start + tl.arange(0, BLOCK_M)
        mask_r = offset_row < TOTAL_BATCH
        mask_row = mask_r[:, None]
        offset_hidden = offset_row[:, None] * hidden_state_stride_bs + offset_d[None, :]

        x = tl.load(hidden_state_ptr + offset_hidden, mask=mask_row)

        variance = tl.sum(x * x, axis=-1) / DIM
        output = x * tl.rsqrt(variance[:, None] + variance_epsilon)

        tl.store(norm_output_ptr + offset_hidden, output, mask=mask_row)


def triton_q_rms(
    q,  # bs, 64, 512
    variance_epsilon,
):
    _t0 = time.perf_counter()
    bs, head_num, dim = q.shape
    total_batch = bs * head_num
    q = q.view(total_batch, dim)

    if dim > 2048:
        raise NotImplementedError("dim > 2048 not supported")

    device_properties = triton.runtime.driver.active.utils.get_device_properties(q.device)
    num_vectorcore = device_properties.get("num_vectorcore", -1)

    ROW_BLOCK_SIZE = 16  # A safe default balancing parallelism and register pressure.
    batch_per_core = triton.cdiv(total_batch, num_vectorcore)
    BLOCK_M = min(ROW_BLOCK_SIZE, batch_per_core)

    grid = (num_vectorcore,)
    _t1 = time.perf_counter()
    norm_output = torch.empty_like(q)
    _t2 = time.perf_counter()

    triton_rms_kernel[grid](
        q,
        q.stride(0),
        norm_output,
        variance_epsilon,
        total_batch,
        dim,
        BLOCK_M,
    )
    _t3 = time.perf_counter()

    # DIAG: split the function into setup / allocation / kernel-launch host
    # time.  A large `launch` on a *new* shape => triton JIT recompile;
    # a large `alloc` => memory allocation stalling on the allocator/device.
    _key = (total_batch, dim, BLOCK_M, num_vectorcore)
    _is_new = _key not in _seen_shapes
    _seen_shapes.add(_key)
    _setup_ms = (_t1 - _t0) * 1000.0
    _alloc_ms = (_t2 - _t1) * 1000.0
    _launch_ms = (_t3 - _t2) * 1000.0
    if _is_new or max(_alloc_ms, _launch_ms) > 50.0:
        _diag_launch_log(_is_new, _key, _setup_ms, _alloc_ms, _launch_ms)

    return norm_output.view(bs, head_num, dim)
