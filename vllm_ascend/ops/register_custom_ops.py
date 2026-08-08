import torch
import torch.nn.functional as F
import torch_npu
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.ops.rotary_embedding import rope_forward_oot
from vllm_ascend.ops.triton.muls_add import muls_add_triton
from vllm_ascend.ops.weight_prefetch import maybe_npu_prefetch
from vllm_ascend.utils import enable_sp_by_pass, is_vl_model, npu_stream_switch, prefetch_stream


def _maybe_chunk_residual_impl(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    try:
        get_forward_context()
    except AssertionError:
        return residual

    if x.size(0) != residual.size(0):
        pad_size = _EXTRA_CTX.pad_size
        if pad_size > 0:
            residual = F.pad(residual, (0, 0, 0, pad_size))
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        residual = torch.chunk(residual, tp_size, dim=0)[tp_rank]

    return residual


def _maybe_all_gather_and_maybe_unpad_impl(x: torch.Tensor, label: bool, is_ep_comm: bool = False) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return x

    flash_comm_v1_enabled = _EXTRA_CTX.flash_comm_v1_enabled or (enable_sp_by_pass() and is_ep_comm)
    if flash_comm_v1_enabled and label:
        dp_metadata = forward_context.dp_metadata
        if dp_metadata is None or not is_ep_comm:
            import os as _hdos
            from vllm_ascend.ascend_forward_context import get_hetero_dump_dir

            _hdd = get_hetero_dump_dir()
            _first = _hdd and not globals().get("_TP_GATHER_DUMPED")
            if _first:
                globals()["_TP_GATHER_DUMPED"] = True
                torch.save(x.detach().cpu(), _hdos.path.join(_hdd, "tp_gather_in.pt"))
                torch.save(torch.tensor([x.shape[0], _EXTRA_CTX.pad_size]),
                           _hdos.path.join(_hdd, "tp_gather_meta.pt"))
            x = tensor_model_parallel_all_gather(x, 0)
            pad_size = _EXTRA_CTX.pad_size
            if pad_size > 0:
                x = x[:-pad_size]
            if _first:
                torch.save(x.detach().cpu(), _hdos.path.join(_hdd, "tp_gather_out.pt"))
        else:
            num_tokens_across_dp_cpu = dp_metadata.num_tokens_across_dp_cpu
            per_dp = getattr(_EXTRA_CTX, 'per_dp_padded_lengths', None)
            tp_sizes = getattr(_EXTRA_CTX, 'per_dp_tp_sizes', None)
            if per_dp is not None and tp_sizes is not None:
                # Heterogeneous TP: after sequence_parallel_chunk each DP rank
                # holds ceil(N_dp / tp_dp) tokens, which differs across DP
                # groups. The EP all_gather requires every rank to contribute
                # the SAME number of rows, so pad each rank to a uniform
                # per-rank size first, then unpad rank-by-rank (the homogeneous
                # code walks per-DP strides over a uniform layout).
                uniform_rank = max(
                    (per_dp[i] + tp_sizes[i] - 1) // tp_sizes[i]
                    for i in range(len(tp_sizes))
                )
                if x.shape[0] < uniform_rank:
                    # Pad only dim 0 (token dim); tensors here may be 1-D
                    # (e.g. pertoken_scale) or 2-D+ (hidden states).
                    pad = uniform_rank - x.shape[0]
                    x = F.pad(x, (0, 0) * (x.dim() - 1) + (0, pad))
                x = get_ep_group().all_gather(x, 0)
                if enable_sp_by_pass():  # TODO: do unpad
                    return x
                result = torch.empty(
                    (num_tokens_across_dp_cpu.sum(), *x.shape[1:]),
                    device=x.device,
                    dtype=x.dtype,
                )
                result_offset = 0
                x_offset = 0
                for i in range(len(tp_sizes)):
                    actual_i = int(num_tokens_across_dp_cpu[i].item())
                    chunk = per_dp[i] // tp_sizes[i]
                    for r in range(tp_sizes[i]):
                        start = r * chunk
                        if start >= actual_i:
                            break
                        n = min(chunk, actual_i - start)
                        result[result_offset : result_offset + n] = x[x_offset : x_offset + n]
                        result_offset += n
                        x_offset += uniform_rank
                return result
            x = get_ep_group().all_gather(x, 0)
            if enable_sp_by_pass():  # TODO: do unpad
                return x
            # unpad
            result = torch.empty((num_tokens_across_dp_cpu.sum(), *x.shape[1:]), device=x.device, dtype=x.dtype)
            dp_size = get_dp_group().world_size
            x = x.view(dp_size, _EXTRA_CTX.padded_length, *x.shape[1:])
            offset = 0
            for idx in range(dp_size):
                num_tokens_dp = num_tokens_across_dp_cpu[idx]
                result[offset : offset + num_tokens_dp] = x[idx, :num_tokens_dp]
                offset += num_tokens_dp
            x = result

    return x


def _maybe_pad_and_reduce_impl(x: torch.Tensor, is_ep_comm: bool = False) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return tensor_model_parallel_all_reduce(x)

    flash_comm_v1_enabled = getattr(forward_context, "flash_comm_v1_enabled", False) or (
        enable_sp_by_pass() and is_ep_comm
    )

    if not flash_comm_v1_enabled or (forward_context.is_draft_model and is_vl_model() and not is_ep_comm):
        return tensor_model_parallel_all_reduce(x)

    dp_metadata = forward_context.dp_metadata
    if dp_metadata is None or not is_ep_comm:
        pad_size = _EXTRA_CTX.pad_size
        if pad_size > 0:
            x = F.pad(x, (0, 0, 0, pad_size))
        # Under heterogeneous TP, pad_size may have been aligned to a
        # different DP rank's tp_size (via per_dp_padded_lengths).
        # Ensure the result is divisible by the LOCAL tp_size before
        # reduce_scatter, or the communicator will assert-fail.
        tp_size = get_tensor_model_parallel_world_size()
        if x.shape[0] % tp_size != 0:
            extra = tp_size - (x.shape[0] % tp_size)
            x = F.pad(x, (0, 0, 0, extra))
        return tensor_model_parallel_reduce_scatter(x, 0)
    else:
        num_tokens_across_dp_cpu = get_forward_context().dp_metadata.num_tokens_across_dp_cpu
        per_dp = getattr(_EXTRA_CTX, 'per_dp_padded_lengths', None)
        tp_sizes = getattr(_EXTRA_CTX, 'per_dp_tp_sizes', None)
        if per_dp is not None and tp_sizes is not None:
            if enable_sp_by_pass():
                return get_ep_group().reduce_scatter(x.view(-1, *x.shape[1:]), 0)
            # Heterogeneous TP: reduce_scatter hands every rank the SAME
            # number of rows, but each DP rank must receive its own
            # ceil(N_dp / tp_dp) share. Pack each rank's real share into a
            # uniform per-rank slot, reduce_scatter, then slice the local
            # share.
            uniform_rank = max(
                (per_dp[i] + tp_sizes[i] - 1) // tp_sizes[i]
                for i in range(len(tp_sizes))
            )
            ep_world_size = get_ep_group().world_size
            padded_x = torch.empty(
                (ep_world_size * uniform_rank, *x.shape[1:]),
                device=x.device,
                dtype=x.dtype,
            )
            x_offset = 0
            padded_offset = 0
            for i in range(len(tp_sizes)):
                actual_i = int(num_tokens_across_dp_cpu[i].item())
                chunk = per_dp[i] // tp_sizes[i]
                for r in range(tp_sizes[i]):
                    start = r * chunk
                    if start >= actual_i:
                        break
                    n = min(chunk, actual_i - start)
                    padded_x[padded_offset : padded_offset + n] = x[x_offset : x_offset + n]
                    x_offset += n
                    padded_offset += uniform_rank
            x = get_ep_group().reduce_scatter(padded_x, 0)
            # Slice this rank's real share from its uniform slot.
            ep_rank = get_ep_group().rank_in_group
            i = 0
            while ep_rank >= tp_sizes[i]:
                ep_rank -= tp_sizes[i]
                i += 1
            local_dp, local_tp = i, ep_rank
            chunk_local = per_dp[local_dp] // tp_sizes[local_dp]
            actual_local = int(num_tokens_across_dp_cpu[local_dp].item())
            n_local = min(chunk_local, actual_local - local_tp * chunk_local)
            return x[: max(n_local, 0)]
        if enable_sp_by_pass():
            return get_ep_group().reduce_scatter(x.view(-1, *x.shape[1:]), 0)
        # padding
        dp_size = get_dp_group().world_size
        padded_x = torch.empty((dp_size, _EXTRA_CTX.padded_length, *x.shape[1:]), device=x.device, dtype=x.dtype)
        offset = 0
        for idx in range(dp_size):
            num_tokens_dp = num_tokens_across_dp_cpu[idx]
            padded_x[idx, :num_tokens_dp] = x[offset : offset + num_tokens_dp]
            offset += num_tokens_dp
        padded_x = padded_x.view(-1, *x.shape[1:])

        return get_ep_group().reduce_scatter(padded_x, 0)


def _maybe_all_gather_and_maybe_unpad_fake(x: torch.Tensor, label: bool, is_ep_comm: bool = False) -> torch.Tensor:
    if _EXTRA_CTX.flash_comm_v1_enabled and label:
        return torch.empty(
            (x.shape[0] * get_tensor_model_parallel_world_size(), *x.shape[1:]), device=x.device, dtype=x.dtype
        )

    return x


def _maybe_pad_and_reduce_fake(x: torch.Tensor, is_ep_comm: bool = False) -> torch.Tensor:
    if _EXTRA_CTX.flash_comm_v1_enabled or enable_sp_by_pass():
        return torch.empty(
            (x.shape[0] // get_tensor_model_parallel_world_size(), *x.shape[1:]), device=x.device, dtype=x.dtype
        )

    return x


def _prefetch_preprocess_impl(weight: torch.Tensor, start_flag: torch.Tensor, max_weight_size: int) -> None:
    calculation_stream = torch_npu.npu.current_stream()
    weight_prefetch_stream = prefetch_stream()
    weight_prefetch_stream.wait_stream(calculation_stream)
    with npu_stream_switch(weight_prefetch_stream):
        maybe_npu_prefetch(inputs=weight, dependency=start_flag, max_size=max_weight_size)


def _prefetch_preprocess_impl_fake(weight: torch.Tensor, start_flag: torch.Tensor, max_weight_size: int) -> None:
    return


def _prefetch_postprocess_impl(stop_flag: torch.Tensor) -> None:
    calculation_stream = torch_npu.npu.current_stream()
    weight_prefetch_stream = prefetch_stream()
    calculation_stream.wait_stream(weight_prefetch_stream)


def _prefetch_postprocess_impl_fake(stop_flag: torch.Tensor) -> None:
    return


def _maybe_all_reduce_tensor_model_parallel_impl(final_hidden_states: torch.Tensor) -> torch.Tensor:
    moe_comm_type = _EXTRA_CTX.moe_comm_type
    if (
        moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2}
        or _EXTRA_CTX.flash_comm_v1_enabled
    ):
        return final_hidden_states
    else:
        return tensor_model_parallel_all_reduce(final_hidden_states)


def _matmul_and_reduce_impl(input_parallel: torch.Tensor, layer_name: str) -> torch.Tensor:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    assert self.custom_op is not None
    bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
    output = self.custom_op.matmul_and_reduce(input_parallel, bias_)

    return output


def _matmul_and_reduce_impl_fake(input_parallel: torch.Tensor, layer_name: str) -> torch.Tensor:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    num_tokens = input_parallel.size(0)
    if _EXTRA_CTX.flash_comm_v1_enabled:
        num_tokens = num_tokens // self.tp_size
    output = torch.empty(
        size=(num_tokens, self.output_size_per_partition), device=input_parallel.device, dtype=input_parallel.dtype
    )

    return output


# TODO(Angazenn): The reason why we use a custom op to encapsulate npu_quantize
# is that aclnnAscendQuantV3(npu_quantize) use div_mode=False, while
# aclnnAddRmsNormQuantV2(npu_add_rms_norm_quant) use div_moe=True. We have to
# pass input_scale and input_scale_reciprocal at the same time to avoid redundant
# reciprocal calculation in fussion pass. We shall remove this once
# aclnnAddRmsNormQuantV2 supports div_moe=False.
def _quantize_impl(
    in_tensor: torch.Tensor, input_scale: torch.Tensor, input_scale_reciprocal: torch.Tensor, input_offset: torch.Tensor
) -> torch.Tensor:
    return torch_npu.npu_quantize(in_tensor, input_scale_reciprocal, input_offset, torch.qint8, -1, False)


def _quantize_impl_fake(
    in_tensor: torch.Tensor, input_scale: torch.Tensor, input_scale_reciprocal: torch.Tensor, input_offset: torch.Tensor
) -> torch.Tensor:
    return torch_npu.npu_quantize(in_tensor, input_scale_reciprocal, input_offset, torch.qint8, -1, False)


def _rope_forward_oot_impl_fake(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    return query, key


def _muls_add_impl_fake(
    x: torch.Tensor,
    y: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="maybe_chunk_residual",
    op_func=_maybe_chunk_residual_impl,
    fake_impl=lambda x, residual: torch.empty_like(x),
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="maybe_all_gather_and_maybe_unpad",
    op_func=_maybe_all_gather_and_maybe_unpad_impl,
    fake_impl=_maybe_all_gather_and_maybe_unpad_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="maybe_pad_and_reduce",
    op_func=_maybe_pad_and_reduce_impl,
    fake_impl=_maybe_pad_and_reduce_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="prefetch_preprocess",
    op_func=_prefetch_preprocess_impl,
    fake_impl=_prefetch_preprocess_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="prefetch_postprocess",
    op_func=_prefetch_postprocess_impl,
    fake_impl=_prefetch_postprocess_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="maybe_all_reduce_tensor_model_parallel",
    op_func=_maybe_all_reduce_tensor_model_parallel_impl,
    fake_impl=lambda x: x,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="matmul_and_reduce",
    op_func=_matmul_and_reduce_impl,
    fake_impl=_matmul_and_reduce_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="quantize",
    op_func=_quantize_impl,
    fake_impl=_quantize_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="npu_rotary_embedding",
    op_func=rope_forward_oot,
    fake_impl=_rope_forward_oot_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="muls_add",
    op_func=muls_add_triton,
    fake_impl=_muls_add_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
