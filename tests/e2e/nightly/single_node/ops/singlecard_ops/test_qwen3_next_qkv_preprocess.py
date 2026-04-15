"""
E2E test for Qwen3NextQKVPreprocess operator.

Tests the NPU operator against PyTorch reference implementation for:
- QKV split with attnOutputGate=true/false
- Per-head RMSNorm
- Interleaved Rotary Embedding
- All three dtype variants (FP16/BF16/FP32)
"""

import gc
import random

import numpy as np
import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

# Test parameters
NUM_TOKENS_LIST = [1, 4, 8]
NUM_QKV_HEADS_LIST = [(8, 2), (16, 2), (32, 8)]
HEAD_DIM_LIST = [128]
EPS = [1e-6]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
SEEDS = [42, 123]
DEVICES = ["npu:0"]

# Tolerances per dtype
ATOL_RTOL = {
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1e-2, 1e-2),
    torch.float32: (1e-4, 1e-4),
}


def qwen3_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """PyTorch reference: per-head RMSNorm.

    Args:
        x: Tensor of shape [numTokens, numHeads, headDim]
        weight: Tensor of shape [headDim]
        eps: epsilon for numerical stability
    Returns:
        Normalized tensor of same shape as x
    """
    x_fp32 = x.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    # RMSNorm: rstd = 1/sqrt(mean(x^2) + eps)
    x_sq = x_fp32.pow(2)
    x_sq_mean = x_sq.mean(dim=-1, keepdim=True)
    rstd = 1.0 / torch.sqrt(x_sq_mean + eps)
    # Normalize and apply weight per head
    out = x_fp32 * rstd * weight_fp32
    return out.to(x.dtype)


def qwen3_rotary_emb_interleaved(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference: interleaved rotary embedding.

    Formula (interleaved layout):
        rotary[2i]   = x[2i]*cos[2i]   - x[2i+1]*sin[2i+1]
        rotary[2i+1] = x[2i]*sin[2i]   + x[2i+1]*cos[2i+1]

    Args:
        x: Tensor of shape [numTokens, numHeads, headDim]
        cos: cos values of shape [numPositions, headDim]
        sin: sin values of shape [numPositions, headDim]
    Returns:
        Rotated tensor of same shape as x
    """
    x_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    cos_fp32 = cos.to(torch.float32)
    sin_fp32 = sin.to(torch.float32)

    rotary_dim = cos_fp32.shape[-1]  # typically headDim or portion thereof
    assert rotary_dim * 2 == x_fp32.shape[-1], (
        f"rotary_dim={rotary_dim}, x.shape[-1]={x_fp32.shape[-1]}"
    )

    # Split into two halves: [numTokens, numHeads, rotary_dim] each
    x0 = x_fp32[..., :rotary_dim]   # first half
    x1 = x_fp32[..., rotary_dim:]   # second half

    # cos/sin are [numPositions, rotary_dim]
    # x is [numTokens, numHeads, rotary_dim]
    # Interleaved: x0 * cos - x1 * sin for even indices, x0 * sin + x1 * cos for odd
    x0_cos = x0 * cos_fp32.unsqueeze(1)   # broadcast cos over heads
    x0_sin = x0 * sin_fp32.unsqueeze(1)
    x1_cos = x1 * cos_fp32.unsqueeze(1)
    x1_sin = x1 * sin_fp32.unsqueeze(1)

    # rotary[2i] = x0[i]*cos[i] - x1[i]*sin[i]
    # rotary[2i+1] = x0[i]*sin[i] + x1[i]*cos[i]
    rotary = torch.empty_like(x_fp32)
    rotary[..., :rotary_dim] = x0_cos - x1_sin
    rotary[..., rotary_dim:] = x0_sin + x1_cos

    return rotary.to(x_dtype)


def qwen3_next_qkv_preprocess_reference(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_cos: torch.Tensor,
    q_sin: torch.Tensor,
    k_cos: torch.Tensor,
    k_sin: torch.Tensor,
    gate: torch.Tensor,
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_size: int,
    kv_size: int,
    attn_output_gate: bool,
    epsilon: float = 1e-6,
) -> tuple:
    """PyTorch reference implementation of qwen3_next QKV preprocessing.

    Args:
        qkv: Input tensor of shape [numTokens, qkvTotalSize]
             Layout: [q(qSize), k(kvSize), v(kvSize)]
        q_weight: Q RMSNorm weight of shape [headDim]
        k_weight: K RMSNorm weight of shape [headDim]
        q_cos: Q cos of shape [numTokens, headDim]
        q_sin: Q sin of shape [numTokens, headDim]
        k_cos: K cos of shape [numTokens, headDim]
        k_sin: K sin of shape [numTokens, headDim]
        gate: Gate tensor of shape [numTokens, qSize] when attn_output_gate=True, else empty
        attn_output_gate: If True, gate is passed separately
                         If False, gate is empty tensor
    Returns:
        (q_out, k_out, v_out, gate_out) tensors
    """
    # Split qkv into [q, k, v]
    q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)

    # Reshape for per-head RMSNorm: [numTokens, numHeads, headDim]
    q_reshaped = q.reshape(num_tokens, num_heads, head_dim)
    k_reshaped = k.reshape(num_tokens, num_kv_heads, head_dim)

    # Apply RMSNorm per head
    q_normed = qwen3_rms_norm(q_reshaped, q_weight, epsilon)
    k_normed = qwen3_rms_norm(k_reshaped, k_weight, epsilon)

    # Apply interleaved rotary embedding
    q_rot = qwen3_rotary_emb_interleaved(q_normed, q_cos, q_sin)
    k_rot = qwen3_rotary_emb_interleaved(k_normed, k_cos, k_sin)

    # Reshape back to flat
    q_out = q_rot.reshape(num_tokens, q_size)
    k_out = k_rot.reshape(num_tokens, kv_size)
    v_out = v  # V is pass-through

    return q_out, k_out, v_out, gate


def qwen3_next_qkv_preprocess_npu(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_cos: torch.Tensor,
    q_sin: torch.Tensor,
    k_cos: torch.Tensor,
    k_sin: torch.Tensor,
    gate: torch.Tensor,
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_size: int,
    kv_size: int,
    qkv_size: int,
    attn_output_gate: bool,
    epsilon: float = 1e-6,
) -> tuple:
    """Call NPU qwen3_next_qkv_preprocess operator.

    Registered in torch_binding.cpp as torch.ops._C_ascend.npu_qwen3_next_qkv_preprocess.

    Input layout:
        qkv: [q(qSize), k(kvSize), v(kvSize)]
        gate: [qSize] when attn_output_gate=True, empty tensor otherwise
    """
    return torch.ops._C_ascend.npu_qwen3_next_qkv_preprocess(
        qkv,
        q_weight,
        k_weight,
        q_cos,
        q_sin,
        k_cos,
        k_sin,
        gate,
        epsilon,
        num_tokens,
        num_heads,
        num_kv_heads,
        head_dim,
        q_size,
        kv_size,
        qkv_size,
        attn_output_gate,
    )


@pytest.mark.parametrize("num_tokens", NUM_TOKENS_LIST)
@pytest.mark.parametrize("num_q_heads, num_kv_heads", NUM_QKV_HEADS_LIST)
@pytest.mark.parametrize("head_dim", HEAD_DIM_LIST)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("attn_output_gate", [True, False])
@torch.inference_mode()
def test_qwen3_next_qkv_preprocess(
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    attn_output_gate: bool,
    epsilon: float = 1e-6,
):
    """Test qwen3_next_qkv_preprocess NPU operator against PyTorch reference."""
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)

    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    # qkv layout: [q(qSize), k(kvSize), v(kvSize)]
    qkv_size = q_size + kv_size * 2

    atol, rtol = ATOL_RTOL[dtype]

    # Create input tensors on NPU
    # qkv: [q, k, v] layout
    qkv = torch.randn(num_tokens, qkv_size, dtype=dtype, device=device)
    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)

    # cos/sin: [numPositions, headDim]
    q_cos = torch.randn(num_tokens, head_dim, dtype=dtype, device=device)
    q_sin = torch.randn(num_tokens, head_dim, dtype=dtype, device=device)
    k_cos = torch.randn(num_tokens, head_dim, dtype=dtype, device=device)
    k_sin = torch.randn(num_tokens, head_dim, dtype=dtype, device=device)

    # gate: separate tensor when attn_output_gate=True
    if attn_output_gate:
        gate = torch.randn(num_tokens, q_size, dtype=dtype, device=device)
    else:
        gate = torch.empty(0, dtype=dtype, device=device)

    # Reference result (computed on CPU/NPU)
    q_ref, k_ref, v_ref, gate_ref = qwen3_next_qkv_preprocess_reference(
        qkv,
        q_weight,
        k_weight,
        q_cos,
        q_sin,
        k_cos,
        k_sin,
        gate,
        num_tokens,
        num_q_heads,
        num_kv_heads,
        head_dim,
        q_size,
        kv_size,
        attn_output_gate,
        epsilon,
    )

    # NPU result
    q_npu, k_npu, v_npu, gate_npu = qwen3_next_qkv_preprocess_npu(
        qkv,
        q_weight,
        k_weight,
        q_cos,
        q_sin,
        k_cos,
        k_sin,
        gate,
        num_tokens,
        num_q_heads,
        num_kv_heads,
        head_dim,
        q_size,
        kv_size,
        qkv_size,
        attn_output_gate,
        epsilon,
    )

    # Verify Q output
    torch.testing.assert_close(
        q_npu.to(torch.float32).cpu(),
        q_ref.to(torch.float32),
        atol=atol,
        rtol=rtol,
        msg=f"Q output mismatch (attnOutputGate={attn_output_gate}, dtype={dtype})",
    )

    # Verify K output
    torch.testing.assert_close(
        k_npu.to(torch.float32).cpu(),
        k_ref.to(torch.float32),
        atol=atol,
        rtol=rtol,
        msg=f"K output mismatch (attnOutputGate={attn_output_gate}, dtype={dtype})",
    )

    # Verify V output (pass-through)
    # V is at offset q_size + kv_size in the [q, k, v] layout
    v_expected = qkv[..., q_size + kv_size:q_size + kv_size * 2]
    torch.testing.assert_close(
        v_npu.to(torch.float32).cpu(),
        v_expected.to(torch.float32),
        atol=atol,
        rtol=rtol,
        msg=f"V output mismatch (attnOutputGate={attn_output_gate}, dtype={dtype})",
    )

    # Verify gate output (only when attnOutputGate=True)
    if attn_output_gate:
        torch.testing.assert_close(
            gate_npu.to(torch.float32).cpu(),
            gate.to(torch.float32),
            atol=atol,
            rtol=rtol,
            msg=f"Gate output mismatch (dtype={dtype})",
        )
    else:
        assert gate_npu.shape == (0,) or gate_npu.numel() == 0, (
            f"Gate should be empty when attnOutputGate=False, got shape {gate_npu.shape}"
        )

    gc.collect()
    torch.npu.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
