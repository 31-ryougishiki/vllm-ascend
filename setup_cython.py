# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Setup script to compile vllm-ascend core modules to binary (.pyd/.so).

This compiles the most critical proprietary modules to protect core
algorithms for Ascend NPU fused MoE, token dispatch, and split
attention-moe cross-group communication.

Usage:
    # Build all core modules in-place
    python setup_cython.py build_ext --inplace

    # Clean build artifacts
    python setup_cython.py clean --all

Requirements:
    pip install cython setuptools
"""

import sys
from setuptools import setup, Extension
from Cython.Build import cythonize

# ---------------------------------------------------------------------------
# Platform-specific compiler settings
# ---------------------------------------------------------------------------
is_windows = sys.platform.startswith("win")
is_linux = sys.platform.startswith("linux")

extra_compile_args = []
if is_windows:
    extra_compile_args = ["/EHsc", "/O2", "/MD"]
elif is_linux:
    extra_compile_args = ["-O3", "-march=native", "-fvisibility=hidden"]

# ---------------------------------------------------------------------------
# Extension definitions — one per compiled module
# ---------------------------------------------------------------------------
extensions = [
    # Tier 1: Fused MoE token dispatcher (3 dispatch strategies)
    Extension(
        "vllm_ascend.ops.fused_moe.token_dispatcher_core",
        sources=["vllm_ascend/ops/fused_moe/token_dispatcher_core.pyx"],
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
    # Tier 1: Fused MoE core implementation
    Extension(
        "vllm_ascend.ops.fused_moe.fused_moe_core",
        sources=["vllm_ascend/ops/fused_moe/fused_moe_core.pyx"],
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
    # Tier 1: Split attn/moe cross-group communication
    Extension(
        "vllm_ascend.distributed.split_attn_moe_communicator_core",
        sources=[
            "vllm_ascend/distributed/split_attn_moe_communicator_core.pyx"
        ],
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
]

# ---------------------------------------------------------------------------
# Cython compiler directives
# ---------------------------------------------------------------------------
cython_directives = {
    "language_level": "3",          # Python 3 semantics
    "embedsignature": True,         # Preserve function signatures for debugging
    "boundscheck": False,           # Skip bounds checking (performance)
    "wraparound": False,            # Skip negative index wrapping
}

compiled_modules = cythonize(
    extensions,
    compiler_directives=cython_directives,
    nthreads=4,                     # Parallel compilation
)

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
setup(
    name="vllm_ascend_core",
    version="1.0.0",
    description="Cython-compiled core modules for vllm-ascend",
    ext_modules=compiled_modules,
    zip_safe=False,
)
