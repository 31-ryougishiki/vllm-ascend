/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for the details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "qwen3_next_qkv_preprocess_kernel.h"

using namespace AscendC;

#define FUSED_ATTN_OP_IMPL(templateClass, ...) \
    do {                                       \
        templateClass<__VA_ARGS__> op(&pipe);  \
        op.Init(qkv, qNormWeight, kNormWeight, cosSinCache, \
                positions, qOut, kOut, vOut, gateOut, &tilingData);   \
        op.Process();                          \
    } while (0)

extern "C" __global__ __aicore__ void qwen3_next_qkv_preprocess(
    GM_ADDR qkv, GM_ADDR qNormWeight, GM_ADDR kNormWeight,
    GM_ADDR cosSinCache, GM_ADDR positions,
    GM_ADDR qOut, GM_ADDR kOut, GM_ADDR vOut, GM_ADDR gateOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    REGISTER_TILING_DEFAULT(Qwen3NextQKVPreprocessTilingData);
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(10)) {
        FUSED_ATTN_OP_IMPL(KernelQwen3NextQKVPreprocess, half);
    } else if (TILING_KEY_IS(20)) {
        FUSED_ATTN_OP_IMPL(KernelQwen3NextQKVPreprocess, float);
    } else if (TILING_KEY_IS(30)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        FUSED_ATTN_OP_IMPL(KernelQwen3NextQKVPreprocess, bfloat16_t);
#endif
    }
}
