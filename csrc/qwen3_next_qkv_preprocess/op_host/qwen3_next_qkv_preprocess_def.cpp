/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for the details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl.h"
#include "aclnn_ops.h"

// ACLNN API definition for qwen3_next_qkv_preprocess
// This is the host-side API that will be called from PyTorch

extern "C" {

/**
 * @brief Qwen3Next Fused Attention Operator
 *
 * This operator implements the fused attention computation for Qwen3Next models:
 * 1. QKV split (with optional gate splitting)
 * 2. Q RMSNorm
 * 3. K RMSNorm
 * 4. Rotary Embedding
 * 5. Attention computation
 * 6. Optional gate multiplication
 *
 * @param qkv[input] - QKV tensor after projection (numTokens, qkvSize)
 * @param qNormWeight[input] - Q normalization weight (headDim)
 * @param kNormWeight[input] - K normalization weight (headDim)
 * @param qCos[input] - Q rotary embedding cos (numTokens, headDim)
 * @param qSin[input] - Q rotary embedding sin (numTokens, headDim)
 * @param kCos[input] - K rotary embedding cos (numTokens, headDim)
 * @param kSin[input] - K rotary embedding sin (numTokens, headDim)
 * @param gate[input] - Optional gate tensor (numTokens, qSize), null if attnOutputGate is false
 * @param output[output] - Attention output (numTokens, hiddenSize)
 * @param workspace[workspace] - Workspace buffer
 * @param tiling[tiling] - Tiling data
 *
 * @return aclnnStatus - ACLNN status code
 */
aclnnStatus aclnnQwen3NextQKVPreprocess(
    const aclTensor* qkv,
    const aclTensor* qNormWeight,
    const aclTensor* kNormWeight,
    const aclTensor* qCos,
    const aclTensor* qSin,
    const aclTensor* kCos,
    const aclTensor* kSin,
    const aclTensor* gate,  // optional, can be null
    const aclScalar* epsilon,
    const aclIntArray* numTokens,
    const aclIntArray* numHeads,
    const aclIntArray* numKvHeads,
    const aclIntArray* headDim,
    const aclIntArray* qSize,
    const aclIntArray* kvSize,
    const aclIntArray* hiddenSize,
    const aclIntArray* qkvSize,
    const aclIntArray* attnOutputGate,
    aclTensor* output,
    aclBaseSpace* workspace,
    const aclTensor* tiling);
}  // extern "C"
