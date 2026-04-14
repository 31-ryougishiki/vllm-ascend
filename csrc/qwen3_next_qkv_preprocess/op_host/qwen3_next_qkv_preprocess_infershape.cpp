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
#include "error_log.h"
#include "log/ops_log.h"

namespace optiling {

constexpr size_t NUM_INPUTS = 8;
constexpr size_t NUM_OUTPUTS = 1;

static ge::graphStatus InferShapeAddRmsNormBias(
    const gert::TilingContext* context,
    std::vector<int64_t>& outputShape)
{
    const gert::StorageShape* qkv_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, qkv_shape);

    // Output shape is the same as input hidden dimension
    // The actual output shape depends on the model configuration
    auto numTokens = qkv_shape->GetDim(0);
    outputShape = {numTokens};

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeQwen3NextQKVPreprocess(
    const gert::TilingContext* context)
{
    OP_LOGI("InferShapeQwen3NextQKVPreprocess", "Enter InferShapeQwen3NextQKVPreprocess");

    // Check input shape
    const gert::StorageShape* qkv_shape = context->GetInputShape(0);
    const gert::StorageShape* output_shape = context->GetOutputShape(0);

    OP_CHECK_NULL_WITH_CONTEXT(context, qkv_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    // Verify dimensions
    size_t qkvDimNum = qkv_shape->GetDimNum();
    OP_CHECK_IF(
        qkvDimNum < 2,
        OP_LOGE(context, "QKV tensor must have at least 2 dimensions."),
        return ge::GRAPH_FAILED);

    // Get number of tokens from first dimension
    int64_t numTokens = qkv_shape->GetDim(0);

    // Output shape should be (numTokens, hiddenSize)
    // This is inferred from the model configuration passed through attributes
    OP_LOGI(context, "Qwen3NextQKVPreprocess infer shape: numTokens=%ld", numTokens);

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
