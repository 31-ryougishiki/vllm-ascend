/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for the details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qwen3_next_qkv_preprocess_infershape.cpp
 * \brief InferShape and InferDataType for Qwen3NextQKVPreprocess
 */

#include "register/op_def_registry.h"
#include "log/ops_log.h"

#define unlikely(x) __builtin_expect((x), 0)
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)                                              \
    do {                                                                                     \
        if (unlikely((ptr) == nullptr)) {                                                    \
            const char* name = (unlikely(((context) == nullptr) ||                           \
                                         (context)->GetNodeName() == nullptr))               \
                                   ? "nil"                                                   \
                                   : (context)->GetNodeName();                              \
            OPS_LOG_E(name, "%s is nullptr!", #ptr);                                         \
            return ge::GRAPH_FAILED;                                                         \
        }                                                                                    \
    } while (0)

// Attribute indices (matching op_def.cpp order)
static constexpr int ATTR_EPSILON = 0;
static constexpr int ATTR_NUM_TOKENS = 1;
static constexpr int ATTR_NUM_HEADS = 2;
static constexpr int ATTR_NUM_KV_HEADS = 3;
static constexpr int ATTR_HEAD_DIM = 4;
static constexpr int ATTR_Q_SIZE = 5;
static constexpr int ATTR_KV_SIZE = 6;
static constexpr int ATTR_QKV_SIZE = 7;
static constexpr int ATTR_ATTN_OUTPUT_GATE = 8;

static constexpr int NUM_OUTPUTS = 4;  // qOut, kOut, vOut, gateOut

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4Qwen3NextQKVPreprocess(gert::InferShapeContext* context)
{
    OPS_LOG_D(context, "Begin InferShape4Qwen3NextQKVPreprocess");

    // Get input shape
    const gert::Shape* qkvShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, qkvShape);

    int64_t numTokens = qkvShape->GetDim(0);

    // Get all shape parameters from op attributes
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    int64_t qSize = *attrs->GetAttrPointer<int64_t>(ATTR_Q_SIZE);
    int64_t kvSize = *attrs->GetAttrPointer<int64_t>(ATTR_KV_SIZE);
    int64_t attnOutputGate = *attrs->GetAttrPointer<int64_t>(ATTR_ATTN_OUTPUT_GATE);

    // Set output shapes:
    //   Output 0 (qOut): [numTokens, qSize]
    //   Output 1 (kOut): [numTokens, kvSize]
    //   Output 2 (vOut): [numTokens, kvSize]
    //   Output 3 (gateOut): [numTokens, qSize] if attnOutputGate=true, else []

    gert::Shape* qOutShape = context->GetOutputShape(0);
    gert::Shape* kOutShape = context->GetOutputShape(1);
    gert::Shape* vOutShape = context->GetOutputShape(2);
    gert::Shape* gateOutShape = context->GetOutputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, qOutShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, kOutShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, vOutShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gateOutShape);

    qOutShape->SetDimNum(2);
    qOutShape->SetDim(0, numTokens);
    qOutShape->SetDim(1, qSize);

    kOutShape->SetDimNum(2);
    kOutShape->SetDim(0, numTokens);
    kOutShape->SetDim(1, kvSize);

    vOutShape->SetDimNum(2);
    vOutShape->SetDim(0, numTokens);
    vOutShape->SetDim(1, kvSize);

    if (attnOutputGate != 0) {
        gateOutShape->SetDimNum(2);
        gateOutShape->SetDim(0, numTokens);
        gateOutShape->SetDim(1, qSize);
    } else {
        gateOutShape->SetDimNum(1);
        gateOutShape->SetDim(0, 0);
    }

    OPS_LOG_D(context, "InferShape done: qOut=[%ld, %ld], kOut=[%ld, %ld], "
            "vOut=[%ld, %ld], gateOut=%s",
            numTokens, qSize, numTokens, kvSize, numTokens, kvSize,
            attnOutputGate != 0 ? "[numTokens, qSize]" : "[]");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4Qwen3NextQKVPreprocess(gert::InferDataTypeContext* context)
{
    // All outputs inherit dtype from the QKV input (input 0)
    for (int i = 0; i < NUM_OUTPUTS; ++i) {
        context->SetOutputDataType(i, context->GetInputDataType(0));
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Qwen3NextQKVPreprocess)
    .InferShape(InferShape4Qwen3NextQKVPreprocess)
    .InferDataType(InferDataType4Qwen3NextQKVPreprocess);

}  // namespace ops
