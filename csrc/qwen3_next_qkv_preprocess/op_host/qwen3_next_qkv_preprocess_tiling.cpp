/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for the details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR
 * FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/*!
 * \file qwen3_next_qkv_preprocess_tiling.cpp
 * \brief Tiling strategy for Qwen3NextQKVPreprocess
 */

#include "log/ops_log.h"
#include "../tiling_base/tiling_templates_registry.h"
#include "../tiling_base/tiling_util.h"
#include "../tiling_base/error_log.h"
#include "tiling/qwen3_next_qkv_preprocess_tiling.h"

namespace optiling {

constexpr uint32_t DTYPE_KEY_FP16 = 1;
constexpr uint32_t DTYPE_KEY_FP32 = 2;
constexpr uint32_t DTYPE_KEY_BF16 = 3;

constexpr uint32_t BLOCK_ALIGN_NUM = 16;
constexpr uint32_t FLOAT_BLOCK_ALIGN_NUM = 8;

platform_ascendc::SocVersion qwen3NextQKVPreprocessSocVersion;

static void SetByDtype(ge::DataType dataType, uint32_t& dtypeKey, uint32_t& dataPerBlock)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            dtypeKey = DTYPE_KEY_FP16;
            dataPerBlock = BLOCK_ALIGN_NUM;
            break;
        case ge::DT_BF16:
            dtypeKey = DTYPE_KEY_BF16;
            dataPerBlock = BLOCK_ALIGN_NUM;
            break;
        default:
            dtypeKey = DTYPE_KEY_FP32;
            dataPerBlock = FLOAT_BLOCK_ALIGN_NUM;
            break;
    }
}

static bool CheckInputOutputShape(const gert::TilingContext* context)
{
    const gert::StorageShape* qkv_shape = context->GetInputShape(0);
    const gert::StorageShape* output_shape = context->GetOutputShape(0);

    OP_CHECK_NULL_WITH_CONTEXT(context, qkv_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    size_t qkvDimNum = qkv_shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        qkvDimNum < 2,
        OP_LOGE(context, "QKV tensor must have at least 2 dimensions."),
        return false);

    return true;
}

static void GetCompileParameters(
    gert::TilingContext* context, uint32_t& numCore, uint64_t& ubSize)
{
    auto ptrCompileInfo = reinterpret_cast<const Qwen3NextQKVPreprocessCompileInfo*>(context->GetCompileInfo());
    if (ptrCompileInfo == nullptr) {
        auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        qwen3NextQKVPreprocessSocVersion = ascendc_platform.GetSocVersion();
        numCore = ascendc_platform.GetCoreNumAiv();
        ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    } else {
        numCore = ptrCompileInfo->totalCoreNum;
        ubSize = ptrCompileInfo->totalUbSize;
        qwen3NextQKVPreprocessSocVersion = ptrCompileInfo->socVersion;
    }
}

static void CalculateTilingParameters(
    gert::TilingContext* context,
    Qwen3NextQKVPreprocessTilingData* tiling,
    uint32_t numCore)
{
    const gert::Shape qkv_shape = context->GetInputShape(0)->GetStorageShape();
    const gert::Shape output_shape = context->GetOutputShape(0)->GetStorageShape();

    uint32_t numTokens = qkv_shape.GetDim(0);
    uint32_t qkvSize = qkv_shape.GetDim(qkv_shape.GetDimNum() - 1);
    uint32_t hiddenSize = output_shape.GetDim(output_shape.GetDimNum() - 1);

    // Read all shape and attribute parameters from op attributes
    auto attrs = context->GetAttrs();

    float epsilon = 1e-6f;
    uint32_t numHeads = 0;
    uint32_t numKvHeads = 0;
    uint32_t headDim = 0;
    uint32_t qSize = 0;
    uint32_t kvSize = 0;
    uint32_t attnOutputGate = 0;

    if (attrs != nullptr) {
        epsilon = *attrs->GetAttrPointer<float>(0);        // epsilon
        numTokens = *attrs->GetAttrPointer<int64_t>(1);    // numTokens
        numHeads = *attrs->GetAttrPointer<int64_t>(2);     // numHeads
        numKvHeads = *attrs->GetAttrPointer<int64_t>(3);   // numKvHeads
        headDim = *attrs->GetAttrPointer<int64_t>(4);      // headDim
        qSize = *attrs->GetAttrPointer<int64_t>(5);        // qSize
        kvSize = *attrs->GetAttrPointer<int64_t>(6);       // kvSize
        attnOutputGate = *attrs->GetAttrPointer<int64_t>(8); // attnOutputGate
    }

    uint32_t tokensPerCore = CeilDiv(numTokens, numCore);

    tiling->set_num_tokens(numTokens);
    tiling->set_num_heads(numHeads);
    tiling->set_num_kv_heads(numKvHeads);
    tiling->set_head_dim(headDim);
    tiling->set_q_size(qSize);
    tiling->set_kv_size(kvSize);
    tiling->set_hidden_size(hiddenSize);
    tiling->set_qkv_size(qSize * 2 + kvSize * 2);  // attnOutputGate=true layout
    tiling->set_block_dim(numCore);
    tiling->set_attn_output_gate(attnOutputGate);
    tiling->set_epsilon(epsilon);

    OPS_LOG_I(context, "Tiling: numTokens=%u, numHeads=%u, numKvHeads=%u, headDim=%u, "
              "qSize=%u, kvSize=%u, qkvSize=%u, attnOutputGate=%u, blockDim=%u",
              numTokens, numHeads, numKvHeads, headDim, qSize, kvSize,
              tiling->get_qkv_size(), attnOutputGate, numCore);
}

static void SaveTilingData(
    gert::TilingContext* context, Qwen3NextQKVPreprocessTilingData* tiling, uint32_t dtypeKey)
{
    uint32_t tilingKey = dtypeKey * 10;
    context->SetTilingKey(tilingKey);
    uint8_t* tilingData = reinterpret_cast<uint8_t*>(context->GetRawTilingData()->GetData());
    tiling->SaveToBuffer(tilingData, context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling->GetDataSize());
}

static void SetWorkspaceSize(gert::TilingContext* context)
{
    constexpr size_t sysWorkspaceSize = 16 * 1024 * 1024;  // 16MB minimum
    constexpr size_t usrSize = 256;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
}

static void LogTilingResults(
    gert::TilingContext* context, Qwen3NextQKVPreprocessTilingData* tiling, uint32_t dtypeKey)
{
    OPS_LOG_I(context, "Tiling Key: %u", dtypeKey * 10);
    OPS_LOG_I(context, "Block Dim: %u", tiling->get_block_dim());
    OPS_LOG_I(context, "numTokens: %u, numHeads: %u, numKvHeads: %u, headDim: %u, "
              "qSize: %u, kvSize: %u, attnOutputGate: %u, epsilon: %f",
              tiling->get_num_tokens(), tiling->get_num_heads(), tiling->get_num_kv_heads(),
              tiling->get_head_dim(), tiling->get_q_size(), tiling->get_kv_size(),
              tiling->get_attn_output_gate(), tiling->get_epsilon());
}

static ge::graphStatus Tiling4Qwen3NextQKVPreprocess(gert::TilingContext* context)
{
    OP_LOGI("Tiling4Qwen3NextQKVPreprocess", "Enter Tiling4Qwen3NextQKVPreprocess");
    OPS_LOG_D(context, "Tiling4Qwen3NextQKVPreprocess running.");

    OP_CHECK_IF(
        !CheckInputOutputShape(context),
        OP_LOGE(context, "Input shape invalid."),
        return ge::GRAPH_FAILED);

    Qwen3NextQKVPreprocessTilingData tiling;

    uint32_t numCore;
    uint64_t ubSize;
    GetCompileParameters(context, numCore, ubSize);

    uint32_t dtypeKey;
    uint32_t dataPerBlock;
    ge::DataType dataType = context->GetInputDesc(0)->GetDataType();
    SetByDtype(dataType, dtypeKey, dataPerBlock);

    CalculateTilingParameters(context, &tiling, numCore);

    SaveTilingData(context, &tiling, dtypeKey);
    SetWorkspaceSize(context);
    LogTilingResults(context, &tiling, dtypeKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4Qwen3NextQKVPreprocess(gert::TilingParseContext* context)
{
    OPS_LOG_D(context, "TilingPrepare4Qwen3NextQKVPreprocess running.");
    OP_LOGI(context, "TilingPrepare4Qwen3NextQKVPreprocess running.");
    auto compileInfo = context->GetCompiledInfo<Qwen3NextQKVPreprocessCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->socVersion = ascendcPlatform.GetSocVersion();
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->totalUbSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Qwen3NextQKVPreprocess)
    .Tiling(Tiling4Qwen3NextQKVPreprocess)
    .TilingParse<Qwen3NextQKVPreprocessCompileInfo>(TilingPrepare4Qwen3NextQKVPreprocess);

}  // namespace optiling
