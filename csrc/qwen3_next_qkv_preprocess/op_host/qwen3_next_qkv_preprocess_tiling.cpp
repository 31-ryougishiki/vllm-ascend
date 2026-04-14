/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for the details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "qwen3_next_qkv_preprocess_tiling.h"
#include "log/ops_log.h"
#include "tiling/platform/platform_ascendc.h"

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

    size_t qkvDimNum = qkv_shape->GetDimNum();
    OP_CHECK_IF(
        qkvDimNum < 2,
        OP_LOGE(context, "QKV tensor must have at least 2 dimensions."),
        return false);

    return true;
}

static void GetCompileParameters(
    gert::TilingContext* context, uint32_t& numCore, uint64_t& ubSize)
{
    auto ptrCompileInfo = reinterpret_cast<const void*>(context->GetCompileInfo());
    if (ptrCompileInfo == nullptr) {
        auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        qwen3NextQKVPreprocessSocVersion = ascendc_platform.GetSocVersion();
        numCore = ascendc_platform.GetCoreNumAiv();
        ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    } else {
        // Use default values if compile info is not available
        auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        qwen3NextQKVPreprocessSocVersion = ascendc_platform.GetSocVersion();
        numCore = ascendc_platform.GetCoreNumAiv();
        ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
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

    // These values should come from model configuration passed via attributes
    // For now, we use placeholder values that need to be set by the caller
    auto attrs = context->GetAttrs();
    float epsilon = 1e-6f;
    if (attrs != nullptr && attrs->GetSize() > 0) {
        epsilon = *attrs->GetFloat(0);
    }

    // Calculate block distribution
    uint32_t blockFactor = 1;
    uint32_t tokensPerCore = CeilDiv(numTokens, numCore);
    blockFactor = tokensPerCore;

    tiling->set_num_tokens(numTokens);
    tiling->set_block_dim(numCore);
    tiling->set_epsilon(epsilon);

    OP_LOGI(context, "Tiling: numTokens=%u, qkvSize=%u, hiddenSize=%u, blockDim=%u",
            numTokens, qkvSize, hiddenSize, numCore);
}

static void SaveTilingData(
    gert::TilingContext* context, Qwen3NextQKVPreprocessTilingData* tiling, uint32_t dtypeKey)
{
    // Use dtype key as part of tiling key
    uint32_t tilingKey = dtypeKey * 10;  // Base tiling key
    context->SetTilingKey(tilingKey);
    tiling->SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling->GetDataSize());
}

static void SetWorkspaceSize(gert::TilingContext* context)
{
    constexpr size_t sysWorkspaceSize = 16 * 1024 * 1024;  // 16MB
    constexpr size_t usrSize = 256;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
}

static void LogTilingResults(
    gert::TilingContext* context, Qwen3NextQKVPreprocessTilingData* tiling, uint32_t dtypeKey)
{
    OPS_LOG_I(context, "Tiling Key: %u", dtypeKey * 10);
    OPS_LOG_I(context, "Block Dim: %u", tiling->get_block_dim());
    OPS_LOG_I(context, "numTokens: %u, epsilon: %f",
              tiling->get_num_tokens(), tiling->get_epsilon());
}

static ge::graphStatus Tiling4Qwen3NextQKVPreprocess(gert::TilingContext* context)
{
    OP_LOGI("Tiling4Qwen3NextQKVPreprocess", "Enter Tiling4Qwen3NextQKVPreprocess");
    OPS_LOG_D(context, "Tiling4Qwen3NextQKVPreprocess running.");

    // Check input/output shape
    OP_CHECK_IF(
        !CheckInputOutputShape(context),
        OP_LOGE(context, "Input shape invalid."),
        return ge::GRAPH_FAILED);

    Qwen3NextQKVPreprocessTilingData tiling;

    uint32_t numCore;
    uint64_t ubSize;
    GetCompileParameters(context, numCore, ubSize);

    // Set data type parameters
    uint32_t dtypeKey;
    uint32_t dataPerBlock;
    ge::DataType dataType = context->GetInputDesc(0)->GetDataType();
    SetByDtype(dataType, dtypeKey, dataPerBlock);

    // Calculate tiling parameters
    CalculateTilingParameters(context, &tiling, numCore);

    // Save tiling data
    SaveTilingData(context, &tiling, dtypeKey);
    SetWorkspaceSize(context);
    LogTilingResults(context, &tiling, dtypeKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4Qwen3NextQKVPreprocess(gert::TilingParseContext* context)
{
    OPS_LOG_D(context, "TilingPrepare4Qwen3NextQKVPreprocess running.");
    OP_LOGI(context, "TilingPrepare4Qwen3NextQKVPreprocess running.");

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    qwen3NextQKVPreprocessSocVersion = ascendcPlatform.GetSocVersion();

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Qwen3NextQKVPreprocess).Tiling(Tiling4Qwen3NextQKVPreprocess).TilingParse<void>(TilingPrepare4Qwen3NextQKVPreprocess);

}  // namespace optiling
