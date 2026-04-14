/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR
 * PURPOSE. See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QWEN3_NEXT_QKV_PREPROCESS_TILING_H_
#define QWEN3_NEXT_QKV_PREPROCESS_TILING_H_

#include "register/tilingdata_base.h"
#include "error_log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Qwen3NextQKVPreprocessTilingData)
TILING_DATA_FIELD_DEF(uint32_t, numTokens);
TILING_DATA_FIELD_DEF(uint32_t, numHeads);
TILING_DATA_FIELD_DEF(uint32_t, numKvHeads);
TILING_DATA_FIELD_DEF(uint32_t, headDim);
TILING_DATA_FIELD_DEF(uint32_t, qSize);
TILING_DATA_FIELD_DEF(uint32_t, kvSize);
TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
TILING_DATA_FIELD_DEF(uint32_t, qkvSize);
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
TILING_DATA_FIELD_DEF(uint32_t, attnOutputGate);
TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

struct Qwen3NextQKVPreprocessCompileInfo {
    uint32_t totalCoreNum = 0;
    uint64_t totalUbSize = 0;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::VLLM_ASCEND_950_SOC_ENUM;
};

REGISTER_TILING_DATA_CLASS(Qwen3NextQKVPreprocess, Qwen3NextQKVPreprocessTilingData)
}  // namespace optiling

#endif  // QWEN3_NEXT_QKV_PREPROCESS_TILING_H_