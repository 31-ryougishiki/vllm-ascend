/**
 * This program is free software; you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for the details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qwen3_next_qkv_preprocess_def.cpp
 * \brief Op definition for Qwen3NextQKVPreprocess
 */

#include "register/op_def_registry.h"

namespace ops {
class Qwen3NextQKVPreprocess : public OpDef {
public:
    explicit Qwen3NextQKVPreprocess(const char* name) : OpDef(name)
    {
        this->Input("qkv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("qNormWeight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("kNormWeight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("qCos")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("qSin")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("kCos")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("kSin")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gate")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("qOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("kOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("vOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("gateOut")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-6);
        this->Attr("numTokens").AttrType(OPTIONAL).Int(0);
        this->Attr("numHeads").AttrType(OPTIONAL).Int(0);
        this->Attr("numKvHeads").AttrType(OPTIONAL).Int(0);
        this->Attr("headDim").AttrType(OPTIONAL).Int(0);
        this->Attr("qSize").AttrType(OPTIONAL).Int(0);
        this->Attr("kvSize").AttrType(OPTIONAL).Int(0);
        this->Attr("qkvSize").AttrType(OPTIONAL).Int(0);
        this->Attr("attnOutputGate").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("coreType.value", "AiCore");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
    }
};
OP_ADD(Qwen3NextQKVPreprocess);

} // namespace ops