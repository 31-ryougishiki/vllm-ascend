/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for the details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR
 * PURPOSE. See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_TILING_ERROR_LOG_H_
#define OPS_BUILT_IN_OP_TILING_ERROR_LOG_H_

#include <string>
#include "toolchain/slog.h"

#define OP_LOGI(opname, ...)
#define OP_LOGW(opname, ...)             \
    do {                                 \
        printf("[WARN][%s] ", (opname), ##__VA_ARGS__); \
        printf("\n");                    \
    } while (0)

#define OP_LOGE_WITHOUT_REPORT(opname, ...) \
    do {                                    \
        printf("[ERRORx][%s] ", (opname), ##__VA_ARGS__);  \
        printf("\n");                       \
    } while (0)

#define OP_LOGE(opname, ...)              \
    do {                                  \
        printf("[ERROR][%s] ", (opname), ##__VA_ARGS__);    \
        printf("\n");                     \
    } while (0)

#define OP_LOGD(opname, ...)

namespace optiling {

#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...)   \
    do {                                                         \
        OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__); \
    } while (0)


#define OP_CHECK_IF(cond, log_func, expr) \
    do {                                      \
        if (cond) {                           \
            log_func;                         \
            expr;                             \
        }                                     \
    } while (0)


#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)                          \
    do {                                                                  \
        if ((ptr) == nullptr) {                                           \
            OP_LOGE(context->GetNodeType(), "%s is null", #ptr);               \
            return ge::GRAPH_FAILED;                                      \
        }                                                                 \
    } while (0)

}  // namespace optiling

template <typename T>
T CeilAlign(T a, T b)
{
    return (a + b - 1) / b * b;
}

template <typename T>
T CeilDiv(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

#endif  // OPS_BUILT_IN_OP_TILING_ERROR_LOG_H_