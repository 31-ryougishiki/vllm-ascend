/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef QWEN3_NEXT_QKV_PREPROCESS_TORCH_ADPT_H
#define QWEN3_NEXT_QKV_PREPROCESS_TORCH_ADPT_H

namespace vllm_ascend {

/**
 * @brief Qwen3Next QKV Preprocessing Operator
 *
 * Implements qwen3_next.py lines 918-937:
 * 1. QKV split based on attn_output_gate
 * 2. Q RMSNorm
 * 3. K RMSNorm
 * 4. Rotary Embedding
 *
 * Input layout (matches vLLM's qkv_proj output):
 *   attnOutputGate=true:  qkv=[q_gate(2*qSize), k(kvSize), v(kvSize)]
 *   attnOutputGate=false: qkv=[q(qSize), k(kvSize), v(kvSize)]
 *
 * Returns Q, K, V tensors (and gate if attn_output_gate is enabled)
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_qwen3_next_qkv_preprocess(
    const at::Tensor& qkv,
    const at::Tensor& qNormWeight,
    const at::Tensor& kNormWeight,
    const at::Tensor& qCos,
    const at::Tensor& qSin,
    const at::Tensor& kCos,
    const at::Tensor& kSin,
    double epsilon,
    int64_t numTokens,
    int64_t numHeads,
    int64_t numKvHeads,
    int64_t headDim,
    int64_t qSize,
    int64_t kvSize,
    int64_t qkvSize,
    bool attnOutputGate)
{
    // Output tensors
    at::Tensor qOut = at::empty({numTokens, qSize}, qkv.options());
    at::Tensor kOut = at::empty({numTokens, kvSize}, qkv.options());
    at::Tensor vOut = at::empty({numTokens, kvSize}, qkv.options());
    at::Tensor gateOut;

    if (attnOutputGate) {
        gateOut = at::empty({numTokens, qSize}, qkv.options());
    }

    // Call the NPU operator
    // qkv layout is passed directly from vLLM's qkv_proj output
    EXEC_NPU_CMD(aclnnQwen3NextQKVPreprocess,
                  qkv,
                  qNormWeight,
                  kNormWeight,
                  qCos,
                  qSin,
                  kCos,
                  kSin,
                  epsilon,
                  numTokens,
                  numHeads,
                  numKvHeads,
                  headDim,
                  qSize,
                  kvSize,
                  qkvSize,
                  attnOutputGate ? 1 : 0,
                  qOut,
                  kOut,
                  vOut,
                  gateOut);

    return std::make_tuple(qOut, kOut, vOut, gateOut);
}

}  // namespace vllm_ascend

#endif  // QWEN3_NEXT_QKV_PREPROCESS_TORCH_ADPT_H