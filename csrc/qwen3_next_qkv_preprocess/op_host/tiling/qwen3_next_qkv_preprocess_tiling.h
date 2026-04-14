/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for the details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR
 * PURPOSE. See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QWEN3_NEXT_QKV_PREPROCESS_TILING_H_
#define QWEN3_NEXT_QKV_PREPROCESS_TILING_H_

#include <cstdint>

// Compile info structure for Qwen3NextQKVPreprocess
struct Qwen3NextQKVPreprocessCompileInfo {
    uint32_t coreNum;
    uint64_t ubSize;
};

// Tiling data for Qwen3Next Fused Attention
// This structure is passed from host to kernel
struct Qwen3NextQKVPreprocessTilingData {
    uint32_t numTokens;       // number of tokens (batch * seq_len)
    uint32_t numHeads;        // number of query heads
    uint32_t numKvHeads;      // number of key/value heads
    uint32_t headDim;         // head dimension
    uint32_t qSize;           // q_size = numHeads * headDim
    uint32_t kvSize;          // kv_size = numKvHeads * headDim
    uint32_t hiddenSize;       // hidden size
    uint32_t qkvSize;         // total QKV size (qSize + 2*kvSize or qSize*2 + 2*kvSize)
    uint32_t blockDim;        // block dimension for parallelization
    uint32_t attnOutputGate;  // whether attn output gate is enabled (0 or 1)
    float epsilon;            // RMSNorm epsilon

    // Getters and setters for tiling data serialization
    uint32_t get_num_tokens() const { return numTokens; }
    void set_num_tokens(uint32_t val) { numTokens = val; }

    uint32_t get_num_heads() const { return numHeads; }
    void set_num_heads(uint32_t val) { numHeads = val; }

    uint32_t get_num_kv_heads() const { return numKvHeads; }
    void set_num_kv_heads(uint32_t val) { numKvHeads = val; }

    uint32_t get_head_dim() const { return headDim; }
    void set_head_dim(uint32_t val) { headDim = val; }

    uint32_t get_q_size() const { return qSize; }
    void set_q_size(uint32_t val) { qSize = val; }

    uint32_t get_kv_size() const { return kvSize; }
    void set_kv_size(uint32_t val) { kvSize = val; }

    uint32_t get_hidden_size() const { return hiddenSize; }
    void set_hidden_size(uint32_t val) { hiddenSize = val; }

    uint32_t get_qkv_size() const { return qkvSize; }
    void set_qkv_size(uint32_t val) { qkvSize = val; }

    uint32_t get_block_dim() const { return blockDim; }
    void set_block_dim(uint32_t val) { blockDim = val; }

    uint32_t get_attn_output_gate() const { return attnOutputGate; }
    void set_attn_output_gate(uint32_t val) { attnOutputGate = val; }

    float get_epsilon() const { return epsilon; }
    void set_epsilon(float val) { epsilon = val; }

    static constexpr uint32_t GetDataSize() {
        return sizeof(Qwen3NextQKVPreprocessTilingData);
    }

    void SaveToBuffer(uint8_t* buffer, uint32_t bufferSize) const {
        if (bufferSize >= sizeof(Qwen3NextQKVPreprocessTilingData)) {
            std::memcpy(buffer, this, sizeof(Qwen3NextQKVPreprocessTilingData));
        }
    }

    void LoadFromBuffer(const uint8_t* buffer, uint32_t bufferSize) {
        if (bufferSize >= sizeof(Qwen3NextQKVPreprocessTilingData)) {
            std::memcpy(this, buffer, sizeof(Qwen3NextQKVPreprocessTilingData));
        }
    }
};

#endif // QWEN3_NEXT_QKV_PREPROCESS_TILING_H_