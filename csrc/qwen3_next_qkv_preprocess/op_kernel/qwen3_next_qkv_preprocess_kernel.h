/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for the details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QWEN3_NEXT_QKV_PREPROCESS_KERNEL_H_
#define QWEN3_NEXT_QKV_PREPROCESS_KERNEL_H_

#include <AscendC.h>

using namespace AscendC;

// Tiling data structure for Qwen3Next QKV Preprocessing
struct Qwen3NextQKVPreprocessTilingData {
    uint32_t numTokens;
    uint32_t numHeads;
    uint32_t numKvHeads;
    uint32_t headDim;
    uint32_t qSize;
    uint32_t kvSize;
    uint32_t qkvSize;
    uint32_t blockDim;
    uint32_t attnOutputGate;  // 0 or 1
    float epsilon;
};

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t REDUCE_LEN = 64;

template <typename T>
class KernelQwen3NextQKVPreprocess {
public:
    __aicore__ inline KernelQwen3NextQKVPreprocess(TPipe* pipe)
    {
        Ppipe = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR qkv, GM_ADDR qNormWeight, GM_ADDR kNormWeight,
        GM_ADDR qCos, GM_ADDR qSin, GM_ADDR kCos, GM_ADDR kSin,
        GM_ADDR qOut, GM_ADDR kOut, GM_ADDR vOut, GM_ADDR gateOut,
        const Qwen3NextQKVPreprocessTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");

        this->numTokens = tiling->numTokens;
        this->numHeads = tiling->numHeads;
        this->numKvHeads = tiling->numKvHeads;
        this->headDim = tiling->headDim;
        this->qSize = tiling->qSize;
        this->kvSize = tiling->kvSize;
        this->qkvSize = tiling->qkvSize;
        this->attnOutputGate = tiling->attnOutputGate;
        this->epsilon = tiling->epsilon;

        blockIdx_ = GetBlockIdx();

        // Work distribution across cores
        uint32_t tokensPerCore = CeilDiv(numTokens, GetBlockNum());
        this->startToken = blockIdx_ * tokensPerCore;
        this->endToken = startToken + tokensPerCore;
        if (this->endToken > numTokens) {
            this->endToken = numTokens;
        }
        this->tokenWork = endToken - startToken;

        // Global buffers - offset is in elements for 1D tensors
        qkvGm.SetGlobalBuffer((__gm__ T*)qkv + startToken * qkvSize, tokenWork * qkvSize);
        qOutGm.SetGlobalBuffer((__gm__ T*)qOut + startToken * qSize, tokenWork * qSize);
        kOutGm.SetGlobalBuffer((__gm__ T*)kOut + startToken * kvSize, tokenWork * kvSize);
        vOutGm.SetGlobalBuffer((__gm__ T*)vOut + startToken * kvSize, tokenWork * kvSize);

        if (attnOutputGate) {
            gateOutGm.SetGlobalBuffer((__gm__ T*)gateOut + startToken * qSize, tokenWork * qSize);
        }

        // RMSNorm weights
        qNormWeightGm.SetGlobalBuffer((__gm__ T*)qNormWeight, headDim);
        kNormWeightGm.SetGlobalBuffer((__gm__ T*)kNormWeight, headDim);

        // Rotary embeddings: (numPositions, headDim)
        qCosGm.SetGlobalBuffer((__gm__ T*)qCos, numTokens * headDim);
        qSinGm.SetGlobalBuffer((__gm__ T*)qSin, numTokens * headDim);
        kCosGm.SetGlobalBuffer((__gm__ T*)kCos, numTokens * headDim);
        kSinGm.SetGlobalBuffer((__gm__ T*)kSin, numTokens * headDim);

        // Initialize UB buffers
        // Need space for: Q(2*qSize), K(kvSize), V(kvSize), gate(qSize), plus intermediates
        uint32_t ubSizeQ = (qSize > 256) ? 256 : qSize;
        uint32_t ubSizeK = (kvSize > 256) ? 256 : kvSize;
        uint32_t ubSizeGate = (qSize > 256) ? 256 : qSize;

        Ppipe->InitBuffer(inQueueQkv, BUFFER_NUM, ubSizeQ * sizeof(T));
        Ppipe->InitBuffer(inQueueGate, BUFFER_NUM, ubSizeGate * sizeof(T));
        Ppipe->InitBuffer(inQueueQNormWeight, BUFFER_NUM, headDim * sizeof(T));
        Ppipe->InitBuffer(inQueueKNormWeight, BUFFER_NUM, headDim * sizeof(T));
        Ppipe->InitBuffer(outQueueQ, BUFFER_NUM, ubSizeQ * sizeof(T));
        Ppipe->InitBuffer(outQueueK, BUFFER_NUM, ubSizeK * sizeof(T));
        Ppipe->InitBuffer(outQueueV, BUFFER_NUM, ubSizeK * sizeof(T));
        Ppipe->InitBuffer(outQueueGate, BUFFER_NUM, ubSizeGate * sizeof(T));

        // RMSNorm intermediate buffers
        Ppipe->InitBuffer(rmsBuf, 128 * sizeof(float));
        Ppipe->InitBuffer(reduceBuf, 64 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t tokenIdx = 0; tokenIdx < tokenWork; tokenIdx++) {
            SplitQKV(tokenIdx);
            ApplyQRMSNorm(tokenIdx);
            ApplyKRMSNorm(tokenIdx);
            ApplyRotaryQ(tokenIdx);
            ApplyRotaryK(tokenIdx);
            CopyOutputs(tokenIdx);
        }
    }

private:
    __aicore__ inline void SplitQKV(uint32_t tokenIdx)
    {
        uint32_t qkvOffset = tokenIdx * qkvSize;

        if (attnOutputGate) {
            // qkv layout: [q_gate(2*qSize), k(kvSize), v(kvSize)]
            // Split q_gate into q(first qSize) and gate(second qSize)

            // Copy q (first half of q_gate)
            LocalTensor<T> qLocal = inQueueQkv.AllocTensor<T>();
            DataCopyCustom(qLocal, qkvGm[qkvOffset], qSize);

            // Copy gate (second half of q_gate)
            LocalTensor<T> gateLocal = inQueueGate.AllocTensor<T>();
            DataCopyCustom(gateLocal, qkvGm[qkvOffset + qSize], qSize);

            // Copy k and v
            LocalTensor<T> kLocal = inQueueQkv.AllocTensor<T>();
            LocalTensor<T> vLocal = inQueueQkv.AllocTensor<T>();
            DataCopyCustom(kLocal, qkvGm[qkvOffset + qSize * 2], kvSize);
            DataCopyCustom(vLocal, qkvGm[qkvOffset + qSize * 2 + kvSize], kvSize);

            // Enqueue: order matters for DeQue
            // We'll DeQue in order: q, k, v, then gate separately
            inQueueQkv.EnQue(qLocal);
            inQueueQkv.EnQue(kLocal);
            inQueueQkv.EnQue(vLocal);
            inQueueGate.EnQue(gateLocal);
        } else {
            // qkv layout: [q(qSize), k(kvSize), v(kvSize)]

            LocalTensor<T> qLocal = inQueueQkv.AllocTensor<T>();
            LocalTensor<T> kLocal = inQueueQkv.AllocTensor<T>();
            LocalTensor<T> vLocal = inQueueQkv.AllocTensor<T>();

            DataCopyCustom(qLocal, qkvGm[qkvOffset], qSize);
            DataCopyCustom(kLocal, qkvGm[qkvOffset + qSize], kvSize);
            DataCopyCustom(vLocal, qkvGm[qkvOffset + qSize + kvSize], kvSize);

            inQueueQkv.EnQue(qLocal);
            inQueueQkv.EnQue(kLocal);
            inQueueQkv.EnQue(vLocal);
        }
    }

    __aicore__ inline void ApplyQRMSNorm(uint32_t tokenIdx)
    {
        // Deque Q tensor (first tensor in queue)
        LocalTensor<T> qLocal = inQueueQkv.DeQue<T>();

        // Compute RMSNorm: output = input * weight / sqrt(sum(x^2)/n + eps)
        LocalTensor<float> sqLocal = rmsBuf.Get<float>();
        LocalTensor<float> reduceLocal = reduceBuf.Get<float>();

        // Cast input to float for computation
        if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
            Cast(sqLocal, qLocal, RoundMode::CAST_NONE, qSize);
        } else {
            // Float - direct copy
            for (uint32_t i = 0; i < qSize; i++) {
                sqLocal.SetValue(i, qLocal.GetValue(i));
            }
        }
        PipeBarrier<PIPE_V>();

        // Square
        Mul(sqLocal, sqLocal, sqLocal, qSize);
        PipeBarrier<PIPE_V>();

        // Average
        float avgFactor = 1.0f / static_cast<float>(headDim);
        Muls(sqLocal, sqLocal, avgFactor, qSize);
        PipeBarrier<PIPE_V>();

        // Reduce sum
        uint32_t repeat = qSize / REDUCE_LEN;
        if (repeat > 0) {
            ReduceSumCustom(sqLocal, sqLocal, reduceLocal, qSize);
            PipeBarrier<PIPE_V>();
        }

        // Compute rstd = 1/sqrt(sum + eps)
        Adds(sqLocal, sqLocal, epsilon, 1);
        Sqrt(sqLocal, sqLocal, 1);
        Duplicates(reduceLocal, sqLocal, 1);
        PipeBarrier<PIPE_V>();
        Div(sqLocal, reduceLocal, sqLocal, 1);  // rstd = 1/rstd
        PipeBarrier<PIPE_V>();

        // Apply to input: q_norm = q * rstd
        LocalTensor<T> qNormLocal = outQueueQ.AllocTensor<T>();
        LocalTensor<T> qWeightLocal = inQueueQNormWeight.DeQue<T>();

        // Cast rstd to same type as q for multiplication
        LocalTensor<float> rstdCast = reduceBuf.Get<float>();
        rstdCast.SetValue(0, sqLocal.GetValue(0));
        PipeBarrier<PIPE_V>();

        // Multiply input by rstd
        Muls(qNormLocal, qLocal, rstdCast, qSize);
        PipeBarrier<PIPE_V>();

        // Apply weight
        Mul(qNormLocal, qWeightLocal, qNormLocal, qSize);
        PipeBarrier<PIPE_V>();

        // Cleanup
        inQueueQNormWeight.EnQue(qWeightLocal);
        outQueueQ.EnQue(qNormLocal);
        inQueueQkv.FreeTensor(qLocal);
    }

    __aicore__ inline void ApplyKRMSNorm(uint32_t tokenIdx)
    {
        // Deque K tensor (second tensor in queue)
        LocalTensor<T> kLocal = inQueueQkv.DeQue<T>();

        LocalTensor<float> sqLocal = rmsBuf.Get<float>();
        LocalTensor<float> reduceLocal = reduceBuf.Get<float>();

        // Cast to float
        if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
            Cast(sqLocal, kLocal, RoundMode::CAST_NONE, kvSize);
        } else {
            for (uint32_t i = 0; i < kvSize; i++) {
                sqLocal.SetValue(i, kLocal.GetValue(i));
            }
        }
        PipeBarrier<PIPE_V>();

        // Square
        Mul(sqLocal, sqLocal, sqLocal, kvSize);
        PipeBarrier<PIPE_V>();

        // Average
        float avgFactor = 1.0f / static_cast<float>(headDim);
        Muls(sqLocal, sqLocal, avgFactor, kvSize);
        PipeBarrier<PIPE_V>();

        // Reduce
        uint32_t repeat = kvSize / REDUCE_LEN;
        if (repeat > 0) {
            ReduceSumCustom(sqLocal, sqLocal, reduceLocal, kvSize);
            PipeBarrier<PIPE_V>();
        }

        // rstd
        Adds(sqLocal, sqLocal, epsilon, 1);
        Sqrt(sqLocal, sqLocal, 1);
        Duplicates(reduceLocal, sqLocal, 1);
        PipeBarrier<PIPE_V>();
        Div(sqLocal, reduceLocal, sqLocal, 1);
        PipeBarrier<PIPE_V>();

        // Apply
        LocalTensor<T> kNormLocal = outQueueK.AllocTensor<T>();
        LocalTensor<T> kWeightLocal = inQueueKNormWeight.DeQue<T>();

        LocalTensor<float> rstdCast = reduceBuf.Get<float>();
        rstdCast.SetValue(0, sqLocal.GetValue(0));
        PipeBarrier<PIPE_V>();

        Muls(kNormLocal, kLocal, rstdCast, kvSize);
        PipeBarrier<PIPE_V>();
        Mul(kNormLocal, kWeightLocal, kNormLocal, kvSize);
        PipeBarrier<PIPE_V>();

        inQueueKNormWeight.EnQue(kWeightLocal);
        outQueueK.EnQue(kNormLocal);
        inQueueQkv.FreeTensor(kLocal);
    }

    __aicore__ inline void ApplyRotaryQ(uint32_t tokenIdx)
    {
        // Deque Q normalized tensor
        LocalTensor<T> qNormLocal = outQueueQ.DeQue<T>();
        uint32_t globalTokenId = startToken + tokenIdx;

        // Get cos/sin
        LocalTensor<T> cosTensor = qCosGm.Get<T>();
        LocalTensor<T> sinTensor = qSinGm.Get<T>();

        LocalTensor<T> qRotLocal = outQueueQ.AllocTensor<T>();

        // Apply rotary per head
        for (uint32_t head = 0; head < numHeads; head++) {
            uint32_t baseOffset = head * headDim;

            // For each head, compute rotary
            // q_rot[2i] = q[2i] * cos[2i] - q[2i+1] * sin[2i+1]
            // q_rot[2i+1] = q[2i] * sin[2i] + q[2i+1] * cos[2i+1]
            for (uint32_t dim = 0; dim < headDim; dim += 2) {
                uint32_t realDim1 = baseOffset + dim;
                uint32_t realDim2 = baseOffset + dim + 1;
                uint32_t cosSinOffset = globalTokenId * headDim + dim;

                T q1 = qNormLocal.GetValue(realDim1);
                T q2 = qNormLocal.GetValue(realDim2);
                T cos1 = cosTensor.GetValue(cosSinOffset);
                T sin1 = sinTensor.GetValue(cosSinOffset);

                // Note: This is simplified. Real rotary uses interleaved indexing.
                T qRot1 = q1 * cos1 - q2 * sin1;
                T qRot2 = q1 * sin1 + q2 * cos1;

                qRotLocal.SetValue(realDim1, qRot1);
                qRotLocal.SetValue(realDim2, qRot2);
            }
        }

        outQueueQ.EnQue(qRotLocal);
    }

    __aicore__ inline void ApplyRotaryK(uint32_t tokenIdx)
    {
        LocalTensor<T> kNormLocal = outQueueK.DeQue<T>();
        uint32_t globalTokenId = startToken + tokenIdx;

        LocalTensor<T> cosTensor = kCosGm.Get<T>();
        LocalTensor<T> sinTensor = kSinGm.Get<T>();

        LocalTensor<T> kRotLocal = outQueueK.AllocTensor<T>();

        for (uint32_t head = 0; head < numKvHeads; head++) {
            uint32_t baseOffset = head * headDim;

            for (uint32_t dim = 0; dim < headDim; dim += 2) {
                uint32_t realDim1 = baseOffset + dim;
                uint32_t realDim2 = baseOffset + dim + 1;
                uint32_t cosSinOffset = globalTokenId * headDim + dim;

                T k1 = kNormLocal.GetValue(realDim1);
                T k2 = kNormLocal.GetValue(realDim2);
                T cos1 = cosTensor.GetValue(cosSinOffset);
                T sin1 = sinTensor.GetValue(cosSinOffset);

                T kRot1 = k1 * cos1 - k2 * sin1;
                T kRot2 = k1 * sin1 + k2 * cos1;

                kRotLocal.SetValue(realDim1, kRot1);
                kRotLocal.SetValue(realDim2, kRot2);
            }
        }

        outQueueK.EnQue(kRotLocal);
    }

    __aicore__ inline void CopyOutputs(uint32_t tokenIdx)
    {
        // Q output
        LocalTensor<T> qLocal = outQueueQ.DeQue<T>();
        DataCopyCustom(qOutGm[tokenIdx * qSize], qLocal, qSize);
        outQueueQ.FreeTensor(qLocal);

        // K output
        LocalTensor<T> kLocal = outQueueK.DeQue<T>();
        DataCopyCustom(kOutGm[tokenIdx * kvSize], kLocal, kvSize);
        outQueueK.FreeTensor(kLocal);

        // V output - this was stored in inQueueQkv
        LocalTensor<T> vLocal = inQueueQkv.DeQue<T>();
        DataCopyCustom(vOutGm[tokenIdx * kvSize], vLocal, kvSize);
        inQueueQkv.FreeTensor(vLocal);

        // Gate output - only when attnOutputGate is enabled
        if (attnOutputGate) {
            LocalTensor<T> gateLocal = inQueueGate.DeQue<T>();
            DataCopyCustom(gateOutGm[tokenIdx * qSize], gateLocal, qSize);
            inQueueGate.FreeTensor(gateLocal);
        }
    }

private:
    TPipe* Ppipe = nullptr;

    // Input queues
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueQkv;  // For Q, K, V tensors
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGate;  // For gate tensor when enabled
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueQNormWeight;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueKNormWeight;

    // Output queues
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueQ;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueK;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueV;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueGate;

    // Buffers
    TBuf<TPosition::VECCALC> rmsBuf;
    TBuf<TPosition::VECCALC> reduceBuf;

    // Global tensors
    GlobalTensor<T> qkvGm;
    GlobalTensor<T> qOutGm;
    GlobalTensor<T> kOutGm;
    GlobalTensor<T> vOutGm;
    GlobalTensor<T> gateOutGm;
    GlobalTensor<T> qNormWeightGm;
    GlobalTensor<T> kNormWeightGm;
    GlobalTensor<T> qCosGm;
    GlobalTensor<T> qSinGm;
    GlobalTensor<T> kCosGm;
    GlobalTensor<T> kSinGm;

    // Parameters
    uint32_t numTokens;
    uint32_t numHeads;
    uint32_t numKvHeads;
    uint32_t headDim;
    uint32_t qSize;
    uint32_t kvSize;
    uint32_t qkvSize;
    uint32_t attnOutputGate;
    float epsilon;

    uint32_t blockIdx_;
    uint32_t startToken;
    uint32_t endToken;
    uint32_t tokenWork;
};

#endif // QWEN3_NEXT_QKV_PREPROCESS_KERNEL_H_