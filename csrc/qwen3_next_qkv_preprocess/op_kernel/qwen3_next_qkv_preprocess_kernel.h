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

#include "kernel_operator.h"
#include "rms_norm_base.h"
#include <cmath>

using namespace AscendC;
using RmsNorm::ReduceSumCustom;
using RmsNorm::DataCopyCustom;
using RmsNorm::is_same;
using RmsNorm::CeilDiv;

#ifndef QWEN3_NEXT_QKV_PREPROCESS_TILING_DATA_H_
#define QWEN3_NEXT_QKV_PREPROCESS_TILING_DATA_H_

// Tiling data structure for Qwen3Next QKV Preprocessing
// Must match the host-side Qwen3NextQKVPreprocessTilingData exactly (binary-compatible)
struct Qwen3NextQKVPreprocessTilingData {
    uint32_t numTokens;
    uint32_t numHeads;
    uint32_t numKvHeads;
    uint32_t headDim;
    uint32_t qSize;
    uint32_t kvSize;
    uint32_t hiddenSize;   // must match host struct field order (between kvSize and qkvSize)
    uint32_t qkvSize;
    uint32_t blockDim;
    uint32_t attnOutputGate;  // 0 or 1
    float epsilon;
};
#endif  // QWEN3_NEXT_QKV_PREPROCESS_TILING_DATA_H_

constexpr uint32_t QBUFFER_NUM = 2;

template <typename T>
class KernelQwen3NextQKVPreprocess {
public:
    __aicore__ inline KernelQwen3NextQKVPreprocess(TPipe* pipe)
    {
        Ppipe = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR qkv, GM_ADDR qNormWeight, GM_ADDR kNormWeight,
        GM_ADDR cosSinCache, GM_ADDR positions,
        GM_ADDR qOut, GM_ADDR kOut, GM_ADDR vOut, GM_ADDR gateOut,
        const Qwen3NextQKVPreprocessTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");

        this->numTokens = tiling->numTokens;
        this->numHeads = tiling->numHeads;
        this->numKvHeads = tiling->numKvHeads;
        this->headDim = tiling->headDim;
        this->invHeadDim = 1.0f / this->headDim;
        this->qSize = tiling->qSize;
        this->kvSize = tiling->kvSize;
        this->qkvSize = tiling->qkvSize;
        this->attnOutputGate = tiling->attnOutputGate;
        this->epsilon = tiling->epsilon;

        blockIdx_ = GetBlockIdx();

        // Work distribution across cores
        uint32_t tokensPerCore = CeilDiv(numTokens, static_cast<uint32_t>(GetBlockNum()));
        this->startToken = blockIdx_ * tokensPerCore;
        this->endToken = startToken + tokensPerCore;
        if (this->endToken > numTokens) {
            this->endToken = numTokens;
        }
        this->tokenWork = endToken - startToken;

        // Global buffers
        // qkvSize when attnOutputGate=true: 2*qSize + 2*kvSize (q_gate + k + v)
        // qkvSize when attnOutputGate=false: qSize + 2*kvSize (q + k + v)
        qkvGm.SetGlobalBuffer((__gm__ T*)qkv + startToken * qkvSize, tokenWork * qkvSize);

        // gateOut is always allocated by the caller when attnOutputGate=true
        gateOutGm.SetGlobalBuffer((__gm__ T*)gateOut + startToken * qSize, tokenWork * qSize);

        qOutGm.SetGlobalBuffer((__gm__ T*)qOut + startToken * qSize, tokenWork * qSize);
        kOutGm.SetGlobalBuffer((__gm__ T*)kOut + startToken * kvSize, tokenWork * kvSize);
        vOutGm.SetGlobalBuffer((__gm__ T*)vOut + startToken * kvSize, tokenWork * kvSize);

        // RMSNorm weights - shape [headDim]
        qNormWeightGm.SetGlobalBuffer((__gm__ T*)qNormWeight, headDim);
        kNormWeightGm.SetGlobalBuffer((__gm__ T*)kNormWeight, headDim);

        // Rotary embeddings: [maxPosition, headDim * 2] - first half is cos, second half is sin
        // cosSinCache is indexed as: positionId * headDim * 2 + dim (for cos) or + headDim + dim (for sin)
        cosSinCacheGm.SetGlobalBuffer((__gm__ T*)cosSinCache, numTokens * headDim * 2);

        // Positions: [numTokens] - position IDs for each token in rotary embedding
        positionsGm.SetGlobalBuffer((__gm__ int64_t*)positions, numTokens);

        // UB buffer sizing: inQueueQkv holds Q (qSize) and K (kvSize) in separate buffers
        // Use max(qSize, kvSize) so either Q or K fits in a single buffer slot
        uint32_t maxQKSize = (qSize > kvSize) ? qSize : kvSize;
        uint32_t ubSizeQK = (maxQKSize > 256) ? 256 : maxQKSize;
        uint32_t ubSizeGate = (qSize > 256) ? 256 : qSize;

        Ppipe->InitBuffer(inQueueQkv, QBUFFER_NUM, ubSizeQK * sizeof(T));
        Ppipe->InitBuffer(inQueueQNormWeight, QBUFFER_NUM, headDim * sizeof(T));
        Ppipe->InitBuffer(inQueueKNormWeight, QBUFFER_NUM, headDim * sizeof(T));
        Ppipe->InitBuffer(outQueueQ, QBUFFER_NUM, ubSizeQK * sizeof(T));
        Ppipe->InitBuffer(outQueueK, QBUFFER_NUM, ubSizeQK * sizeof(T));
        Ppipe->InitBuffer(outQueueV, QBUFFER_NUM, ubSizeQK * sizeof(T));
        Ppipe->InitBuffer(outQueueGate, QBUFFER_NUM, ubSizeGate * sizeof(T));

        // Float scratch buffers: headDim floats for RMS computation + 64 floats for reduce work
        uint32_t rmsBufSize = (headDim > 128) ? headDim : 128;
        Ppipe->InitBuffer(rmsBuf, rmsBufSize * sizeof(float));
        Ppipe->InitBuffer(reduceBuf, 256 * sizeof(float));  // Need 256 floats for rotary computation
        Ppipe->InitBuffer(weightFloatBuf, headDim * sizeof(float));  // for casting T weight to float
        Ppipe->InitBuffer(rotaryTmpBuf, 256 * sizeof(float));  // for rotary computation
        Ppipe->InitBuffer(rmsTmpBuf, headDim * sizeof(float));  // temp for x * (1 + w) in RMSNorm
    }

    __aicore__ inline void Process()
    {
        // Load norm weights once; they are reused across all tokens via EnQue/DeQue cycle
        LocalTensor<T> qWeightInit = inQueueQNormWeight.AllocTensor<T>();
        DataCopyCustom<T>(qWeightInit, qNormWeightGm, headDim);
        inQueueQNormWeight.EnQue(qWeightInit);

        LocalTensor<T> kWeightInit = inQueueKNormWeight.AllocTensor<T>();
        DataCopyCustom<T>(kWeightInit, kNormWeightGm, headDim);
        inQueueKNormWeight.EnQue(kWeightInit);

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
    // Split input QKV and route each tensor to its destination queue.
    // V goes directly to outQueueV to avoid exceeding inQueueQkv's QBUFFER_NUM=2 capacity.
    //
    // QKV layout (matches qwen3_next.py qkv_proj output):
    //   attnOutputGate=true:  qkv=[q_gate(2*qSize), k(kvSize), v(kvSize)]
    //                         where q_gate is split into q(first qSize) and gate(second qSize)
    //   attnOutputGate=false: qkv=[q(qSize), k(kvSize), v(kvSize)]
    __aicore__ inline void SplitQKV(uint32_t tokenIdx)
    {
        uint32_t qkvOffset = tokenIdx * qkvSize;

        if (attnOutputGate) {
            // QKV layout: [q_gate(2*qSize), k(kvSize), v(kvSize)]
            // q_gate = qkv[0:2*qSize], split into q and gate (each qSize)
            // k starts at offset 2*qSize
            // v starts at offset 2*qSize + kvSize

            // Extract q (first half of q_gate) - goes to inQueueQkv for RMSNorm
            LocalTensor<T> qLocal = inQueueQkv.AllocTensor<T>();
            DataCopyCustom<T>(qLocal, qkvGm[qkvOffset], qSize);
            inQueueQkv.EnQue(qLocal);

            // Extract gate (second half of q_gate) - goes directly to outQueueGate
            LocalTensor<T> gateLocal = outQueueGate.AllocTensor<T>();
            DataCopyCustom<T>(gateLocal, qkvGm[qkvOffset + qSize], qSize);
            outQueueGate.EnQue(gateLocal);

            // Extract k - starts at offset 2*qSize
            LocalTensor<T> kLocal = inQueueQkv.AllocTensor<T>();
            DataCopyCustom<T>(kLocal, qkvGm[qkvOffset + 2 * qSize], kvSize);
            inQueueQkv.EnQue(kLocal);

            // Extract v - starts at offset 2*qSize + kvSize
            LocalTensor<T> vLocal = outQueueV.AllocTensor<T>();
            DataCopyCustom<T>(vLocal, qkvGm[qkvOffset + 2 * qSize + kvSize], kvSize);
            outQueueV.EnQue(vLocal);
        } else {
            // QKV layout: [q(qSize), k(kvSize), v(kvSize)]

            LocalTensor<T> qLocal = inQueueQkv.AllocTensor<T>();
            DataCopyCustom<T>(qLocal, qkvGm[qkvOffset], qSize);
            inQueueQkv.EnQue(qLocal);

            LocalTensor<T> kLocal = inQueueQkv.AllocTensor<T>();
            DataCopyCustom<T>(kLocal, qkvGm[qkvOffset + qSize], kvSize);
            inQueueQkv.EnQue(kLocal);

            // V goes to outQueueV directly
            LocalTensor<T> vLocal = outQueueV.AllocTensor<T>();
            DataCopyCustom<T>(vLocal, qkvGm[qkvOffset + qSize + kvSize], kvSize);
            outQueueV.EnQue(vLocal);
        }
    }

    // Apply RMSNorm per head to Q tensor.
    // GemmaRMSNorm formula: x = x * rsqrt(mean(x^2) + eps) * (1 + weight)
    //                       = x * rstd * (1 + w) = x_norm + x_norm * w
    // Reduce dimension is headDim (not qSize) - one rstd per head.
    __aicore__ inline void ApplyQRMSNorm(uint32_t tokenIdx)
    {
        LocalTensor<T> qLocal = inQueueQkv.DeQue<T>();
        LocalTensor<T> qNormLocal = outQueueQ.AllocTensor<T>();
        LocalTensor<T> qWeightLocal = inQueueQNormWeight.DeQue<T>();

        LocalTensor<float> headBuf = rmsBuf.Get<float>();
        LocalTensor<float> workBuf = reduceBuf.Get<float>();
        LocalTensor<float> qBuf = reduceBuf.Get<float>();
        LocalTensor<float> qNormBuf = reduceBuf.Get<float>();
        LocalTensor<float> weightFloatBufLocal = weightFloatBuf.Get<float>();
        LocalTensor<float> tmpBuf = rmsTmpBuf.Get<float>();

        // Cast weight from T to float once per token (reused across all heads)
        if constexpr (is_same<T, float>::value) {
            // Already float, use directly
            weightFloatBufLocal = qWeightLocal;
        } else {
            Cast(weightFloatBufLocal, qWeightLocal, RoundMode::CAST_NONE, headDim);
        }
        PipeBarrier<PIPE_V>();

        for (uint32_t head = 0; head < numHeads; head++) {
            uint32_t headOffset = head * headDim;

            // Cast this head's elements to float for numerically stable reduction
            if constexpr (is_same<T, float>::value) {
                Muls(headBuf, qLocal[headOffset], 1.0f, headDim);
            } else {
                Cast(headBuf, qLocal[headOffset], RoundMode::CAST_NONE, headDim);
            }
            PipeBarrier<PIPE_V>();

            // Square each element
            Mul(headBuf, headBuf, headBuf, headDim);
            PipeBarrier<PIPE_V>();

            // Scale by 1/headDim to get mean of squares
            Muls(headBuf, headBuf, invHeadDim, headDim);
            PipeBarrier<PIPE_V>();

            // Reduce sum over headDim → result in headBuf[0]
            ReduceSumCustom(headBuf, headBuf, workBuf, headDim);
            PipeBarrier<PIPE_V>();

            // rstd = 1 / sqrt(mean_of_squares + epsilon)
            float ms = headBuf.GetValue(0);
            float rstdVal = 1.0f / sqrt(ms + epsilon);

            // Cast input to float, apply rstd in float
            if constexpr (is_same<T, float>::value) {
                Muls(qNormBuf, qLocal[headOffset], rstdVal, headDim);
            } else {
                Cast(qBuf, qLocal[headOffset], RoundMode::CAST_NONE, headDim);
                PipeBarrier<PIPE_V>();
                Muls(qNormBuf, qBuf, rstdVal, headDim);
            }
            PipeBarrier<PIPE_V>();

            // Apply GemmaRMSNorm: x * (1 + w) = x + x * w
            // Compute x * w in tmpBuf
            Mul(tmpBuf, qNormBuf, weightFloatBufLocal, headDim);
            PipeBarrier<PIPE_V>();
            // Add: x + x * w
            Add(qNormBuf, qNormBuf, tmpBuf, headDim);
            PipeBarrier<PIPE_V>();

            // Cast result back to T
            Cast(qNormLocal[headOffset], qNormBuf, RoundMode::CAST_NONE, headDim);
            PipeBarrier<PIPE_V>();
        }

        // Return weight buffer for reuse on the next token
        inQueueQNormWeight.EnQue(qWeightLocal);
        outQueueQ.EnQue(qNormLocal);
        inQueueQkv.FreeTensor(qLocal);
    }

    // Apply RMSNorm per head to K tensor.
    // Apply RMSNorm per head to K tensor.
    // GemmaRMSNorm formula: x = x * rsqrt(mean(x^2) + eps) * (1 + weight)
    //                       = x * rstd * (1 + w) = x_norm + x_norm * w
    __aicore__ inline void ApplyKRMSNorm(uint32_t tokenIdx)
    {
        LocalTensor<T> kLocal = inQueueQkv.DeQue<T>();
        LocalTensor<T> kNormLocal = outQueueK.AllocTensor<T>();
        LocalTensor<T> kWeightLocal = inQueueKNormWeight.DeQue<T>();

        LocalTensor<float> headBuf = rmsBuf.Get<float>();
        LocalTensor<float> workBuf = reduceBuf.Get<float>();
        LocalTensor<float> kBuf = reduceBuf.Get<float>();
        LocalTensor<float> kNormBuf = reduceBuf.Get<float>();
        LocalTensor<float> weightFloatBufLocal = weightFloatBuf.Get<float>();
        LocalTensor<float> tmpBuf = rmsTmpBuf.Get<float>();

        // Cast weight from T to float once per token
        if constexpr (is_same<T, float>::value) {
            weightFloatBufLocal = kWeightLocal;
        } else {
            Cast(weightFloatBufLocal, kWeightLocal, RoundMode::CAST_NONE, headDim);
        }
        PipeBarrier<PIPE_V>();

        for (uint32_t head = 0; head < numKvHeads; head++) {
            uint32_t headOffset = head * headDim;

            if constexpr (is_same<T, float>::value) {
                Muls(headBuf, kLocal[headOffset], 1.0f, headDim);
            } else {
                Cast(headBuf, kLocal[headOffset], RoundMode::CAST_NONE, headDim);
            }
            PipeBarrier<PIPE_V>();

            Mul(headBuf, headBuf, headBuf, headDim);
            PipeBarrier<PIPE_V>();

            Muls(headBuf, headBuf, invHeadDim, headDim);
            PipeBarrier<PIPE_V>();

            ReduceSumCustom(headBuf, headBuf, workBuf, headDim);
            PipeBarrier<PIPE_V>();

            float ms = headBuf.GetValue(0);
            float rstdVal = 1.0f / sqrt(ms + epsilon);

            // Cast input to float, apply rstd in float
            if constexpr (is_same<T, float>::value) {
                Muls(kNormBuf, kLocal[headOffset], rstdVal, headDim);
            } else {
                Cast(kBuf, kLocal[headOffset], RoundMode::CAST_NONE, headDim);
                PipeBarrier<PIPE_V>();
                Muls(kNormBuf, kBuf, rstdVal, headDim);
            }
            PipeBarrier<PIPE_V>();

            // Apply GemmaRMSNorm: x * (1 + w) = x + x * w
            // Compute x * w in tmpBuf
            Mul(tmpBuf, kNormBuf, weightFloatBufLocal, headDim);
            PipeBarrier<PIPE_V>();
            // Add: x + x * w
            Add(kNormBuf, kNormBuf, tmpBuf, headDim);
            PipeBarrier<PIPE_V>();

            // Cast result back to T
            Cast(kNormLocal[headOffset], kNormBuf, RoundMode::CAST_NONE, headDim);
            PipeBarrier<PIPE_V>();
        }

        inQueueKNormWeight.EnQue(kWeightLocal);
        outQueueK.EnQue(kNormLocal);
        inQueueQkv.FreeTensor(kLocal);
    }

    // Apply interleaved rotary embedding to Q:
    //   rotary[2i]   = q[2i]*cos[2i]   - q[2i+1]*sin[2i+1]
    //   rotary[2i+1] = q[2i]*sin[2i]   + q[2i+1]*cos[2i+1]
    // cosSinCache: [maxPosition, headDim * 2], first half is cos, second half is sin
    // cos for position: cosSinCache[positionId * headDim * 2 + dim]
    // sin for position: cosSinCache[positionId * headDim * 2 + headDim + dim]
    __aicore__ inline void ApplyRotaryQ(uint32_t tokenIdx)
    {
        LocalTensor<T> qNormLocal = outQueueQ.DeQue<T>();
        uint32_t positionId = positionsGm.GetValue(tokenIdx);
        uint32_t cosSinBase = positionId * headDim * 2;  // base offset for cos/sin in cosSinCache

        LocalTensor<T> qRotLocal = outQueueQ.AllocTensor<T>();
        // Use separate buffers to avoid data overlap
        LocalTensor<float> qBuf = reduceBuf.Get<float>();          // for q data
        LocalTensor<float> cosSinBuf = rotaryTmpBuf.Get<float>(); // for cos/sin data
        // Use inQueueQkv as temp T storage for cos/sin GM loads
        LocalTensor<T> cosTmpLocal = inQueueQkv.AllocTensor<T>();

        for (uint32_t head = 0; head < numHeads; head++) {
            uint32_t baseOffset = head * headDim;

            // Cast qNorm to float for this head (data copied to qBuf[0..headDim-1])
            if constexpr (is_same<T, float>::value) {
                DataCopyCustom<float>(qBuf, qNormLocal[baseOffset], headDim);
            } else {
                Cast(qBuf, qNormLocal[baseOffset], RoundMode::CAST_NONE, headDim);
            }
            PipeBarrier<PIPE_V>();

            // Load cos from GM to temp T buffer
            DataCopyCustom<T>(cosTmpLocal, cosSinCacheGm[cosSinBase], headDim);

            // Cast to float (data in cosSinBuf[0..headDim-1])
            if constexpr (is_same<T, float>::value) {
                DataCopyCustom<float>(cosSinBuf, cosTmpLocal, headDim);
            } else {
                Cast(cosSinBuf, cosTmpLocal, RoundMode::CAST_NONE, headDim);
            }
            PipeBarrier<PIPE_V>();

            // Also load sin to cosSinBuf[headDim..2*headDim-1]
            DataCopyCustom<T>(cosTmpLocal, cosSinCacheGm[cosSinBase + headDim], headDim);
            if constexpr (is_same<T, float>::value) {
                DataCopyCustom<float>(cosSinBuf[headDim], cosTmpLocal, headDim);
            } else {
                Cast(cosSinBuf[headDim], cosTmpLocal, RoundMode::CAST_NONE, headDim);
            }
            PipeBarrier<PIPE_V>();

            // Compute rotary in float (in-place on qBuf)
            for (uint32_t dim = 0; dim < headDim; dim += 2) {
                float q0 = qBuf.GetValue(dim);
                float q1 = qBuf.GetValue(dim + 1);
                float cosVal = cosSinBuf.GetValue(dim);
                float sinVal = cosSinBuf.GetValue(headDim + dim);  // sin values start at offset headDim

                float rot0 = q0 * cosVal - q1 * sinVal;
                float rot1 = q0 * sinVal + q1 * cosVal;

                qBuf.SetValue(dim, rot0);
                qBuf.SetValue(dim + 1, rot1);
            }
            PipeBarrier<PIPE_V>();

            // Cast result back to T and copy to output
            Cast(qRotLocal[baseOffset], qBuf[0], RoundMode::CAST_NONE, headDim);
            PipeBarrier<PIPE_V>();
        }

        inQueueQkv.FreeTensor(cosTmpLocal);
        outQueueQ.FreeTensor(qNormLocal);
        outQueueQ.EnQue(qRotLocal);
    }

    // Apply interleaved rotary embedding to K over all numKvHeads.
    // cosSinCache: [maxPosition, headDim * 2], first half is cos, second half is sin
    // cos for position: cosSinCache[positionId * headDim * 2 + dim]
    // sin for position: cosSinCache[positionId * headDim * 2 + headDim + dim]
    __aicore__ inline void ApplyRotaryK(uint32_t tokenIdx)
    {
        LocalTensor<T> kNormLocal = outQueueK.DeQue<T>();
        uint32_t positionId = positionsGm.GetValue(tokenIdx);
        uint32_t cosSinBase = positionId * headDim * 2;  // base offset for cos/sin in cosSinCache

        LocalTensor<T> kRotLocal = outQueueK.AllocTensor<T>();
        // Use separate buffers to avoid data overlap
        LocalTensor<float> kBuf = reduceBuf.Get<float>();          // for k data
        LocalTensor<float> cosSinBuf = rotaryTmpBuf.Get<float>(); // for cos/sin data
        // Use inQueueQkv as temp T storage for cos/sin GM loads
        LocalTensor<T> cosTmpLocal = inQueueQkv.AllocTensor<T>();

        for (uint32_t head = 0; head < numKvHeads; head++) {
            uint32_t baseOffset = head * headDim;

            // Cast kNorm to float for this head (data copied to kBuf[0..headDim-1])
            if constexpr (is_same<T, float>::value) {
                DataCopyCustom<float>(kBuf, kNormLocal[baseOffset], headDim);
            } else {
                Cast(kBuf, kNormLocal[baseOffset], RoundMode::CAST_NONE, headDim);
            }
            PipeBarrier<PIPE_V>();

            // Load cos from GM to temp T buffer
            DataCopyCustom<T>(cosTmpLocal, cosSinCacheGm[cosSinBase], headDim);

            // Cast to float (cos data in cosSinBuf[0..headDim-1])
            if constexpr (is_same<T, float>::value) {
                DataCopyCustom<float>(cosSinBuf, cosTmpLocal, headDim);
            } else {
                Cast(cosSinBuf, cosTmpLocal, RoundMode::CAST_NONE, headDim);
            }
            PipeBarrier<PIPE_V>();

            // Also load sin to cosSinBuf[headDim..2*headDim-1]
            DataCopyCustom<T>(cosTmpLocal, cosSinCacheGm[cosSinBase + headDim], headDim);
            if constexpr (is_same<T, float>::value) {
                DataCopyCustom<float>(cosSinBuf[headDim], cosTmpLocal, headDim);
            } else {
                Cast(cosSinBuf[headDim], cosTmpLocal, RoundMode::CAST_NONE, headDim);
            }
            PipeBarrier<PIPE_V>();

            // Compute rotary: k_buf[2i] = k[2i]*cos[2i] - k[2i+1]*sin[2i+1]
            for (uint32_t dim = 0; dim < headDim; dim += 2) {
                float k0 = kBuf.GetValue(dim);
                float k1 = kBuf.GetValue(dim + 1);
                float cosVal = cosSinBuf.GetValue(dim);
                float sinVal = cosSinBuf.GetValue(headDim + dim);  // sin values start at offset headDim

                float rot0 = k0 * cosVal - k1 * sinVal;
                float rot1 = k0 * sinVal + k1 * cosVal;

                kBuf.SetValue(dim, rot0);
                kBuf.SetValue(dim + 1, rot1);
            }
            PipeBarrier<PIPE_V>();

            // Cast result back to T
            Cast(kRotLocal[baseOffset], kBuf[0], RoundMode::CAST_NONE, headDim);
            PipeBarrier<PIPE_V>();
        }

        inQueueQkv.FreeTensor(cosTmpLocal);
        outQueueK.FreeTensor(kNormLocal);
        outQueueK.EnQue(kRotLocal);
    }

    __aicore__ inline void CopyOutputs(uint32_t tokenIdx)
    {
        // Q output (after RMSNorm + Rotary)
        LocalTensor<T> qLocal = outQueueQ.DeQue<T>();
        DataCopyCustom<T>(qOutGm[tokenIdx * qSize], qLocal, qSize);
        outQueueQ.FreeTensor(qLocal);

        // K output (after RMSNorm + Rotary)
        LocalTensor<T> kLocal = outQueueK.DeQue<T>();
        DataCopyCustom<T>(kOutGm[tokenIdx * kvSize], kLocal, kvSize);
        outQueueK.FreeTensor(kLocal);

        // V output (pass-through, stored in outQueueV by SplitQKV)
        LocalTensor<T> vLocal = outQueueV.DeQue<T>();
        DataCopyCustom<T>(vOutGm[tokenIdx * kvSize], vLocal, kvSize);
        outQueueV.FreeTensor(vLocal);

        // Gate output (extracted from qkv when attnOutputGate is enabled)
        if (attnOutputGate) {
            LocalTensor<T> gateLocal = outQueueGate.DeQue<T>();
            DataCopyCustom<T>(gateOutGm[tokenIdx * qSize], gateLocal, qSize);
            outQueueGate.FreeTensor(gateLocal);
        }
    }

private:
    TPipe* Ppipe = nullptr;

    // Input queues (QBUFFER_NUM=2 each)
    TQue<QuePosition::VECIN, QBUFFER_NUM> inQueueQkv;        // Q and K tensors
    TQue<QuePosition::VECIN, QBUFFER_NUM> inQueueQNormWeight;
    TQue<QuePosition::VECIN, QBUFFER_NUM> inQueueKNormWeight;

    // Output queues
    TQue<QuePosition::VECOUT, QBUFFER_NUM> outQueueQ;
    TQue<QuePosition::VECOUT, QBUFFER_NUM> outQueueK;
    TQue<QuePosition::VECOUT, QBUFFER_NUM> outQueueV;    // V is directly routed here by SplitQKV
    TQue<QuePosition::VECOUT, QBUFFER_NUM> outQueueGate;

    // Float scratch buffers for RMSNorm computation
    TBuf<TPosition::VECCALC> rmsBuf;    // headDim floats for per-head intermediate values
    TBuf<TPosition::VECCALC> reduceBuf; // work buffer for ReduceSumCustom + rstd scalar
    TBuf<TPosition::VECCALC> weightFloatBuf;  // for casting T weight to float before Mul
    TBuf<TPosition::VECCALC> rotaryTmpBuf;    // temporary buffer for rotary computation
    TBuf<TPosition::VECCALC> rmsTmpBuf;      // temporary buffer for x * (1 + w) computation

    // Global tensors
    GlobalTensor<T> qkvGm;
    GlobalTensor<T> qOutGm;
    GlobalTensor<T> kOutGm;
    GlobalTensor<T> vOutGm;
    GlobalTensor<T> gateOutGm;
    GlobalTensor<T> qNormWeightGm;
    GlobalTensor<T> kNormWeightGm;
    GlobalTensor<T> cosSinCacheGm;  // [maxPosition, headDim * 2] - cos and sin concatenated
    GlobalTensor<int64_t> positionsGm;  // position IDs for rotary embedding

    // Parameters from tiling
    uint32_t numTokens;
    uint32_t numHeads;
    uint32_t numKvHeads;
    uint32_t headDim;
    float invHeadDim;  // precomputed 1.0f / headDim to avoid static_cast in inner loop
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
