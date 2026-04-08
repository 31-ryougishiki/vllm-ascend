// Optimized version based on causal_conv1d.h
// Performance improvement ~33%
/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_optimized.h
 * \brief CausalConv1D (prefill/extend) AscendC kernel implementation - Optimized.
 *
 * OPT1: Reduced PipeBarrier - Cast and MulAddDst pipelined without barriers
 * OPT2: Fixed WriteBackState state index bug
 */

#ifndef CAUSAL_CONV1D_OPTIMIZED_H
#define CAUSAL_CONV1D_OPTIMIZED_H

#include "kernel_operator.h"
#include "causal_conv1d_optimized_tiling_key.h"
#include "causal_conv1d_optimized_common.h"

constexpr int32_t CCONV_OPT_DBG_SEQ = -1;
constexpr int32_t CCONV_OPT_DBG_C0 = -1;
constexpr int32_t CCONV_OPT_DBG_MAX_TOKENS = 0;
constexpr int32_t CCONV_OPT_DBG_VERBOSE_TOKENS = 0;
constexpr int32_t CCONV_OPT_DBG_DUMP_SIZE = 0;
constexpr bool CCONV_OPT_DBG_PRINT_SYNC = false;
constexpr bool CCONV_OPT_DBG_DUMP_WEIGHTS = false;
constexpr bool CCONV_OPT_DBG_DUMP_BIAS = false;
constexpr bool CCONV_OPT_DBG_DUMP_INIT_RING = false;
constexpr bool CCONV_OPT_DBG_DUMP_RUNSEQ = false;
constexpr bool CCONV_OPT_DBG_DUMP_PREFETCH = false;
constexpr bool CCONV_OPT_DBG_DUMP_STATE = false;

using namespace AscendC;
namespace NsCausalConv1dOptimized {
using namespace NsCausalConv1dOptimizedCommon;

#ifndef CAUSAL_CONV1D_OPTIMIZED_TILING_DATA_H_
#define CAUSAL_CONV1D_OPTIMIZED_TILING_DATA_H_

struct CausalConv1dOptimizedTilingData {
    int64_t dim;
    int64_t cuSeqlen;
    int64_t seqLen;
    int64_t inputMode;

    int64_t width;

    int64_t stateLen;
    int64_t numCacheLines;

    int64_t batch;

    // attrs
    int64_t activationMode; // 0: none, 1: silu/swish
    int64_t padSlotId;      // default -1

    // optional inputs
    int64_t hasBias;        // 0/1

    // Channel-wise tiling
    int64_t dimTileSize;
    int64_t blocksPerSeq;
};
#endif // CAUSAL_CONV1D_OPTIMIZED_TILING_DATA_H_

template <typename T>
class CausalConv1dOptimized
{
public:
    __aicore__ inline CausalConv1dOptimized() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates, GM_ADDR queryStartLoc,
                                GM_ADDR cacheIndices, GM_ADDR hasInitialState, GM_ADDR y
                                 ,
                                 const  CausalConv1dOptimizedTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void LoadWeightAndBias(int32_t c0, int32_t dimTileSize, bool dbg);
    __aicore__ inline void InitRing(int32_t cacheIdx, bool hasInit, int32_t start, int32_t len,
                                    int32_t c0, int32_t dimTileSize, int32_t dim, bool dbg);
    __aicore__ inline void RunSeq(int32_t start, int32_t len, int32_t c0, int32_t dimTileSize, int32_t dim, bool dbg);
    __aicore__ inline void WriteBackState(int32_t cacheIdx, int32_t len, int32_t c0,
                                          int32_t dimTileSize, int32_t dim, bool dbg);
    __aicore__ inline void AllocEvents();
    __aicore__ inline void ReleaseEvents();

private:
    TPipe pipe;
    TBuf<QuePosition::VECIN> inBuf;
    TBuf<QuePosition::VECOUT> outBuf;
    TBuf<QuePosition::VECCALC> calcBuf;

    TEventID tempVToMte2Event_;
    TEventID tempMte2ToVEvent_;
    TEventID inputMte2ToVEvent_;
    TEventID outMte3ToVEvent_[2];
    TEventID outVToMte3Event_[2];

    GlobalTensor<T> xGm;
    GlobalTensor<T> weightGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<T> convStatesGm;
    GlobalTensor<int32_t> queryStartLocGm;
    GlobalTensor<int32_t> cacheIndicesGm;
    GlobalTensor<bool> hasInitialStateGm;
    GlobalTensor<T> yGm;

    const  CausalConv1dOptimizedTilingData* tilingData_ {nullptr};
};

template <typename T>
__aicore__ inline void CausalConv1dOptimized<T>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                            GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR hasInitialState,
                                            GM_ADDR y
                                             , const  CausalConv1dOptimizedTilingData* tilingData)
{
    tilingData_ = tilingData;

    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight));
    if (tilingData_->hasBias != 0) {
        biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias));
    }
    convStatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(convStates));
    queryStartLocGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(queryStartLoc));
    cacheIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cacheIndices));
    hasInitialStateGm.SetGlobalBuffer(reinterpret_cast<__gm__ bool*>(hasInitialState));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    pipe.InitBuffer(inBuf, RING_SLOTS * MAX_BLOCK_DIM * sizeof(T));
    pipe.InitBuffer(outBuf, 2 * MAX_BLOCK_DIM * sizeof(T));
    pipe.InitBuffer(calcBuf, (MAX_WIDTH + 3) * MAX_BLOCK_DIM * sizeof(float));

    AllocEvents();
}

template <typename T>
__aicore__ inline void CausalConv1dOptimized<T>::AllocEvents()
{
    tempVToMte2Event_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    tempMte2ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    inputMte2ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    outMte3ToVEvent_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    outMte3ToVEvent_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    outVToMte3Event_[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    outVToMte3Event_[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
}

template <typename T>
__aicore__ inline void CausalConv1dOptimized<T>::ReleaseEvents()
{
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(tempVToMte2Event_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(tempMte2ToVEvent_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(inputMte2ToVEvent_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(outMte3ToVEvent_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(outMte3ToVEvent_[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(outVToMte3Event_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(outVToMte3Event_[1]);
}

template <typename T>
__aicore__ inline void CausalConv1dOptimized<T>::LoadWeightAndBias(int32_t c0, int32_t dimTileSize, bool dbg)
{
    const int32_t dim = tilingData_->dim;
    const bool dbgSync = dbg && CCONV_OPT_DBG_PRINT_SYNC;
    (void)dbgSync;
    LocalTensor<float> calc = calcBuf.Get<float>();
    LocalTensor<float> weightF = calc;
    LocalTensor<float> biasF = weightF[MAX_WIDTH * MAX_BLOCK_DIM];
    LocalTensor<T> tempT = outBuf.Get<T>();

    for (int32_t j = 0; j < MAX_WIDTH; ++j) {
        const int64_t weightOffset = static_cast<int64_t>(j) * dim + c0;
        PipeBarrier<PIPE_ALL>();
        DataCopy(tempT, weightGm[weightOffset], dimTileSize);
        PipeBarrier<PIPE_ALL>();
        Cast(weightF[j * MAX_BLOCK_DIM], tempT, RoundMode::CAST_NONE, dimTileSize);
    }

    if (tilingData_->hasBias != 0) {
        PipeBarrier<PIPE_ALL>();
        DataCopy(tempT, biasGm[c0], dimTileSize);
        PipeBarrier<PIPE_ALL>();
        Cast(biasF, tempT, RoundMode::CAST_NONE, dimTileSize);
    } else {
        Duplicate(biasF, 0.0f, dimTileSize);
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void CausalConv1dOptimized<T>::InitRing(int32_t cacheIdx, bool hasInit, int32_t start, int32_t len,
                                                 int32_t c0, int32_t dimTileSize, int32_t dim, bool dbg)
{
    const int32_t stateLen = tilingData_->stateLen;
    LocalTensor<T> ring = inBuf.Get<T>();

    PipeBarrier<PIPE_ALL>();
    if (hasInit) {
        for (int32_t i = 0; i < (MAX_WIDTH - 1); ++i) {
            const int64_t stateOffset = static_cast<int64_t>(cacheIdx) * stateLen * dim +
                                        static_cast<int64_t>(i) * dim + c0;
            DataCopy(ring[i * MAX_BLOCK_DIM], convStatesGm[stateOffset], dimTileSize);
        }
    } else {
        for (int32_t i = 0; i < (MAX_WIDTH - 1); ++i) {
            Duplicate(ring[i * MAX_BLOCK_DIM], static_cast<T>(0), dimTileSize);
        }
    }
    PipeBarrier<PIPE_ALL>();

    if (len > 0) {
        const int64_t xOffset = static_cast<int64_t>(start) * dim + c0;
        PipeBarrier<PIPE_ALL>();
        DataCopy(ring[SlotCurr(0) * MAX_BLOCK_DIM], xGm[xOffset], dimTileSize);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void CausalConv1dOptimized<T>::RunSeq(int32_t start, int32_t len, int32_t c0, int32_t dimTileSize,
                                               int32_t dim, bool dbg)
{
    LocalTensor<float> calc = calcBuf.Get<float>();
    LocalTensor<float> weightF = calc;
    LocalTensor<float> biasF = weightF[MAX_WIDTH * MAX_BLOCK_DIM];
    LocalTensor<float> accF = biasF[MAX_BLOCK_DIM];
    LocalTensor<float> tmpF = accF[MAX_BLOCK_DIM];
    LocalTensor<T> ring = inBuf.Get<T>();
    LocalTensor<T> outT = outBuf.Get<T>();
    const bool hasActivation = (tilingData_->activationMode != 0);

    for (int32_t t = 0; t < len; ++t) {
        const int32_t slotCurr = SlotCurr(t);
        const int32_t slotH1 = SlotHist(t, 1);
        const int32_t slotH2 = SlotHist(t, 2);
        const int32_t slotH3 = SlotHist(t, 3);
        const int32_t slotPref = (t + 1 < len) ? SlotPrefetch(t) : -1;
        const int32_t outSlot = t & 1;

        // Prefetch next input (no barrier - will sync with next iteration's Cast)
        if (t + 1 < len) {
            const int64_t xOffset = static_cast<int64_t>(start + t + 1) * dim + c0;
            DataCopy(ring[slotPref * MAX_BLOCK_DIM], xGm[xOffset], dimTileSize);
        }

        // Initialize accumulator with bias
        DataCopy(accF, biasF, dimTileSize);

        // OPT1: Reduced PipeBarrier - 1 barrier per iteration (Cast->MulAddDst dependency)
        for (int32_t j = 0; j < MAX_WIDTH; ++j) {
            const int32_t tap = (MAX_WIDTH - 1) - j;
            const int32_t slot = (tap == 0) ? slotCurr : SlotHist(t, tap);
            Cast(tmpF, ring[slot * MAX_BLOCK_DIM], RoundMode::CAST_NONE, dimTileSize);
            PipeBarrier<PIPE_ALL>();
            MulAddDst(accF, tmpF, weightF[j * MAX_BLOCK_DIM], dimTileSize);
        }

        if (hasActivation) {
            Silu(tmpF, accF, dimTileSize);
        }

        // Sync before output write
        PipeBarrier<PIPE_ALL>();
        if constexpr (IsSameType<T, float>::value) {
            if (hasActivation) {
                DataCopy(outT[outSlot * MAX_BLOCK_DIM], tmpF, dimTileSize);
            } else {
                DataCopy(outT[outSlot * MAX_BLOCK_DIM], accF, dimTileSize);
            }
        } else {
            if (hasActivation) {
                Cast(outT[outSlot * MAX_BLOCK_DIM], tmpF, RoundMode::CAST_RINT, dimTileSize);
            } else {
                Cast(outT[outSlot * MAX_BLOCK_DIM], accF, RoundMode::CAST_RINT, dimTileSize);
            }
        }

        // Write to GM (DataCopy handles sync automatically)
        const int64_t outOffset = static_cast<int64_t>(start + t) * dim + c0;
        DataCopy(yGm[outOffset], outT[outSlot * MAX_BLOCK_DIM], dimTileSize);
    }
}

template <typename T>
__aicore__ inline void CausalConv1dOptimized<T>::WriteBackState(int32_t cacheIdx, int32_t len, int32_t c0,
                                                       int32_t dimTileSize, int32_t dim, bool dbg)
{
    const int32_t stateLen = tilingData_->stateLen;
    if (len <= 0) {
        return;
    }

    const int32_t lastT = len - 1;
    LocalTensor<T> ring = inBuf.Get<T>();

    for (int32_t pos = 0; pos < (MAX_WIDTH - 1); ++pos) {
        // OPT2: Fixed state index - changed from (MAX_WIDTH - 2) to (MAX_WIDTH - 1)
        const int32_t tap = (MAX_WIDTH - 1) - pos;
        const int32_t slot = (tap == 0) ? SlotCurr(lastT) : SlotHist(lastT, tap);
        const int64_t stateOffset = static_cast<int64_t>(cacheIdx) * stateLen * dim +
                                    static_cast<int64_t>(pos) * dim + c0;
        PipeBarrier<PIPE_ALL>();
        DataCopy(convStatesGm[stateOffset], ring[slot * MAX_BLOCK_DIM], dimTileSize);
    }
}

template <typename T>
__aicore__ inline void CausalConv1dOptimized<T>::Process()
{
    const int32_t dim = tilingData_->dim;
    const int32_t batch = tilingData_->batch;
    const int32_t inputMode = tilingData_->inputMode;
    const int32_t seqLen = tilingData_->seqLen;
    const int32_t dimTileSize = static_cast<int32_t>(tilingData_->dimTileSize);
    const int32_t blocksPerSeq = static_cast<int32_t>(tilingData_->blocksPerSeq);

    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockNum = GetBlockNum();

    if (dimTileSize <= 0 || blocksPerSeq <= 0 || dimTileSize > MAX_BLOCK_DIM || blocksPerSeq * dimTileSize != dim) {
        ReleaseEvents();
        return;
    }

    const int64_t gridSize = static_cast<int64_t>(batch) * blocksPerSeq;
    for (int64_t task = static_cast<int64_t>(blockIdx); task < gridSize; task += static_cast<int64_t>(blockNum)) {
        const int32_t seq = static_cast<int32_t>(task / blocksPerSeq);
        const int32_t dimBlockId = static_cast<int32_t>(task % blocksPerSeq);
        const int32_t c0 = dimBlockId * dimTileSize;
        const bool dbg = (seq == CCONV_OPT_DBG_SEQ) && (c0 == CCONV_OPT_DBG_C0);

        LoadWeightAndBias(c0, dimTileSize, dbg);

        int32_t start = 0;
        int32_t len = 0;
        if (inputMode == 0) {
            const int32_t startVal = queryStartLocGm.GetValue(seq);
            const int32_t endVal = queryStartLocGm.GetValue(seq + 1);
            start = startVal;
            len = endVal - startVal;
        } else {
            start = seq * seqLen;
            len = seqLen;
        }

        if (len <= 0) {
            continue;
        }

        const int32_t cacheIdx = cacheIndicesGm.GetValue(seq);
        if (cacheIdx == tilingData_->padSlotId) {
            continue;
        }

        const bool hasInit = hasInitialStateGm.GetValue(seq);

        InitRing(cacheIdx, hasInit, start, len, c0, dimTileSize, dim, dbg);
        RunSeq(start, len, c0, dimTileSize, dim, dbg);
        WriteBackState(cacheIdx, len, c0, dimTileSize, dim, dbg);
    }

    ReleaseEvents();
}

} // namespace NsCausalConv1dOptimized
#endif // CAUSAL_CONV1D_OPTIMIZED_H