//===- KernelInfo.cpp -- Definition for Kernel Info -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements kernel info definition.
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/KernelInfo.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/AutoScheduleAttrDefs.h"
#include "mtir/Dialect/Utils/Util.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mtfusion-auto-schedule"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] [Base Scheduler] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::mtfusion;

//===----------------------------------------------------------------------===//
// KernelInfo
//===----------------------------------------------------------------------===//

void KernelInfo::FusableProducers::dump() {
  LDBG("--- FusableProducers group:" << this->groupName);
  for (const auto *op : operations)
    LDBG("----- operation: " << *op);
  for (const auto &idx : blockArguments)
    LDBG("----- block arg idx: " << idx);
}

std::optional<std::pair<int64_t, int64_t>>
KernelInfo::getInputValueIdxWithMaxRank() {
  int64_t inputValueIdx = -1;
  int64_t maxRank = -1;
  for (const auto &[idx, inputValue] : llvm::enumerate(this->inputValues)) {
    Type inputType = inputValue.getType();
    auto shapedTy = dyn_cast<ShapedType>(inputType);
    if (!shapedTy)
      continue;
    int64_t curRank = shapedTy.getRank();
    if (maxRank == -1 || maxRank <= curRank) {
      maxRank = curRank;
      inputValueIdx = idx;
    }
  }
  if (inputValueIdx == -1) {
    return std::nullopt;
  }
  return std::pair{inputValueIdx, maxRank};
}

int64_t KernelInfo::getSmallestElementTypeBits() {
  assert(smallestElementType != Type());
  LDBG("Smallest tensor element type is " << smallestElementType);
  return smallestElementType.getIntOrFloatBitWidth();
}

std::optional<KernelInfo::DimAndAlignment> KernelInfo::getAlignments() {
  std::optional<KernelInfo::DimAndAlignment> brcAlignment =
      getAlignmentsForBroadcastOp();
  std::optional<KernelInfo::DimAndAlignment> reduceAlignment =
      getAlignmentsForReduceOp();

  if (brcAlignment.has_value() ^ reduceAlignment.has_value()) {
    return brcAlignment.has_value() ? brcAlignment : reduceAlignment;
  }
  if (!brcAlignment.has_value() && !reduceAlignment.has_value()) {
    return std::nullopt;
  }
  // only need to align the lowest dimension
  return brcAlignment.value().first < reduceAlignment.value().first
             ? reduceAlignment
             : brcAlignment;
}

std::optional<KernelInfo::DimAndAlignment>
KernelInfo::getAlignmentsForReduceOp() {
  if (reductionOps.empty())
    return std::nullopt;

  // Assume that all reduction ops have the same rank and reduction dims.
  const auto pair = *(reductionOps.begin());
  auto reductionDims = llvm::to_vector(pair.second.reductionDims);
  llvm::sort(reductionDims);
  if (reductionDims.empty())
    return std::nullopt;

  auto lastReductionDim = reductionDims.back();
  auto totalDims = dyn_cast<linalg::LinalgOp>(pair.first).getNumLoops();
  if (totalDims == 1) {
    // Special Case:
    // If there is only one loop, and it's a reduction loop, no need to align.
    // For example:
    // tensor<15xf32> reduced to tensor<1xf32>, reduction dims is 1.
    return std::nullopt;
  }

  int32_t alignDim;
  if (lastReductionDim == totalDims - 1) {
    // For last axis reduce, need to align the penultimate axis.
    // For example:
    // tensor<?x15xf32> reduced to tensor<?x1xf32>, reduction dims is 1.
    alignDim = lastReductionDim - 1;
  } else {
    // For n-last axis reduce, need to align the last reduce axis.
    // For example:
    // tensor<15x15xf32> reduced to tensor<1x15xf32>, reduction dims is 0.
    alignDim = lastReductionDim;
  }

  assert(alignDim < static_cast<int32_t>(totalDims));
  LDBG("[Alignment Info] dimension to align: " << alignDim);
  TensorType typeToAlign =
      cast<TensorType>(pair.first->getOperand(0).getType());
  typeToAlign =
      typeToAlign.clone(typeToAlign.getShape(),
                        getSmallestElementTypeBits() < kNumBitsInByte
                            ? IntegerType::get(getContext(), kNumBitsInByte)
                            : smallestElementType);
  LDBG("[Alignment Info] type before alignment: " << typeToAlign);

  // Note: the input to `collectAlignUnits` is the axis to which the stride
  // should be aligned. The stride is aligned by aligning the **next**
  // dimension. So in terms of the shape, we need to align the next dimension.
  SmallVector<int32_t> alignDims{static_cast<int32_t>(alignDim)};
  SmallVector<int32_t> alignBytes{static_cast<int32_t>(kUBAlignSizeInBytes)};
  auto alignment =
      hivm::util::collectAlignUnits(alignDims, alignBytes, typeToAlign);
  LDBG("[Alignment Info] alignment unit: " << alignment[alignDim + 1]);
  return std::make_pair(static_cast<int>(alignDim + 1),
                        alignment[alignDim + 1]);
}

std::optional<KernelInfo::DimAndAlignment>
KernelInfo::getAlignmentsForBroadcastOp() {
  if (broadcastOps.empty())
    return std::nullopt;

  TensorType typeWithMaxRankAfterBroadcast;
  SetVector<int64_t> broadcastDims;
  for (auto [op, brcInfo] : broadcastOps) {
    auto currentType = cast<TensorType>(op->getResult(0).getType());
    if (typeWithMaxRankAfterBroadcast == TensorType() ||
        typeWithMaxRankAfterBroadcast.getRank() < currentType.getRank()) {
      typeWithMaxRankAfterBroadcast = currentType;
    }
    broadcastDims.insert(brcInfo.broadcastDims.begin(),
                         brcInfo.broadcastDims.end());
  }
  auto broadcastDimsVec = llvm::to_vector(broadcastDims);
  llvm::sort(broadcastDimsVec);
  if (broadcastDimsVec.empty())
    return std::nullopt;

  int32_t alignDim;
  // The last broadcast dimension needs to be aligned.
  auto lastBroadcastDim = broadcastDimsVec.back();
  auto totalDims = typeWithMaxRankAfterBroadcast.getRank();
  if (lastBroadcastDim == totalDims - 1) {
    // Special Case:
    // If there is only a single broadcast dimension, and it's also the final
    // dimension of the tensor, there is no need to do alignment.
    // For example:
    // tensor<1xf32> to tensor<15xf32>, broadcast dim is 0.
    if (llvm::hasSingleElement(broadcastDimsVec)) {
      return std::nullopt;
    }
    // Otherwise, sill need to align the penultimate broadcast axis.
    alignDim = *(broadcastDimsVec.rbegin() + 1);
  } else {
    alignDim = lastBroadcastDim;
  }

  assert(lastBroadcastDim < typeWithMaxRankAfterBroadcast.getRank());
  LDBG("[Alignment Info] dimension to align: " << alignDim);
  TensorType typeToAlign = typeWithMaxRankAfterBroadcast.clone(
      typeWithMaxRankAfterBroadcast.getShape(),
      getSmallestElementTypeBits() < kNumBitsInByte
          ? IntegerType::get(getContext(), kNumBitsInByte)
          : smallestElementType);
  LDBG("[Alignment Info] type before alignment: " << typeToAlign);

  // Note: the input to `collectAlignUnits` is the axis to which the stride
  // should be aligned. The stride is aligned by aligning the **next**
  // dimension. So in terms of the shape, we need to align the next dimension.
  SmallVector<int32_t> alignDims{static_cast<int32_t>(alignDim)};
  SmallVector<int32_t> alignBytes{static_cast<int32_t>(kUBAlignSizeInBytes)};
  auto alignment =
      hivm::util::collectAlignUnits(alignDims, alignBytes, typeToAlign);
  LDBG("[Alignment Info] alignment unit: " << alignment[alignDim + 1]);
  return std::make_pair(static_cast<int>(alignDim + 1),
                        alignment[alignDim + 1]);
}