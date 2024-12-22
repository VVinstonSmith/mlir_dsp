//===- AnyPBSchedule.cpp -- Auto-schedule fused kernels ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements auto schedule policy for any axis pointwise/broadcast
// kernels.
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/AnyPBSchedule.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/AutoScheduleAttrDefs.h"
#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

#define DEBUG_TYPE "mtfusion-auto-schedule"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] [Any PB] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::mtfusion;

LogicalResult AnyPBScheduler::analyzeAndVerifyKernelImpl() {
  auto *kernelInfo = dyn_cast_or_null<AnyPBKernelInfo>(getKernelInfo());
  assert(kernelInfo != nullptr);
  auto idxWithMaxRank = kernelInfo->getInputValueIdxWithMaxRank();
  if (!idxWithMaxRank.has_value()) {
    return getOriginalKernel()->emitError(
        "Cannot find a single input value with shaped type!");
  }
  auto [inputValueIdx, maxRank] = idxWithMaxRank.value();
  kernelInfo->inputValueIdxWithHighestOrderDim = inputValueIdx;
  kernelInfo->tileableDimSize = maxRank;
  return SchedulerBase::analyzeAndVerifyKernelImpl();
}

TilingComputeFn AnyPBScheduler::calculateTilingImpl() {
  return [](KernelInfo *kernelInfo,
            ExprBuilder *opBuilder) -> TilingFnResultTy {
    OpBuilder::InsertionGuard g(*opBuilder);
    auto *anyPBInfo = dyn_cast_or_null<AnyPBKernelInfo>(kernelInfo);
    assert(anyPBInfo != nullptr);

    // The number of tiling cases is equal to the number of tileable axes.
    size_t numTilingCases = anyPBInfo->tileableDimSize;
    assert(numTilingCases > 0 &&
           "The number of tileable axes should be greater than 0");
    auto maxBufferCnt = kernelInfo->maxBufferCnt;
    assert(maxBufferCnt > 0 && "buffer count should be greater than zero!");

    // Calculate tiling data.
    MLIRContext *ctx = opBuilder->getContext();

    // Get all dimension size
    size_t inputValueIdx = anyPBInfo->inputValueIdxWithHighestOrderDim;
    SmallVector<Expr> dims =
        opBuilder->createDimSymbolExprs(inputValueIdx, 0, numTilingCases);

    // Align dimension
    std::optional<KernelInfo::DimAndAlignment> alignmentInfo =
        anyPBInfo->getAlignments();
    if (alignmentInfo.has_value()) {
      auto [idx, alignment] = alignmentInfo.value();
      assert(idx < static_cast<int>(dims.size()));
      LDBG("[Alignment Info] dim: " << idx << " is aligned to: " << alignment);
      dims[idx] = dims[idx].alignTo(alignment);
    }

    Expr smallestTypeBits =
        opBuilder->createConstExpr(kernelInfo->getSmallestElementTypeBits());
    Expr ubMaxSizeInBits = opBuilder->createConstExpr(kUBMaxSizeInBits);
    Expr ubAvailableNumInSmallestTypeBits =
        ubMaxSizeInBits.floorDiv(smallestTypeBits).floorDiv(maxBufferCnt);
    // ub avaliable number align down to block size
    Expr alignedBufferSizeInBits =
        (ubAvailableNumInSmallestTypeBits * smallestTypeBits)
            .alignDown(kUBAlignSizeInBytes * kNumBitsInByte);
    ubAvailableNumInSmallestTypeBits =
        alignedBufferSizeInBits.floorDiv(smallestTypeBits);

    size_t numTilingData =
        numTilingCases + /*numTilingKey=*/1 + /*numBufferSize=*/1;
    TilingStruct s = SmallVector<TilingDataPtr>(numTilingData);
    TilingCases c;

    // The constructed array holds the accumulated number of elements up to a
    // certain dimension.
    auto accumulatedDims =
        tiling::getAccumulatedDims(llvm::to_vector(llvm::reverse(dims)));

    /// For the i-th tiling case, assume that we can load all the data
    /// from the i-th to {N-1}-th axes to UB. In other words,
    /// For Tiling Case N-1:
    /// Tiling Sizes = [1, ..., 1, ubAvailableNum]
    ///
    /// For Tiling Case N-2:
    /// Tiling Sizes = [1, ..., ubAvailableNum / dim_{N-2}, dim_{N-1}]
    ///
    /// For Tiling Case 0:
    /// Tiling Sizes = [ubAvailableNum / (dim_{1} *
    ///                                   dim_{2} * ... * dim_{N-1}),
    ///                 dim1, ..., dim_{N-1}]
    ///
    /// Therefore, the tiling key value selection logic is:
    ///   if (ubAvailableNum <= dim_{N-1})
    ///      tilingKey = N - 1
    ///   else if (ubAvailableNum <= dim_{N-1} * dim_{N-2}:
    ///      tilingKey = N - 2
    ///   ...
    ///   else:
    ///      tilingKey = 0
    /// This can also be constructed using a nested selection statement.
    Expr tilingKey = opBuilder->createConstExpr(0);
    for (const auto &[idx, accumulatedValue] :
         llvm::enumerate(llvm::drop_begin(llvm::reverse(accumulatedDims)))) {
      Expr tilingCase = opBuilder->createConstExpr(idx + 1);
      tilingKey = select(ubAvailableNumInSmallestTypeBits <= accumulatedValue,
                         tilingCase, tilingKey);
    }

    auto tilingDataType = IntegerType::get(ctx, 64);
    Expr ubRemainingNum = ubAvailableNumInSmallestTypeBits;
    for (int64_t dimIdx = numTilingCases - 1; dimIdx >= 0; --dimIdx) {
      c.insert(dimIdx);
      LDBG("Added tiling case: " << dimIdx);
      /// Consider the following three cases:
      ///
      ///                        (b) dimIdx
      ///                         tilingKey
      ///                            |      (c) dimIdx
      ///        (a) dimIdx          |           |
      ///             |              |           |
      /// [dim_{0}, dim_{1}, ..., dim_{N-2}, dim_{N-1}]
      ///
      /// For case (a), since the selected tiling key is larger than the dim
      /// index, we cannot load more than one line of dim_{1}. Thus the tile
      /// size is 1.
      /// For case (b), since the tiling key is equal to the dim index, we
      /// can partially load dim_{N-2} to the UB, and the tiling size is:
      /// ubAvailableNum / dim_{N-2}.
      /// For case (c), since the tiling key is less than the dim index, we
      /// can fully load dim_{N-1}, and the tiling size is dim_{N-1}.
      Expr tilingKeyGreaterThanDim = tilingKey > dimIdx;
      Expr tilingKeyEqualToDim = tilingKey == dimIdx;
      // FIXME: Don't need to cut full-load dimensions.
      Expr tileSize =
          select(tilingKeyGreaterThanDim, opBuilder->createConstExpr(1),
                 select(tilingKeyEqualToDim, ubRemainingNum, dims[dimIdx]));
      s[dimIdx + 1] = std::make_unique<TilingData>(
          TilingData(std::move(tileSize), tilingDataType)); // tile size
      ubRemainingNum = ubRemainingNum.floorDiv(dims[dimIdx]);
    }

    s[0] = std::make_unique<TilingData>(
        TilingData(std::move(tilingKey), tilingDataType)); // tiling key

    // TODO: Move buffer size out of tiling data as it's always a compile-time
    // constant.
    s.back() = std::make_unique<TilingData>(
        TilingData(alignedBufferSizeInBits.floorDiv(kNumBitsInByte),
                   tilingDataType)); // buffer size

    return TilingFnResultTy{std::move(c), std::move(s)};
  };
}

ValueHandleFoldResults
AnyPBScheduler::getTilingFactors(const AnyPBKernelInfo *anyPBInfo,
                                 const SmallVector<TilingData *> &tilingData,
                                 const ValueHandles &tilingDataHandles) const {
  ValueHandleFoldResults results;
  // Drop the first tiling data, which is tiling key
  auto tilingFactorsHandles =
      llvm::to_vector(llvm::drop_begin(tilingDataHandles));
  auto tileSizeTilingData = llvm::to_vector(llvm::drop_begin(tilingData));
  for (size_t idx = 0; idx < anyPBInfo->tileableDimSize; ++idx) {
    TilingData *td = tileSizeTilingData[idx];
    // If the tiling factor is a const 1, tile it with constant value. Otherwise
    // the constantize might fail as IR gets complicated.
    if (td->isConst() && td->getConst() == 1) {
      results.push_back(ValueHandleFoldResult(1, getContext()));
    } else {
      results.push_back(ValueHandleFoldResult{tilingFactorsHandles[idx]});
    }
  }
  return results;
}

LogicalResult AnyPBScheduler::createScheduleImpl(TilingKey key,
                                                 OpBuilder &opBuilder) {
  TilingInfo *tilingInfo = getTilingInfo();
  assert(tilingInfo != nullptr);
  auto *anyPBInfo = dyn_cast_or_null<AnyPBKernelInfo>(getKernelInfo());
  assert(anyPBInfo != nullptr);

  // Get handles to tiling data.
  ValueHandles tilingDataHandles =
      getTilingStructHandles(tilingInfo->getTilingStruct(), opBuilder);

  // Step 1: Cache read input arguments.
  CacheIOResult cacheReadResult = {getOpsWithIdentifier(
      kCacheReadTagName, IdentifierType::kAttribute, opBuilder)};

  // Step 2: Cache write kernel results.
  CacheIOResult cacheWriteResult = {getOpsWithIdentifier(
      kCacheWriteTagName, IdentifierType::kAttribute, opBuilder)};

  // Step 3: Tile cache writes using `scf.for` op.
  ValueHandles splitCachedOps = splitHandle(
      cacheWriteResult.cachedOps, getKernelInfo()->numOutputs, opBuilder);

  auto ubTileSizes = getTilingFactors(anyPBInfo, tilingInfo->getTilingStruct(),
                                      tilingDataHandles);
  ForTilingResult tileUsingForResult =
      tileUsingFor(splitCachedOps, ubTileSizes, opBuilder);

  // Disabled `kSimplifyTrivialLoops` because loop handles might be invalidate
  // if the tiled loop is trivial during compile-time
  applyPatterns(
      getFuncHandle(opBuilder),
      /*patterns=*/
      SmallVector<TransformPatternKind>{
          TransformPatternKind::CSE, TransformPatternKind::CANONICALIZATION,
          TransformPatternKind::MERGE_CONSECUTIVE_INSERT_EXTRACT_SLICE},
      opBuilder,
      /*disablePatterns=*/
      SmallVector<CanonicalizationPatternKind>{
          CanonicalizationPatternKind::kSimplifyTrivialLoops});

  // Step 4: Fuse independent `scf.for` ops for every dimension.
  ValueHandles fusedLoops;
  for (size_t dimIdx = 0; dimIdx < anyPBInfo->tileableDimSize; ++dimIdx) {
    ValueHandles currentLoops;
    llvm::transform(tileUsingForResult.loops, std::back_inserter(currentLoops),
                    [&](ValueHandles vhs) {
                      auto *namedHandle = cast<NamedValueHandle>(vhs[dimIdx]);
                      return getOpsWithIdentifier(namedHandle->getName(),
                                                  IdentifierType::kAttribute,
                                                  opBuilder);
                    });
    fusedLoops.push_back(fuseLoops(currentLoops, opBuilder));
  }

  // Step 5: Coalesce loops starting from outermost loop and normalize it.
  auto *coalescedLoop = coalesceLoops(fusedLoops.front(), opBuilder);
  // normalizeLoop(coalescedLoop, opBuilder);

  // Step 6: Fuse producers into `scf.for` op.
  MatchOptions matchOptions;
  matchOptions.needsReverse = true;
  ValueHandle *producerOps =
      getOpsWithIdentifier(kIntermediateProducerTagName,
                           IdentifierType::kAttribute, opBuilder, matchOptions);
  ValueHandles targetsToFuseInto = {producerOps, cacheReadResult.cachedOps};
  ValueHandles fusedLoopList = {coalescedLoop};
  fuseIntoContaining(targetsToFuseInto, fusedLoopList, opBuilder);
  // Handle to outermost loop is invalidated, needs rematching.
  coalescedLoop->setStatus(HandleStatus::kNeedsRematch);

  // Step 7: Tile block.dim axes and map to forall.
  // auto tileResult =
  //     tileLoop(coalescedLoop, tilingInfo->getBlockDim(), opBuilder,
  //              LoopTileOptions{/*mode=*/LoopTileMode::kFactorMode,
  //                              /*isReorderMode=*/true});
  // normalizeLoop(tileResult.outerLoop, opBuilder);
  // auto mapping =
  //     mtfusion::BlockMappingAttr::get(getContext(), mtfusion::MappingId::DimX);
  // mapForToForall(tileResult.outerLoop, opBuilder,
  //                MapForToForallOptions{mapping, /*annotate_only=*/true});

  // Step 8: Set buffer size.
  ValueHandles targetsToSetBufferSize = {producerOps,
                                         cacheReadResult.cachedOps};
  // The buffer size is the last tiling data
  TilingData *bufferSize = tilingInfo->getTilingData(tilingInfo->size() - 1);
  assert(bufferSize->isConst() && "buffer size should be const");
  uint64_t bufferSizeConst = bufferSize->getConst();
  if (bufferSizeConst == 0u)
    return getToBeScheduledKernel().emitError(
        "Buffer size is less than or equal to zero. Possibly because there is "
        "not enough space on local memory!");

  // Apply canonicalize before setting buffer size to make sure that dead
  // operations are erased.
  applyCanonicalization(opBuilder);
  // Rematch handles to make sure they are valid.
  setStatusTo(targetsToSetBufferSize, HandleStatus::kNeedsRematch);
  SetBufferSizeOptions bufferSizeOptions{transform::SetBufferSizeMode::kPerByte,
                                         getKernelInfo()->smallestElementType};
  setBufferSize(targetsToSetBufferSize, bufferSizeConst, opBuilder,
                bufferSizeOptions);
  return success();
}