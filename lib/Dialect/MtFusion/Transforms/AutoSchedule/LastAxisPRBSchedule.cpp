//===- LastAxisPRBSchedule.cpp -- Auto-schedule fused kernels --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements auto schedule policy for last axis pointwise,
// broadcast, and reduction kernels.
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/LastAxisPBRSchedule.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/AutoScheduleAttrDefs.h"
#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/Utils/Utils.h"
#include "mtir/Dialect/Utils/Util.h"

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#include <algorithm>
#include <functional>
#include <numeric>

#define DEBUG_TYPE "mtfusion-auto-schedule"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] [Last Axis PBR] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::mtfusion;

LogicalResult LastAxisPBRScheduler::analyzeAndVerifyKernelImpl() {
  // Collect base information first.
  if (failed(SchedulerBase::analyzeAndVerifyKernelImpl())) {
    return failure();
  }
  auto *kernelInfo = dyn_cast_or_null<LastAxisPBRKernelInfo>(getKernelInfo());
  assert(kernelInfo != nullptr);
  auto idxWithMaxRank = kernelInfo->getInputValueIdxWithMaxRank();
  if (!idxWithMaxRank.has_value()) {
    return getOriginalKernel()->emitError(
        "Cannot find a single input value with shaped type!");
  }
  auto [inputValueIdx, maxRank] = idxWithMaxRank.value();
  kernelInfo->inputValueIdxWithHighestOrderDim = inputValueIdx;
  // Verify Reduction Op
  const auto &reductionOpsInfo = kernelInfo->reductionOps;
  if (!reductionOpsInfo.empty()) {
    kernelInfo->tilebaleReductionDimSize =
        (*(reductionOpsInfo.begin())).second.reductionDims.size();
    for (auto [_, info] : llvm::drop_begin(kernelInfo->reductionOps)) {
      if (kernelInfo->tilebaleReductionDimSize != info.reductionDims.size()) {
        return getOriginalKernel()->emitError(
            "LastPBR currently don't support reduction ops with different "
            "reduction dims!");
      }
    }
  }
  if (kernelInfo->tilebaleReductionDimSize > 1) {
    return getOriginalKernel()->emitError(
        "LastPBR currently don't support reduction with more than one "
        "dimensions!");
  }
  LDBG("Number of tileable reduction axes is "
       << kernelInfo->tilebaleReductionDimSize);
  kernelInfo->tileableParallelDimSize =
      maxRank - kernelInfo->tilebaleReductionDimSize;
  if (kernelInfo->tileableParallelDimSize <= 0) {
    return getOriginalKernel()->emitError("Parallel axes size is less than 1!");
  }
  LDBG("Number of tileable parallel axes is "
       << kernelInfo->tileableParallelDimSize);
  kernelInfo->totalTilableDimSize = maxRank;
  return success();
}

TilingComputeFn LastAxisPBRScheduler::calculateTilingImpl() {
  return [](KernelInfo *kernelInfo,
            ExprBuilder *opBuilder) -> TilingFnResultTy {
    OpBuilder::InsertionGuard g(*opBuilder);
    auto *lastAxisPBRInfo = dyn_cast_or_null<LastAxisPBRKernelInfo>(kernelInfo);
    assert(lastAxisPBRInfo != nullptr);

    // The number of tiling cases is equal to the number of tileable axes.
    size_t numTilingCases = lastAxisPBRInfo->totalTilableDimSize;
    assert(numTilingCases > 0 &&
           "The number of tileable axes should be greater than 0");
    auto maxBufferCnt = kernelInfo->maxBufferCnt;
    assert(maxBufferCnt > 0 && "buffer count should be greater than zero!");

    // Calculate tiling data.
    MLIRContext *ctx = opBuilder->getContext();

    // Get all dimension size
    size_t inputValueIdx = lastAxisPBRInfo->inputValueIdxWithHighestOrderDim;
    auto dims =
        opBuilder->createDimSymbolExprs(inputValueIdx, 0, numTilingCases);

    // Align dimension
    std::optional<KernelInfo::DimAndAlignment> alignmentInfo =
        lastAxisPBRInfo->getAlignments();
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

    // TODO: Refactor. This code is straight copied from AnyPB schedule.

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

bool LastAxisPBRScheduler::isSplitDTilingCase(TilingKey key) const {
  auto *lastAxisPBRInfo =
      dyn_cast_or_null<LastAxisPBRKernelInfo>(getKernelInfo());
  assert(lastAxisPBRInfo != nullptr);
  return key == static_cast<int64_t>(lastAxisPBRInfo->totalTilableDimSize) - 1;
}

LogicalResult LastAxisPBRScheduler::createScheduleImpl(TilingKey key,
                                                       OpBuilder &opBuilder) {
  if (isSplitDTilingCase(key)) {
    LDBG("Generating schedule for split-d tiling case, with key=" << key);
    return createScheduleImplForSplitD(opBuilder);
  }
  LDBG("Generating schedule for split-n tiling case, with key=" << key);
  return createScheduleImplForSplitN(key, opBuilder);
}

size_t LastAxisPBRScheduler::getReductionTilingFactorTilingDataIdx() const {
  const auto *lastAxisPBRInfo =
      dyn_cast_or_null<LastAxisPBRKernelInfo>(getKernelInfo());
  assert(lastAxisPBRInfo != nullptr);
  return lastAxisPBRInfo->tileableParallelDimSize + 1;
}

ValueHandleFoldResults LastAxisPBRScheduler::getTilingFactorsForParallelOp(
    const TilingInfo *tilingInfo, ValueHandles tilingDataHandles,
    size_t parallelDims, TilingAxesKind tilingAxesKind,
    const SmallVector<TilingData *> &tilingData) const {
  if (tilingAxesKind == TilingAxesKind::kParallel) {
    assert(!tilingData.empty());

    ValueHandleFoldResults results;
    // Drop the first tiling data, which is tiling key
    auto tilingFactorsHandles =
        llvm::to_vector(llvm::drop_begin(tilingDataHandles));
    auto tileSizeTilingData = llvm::to_vector(llvm::drop_begin(tilingData));
    for (size_t idx = 0; idx < parallelDims; ++idx) {
      TilingData *td = tileSizeTilingData[idx];
      // If the tiling factor is a const 1, tile it with constant value.
      // Otherwise the constantize might fail as IR gets complicated.
      if (td->isConst() && td->getConst() == 1) {
        results.push_back(ValueHandleFoldResult(1, getContext()));
      } else {
        results.push_back(ValueHandleFoldResult{tilingFactorsHandles[idx]});
      }
    }
    return results;
  }

  assert(tilingInfo != nullptr);
  TilingData *ubTileSizeDimD =
      tilingInfo->getTilingData(getReductionTilingFactorTilingDataIdx());
  assert(ubTileSizeDimD->isConst());

  ValueHandleFoldResults tileSizes(parallelDims,
                                   ValueHandleFoldResult(0, getContext()));
  tileSizes.back() =
      ValueHandleFoldResult(ubTileSizeDimD->getConst(), getContext());
  return tileSizes;
}

std::vector<int64_t> LastAxisPBRScheduler::getTilingFactorsForReductionOp(
    const TilingInfo *tilingInfo, size_t parallelDims) const {
  assert(tilingInfo != nullptr);
  // TODO: Use dynamic value after tileReductionUsingFor support dynamic shape
  // The reduction tile size is stored after tiling key and the tiling sizes
  const TilingData *ubTileSizeDimD =
      tilingInfo->getTilingData(getReductionTilingFactorTilingDataIdx());
  assert(ubTileSizeDimD->isConst());
  std::vector<int64_t> rfactorSizes(parallelDims + 1, 0);
  rfactorSizes.back() = ubTileSizeDimD->getConst();
  return rfactorSizes;
}

void LastAxisPBRScheduler::applyCanonicalization(OpBuilder &opBuilder) {
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
}

LogicalResult
LastAxisPBRScheduler::createScheduleImplForSplitN(TilingKey key,
                                                  OpBuilder &opBuilder) {
  TilingInfo *tilingInfo = getTilingInfo();
  assert(tilingInfo != nullptr);

  const auto *lastAxisPBRInfo =
      dyn_cast_or_null<LastAxisPBRKernelInfo>(getKernelInfo());
  assert(lastAxisPBRInfo != nullptr);

  // Get handles to tiling data.
  ValueHandles tilingDataHandles =
      getTilingStructHandles(tilingInfo->getTilingStruct(), opBuilder);

  // Step 1: Cache read input arguments.
  CacheIOResult cacheReadResult = {getOpsWithIdentifier(
      kCacheReadTagName, IdentifierType::kAttribute, opBuilder)};

  // Step 2: Cache write kernel results.
  CacheIOResult cacheWriteResult = {getOpsWithIdentifier(
      kCacheWriteTagName, IdentifierType::kAttribute, opBuilder)};

  ValueHandle *producerOps = nullptr;
  auto tileSizes = getTilingFactorsForParallelOp(
      tilingInfo, tilingDataHandles, lastAxisPBRInfo->tileableParallelDimSize,
      TilingAxesKind::kParallel, tilingInfo->getTilingStruct());

  // Step 3: Tile cache writes using `scf.for` op.
  ValueHandles splitCachedOps = splitHandle(
      cacheWriteResult.cachedOps, getKernelInfo()->numOutputs, opBuilder);
  ForTilingResult tileUsingForResult =
      tileUsingFor(splitCachedOps, tileSizes, opBuilder);

  applyCanonicalization(opBuilder);

  // Step 4: Fuse independent `scf.for` ops for every dimension.
  ValueHandles fusedLoops;
  for (size_t dimIdx = 0; dimIdx < lastAxisPBRInfo->tileableParallelDimSize;
       ++dimIdx) {
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
  producerOps =
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

  // Final Step: Set buffer size.
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

mtfusion::detail::ForReductionTilingResult
LastAxisPBRScheduler::tileReductionAndFuseProducers(
    const LastAxisPBRKernelInfo *kernelInfo, const TilingInfo *tilingInfo,
    OpBuilder &opBuilder) {
  ForReductionTilingResult combinedResult;
  // Loop over all reduction ops, tiling one at each time.
  for (auto [_, reductionInfo] : kernelInfo->reductionOps) {
    auto reduceOpsHandle = ValueHandles{getOpsWithIdentifier(
        reductionInfo.key, IdentifierType::kAttribute, opBuilder)};
    ForReductionTilingResult reductionTileResult = tileReductionUsingFor(
        reduceOpsHandle,
        getTilingFactorsForReductionOp(tilingInfo,
                                       /*parallelDims=*/reductionInfo.numLoops -
                                           reductionInfo.reductionDims.size()),
        opBuilder,
        /*multiReduceNum=*/reductionInfo.numResults);
    combinedResult.partialReductionOp.push_back(
        reductionTileResult.partialReductionOp.front());
    combinedResult.finalReductionOp.push_back(
        reductionTileResult.finalReductionOp.front());
    combinedResult.reductionInitOp.push_back(
        reductionTileResult.reductionInitOp.front());
    combinedResult.loops.push_back(reductionTileResult.loops.front());
  }

  // Fuse `linalg.reduce` op's fusable producers into its own partial loop.
  if (kernelInfo->fusableProducerInfos.count(
          KernelInfo::ConsumerType::kReduction))
    fuseProducersIntoAxisDLoop(combinedResult.loops,
                               kernelInfo->fusableProducerInfos.at(
                                   KernelInfo::ConsumerType::kReduction),
                               opBuilder);
  return combinedResult;
}

LogicalResult
LastAxisPBRScheduler::createScheduleImplForSplitD(OpBuilder &opBuilder) {
  TilingInfo *tilingInfo = getTilingInfo();
  assert(tilingInfo != nullptr);

  const auto *lastAxisPBRInfo =
      dyn_cast_or_null<LastAxisPBRKernelInfo>(getKernelInfo());
  assert(lastAxisPBRInfo != nullptr);

  // Get handles to tiling data.
  ValueHandles tilingDataHandles =
      getTilingStructHandles(tilingInfo->getTilingStruct(), opBuilder);

  // Step 1: Cache read input arguments.
  CacheIOResult cacheReadResult = {getOpsWithIdentifier(
      kCacheReadTagName, IdentifierType::kAttribute, opBuilder)};

  // Step 2: Cache write kernel results.
  CacheIOResult cacheWriteResult = {getOpsWithIdentifier(
      kCacheWriteTagName, IdentifierType::kAttribute, opBuilder)};

  ValueHandle *producerOps = nullptr;
  auto tileSizes = getTilingFactorsForParallelOp(
      tilingInfo, tilingDataHandles, lastAxisPBRInfo->tileableParallelDimSize,
      TilingAxesKind::kParallel, tilingInfo->getTilingStruct());

  // Step 3: Tile cache writes' parallel axes using `scf.for` op.
  ValueHandles splitCachedOps = splitHandle(
      cacheWriteResult.cachedOps, getKernelInfo()->numOutputs, opBuilder);
  ForTilingResult tileUsingForResult =
      tileUsingFor(splitCachedOps, tileSizes, opBuilder);

  applyCanonicalization(opBuilder);

  // Step 4: Fuse independent `scf.for` ops for every dimension.
  ValueHandles fusedLoops;
  for (size_t dimIdx = 0; dimIdx < lastAxisPBRInfo->tileableParallelDimSize;
       ++dimIdx) {
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
  producerOps =
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

  // Tile axis D.
  applyCanonicalization(opBuilder);

  // Tile Reduction Ops and fuse fusable producers.
  ForReductionTilingResult reductionTileResult =
      tileReductionAndFuseProducers(lastAxisPBRInfo, tilingInfo, opBuilder);

  // Tile (N, D) shaped cache writes' axis D and fuse fusable producers.
  if (lastAxisPBRInfo->fusableProducerInfos.count(
          KernelInfo::ConsumerType::kOutput)) {
    // Rematch cached ops in case previous transform invalidated it
    cacheWriteResult.cachedOps->setStatus(HandleStatus::kNeedsRematch);
    ValueHandles splitCachedOps = splitHandle(
        cacheWriteResult.cachedOps, lastAxisPBRInfo->numOutputs, opBuilder);
    for (const KernelInfo::FusableProducers &fp :
         lastAxisPBRInfo->fusableProducerInfos.at(
             KernelInfo::ConsumerType::kOutput)) {
      assert(fp.idx < lastAxisPBRInfo->numOutputs);
      ValueHandles targetCachedOp = {splitCachedOps[fp.idx]};
      auto outputType =
          lastAxisPBRInfo->outputTypes[lastAxisPBRInfo->outputOrdering[fp.idx]];
      assert(isa<RankedTensorType>(outputType));
      auto tileSizes = getTilingFactorsForParallelOp(
          tilingInfo, tilingDataHandles,
          cast<RankedTensorType>(outputType).getRank(),
          TilingAxesKind::kReduction);
      auto splitDResult = tileUsingFor(targetCachedOp, tileSizes, opBuilder);
      auto loop = splitDResult.loops.back();
      fuseProducersIntoAxisDLoop(loop, {fp}, opBuilder);
    }
  }

  // Set buffer size.
  ValueHandles targetsToSetBufferSize = {producerOps,
                                         cacheReadResult.cachedOps};
  for (const auto &inits : reductionTileResult.reductionInitOp) {
    targetsToSetBufferSize.append(inits);
  }
  targetsToSetBufferSize.append(reductionTileResult.partialReductionOp);
  targetsToSetBufferSize.append(reductionTileResult.finalReductionOp);
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

void LastAxisPBRScheduler::annotateFusableCacheReadsForSplitD(
    const KernelInfo *info, OpBuilder &opBuilder) {
  assert(info);
  for (auto argIdx : info->cacheReadFuncArgIndices) {
    auto targetCacheRead =
        matchByIdentifier(getFuncValue(opBuilder), getCacheReadTag(argIdx),
                          IdentifierType::kAttribute, opBuilder);
    for (const auto &pair : info->fusableProducerInfos) {
      for (const auto &fusableProducer : pair.second) {
        if (!fusableProducer.blockArguments.contains(argIdx)) {
          continue;
        }
        annotateByAttr(targetCacheRead, fusableProducer.groupName, opBuilder);
      }
    }
  }
}

void LastAxisPBRScheduler::fuseProducersIntoAxisDLoop(
    ValueHandles forOps,
    const std::vector<KernelInfo::FusableProducers> &producersInfo,
    OpBuilder &opBuilder) {
  assert(forOps.size() == producersInfo.size());
  MatchOptions options;
  options.needsReverse = true;
  for (auto [loopValueHandle, producerInfo] :
       llvm::zip(forOps, producersInfo)) {
    // We only want to get producers ops that are ancestors to the current loop.
    options.childHandleOrValue = loopValueHandle;
    ValueHandles producerHandles = {
        getOpsWithIdentifier(producerInfo.groupName, IdentifierType::kAttribute,
                             opBuilder, options)};
    ValueHandles handles = {loopValueHandle};
    fuseIntoContaining(producerHandles, handles, opBuilder,
                       /*duplicateProducers=*/true,
                       /*applyCanonicalizeAfterEachFusion=*/true);
  }
}