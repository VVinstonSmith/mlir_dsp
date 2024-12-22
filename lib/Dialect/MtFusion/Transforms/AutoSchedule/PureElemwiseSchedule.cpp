//===- PureElemwiseSchedule.cpp -- Auto-schedule fused kernels --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements auto schedule policy for pure elementwise kernels.
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/PureElemwiseSchedule.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/AutoScheduleAttrDefs.h"
#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/TransformOps/MtFusionTransformOps.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

#define DEBUG_TYPE "mtfusion-auto-schedule"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] [Pure Elemwise] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::mtfusion;

namespace {
/// Tiling Data is organized as:
///   1. Tiling Key
///   2. UB Tile Size
///   3. UB Buffer Size
constexpr size_t kTilingKeyPos = 0;
constexpr size_t kUBTileSizePos = 1;
constexpr size_t kUBBufferSizePos = 2;

/// Tiling Key
constexpr int64_t kTilingCaseKey100 = 100;

} // namespace

//===----------------------------------------------------------------------===//
// PureElemwiseScheduler
//===----------------------------------------------------------------------===//

TilingComputeFn PureElemwiseScheduler::calculateTilingImpl() {
  return [](KernelInfo *kernelInfo,
            ExprBuilder *opBuilder) -> TilingFnResultTy {
    OpBuilder::InsertionGuard g(*opBuilder);

    int64_t maxBufferCnt = kernelInfo->maxBufferCnt;
    assert(maxBufferCnt > 0 && "buffer count should be greater than zero!");

    // Calculate tiling data.
    MLIRContext *ctx = opBuilder->getContext();
    // The number of element that can be store on unified buffer.

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

    Expr const100 = opBuilder->createConstExpr(kTilingCaseKey100);
    Expr tilingKeyExpr = (ubAvailableNumInSmallestTypeBits > 0) * const100;

    auto tilingDataType = IntegerType::get(ctx, 64);
    TilingData tilingData0 =
        TilingData(std::move(tilingKeyExpr), tilingDataType);
    TilingData tilingData1 =
        TilingData(std::move(ubAvailableNumInSmallestTypeBits), tilingDataType);
    TilingData tilingData2 = TilingData(
        alignedBufferSizeInBits.floorDiv(kNumBitsInByte), tilingDataType);

    // Build tiling struct.
    TilingStruct s;
    s.push_back(std::make_unique<TilingData>(std::move(tilingData0)));
    s.push_back(std::make_unique<TilingData>(std::move(tilingData1)));
    s.push_back(std::make_unique<TilingData>(std::move(tilingData2)));

    // Set tiling keys.
    TilingCases c;
    c.insert(kTilingCaseKey100);

    return TilingFnResultTy{std::move(c), std::move(s)};
  };
}

LogicalResult PureElemwiseScheduler::createScheduleImpl(TilingKey key,
                                                        OpBuilder &opBuilder) {
  TilingInfo *tilingInfo = getTilingInfo();
  assert(tilingInfo != nullptr);

  // Pure Elemwise only have one tiling case
  if (key != kTilingCaseKey100)
    return failure();

  // Get handles to tiling data.
  ValueHandles tilingDataHandles =
      getTilingStructHandles(tilingInfo->getTilingStruct(), opBuilder);

  // Step 1: Cache read input arguments.
  CacheIOResult cacheReadResult = {getOpsWithIdentifier(
      kCacheReadTagName, IdentifierType::kAttribute, opBuilder)};

  // Step 2: Cache write kernel results.
  CacheIOResult cacheWriteResult = {getOpsWithIdentifier(
      kCacheWriteTagName, IdentifierType::kAttribute, opBuilder)};

  // Step 3: Tile cache writes using `scf.forall` op.
  ValueHandles splitCachedOps = splitHandle(
      cacheWriteResult.cachedOps, getKernelInfo()->numOutputs, opBuilder);
  ForallTilingResult tileUsingForAllResult =
      tileUsingForAll(splitCachedOps, tilingInfo->getBlockDim(), opBuilder);

  // Step 4: Fuse independent `scf.forall` ops.
  ValueHandle *fusedLoop = fuseLoops(tileUsingForAllResult.loops, opBuilder);
  // Handle to cached ops is invalidated after loop fuse, needs rematching.
  cacheWriteResult.cachedOps->setStatus(HandleStatus::kNeedsRematch);

  // Step 5: Fuse producers into `scf.forall` op.
  // We wish to fuse producers ops by reverse topological ordering.
  MatchOptions matchOptions;
  matchOptions.needsReverse = true;
  ValueHandle *producerOps =
      getOpsWithIdentifier(kIntermediateProducerTagName,
                           IdentifierType::kAttribute, opBuilder, matchOptions);
  ValueHandles targetsToFuseInto = {producerOps, cacheReadResult.cachedOps};
  ValueHandles fusedLoopList = {fusedLoop};
  fuseIntoContaining(targetsToFuseInto, fusedLoopList, opBuilder,
                     /*duplicateProducers=*/true,
                     /*applyCanonicalizeAfterEachFusion=*/true);

  // Step 6: Tile cache writes again using `scf.for` op.
  splitCachedOps = splitHandle(cacheWriteResult.cachedOps,
                               getKernelInfo()->numOutputs, opBuilder);
  // For Pure Elemwise schedule, the tile size should be one dimensional
  auto ubTilingDataHandle =
      ValueHandleFoldResults{tilingDataHandles[kUBTileSizePos]};
  ForTilingResult tileUsingForResult =
      tileUsingFor(splitCachedOps, ubTilingDataHandle, opBuilder);

  // Step 7: Apply canonicalize patterns.
  //         Disabled `kSimplifyTrivialLoops` because loop handles might be
  //         invalidate if the tiled loop is trivial during compile-time
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

  // Step 8: Fuse independent `scf.for` ops.
  auto loops = llvm::map_to_vector(tileUsingForResult.loops,
                                   [](ValueHandles hs) { return hs.front(); });
  fusedLoop = fuseLoops(loops, opBuilder);
  // Handle are invalidated after loop fuse, needs rematching.
  fusedLoop->setStatus(HandleStatus::kNeedsRematch);
  cacheWriteResult.cachedOps->setStatus(HandleStatus::kNeedsRematch);

  // Step 9: Fuse producers into `scf.for` op.
  fusedLoopList = {fusedLoop};
  fuseIntoContaining(targetsToFuseInto, fusedLoopList, opBuilder,
                     /*duplicateProducers=*/true,
                     /*applyCanonicalizeAfterEachFusion=*/true);

  // Step 10: Set buffer size.
  ValueHandles targetsToSetBufferSize = {producerOps,
                                         cacheReadResult.cachedOps};
  TilingData *bufferSize = tilingInfo->getTilingData(kUBBufferSizePos);
  assert(bufferSize->isConst() && "buffer size should be const");
  uint64_t bufferSizeConst = bufferSize->getConst();
  if (bufferSizeConst == 0u)
    return getToBeScheduledKernel().emitError(
        "Buffer size is less than or equal to zero. Possibly because there is "
        "not enough space on local memory!");

  // Rematch handles to make sure they are valid.
  setStatusTo(targetsToSetBufferSize, HandleStatus::kNeedsRematch);
  SetBufferSizeOptions bufferSizeOptions{transform::SetBufferSizeMode::kPerByte,
                                         getKernelInfo()->smallestElementType};
  setBufferSize(targetsToSetBufferSize, bufferSizeConst, opBuilder,
                bufferSizeOptions);
  return success();
}