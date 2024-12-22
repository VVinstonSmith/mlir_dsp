//===- BlockPureElemwiseSchedule.cpp -- Auto-schedule kernels ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements auto schedule policy for block pure elementwise kernels.
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/BlockPureElemwiseSchedule.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/AutoScheduleAttrDefs.h"
#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/TransformOps/MtFusionTransformOps.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

#define DEBUG_TYPE "mtfusion-auto-schedule"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] [Block Pure Elemwise] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::mtfusion;

namespace {
/// Tiling Data is organized as:
///   1. Tiling Key
///   2. UB Tile Size
///   3. UB Buffer Size
constexpr size_t kTilingKeyPos = 0;
constexpr size_t kUBTileSizePos1 = 1;
constexpr size_t kUBTileSizePos2 = 2;
constexpr size_t kUBBufferSizePos = 3;

/// Tiling Key
constexpr int64_t kTilingCaseKey1000 = 1000;
} // namespace

//===----------------------------------------------------------------------===//
// BlockPureElemwiseScheduler
//===----------------------------------------------------------------------===//

LogicalResult BlockPureElemwiseScheduler::analyzeAndVerifyKernelImpl() {
  auto *kernelInfo = getKernelInfo();
  assert(kernelInfo != nullptr);
  func::FuncOp originalKernel = getOriginalKernel();
  return SchedulerBase::analyzeAndVerifyKernelImpl();
}

TilingComputeFn BlockPureElemwiseScheduler::calculateTilingImpl() {
  return [](KernelInfo *kernelInfo,
            ExprBuilder *opBuilder) -> TilingFnResultTy {
    OpBuilder::InsertionGuard g(*opBuilder);

    auto maxBufferCnt = kernelInfo->maxBufferCnt;
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
    //   Expr blockActualN = opBuilder->createDimSymbolExpr(maxBufferCnt - 2,
    //   1);
    Expr blockSizeN = opBuilder->createConstExpr(256);
    Expr ubAvailableBitsDivBlockN =
        ubAvailableNumInSmallestTypeBits.floorDiv(blockSizeN);
    Expr const1000 = opBuilder->createConstExpr(kTilingCaseKey1000);
    Expr tilingKeyExpr = (ubAvailableNumInSmallestTypeBits > 0) * const1000;
    auto tilingDataType = IntegerType::get(ctx, 64);
    TilingData tilingData0 =
        TilingData(std::move(tilingKeyExpr), tilingDataType);
    TilingData tilingData1 =
        TilingData(std::move(ubAvailableBitsDivBlockN), tilingDataType);
    TilingData tilingData2 =
        TilingData(std::move(ubAvailableNumInSmallestTypeBits), tilingDataType);
    TilingData tilingData3 = TilingData(
        alignedBufferSizeInBits.floorDiv(kNumBitsInByte), tilingDataType);

    // Build tiling struct.
    TilingStruct s;
    s.push_back(std::make_unique<TilingData>(std::move(tilingData0)));
    s.push_back(std::make_unique<TilingData>(std::move(tilingData1)));
    s.push_back(std::make_unique<TilingData>(std::move(tilingData2)));
    s.push_back(std::make_unique<TilingData>(std::move(tilingData3)));

    // Set tiling keys.
    TilingCases c;
    c.insert(kTilingCaseKey1000);

    return TilingFnResultTy{std::move(c), std::move(s)};
  };
}

LogicalResult
BlockPureElemwiseScheduler::createScheduleImpl(TilingKey key,
                                               OpBuilder &opBuilder) {
  TilingInfo *tilingInfo = getTilingInfo();
  assert(tilingInfo != nullptr);
  // Pure Elemwise only have one tiling case
  if (key != kTilingCaseKey1000)
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

  // Step 3: Tile cache writes using `scf.for` op.
  ValueHandles splitCachedOps = splitHandle(
      cacheWriteResult.cachedOps, getKernelInfo()->numOutputs, opBuilder);
  auto ubTilingDataHandle = ValueHandleFoldResults{
      tilingDataHandles[kUBTileSizePos1], tilingDataHandles[kUBTileSizePos2]};
  ForTilingResult tileUsingForResult =
      tileUsingFor(splitCachedOps, ubTilingDataHandle, opBuilder);

  // Step 4: Apply canonicalize patterns.
  applyPatterns(
      getFuncHandle(opBuilder),
      /*patterns=*/
      SmallVector<TransformPatternKind>{
          TransformPatternKind::CSE, TransformPatternKind::CANONICALIZATION,
          TransformPatternKind::MERGE_CONSECUTIVE_INSERT_EXTRACT_SLICE},
      opBuilder);

  // Step 5: Fuse producers into `scf.for` op.
  // We wish to fuse producers ops by reverse topological ordering.
  MatchOptions matchOptions;
  matchOptions.needsReverse = true;
  ValueHandle *producerOps =
      getOpsWithIdentifier(kIntermediateProducerTagName,
                           IdentifierType::kAttribute, opBuilder, matchOptions);
  ValueHandles targetsToFuseInto = {producerOps, cacheReadResult.cachedOps};
  ValueHandles fusedLoopList = {tileUsingForResult.loops.front().back()};
  fuseIntoContaining(targetsToFuseInto, fusedLoopList, opBuilder,
                     /*duplicateProducers=*/true,
                     /*applyCanonicalizeAfterEachFusion=*/true);

  // Step 6: Clean up IR.
  applyCSE(opBuilder);
  applyCanonicalization(opBuilder);

  // Step 7: Set buffer size.
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