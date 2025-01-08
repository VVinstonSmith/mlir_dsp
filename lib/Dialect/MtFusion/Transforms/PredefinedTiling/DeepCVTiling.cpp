//===- DeepCVTiling.cpp -- Tiling for deep fused CV kernels ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements tiling policy for deep fused CV kernels.
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Transforms/PredefinedTiling/DeepCVTiling.h"
#include "mtir/Dialect/MtFusion/Transforms/PredefinedTiling/TilingBase.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/AutoScheduleAttrDefs.h"
#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/ValueHandle.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"

#include <map>
#include <utility>
#include <iostream>

#define DEBUG_TYPE "mtfusion-predefined-tiling"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] [Deep CV] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::mtfusion;

const std::map<mtfusion::Cache, std::string> CacheToStr = {
    {Cache::DDR, "DDR"}, {Cache::GSM, "GSM"},
    {Cache::AM, "AM"}, {Cache::SM, "SM"}};

const std::map<mtfusion::Matrix, int64_t> MatToOprId = {
    {Matrix::MatA, 0}, {Matrix::MatB, 1}, {Matrix::MatC, 2}};

static std::string stringfyDataFlow(mtfusion::Cache src, mtfusion::Cache dst) {
  return CacheToStr.at(src) + " : " + CacheToStr.at(dst);
}

//===----------------------------------------------------------------------===//
// DeepCVTiler
//===----------------------------------------------------------------------===//

void DeepCVTiler::applyCanonicalization(OpBuilder &opBuilder) {
  applyPatterns(
      getFuncHandle(opBuilder),
      /*patterns=*/
      SmallVector<TransformPatternKind>{
          TransformPatternKind::CSE, TransformPatternKind::CANONICALIZATION},
          // TransformPatternKind::MERGE_CONSECUTIVE_INSERT_EXTRACT_SLICE},
      opBuilder,
      /*disablePatterns=*/
      SmallVector<CanonicalizationPatternKind>{
          CanonicalizationPatternKind::kSimplifyTrivialLoops});
}

ValueHandles DeepCVTiler::matchMatmulOps(OpBuilder &opBuilder) {
  /// Match linalg.matmul ops.
  MatmulResult matmulResult = {getOpsWithIdentifier(
      "linalg.matmul", IdentifierType::kOperation, opBuilder)};
  // Tile linalg.matmul writes using `scf.forall` op.
  int64_t nMatmulOps = 
      llvm::count_if(getToBeTiledKernel().getBody().getOps(), [&](Operation& op) {
        return isa<linalg::MatmulOp>(op);
      });
  ValueHandles splitMatmulOps = splitHandle(
      matmulResult.matmulOps, nMatmulOps, opBuilder);
  return splitMatmulOps;
}

LogicalResult DeepCVTiler::createTilingImpl(OpBuilder &opBuilder) {
  func::FuncOp toBeTiledKenrel = getToBeTiledKernel();
  
  /// Generate tiling argument positions.
  SmallVector<size_t> tilingArgsPos;
  for (auto [argIdx, tilingArg] : llvm::enumerate(toBeTiledKenrel.getArguments())) {
    if (toBeTiledKenrel.getArgAttr(argIdx, mtfusion::TilingDataAttr::name)) {
      tilingArgsPos.push_back(argIdx);
    }
  }

  /// Match linalg.matmul ops.
  ValueHandles splitMatmulOps = matchMatmulOps(opBuilder);

  /// Match ops with tiling_data attr.
  TilingDataResult tilingParams = {getOpsWithIdentifier(
      "mtfusion.tiling_data", IdentifierType::kAttribute, opBuilder)};
  ValueHandles splitTilingParams = splitHandle(
      tilingParams.tilingData , tilingArgsPos.size(), opBuilder);

  std::map<mtfusion::Matrix, mtfusion::Cache> matDataLoc = {
    {Matrix::MatA, Cache::DDR}, {Matrix::MatB, Cache::DDR}, 
    {Matrix::MatC, Cache::DDR}};

  for(size_t argIdx = 0; argIdx < tilingArgsPos.size(); argIdx++) {
    /// Get tiling sizes from castOp and tiling axis from TilingAxisAttr.
    ValueHandleFoldResults tileSizes(3,
        ValueHandleFoldResult(0, getContext()));
    if(auto axisAttr = toBeTiledKenrel.getArgAttr(
        tilingArgsPos[argIdx], mtfusion::TilingAxisAttr::name)) {
      switch(cast<mtfusion::TilingAxisAttr>(axisAttr).getAxis()) {
        case mtfusion::Axis::M :
          tileSizes[0] = ValueHandleFoldResult{splitTilingParams[argIdx]};
          // std::cout << "mtfusion::Axis::M" << std::endl;
          break;
        case mtfusion::Axis::N :
          tileSizes[1] = ValueHandleFoldResult{splitTilingParams[argIdx]};
          break;
        case mtfusion::Axis::K :
          tileSizes[2] = ValueHandleFoldResult{splitTilingParams[argIdx]};
          break;
        default:
          return toBeTiledKenrel->emitError("Unknow tiling axis.");
      }
    }

    /// Create transform::tileUsingForOp
    ForTilingResult tileUsingForResult =
        tileUsingFor(splitMatmulOps, tileSizes, opBuilder);
    
    /// Add data copy according to copy_mat and copy_dst.
    if(auto copyMatAttr = toBeTiledKenrel.getArgAttr(
        tilingArgsPos[argIdx], mtfusion::CopyMatAttr::name)) {
      auto copyMats = copyMatAttr.cast<CopyMatAttr>().getMatrices();
      auto copyDsts = toBeTiledKenrel.getArgAttr(tilingArgsPos[argIdx], 
          mtfusion::CopyDstAttr::name).cast<CopyDstAttr>().getCaches();
      SmallVector<int64_t> readOprIndices;
      SmallVector<std::string> readOpAttrs;
      SmallVector<std::pair<std::string, std::string>> readAndWriteOpAttrs;
      for(size_t cpyId = 0; cpyId < copyMats.size(); cpyId++) {
        switch(copyMats[cpyId]) {
          case mtfusion::Matrix::MatA :
          case mtfusion::Matrix::MatB :
            readOprIndices.push_back(MatToOprId.at(copyMats[cpyId]));
            readOpAttrs.push_back(
                stringfyDataFlow(matDataLoc[copyMats[cpyId]], copyDsts[cpyId]));
            break;
          case mtfusion::Matrix::MatC :
            readAndWriteOpAttrs.push_back({
              stringfyDataFlow(matDataLoc[copyMats[cpyId]], copyDsts[cpyId]),
              stringfyDataFlow(copyDsts[cpyId], matDataLoc[copyMats[cpyId]])
            });
            break;
          default:
            return toBeTiledKenrel->emitError("Error matrix attr.");
        };
        matDataLoc[copyMats[cpyId]] = copyDsts[cpyId];
      }
      for(size_t i = 0; i < readOprIndices.size(); i++)
        cacheRead(splitMatmulOps, readOprIndices[i], readOpAttrs[i], opBuilder);
      for(size_t i = 0; i < readAndWriteOpAttrs.size(); i++) {
        auto [readOpAttr, writeOpAttr] = readAndWriteOpAttrs[i];
        cacheReadAndWrite(splitMatmulOps, 0 /*resIndex*/, readOpAttr, writeOpAttr, opBuilder);
      }
    }
  }
  applyCanonicalization(opBuilder);

  return success();
}
