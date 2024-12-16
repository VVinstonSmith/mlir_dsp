//===- InferFuncFuseKind.cpp -- label host function to a fusion kind ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements fusion kind inferring and labeling
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/Transforms/OpFusion/FusableBlock.h"
#include "mtir/Dialect/MtFusion/Transforms/OpFusion/FusableBlockAnalyzer.h"
#include "mtir/Dialect/MtFusion/Transforms/OpFusion/FusableHelper.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"
#include "mtir/Dialect/MtFusion/Utils/Utils.h"
#include "mtir/Dialect/Utils/Util.h"

#include "llvm/Support/Debug.h"

#include <numeric>
#include <queue>

#define DEBUG_TYPE "mtfusion-infer-func"

namespace mlir {
#define GEN_PASS_DEF_INFERFUNCFUSIONKIND
#include "mtir/Dialect/MtFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace mlir {
using namespace mtfusion;
using namespace mtfusion::opfusion;
struct InferFuncFusionKindPass
    : public impl::InferFuncFusionKindBase<InferFuncFusionKindPass> {
public:
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (tryGetFusionKind(func)) {
      LLVM_DEBUG(llvm::dbgs() << "Fusion Kind attribute found\n";);
      LLVM_DEBUG(llvm::dbgs() << "Skipping " << func << "\n";);
      return;
    }
    labelFunction(func);
  }

private:
  void labelFunction(func::FuncOp func) {

    int opCount = 0;
    Operation *firstOp;
    func.walk([&](Operation *op) {
      LLVM_DEBUG(llvm::dbgs() << "Checking op " << *op << "\n";);
      if (FusableHelper::isSingleOutlinable(op)) {
        firstOp = op;
        LLVM_DEBUG(llvm::dbgs() << "Outlinable\n";);
        opCount++;
      }
    });

    if (opCount == 0) {
      LLVM_DEBUG(llvm::dbgs() << "No outlinable op found, skipping\n";);
      trySetFusionKind(func, FusionKind::AnyPB);
      return;
    }

    // Single outlinable corner case
    if (opCount == 1) {
      LLVM_DEBUG(llvm::dbgs() << "Single outlinable function found\n";);
      trySetFusionKind(func, FusableHelper::getSingleFusionKind(firstOp));
      return;
    }

    for (uint32_t i = 1; i < getMaxEnumValForFusionKind(); i++) {
      // Loop through all this and apply fusion to it
      auto fusionKind = symbolizeFusionKind(i).value();
      FusableHelper fusableHelper(fusionKind, false, -1);
      auto fusableBlocks = getFusableBlocks(func, fusableHelper);
      if (fusableBlocks.size() != 1)
        continue;

      int fusableCount = 0;
      for (auto op : fusableBlocks[0].getOps()) {
        if (FusableHelper::isSingleOutlinable(op)) {
          fusableCount++;
        }
      }

      if (fusableCount == opCount) {
        trySetFusionKind(func, fusionKind);
        LLVM_DEBUG(llvm::dbgs() << "Found fusable label\n";);
        return;
      }
      // Check if all operations here is outlined.
    }

    trySetFusionKind(func, FusionKind::Unknown);
    LLVM_DEBUG(llvm::dbgs() << "This function cannot be labeled\n";);
    return;
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::mtfusion::createInferFuncFusionKind() {
  return std::make_unique<InferFuncFusionKindPass>();
}