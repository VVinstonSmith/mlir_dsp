//===- SingleOpOutline.cpp - Outline Single Operations --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Utils/Utils.h"
#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/Transforms/OpFusion/FusableBlockOutliner.h"
#include "mtir/Dialect/MtFusion/Transforms/OpFusion/FusableHelper.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"
#include "mtir/Dialect/Utils/Util.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mtfusion-single-op-fusion"

namespace mlir {
#define GEN_PASS_DEF_SINGLEOPOUTLINE
#include "mtir/Dialect/MtFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace mlir {
using namespace mtfusion;
using namespace mtfusion::opfusion;
struct SingleOpOutlinePass
    : public impl::SingleOpOutlineBase<SingleOpOutlinePass> {
public:
  void runOnOperation() override {

    func::FuncOp func = getOperation();

    LLVM_DEBUG(llvm::dbgs() << "Running single op\n";);
    if (!mtfusion::isHost(func)) {
      return;
    }

    func.walk([&](Operation *op) {
      LLVM_DEBUG(llvm::dbgs() << "Checking op " << *op << "\n";);
      if (opfusion::FusableHelper::isSingleOutlinable(op)) {
        LLVM_DEBUG(llvm::dbgs() << "Outlinable\n";);
        outlineSingleOp(func, op);
      }
    });
  }

private:
  size_t funcCnt_{0};

  void outlineSingleOp(func::FuncOp &func, Operation *op) {
    OpBuilder builder(func.getContext());
    builder.setInsertionPoint(func);
    // RVO will optimize this
    FusableHelper fusableHelper(
        opfusion::FusableHelper::getSingleFusionKind(op), true, 0);
    SmallVector<Operation *> ops = {op};
    FusableBlocks fusableBlocks = {FusableBlock(ops, &fusableHelper, true)};
    FusableBlockOutliner outliner(fusableBlocks, OutputMode::Multiple, false);
    if (!outliner.outline("_single_outlined_" + std::to_string(funcCnt_++)))
      return signalPassFailure();
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::mtfusion::createSingleOpOutlinePass() {
  return std::make_unique<SingleOpOutlinePass>();
}
