//===- OpFusion.cpp -- Outline fusable ops into kernels -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements op fusion algorithm and outline into functions.
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Utils/Utils.h"
#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/Transforms/OpFusion/FusableBlock.h"
#include "mtir/Dialect/MtFusion/Transforms/OpFusion/FusableBlockAnalyzer.h"
#include "mtir/Dialect/MtFusion/Transforms/OpFusion/FusableBlockOutliner.h"
#include "mtir/Dialect/MtFusion/Transforms/OpFusion/FusableHelper.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"
#include "mtir/Dialect/Utils/Util.h"
#include <iostream>

#include "llvm/Support/Debug.h"

#include <optional>

#define DEBUG_TYPE "mtfusion-op-fusion"

namespace mlir {
#define GEN_PASS_DEF_MTFUSIONOPFUSION
#include "mtir/Dialect/MtFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::mtfusion;

namespace mlir {
namespace mtfusion {

using namespace opfusion;

namespace {

inline std::optional<MtFusionOpFusionOptions>
getOptionFromLabel(func::FuncOp func, const MtFusionOpFusionOptions &options) {
  MtFusionOpFusionOptions newOptions = options;
  auto fusionKindAttr =
      func->getAttrOfType<FusionKindAttr>(FusionKindAttr::name);
  if (!fusionKindAttr)
    return std::nullopt;
  auto fusionKind = fusionKindAttr.getFusionKind();
  newOptions.fusionMode = fusionKind;
  return newOptions;
}
} // namespace

LogicalResult outlineFusedFuncs(func::FuncOp entryFunc,
                                const MtFusionOpFusionOptions &options,
                                SmallVector<func::FuncOp> &outlinedFuncs) {
  if (options.fusionMode == FusionKind::Unknown)
    return failure();

  FusableHelper fusableHelper(options.fusionMode, options.moveOutToParam,
                              options.maxHorizontalFusionSize);
  
  // RVO will optimize this
  FusableBlocks fusableBlocks = getFusableBlocks(entryFunc, fusableHelper);
  if (fusableBlocks.empty())
    return success();

  // for(auto block : fusableBlocks)
  //   block.dump();
  
  FusableBlockOutliner outliner(fusableBlocks, options.outputMode,
                                options.alwaysInline, true);

  if (!outliner.outline())
    return failure();

  outlinedFuncs.append(outliner.getOutlinedFuncs());
  return success();
}

} // namespace mtfusion

//===---------------------------------------------------------------------===//
// Pass
//===---------------------------------------------------------------------===//

struct MtFusionOpFusionPass
    : public impl::MtFusionOpFusionBase<MtFusionOpFusionPass> {
  explicit MtFusionOpFusionPass(const MtFusionOpFusionOptions &options)
      : MtFusionOpFusionBase(options) {}
  void initOptions() {
    options_.outputMode = this->outputMode;
    options_.fusionMode = this->fusionMode;
    options_.alwaysInline = this->alwaysInline;
    options_.moveOutToParam = this->moveOutToParam;
    options_.maxHorizontalFusionSize = this->maxHorizontalFusionSize;
  }
  void runOnOperation() override {
    initOptions();
    // This is a module pass to avoid function making and calling issues
    getOperation()->walk([&](func::FuncOp func) -> void {
      if (!mtfusion::isHost(func)) {
        func->dump();
        std::cout<<"is not a host function."<<std::endl;
        return;
      }
      [[maybe_unused]] SmallVector<func::FuncOp> outlinedFuncs;
      auto newOption = mtfusion::getOptionFromLabel(func, options_);
      // Return by reference of this outlinedFuncs
      if (newOption &&
          failed(outlineFusedFuncs(func, newOption.value(), outlinedFuncs)))
        return signalPassFailure();
    });
  }

private:
  MtFusionOpFusionOptions options_;
};
} // namespace mlir

std::unique_ptr<Pass> mlir::mtfusion::createMtFusionOpFusionPass(
    const MtFusionOpFusionOptions &options) {
  return std::make_unique<MtFusionOpFusionPass>(options);
}
