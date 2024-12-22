//===----------------- CacheIO.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Transforms/Passes.h"
#include "mtir/Dialect/MtFusion/Utils/Utils.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir {
#define GEN_PASS_DEF_CACHEIO
#include "mtir/Dialect/MtFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
constexpr static char kFuncArgIdxFormat[] = "__arg{0}__";
constexpr static llvm::StringLiteral kCacheReadTagName = "__cache_read__";
constexpr static llvm::StringLiteral kCacheWriteTagName = "__cache_write__";

/// Set a unit attribute named \c attrName to \c op.
void setNamedUnitAttr(Operation *op, StringRef attrName) {
  assert(op != nullptr);
  op->setAttr(attrName, UnitAttr::get(op->getContext()));
}
} // namespace

void mtfusion::cacheFuncIO(func::FuncOp funcOp, bool annotate = true) {
  OpBuilder builder(funcOp.getContext());
  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    if (!arg.getType().isa<TensorType>()) {
      continue;
    }
    bool funcArgIsReshaped = false;
    bool funcResultIsReshaped = false;
    if (auto resultIdx = mtfusion::getFuncArgTiedResultReturnIdx(
            arg, funcArgIsReshaped, funcResultIsReshaped)) {
      continue;
    }

    auto maybeTracedArg = traceReshapeOrSliceSingleConsumer(arg);
    auto tracedArg = failed(maybeTracedArg) ? arg : maybeTracedArg.value();
    if (tracedArg.isa<BlockArgument>()) {
      builder.setInsertionPoint(
          &(tracedArg.cast<BlockArgument>().getParentBlock()->front()));
    } else if (tracedArg.isa<OpResult>()) {
      builder.setInsertionPoint(tracedArg.cast<OpResult>().getDefiningOp());
    }
    Operation *cachedOp =
        mtfusion::createCacheRead(builder, tracedArg, tracedArg.getLoc());
    if (annotate) {
      setNamedUnitAttr(cachedOp, llvm::formatv(kFuncArgIdxFormat, idx).str());
      setNamedUnitAttr(cachedOp, kCacheReadTagName);
    }
  }

  func::ReturnOp returnOp = nullptr;
  funcOp->walk([&returnOp](func::ReturnOp op) { returnOp = op; });
  for (auto res : returnOp.getOperands()) {
    auto maybeTracedRes = traceReshapeOrSliceSingleProducer(res);
    auto tracedRes = failed(maybeTracedRes) ? res : maybeTracedRes.value();
    builder.setInsertionPointAfter(tracedRes.getDefiningOp());
    auto cachedOp = mtfusion::createCacheWrite(
        builder, tracedRes.cast<OpResult>(), true, true);
    if (succeeded(cachedOp) && annotate) {
      setNamedUnitAttr(cachedOp.value(), kCacheWriteTagName);
    }
  }
}

namespace mlir {
struct CacheIOPass : public impl::CacheIOBase<CacheIOPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    mtfusion::cacheFuncIO(funcOp);
  }
};
} // namespace mlir

std::unique_ptr<Pass> mlir::mtfusion::createCacheIO() {
  return std::make_unique<CacheIOPass>();
}