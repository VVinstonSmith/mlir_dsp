//===- TensorResToOutParams.cpp - Move tensor results to function params --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Transforms/Passes.h"
#include "mtir/Dialect/MtFusion/Utils/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mtfusion-tensor-results-to-out-params"

namespace mlir {
#define GEN_PASS_DEF_TENSORRESTOOUTPARAMS
#include "mtir/Dialect/MtFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::mtfusion;

namespace {
struct TensorResToOutParamsPass
    : public impl::TensorResToOutParamsBase<TensorResToOutParamsPass> {
public:
  explicit TensorResToOutParamsPass() : TensorResToOutParamsBase() {}
  explicit TensorResToOutParamsPass(ArrayRef<std::string> includeSymbols);
  LogicalResult initialize(MLIRContext *context) override;
  void runOnOperation() final;

private:
  void updateFuncSignature(func::FuncOp func);
  void tryModify(func::FuncOp func);
  void fixCallSites();

private:
  DenseSet<func::FuncOp> modifiedFuncs;
  DenseSet<StringAttr> includeSymbolSet;
};

LogicalResult TensorResToOutParamsPass::initialize(MLIRContext *context) {
  for (const std::string &symbol : includeSymbols)
    includeSymbolSet.insert(StringAttr::get(context, symbol));
  return success();
}

TensorResToOutParamsPass::TensorResToOutParamsPass(
    llvm::ArrayRef<std::string> includeSymbols) {
  this->includeSymbols = includeSymbols;
}

void TensorResToOutParamsPass::updateFuncSignature(func::FuncOp func) {
  LLVM_DEBUG(llvm::dbgs() << "Update function type @" << func.getSymName()
                          << "\n");
  SetVector<Value> values;
  for (Block &block : func.getBody()) {
    for (Value v : block.getArguments()) {
      if (!v.getDefiningOp())
        values.insert(v);
    }
  }

  FunctionType newFuncTy = func.getFunctionType().clone(
      TypeRange(ValueRange(values.getArrayRef())), func.getResultTypes());
  func.setType(newFuncTy);
  LLVM_DEBUG(func.getFunctionType().dump());
}

/// get dps result for each return value and corresponding op passed through by
/// tracing upwards from ReturnOp
static SmallVector<std::pair<Value, SmallVector<Operation *>>>
getReturnDpsResultAndTraces(func::ReturnOp ret) {
  SmallVector<std::pair<Value, SmallVector<Operation *>>> reshapeTraces;
  for (Value v : ret.getOperands()) {
    if (!isa<mlir::TensorType>(v.getType())) {
      ret->emitWarning("return operand is not tensor type");
      continue;
    }
    Operation *op = v.getDefiningOp();
    if (!isReshapeOp(op)) {
      reshapeTraces.push_back({v, {}});
      continue;
    }
    SmallVector<Operation *> trace =
        mtfusion::getReshapeOrSliceOpProduceTrace(v);
    assert(!trace.empty() && "reshape produce trace must not be empty");
    Value source = mtfusion::getReshapeSource(trace.back());
    reshapeTraces.push_back({source, trace});
  }
  return reshapeTraces;
}

/// create a new block argument and return the value traced from the new
/// argument to replace the origin `dpsInitV`. The new replacement value is
/// computed by reverse the expandOp/collapseOp in provided `trace`.
static Value getInitValueReplacement(Value dpsInitV,
                                     const SmallVector<Operation *> &trace,
                                     OpBuilder &builder) {
  Block *block = dpsInitV.getDefiningOp()->getBlock();
  if (trace.empty()) {
    BlockArgument newArg =
        block->addArgument(dpsInitV.getType(), dpsInitV.getLoc());
    return newArg;
  }

  // reshape op in trace requires extra collapse/expand when replacing uses with
  // new block args
  Operation *retDefOp = trace.front();
  Type newArgType = retDefOp->getResult(0).getType();
  BlockArgument newArg = block->addArgument(newArgType, dpsInitV.getLoc());
  Location loc = dpsInitV.getDefiningOp()->getLoc();

  Value src = newArg;
  for (Operation *op : trace) {
    if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(op)) {
      src = builder.create<tensor::CollapseShapeOp>(
          loc, /*resultType=*/expandOp.getSrcType(), /*src=*/src,
          /*reassociation=*/expandOp.getReassociation());
    } else if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(op)) {
      src = builder.create<tensor::ExpandShapeOp>(
          loc, /*resultType=*/collapseOp.getSrcType(), /*src=*/src,
          /*reassociation=*/collapseOp.getReassociationIndices());
    } else {
      llvm_unreachable(
          "only support reshape Op including tensor::ExpandShapeOp "
          "and tensor::CollapseShapeOp");
    }
  }
  return src;
}

void TensorResToOutParamsPass::tryModify(func::FuncOp func) {
  bool modified = false;
  for (func::ReturnOp ret : func.getBody().getOps<func::ReturnOp>()) {
    LLVM_DEBUG(ret->dump());
    for (const auto &[dpsValue, trace] : getReturnDpsResultAndTraces(ret)) {
      Operation *op = dpsValue.getDefiningOp();
      if (!op || !isa<DestinationStyleOpInterface>(op))
        continue;
      OpResult dpsResult = cast<OpResult>(dpsValue);

      // update specified init of target retDefOp, use `initIdx` to update the
      // right init value when handling multiple return values
      auto dpsOp = cast<DestinationStyleOpInterface>(op);
      unsigned int initIdx = dpsResult.getResultNumber();
      OpOperand *initV = dpsOp.getDpsInitOperand(initIdx);
      auto initSource = traceReshapeOrSliceSingleProducerOrSelf(initV->get());
      if (isa<BlockArgument>(initSource))
        continue;

      OpBuilder builder(dpsOp);
      Value replacement = getInitValueReplacement(initV->get(), trace, builder);
      dpsOp.setDpsInitOperand(initIdx, replacement);
      modified = true;
    }
    updateFuncSignature(func);
  }
  if (modified)
    modifiedFuncs.insert(func);
}

void TensorResToOutParamsPass::fixCallSites() {
  for (func::FuncOp func : modifiedFuncs) {
    LLVM_DEBUG(llvm::dbgs()
               << "Fix callsites of @" << func.getSymName() << "\n");

    DenseMap<func::FuncOp, tiling::CallerInfo> workList;
    tiling::getCallerInfo(func, getOperation(), workList);

    DenseSet<Operation *> processed;
    while (!workList.empty()) {
      auto &[caller, callerInfo] = *(workList.begin());
      if (processed.contains(caller)) {
        return;
      }

      DenseMap<Block *, size_t> updatedBlocks;
      for (auto oldCall : callerInfo.callSites) {
        OpBuilder opBuilder(oldCall);
        Block *block = opBuilder.getInsertionBlock();
        if (!updatedBlocks.contains(block)) {
          const size_t orgNumBlockArgs = block->getNumArguments();
          ArrayRef<Type> calleeArgTypes = func.getArgumentTypes();
          for (auto it = calleeArgTypes.begin() + oldCall->getNumOperands(),
                    itEnd = calleeArgTypes.end();
               it != itEnd; ++it)
            block->addArgument(*it, oldCall.getLoc());
          updatedBlocks.insert(std::make_pair(block, orgNumBlockArgs));
          updateFuncSignature(oldCall->getParentOfType<func::FuncOp>());
        }

        LLVM_DEBUG(llvm::dbgs() << "Fix caller\n");
        SmallVector<Value> newCallArgs(oldCall->getOperands());
        newCallArgs.append(block->args_begin() + updatedBlocks.at(block),
                           block->args_end());
        func::CallOp newCall =
            opBuilder.create<func::CallOp>(oldCall.getLoc(), func, newCallArgs);
        oldCall.replaceAllUsesWith(newCall);
        oldCall->erase();
        LLVM_DEBUG(newCall->dump());
      }

      tiling::getCallerInfo(caller, getOperation(), workList);
      processed.insert(caller);
      workList.erase(caller);
    }
  }
}

void TensorResToOutParamsPass::runOnOperation() {
  bool applyToAll = includeSymbolSet.empty();
  getOperation()->walk([&](func::FuncOp func) {
    if (applyToAll || includeSymbolSet.contains(func.getSymNameAttr()))
      tryModify(func);
  });
  fixCallSites();
}

} // anonymous namespace

std::unique_ptr<Pass> mtfusion::createTensorResToOutParamsPass() {
  return std::make_unique<TensorResToOutParamsPass>();
}

std::unique_ptr<Pass>
mtfusion::createTensorResToOutParamsPass(ArrayRef<std::string> includeSymbols) {
  return std::make_unique<TensorResToOutParamsPass>(includeSymbols);
}