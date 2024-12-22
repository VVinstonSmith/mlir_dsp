//===- ConstantizeTilingData.cpp -- Optimize cst tiling data -----*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic to optimize tiling data that are compile-time
// constants.
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Transforms/Passes.h"
#include "mtir/Dialect/MtFusion/Utils/Utils.h"
#include "mtir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"

#define DEBUG_TYPE "mtfusion-constantize-tiling-data"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

namespace mlir {
namespace mtfusion {

#define GEN_PASS_DEF_CONSTANTIZETILINGDATA
#include "mtir/Dialect/MtFusion/Transforms/Passes.h.inc"

namespace {

class ConstantizeTilingDataPass
    : public impl::ConstantizeTilingDataBase<ConstantizeTilingDataPass> {

  void runOnOperation() final;

private:
  LogicalResult propagateCalcFuncRes(func::FuncOp &calcFunc,
                                     SmallVector<func::FuncOp> &deviceFuncs);
  LogicalResult removeDeadResults(func::FuncOp &calcFunc,
                                  SmallVector<func::FuncOp> &deviceFuncs);
  LogicalResult removeCalcFuncDeadValues(func::FuncOp &calcFunc,
                                         const BitVector &deadIndices);
  LogicalResult removeDeviceFuncDeadValues(func::FuncOp &deviceFunc,
                                           const BitVector &calcEraseIndices);
  LogicalResult
  removeDeviceCallerDeadArguments(func::FuncOp &deviceFunc,
                                  const BitVector &calcEraseIndices);
};

void ConstantizeTilingDataPass::runOnOperation() {
  ModuleOp mod = getOperation();
  std::map<std::string, SmallVector<func::FuncOp>> tilingFuncNameToDeviceFunc;
  mod.walk([&](func::FuncOp func) {
    if (Attribute attr = func->getAttr(mtfusion::TilingFuncAttr::name)) {
      StringRef calcFuncName = cast<StringAttr>(attr).getValue();
      tilingFuncNameToDeviceFunc[calcFuncName.str()].push_back(func);
    }
  });

  for (auto &[calcFuncName, devFuncs] : tilingFuncNameToDeviceFunc) {
    if (auto calcFunc = mod.lookupSymbol<func::FuncOp>(calcFuncName)) {
      if (failed(tiling::verifyCalcTiling(calcFunc, devFuncs)))
        return signalPassFailure();
      if (failed(propagateCalcFuncRes(calcFunc, devFuncs)))
        return signalPassFailure();
      if (failed(removeDeadResults(calcFunc, devFuncs)))
        return signalPassFailure();
    } else {
      mod.emitError("Corresponding calcFunc not found: ") << calcFuncName;
      return signalPassFailure();
    }
  }
}

static BitVector generateEraseIndicesForFuncBasedOnDeadTilingIndices(
    func::FuncOp f, const BitVector &calcEraseIndices) {
  int tilingIndex = 0;
  BitVector erasedIndices(f.getNumArguments(), false);
  int EraseIndices = calcEraseIndices.size();
  for (auto [idx, arg] : llvm::enumerate(f.getArguments())) {
    if (f.getArgAttr(idx, mtfusion::TilingDataAttr::name) && EraseIndices != 0) {
      if (calcEraseIndices[tilingIndex])
        erasedIndices.set(idx);
      tilingIndex++;
      EraseIndices--;
    }
  }
  return erasedIndices;
}

static SmallVector<Value> newArgsBuilder(const BitVector &erasedIndices,
                                         func::CallOp callSite) {
  SmallVector<Value> newArgs;
  for (const auto [idx, operand] : llvm::enumerate(callSite.getOperands())) {
    if (!erasedIndices[idx]) {
      newArgs.push_back(operand);
    }
  }
  return newArgs;
}

LogicalResult ConstantizeTilingDataPass::removeDeviceFuncDeadValues(
    func::FuncOp &deviceFunc, const BitVector &calcEraseIndices) {
  LDBG("Removing dead values in device func\n" << *deviceFunc);

  // Get callers of the device func.
  DenseMap<func::FuncOp, tiling::CallerInfo> workList;
  tiling::getCallerInfo(deviceFunc, getOperation(), workList);

  // Bail out on trivial case where there is no caller
  if (workList.empty()) {
    BitVector erasedIndices =
        generateEraseIndicesForFuncBasedOnDeadTilingIndices(deviceFunc,
                                                            calcEraseIndices);
    deviceFunc.eraseArguments(erasedIndices);
    LDBG("Removed dead values from device func:\n" << *deviceFunc);
    return success();
  }

  // Repeatedly modify the caller and call site, until there is no caller.
  DenseMap<Operation *, Operation *> irMap;
  DenseSet<Operation *> processed;
  while (!workList.empty()) {
    auto &[caller, callerInfo] = *(workList.begin());
    if (processed.contains(caller)) {
      LDBG("Cyclic call detected");
      return failure();
    }

    LDBG("Fixing call site in: \n" << *caller);
    BitVector erasedIndices =
        generateEraseIndicesForFuncBasedOnDeadTilingIndices(callerInfo.callee,
                                                            calcEraseIndices);
    tiling::CallSiteBuilderInfo builderInfo{
        /*argBuilderFn=*/
        std::bind(newArgsBuilder, erasedIndices,
                  std::placeholders::_1), /*siteBuilderFn=*/
        tiling::callSiteBuilderFnForTilingModification};

    OpBuilder opBuilder(caller);
    if (failed(
            tiling::doFixCallSite(callerInfo, builderInfo, irMap, opBuilder)))
      return failure();

    for (auto &[oldOp, newOp] : irMap) {
      oldOp->replaceAllUsesWith(newOp);
      oldOp->erase();
    }
    irMap.clear();

    tiling::getCallerInfo(caller, getOperation(), workList);
    processed.insert(caller);
    workList.erase(caller);
  }

  BitVector erasedIndices = generateEraseIndicesForFuncBasedOnDeadTilingIndices(
      deviceFunc, calcEraseIndices);
  deviceFunc.eraseArguments(erasedIndices);
  LDBG("Removed dead values from device func:\n" << *deviceFunc);
  return success();
}

LogicalResult ConstantizeTilingDataPass::removeCalcFuncDeadValues(
    func::FuncOp &calcFunc, const BitVector &deadIndices) {
  auto returnOp = cast<func::ReturnOp>(calcFunc.getBody().front().back());
  SmallVector<Value, 4> newReturnValues;
  SmallVector<Type, 4> newReturnTypes;

  for (auto [i, value] : llvm::enumerate(returnOp.getOperands())) {
    if (!deadIndices[i]) {
      newReturnValues.push_back(value);
      newReturnTypes.push_back(value.getType());
    }
  }

  OpBuilder builder(returnOp);
  builder.create<func::ReturnOp>(returnOp.getLoc(), newReturnValues);
  returnOp.erase();

  auto newFuncType =
      FunctionType::get(calcFunc.getContext(),
                        calcFunc.getFunctionType().getInputs(), newReturnTypes);
  calcFunc.setType(newFuncType);

  getOperation().walk([&](func::CallOp callOp) {
    if (callOp.getCallee() == calcFunc.getName()) {
      SmallVector<Value, 4> newResults;
      for (auto [i, result] : llvm::enumerate(callOp.getResults())) {
        if (!deadIndices[i])
          newResults.push_back(result);
      }
      OpBuilder builder(callOp);
      auto newCall = builder.create<func::CallOp>(callOp.getLoc(), calcFunc,
                                                  callOp.getOperands());
      for (auto [oldResult, newResult] :
           llvm::zip(newResults, newCall.getResults()))
        oldResult.replaceAllUsesWith(newResult);
      callOp.erase();
    }
  });

  return success();
}

LogicalResult ConstantizeTilingDataPass::removeDeviceCallerDeadArguments(
    func::FuncOp &deviceFunc, const BitVector &calcEraseIndices) {
  LDBG("Begin removing device caller dead arguments for:\n" << *deviceFunc);

  // Get callers of the device func.
  DenseMap<func::FuncOp, tiling::CallerInfo> workList;
  tiling::getCallerInfo(deviceFunc, getOperation(), workList);

  DenseSet<Operation *> processed;
  while (!workList.empty()) {
    auto &[caller, callerInfo] = *(workList.begin());
    if (processed.contains(caller)) {
      LDBG("Cyclic call detected");
      return failure();
    }

    LDBG("Processing func:\n" << *caller);
    auto erasedIndices = generateEraseIndicesForFuncBasedOnDeadTilingIndices(
        caller, calcEraseIndices);
    caller.eraseArguments(erasedIndices);
    tiling::getCallerInfo(caller, getOperation(), workList);

    processed.insert(caller);
    workList.erase(caller);
  }
  return success();
}

LogicalResult ConstantizeTilingDataPass::removeDeadResults(
    func::FuncOp &calcFunc, SmallVector<func::FuncOp> &deviceFuncs) {
  BitVector deadIndices(calcFunc.getNumResults(), true);
  for (auto deviceFunc : deviceFuncs) {
    int tilingIndex = 0;
    for (auto [idx, arg] : llvm::enumerate(deviceFunc.getArguments())) {
      if (deviceFunc.getArgAttr(idx, mtfusion::TilingDataAttr::name)) {
        if (!arg.use_empty())
          deadIndices.reset(tilingIndex);
        tilingIndex++;
      }
    }
  }

  for (auto deviceFunc : deviceFuncs) {
    if (failed(removeDeviceFuncDeadValues(deviceFunc, deadIndices)))
      return failure();
    if (failed(removeDeviceCallerDeadArguments(deviceFunc, deadIndices)))
      return failure();
  }

  if (failed(removeCalcFuncDeadValues(calcFunc, deadIndices)))
    return failure();

  return success();
}

LogicalResult ConstantizeTilingDataPass::propagateCalcFuncRes(
    func::FuncOp &calcFunc, SmallVector<func::FuncOp> &deviceFuncs) {
  auto returnOp = cast<func::ReturnOp>(calcFunc.getBody().front().back());
  SmallVector<std::pair<int, Operation *>> constantCalcFuncRets;
  // Extract the constant return value
  for (auto [idx, returnVal] : llvm::enumerate(returnOp.getOperands())) {
    OpFoldResult foldRes = getAsOpFoldResult(returnVal);
    // Simply Case: return value's defining op is constant op
    if (auto constOp = returnVal.getDefiningOp<arith::ConstantOp>()) {
      LDBG("Tiling function's return is arith const op");
      constantCalcFuncRets.emplace_back(idx, constOp);
      continue;
    }
  }
  for (auto deviceFunc : deviceFuncs) {
    LDBG("Processing device func:\n" << *deviceFunc);
    // Extract the tiling data index
    SmallVector<int> tilingDataDeviceIndex;
    for (auto [deviceIdx, arg] : llvm::enumerate(deviceFunc.getArguments())) {
      if (deviceFunc.getArgAttr(deviceIdx, mtfusion::TilingDataAttr::name)) {
        tilingDataDeviceIndex.push_back(deviceIdx);
      }
    }
    assert(tilingDataDeviceIndex.size() == returnOp.getNumOperands());
    OpBuilder builder(&deviceFunc.getBody().front().front());

    // For each of device function clone all the constants
    for (auto [idx, constOp] : constantCalcFuncRets) {
      // Clone the constant op
      auto clonedConstOp = cast<arith::ConstantOp>(builder.clone(*constOp));
      Value replacement = clonedConstOp->getResult(0);
      auto replacementTarget =
          deviceFunc.getArgument(tilingDataDeviceIndex[idx]);
      if (clonedConstOp.getType() != replacementTarget.getType()) {
        replacement = convertScalarToDtype(
            builder, replacement.getLoc(), replacement,
            replacementTarget.getType(), /*isUnsignedCast=*/true);
      }
      // Replace uses of the corresponding argument in the device function
      deviceFunc.getArgument(tilingDataDeviceIndex[idx])
          .replaceAllUsesWith(replacement);
    }
    LDBG("Successfully modified device func:\n" << *deviceFunc);
  }

  return success();
}

} // namespace

std::unique_ptr<Pass> createConstantizeTilingDataPass() {
  return std::make_unique<ConstantizeTilingDataPass>();
}

} // namespace mtfusion
} // namespace mlir