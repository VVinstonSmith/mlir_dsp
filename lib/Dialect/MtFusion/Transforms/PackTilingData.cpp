//===- PackTilingData.cpp ------- Pack Tiling Data Pass -------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to pack dynamic tiling information into a struct.
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Transforms/Passes.h"
#include "mtir/Dialect/MtFusion/Utils/Utils.h"
#include "mtir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
namespace mtfusion {

#define GEN_PASS_DEF_PACKTILINGDATA
#include "mtir/Dialect/MtFusion/Transforms/Passes.h.inc"

namespace {

// TODO: Refactor to use tiling utils.
struct PackTilingDataPass
    : public impl::PackTilingDataBase<PackTilingDataPass> {

  void runOnOperation() final;

private:
  LogicalResult modifyCalculateTilingFunc(func::FuncOp &curCalcFunc);
  LogicalResult modifyDeviceFunc(func::FuncOp &curDevFunc);
  LogicalResult fixTilingCallSites(func::FuncOp &curCalcFunc);
  LogicalResult fixDeviceCallSites(func::FuncOp &curDevFunc);

  inline DictionaryAttr getTilingStructAttr(OpBuilder &builder);

private:
  MemRefType curTilingInfoType{};
  BitVector curArgIndicesToBeErased;
};

DictionaryAttr PackTilingDataPass::getTilingStructAttr(OpBuilder &builder) {
  return builder.getDictionaryAttr(
      NamedAttribute{builder.getStringAttr(mtfusion::TilingStructAttr::name),
                     builder.getUnitAttr()});
}

static bool noTilingData(func::FuncOp deviceFn) {
  for (size_t idx = 0; idx < deviceFn.getNumArguments(); idx++) {
    if (deviceFn.getArgAttr(idx, mtfusion::TilingDataAttr::name)) {
      return false;
    }
  }
  return true;
}

void PackTilingDataPass::runOnOperation() {

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
      // Pack for tiling data calculation functions
      if (failed(tiling::verifyCalcTiling(calcFunc, devFuncs)))
        return signalPassFailure();
      if (failed(modifyCalculateTilingFunc(calcFunc)))
        signalPassFailure();
      for (auto &devFunc : devFuncs) {
        // Bail out if device func doesn't contain any tiling data
        if (noTilingData(devFunc))
          continue;
        // Unpack for scheduled functions
        if (failed(modifyDeviceFunc(devFunc)))
          signalPassFailure();
        // Fix callsites
        if (failed(fixDeviceCallSites(devFunc)))
          signalPassFailure();
      }
      if (failed(fixTilingCallSites(calcFunc)))
        signalPassFailure();
    } else {
      mod.emitError("Corresponding calcFunc not found: ") << calcFuncName;
      return signalPassFailure();
    }
  }
}

LogicalResult
PackTilingDataPass::modifyCalculateTilingFunc(func::FuncOp &curCalcFunc) {
  const unsigned returnNum = curCalcFunc.getNumResults();

  // Bail out on empty host tiling func
  if (returnNum == 0)
    return success();

  OpBuilder builder(curCalcFunc);
  curTilingInfoType = MemRefType::get({returnNum}, builder.getIntegerType(64));

  // Update function signature
  curCalcFunc.insertArgument(curCalcFunc.getNumArguments(), curTilingInfoType,
                             getTilingStructAttr(builder),
                             curCalcFunc.getLoc());
  curCalcFunc.setFunctionType(
      curCalcFunc.getFunctionType().clone(curCalcFunc.getArgumentTypes(), {}));
  auto tilingStruct = curCalcFunc.getArguments().back();

  // Replace terminators
  for (func::ReturnOp returnOp : curCalcFunc.getOps<func::ReturnOp>()) {
    builder.setInsertionPoint(returnOp);
    for (unsigned i = 0; i < returnNum; ++i) {
      Value idx =
          builder.create<arith::ConstantIndexOp>(curCalcFunc.getLoc(), i);
      builder.create<memref::StoreOp>(curCalcFunc.getLoc(),
                                      returnOp.getOperand(i), tilingStruct,
                                      ValueRange{idx});
    }
    returnOp->setOperands({});
  }
  return success();
}

LogicalResult PackTilingDataPass::fixDeviceCallSites(func::FuncOp &curDevFunc) {
  SetVector<func::FuncOp> visited;
  SymbolTable::UseRange uses = *curDevFunc.getSymbolUses(getOperation());
  for (SymbolTable::SymbolUse use : uses) {
    func::CallOp call = cast<func::CallOp>(use.getUser());

    SmallVector<Value> newOperands;
    SmallVector<Value> tilingOperands;

    for (size_t i = 0, e = call.getNumOperands(); i < e; ++i) {
      const auto oldOperand = call.getOperand(i);
      if (!curArgIndicesToBeErased.test(i)) {
        newOperands.push_back(oldOperand);
      } else {
        tilingOperands.push_back(oldOperand);
      }
    }

    assert(!tilingOperands.empty());
    auto calcTilingOp = tilingOperands[0].getDefiningOp<func::CallOp>();
    if (failed(tiling::checkCallCalcTilingWithTilingOperands(calcTilingOp,
                                                             tilingOperands)))
      return failure();

    func::FuncOp callSite = call->getParentOfType<func::FuncOp>();
    OpBuilder builder(call);

    // Add memref in host function if exist
    if (!visited.contains(callSite)) {
      callSite.insertArgument(callSite.getNumArguments(), curTilingInfoType,
                              getTilingStructAttr(builder), callSite.getLoc());
      visited.insert(callSite);
    }
    newOperands.push_back(callSite.getArguments().back());
    func::CallOp newCall = builder.create<func::CallOp>(
        call.getLoc(), call.getCalleeAttr(), call.getResultTypes(),
        ValueRange(newOperands));
    newCall.replaceAllUsesWith(call);
    call->erase();
  }
  return success();
}

LogicalResult
PackTilingDataPass::fixTilingCallSites(func::FuncOp &curCalcFunc) {
  SymbolTable::UseRange uses = *curCalcFunc.getSymbolUses(getOperation());
  for (SymbolTable::SymbolUse use : uses) {
    func::CallOp call = cast<func::CallOp>(use.getUser());
    func::FuncOp callSite = call->getParentOfType<func::FuncOp>();
    OpBuilder builder(call);
    SmallVector<Value> newOperands(call->getOperands());
    newOperands.push_back(callSite.getArguments().back());
    builder.create<func::CallOp>(call.getLoc(), call.getCalleeAttr(),
                                 TypeRange(), ValueRange(newOperands));
    call->erase();
  }
  return success();
}

LogicalResult PackTilingDataPass::modifyDeviceFunc(func::FuncOp &curDevFunc) {
  // Collect arguments indices that contain "mtfusion.tiling_data" attribute
  SmallVector<size_t> tilingDataIndices;
  OpBuilder builder(curDevFunc);
  curDevFunc.insertArgument(curDevFunc.getNumArguments(), curTilingInfoType,
                            getTilingStructAttr(builder), curDevFunc.getLoc());

  curArgIndicesToBeErased.reset();
  curArgIndicesToBeErased.resize(curDevFunc.getNumArguments(), false);

  for (size_t i = 0, e = curDevFunc.getNumArguments(); i != e; ++i) {
    if (curDevFunc.getArgAttr(i, mtfusion::TilingDataAttr::name)) {
      tilingDataIndices.push_back(i);
      curArgIndicesToBeErased.set(i);
    }
  }
  const size_t numFields = curTilingInfoType.getShape()[0];
  assert(tilingDataIndices.size() == numFields);

  builder.setInsertionPointToStart(&curDevFunc.front());
  for (size_t i = 0; i < numFields; ++i) {
    Value idx = builder.create<arith::ConstantIndexOp>(curDevFunc.getLoc(), i);
    Value loadedField = builder.create<memref::LoadOp>(
        curDevFunc.getLoc(), curDevFunc.getArguments().back(), ValueRange{idx});
    curDevFunc.getArgument(tilingDataIndices[i])
        .replaceAllUsesWith(loadedField);
  }
  curDevFunc.eraseArguments(curArgIndicesToBeErased);
  return success();
}

} // namespace

std::unique_ptr<Pass> createPackTilingDataPass() {
  return std::make_unique<PackTilingDataPass>();
}

} // namespace mtfusion
} // namespace mlir