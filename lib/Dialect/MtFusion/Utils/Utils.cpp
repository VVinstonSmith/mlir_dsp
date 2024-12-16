//===-----------------------------Utils.cpp--------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Utils/Utils.h"
#include "mtir/Dialect/Utils/Util.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"

#include <optional>
#include <unordered_set>

using namespace mlir;
using namespace mlir::mtfusion;

#define DEBUG_TYPE "mtfusion-utils"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

// TODO: Refactor ArithToMtFusion pass to use this util

tensor::EmptyOp mtfusion::createEmptyOpWithTargetElemType(OpBuilder &builder,
                                                         Location loc,
                                                         Value source,
                                                         Type targetElemType) {
  auto tensorType = source.getType().cast<TensorType>();
  ArrayRef<int64_t> staticShapes = tensorType.getShape();
  llvm::SmallVector<Value, 2> dynamicSizes;
  for (size_t i = 0; i < staticShapes.size(); i++) {
    if (staticShapes[i] == ShapedType::kDynamic) {
      auto dynDim = builder.create<tensor::DimOp>(loc, source, i);
      dynamicSizes.push_back(dynDim);
    }
  }
  auto emptyOp = builder.create<tensor::EmptyOp>(loc, staticShapes,
                                                 targetElemType, dynamicSizes);
  return emptyOp;
}

tensor::EmptyOp mtfusion::createEmptyOp(OpBuilder &builder, Location loc,
                                       Value source) {
  auto elementType = getElementTypeOrSelf(source);
  auto emptyOp =
      createEmptyOpWithTargetElemType(builder, loc, source, elementType);
  return emptyOp;
}

Value mtfusion::castTo(OpBuilder &builder, Value src, Type targetElemType,
                      mtfusion::RoundMode roundMode, std::optional<Value> dst) {
  Location loc = src.getLoc();
  if (!isa<TensorType>(src.getType())) {
    assert(src.getType().isIntOrIndexOrFloat());
    return convertScalarToDtype(builder, loc, src, targetElemType,
                                /*isUnsignedCast=*/false);
  }

  Value targetTensor;
  if (dst.has_value()) {
    targetTensor = dst.value();
  } else {
    targetTensor =
        createEmptyOpWithTargetElemType(builder, loc, src, targetElemType);
  }

  auto roundingAttr = builder.getAttr<mtfusion::RoundModeAttr>(roundMode);
  auto modeAttr =
      builder.getNamedAttr(mtfusion::RoundModeAttr::getMnemonic(), roundingAttr);
  auto vcastOp = builder.create<mtfusion::CastOp>(
      loc, SmallVector<Type>{targetTensor.getType()}, src, targetTensor,
      modeAttr);
  return vcastOp->getResult(0);
}

Value mtfusion::castToIndex(Value v, OpBuilder &opBuilder, bool isUnsigned) {
  return isUnsigned ? 
    opBuilder.create<arith::IndexCastUIOp>(
      v.getLoc(), opBuilder.getIndexType(), v)->getResult(0)
    : opBuilder.create<arith::IndexCastOp>(
      v.getLoc(), opBuilder.getIndexType(), v)->getResult(0);
}

Value mtfusion::castIndexTo(Value v, Type t, OpBuilder &opBuilder,
                           bool isUnsigned) {
  return isUnsigned ? opBuilder.create<arith::IndexCastUIOp>(v.getLoc(), t, v)
                          ->getResult(0)
                    : opBuilder.create<arith::IndexCastOp>(v.getLoc(), t, v)
                          ->getResult(0);
}

void tiling::getCallerInfo(func::FuncOp callee, ModuleOp enclosingModule,
                           DenseMap<func::FuncOp, CallerInfo> &info) {
  std::optional<SymbolTable::UseRange> maybeUses =
      callee.getSymbolUses(enclosingModule);
  for (SymbolTable::SymbolUse use : maybeUses.value()) {
    func::CallOp callSite = cast<func::CallOp>(use.getUser());
    auto callerOp = callSite->getParentOfType<func::FuncOp>();
    assert(callerOp != nullptr && "Caller should not be empty!");
    auto &callerInfo = info[callerOp];
    callerInfo.caller = callerOp;
    callerInfo.callee = callee;
    callerInfo.callSites.push_back(callSite);
  }
}

SmallVector<Value> tiling::getCalleeTilingArguments(func::FuncOp callee,
                                                    func::CallOp callSite) {
  SmallVector<Value> tilingOperands;
  for (const auto [idx, operand] : llvm::enumerate(callSite.getOperands())) {
    if (callee.getArgAttr(idx, mtfusion::TilingDataAttr::name)) {
      tilingOperands.push_back(operand);
    }
  }
  return tilingOperands;
}

LogicalResult tiling::doFixCallSite(tiling::CallerInfo &callerInfo,
                                    tiling::CallSiteBuilderInfo &builderInfo,
                                    DenseMap<Operation *, Operation *> &irMap,
                                    OpBuilder &opBuilder) {
  for (func::CallOp callSite : callerInfo.callSites) {
    LDBG("fixing call site: " << *callSite);
    auto newArgs = builderInfo.argBuilderFn(callSite, opBuilder);
    auto tilingArgs =
        tiling::getCalleeTilingArguments(callerInfo.callee, callSite);
    // If the arguments in the callee is the result of calling host tiling func,
    // check the validity.
    for (auto arg : tilingArgs) {
      auto calcTilingOp = arg.getDefiningOp<func::CallOp>();
      if (calcTilingOp && failed(checkCallCalcTilingWithTilingOperands(
                              calcTilingOp, tilingArgs))) {
        return failure();
      }
    }
    opBuilder.setInsertionPoint(callSite);
    if (failed(builderInfo.siteBuilderFn(callSite, opBuilder, newArgs, irMap)))
      return failure();
  }
  return success();
}

LogicalResult tiling::callSiteBuilderFnForTilingModification(
    func::CallOp callSite, OpBuilder &opBuilder,
    const SmallVector<Value> &newArguments,
    DenseMap<Operation *, Operation *> &irMap) {
  func::CallOp newCallSite = opBuilder.create<func::CallOp>(
      callSite.getLoc(), callSite.getResultTypes(), callSite.getCallee(),
      newArguments);
  LDBG("Generated new call site:\n" << *newCallSite);
  irMap.insert(std::make_pair(callSite, newCallSite));
  return success();
}

LogicalResult
tiling::checkCallCalcTilingWithTilingOperands(Operation *calcTilingOp,
                                              ArrayRef<Value> tilingOperands) {
  assert(tilingOperands.size() && isa<func::CallOp>(calcTilingOp));
  assert(calcTilingOp->getNumResults() == tilingOperands.size());
  for (auto [idx, res] : llvm::enumerate(calcTilingOp->getResults())) {
    if (res != tilingOperands[idx]) {
      return calcTilingOp->emitError(
          "Calc tiling order and usage inconsistency");
    }
    if (!res.getType().isInteger(64)) {
      return calcTilingOp->emitError("Non i64 calculate tiling return type");
    }
  }
  return success();
}

LogicalResult tiling::verifyCalcTiling(func::FuncOp &calcFunc,
                                       SmallVector<func::FuncOp> &deviceFuncs) {
  // verify
  for (auto [idx, res] :
       llvm::enumerate(calcFunc.getFunctionType().getResults())) {
    if (!res.isInteger(64)) {
      return calcFunc.emitError("Non i64 calculate tiling return type");
    }
  }
  for (auto [idx, deviceFunc] : llvm::enumerate(deviceFuncs)) {
    int tilingDataDeviceCount = 0;
    for (auto [idxArg, devArg] : llvm::enumerate(deviceFunc.getArguments())) {
      if (deviceFunc.getArgAttr(idxArg, mtfusion::TilingDataAttr::name)) {
        tilingDataDeviceCount++;
        if (!devArg.getType().isInteger(64))
          return deviceFunc.emitError("Non i64 device tiling data args");
      }
    }
    if (tilingDataDeviceCount != calcFunc.getNumResults()) {
      return calcFunc.emitError("Calc tiling order and usage inconsistency");
    }
  }
  return success();
}

bool mtfusion::isReshapeOp(Operation *op) {
  if (!op)
    return false;
  return isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(op);
}

bool mtfusion::isReshapeOrSliceOp(Operation *op) {
  if (!op)
    return false;
  return mtfusion::isReshapeOp(op) || reshape_utils::isSlicingOp(op);
}

Value mtfusion::getReshapeSource(Operation *op) {
  return TypeSwitch<Operation *, Value>(op)
      .Case([](tensor::ExpandShapeOp expand) { return expand.getSrc(); })
      .Case([](tensor::CollapseShapeOp collapse) { return collapse.getSrc(); })
      .Default([](Operation *op) {
        llvm_unreachable("Unsupported reshape op");
        return Value();
      });
}

Value mtfusion::getReshapeResult(Operation *op) {
  return TypeSwitch<Operation *, Value>(op)
      .Case([](tensor::ExpandShapeOp expand) { return expand.getResult(); })
      .Case(
          [](tensor::CollapseShapeOp collapse) { return collapse.getResult(); })
      .Default([](Operation *op) {
        llvm_unreachable("Unsupported reshape op");
        return Value();
      });
}

Value mtfusion::getReshapeOrSliceSource(Operation *op) {
  return TypeSwitch<Operation *, Value>(op)
      .Case([](tensor::ExpandShapeOp expand) { return expand.getSrc(); })
      .Case([](tensor::CollapseShapeOp collapse) { return collapse.getSrc(); })
      .Case([](tensor::ExtractSliceOp extract) { return extract.getSource(); })
      .Case([](tensor::InsertSliceOp insert) { return insert.getSource(); })
      .Default([](Operation *op) {
        llvm_unreachable("Unsupported reshape or slice op");
        return Value();
      });
}

Value mtfusion::getReshapeOrSliceResult(Operation *op) {
  return TypeSwitch<Operation *, Value>(op)
      .Case([](tensor::ExpandShapeOp expand) { return expand.getResult(); })
      .Case(
          [](tensor::CollapseShapeOp collapse) { return collapse.getResult(); })
      .Case([](tensor::ExtractSliceOp extract) { return extract.getResult(); })
      .Case([](tensor::InsertSliceOp insert) { return insert.getResult(); })
      .Default([](Operation *op) {
        llvm_unreachable("Unsupported reshape or slice op");
        return Value();
      });
}

Value mtfusion::traceReshapeOrSliceSingleProducerOrSelf(Value input) {
  auto maybeValue = mtfusion::traceReshapeOrSliceSingleProducer(input);
  if (succeeded(maybeValue))
    return maybeValue.value();
  return input;
}

FailureOr<Value> mtfusion::traceReshapeOrSliceSingleProducer(Value input) {
  LDBG("Tracing reshape single producer for " << input);
  if (isa<BlockArgument>(input)) {
    LDBG("Input is a block argument");
    return failure();
  }

  auto result = cast<OpResult>(input);
  auto *definingOp = result.getOwner();
  if (!mtfusion::isReshapeOrSliceOp(definingOp)) {
    LDBG("Defining op is not reshape");
    return failure();
  }

  auto reshapeSource = mtfusion::getReshapeOrSliceSource(definingOp);
  return mtfusion::traceReshapeOrSliceSingleProducerOrSelf(reshapeSource);
}

SmallVector<Operation *> mtfusion::getReshapeOrSliceOpProduceTrace(Value input) {
  Operation *curOp = input.getDefiningOp();
  SmallVector<Operation *> trace;
  while (mtfusion::isReshapeOrSliceOp(curOp)) {
    trace.push_back(curOp);
    Value reshapeSrc = mtfusion::getReshapeOrSliceSource(curOp);
    if (isa<BlockArgument>(reshapeSrc)) {
      break;
    }
    curOp = reshapeSrc.getDefiningOp();
  };
  return trace;
}

Value mtfusion::traceReshapeOrSliceSingleConsumerOrSelf(Value input) {
  auto maybeValue = mtfusion::traceReshapeOrSliceSingleConsumer(input);
  if (succeeded(maybeValue))
    return maybeValue.value();
  return input;
}

FailureOr<Value> mtfusion::traceReshapeOrSliceSingleConsumer(Value input) {
  LDBG("Tracing reshape or slice single consumer for " << input);
  auto reshapeUsers =
      llvm::make_filter_range(input.getUsers(), [&](Operation *user) {
        return mtfusion::isReshapeOrSliceOp(user);
      });

  if (!llvm::hasSingleElement(reshapeUsers)) {
    LDBG("Input has none or more than one reshape users");
    return failure();
  }

  auto *singleReshape = *(reshapeUsers.begin());
  auto result = mtfusion::getReshapeOrSliceResult(singleReshape);
  return mtfusion::traceReshapeOrSliceSingleConsumerOrSelf(result);
}

bool mtfusion::isCommutativeOp(const Operation *op) {
  auto binLinalgOp = dyn_cast<linalg::ElemwiseBinaryOp>(op);
  if (!binLinalgOp) {
    return false;
  }
  static std::unordered_set<linalg::BinaryFn> supportLinalgOps = {
      linalg::BinaryFn::add,          linalg::BinaryFn::mul,
      linalg::BinaryFn::max_signed,   linalg::BinaryFn::min_signed,
      linalg::BinaryFn::max_unsigned, linalg::BinaryFn::min_unsigned};

  static std::unordered_set<mtfusion::BinaryFn> supportHfusionOps = {
      mtfusion::BinaryFn::maxf, mtfusion::BinaryFn::minf};
  if (binLinalgOp) {
    linalg::BinaryFn binLinalgFn = binLinalgOp.getFunAttr().getValue();
    return supportLinalgOps.find(binLinalgFn) != supportLinalgOps.end();
  }
  return false;
}

bool mtfusion::isHWSupportVSOp(const Operation *op) {
  // only support vs for elemwise binary op currently
  auto binLinalgOp = dyn_cast<linalg::ElemwiseBinaryOp>(op);
  if (!binLinalgOp) {
    return false;
  }
  static std::unordered_set<linalg::BinaryFn> supportLinalgOps = {
      linalg::BinaryFn::add,          linalg::BinaryFn::mul,
      linalg::BinaryFn::max_signed,   linalg::BinaryFn::min_signed,
      linalg::BinaryFn::max_unsigned, linalg::BinaryFn::min_unsigned,
  };

  static std::unordered_set<mtfusion::BinaryFn> supportHfusionOps = {
      mtfusion::BinaryFn::maxf, mtfusion::BinaryFn::minf,
      mtfusion::BinaryFn::powf};
  if (binLinalgOp) {
    linalg::BinaryFn binLinalgFn = binLinalgOp.getFunAttr().getValue();
    return supportLinalgOps.find(binLinalgFn) != supportLinalgOps.end();
  }
  return false;
}

std::optional<FusionKind> mtfusion::tryGetFusionKind(func::FuncOp &func) {
  auto fusionKindAttr =
      func->getAttrOfType<FusionKindAttr>(FusionKindAttr::name);
  if (!fusionKindAttr)
    return std::nullopt;
  return fusionKindAttr.getFusionKind();
}

void mtfusion::trySetFuncKind(func::FuncOp &func,
                               const FuncKind &funcKind) {
  if (func->hasAttr(FuncKindAttr::name)) {
    LLVM_DEBUG(llvm::dbgs()
                   << "Function already has a funcKind, replacing with: "
                   << funcKind << "\n";);
  }
  func->setAttr(FuncKindAttr::name,
                FuncKindAttr::get(func->getContext(), funcKind));
  return;
}

bool mtfusion::isHost(func::FuncOp &func) {
  auto funcKindAttr = func->getAttrOfType<FuncKindAttr>(FuncKindAttr::name);
  if(!funcKindAttr || funcKindAttr.getFunctionKind() != FuncKind::Host)
    return false;
  return true;
}

bool mtfusion::isDevice(func::FuncOp &func) {
  auto funcKindAttr = func->getAttrOfType<FuncKindAttr>(FuncKindAttr::name);
  if(!funcKindAttr || funcKindAttr.getFunctionKind() != FuncKind::Device)
    return false;
  return true;
}

void mtfusion::trySetFusionKind(func::FuncOp &func,
                               const FusionKind &fusionKind) {
  if (func->hasAttr(FusionKindAttr::name)) {
    LLVM_DEBUG(llvm::dbgs()
                   << "Function already has a fusionKind, replacing with: "
                   << fusionKind << "\n";);
  }
  func->setAttr(FusionKindAttr::name,
                FusionKindAttr::get(func->getContext(), fusionKind));
  return;
}

bool reshape_utils::isInitOp(Operation *op) { return isa<tensor::EmptyOp>(op); }

bool reshape_utils::isReshapingOp(Operation *op) {
  return isa<tensor::CollapseShapeOp, tensor::ReshapeOp, tensor::ExpandShapeOp>(
      op);
}

bool reshape_utils::isSlicingOp(Operation *op) {
  return isa<tensor::ExtractSliceOp, tensor::InsertSliceOp>(op);
}

bool reshape_utils::isArgOp(Operation *op) {
  return isReshapingOp(op) || isSlicingOp(op) || isInitOp(op) ||
         isa<arith::ConstantOp, bufferization::ToTensorOp>(op);
}

bool reshape_utils::isStopPropagatable(Operation *op) {
  return isSlicingOp(op) || isInitOp(op) ||
         isa<arith::ConstantOp, bufferization::ToTensorOp>(op);
}

bool reshape_utils::isOutOp(Operation *op) {
  return isReshapingOp(op) || isReturnOp(op);
}

bool reshape_utils::isSkippableOp(Operation *op) {
  return isOutOp(op) || isArgOp(op);
}

bool reshape_utils::isContainerAllocator(Operation *op) {
  return isa<tensor::EmptyOp>(op);
}

bool reshape_utils::isElementwiseOp(Operation *op) {
  if (!isAllParallelOp(op))
    return false;
  auto genericOp = dyn_cast<linalg::LinalgOp>(op);

  LLVM_DEBUG(llvm::dbgs() << *op << "\n";);
  if (llvm::any_of(genericOp.getIndexingMapsArray(),
                   [](AffineMap map) { return !map.isIdentity(); })) {
    return false;
  }
  return true;
}

bool reshape_utils::isMarkedAsElementwiseOp(Operation *op) {
  // This would handle scalar as well
  return isa<linalg::ElemwiseBinaryOp, linalg::ElemwiseUnaryOp,
             mtfusion::CastOp, linalg::FillOp>(op);
}

bool reshape_utils::isAllParallelOp(Operation *op) {
  // Check if it's a Linalg op with all parallel loops
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    bool isAllParallelLoops =
        (linalgOp.getNumLoops() == linalgOp.getNumParallelLoops());
    return isAllParallelLoops;
  }
  return false;
}

bool reshape_utils::isLegalOp(Operation *op) {
  if (isa<linalg::MapOp, linalg::FillOp, linalg::GenericOp,
          linalg::ElemwiseBinaryOp, linalg::ElemwiseUnaryOp,
          linalg::BroadcastOp, linalg::ReduceOp, linalg::TransposeOp,
          mtfusion::CastOp, linalg::MatmulOp,
          linalg::MatmulTransposeAOp, linalg::MatmulTransposeBOp>(op)) {
    return true;
  }
  LLVM_DEBUG(llvm::dbgs() << "Warning: unchecked operation " << *op << "\n");
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    bool isAllParallelLoops =
        linalgOp.getNumLoops() == linalgOp.getNumParallelLoops();
    if (isAllParallelLoops) {
      return true;
    }
  }
  return false;
}

bool reshape_utils::isReturnOp(Operation *op) {
  return isSlicingOp(op) ||
         isa<func::ReturnOp, bufferization::MaterializeInDestinationOp>(op);
}

void mtfusion::setInsertionPointBeforeOrAfter(OpBuilder &builder, Value &value,
                                             bool isAfter) {
  if (BlockArgument blockArg = dyn_cast<BlockArgument>(value)) {

    LLVM_DEBUG(llvm::dbgs() << "here set\n";);
    // If it's a block argument, set insertion point to the start of the block
    builder.setInsertionPointToStart(blockArg.getOwner());
  } else {
    Operation *definingOp = value.getDefiningOp();
    if (definingOp) {
      if (isAfter)
        builder.setInsertionPointAfter(definingOp);
      else
        builder.setInsertionPoint(definingOp);
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "Warning: Non-block argument with no defining op\n");
      if (Operation *parentOp = value.getParentRegion()->getParentOp()) {
        if (!parentOp->getRegions().empty() &&
            !parentOp->getRegion(0).empty()) {
          builder.setInsertionPointToStart(&parentOp->getRegion(0).front());
        }
      }
    }
  }
}

void mtfusion::setInsertionPointAfterValue(OpBuilder &builder, Value &value) {
  setInsertionPointBeforeOrAfter(builder, value, true);
}

void mtfusion::setInsertionPointBeforeValue(OpBuilder &builder, Value &value) {
  setInsertionPointBeforeOrAfter(builder, value, false);
}

/// If the function argument is used as the dps init operand
/// of linalg/mtfusion ops, and the tied result value is returned from from
/// the function, return its result index.
/// This function will also consider the following cases:
///   1) The input argument is reshaped before use
///   2) The result tied to the init operand is reshaped before return
///
/// If the function argument is "tied to" multiple return values, only
/// the first index will be returned.
///
/// For example:
/// ```mlir
///    func.func @foo(%arg0, %arg1)
///      %ret0:N = linalg.ops ins(...) outs(%arg1, ...)
///      func.return %some_value, %ret#0
///  ```
/// The result is 1 (start counting from zero).
std::optional<int64_t> mtfusion::getFuncArgTiedResultReturnIdx(
    BlockArgument &ba, bool &funcArgIsReshaped, bool &funcResultIsReshaped) {
  auto maybeArgReshaped = mtfusion::traceReshapeOrSliceSingleConsumerOrSelf(ba);
  if (mtfusion::isReshapeOrSliceOp(maybeArgReshaped.getDefiningOp()))
    funcArgIsReshaped = true;

  for (OpOperand &use : maybeArgReshaped.getUses()) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(use.getOwner());
    if (!linalgOp)
      continue;

    if (!linalgOp.isDpsInit(&use))
      continue;

    // Check to see if tied result is used by `func.return`
    auto tiedResult = linalgOp.getTiedOpResult(&use);
    auto reshapeOrSelf =
        mtfusion::traceReshapeOrSliceSingleConsumerOrSelf(tiedResult);
    if (mtfusion::isReshapeOrSliceOp(reshapeOrSelf.getDefiningOp()))
      funcResultIsReshaped = true;

    auto maybeOperandOfReturnOp =
        llvm::find_if(reshapeOrSelf.getUses(), [&](OpOperand &operand) {
          return isa<func::ReturnOp>(operand.getOwner());
        });

    if (maybeOperandOfReturnOp == reshapeOrSelf.getUses().end())
      continue;

    // Return the index of the operand in `func.return`
    return static_cast<int64_t>(maybeOperandOfReturnOp->getOperandNumber());
  }
  return std::nullopt;
}

/// Use `operand`'s shape information to create an `tensor.empty` op
/// with the exact same shape.
tensor::EmptyOp
mtfusion::createEmptyOpWithSameShape(OpBuilder &rewriter, Value operand,
                                    SmallPtrSet<Operation *, 4> &newOps,
                                    Location loc) {
  auto tensorType = operand.getType().cast<TensorType>();
  ArrayRef<int64_t> staticShapes = tensorType.getShape();
  llvm::SmallVector<Value, 2> dynamicSizes;
  for (size_t i = 0; i < staticShapes.size(); i++) {
    if (staticShapes[i] == ShapedType::kDynamic) {
      auto dynDim = rewriter.create<tensor::DimOp>(loc, operand, i);
      newOps.insert(dynDim.getOperation());
      dynamicSizes.push_back(dynDim);
    }
  }
  auto emptyOp = rewriter.create<tensor::EmptyOp>(
      loc, staticShapes, tensorType.getElementType(), dynamicSizes);
  return emptyOp;
}

linalg::CopyOp mtfusion::createCacheRead(OpBuilder &rewriter, Value operand,
                                        Location loc) {
  SmallPtrSet<Operation *, 4> newOps;
  auto emptyOp =
      mtfusion::createEmptyOpWithSameShape(rewriter, operand, newOps, loc);
  auto cachedOp = rewriter.create<linalg::CopyOp>(loc, ValueRange{operand},
                                                  ValueRange{emptyOp});
  newOps.insert(emptyOp);
  newOps.insert(cachedOp);
  operand.replaceAllUsesExcept(cachedOp.getResult(0), newOps);
  return cachedOp;
}

FailureOr<linalg::CopyOp>
mtfusion::createCacheWrite(OpBuilder &rewriter, OpResult result, bool outputOnly,
                          bool cacheWriteToOutputInit) {
  auto definingOp = dyn_cast<linalg::LinalgOp>(result.getOwner());
  if (!definingOp)
    return {};

  Location loc = definingOp->getLoc();
  OpBuilder::InsertionGuard guard(rewriter);

  auto initOperand =
      definingOp.getDpsInitOperand(result.getResultNumber())->get();

  linalg::CopyOp cachedOp;
  tensor::EmptyOp emptyOp;
  SmallPtrSet<Operation *, 4> exceptions;

  // If the cache write mode is output-only ...
  if (outputOnly) {
    FailureOr<Value> maybeReshape =
        mtfusion::traceReshapeOrSliceSingleConsumer(result);
    if (succeeded(maybeReshape)) {
      /// If the result is reshaped before return, then the cached result is
      /// only used to replace the original op in the reshape op being returned.
      ///
      /// Before cache write:
      ///
      ///   %result = ...
      ///   some_use(%result)
      ///   %reshaped = reshape(%result)
      ///   %reshaped1 = reshape(%reshaped)
      ///   func.return %reshaped1
      ///
      /// After cache write:
      ///
      ///   %result = ...
      ///   %cached = ...
      ///   %reshaped = reshape(%cached)
      ///   %reshaped1 = reshape(%reshaped)
      ///   func.return %reshaped1
      bool reshapeIsReturned =
          llvm::any_of(maybeReshape->getUsers(), [](Operation *user) {
            return isa<func::ReturnOp>(user);
          });
      if (reshapeIsReturned)
        llvm::for_each(result.getUsers(), [&exceptions](Operation *user) {
          if (mtfusion::isReshapeOrSliceOp(user))
            return;
          exceptions.insert(user);
        });
    } else {
      /// Otherwise, the cached result is only used to replace the original op
      /// in `func.return` op.
      ///
      /// Before cache write:
      ///
      ///   %result = ...
      ///   some_use(%result)
      ///   func.return %result
      ///
      /// After cache write:
      ///
      ///   %result = ...
      ///   %cached = ...
      ///   some_use(%result)
      ///   func.return %cached
      llvm::for_each(result.getUsers(), [&exceptions](Operation *user) {
        if (isa<func::ReturnOp>(user))
          return;
        exceptions.insert(user);
      });
    }
  }

  if (cacheWriteToOutputInit) {
    // Input:
    //   %ret = linalg.op ins(...) outs(%init)
    //
    // After performing cache write to %ret:
    //   %dim = tensor.dim %init
    //   %empty = tensor.empty(%dim)
    //   %ret = linalg.op ins(...) outs(%empty)
    //   linalg.copy ins(%ret) outs(%init)
    rewriter.setInsertionPoint(definingOp);
    // for dynamic shape scenario, need to use `initOperand` to create
    // tensor.dim ops
    emptyOp =
        createEmptyOpWithSameShape(rewriter, initOperand, exceptions, loc);
    definingOp->replaceUsesOfWith(initOperand, emptyOp);
    rewriter.setInsertionPointAfter(definingOp);
    cachedOp = rewriter.create<linalg::CopyOp>(loc, ValueRange{result},
                                               ValueRange{initOperand});
  } else {
    // Input:
    //   %ret = linalg.op ins(...) outs(%init)
    //
    // After performing cache write to %ret:
    //   %ret = linalg.op ins(...) outs(%init)
    //   %dim = tensor.dim %ret
    //   %empty = tensor.empty(%dim)
    //   linalg.copy ins(%ret) outs(%empty)
    rewriter.setInsertionPointAfter(definingOp);
    emptyOp = createEmptyOpWithSameShape(rewriter, result, exceptions, loc);
    cachedOp = rewriter.create<linalg::CopyOp>(loc, ValueRange{result},
                                               ValueRange{emptyOp});
  }

  exceptions.insert(emptyOp);
  exceptions.insert(cachedOp);
  result.replaceAllUsesExcept(cachedOp.getResult(0), exceptions);
  return cachedOp;
}