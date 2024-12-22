//===- TilingUtils.cpp -- Utilities for Auto Schedule Tiling ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements tiling utilties for auto scheduler.
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/TilingUtils.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/KernelInfo.h"

#include "mtir/Dialect/MtFusion/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::mtfusion;

namespace {

/// By convension, tiling key is the first tiling data.
constexpr size_t kTilingKeyPos = 0;

Value evaluateAffineExpr(AffineExpr e, const SmallVector<OpFoldResult> &symbols,
                         OpBuilder &opBuilder) {
  return affine::makeComposedAffineApply(opBuilder, opBuilder.getUnknownLoc(),
                                         e, symbols)
      ->getResult(0);
}

} // namespace

NamedAttribute mtfusion::getTilingDataAttr(OpBuilder &opBuilder) {
  return NamedAttribute{opBuilder.getStringAttr(mtfusion::TilingDataAttr::name),
                        opBuilder.getUnitAttr()};
}

NamedAttribute mtfusion::getTilingKeyAttr(OpBuilder &opBuilder) {
  return NamedAttribute{opBuilder.getStringAttr(mtfusion::TilingKeyAttr::name),
                        opBuilder.getUnitAttr()};
}

//===----------------------------------------------------------------------===//
// Expr Implementation
//===----------------------------------------------------------------------===//

Expr Expr::operator+(int64_t cst) {
  MLIRContext *ctx = getContext();
  AffineExpr lhs = getAffineSymbolExpr(0, ctx);
  AffineExpr rhs = getAffineConstantExpr(cst, ctx);
  AffineExpr result = lhs + rhs;
  return Expr(evaluateAffineExpr(result, /*symbols=*/{this->v_}, *builder_),
              ExprKind::kRegular, builder_);
}

Expr Expr::operator+(const Expr &other) {
  MLIRContext *ctx = getContext();
  AffineExpr lhs = getAffineSymbolExpr(0, ctx);
  AffineExpr rhs = getAffineSymbolExpr(1, ctx);
  AffineExpr result = lhs + rhs;
  return Expr(
      evaluateAffineExpr(result, /*symbols=*/{this->v_, other.v_}, *builder_),
      ExprKind::kRegular, builder_);
}

Expr Expr::operator-(const Expr &other) {
  MLIRContext *ctx = getContext();
  AffineExpr lhs = getAffineSymbolExpr(0, ctx);
  AffineExpr rhs = getAffineSymbolExpr(1, ctx);
  AffineExpr result = lhs - rhs;
  return Expr(
      evaluateAffineExpr(result, /*symbols=*/{this->v_, other.v_}, *builder_),
      ExprKind::kRegular, builder_);
}

Expr Expr::floorDiv(uint64_t cst) {
  MLIRContext *ctx = getContext();
  AffineExpr lhs = getAffineSymbolExpr(0, ctx);
  AffineExpr rhs = getAffineConstantExpr(cst, ctx);
  AffineExpr result = lhs.floorDiv(rhs);
  return Expr(evaluateAffineExpr(result, /*symbols=*/{this->v_}, *builder_),
              ExprKind::kRegular, builder_);
}

Expr Expr::floorDiv(const Expr &other) {
  MLIRContext *ctx = getContext();
  AffineExpr lhs = getAffineSymbolExpr(0, ctx);
  AffineExpr rhs = getAffineSymbolExpr(1, ctx);
  AffineExpr result = lhs.floorDiv(rhs);
  return Expr(
      evaluateAffineExpr(result, /*symbols=*/{this->v_, other.v_}, *builder_),
      ExprKind::kRegular, builder_);
}

Expr Expr::operator*(int64_t cst) {
  MLIRContext *ctx = getContext();
  AffineExpr lhs = getAffineSymbolExpr(0, ctx);
  AffineExpr rhs = getAffineConstantExpr(cst, ctx);
  AffineExpr result = lhs * rhs;
  return Expr(evaluateAffineExpr(result, /*symbols=*/{this->v_}, *builder_),
              ExprKind::kRegular, builder_);
}

Expr Expr::operator*(const Expr &other) {
  MLIRContext *ctx = getContext();
  AffineExpr lhs = getAffineSymbolExpr(0, ctx);
  AffineExpr rhs = getAffineSymbolExpr(1, ctx);
  AffineExpr result = lhs * rhs;
  return Expr(
      evaluateAffineExpr(result, /*symbols=*/{this->v_, other.v_}, *builder_),
      ExprKind::kRegular, builder_);
}

Expr Expr::alignTo(uint64_t align) {
  MLIRContext *ctx = getContext();
  assert(align != 0u && "Align can't be 0.");
  AffineExpr alignExpr = getAffineConstantExpr(align, ctx);
  AffineExpr self = getAffineSymbolExpr(0, ctx);
  AffineExpr result = ((self + alignExpr) - 1).floorDiv(alignExpr) * alignExpr;
  return Expr(evaluateAffineExpr(result, /*symbols=*/{this->v_}, *builder_),
              ExprKind::kRegular, builder_);
}

Expr Expr::alignDown(uint64_t align) {
  MLIRContext *ctx = getContext();
  assert(align != 0u && "Align can't be 0.");
  AffineExpr alignExpr = getAffineConstantExpr(align, ctx);
  AffineExpr self = getAffineSymbolExpr(0, ctx);
  AffineExpr result = self.floorDiv(alignExpr) * alignExpr;
  return Expr(evaluateAffineExpr(result, /*symbols=*/{this->v_}, *builder_),
              ExprKind::kRegular, builder_);
}

Expr Expr::operator>(int64_t cst) {
  Expr vExpr = builder_->createConstExpr(cst);
  Value result = builder_->create<arith::CmpIOp>(
      this->v_.getLoc(), ::arith::CmpIPredicate::sgt, this->v_, vExpr.v_);
  return Expr(castToIndex(result, *builder_), ExprKind::kRegular, builder_);
}

Expr Expr::operator==(const Expr &other) {
  Value result = builder_->create<arith::CmpIOp>(
      this->v_.getLoc(), ::arith::CmpIPredicate::eq, this->v_, other.v_);
  return Expr(castToIndex(result, *builder_), ExprKind::kRegular, builder_);
}

Expr Expr::operator==(int64_t cst) {
  Expr vExpr = builder_->createConstExpr(cst);
  return this->operator==(vExpr);
}

Expr Expr::operator<=(const Expr &other) {
  Value result = builder_->create<arith::CmpIOp>(
      this->v_.getLoc(), ::arith::CmpIPredicate::sle, this->v_, other.v_);
  return Expr(castToIndex(result, *builder_), ExprKind::kRegular, builder_);
}

Expr mtfusion::max(Expr lhs, int64_t rhs) {
  ExprBuilder &builder = lhs.getBuilder();
  Expr rhsExpr = builder.createConstExpr(rhs);
  Value result = builder.create<arith::MaxSIOp>(
      lhs.getMaterializedValue().getLoc(), lhs.getMaterializedValue(),
      rhsExpr.getMaterializedValue());
  return Expr(result, ExprKind::kRegular, &builder);
}

Expr mtfusion::min(Expr lhs, Expr rhs) {
  ExprBuilder &builder = lhs.getBuilder();
  Value result = builder.create<arith::MinSIOp>(
      lhs.getMaterializedValue().getLoc(), lhs.getMaterializedValue(),
      rhs.getMaterializedValue());
  return Expr(result, ExprKind::kRegular, &builder);
}

Expr mtfusion::select(Expr condition, Expr trueValue, Expr falseValue) {
  ExprBuilder &builder = condition.getBuilder();
  return condition * trueValue +
         (builder.createConstExpr(1) - condition) * falseValue;
}

//===----------------------------------------------------------------------===//
// ExprBuilder Implementation
//===----------------------------------------------------------------------===//

Expr ExprBuilder::createConstExpr(int64_t cst) {
  return Expr(
      evaluateAffineExpr(mlir::getAffineConstantExpr(cst, this->getContext()),
                         /*symbols=*/{}, *this),
      ExprKind::kRegular, this);
}

Expr ExprBuilder::createDimSymbolExpr(Value tensorValue, size_t dimIdx) {
  assert(tilingInfo_);
  assert(isa<ShapedType>(tensorValue.getType()) &&
         "source value must be shaped type!");
  auto dimValue =
      this->create<tensor::DimOp>(this->getUnknownLoc(), tensorValue, dimIdx)
          ->getOpResult(0);
  return DimSymbol(dimValue, this);
}

Expr ExprBuilder::createDimSymbolExpr(size_t tensorIdx, size_t dimIdx) {
  assert(tilingInfo_);
  Value hostTilingArg = tilingInfo_->getHostTilingFuncArg(tensorIdx);
  Value maybeReshapedInput = this->kernelInfo_->inputValues[tensorIdx];
  if (isa<BlockArgument>(maybeReshapedInput)) {
    // create symbol from tensor arg directly
    return createDimSymbolExpr(hostTilingArg, dimIdx);
  }
  // create symbol from reshape op result
  SmallVector<Operation *> reshapeTrace =
      mtfusion::getReshapeOrSliceOpProduceTrace(maybeReshapedInput);
  assert(!reshapeTrace.empty() && "reshape trace must not be empty");
  Value reshapeProducer = mtfusion::getReshapeOrSliceSource(reshapeTrace.back());
  assert(isa<BlockArgument>(reshapeProducer) &&
         "src of reshape op should be block argument");

  Value hostTilingV = hostTilingArg;
  Expr expr;
  // replace reshape op one by one from arg to users
  for (Operation *reshapeOp : llvm::reverse(reshapeTrace)) {
    Value reshapeSrc = mtfusion::getReshapeOrSliceSource(reshapeOp);
    // clone reshape op and replace src with value in host tiling func
    IRMapping mapper;
    mapper.map(reshapeSrc, hostTilingV);
    Operation *clonedReshapeOp = this->clone(*reshapeOp, mapper);
    Value clonedReshapeV = clonedReshapeOp->getResult(0);
    expr = createDimSymbolExpr(clonedReshapeV, dimIdx);
    // update current host tiling value using the result reshaped value
    hostTilingV = clonedReshapeV;
  }
  return expr;
}

SmallVector<Expr> ExprBuilder::createDimSymbolExprs(size_t tensorIdx,
                                                    size_t startDim,
                                                    size_t endDim) {
  return llvm::map_to_vector(
      llvm::to_vector(llvm::seq<size_t>(startDim, endDim)),
      [this, &tensorIdx](size_t idx) -> Expr {
        return this->createDimSymbolExpr(
            /*tensorIdx=*/tensorIdx, /*dimIdx=*/idx);
      });
}

SmallVector<Expr> mtfusion::tiling::getAccumulatedDims(SmallVector<Expr> dims) {
  SmallVector<Expr> accumulatedDims = {dims.front()};
  for (const auto &dim : llvm::drop_begin(dims)) {
    auto accumulatedValue = accumulatedDims.back() * dim;
    accumulatedDims.push_back(accumulatedValue);
  }
  return accumulatedDims;
}

//===----------------------------------------------------------------------===//
// TilingData Implementation
//===----------------------------------------------------------------------===//

Expr *TilingData::getExpr() const {
  assert(!isConst());
  return std::get<std::unique_ptr<Expr>>(data_).get();
}

int64_t TilingData::getConst() const {
  assert(isConst());
  return std::get<int64_t>(data_);
}

//===----------------------------------------------------------------------===//
// TilingInfo Implementation
//===----------------------------------------------------------------------===//

SmallVector<Value> TilingInfo::evaluateTilingComputation(TilingComputeFn fn,
                                                         KernelInfo *kernelInfo,
                                                         ExprBuilder *builder) {
  std::tie(this->caseKeys_, this->struct_) = fn(kernelInfo, builder);
  SmallVector<Value> returns;
  for (std::unique_ptr<TilingData> &data : struct_) {
    Value exprValue = data->getExpr()->getMaterializedValue();
    Value castedValue = castIndexTo(exprValue, data->getType(), *builder);
    returns.push_back(castedValue);
  }
  return returns;
}

SmallVector<TilingData *> TilingInfo::getTilingStruct() {
  return llvm::map_to_vector(struct_,
                             [](TilingDataPtr &td) { return td.get(); });
}

BlockArgument TilingInfo::getHostTilingFuncArg(size_t idx) {
  assert(idx < hostTilingFunc_.getNumArguments());
  return hostTilingFunc_.getArgument(idx);
}

void TilingInfo::recordKernelFunc(TilingKey key, func::FuncOp f) {
  tilingKey2Kernel_.insert({key, f});
}

DenseMap<TilingKey, func::FuncOp> TilingInfo::getTilingKey2KernelMap() {
  return tilingKey2Kernel_;
}

LogicalResult TilingInfo::trySimplifyTilingFunc() {
  // Simplify host tiling func
  PassManager pm(hostTilingFunc_->getContext());
  CanonicalizerOptions options;
  // options.enableExtendedPattern = true;
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  if (failed(pm.run(hostTilingFunc_)))
    return failure();

  // Get return value and see if it's constant
  func::ReturnOp returnOp;
  hostTilingFunc_.walk([&returnOp](func::ReturnOp op) { returnOp = op; });

  for (auto [returnVal, tilingDataPtr] :
       llvm::zip_equal(returnOp->getOperands(), this->struct_)) {
    std::optional<int64_t> maybeConst =
        getConstantIntValue(getAsOpFoldResult(returnVal));
    if (!maybeConst.has_value())
      continue;
    tilingDataPtr->setData(maybeConst.value());
  }
  return success();
}

void TilingInfo::pruneTilingExcept(int64_t keepKey) {
  caseKeys_.remove_if([&](int64_t key) { return key != keepKey; });
}

TilingData *TilingInfo::getTilingData(unsigned idx) {
  assert(idx < size());
  return struct_[idx].get();
}

TilingData *TilingInfo::getTilingData(unsigned idx) const {
  assert(idx < size());
  return struct_[idx].get();
}

TilingData *TilingInfo::getTilingKey() { return getTilingData(kTilingKeyPos); }