//===- GenericToNamedConversion.cpp - Linalg Generic To Named ops Pass ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <set>

namespace mlir {
#define GEN_PASS_DEF_GENERICTONAMEDCONVERSION
#include "mtir/Dialect/MtFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::mtfusion;

static constexpr int kUnaryInputSize = 1;
static constexpr int kBinaryInputSize = 2;

std::set<std::string> linalgUnarySet = {"math.exp", "math.absf", "math.log"};
std::set<std::string> linalgBinarySet = {
    "arith.addf",  "arith.mulf",  "arith.subf",  "arith.divf",
    "arith.maxui", "arith.maxsi", "arith.minui", "arith.minsi"};
std::set<std::string> mtfusionUnarySet = {
    "math.sqrt", "math.rsqrt", "arith.divf", "arith.maximumf", "arith.xori"};
std::set<std::string> mtfusionBinarySet = {"arith.andi", "arith.ori"};

class GenericToNamedConversion
    : public impl::GenericToNamedConversionBase<GenericToNamedConversion> {
public:
  void runOnOperation() override;
};

static linalg::UnaryFn getLinalgUnaryFnKind(std::string opName) {
  linalg::UnaryFn kind;
  if (opName == "math.exp") {
    kind = linalg::UnaryFn::exp;
  } else if (opName == "math.absf") {
    kind = linalg::UnaryFn::abs;
  } else if (opName == "math.log") {
    kind = linalg::UnaryFn::log;
  }
  return kind;
}

static linalg::BinaryFn getLinalgBinaryFnKind(std::string opName) {
  linalg::BinaryFn kind;
  if (opName == "arith.addf") {
    kind = linalg::BinaryFn::add;
  } else if (opName == "arith.mulf") {
    kind = linalg::BinaryFn::mul;
  } else if (opName == "arith.subf") {
    kind = linalg::BinaryFn::sub;
  } else if (opName == "arith.divf") {
    kind = linalg::BinaryFn::div;
  } else if (opName == "arith.maxui") {
    kind = linalg::BinaryFn::max_unsigned;
  } else if (opName == "arith.maxsi") {
    kind = linalg::BinaryFn::max_signed;
  } else if (opName == "arith.minui") {
    kind = linalg::BinaryFn::min_unsigned;
  } else if (opName == "arith.minsi") {
    kind = linalg::BinaryFn::min_signed;
  }
  return kind;
}

static mtfusion::UnaryFn getMtFusionUnaryFnKind(std::string opName) {
  mtfusion::UnaryFn kind;
  if (opName == "math.sqrt") {
    kind = mtfusion::UnaryFn::sqrt;
  } else if (opName == "math.rsqrt") {
    kind = mtfusion::UnaryFn::rsqrt;
  } else if (opName == "arith.divf") {
    kind = mtfusion::UnaryFn::rec;
  } else if (opName == "arith.maximumf") {
    kind = mtfusion::UnaryFn::relu;
  } else if (opName == "arith.xori") {
    kind = mtfusion::UnaryFn::vnot;
  }
  return kind;
}

static mtfusion::BinaryFn getMtFusionBinaryFnKind(std::string opName) {
  mtfusion::BinaryFn kind;
  if (opName == "arith.andi") {
    kind = mtfusion::BinaryFn::vand;
  } else if (opName == "arith.ori") {
    kind = mtfusion::BinaryFn::vor;
  }
  return kind;
}

static Operation *getSingleComputeOp(linalg::GenericOp genericOp) {
  Block &body = genericOp.getRegion().front();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  auto bodyOp = yieldOp.getValues()[0].getDefiningOp();
  assert(bodyOp != nullptr);
  return bodyOp;
}

template <typename T>
static bool isConstantValue(Value oper, T cstValue) {
  auto defOp = oper.getDefiningOp();
  if (defOp && isa<arith::ConstantOp>(defOp)) {
    auto cstOp = cast<arith::ConstantOp>(defOp);
    auto floatAttr = cast<FloatAttr>(cstOp.getValue());
    if (floatAttr.getValueAsDouble() == cstValue)
      return true;
  }
  return false;
}

template <typename T>
static bool isConstantOp(linalg::GenericOp genericOp, Operation *bodyOp,
                         T cstValue) {
  for (auto oper : genericOp.getInputs()) {
    if (isConstantValue(oper, cstValue))
      return true;
  }
  for (auto oper : bodyOp->getOperands()) {
    if (isConstantValue(oper, cstValue))
      return true;
  }
  return false;
}

static bool isReluOp(linalg::GenericOp genericOp) {
  auto bodyOp = getSingleComputeOp(genericOp);
  auto opName = bodyOp->getName().getStringRef().str();
  if (opName != "arith.maximumf")
    return false;
  double zero = 0.0;
  if (isConstantOp(genericOp, bodyOp, zero))
    return true;
  return false;
}

static bool isRecOp(linalg::GenericOp genericOp) {
  auto bodyOp = getSingleComputeOp(genericOp);
  auto opName = bodyOp->getName().getStringRef().str();
  if (opName != "arith.divf")
    return false;
  double one = 1.0;
  if (isConstantOp(genericOp, bodyOp, one))
    return true;
  return false;
}

static Attribute getElemwiseFunAttr(linalg::GenericOp genericOp,
                                    std::string opName) {
  Attribute attr;
  if (opName == "arith.maximumf" || opName == "arith.divf") {
    if (isReluOp(genericOp) || isRecOp(genericOp)) {
      mtfusion::UnaryFn kind = getMtFusionUnaryFnKind(opName);
      attr = mtfusion::UnaryFnAttr::get(genericOp.getContext(), kind);
    } else {
      linalg::BinaryFn kind = getLinalgBinaryFnKind(opName);
      attr = linalg::BinaryFnAttr::get(genericOp.getContext(), kind);
    }
    return attr;
  }

  if (mtfusionUnarySet.count(opName)) {
    mtfusion::UnaryFn kind = getMtFusionUnaryFnKind(opName);
    attr = mtfusion::UnaryFnAttr::get(genericOp.getContext(), kind);
  } else if (linalgUnarySet.count(opName)) {
    linalg::UnaryFn kind = getLinalgUnaryFnKind(opName);
    attr = linalg::UnaryFnAttr::get(genericOp.getContext(), kind);
  } else if (mtfusionBinarySet.count(opName)) {
    mtfusion::BinaryFn kind = getMtFusionBinaryFnKind(opName);
    attr = mtfusion::BinaryFnAttr::get(genericOp.getContext(), kind);
  } else if (linalgBinarySet.count(opName)) {
    linalg::BinaryFn kind = getLinalgBinaryFnKind(opName);
    attr = linalg::BinaryFnAttr::get(genericOp.getContext(), kind);
  }
  return attr;
}

static SmallVector<Value> getInputsValue(linalg::GenericOp genericOp,
                                         int inputSize, Attribute attr) {
  SmallVector<Value> src;
  auto bodyOp = getSingleComputeOp(genericOp);
  if (auto unaryFnAttr = dyn_cast<mtfusion::UnaryFnAttr>(attr)) {
    if ((unaryFnAttr.getValue() == mtfusion::UnaryFn::rec ||
         unaryFnAttr.getValue() == mtfusion::UnaryFn::relu) &&
        inputSize == kBinaryInputSize) {
      auto oper0 = genericOp.getInputs()[0];
      auto oper1 = genericOp.getInputs()[1];
      if (oper0.getDefiningOp())
        src = {oper1};
      else
        src = {oper0};
      return src;
    }
  } else if ((isa<mtfusion::BinaryFnAttr>(attr) ||
              isa<linalg::BinaryFnAttr>(attr)) &&
             inputSize == kUnaryInputSize) {
    auto oper0 = bodyOp->getOperands()[0];
    auto oper1 = bodyOp->getOperands()[1];
    if (oper0.getDefiningOp())
      src = {oper0, genericOp.getInputs()[0]};
    else
      src = {genericOp.getInputs()[0], oper1};
    return src;
  }
  return genericOp.getInputs();
}

static Operation *createElemwiseOp(linalg::GenericOp genericOp,
                                   PatternRewriter &rewriter, int inputSize,
                                   Attribute attr) {
  auto loc = genericOp.getLoc();
  SmallVector<Value> src = getInputsValue(genericOp, inputSize, attr);
  SmallVector<Value> dst = genericOp.getOutputs();
  SmallVector<NamedAttribute> attrs;
  auto nameAttr = StringAttr::get(genericOp.getContext(), "fun");
  attrs.push_back({nameAttr, attr});
  Operation *namedOp = nullptr;
  if (isa<mtfusion::UnaryFnAttr>(attr) || isa<linalg::UnaryFnAttr>(attr)) {
    namedOp = rewriter.create<linalg::ElemwiseUnaryOp>(loc, src, dst, attrs);
  } else if (isa<mtfusion::BinaryFnAttr>(attr) || isa<linalg::BinaryFnAttr>(attr)) {
    namedOp = rewriter.create<linalg::ElemwiseBinaryOp>(loc, src, dst, attrs);
  }
  return namedOp;
}

static LogicalResult generateNamedOp(linalg::GenericOp genericOp,
                                     PatternRewriter &rewriter) {
  auto bodyOp = getSingleComputeOp(genericOp);
  auto opName = bodyOp->getName().getStringRef().str();
  auto inputSize = genericOp.getInputs().size();
  Attribute attr = getElemwiseFunAttr(genericOp, opName);
  Operation *namedOp = createElemwiseOp(genericOp, rewriter, inputSize, attr);
  rewriter.replaceOp(genericOp, namedOp->getResults());
  return success();
}

static bool atLeastOneComputeOp(linalg::GenericOp genericOp) {
  Block &body = genericOp.getRegion().front();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  auto bodyOp = yieldOp.getValues()[0].getDefiningOp();
  return bodyOp != nullptr;
}

struct ConvertElemwiseLinalgGenericOps
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!linalg::isElementwise(op) || !atLeastOneComputeOp(op)) {
      llvm::dbgs() << "unsupport named structure for this generic type.";
      return failure();
    }
    return generateNamedOp(op, rewriter);
  }
};

void GenericToNamedConversion::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<ConvertElemwiseLinalgGenericOps>(patterns.getContext());
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<Pass> mlir::mtfusion::createGenericToNamedConversionPass() {
  return std::make_unique<GenericToNamedConversion>();
}