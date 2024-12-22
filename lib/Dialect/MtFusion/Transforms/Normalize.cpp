//===- Normalize .cpp -------------------- Normalize MtFusion  -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is for normalizing MtFusion.
//
//===----------------------------------------------------------------------===//
#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"
#include "mtir/Dialect/MtFusion/Utils/Utils.h"
#include "mtir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_NORMALIZE
#include "mtir/Dialect/MtFusion/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "mtfusion-normalize-ops"

using namespace mlir;
using namespace mlir::mtfusion;

/// normalize negf op to mul op
/// eg.
///  y = linalg.elemwise_unary {negf} (x)
///  is normalized to
///  y = linalg.elemwise_binary {mul} (x, -1)
struct NormalizeNegToMul : public OpRewritePattern<linalg::ElemwiseUnaryOp> {
public:
  using OpRewritePattern<linalg::ElemwiseUnaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != linalg::UnaryFn::negf) {
      return failure();
    }

    auto input = op.getDpsInputs()[0];
    auto elementType = getElementTypeOrSelf(input.getType());
    Value one = rewriter.create<arith::ConstantOp>(
        op->getLoc(), elementType, rewriter.getFloatAttr(elementType, -1.0));

    linalg::BinaryFn fun = linalg::BinaryFn::mul;
    auto funAttr = rewriter.getAttr<linalg::BinaryFnAttr>(fun);
    auto mulAttr = rewriter.getNamedAttr("fun", funAttr);
    auto mulOP = rewriter.create<linalg::ElemwiseBinaryOp>(
        op->getLoc(), TypeRange(op.getResults()), ValueRange{input, one},
        op.getDpsInits()[0], ArrayRef{mulAttr});
    rewriter.replaceOp(op, mulOP);
    return success();
  }
};

/// normalize div(s, v) to div(fill(s), v)
/// eg.
///  y = linalg.div(1, x)
///  is normalized to
///  y = linalg.div(fill(1), x)
struct NormalizeVDivSVToVV : public OpRewritePattern<linalg::ElemwiseBinaryOp> {
  using OpRewritePattern<linalg::ElemwiseBinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return rewriter.notifyMatchFailure(op, "must have pure tensor semantic!");
    }
    if (op.getFun() != linalg::BinaryFn::div &&
        op.getFun() != linalg::BinaryFn::div_unsigned) {
      return rewriter.notifyMatchFailure(op, "not vdiv op!");
    }
    auto *src0Operand = op.getDpsInputOperand(0);
    if (isa<ShapedType>(src0Operand->get().getType())) {
      return rewriter.notifyMatchFailure(op, "src0 is not a scalar!");
    }
    rewriter.setInsertionPoint(op);
    tensor::EmptyOp empty =
        createEmptyOp(rewriter, op->getLoc(), op.getDpsInitOperand(0)->get());
    auto fillOp = rewriter.create<linalg::FillOp>(
        op->getLoc(),
        /*resultTensorTypes=*/SmallVector<Type>{empty.getType()},
        /*inputs=*/SmallVector<Value>{src0Operand->get()},
        /*outputs=*/SmallVector<Value>{empty});
    rewriter.modifyOpInPlace(
        op, [&]() { src0Operand->assign(fillOp->getResult(0)); });
    return success();
  }
};

struct NormalizeCeilandFloorOp
    : public OpRewritePattern<linalg::ElemwiseUnaryOp> {
public:
  using OpRewritePattern<linalg::ElemwiseUnaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ElemwiseUnaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    if (op.getFun() != linalg::UnaryFn::ceil &&
        op.getFun() != linalg::UnaryFn::floor) {
      return failure();
    }

    auto inType = getElementTypeOrSelf(op.getInputs()[0].getType());
    auto outType = getElementTypeOrSelf(op.getOutputs()[0].getType());

    OpBuilder builder(op);
    Value src = op.getInputs()[0];
    if ((inType.isF16() || inType.isBF16() || inType.isSignlessInteger(16)) &&
        inType == outType) {
      // cast to fp32 to do ceil or floor, then cast back
      src = mtfusion::castTo(builder, src, rewriter.getF32Type(),
                            mtfusion::RoundMode::NORMAL);
    }
    mtfusion::RoundMode roundMode = op.getFun() == linalg::UnaryFn::ceil
                                       ? mtfusion::RoundMode::CEIL
                                       : mtfusion::RoundMode::FLOOR;
    auto castOp =
        mtfusion::castTo(builder, src, outType, roundMode, op.getOutputs()[0]);
    rewriter.replaceOp(op, castOp);
    return success();
  }
};

/// normalize integer Div by float div
/// c = a / b
///  is normalized to
/// fa = castTo<f32>(a)
/// fb = castTo<f32>(b)
/// fc = fa / fb
/// c = castTo<integer>(fc)
struct NormalizeDivInteger : public OpRewritePattern<linalg::ElemwiseBinaryOp> {
private:
  LogicalResult castOperand(OpBuilder &builder, Location loc, mlir::Value val,
                            mlir::Type elemTySrc, mlir::FloatType elemTyDst,
                            ArrayRef<int64_t> tensorShape,
                            mlir::Value &castVal) const {
    auto valTy = val.getType();
    mlir::Value tensorVal;
    if (isa<ShapedType>(valTy)) {
      tensorVal = val;
    } else {
      // Convert scalar to 0d tensor to adapt to mtfusion::CastOp
      auto valTensor = builder.create<tensor::EmptyOp>(loc, tensorShape, valTy);
      auto tensorFill =
          builder.create<linalg::FillOp>(loc, val, valTensor.getResult());
      tensorVal = tensorFill.getResultTensors()[0];
    }

    mtfusion::RoundMode rounding =
        mlir::utils::selectRoundMode<mtfusion::RoundMode>(elemTySrc, elemTyDst);
    castVal = mtfusion::castTo(builder, tensorVal, elemTyDst, rounding);

    return success();
  }

public:
  using OpRewritePattern<linalg::ElemwiseBinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }
    if (op.getFun() != linalg::BinaryFn::div) {
      return failure();
    }
    auto loc = op->getLoc();
    // linalg::ElemwiseBinaryOp's Outputs and Results must be
    // variadic of ranked tensor of any type values.
    // If the Outputs operand is a scalar, mlir crashes.
    // If the Results operand is a scalar, the verifier reports error.
    auto res = op.getResultTensors()[0];
    auto resTy = dyn_cast<TensorType>(res.getType());
    auto elemTySrc = getElementTypeOrSelf(resTy);
    if(!elemTySrc.isa<IntegerType>()) {
      return failure();
    }
    
    FloatType elemTyDst;
    if (elemTySrc.isInteger(32)) {
      elemTyDst = rewriter.getF32Type();
    } else if (elemTySrc.isInteger(16)) {
      elemTyDst = rewriter.getF16Type();
    } else {
      // Only [I32, I16] are supported
      return failure();
    }

    rewriter.setInsertionPoint(op);
    auto resShape = resTy.getShape();
    auto inputs = op.getDpsInputs();
    mlir::Value divFLhs;
    mlir::Value divFRhs;
    if (castOperand(rewriter, loc, inputs[0], elemTySrc, elemTyDst, resShape,
                    divFLhs)
            .failed()) {
      return failure();
    }
    if (castOperand(rewriter, loc, inputs[1], elemTySrc, elemTyDst, resShape,
                    divFRhs)
            .failed()) {
      return failure();
    }

    auto resElemTy = getElementTypeOrSelf(resTy);
    auto resFloatElemTy = getElementTypeOrSelf(divFLhs.getType());
    auto resFloatEmpty =
        rewriter.create<tensor::EmptyOp>(loc, resShape, resFloatElemTy);
    linalg::BinaryFn fun = linalg::BinaryFn::div;
    auto funAttr = rewriter.getAttr<linalg::BinaryFnAttr>(fun);
    auto divAttr = rewriter.getNamedAttr("fun", funAttr);
    auto divFOp = rewriter.create<linalg::ElemwiseBinaryOp>(
        loc, ValueRange{divFLhs, divFRhs}, ValueRange{resFloatEmpty},
        ArrayRef{divAttr});

    mtfusion::RoundMode rounding =
        mlir::utils::selectRoundMode<mtfusion::RoundMode>(resElemTy,
                                                         resFloatElemTy);
    auto castResOp =
        mtfusion::castTo(rewriter, divFOp->getResults()[0], resElemTy, rounding);

    rewriter.replaceOp(op, castResOp);
    return success();
  }
};

/// Returns whether the input value `v` is rec-like: Rec op or div op
/// with numerator of constant one. Set the denominator in place if true
static bool isRecLike(mlir::Value v, mlir::Value &denominator) {
  Operation *op = v.getDefiningOp();
  auto binOp = dyn_cast_or_null<linalg::ElemwiseBinaryOp>(op);
  if (!binOp) {
    return false;
  }
  if (binOp.getFun() != linalg::BinaryFn::div) {
    return false;
  }
  auto inputs = binOp.getDpsInputs();
  mlir::Value divLhs = inputs[0];
  mlir::Value divRhs = inputs[1];
  auto lhsConstOp = dyn_cast_or_null<arith::ConstantOp>(divLhs.getDefiningOp());
  if (!lhsConstOp) {
    return false;
  }

  denominator = divRhs;
  if (auto constFloatAttr = lhsConstOp.getValue().dyn_cast<FloatAttr>()) {
    llvm::APFloat floatOne(constFloatAttr.getValue().getSemantics(), 1);
    return constFloatAttr.getValue() == floatOne;
  } else if (auto constIntAttr =
                 lhsConstOp.getValue().dyn_cast<IntegerAttr>()) {
    return constIntAttr.getInt() == 1;
  }
  return false;
}

// replace `mulOp` with `newDivLhs/newDivRhs`
static void normalizeMulRecLikeByDiv(linalg::ElemwiseBinaryOp mulOp,
                                     Value newDivLhs, Value newDivRhs,
                                     PatternRewriter &rewriter) {

  assert(mulOp.getFun() == linalg::BinaryFn::mul &&
         "only support div-by-one used by mul bin op");

  auto mulRes = mulOp.getResultTensors()[0];
  auto mulResType = dyn_cast<TensorType>(mulRes.getType());
  auto newDivResult = rewriter.create<tensor::EmptyOp>(
      mulOp.getLoc(), mulResType.getShape(), getElementTypeOrSelf(mulResType));

  auto funAttr = rewriter.getAttr<linalg::BinaryFnAttr>(linalg::BinaryFn::div);
  auto divAttr = rewriter.getNamedAttr("fun", funAttr);

  auto newDivOp = rewriter.create<linalg::ElemwiseBinaryOp>(
      mulOp.getLoc(), ValueRange{newDivLhs, newDivRhs},
      ValueRange{newDivResult}, ArrayRef{divAttr});
  rewriter.replaceOp(mulOp, newDivOp);
}

/// normalize mul rec(div-by-one)
/// (1/b) * a -> a/b
/// a * (1/b) -> a/b
struct NormalizeMulRec : public OpRewritePattern<linalg::ElemwiseBinaryOp> {

public:
  using OpRewritePattern<linalg::ElemwiseBinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ElemwiseBinaryOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }
    if (op.getFun() != linalg::BinaryFn::mul) {
      return failure();
    }
    auto inputs = op.getDpsInputs();
    mlir::Value mulLhs = inputs[0];
    mlir::Value mulRhs = inputs[1];
    mlir::Value denominator;
    if (isRecLike(mulLhs, denominator)) {
      /// (1/b) * a -> a/b
      normalizeMulRecLikeByDiv(op, mulRhs, denominator, rewriter);
      return success();
    } else if (isRecLike(mulRhs, denominator)) {
      /// a * (1/b) -> a/b
      normalizeMulRecLikeByDiv(op, mulLhs, denominator, rewriter);
      return success();
    }
    return failure();
  }
};

static LogicalResult castInToF32ToOut(mtfusion::CastOp &op,
                                      PatternRewriter &rewriter) {
  auto srcTy = getElementTypeOrSelf(op.getDpsInputOperand(0)->get());
  auto dstTy = getElementTypeOrSelf(op.getDpsInitOperand(0)->get());
  auto castSrcToF32 =
      castTo(rewriter, op.getDpsInputOperand(0)->get(), rewriter.getF32Type(),
             mlir::utils::selectRoundMode<mtfusion::RoundMode>(
                 srcTy, rewriter.getF32Type()));
  auto castF32ToDst = castTo(rewriter, castSrcToF32, dstTy, op.getRoundMode());
  rewriter.replaceOp(op, castF32ToDst);
  return success();
}

// i8 -> f16 -> f32 -> i64
static LogicalResult castI8ToF16ToI64(mtfusion::CastOp &op,
                                      PatternRewriter &rewriter) {
  auto srcTy = getElementTypeOrSelf(op.getDpsInputOperand(0)->get());
  auto dstTy = getElementTypeOrSelf(op.getDpsInitOperand(0)->get());
  assert(srcTy.isInteger(8) && dstTy.isInteger(64) &&
         "this func only works for i8 to i64 cast");

  // i8->f16 only support Normal rounding mode
  Type f16Type = rewriter.getF16Type();
  auto roundI8ToF16 =
      mlir::utils::selectRoundMode<mtfusion::RoundMode>(srcTy, f16Type);
  Value dpsInput = op.getDpsInputOperand(0)->get();
  auto castSrcToF16 = castTo(rewriter, dpsInput, f16Type, roundI8ToF16);

  // f16->f32 only support Normal rounding mode
  Type f32Type = rewriter.getF32Type();
  auto roundF16ToF32 =
      mlir::utils::selectRoundMode<mtfusion::RoundMode>(f16Type, f32Type);
  auto castF16ToF32 = castTo(rewriter, castSrcToF16, f32Type, roundF16ToF32);

  // f32->i64
  auto roundF32ToI64 =
      mlir::utils::selectRoundMode<mtfusion::RoundMode>(f32Type, dstTy);
  auto castF32ToDst = castTo(rewriter, castF16ToF32, dstTy, roundF32ToI64);
  rewriter.replaceOp(op, castF32ToDst);
  return success();
}

struct NormalizeCastLoweringOp : public OpRewritePattern<mtfusion::CastOp> {
public:
  using OpRewritePattern<CastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mtfusion::CastOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) {
      return failure();
    }

    auto inType = getElementTypeOrSelf(op.getInputs()[0].getType());
    auto outType = getElementTypeOrSelf(op.getOutputs()[0].getType());
    const bool isI64ToF16OrBF16 =
        inType.isInteger(64) && (outType.isF16() || outType.isBF16());

    if (isI64ToF16OrBF16) {
      LLVM_DEBUG(llvm::dbgs()
                 << "match compound cast pattern from " << inType << " to "
                 << outType << ", and rewrite to cast (from " << inType
                 << " to f16) "
                 << "\n ");
      return castInToF32ToOut(op, rewriter);
    }

    const bool isI8ToI64 = inType.isInteger(8) && outType.isInteger(64);
    if (isI8ToI64) {
      LLVM_DEBUG(llvm::dbgs()
                 << "match compound cast pattern from " << inType << " to "
                 << outType << ", and rewrite to cast (from " << inType
                 << " to f16 to f32 to " << outType << ")\n");
      return castI8ToF16ToI64(op, rewriter);
    }
    return failure();
  }
};

/// normalize operand order in [tensorOper, scalarOper, outputOper]
template <typename SrcOp>
struct NormalizeBinaryOp : public OpRewritePattern<SrcOp> {
  using OpRewritePattern<SrcOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const override {
    if (!isCommutativeOp(op)) {
      return failure();
    }

    if (!isSVOp(op) || !isHWSupportVSOp(op)) {
      return failure();
    }

    llvm::SmallVector<Value> inputs = op.getDpsInputs();
    llvm::SmallVector<Value> outputs = op.getOutputs();
    SmallVector<Value> newOperands{inputs[1], inputs[0], outputs[0]};
    op->setOperands(newOperands);
    return success();
  }
};

/// normalize inf op
/// eg fp 16
/// ignore mask sign bit
/// 1. vdup(7FFF)
/// 2. vand(input, input, vdup)
/// compare with inf(include negeative and positive inf)
/// 3.vadd(input, input, 0xFC00). FC00 is negative inf
/// 4.vabs(input, input)
/// If is_inf, the result after vabs is 0, otherwise is greater than 1.
/// 5.vmin(input, input, 1)
/// If is_inf, the result after vmin is 0, otherwise is 1.
/// 6.vmuls(input, input, -1)
/// If is_inf, the result after vmin is 0, otherwise is -1.
/// 7.vadds(input, input, 1)
/// If is_inf, the result after vmin is 1, otherwise is 0.
/// 8.cast(input, fp->i1)
inline bool isInf(Value input) {
  auto fillop = input.getDefiningOp<linalg::FillOp>();
  if (fillop == nullptr) {
    return false;
  }
  auto cstOp = fillop.getInputs()[0].getDefiningOp<arith::ConstantOp>();
  assert(cstOp != nullptr);
  llvm::APFloat cstValue = dyn_cast<FloatAttr>(cstOp.getValueAttr()).getValue();
  return cstValue.isNegInfinity();
}

/// Convert dense tensor/memref with only 1 element to scalar.
static std::optional<Value>
singleElemDenseTensorToScalar(Value operand, PatternRewriter &rewriter) {
  auto constantOp = operand.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    return std::nullopt;

  auto shapedType = dyn_cast<ShapedType>(constantOp.getType());
  if (!shapedType)
    return std::nullopt;

  auto shape = shapedType.getShape();
  if (shape.size() > 1 || (!shape.empty() && shape[0] > 1))
    return std::nullopt;

  auto denseAttr = constantOp.getValue().dyn_cast<DenseIntOrFPElementsAttr>();
  if (!denseAttr) {
    return std::nullopt;
  }

  auto elemType = denseAttr.getElementType();
  if (!elemType.isIntOrIndexOrFloat()) {
    return std::nullopt;
  }

  TypedAttr typedAttr =
      elemType.isIntOrIndex()
          ? (TypedAttr)*denseAttr.getValues<IntegerAttr>().begin()
          : (TypedAttr)*denseAttr.getValues<FloatAttr>().begin();

  return rewriter.create<arith::ConstantOp>(operand.getLoc(), elemType,
                                            typedAttr);
}

template <typename OpType>
struct NormalizeScalarLikeTensorOp : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    bool isConverted = false;
    SmallVector<Value> inputsNew;
    for (auto inp : op.getInputs()) {
      auto inpNew = singleElemDenseTensorToScalar(inp, rewriter);
      if (inpNew.has_value()) {
        inputsNew.push_back(*inpNew);
        isConverted = true;
      } else {
        inputsNew.push_back(inp);
      }
    }

    SmallVector<Value> outputsNew;
    for (auto out : op.getOutputs()) {
      auto outNew = singleElemDenseTensorToScalar(out, rewriter);
      if (outNew.has_value()) {
        outputsNew.push_back(*outNew);
        isConverted = true;
      } else {
        outputsNew.push_back(out);
      }
    }

    if (!isConverted)
      return failure();

    IRMapping mapper;
    mapper.map(op.getInputs(), ValueRange(inputsNew));
    mapper.map(op.getOutputs(), ValueRange(outputsNew));

    Operation *clonedOp = rewriter.clone(*op, mapper);
    rewriter.replaceOp(op, clonedOp);
    return success();
  }
};

/// Convert linalg.broadcast to linalg.fill if input operand only has one elem.
struct NormalizeScalarLikeTensorLinalgBrcOp
    : public OpRewritePattern<linalg::BroadcastOp> {
public:
  using OpRewritePattern<linalg::BroadcastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    auto optInpNew = singleElemDenseTensorToScalar(op.getInput(), rewriter);
    if (!optInpNew.has_value())
      return failure();

    auto fillOp = rewriter.create<linalg::FillOp>(
        op->getLoc(), ValueRange(*optInpNew), op.getInit());
    rewriter.replaceOp(op, fillOp);
    return success();
  }
};

// Normalize scalar like tensor for linalg and mtfusion ops.
void populateNormalizeScalarLikeMtFusionPatterns(RewritePatternSet &patterns) {
  patterns.add<NormalizeScalarLikeTensorOp<mtfusion::CastOp>>(
      patterns.getContext());
  patterns.add<NormalizeScalarLikeTensorOp<linalg::ElemwiseUnaryOp>>(
      patterns.getContext());
  patterns.add<NormalizeScalarLikeTensorOp<linalg::ElemwiseBinaryOp>>(
      patterns.getContext());
  patterns.add<NormalizeScalarLikeTensorLinalgBrcOp>(patterns.getContext());
}

void populateNormalizeMtFusionPatterns(RewritePatternSet &patterns) {
  patterns.add<NormalizeMulRec>(patterns.getContext());
  patterns.add<NormalizeNegToMul>(patterns.getContext());
  patterns.add<NormalizeVDivSVToVV>(patterns.getContext());
  patterns.add<NormalizeCeilandFloorOp>(patterns.getContext());
  patterns.add<NormalizeDivInteger>(patterns.getContext());
  patterns.add<NormalizeCastLoweringOp>(patterns.getContext());
  patterns.add<NormalizeBinaryOp<linalg::ElemwiseBinaryOp>>(
      patterns.getContext());
  populateNormalizeScalarLikeMtFusionPatterns(patterns);
}

namespace {
struct NormalizeMtFusionPass : public impl::NormalizeBase<NormalizeMtFusionPass> {
public:
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    populateNormalizeMtFusionPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::mtfusion::createMtFusionNormalizeOpsPass() {
  return std::make_unique<NormalizeMtFusionPass>();
}