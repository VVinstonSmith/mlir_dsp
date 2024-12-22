//===- InlineBrc.cpp - Inline Broadcast-like Ops For MtFusion Ops ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"
#include "mtir/Dialect/MtFusion/Utils/Utils.h"
#include "mtir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mtfusion-inline-brc"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_MTFUSIONINLINEBRC
#include "mtir/Dialect/MtFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::mtfusion;

// TODO : add platform information
// check whether brc op can be inlined to current brc user op
static bool canInlineBrc(Operation *useOp, OpOperand *oper,
                         bool isScalar = false) {
  if (!isHWSupportVSOp(useOp)) {
    return false;
  }
  // only support inline brc to elemwise binary op currently
  if (!isa<linalg::ElemwiseBinaryOp>(useOp)) {
    return false;
  }

  Value lhs = useOp->getOperand(0);
  Value rhs = useOp->getOperand(1);
  // already vector-scalar or scalar-vector pattern, no need to inline
  if (utils::isScalarLike(rhs) || utils::isScalarLike(lhs)) {
    return false;
  }

  bool isSameAsInit = false;
  auto dstStyleOp = cast<DestinationStyleOpInterface>(useOp);
  for (auto initOper : dstStyleOp.getDpsInits()) {
    if (initOper == oper->get()) {
      isSameAsInit = true;
    }
  }

  if (isSameAsInit && isScalar) {
    // init operand cannot be scalar
    return false;
  }
  // should not inline scalar when lhs and rhs are the same vector
  // TODO: add `OptSinglePoint` pass in mtfusion to allow inline when `lhs==rhs`
  return lhs != rhs;
}

// extract the scalar value to replace origin brc/fill op
static std::optional<Value> getScalarValue(PatternRewriter &rewriter,
                                           Location loc, Value value) {
  Type type = value.getType();
  Value scalarValue = nullptr;
  SmallVector<Value> indices;

  if (type.isIntOrIndexOrFloat()) {
    scalarValue = value;
  } else {
    std::optional<size_t> rankMaybe = utils::getShapeRank(type);
    if (!rankMaybe.has_value()) {
      return std::nullopt;
    }
    size_t rank = rankMaybe.value();
    for (size_t i = 0; i < rank; ++i) {
      indices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }
    scalarValue = rewriter.create<tensor::ExtractOp>(loc, value, indices);
  }
  assert(scalarValue && "scalar value must not be nullptr");
  return scalarValue;
}

static void findUsersToInline(Value src, DenseSet<OpOperand *> &usesToInline,
                              bool isInlineScalar, int level = 0) {
  assert(level < 100 && "findUsersToInline in infinite recursion");
  assert(src && "src must not be nullptr");
  for (OpOperand &use : src.getUses()) {
    Operation *user = use.getOwner();
    if (canInlineBrc(user, &use, isInlineScalar)) {
      usesToInline.insert(&use);
      continue;
    }
    if (auto mtfusionCast = dyn_cast<mtfusion::CastOp>(user)) {
      Value castResult = mtfusionCast.getResult(0);
      findUsersToInline(castResult, usesToInline, isInlineScalar, level + 1);
    } else if (auto tensorCast = dyn_cast<tensor::CastOp>(user)) {
      Value castResult = tensorCast.getResult();
      findUsersToInline(castResult, usesToInline, isInlineScalar, level + 1);
    }
  }
}

// replace brc result in current user op with brc scalar input
static LogicalResult replaceBrcWithInput(Operation *brcOp, Value brcResult,
                                         Value input, Location loc,
                                         PatternRewriter &rewriter) {
  if (!utils::isScalarLike(input)) {
    return rewriter.notifyMatchFailure(brcOp, "input is not scalar like.");
  }

  DenseSet<OpOperand *> usesToInline;
  bool isScalar = input.getType().isIntOrFloat();
  findUsersToInline(brcResult, usesToInline, isScalar);
  if (usesToInline.empty()) {
    return rewriter.notifyMatchFailure(brcOp, "cannot find users to inline.");
  }

  auto scalarMaybe = getScalarValue(rewriter, loc, input);
  if (!scalarMaybe.has_value()) {
    return rewriter.notifyMatchFailure(brcOp, "failed to get scalar value.");
  }

  Value scalar = scalarMaybe.value();
  Type srcElemType = getElementTypeOrSelf(scalar.getType());

  for (OpOperand *use : usesToInline) {
    Type dstElemType = getElementTypeOrSelf(use->get().getType());
    Value replacement = scalar;
    if (srcElemType != dstElemType) {
      mtfusion::RoundMode roundMode =
          mlir::utils::selectRoundMode<mtfusion::RoundMode>(srcElemType,
                                                           dstElemType);
      replacement = mtfusion::castTo(rewriter, scalar, dstElemType, roundMode);
    }
    rewriter.modifyOpInPlace(use->getOwner(), [&]() { use->set(replacement); });
  }
  return success();
}

struct InlineBroadcastOpWithScalarInput
    : public OpRewritePattern<linalg::BroadcastOp> {
public:
  using OpRewritePattern<linalg::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BroadcastOp brcOp,
                                PatternRewriter &rewriter) const override {
    LDBG("Got BroadcastOp: " << brcOp);
    Value input = brcOp.getInput();
    return replaceBrcWithInput(brcOp, brcOp->getResult(0), input,
                               brcOp->getLoc(), rewriter);
  }
};

struct InlineFillOpWithScalarInput : public OpRewritePattern<linalg::FillOp> {
public:
  using OpRewritePattern<linalg::FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::FillOp fillOp,
                                PatternRewriter &rewriter) const override {
    LDBG("Got FillOp: " << fillOp);
    if (fillOp->getNumResults() != 1) {
      return failure();
    }
    ValueRange inputs = fillOp.getInputs();
    if (inputs.size() != 1) {
      return failure();
    }

    Value input = inputs.front();
    return replaceBrcWithInput(fillOp, fillOp->getResult(0), input,
                               fillOp->getLoc(), rewriter);
  }
};

namespace {
struct MtFusionInlineBrcPass
    : public impl::MtFusionInlineBrcBase<MtFusionInlineBrcPass> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<InlineBroadcastOpWithScalarInput>(patterns.getContext());
    patterns.add<InlineFillOpWithScalarInput>(patterns.getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // anonymous namespace

std::unique_ptr<Pass> mlir::mtfusion::createMtFusionInlineBrcPass() {
  return std::make_unique<MtFusionInlineBrcPass>();
}