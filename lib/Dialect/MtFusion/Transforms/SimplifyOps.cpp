//===- SimplifyOps.cpp ------- Simplify operations ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"

#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_SIMPLIFYOPS
#include "mtir/Dialect/MtFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::mtfusion;

namespace {
struct SimplifyOpsPass : public impl::SimplifyOpsBase<SimplifyOpsPass> {
public:
  void runOnOperation() final;
};

struct CastOpPattern : public OpRewritePattern<CastOp> {
public:
  using OpRewritePattern<CastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CastOp castOp,
                                PatternRewriter &rewriter) const final {
    if (isOpTriviallyDead(castOp)) {
      rewriter.eraseOp(castOp);
      return success();
    }

    // Helper function that return the cast op that
    // defines all inputs of the given op (in the same order). Return "nullptr"
    // if there is no such op.
    auto getInputCast = [](CastOp castOp) -> CastOp {
      auto inputCastOp = castOp.getInputs().front().getDefiningOp<CastOp>();
      if (!inputCastOp)
        return {};
      if (inputCastOp.getResults() != castOp.getInputs())
        return {};
      return inputCastOp;
    };

    // Process ops bottom-to-top.

    // Traverse the chain of input cast ops to see if an op with the same
    // input types can be found.
    CastOp nextCast = castOp;
    while (nextCast) {
      if (nextCast.getInputs().getTypes() == castOp.getResultTypes()) {
        // Found a cast where the input types match the output types of the
        // matched op. We can directly use those inputs and the matched op can
        // be removed.
        rewriter.replaceOp(castOp, nextCast.getInputs());
        return success();
      }
      nextCast = getInputCast(nextCast);
    }

    return failure();
  }
};

struct TransposeOpPattern : public OpRewritePattern<linalg::TransposeOp> {
public:
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::TransposeOp transOp,
                                PatternRewriter &rewriter) const final {
    if (isOpTriviallyDead(transOp)) {
      rewriter.eraseOp(transOp);
      return success();
    }

    // Initialize baseline as golden for simulation result comparison
    SmallVector<int64_t> base(transOp.getPermutation().size());
    for (size_t i = 0; i < base.size(); ++i)
      base[i] = i;

    // Helper function simulating permutations on input data
    auto simulate = [&](const ArrayRef<int64_t> input,
                        const ArrayRef<int64_t> permutation) {
      assert(input.size() == permutation.size());
      SmallVector<int64_t> res(permutation.size());
      for (const auto [i, v] : llvm::enumerate(permutation))
        res[v] = input[i];
      return res;
    };

    SmallVector<int64_t> sim(base);
    // Bottom-up searching the chain of transpose ops
    linalg::TransposeOp nextTransOp = transOp;
    while (nextTransOp) {
      sim = simulate(sim, nextTransOp.getPermutation());
      if (sim == base) {
        rewriter.replaceOp(transOp, {nextTransOp.getInput()});
        return success();
      }
      nextTransOp = nextTransOp.getInput().getDefiningOp<linalg::TransposeOp>();
    }

    return failure();
  }
};

bool isConstOne(Value v) {
  auto type = getElementTypeOrSelf(v);
  if (isa<FloatType>(type)) {
    if (matchPattern(v, m_OneFloat())) {
      return true;
    }
  } else if (type.isIntOrIndex()) {
    if (matchPattern(v, m_One())) {
      return true;
    }
  }

  auto defineOp = v.getDefiningOp();
  if (!defineOp) {
    return false;
  }

  auto resIndx = cast<OpResult>(v).getResultNumber();
  if (auto fillOp = dyn_cast<linalg::FillOp>(defineOp)) {
    return isConstOne(fillOp.getOperand(resIndx));
  } else if (auto castOp = dyn_cast<mtfusion::CastOp>(defineOp)) {
    return isConstOne(castOp.getOperand(resIndx));
  }

  return false;
}

bool isConstZero(Value v) {
  auto type = getElementTypeOrSelf(v);
  if (isa<FloatType>(type)) {
    if (matchPattern(v, m_PosZeroFloat()) ||
        matchPattern(v, m_NegZeroFloat())) {
      return true;
    }
  } else if (type.isIntOrIndex()) {
    if (matchPattern(v, m_Zero())) {
      return true;
    }
  }

  auto defineOp = v.getDefiningOp();
  if (!defineOp) {
    return false;
  }

  auto resIndx = cast<OpResult>(v).getResultNumber();
  if (auto fillOp = dyn_cast<linalg::FillOp>(defineOp)) {
    return isConstZero(fillOp.getOperand(resIndx));
  } else if (auto castOp = dyn_cast<mtfusion::CastOp>(defineOp)) {
    return isConstZero(castOp.getOperand(resIndx));
  }

  return false;
}

template <typename AddOP>
LogicalResult simplifyAdd(PatternRewriter &rewriter, AddOP addOp) {
  if (isConstZero(addOp.getOperand(0))) {
    rewriter.replaceOp(addOp, addOp.getOperand(1));
    return success();
  }
  if (isConstZero(addOp.getOperand(1))) {
    rewriter.replaceOp(addOp, addOp.getOperand(0));
    return success();
  }
  return failure();
}

template <typename SubOP>
LogicalResult simplifySub(PatternRewriter &rewriter, SubOP subOp) {
  if (isConstZero(subOp.getOperand(1))) {
    rewriter.replaceOp(subOp, subOp.getOperand(0));
    return success();
  }
  return failure();
}

template <typename DivOP>
LogicalResult simplifyDiv(PatternRewriter &rewriter, DivOP divOp) {
  if (isConstOne(divOp.getOperand(1))) {
    rewriter.replaceOp(divOp, divOp.getOperand(0));
    return success();
  }
  return failure();
}

template <typename MulOP>
LogicalResult simplifyMul(PatternRewriter &rewriter, MulOP mulOp) {
  if (isConstOne(mulOp.getOperand(0))) {
    rewriter.replaceOp(mulOp, mulOp.getOperand(1));
    return success();
  }

  if (isConstOne(mulOp.getOperand(1))) {
    rewriter.replaceOp(mulOp, mulOp.getOperand(0));
    return success();
  }
  return failure();
}

template <typename BINOP>
struct ElemBinaryPattern : public OpRewritePattern<BINOP> {
public:
  using OpRewritePattern<BINOP>::OpRewritePattern;
  LogicalResult matchAndRewrite(BINOP binaryOp,
                                PatternRewriter &rewriter) const final {
    auto binaryFunc = binaryOp.getFun();
    if constexpr (std::is_same_v<BINOP, linalg::ElemwiseBinaryOp>) {
      if (binaryFunc == linalg::BinaryFn::add) {
        return simplifyAdd<BINOP>(rewriter, binaryOp);
      }

      if (binaryFunc == linalg::BinaryFn::mul) {
        return simplifyMul<BINOP>(rewriter, binaryOp);
      }

      if (binaryFunc == linalg::BinaryFn::sub) {
        return simplifySub<BINOP>(rewriter, binaryOp);
      }

      if (binaryFunc == linalg::BinaryFn::div) {
        return simplifyDiv<BINOP>(rewriter, binaryOp);
      }
    }

    return failure();
  }
};

void populateSimplifyOpsPattern(RewritePatternSet &patterns) {
  patterns.add<CastOpPattern>(patterns.getContext());
  patterns.add<TransposeOpPattern>(patterns.getContext());
  patterns.add<ElemBinaryPattern<linalg::ElemwiseBinaryOp>>(
      patterns.getContext());
}

void SimplifyOpsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateSimplifyOpsPattern(patterns);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // anonymous namespace

std::unique_ptr<Pass> mlir::mtfusion::createSimplifyOpsPass() {
  return std::make_unique<SimplifyOpsPass>();
}