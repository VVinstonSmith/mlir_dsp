//===- MtFusionTransformOps.cpp - Implementation of MtFusion transform ops --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/TransformOps/MtFusionTransformOps.h"
// #include "mtir/Dialect/Annotation/IR/Annotation.h"
#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/TransformOps/Syntax.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mtfusion-transform-op"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::transform;
using namespace mlir::mtfusion;

namespace {
static constexpr llvm::StringLiteral kBufferSizeInByteAttr =
    "buffer_size_in_byte";
} // namespace

//===----------------------------------------------------------------------===//
// GetFuncArgumentOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
GetFuncArgumentOp::apply(TransformRewriter &rewriter,
                         TransformResults &transformResults,
                         TransformState &state) {
  auto payloadOps = state.getPayloadOps(getTarget());
  if (!llvm::hasSingleElement(payloadOps))
    return emitDefiniteFailure() << "requires exactly one target handle!";

  auto func = dyn_cast_or_null<func::FuncOp>(*payloadOps.begin());
  if (!func)
    return emitDefiniteFailure()
           << "target handle does not point to `func.func` op";

  Region::BlockArgListType funcArgs = func.getArguments();
  SmallVector<int64_t> operandPositions;
  DiagnosedSilenceableFailure diag = expandTargetSpecification(
      getLoc(), getIsAll(), getIsInverted(), getRawPositionList(),
      func.getNumArguments(), operandPositions);
  if (diag.isSilenceableFailure()) {
    diag.attachNote(func->getLoc())
        << "while considering positions of this payload operation";
    return diag;
  }
  SmallVector<Value> selectedArgs = llvm::map_to_vector(
      operandPositions, [&](int64_t pos) { return Value(funcArgs[pos]); });
  if (getFindReshapeConsumer()) {
    for (auto [idx, v] : llvm::enumerate(selectedArgs)) {
      auto maybeResult = mtfusion::traceReshapeOrSliceSingleConsumer(v);
      if (failed(maybeResult))
        return emitDefiniteFailure()
               << "cannot trace to single reshape consumer for " << v;
      v = maybeResult.value();
    }
  }
  transformResults.setValues(llvm::cast<OpResult>(getOutputs()), selectedArgs);
  return DiagnosedSilenceableFailure::success();
}

void GetFuncArgumentOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTarget(), effects);
  producesHandle(getOutputs(), effects);
  // onlyReadsHandle(getTargetMutable(), effects);
  // producesHandle(getOperation()->getOpResults(), effects);
}

//===----------------------------------------------------------------------===//
// GetFuncResultOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
GetFuncResultOp::apply(TransformRewriter &rewriter,
                       TransformResults &transformResults,
                       TransformState &state) {
  auto payloadOps = state.getPayloadOps(getTarget());
  if (!llvm::hasSingleElement(payloadOps))
    return emitDefiniteFailure() << "requires exactly one target handle!";

  auto func = dyn_cast_or_null<func::FuncOp>(*payloadOps.begin());
  if (!func)
    return emitDefiniteFailure()
           << "target handle does not point to `func.func` op";

  func::ReturnOp returnOp = nullptr;
  func->walk([&returnOp](func::ReturnOp op) { returnOp = op; });
  if (!returnOp)
    return emitDefiniteFailure() << "cannot find return op in func!";

  SmallVector<int64_t> operandPositions;
  DiagnosedSilenceableFailure diag = expandTargetSpecification(
      getLoc(), getIsAll(), getIsInverted(), getRawPositionList(),
      func.getNumResults(), operandPositions);
  if (diag.isSilenceableFailure()) {
    diag.attachNote(func->getLoc())
        << "while considering positions of this payload operation";
    return diag;
  }
  SmallVector<Value> selectedResult =
      llvm::map_to_vector(operandPositions, [&](int64_t pos) {
        return returnOp->getOpOperand(pos).get();
      });
  if (getFindReshapeProducer()) {
    for (auto [idx, v] : llvm::enumerate(selectedResult)) {
      auto maybeResult = mtfusion::traceReshapeOrSliceSingleProducer(v);
      if (failed(maybeResult))
        return emitDefiniteFailure()
               << "cannot trace to single reshape producer for " << v;
      v = maybeResult.value();
    }
  }
  transformResults.setValues(llvm::cast<OpResult>(getOutputs()),
                             selectedResult);
  return DiagnosedSilenceableFailure::success();
}

void GetFuncResultOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTarget(), effects);
  producesHandle(getOutputs(), effects);
  // onlyReadsHandle(getTargetMutable(), effects);
  // producesHandle(getOperation()->getOpResults(), effects);
}

//===----------------------------------------------------------------------===//
// CacheReadOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
CacheReadOp::apply(TransformRewriter &rewriter,
                   TransformResults &transformResults, TransformState &state) {
  SmallVector<Operation *> cachedOps;
  for (Value target : state.getPayloadValues(getTargets())) {
    // skip values that does not have tensor types
    if (!isa<TensorType>(target.getType())) {
      continue;
    }
    linalg::CopyOp cachedOp;
    if (auto opResult = dyn_cast_or_null<OpResult>(target)) {
      auto definingOp = opResult.getOwner();
      rewriter.setInsertionPointAfter(definingOp);
      cachedOp = createCacheRead(rewriter, opResult, definingOp->getLoc());
    } else if (auto blockArgument = dyn_cast_or_null<BlockArgument>(target)) {
      auto insertPoint = &(blockArgument.getParentBlock()->front());
      rewriter.setInsertionPoint(insertPoint);
      cachedOp =
          createCacheRead(rewriter, blockArgument, insertPoint->getLoc());
    } else {
      llvm_unreachable("unsupported type");
    }
    cachedOps.push_back(cachedOp.getOperation());
  }
  transformResults.set(llvm::cast<OpResult>(getCached()), cachedOps);
  return DiagnosedSilenceableFailure::success();
}

void CacheReadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargets(), effects);
  producesHandle(getCached(), effects);
  // onlyReadsHandle(getTargetsMutable(), effects);
  // producesHandle(getOperation()->getOpResults(), effects);
}

//===----------------------------------------------------------------------===//
// CacheWriteOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
CacheWriteOp::apply(TransformRewriter &rewriter,
                    TransformResults &transformResults, TransformState &state) {
  SmallVector<Operation *> cachedOps;
  for (Value target : state.getPayloadValues(getTargets())) {
    // skip values that does not have tensor types
    if (!isa<TensorType>(target.getType())) {
      continue;
    }
    FailureOr<linalg::CopyOp> maybeCachedOp;
    if (auto opResult = dyn_cast_or_null<OpResult>(target)) {
      maybeCachedOp = createCacheWrite(rewriter, opResult, getOutputOnly(),
                                       getCacheWriteToOutputInit());
    } else {
      llvm_unreachable("unsupported type");
    }
    if (failed(maybeCachedOp))
      return DiagnosedSilenceableFailure::definiteFailure();
    cachedOps.push_back((*maybeCachedOp).getOperation());
  }
  transformResults.set(llvm::cast<OpResult>(getCached()), cachedOps);
  return DiagnosedSilenceableFailure::success();
}

void CacheWriteOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargets(), effects);
  producesHandle(getCached(), effects);
  // onlyReadsHandle(getTargetsMutable(), effects);
  // producesHandle(getOperation()->getOpResults(), effects);
}

//===----------------------------------------------------------------------===//
// CacheReadAndWriteOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
CacheReadAndWriteOp::apply(TransformRewriter &rewriter,
                           TransformResults &transformResults, 
                           TransformState &state) {
  SmallVector<Operation *> copyReadOps, copyWriteOps;
  for(Value target : state.getPayloadValues(getTargets())) {
    // skip values that does not have tensor types
    if (!isa<TensorType>(target.getType())) {
      continue;
    }
    if (auto opResult = dyn_cast_or_null<OpResult>(target)) {
      auto [copyReadOp, copyWriteOp] = createCacheReadAndWrite(rewriter, opResult);
      copyReadOps.push_back(copyReadOp.getOperation());
      copyWriteOps.push_back(copyWriteOp.getOperation());
    } else {
      llvm_unreachable("unsupported type");
    }
  }
  transformResults.set(llvm::cast<OpResult>(getCopyReadOp()), copyReadOps);
  transformResults.set(llvm::cast<OpResult>(getCopyWriteOp()), copyWriteOps);
  return DiagnosedSilenceableFailure::success();
}

void CacheReadAndWriteOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargets(), effects);
  producesHandle(getCopyReadOp(), effects);
  producesHandle(getCopyWriteOp(), effects);
}

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure ReverseOp::apply(TransformRewriter &rewriter,
                                             TransformResults &transformResults,
                                             TransformState &state) {
  SmallVector<Operation *> targets =
      llvm::to_vector(state.getPayloadOps(getTarget()));
  SmallVector<Operation *> reversedOperations = {targets.rbegin(),
                                                 targets.rend()};
  transformResults.set(cast<OpResult>(getResult()), reversedOperations);
  return DiagnosedSilenceableFailure::success();
}

void ReverseOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTarget(), effects);
  producesHandle(getResult(), effects);
  // onlyReadsHandle(getTargetMutable(), effects);
  // producesHandle(getOperation()->getOpResults(), effects);
}

//===----------------------------------------------------------------------===//
// ExtendedFuseIntoContainingOp
//===----------------------------------------------------------------------===//

void transform::ExtendedFuseIntoContainingOp::build(OpBuilder &builder,
                                                    OperationState &result,
                                                    Value producerOp,
                                                    Value containingOp) {
  result.addOperands({producerOp, containingOp});
  auto resultType = transform::AnyOpType::get(builder.getContext());
  result.addTypes({resultType, resultType});
}

bool transform::ExtendedFuseIntoContainingOp::allowsRepeatedHandleOperands() {
  // Allow repeated handles since we are fusing everything anyway.
  return true;
}

static SmallVector<Value> recursiveClone(RewriterBase &rewriter,
                                         SmallVector<Value> values,
                                         Operation *clonePoint) {
  SmallVector<Value> newValues;
  for (auto value : values) {
    // If target value is a block argument, we can use it anywhere we want.
    if (value.isa<BlockArgument>()) {
      newValues.push_back(value);
      continue;
    }
    // If target value is a result of a value defined before the target
    // cloning point, we need to recursively clone its operands.
    auto *defOperation = value.getDefiningOp();
    if (clonePoint->getBlock() == defOperation->getBlock() &&
        clonePoint->isBeforeInBlock(defOperation)) {
      auto operands = defOperation->getOperands();
      auto clonedValues = recursiveClone(rewriter, operands, clonePoint);

      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(clonePoint);

      IRMapping mapping;
      mapping.map(operands, clonedValues);
      auto *clonedOp = rewriter.clone(*defOperation, mapping);

      newValues.push_back(
          clonedOp->getResult(cast<OpResult>(value).getResultNumber()));
    } else {
      newValues.push_back(value);
    }
  }

  return newValues;
}

static bool isValidSliceOpInContainingOp(tensor::ExtractSliceOp sliceOp,
                                         Operation *containingOp) {
  if (!sliceOp || !containingOp->isProperAncestor(sliceOp)) {
    return false;
  }

  auto staticStrides = sliceOp.getStaticStrides();
  if (llvm::count_if(staticStrides, [](int64_t s) { return s != 1; }) > 0) {
    // only union extract slice with stride 1
    return false;
  }

  return true;
}

static void getFirstSliceUserInContainingOp(
    Operation *producerOp, Operation *containingOp,
    llvm::DenseMap<Value, tensor::ExtractSliceOp> *result2FirstSliceOp,
    llvm::DenseMap<Value, int> *result2ValidNum) {
  for (auto res : producerOp->getResults()) {
    tensor::ExtractSliceOp firstSliceOp;
    int validNum = 0;
    for (auto user : res.getUsers()) {
      auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
      if (!isValidSliceOpInContainingOp(sliceOp, containingOp)) {
        continue;
      }

      if (!firstSliceOp || sliceOp->isBeforeInBlock(firstSliceOp)) {
        firstSliceOp = sliceOp;
      }

      validNum++;
    }
    result2ValidNum->insert(std::pair(res, validNum));
    if (firstSliceOp) {
      assert(validNum > 0);
      result2FirstSliceOp->insert(std::pair(res, firstSliceOp));
    }
  }
}

enum class MODE {
  UNION_MAX,
  UNION_MIN,
  COMPUTE_SLICE_MAX,
  COMPUTE_SUB,
  COMPUTE_DISTANCE
};

static SmallVector<Value> compute(RewriterBase &rewriter, MODE mode,
                                  SmallVectorImpl<Value> &lhs,
                                  SmallVectorImpl<Value> &rhs, Location loc) {
  auto symA = rewriter.getAffineSymbolExpr(0);
  auto symB = rewriter.getAffineSymbolExpr(1);
  auto one = rewriter.getAffineConstantExpr(1);
  AffineMap map;
  if (mode == MODE::UNION_MAX || mode == MODE::UNION_MIN)
    map = AffineMap::get(0, 2, {symA, symB}, rewriter.getContext());
  else if (mode == MODE::COMPUTE_SLICE_MAX)
    map = AffineMap::get(0, 2, {symA + symB - one}, rewriter.getContext());
  else if (mode == MODE::COMPUTE_SUB)
    map = AffineMap::get(0, 2, {symA - symB}, rewriter.getContext());
  else {
    assert(mode == MODE::COMPUTE_DISTANCE);
    map = AffineMap::get(0, 2, {symA - symB + one}, rewriter.getContext());
  }

  SmallVector<Value> results;
  for (auto it : llvm::zip(lhs, rhs)) {
    auto l = std::get<0>(it);
    auto r = std::get<1>(it);
    Value result;
    switch (mode) {
    case MODE::UNION_MAX:
      result = rewriter.create<affine::AffineMaxOp>(loc, map, ValueRange{l, r});
      break;
    case MODE::UNION_MIN:
      result = rewriter.create<affine::AffineMinOp>(loc, map, ValueRange{l, r});
      break;
    case MODE::COMPUTE_SLICE_MAX:
      result =
          rewriter.create<affine::AffineApplyOp>(loc, map, ValueRange{l, r});
      break;
    case MODE::COMPUTE_SUB:
    case MODE::COMPUTE_DISTANCE:
      result =
          rewriter.create<affine::AffineApplyOp>(loc, map, ValueRange{l, r});
      break;
    }
    results.push_back(result);
  }
  return results;
}

SmallVector<OpFoldResult> convert(SmallVectorImpl<Value> &values) {
  SmallVector<OpFoldResult> results;
  for (auto it : values) {
    results.push_back(OpFoldResult(it));
  }
  return results;
}

static void unionProducerUsers(RewriterBase &rewriter, Diagnostic &diag,
                               Operation *producerOp, Operation *containingOp) {

  llvm::DenseMap<Value, tensor::ExtractSliceOp> result2FirstSliceOp;
  llvm::DenseMap<Value, int> result2ValidNum;
  getFirstSliceUserInContainingOp(producerOp, containingOp,
                                  &result2FirstSliceOp, &result2ValidNum);

  for (auto produceResult : producerOp->getResults()) {
    int validSliceOpNum = result2ValidNum[produceResult];
    LDBG("produce res : " << produceResult
                          << ", slice op number : " << validSliceOpNum);
    if (validSliceOpNum < 2) {
      continue;
    }
    assert(result2FirstSliceOp.find(produceResult) !=
           result2FirstSliceOp.end());
    auto firstSliceOp = result2FirstSliceOp[produceResult];
    LDBG("begin to union \n" << *containingOp);
    LDBG("first SliceOp \n" << firstSliceOp);

    rewriter.setInsertionPoint(firstSliceOp);
    auto unionOffsets = getValueOrCreateConstantIndexOp(
        rewriter, firstSliceOp.getLoc(), firstSliceOp.getMixedOffsets());
    auto sizes = getValueOrCreateConstantIndexOp(
        rewriter, firstSliceOp.getLoc(), firstSliceOp.getMixedSizes());
    auto unionMaxes = compute(rewriter, MODE::COMPUTE_SLICE_MAX, unionOffsets,
                              sizes, firstSliceOp->getLoc());

    for (auto *user : produceResult.getUsers()) {
      auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
      if (!isValidSliceOpInContainingOp(sliceOp, containingOp) ||
          sliceOp == firstSliceOp) {
        continue;
      }

      LDBG("union slice \n" << sliceOp);

      // get and clone offsets if it is defined below inserted point
      auto offsets = getValueOrCreateConstantIndexOp(
          rewriter, sliceOp->getLoc(), sliceOp.getMixedOffsets());
      auto clonedOffsets =
          recursiveClone(rewriter, offsets, firstSliceOp.getOperation());

      // union offsets
      unionOffsets = compute(rewriter, MODE::UNION_MIN, unionOffsets,
                             clonedOffsets, firstSliceOp->getLoc());

      // get and clone sizes if it is defined below inserted point
      auto sizes = getValueOrCreateConstantIndexOp(
          rewriter, firstSliceOp.getLoc(), sliceOp.getMixedSizes());
      auto clonedSizes =
          recursiveClone(rewriter, sizes, firstSliceOp.getOperation());

      // compute max
      auto clonedMaxes =
          compute(rewriter, MODE::COMPUTE_SLICE_MAX, clonedOffsets, clonedSizes,
                  firstSliceOp->getLoc());
      // union max
      unionMaxes = compute(rewriter, MODE::UNION_MAX, unionMaxes, clonedMaxes,
                           firstSliceOp->getLoc());
    }

    auto unionSizes = compute(rewriter, MODE::COMPUTE_DISTANCE, unionMaxes,
                              unionOffsets, firstSliceOp->getLoc());

    auto unionSlice = rewriter.create<tensor::ExtractSliceOp>(
        firstSliceOp.getLoc(), firstSliceOp.getSource(), convert(unionOffsets),
        convert(unionSizes), firstSliceOp.getMixedStrides());

    LDBG("insert union slice \n" << unionSlice);
    LDBG(*containingOp);
    // update users to use union slice result
    for (auto *user : produceResult.getUsers()) {
      auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
      if (!isValidSliceOpInContainingOp(sliceOp, containingOp) ||
          sliceOp == unionSlice) {
        continue;
      }

      rewriter.setInsertionPoint(sliceOp.getOperation());
      auto offsets = getValueOrCreateConstantIndexOp(rewriter, sliceOp.getLoc(),
                                                     sliceOp.getMixedOffsets());
      auto newOffsets = compute(rewriter, MODE::COMPUTE_SUB, offsets,
                                unionOffsets, user->getLoc());
      auto newSlice = rewriter.create<tensor::ExtractSliceOp>(
          sliceOp.getLoc(), unionSlice.getResult(), convert(newOffsets),
          sliceOp.getMixedSizes(), unionSlice.getMixedStrides());
      rewriter.replaceOp(sliceOp.getOperation(), newSlice.getResult());
    }

    LDBG("unioned containingOp: \n" << *containingOp);
  }
}

/// Add new operands to the forall op for users of the producerOp
/// that are dominated by the containing scf.forall op.
static Operation *replaceForAllWithNewSignature(
    RewriterBase &rewriter, Diagnostic &diag, Operation *producerOp,
    Operation *containingOp, TilingResult &tileAndFuseResult,
    int64_t resultNumber, SmallVector<OpFoldResult> &offsets,
    SmallVector<OpFoldResult> &sizes) {

  // Count number of users not including the containing op
  SetVector<Operation *> dominatedUsers;
  DominanceInfo domInfo(containingOp);
  for (Operation *user : producerOp->getResult(resultNumber).getUsers()) {
    if (!containingOp->isAncestor(user) &&
        (domInfo.dominates(containingOp, user))) {
      dominatedUsers.insert(user);
    }
  }
  if (dominatedUsers.empty())
    return nullptr;

  // Create new scf.forall op
  auto forallOp = cast<scf::ForallOp>(containingOp);
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(forallOp);

  // Get new output
  Location loc = forallOp.getLoc();
  auto genericOp = dyn_cast<linalg::GenericOp>(producerOp);
  if (!genericOp)
    return nullptr;
  SmallVector<Value> outputs = genericOp.getOutputs();
  SmallVector<Value> newOuts(forallOp.getOutputs());
  newOuts.push_back(outputs[resultNumber]);

  // Create new scf.forall op
  auto newforallOp = rewriter.create<scf::ForallOp>(
      loc, forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
      forallOp.getMixedStep(), newOuts, forallOp.getMapping());
  rewriter.eraseBlock(newforallOp.getBody());
  newforallOp.getRegion().takeBody(forallOp.getRegion());

  // Add additional block argument for new value being returned
  // and replaces all uses of the new output with corresponding bbArg
  // inside the scf.forall to enable fusion into this new scf.forall.
  newforallOp.getBody()->addArgument(newOuts.back().getType(),
                                     newOuts.back().getLoc());
  auto bbArgs = newforallOp.getBody()->getArguments();
  rewriter.replaceUsesWithIf(newOuts.back(), bbArgs.back(),
                             [&](OpOperand &use) {
                               Operation *op = use.getOwner();
                               return newforallOp->isProperAncestor(op);
                             });

  // Fix terminator
  scf::InParallelOp terminatorOp = newforallOp.getTerminator();
  SmallVector<Operation *> yieldingOps = llvm::to_vector<4>(llvm::map_range(
      terminatorOp.getYieldingOps(), [](Operation &op) { return &op; }));
  Operation *firstYieldOp = yieldingOps.front();
  rewriter.setInsertionPoint(firstYieldOp);
  Value src = tileAndFuseResult.tiledValues[0];
  Value dst = newforallOp.getRegionIterArgs().back();
  SmallVector<OpFoldResult> strides(offsets.size(), rewriter.getIndexAttr(1));
  rewriter.create<tensor::ParallelInsertSliceOp>(firstYieldOp->getLoc(), src,
                                                 dst, offsets, sizes, strides);

  for (auto result : llvm::enumerate(forallOp.getResults())) {
    rewriter.replaceAllUsesWith(result.value(),
                                newforallOp->getResult(result.index()));
  }
  rewriter.replaceUsesWithIf(producerOp->getResult(resultNumber),
                             newforallOp->getResults().back(),
                             [&](OpOperand &use) {
                               Operation *user = use.getOwner();
                               return dominatedUsers.contains(user);
                             });
  return newforallOp;
}

/// Find the first "extract" user of `producerOp` and tile it right before its
/// use. The tiled op is fused under the `containingOp`.
/// Return this fused op on success or nullptr if anything fails.
/// If tiled op has uses that are dominated by `containingOp`, return
/// a new `containingOp` with results of the fused op appended to
/// results of the `containingOp` or nullptr if there are no dominated uses.
/// However, if `duplicateProducer` is set to true, then the `producerOp` is
/// expected to be tiled and fused into all users.
static std::tuple<SmallVector<Operation *>, Operation *>
tileAndFuseFirstExtractUse(RewriterBase &rewriter, Diagnostic &diag,
                           Operation *producerOp, Operation *containingOp,
                           bool duplicateProducer) {
  LLVM_DEBUG(DBGS() << "Try to fuse a direct extract use\n");
  auto tileableProducer = dyn_cast<TilingInterface>(producerOp);
  if (!tileableProducer) {
    diag.attachNote(producerOp->getLoc())
        << "producer is not a TileableInterface: " << *producerOp;
    return {};
  }

  // Search the producer slices accessed within the containing operation.
  // TODO: Generalize to more extract/insert/parallel_insert triples, maybe
  // evolve into an interface.
  auto it = llvm::find_if(tileableProducer->getUsers(), [&](Operation *user) {
    auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
    return sliceOp && containingOp->isProperAncestor(sliceOp);
  });

  // Find a fusion opportunity.
  if (it == tileableProducer->getUsers().end()) {
    diag.attachNote(tileableProducer->getLoc())
        << "could not find fusion opportunity for: " << *tileableProducer;
    return {};
  }
  auto sliceOpToTile = cast<tensor::ExtractSliceOp>(*it);

  // Try to fuse the producer in-place.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(sliceOpToTile);

  // Tile the producer.
  int64_t resultNumber =
      cast<OpResult>(sliceOpToTile.getSource()).getResultNumber();
  LLVM_DEBUG(DBGS() << "resultNumber: " << resultNumber << "\n");

  SmallVector<OpFoldResult> offsets = sliceOpToTile.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = sliceOpToTile.getMixedSizes();

  FailureOr<TilingResult> tileAndFuseResult =
      tileableProducer.generateResultTileValue(rewriter, resultNumber, offsets,
                                               sizes);

  if (failed(tileAndFuseResult)) {
    diag.attachNote(tileableProducer->getLoc())
        << "failed to tile producer op: " << *tileableProducer;
    return {};
  }

#ifndef NDEBUG
  for (auto *tiledOp : tileAndFuseResult->tiledOps) {
    LLVM_DEBUG(DBGS() << "tiledProducer: " << *tiledOp << "\n");
  }
#endif

  // Replace the extract op.
  auto maybeRankReduced = tensor::ExtractSliceOp::rankReduceIfNeeded(
      rewriter, sliceOpToTile->getLoc(), tileAndFuseResult->tiledValues[0],
      cast<RankedTensorType>(sliceOpToTile->getResult(0).getType()).getShape());
  if (failed(maybeRankReduced)) {
    diag.attachNote(producerOp->getLoc())
        << "shape types don't match (missing canonicalization?):\nTiledOp: "
        << tileAndFuseResult->tiledValues[0]
        << "\nSliceOp: " << sliceOpToTile.getOperation() << '\n';
    return {};
  }
  rewriter.replaceOp(sliceOpToTile, *maybeRankReduced);

  if (duplicateProducer)
    return std::make_tuple(tileAndFuseResult->tiledOps, nullptr);

  // Add new outputs to containing op, if required
  Operation *newContainingOp = replaceForAllWithNewSignature(
      rewriter, diag, producerOp, containingOp, *tileAndFuseResult,
      resultNumber, offsets, sizes);

  return std::make_tuple(tileAndFuseResult->tiledOps, newContainingOp);
}

/// First, find the first "scf::ForallOp" user of `producerOp` and ensure
/// it is exactly the `containingOp`, otherwise bail.
/// Then, find the first "extract" user of the tied block argument and tile it
/// right before its "extract" use. The tiled op is fused under the
/// `containingOp`.
/// Return this fused op on success or nullptr if anything fails.
static SmallVector<Operation *>
tileAndFuseFirstExtractUseThroughContainingOpBlockArgument(
    RewriterBase &rewriter, Diagnostic &diag, Operation *producerOp,
    Operation *containingOp) {
  LLVM_DEBUG(DBGS() << "Try to fuse an extract use through block argument\n");

  auto tileableProducer = dyn_cast<TilingInterface>(producerOp);
  if (!tileableProducer) {
    diag.attachNote(producerOp->getLoc())
        << "producer is not a TileableInterface: " << *producerOp;
    return {};
  }

  // Search the first use by a "scf::ForallOp" user.
  scf::ForallOp forallOp;
  auto itProducerUses =
      llvm::find_if(tileableProducer->getUses(), [&](OpOperand &use) {
        forallOp = dyn_cast<scf::ForallOp>(use.getOwner());
        return forallOp;
      });
  // If it's not from the containing op, return.
  if (!forallOp || forallOp != containingOp) {
    diag.attachNote(tileableProducer->getLoc())
        << "could not find a use by the containing op: " << *tileableProducer;
    return {};
  }

  // Search the producer slices accessed within the containing
  // operation.
  // TODO: Generalize to more extract/insert/parallel_insert triples.
  //   Maybe evolve into an interface.
  OpOperand *pUse = &(*itProducerUses);
  BlockArgument bbArg = forallOp.getTiedBlockArgument(pUse);

  // Search the producer slices accessed within the containing operation.
  // TODO: Generalize to more extract/insert/parallel_insert triples, maybe
  // evolve into an interface.
  auto itBBArgUsers = llvm::find_if(bbArg.getUsers(), [&](Operation *user) {
    auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
    return sliceOp && containingOp->isProperAncestor(sliceOp);
  });

  // Find a fusion opportunity.
  if (itBBArgUsers == bbArg.getUsers().end()) {
    diag.attachNote(containingOp->getLoc())
        << "could not find fusion opportunity for bbArg: " << bbArg;
    return {};
  }
  auto sliceOpToTile = cast<tensor::ExtractSliceOp>(*itBBArgUsers);

  // Try to fuse the producer in-place.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(sliceOpToTile);

  // Replace the use in the tileableProducer before tiling: clone, replace and
  // then tile.
  int64_t resultNumber = cast<OpResult>(pUse->get()).getResultNumber();
  LLVM_DEBUG(DBGS() << "resultNumber: " << resultNumber << "\n");

  // Gather destination tensors.
  SmallVector<Value> destinationTensors;
  if (failed(tensor::getOrCreateDestinations(
          rewriter, tileableProducer->getLoc(), tileableProducer,
          destinationTensors))) {
    diag.attachNote(tileableProducer->getLoc())
        << "failed to get destination tensors for: " << *tileableProducer;
    return {};
  }

  IRMapping bvm;
  bvm.map(destinationTensors[resultNumber], bbArg);
  auto tileableProducerClone =
      cast<TilingInterface>(rewriter.clone(*tileableProducer, bvm));
  auto scopeGuard =
      llvm::make_scope_exit([&]() { rewriter.eraseOp(tileableProducerClone); });

  // Tile the producer.
  FailureOr<TilingResult> tileAndFuseResult =
      tileableProducerClone.generateResultTileValue(
          rewriter, resultNumber, sliceOpToTile.getMixedOffsets(),
          sliceOpToTile.getMixedSizes());
  if (failed(tileAndFuseResult)) {
    diag.attachNote(tileableProducer->getLoc())
        << "failed to tile producer op: " << *tileableProducer;
    return {};
  }

  // Replace the extract op.
  auto maybeRankReduced = tensor::ExtractSliceOp::rankReduceIfNeeded(
      rewriter, sliceOpToTile->getLoc(), tileAndFuseResult->tiledValues[0],
      cast<RankedTensorType>(sliceOpToTile->getResult(0).getType()).getShape());
  assert(succeeded(maybeRankReduced) && "unexpected shape");
  rewriter.replaceOp(sliceOpToTile, *maybeRankReduced);

  // Replace the use in containingOp.
  rewriter.modifyOpInPlace(containingOp, [&]() {
    containingOp->setOperand(pUse->getOperandNumber(),
                             destinationTensors.front());
  });

  return tileAndFuseResult->tiledOps;
}

static Operation *cloneAndFuseFirstUse(RewriterBase &rewriter, Diagnostic &diag,
                                       Operation *producerOp,
                                       Operation *containingOp) {
  LLVM_DEBUG(DBGS() << "Try to fuse an use by cloning\n");

  // Gather all uses inside the containing op.
  SmallVector<OpOperand *> uses;
  for (OpResult result : producerOp->getOpResults()) {
    for (OpOperand &use : result.getUses()) {
      if (containingOp->isProperAncestor(use.getOwner())) {
        uses.push_back(&use);
        continue;
      }
      // Cannot clone and fuse if the use is by the containing op itself: fail
      // immediately.
      if (containingOp == use.getOwner()) {
        diag.attachNote(producerOp->getLoc())
            << "producer op use by containing op cannot be fused by cloning";
        return nullptr;
      }
    }
  }

  // Check for a non-empty list of fusion opportunities.
  if (uses.empty()) {
    diag.attachNote(producerOp->getLoc()) << "no fusion opportunity by cloning";
    return nullptr;
  }

  // Clone and fuse inside the containing op.
  Operation *fusedOp = nullptr;
  OpOperand *use = uses.front();
  // Parallel insert slice is not a valid clone destination.
  // TODO: Generalize to other type of ops.
  assert(!isa<tensor::ParallelInsertSliceOp>(use->getOwner()) &&
         "Parallel insert slice is not a valid clone destination");
  unsigned resultNumber = cast<OpResult>(use->get()).getResultNumber();
  LLVM_DEBUG(DBGS() << "resultNumber: " << resultNumber << "\n");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(use->getOwner());
  fusedOp = rewriter.clone(*producerOp);
  rewriter.modifyOpInPlace(
      use->getOwner(), [&] { use->set(fusedOp->getOpResult(resultNumber)); });

  return fusedOp;
}

DiagnosedSilenceableFailure
transform::ExtendedFuseIntoContainingOp::fuseIntoOneContaining(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state,
    size_t index, Operation *containingOp) {
  assert(index < getFusedOp().size());
  assert(index < getNewContainingOp().size());

  SmallVector<Operation *> fusedOps;
  auto producerOps = state.getPayloadOps(getProducerOp());
  // If nothing to fuse, propagate success.
  if (std::empty(producerOps)) {
    results.set(cast<OpResult>(getFusedOp()[index]),
                SmallVector<mlir::Operation *>{});
    results.set(cast<OpResult>(getNewContainingOp()[index]), {containingOp});
    return DiagnosedSilenceableFailure::success();
  }

  // Helper function to find the next producer that should be fused. Take any
  // producer that has a use inside the containing op.
  DenseMap<Operation *, int> producerOp2ResultNumer;
  for (auto producerOp : producerOps) {
    producerOp2ResultNumer.insert(
        std::pair(producerOp, producerOp->getNumResults()));
  }

  SetVector<Operation *> remainingProducers(producerOps.begin(),
                                            producerOps.end());
  auto getNextProducer = [&]() -> FailureOr<Operation *> {
    for (const auto &it : enumerate(remainingProducers)) {
      Operation *producerOp = it.value();
      // The containing op may be a user of producerOp: use isAncestor.
      int64_t numUsesInContainingOp =
          llvm::count_if(producerOp->getUsers(), [&](Operation *op) {
            return containingOp->isAncestor(op);
          });
      if (numUsesInContainingOp > 0) {
        producerOp2ResultNumer[producerOp]--;
        if (producerOp2ResultNumer[producerOp] == 0)
          remainingProducers.erase(remainingProducers.begin() + it.index());
        return producerOp;
      }
    }
    return failure();
  };

  while (!remainingProducers.empty()) {
    auto nextProducer = getNextProducer();
    if (failed(nextProducer)) {
      auto diag = mlir::emitSilenceableFailure(getLoc())
                  << "could not find next producer to fuse into container";
      diag.attachNote(containingOp->getLoc()) << "containing op";
      return diag;
    }

    Operation *producerOp = *nextProducer;

    // Default diagnostic, to be complemented with more failure information.
    Diagnostic diag(producerOp->getLoc(), DiagnosticSeverity::Remark);
    diag << "could not fuse " << *producerOp << " into " << *containingOp;

    // Union the multiple consumers in containing op.
    unionProducerUsers(rewriter, diag, producerOp, containingOp);

    auto [tiledOps, newContainingOp] = tileAndFuseFirstExtractUse(
        rewriter, diag, producerOp, containingOp, getDuplicateProducer());
    if (!tiledOps.empty()) {
      LLVM_DEBUG(DBGS() << "\nFused a direct extract use\n" << *containingOp);
      fusedOps.append(tiledOps);
      if (newContainingOp) {
        // Update handles associated with the containing op so we don't need
        // to invalidate them. This is a hack to support better composability
        // between tiling and fusion while a proper mechanism is being
        // investigated.
        //
        // DO NOT replicate this elsewhere unless you understand what you are
        // doing.
        LogicalResult replacementStatus =
            rewriter.notifyPayloadOperationReplaced(containingOp,
                                                    newContainingOp);
        (void)replacementStatus;
        assert(succeeded(replacementStatus) &&
               "unable to update transform state mapping");
        rewriter.eraseOp(containingOp);
        containingOp = newContainingOp;
      }
      continue;
    }

    SmallVector<Operation *> tiledContainingOpOperand =
        tileAndFuseFirstExtractUseThroughContainingOpBlockArgument(
            rewriter, diag, producerOp, containingOp);
    if (!tiledContainingOpOperand.empty()) {
      LLVM_DEBUG(DBGS() << "\nFused an extract use through block argument\n"
                        << *containingOp);
      fusedOps.append(tiledContainingOpOperand);
      continue;
    }

    Operation *cloned =
        cloneAndFuseFirstUse(rewriter, diag, producerOp, containingOp);
    if (cloned) {
      LLVM_DEBUG(DBGS() << "\nFused an use by cloning\n" << *containingOp);
      fusedOps.push_back(cloned);
      continue;
    }
    return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
  }
  results.set(cast<OpResult>(getFusedOp()[index]), fusedOps);
  results.set(cast<OpResult>(getNewContainingOp()[index]), {containingOp});
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure transform::ExtendedFuseIntoContainingOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  auto containingOps = getContainingOp();
  for (auto it : llvm::enumerate(containingOps)) {
    auto containingOpPayloads = state.getPayloadOps(it.value());
    if (!llvm::hasSingleElement(containingOpPayloads)) {
      return emitDefiniteFailure()
             << "requires exactly one containing_op handle (got "
             << llvm::range_size(containingOpPayloads) << ")";
    }
    Operation *currentOp = *containingOpPayloads.begin();
    auto status =
        fuseIntoOneContaining(rewriter, results, state, it.index(), currentOp);
    if (!status.succeeded())
      return status;
  }
  return DiagnosedSilenceableFailure::success();
}

ParseResult ExtendedFuseIntoContainingOp::parse(OpAsmParser &parser,
                                                OperationState &result) {
  OpAsmParser::UnresolvedOperand producer;
  SmallVector<OpAsmParser::UnresolvedOperand> containingOps;
  FunctionType functionalType;
  llvm::SMLoc producerLoc;
  llvm::SMLoc containingOpsLoc;

  if (parser.getCurrentLocation(&producerLoc) || parser.parseOperand(producer))
    return ParseResult::failure();

  if (parser.parseKeyword("into"))
    return ParseResult::failure();

  if (parser.getCurrentLocation(&containingOpsLoc) ||
      parser.parseOperandList(containingOps))
    return ParseResult::failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return ParseResult::failure();

  if (result.propertiesAttr) {
    NamedAttrList attrs = llvm::cast<DictionaryAttr>(result.propertiesAttr);
    attrs.append("resultSegmentSizes",
                 parser.getBuilder().getDenseI32ArrayAttr(
                     {static_cast<int32_t>(containingOps.size()),
                      static_cast<int32_t>(containingOps.size())}));
    result.propertiesAttr = attrs.getDictionary(parser.getContext());
  } else {
    result.addAttribute("resultSegmentSizes",
                        parser.getBuilder().getDenseI32ArrayAttr(
                            {static_cast<int32_t>(containingOps.size()),
                             static_cast<int32_t>(containingOps.size())}));
  }

  if (parser.parseColonType(functionalType))
    return ParseResult::failure();

  if (parser.resolveOperand(producer, functionalType.getInputs().front(),
                            result.operands) ||
      parser.resolveOperands(containingOps,
                             functionalType.getInputs().drop_front(),
                             containingOpsLoc, result.operands)) {
    return ParseResult::failure();
  }

  result.addTypes(functionalType.getResults());
  return ParseResult::success();
}

void ExtendedFuseIntoContainingOp::print(OpAsmPrinter &p) {
  p << ' ' << getProducerOp();
  p << ' ' << "into";
  p << ' ';
  p.printOperands(getContainingOp());
  p.printOptionalAttrDict((*this)->getAttrs(), {"resultSegmentSizes"});
  p << " : ";
  p.printFunctionalType(getOperands().getTypes(), getResults().getTypes());
}

void transform::ExtendedFuseIntoContainingOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getProducerOp(), effects);
  onlyReadsHandle(getContainingOp(), effects);
  producesHandle(getResults(), effects);
  // consumesHandle(getProducerOpMutable(), effects);
  // onlyReadsHandle(getContainingOpMutable(), effects);
  // producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// SetBufferSizeOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
adjustBufferSizeByReferenceType(int64_t *bufferSize, Type currentElementType,
                                Type referenceElementType, Location loc) {
  auto referenceTypeWidth = referenceElementType.getIntOrFloatBitWidth();
  auto currentTypeWidth = currentElementType.getIntOrFloatBitWidth();
  if (referenceTypeWidth > currentTypeWidth)
    return emitDefiniteFailure(
        loc, "Reference type's bit width should be less than or equal to the "
             "current element type!");
  auto factor = currentTypeWidth / referenceTypeWidth;
  if (currentTypeWidth % referenceTypeWidth != 0)
    factor = (currentTypeWidth + referenceTypeWidth - 1) / referenceTypeWidth;
  *bufferSize = *bufferSize * factor;
  return DiagnosedSilenceableFailure::success();
}

template <typename AllocOpTy>
DiagnosedSilenceableFailure
setBufferSizeForAllocOp(Operation *op, int64_t bufferSize,
                        std::optional<Type> referenceType,
                        transform::TransformRewriter &rewriter) {
  assert(op);
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(op);
  Location loc = op->getLoc();
  auto oldType = dyn_cast<MemRefType>(op->getResultTypes().front());
  assert(oldType);
  // If the memref alloc has static shape, do nothing.
  if (oldType.hasStaticShape()) {
    return DiagnosedSilenceableFailure::success();
  }
  if (referenceType.has_value()) {
    auto diag = adjustBufferSizeByReferenceType(
        &bufferSize, oldType.getElementType(), *referenceType, op->getLoc());
    if (!diag.succeeded())
      return diag;
  }
  // Create new alloc with static size.
  auto newMemrefType =
      MemRefType::get({bufferSize}, rewriter.getI8Type(), mlir::AffineMap{},
                      oldType.getMemorySpace());
  auto newAllocOp = rewriter.create<AllocOpTy>(loc, newMemrefType);
  // Create view from new alloc to old alloc's sizes and replace its use.
  auto startOffset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto viewOp = rewriter.create<memref::ViewOp>(
      loc, oldType, newAllocOp.getResult(), startOffset, op->getOperands());
  rewriter.replaceOp(op, viewOp);
  return DiagnosedSilenceableFailure::success();
}

static DiagnosedSilenceableFailure
setBufferSizeForOp(Operation *op, int64_t bufferSize,
                   std::optional<Type> referenceType,
                   transform::TransformRewriter &rewriter) {
  assert(op);
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(op);
  for (auto result : op->getResults()) {
    // If the op result is a static shape type, do nothing.
    auto maybeShapedType = dyn_cast<ShapedType>(result.getType());
    if (maybeShapedType && maybeShapedType.hasStaticShape()) {
      continue;
    }
    // Each result might have different data type.
    int64_t bufferSizeForResult = bufferSize;
    if (referenceType.has_value()) {
      auto diag = adjustBufferSizeByReferenceType(
          &bufferSizeForResult, maybeShapedType.getElementType(),
          *referenceType, op->getLoc());
      if (!diag.succeeded())
        return diag;
    }
    // auto mark = rewriter.create<annotation::MarkOp>(op->getLoc(), result);
    // mark->setAttr(kBufferSizeInByteAttr,
    //               rewriter.getI64IntegerAttr(bufferSizeForResult));
  }
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
SetBufferSizeOp::apply(transform::TransformRewriter &rewriter,
                       transform::TransformResults &transformResults,
                       transform::TransformState &state) {
  auto staticBufferSizes = getStaticBufferSizes();
  if (getTarget().size() != staticBufferSizes.size())
    return emitDefiniteFailure(
        "Number of operands to set does not match buffer size count!");

  SetBufferSizeMode unitMode = getUnitMode();
  std::optional<Type> maybeReferenceType = getReferenceType();
  for (const auto &targetHandle : llvm::enumerate(getTarget())) {
    auto payloadOps = state.getPayloadOps(targetHandle.value());
    for (Operation *definingOp : payloadOps) {
      auto staticBufferSize = staticBufferSizes[targetHandle.index()];
      if (staticBufferSize < 0)
        return emitDefiniteFailure("buffer size should be greater than 0!");

      if (unitMode == SetBufferSizeMode::kPerElement) {
        int perElementByte =
            getElementTypeOrSelf(definingOp->getResultTypes().front())
                .getIntOrFloatBitWidth() /
            8;
        staticBufferSize = staticBufferSize * perElementByte;
      }

      if (maybeReferenceType.has_value()) {
        if (!(*maybeReferenceType).isIntOrFloat())
          return emitDefiniteFailure(
              "reference type must be an int or float type!");
      }
      DiagnosedSilenceableFailure diag = DiagnosedSilenceableFailure::success();
      if (isa<memref::AllocOp>(definingOp)) {
        diag = setBufferSizeForAllocOp<memref::AllocOp>(
            definingOp, staticBufferSize, maybeReferenceType, rewriter);
      } else if (isa<memref::AllocaOp>(definingOp)) {
        diag = setBufferSizeForAllocOp<memref::AllocaOp>(
            definingOp, staticBufferSize, maybeReferenceType, rewriter);
      } else {
        diag = setBufferSizeForOp(definingOp, staticBufferSize,
                                  maybeReferenceType, rewriter);
      }
      if (!diag.succeeded())
        return diag;
    }
  }
  return DiagnosedSilenceableFailure::success();
}

void SetBufferSizeOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTarget(), effects);
  // consumesHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// MultiBufferOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
MultiBufferOp::apply(transform::TransformRewriter &rewriter,
                     transform::TransformResults &transformResults,
                     transform::TransformState &state) {
  auto factor = getFactor();
  if (factor < 1) {
    emitError("factor should be >= 1.");
    return DiagnosedSilenceableFailure::definiteFailure();
  }

  for (const auto &targetHandle : getTarget()) {
    auto payloadOps = state.getPayloadOps(targetHandle);
    for (Operation *definingOp : payloadOps) {
      assert(definingOp && "definingOp shouldn't be null.");
      if (!definingOp->getResults().empty()) {
        rewriter.setInsertionPointAfter(definingOp);
        for (auto res : definingOp->getResults()) {
          // auto markOp =
          //     rewriter.create<annotation::MarkOp>(definingOp->getLoc(), res);
          // markOp->setAttr(mtfusion::MultiBufferAttr::name,
          //                 rewriter.getI32IntegerAttr(factor));
        }
      }
    }
  }

  return DiagnosedSilenceableFailure::success();
}

void MultiBufferOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTarget(), effects);
  // onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

//===---------------------------------------------------------------------===//
// MatchAncestorOfOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::MatchAncestorOfOp::apply(transform::TransformRewriter &rewriter,
                                    transform::TransformResults &results,
                                    transform::TransformState &state) {
  llvm::StringSet<> strs;
  if (getOps().has_value())
    strs.insert(getOps()->getAsValueRange<StringAttr>().begin(),
                getOps()->getAsValueRange<StringAttr>().end());

  auto payloadOps = state.getPayloadOps(getTarget());
  if (!llvm::hasSingleElement(payloadOps)) {
    return emitDefiniteFailure("requires exactly one target handle");
  }

  auto childOps = state.getPayloadOps(getChild());
  if (!llvm::hasSingleElement(childOps)) {
    return emitDefiniteFailure("requires exactly one child handle");
  }
  Operation *childOp = *childOps.begin();
  // Build dominance info from enclosing function
  func::FuncOp enclosingFunc = childOp->getParentOfType<func::FuncOp>();
  DominanceInfo domInfo(enclosingFunc);

  SmallVector<Operation *> res;
  bool incorrectNumOperandTypes = false;
  auto matchFun = [&](Operation *op) {
    if (getOps().has_value() && !strs.contains(op->getName().getStringRef()))
      return;

    // Interfaces cannot be matched by name, just by ID.
    // So we specifically encode the interfaces we care about for this op.
    if (getInterface().has_value()) {
      auto iface = getInterface().value();
      if (iface == transform::MatchInterfaceEnum::LinalgOp &&
          !isa<linalg::LinalgOp>(op))
        return;
      if (iface == transform::MatchInterfaceEnum::TilingInterface &&
          !isa<TilingInterface>(op))
        return;
      if (iface == transform::MatchInterfaceEnum::LoopLikeInterface &&
          !isa<LoopLikeOpInterface>(op))
        return;
    }

    // Check if all specified attributes match.
    if (getOpAttrs().has_value()) {
      DictionaryAttr opAttrs = getOpAttrs().value();
      for (NamedAttribute attr : opAttrs) {
        if (attr.getName() == getInterfaceAttrName() ||
            attr.getName() == getOpsAttrName())
          continue;
        if (!op->hasAttr(attr.getName()))
          return;
        if (op->getAttr(attr.getName()) != attr.getValue())
          return;
      }
    }

    if (getFilterResultType().has_value()) {
      Type t = getFilterResultType().value();
      if (op->getNumResults() != 1 || op->getResultTypes().front() != t)
        return;
    }

    if (getFilterOperandTypes().has_value()) {
      mlir::ArrayAttr types = getFilterOperandTypes().value();
      auto operandTypes = op->getOperandTypes();

      if (types.size() == 1) {
        // All the operands must be equal to the specified type
        auto typeattr =
            dyn_cast<mlir::TypeAttr>(getFilterOperandTypes().value()[0]);
        Type t = typeattr.getValue().cast<::mlir::Type>();
        if (!llvm::all_of(op->getOperandTypes(),
                          [&](Type operandType) { return operandType == t; }))
          return;
      } else {
        // The operand types must match all the types in the list (in the same
        // order in with they are specified)
        if (types.size() != operandTypes.size()) {
          incorrectNumOperandTypes = true;
          return;
        }

        for (auto [attr, operandType] :
             llvm::zip_equal(getFilterOperandTypes().value(), operandTypes)) {
          auto typeattr = cast<mlir::TypeAttr>(attr);
          Type type = typeattr.getValue().cast<::mlir::Type>();

          if (type != operandType)
            return;
        }
      }
    }

    if (!domInfo.properlyDominates(op, childOp))
      return;

    // All constraints are satisfied.
    res.push_back(op);
    return;
  };

  (*payloadOps.begin())->walk(matchFun);
  if (incorrectNumOperandTypes)
    return emitDefiniteFailure("If filter_operand_types contains more than a "
                               "type, then it must contain as much types as "
                               "the number of operands in the target ops");
  results.set(cast<OpResult>(getResult()), res);
  return DiagnosedSilenceableFailure::success();
}

void transform::MatchAncestorOfOp::build(OpBuilder &builder,
                                         OperationState &result, Value target,
                                         Value child,
                                         ArrayRef<StringRef> opNames) {
  result.addOperands(target);
  result.addOperands(child);
  result.addAttribute(MatchOp::getOpsAttrName(result.name),
                      builder.getStrArrayAttr(opNames));
  result.addTypes(transform::AnyOpType::get(builder.getContext()));
}

void transform::MatchAncestorOfOp::build(OpBuilder &builder,
                                         OperationState &result,
                                         TypeRange resultTypes, Value target,
                                         Value child,
                                         ArrayRef<StringRef> opNames) {
  result.addOperands(target);
  result.addOperands(child);
  result.addAttribute(MatchOp::getOpsAttrName(result.name),
                      builder.getStrArrayAttr(opNames));
  result.addTypes(resultTypes);
}

#include "mtir/Dialect/MtFusion/TransformOps/MtFusionTransformOpsEnums.cpp.inc"
#define GET_OP_CLASSES
#include "mtir/Dialect/MtFusion/TransformOps/MtFusionTransformOps.cpp.inc"