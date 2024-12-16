//===- FusableHelper.cpp - Provide utilities and fusion rules to analyzer -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Transforms/OpFusion/FusableHelper.h"
#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/Utils/Utils.h"
#include "mtir/Dialect/Utils/Util.h"

#include "llvm/Support/Debug.h"
#include <iostream>

#include <numeric>
#include <queue>

#define DEBUG_TYPE "mtfusion-fuse"
namespace mlir {
namespace mtfusion {
namespace opfusion {
using mlir::mtfusion::reshape_utils::isMarkedAsElementwiseOp;

//===---------------------------------------------------------------------===//
// Utils
//===---------------------------------------------------------------------===//

bool FusableHelper::isSingleOutlinable(Operation *op) {
  auto pattern = FusableHelper::getOpPattern(op);
  // op->dump();
  // std::cout << "Checking op " << static_cast<int>(pattern) << "\n" << std::endl;

  LLVM_DEBUG(llvm::dbgs() << "Checking op " << static_cast<uint8_t>(pattern)
                          << "\n";);
  return static_cast<uint8_t>(pattern) >=
         static_cast<uint8_t>(opfusion::OpPattern::kElementWise);
}

FusionKind FusableHelper::getSingleFusionKind(Operation *op) {
  auto pattern = FusableHelper::getOpPattern(op);

  LLVM_DEBUG(llvm::dbgs() << "Trying to single outline\n";);
  switch (pattern) {
  case opfusion::OpPattern::kElementWise:
    return FusionKind::PureElemwise;
  case opfusion::OpPattern::kLastAxisReduce:
    return FusionKind::LastAxisPBR;
  case opfusion::OpPattern::kMatmul:
    return FusionKind::MixCV;
  case opfusion::OpPattern::kLastAxisBroadcast:
  case opfusion::OpPattern::kOtherBroadcast:
    return FusionKind::AnyPB;
  default:
    llvm_unreachable("Invalid operation pattern for outlining");
  }
}

// README: [$FusionType] Means it is for that certain FusionType
FusableHelper::FusableHelper(FusionKind fusionKind, bool bufferToOut,
                             int32_t maxHorizontalFusionSize)
    : fusionKind_(fusionKind), moveOutToParam_(bufferToOut),
      maxHorizontalFusion_(
          maxHorizontalFusionSize == -1 ? INT_MAX : maxHorizontalFusionSize) {}

// [General] This is to check whether out tensor allocation
// should be included in the fusion or not.
//
// If it's true, than all out operator's tensor.empty / allocation
// would be left in the caller (not included in the fusion)
bool FusableHelper::moveOutToParam() const { return moveOutToParam_; }

// [General] This is to check how many max non-dependent function fusion
// should be attempted. -1 to merge all, 0 to separate all.
bool FusableHelper::maxHorizontalFusion() const { return maxHorizontalFusion_; }

// [General] Auxiliary nodes are the arith, etc.
bool FusableHelper::includeAuxiliary(Operation *op) const {
  return getOpPattern(op) == OpPattern::kAuxiliary;
}

// [General] Auxiliary nodes are the empty etc, etc.
bool FusableHelper::includeBuffer(Operation *op) const {
  return getOpPattern(op) == OpPattern::kBuffer;
}

bool FusableHelper::isFusable(Operation *a, Operation *b) const {
  return isFusable(getOpPattern(a), getOpPattern(b));
}

// [MixCV] Node Type Checking
uint8_t FusableHelper::obtainType(Operation *op) const {
  OpPattern pattern = getOpPattern(op);
  TypePattern returnType = TypePattern::kOpaque;
  switch (pattern) {
  case OpPattern::kElementWise:
    returnType = TypePattern::kPureElementWise;
    break;
  case OpPattern::kMatmul:
    returnType = TypePattern::kPureMatmul;
    break;
  default:
    break;
  }
  return static_cast<uint8_t>(returnType);
}

// [MixCV] the matmul allowed is either at the end or at the beginning
uint8_t FusableHelper::adjustType(const uint8_t &typeA,
                                  const uint8_t &typeB) const {
  TypePattern returnType = TypePattern::kOpaque;
  TypePattern patternA = static_cast<TypePattern>(typeA);
  TypePattern patternB = static_cast<TypePattern>(typeB);
  switch (patternA) {
  case TypePattern::kPureElementWise:
    switch (patternB) {
    case TypePattern::kPureElementWise:
      // (Elwise -> .. -> Elwise) + (Elwise -> .. -> Elwise)
      returnType = TypePattern::kPureElementWise;
      break;
    case TypePattern::kPureMatmul:
    case TypePattern::kPrefixElementWise:
      // (Elwise -> .. -> Elwise) + (Matmul)
      // (Elwise -> .. -> Elwise) + (Elwise -> .. -> Elwise -> Matmul)
      returnType = TypePattern::kPrefixElementWise;
      break;
    default:
      break;
    }
    break;
  case TypePattern::kPureMatmul:
    switch (patternB) {
    case TypePattern::kPureElementWise:
      // (Matmul) + (Elwise -> .. -> Elwise)
      returnType = TypePattern::kSuffixElementWise;
      break;
    default:
      break;
    }
    break;
  case TypePattern::kSuffixElementWise:
    if (patternB == TypePattern::kPureElementWise) {
      // (Matmul -> Elwise -> .. -> Elwise) + (Elwise -> .. -> Elwise)
      returnType = TypePattern::kSuffixElementWise;
    }
    break;
  default:
    break;
  }
  return static_cast<uint8_t>(returnType);
}

// [MixCV] the matmul allowed is either at the end or at the beginning
bool FusableHelper::isRestrictedByNodeType(const uint8_t &typeA,
                                           const uint8_t &typeB) const {
  // For other than mix cv, no restriction
  if (fusionKind_ != FusionKind::MixCV)
    return false;

  TypePattern patternA = static_cast<TypePattern>(typeA);
  TypePattern patternB = static_cast<TypePattern>(typeB);

  bool restricted = true;
  switch (patternA) {
  case TypePattern::kPureElementWise:
    switch (patternB) {
    case TypePattern::kPureElementWise:
      restricted = false;
      break;
    case TypePattern::kPureMatmul:
      restricted = false;
      break;
    default:
      break;
    }
    break;
  case TypePattern::kPureMatmul:
    switch (patternB) {
    case TypePattern::kPureElementWise:
      restricted = false;
      break;
    default:
      break;
    }
    break;
  case TypePattern::kPrefixElementWise:
    if (patternB == TypePattern::kPureMatmul) {
      restricted = false;
    }
    break;
  case TypePattern::kSuffixElementWise:
    if (patternB == TypePattern::kPureElementWise) {
      restricted = false;
    }
    break;
  default:
    break;
  }

  return restricted;
}

// [LastAxisPBR] This is to obtain input rank(max rank) of last axis reduce
int FusableHelper::obtainLastReduceRank(Operation *op) const {
  if (getOpPattern(op) == OpPattern::kLastAxisReduce) {
    linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);
    size_t inputRank = getMaxRank(linalgOp.getDpsInputs());

    return inputRank;
  }

  // Return magic -1 for other op
  return -1;
}

// [LastAxisPBR] This is to check whether all reduce op operate with same input
// rank
bool FusableHelper::isRestrictedByReduceRank(const int &a, const int &b) const {
  // For other than lastpbr, no restriction
  if (fusionKind_ != FusionKind::LastAxisPBR)
    return false;

  return (a >= 0 && b >= 0 && a != b);
}

// [ShallowCV] Dynamic shapes are not allowed for ShallowCV
bool FusableHelper::isRestrictedByDynamicShape(Operation *op) const {
  if (fusionKind_ != FusionKind::ShallowCV)
    return false;
  return mlir::mtfusion::util::hasDynamicShapeOperand(op);
}

OpPattern FusableHelper::getOpPattern(Operation *op) {
  if (isa<linalg::ReduceOp>(op)) {
    linalg::ReduceOp reduceOp = cast<linalg::ReduceOp>(op);
    auto dimensions = reduceOp.getDimensions();
    if (dimensions.size() != 1)
      return OpPattern::kOtherReduce;
    const auto &reduceAxis = dimensions[0];
    // TODO: Handle variadic reduce
    TypedValue<ShapedType> init =
        cast<TypedValue<ShapedType>>(*reduceOp.getInits().begin());
    decltype(reduceAxis) lastAxis = init.getType().getShape().size();
    if (reduceAxis == lastAxis)
      return OpPattern::kLastAxisReduce;
    return OpPattern::kOtherReduce;
  }
  if (isa<linalg::BroadcastOp>(op)) {
    linalg::BroadcastOp broadcastOp = cast<linalg::BroadcastOp>(op);
    auto dimensions = broadcastOp.getDimensions();
    if (dimensions.size() != 1)
      return OpPattern::kOtherBroadcast;
    const auto &broadcastAxis = dimensions[0];
    decltype(broadcastAxis) lastAxis =
        broadcastOp.getInput().getType().getShape().size();
    if (broadcastAxis == lastAxis)
      return OpPattern::kLastAxisBroadcast;
    LLVM_DEBUG(llvm::dbgs() << "infer kOtherBroadcast\n";);
    return OpPattern::kOtherBroadcast;
  }
  if (isa<linalg::MatmulOp, linalg::MatmulTransposeAOp,
          linalg::MatmulTransposeBOp>(op))
    return OpPattern::kMatmul;
  if (isMarkedAsElementwiseOp(op))
    return OpPattern::kElementWise;
  if (isa<arith::ConstantOp>(op))
    return OpPattern::kAuxiliary;
  if (isa<tensor::EmptyOp>(op))
    return OpPattern::kBuffer;
  if (isa<tensor::DimOp>(op))
    return OpPattern::kBuffer;
  if (isa<tensor::ExpandShapeOp, tensor::ReshapeOp, tensor::CollapseShapeOp,
          tensor::ExtractSliceOp>(op))
    return OpPattern::kReshape;
  return OpPattern::kOpaque;
}

bool FusableHelper::isImportantPattern(const OpPattern &pattern) {
  switch (pattern) {
  case OpPattern::kElementWise:
  case OpPattern::kLastAxisBroadcast:
  case OpPattern::kLastAxisReduce:
  case OpPattern::kOtherReduce:
  case OpPattern::kOtherBroadcast:
  case OpPattern::kMatmul:
    return true;
  case OpPattern::kReshape:
  case OpPattern::kAuxiliary:
  case OpPattern::kBuffer:
  case OpPattern::kOpaque:
    return false;
  default:
    break;
  }
  return false;
}

bool FusableHelper::isImportantPattern(Operation *op) {
  return FusableHelper::isImportantPattern(getOpPattern(op));
}

FusionKind FusableHelper::getFusionKind() const { return fusionKind_; }

bool FusableHelper::isFusable(const OpPattern &patternA,
                              const OpPattern &patternB) const {
  switch (fusionKind_) {
  case FusionKind::PureElemwise:
    return isPureElemwiseFusable(patternA, patternB);
  case FusionKind::AnyPB:
    return isAnyPBFusable(patternA, patternB);
  case FusionKind::LastAxisPBR:
    return isLastAxisPBRFusable(patternA, patternB);
  case FusionKind::ShallowCV:
    return isShallowCVFusable(patternA, patternB);
  case FusionKind::MixCV:
    return isMixCVFusable(patternA, patternB);
  default:
    llvm_unreachable("Invalid fusion mode");
    return false;
  }
} // namespace opfusion

bool FusableHelper::isPureElemwiseFusable(const OpPattern &patternA,
                                          const OpPattern &patternB) const {
  switch (patternA) {
  case OpPattern::kElementWise:
  case OpPattern::kReshape:
    switch (patternB) {
    case OpPattern::kElementWise:
    case OpPattern::kReshape:
      return true;
    default:
      return false;
    }
  default:
    return false;
  }
}

bool FusableHelper::isLastAxisPBRFusable(const OpPattern &patternA,
                                         const OpPattern &patternB) const {
  switch (patternA) {
  case OpPattern::kElementWise:
  case OpPattern::kReshape:
    switch (patternB) {
    case OpPattern::kOtherBroadcast:
    case OpPattern::kLastAxisBroadcast:
    case OpPattern::kLastAxisReduce:
    case OpPattern::kElementWise:
    case OpPattern::kReshape:
      return true;
    default:
      return false;
    }
  case OpPattern::kLastAxisReduce:
    switch (patternB) {
    case OpPattern::kOtherBroadcast:
    case OpPattern::kLastAxisBroadcast:
    case OpPattern::kElementWise:
    case OpPattern::kReshape:
      return true;
    default:
      return false;
    }
  case OpPattern::kOtherBroadcast:
  case OpPattern::kLastAxisBroadcast:
    switch (patternB) {
    case OpPattern::kElementWise:
    case OpPattern::kReshape:
    case OpPattern::kOtherBroadcast:
    case OpPattern::kLastAxisBroadcast:
      return true;
    default:
      return false;
    }
  default:
    return false;
  }
}

bool FusableHelper::isShallowCVFusable(const OpPattern &patternA,
                                       const OpPattern &patternB) const {
  switch (patternA) {
  case OpPattern::kElementWise:
  case OpPattern::kMatmul:
  case OpPattern::kLastAxisReduce:
  case OpPattern::kLastAxisBroadcast:
  case OpPattern::kOtherBroadcast:
  case OpPattern::kReshape:
    switch (patternB) {
    case OpPattern::kElementWise:
    case OpPattern::kMatmul:
    case OpPattern::kLastAxisReduce:
    case OpPattern::kLastAxisBroadcast:
    case OpPattern::kReshape:
    case OpPattern::kOtherBroadcast:
      return true;
    default:
      return false;
    }
  default:
    return false;
  }
}

bool FusableHelper::isMixCVFusable(const OpPattern &patternA,
                                   const OpPattern &patternB) const {
  switch (patternA) {
  case OpPattern::kElementWise:
  case OpPattern::kReshape:
    switch (patternB) {
    case OpPattern::kElementWise:
    case OpPattern::kMatmul:
    case OpPattern::kReshape:
      return true;
    default:
      return false;
    }
  case OpPattern::kMatmul:
    switch (patternB) {
    case OpPattern::kElementWise:
      return true;
    default:
      return false;
    }
  default:
    return false;
  }
}

bool FusableHelper::isAnyPBFusable(const OpPattern &patternA,
                                   const OpPattern &patternB) const {
  switch (patternA) {
  case OpPattern::kElementWise:
  case OpPattern::kLastAxisBroadcast:
  case OpPattern::kOtherBroadcast:
  case OpPattern::kReshape:
    switch (patternB) {
    case OpPattern::kElementWise:
    case OpPattern::kLastAxisBroadcast:
    case OpPattern::kOtherBroadcast:
    case OpPattern::kReshape:
      return true;
    default:
      return false;
    }
  default:
    return false;
  }
}

size_t FusableHelper::getMaxRank(const SmallVector<Value> &operands) {
  return std::accumulate(operands.begin(), operands.end(), 0,
                         [](const size_t &currentMax, auto nextVal) {
                           if (auto nextRank =
                                   utils::getShapeRank(nextVal.getType())) {
                             return std::max(currentMax, *nextRank);
                           }
                           return currentMax;
                         });
}

} // namespace opfusion
} // namespace mtfusion
} // namespace mlir