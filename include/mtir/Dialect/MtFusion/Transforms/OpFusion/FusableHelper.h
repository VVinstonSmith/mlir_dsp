//===- FusableHelper.h --------------------------------------- --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"

#include "llvm/Support/Debug.h"

#ifndef MTIR_DIALECT_MTFUSION_TRANSFORMS_OPFUSION_FUSABLEHELPER_H
#define MTIR_DIALECT_MTFUSION_TRANSFORMS_OPFUSION_FUSABLEHELPER_H

namespace mlir {
namespace mtfusion {
namespace opfusion {

enum class OpPattern : uint8_t {
  kAuxiliary = 0,
  kBuffer = 1,
  kOpaque = 2,
  kReshape = 50,
  kElementWise = 100,
  kLastAxisReduce = 101,
  kLastAxisBroadcast = 102,
  kOtherReduce = 103,
  kOtherBroadcast = 104,
  kMatmul = 105,
};

enum class TypePattern : uint8_t {
  kPureElementWise = 0,
  kPureMatmul = 1,
  kPrefixElementWise = 2,
  kSuffixElementWise = 3,
  kOpaque = 4,
};

class FusableHelper {
public:
  explicit FusableHelper(FusionKind fusionKind, bool bufferToOut = true,
                         int32_t maxHorizontalFusionSize = -1);

  bool moveOutToParam() const;
  bool maxHorizontalFusion() const;
  bool includeBuffer(Operation *op) const;
  bool includeAuxiliary(Operation *op) const;
  bool isFusable(Operation *a, Operation *b) const;
  uint8_t obtainType(Operation *op) const;
  uint8_t adjustType(const uint8_t &typeA, const uint8_t &typeB) const;
  bool isRestrictedByNodeType(const uint8_t &typeA, const uint8_t &typeB) const;
  bool isRestrictedByDynamicShape(Operation *op) const;
  int obtainLastReduceRank(Operation *op) const;
  bool isRestrictedByReduceRank(const int &a, const int &b) const;
  static OpPattern getOpPattern(Operation *op);
  static bool isImportantPattern(Operation *op);
  static bool isImportantPattern(const OpPattern &pattern);
  static bool isSingleOutlinable(Operation *op);
  static FusionKind getSingleFusionKind(Operation *op);
  FusionKind getFusionKind() const;

private:
  FusionKind fusionKind_;
  bool moveOutToParam_;
  int32_t maxHorizontalFusion_;

  bool isFusable(const OpPattern &patternA, const OpPattern &patternB) const;
  bool isPureElemwiseFusable(const OpPattern &patternA,
                             const OpPattern &patternB) const;
  bool isLastAxisPBRFusable(const OpPattern &patternA,
                            const OpPattern &patternB) const;
  bool isShallowCVFusable(const OpPattern &patternA,
                          const OpPattern &patternB) const;
  bool isMixCVFusable(const OpPattern &patternA,
                      const OpPattern &patternB) const;
  bool isAnyPBFusable(const OpPattern &patternA,
                      const OpPattern &patternB) const;

  static size_t getMaxRank(const SmallVector<Value> &operands);
};
} // namespace opfusion
} // namespace mtfusion
} // namespace mlir

#endif // MTIR_DIALECT_MTFUSION_TRANSFORMS_OPFUSION_FUSABLEHELPER_H
