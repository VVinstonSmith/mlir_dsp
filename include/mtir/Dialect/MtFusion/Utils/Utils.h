//===-----------------------Utils.h----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef MTIR_DIALECT_MTFUSION_UTILS_UTILS_H
#define MTIR_DIALECT_MTFUSION_UTILS_UTILS_H

#include "mtir/Dialect/MtFusion/IR/MtFusion.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

#include <optional>

namespace mlir {
namespace mtfusion {

/// Cast `src` value to the specified element type and rounding mode.
///
/// `src` can be either tensor or scalar.
/// If it's a scalar, casting is done by arith dialect ops.
/// If it's a tensor, casting is done by `mtfusion.cast` op. If `dst` is not
/// provided, the init value is a `tensor.empty` op. Otherwise, it's written
/// to `dst`.
Value castTo(OpBuilder &builder, Value src, Type targetElemType,
             mtfusion::RoundMode roundMode,
             std::optional<Value> dst = std::nullopt);

/// Create tensor.empty op with the same type as source
tensor::EmptyOp createEmptyOp(OpBuilder &builder, Location loc, Value source);

///  Create tensor.empty op with the same shape as source but with element type
///  targetElemType
tensor::EmptyOp createEmptyOpWithTargetElemType(OpBuilder &builder,
                                                Location loc, Value source,
                                                Type targetElemType);

/// Create arith index cast op to cast value `v` to index type.
/// If `isUnsigned` is true, create `arith.index_castui`, otherwise create
/// `arith.index_cast`.
Value castToIndex(Value v, OpBuilder &opBuilder, bool isUnsigned = true);

/// Create `arith.index_cast` op to cast value index-typed value `v` to type
/// `t`.
/// If `isUnsigned` is true, create `arith.index_castui`, otherwise create
/// `arith.index_cast`.
Value castIndexTo(Value v, Type t, OpBuilder &opBuilder,
                  bool isUnsigned = true);

/// Tiling related utilities
namespace tiling {

/// Caller information.
struct CallerInfo {
  func::FuncOp caller;
  /// Function called by the caller.
  func::FuncOp callee;
  /// Call sites within the caller calling callee.
  SmallVector<func::CallOp> callSites;
};

using CallSiteArgsBuilderFn = std::function<SmallVector<Value>(
    /*callSite=*/func::CallOp, OpBuilder &)>;

struct CallSiteBuilderInfo;
using CallSiteBuilderFn = std::function<LogicalResult(
    /*callSite=*/func::CallOp, OpBuilder &,
    /*newArgs=*/const SmallVector<Value> &,
    /*irMap=*/DenseMap<Operation *, Operation *> &)>;

LogicalResult callSiteBuilderFnForTilingModification(
    func::CallOp callSite, OpBuilder &opBuilder,
    const SmallVector<Value> &newArguments,
    DenseMap<Operation *, Operation *> &irMap);

/// Information needed to construct new callee.
struct CallSiteBuilderInfo {
  /// Function to create arguments for new call site.
  CallSiteArgsBuilderFn argBuilderFn;
  /// Function to create new call site.
  CallSiteBuilderFn siteBuilderFn;
};

/// Get callee's caller's information.
void getCallerInfo(func::FuncOp callee, ModuleOp enclosingModule,
                   DenseMap<func::FuncOp, CallerInfo> &info);

/// Get call site arguments that corresponds to tiling data arguments in callee.
SmallVector<Value> getCalleeTilingArguments(func::FuncOp callee,
                                            func::CallOp callSite);
/// Fix the call sites by replacing arguments.
LogicalResult doFixCallSite(CallerInfo &callerInfo,
                            CallSiteBuilderInfo &builderInfo,
                            DenseMap<Operation *, Operation *> &irMap,
                            OpBuilder &opBuilder);

/// Add calculate tiling call crosscheck with device func tiling
LogicalResult
checkCallCalcTilingWithTilingOperands(Operation *calcTilingOp,
                                      ArrayRef<Value> tilingOperands);

/// Add calculate tiling func crosscheck with device func tiling
LogicalResult verifyCalcTiling(func::FuncOp &calcFunc,
                               SmallVector<func::FuncOp> &deviceFuncs);

} // namespace tiling

namespace auto_schedule {
/// Generate payload tag from kernel name.
inline std::string getPayloadRootTag(const std::string &kernelName) {
  return kernelName + "_payload";
}

/// Generate transform tag from kernel name.
inline std::string getTransformRootTag(const std::string &kernelName) {
  return kernelName + "_transform";
}
} // namespace auto_schedule

/// Whether the operation is a `tensor.expand_shape`, `tensor.collapse_shape`.
bool isReshapeOp(Operation *op);

/// Whether the operation is a rehape op or slice op
bool isReshapeOrSliceOp(Operation *op);

Value getReshapeSource(Operation *op);
Value getReshapeResult(Operation *op);

Value getReshapeOrSliceSource(Operation *op);
Value getReshapeOrSliceResult(Operation *op);

/// Trace back use-def chain to get the original value before reshape or slice.
FailureOr<Value> traceReshapeOrSliceSingleProducer(Value input);

/// Trace back use-def chain to get the original value before reshape or slice
/// if possible. Otherwise, return the input itself.
Value traceReshapeOrSliceSingleProducerOrSelf(Value input);

/// Trace back use-def chain to get the reshape or slice operations from current
/// input value to original value.
SmallVector<Operation *> getReshapeOrSliceOpProduceTrace(Value input);

/// Trace the use-def chain to get the value after reshape or slice. The input
/// value should have only one reshape consumer.
FailureOr<Value> traceReshapeOrSliceSingleConsumer(Value input);

/// Trace the use-def chain to get the value after reshape or slice if possible.
/// Otherwise, return the input itself.
Value traceReshapeOrSliceSingleConsumerOrSelf(Value input);

/// Whitelist to filter op that supports vector-scalar pattern.
bool isHWSupportVSOp(const Operation *op);

/// Whitelist to filter op that satisfy commutative property.
bool isCommutativeOp(const Operation *op);

/// Whether is scalar-vector binary op.
template <typename SrcOp>
bool isSVOp(SrcOp op) {
  llvm::SmallVector<Value> inputs = op.getDpsInputs();
  if (inputs.size() != 2) {
    return false;
  }
  return (inputs[0].getType().isIntOrFloat() &&
          llvm::isa<ShapedType>(inputs[1].getType()));
}

void trySetFusionKind(func::FuncOp &func, const FusionKind &fusionKind);
std::optional<FusionKind> tryGetFusionKind(func::FuncOp &func);

namespace reshape_utils {

bool isInitOp(Operation *op);

bool isReshapingOp(Operation *op);

bool isSlicingOp(Operation *op);

bool isArgOp(Operation *op);

bool isStopPropagatable(Operation *op);

bool isOutOp(Operation *op);

bool isSkippableOp(Operation *op);

bool isLegalOp(Operation *op);

bool isReturnOp(Operation *op);

bool isContainerAllocator(Operation *op);

bool isElementwiseOp(Operation *op);

bool isMarkedAsElementwiseOp(Operation *op);

bool isAllParallelOp(Operation *op);

} // namespace reshape_utils

void setInsertionPointBeforeOrAfter(OpBuilder &builder, Value &value,
                                    bool isAfter);

void setInsertionPointAfterValue(OpBuilder &builder, Value &value);

void setInsertionPointBeforeValue(OpBuilder &builder, Value &value);

std::optional<int64_t>
getFuncArgTiedResultReturnIdx(BlockArgument &ba, bool &funcArgIsReshaped,
                              bool &funcResultIsReshaped);

tensor::EmptyOp createEmptyOpWithSameShape(OpBuilder &rewriter, Value operand,
                                           SmallPtrSet<Operation *, 4> &newOps,
                                           Location loc);

linalg::CopyOp createCacheRead(OpBuilder &rewriter, Value operand,
                               Location loc);

FailureOr<linalg::CopyOp> createCacheWrite(OpBuilder &rewriter, OpResult result,
                                           bool outputOnly,
                                           bool cacheWriteToOutputInit);
} // namespace mtfusion
} // namespace mlir

#endif // MTIR_DIALECT_MTFUSION_UTILS_UTILS_H