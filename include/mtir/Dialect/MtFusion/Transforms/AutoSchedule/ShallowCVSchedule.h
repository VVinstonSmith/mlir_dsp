//===- ShallowCVSchedule.h -- Schedule for ShallowCV Op ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MTIR_DIALECT_MTFUSION_TRANSFORMS_AUTOSCHEDULE_SHALLOWCVSCHEDULE_H
#define MTIR_DIALECT_MTFUSION_TRANSFORMS_AUTOSCHEDULE_SHALLOWCVSCHEDULE_H

#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
struct LogicalResult;
class Location;
class OpBuilder;

namespace mtfusion {

class ShallowCVKernelInfo : public KernelInfo {
public:
  ShallowCVKernelInfo(MLIRContext *ctx)
      : KernelInfo(FusionKind::ShallowCV, ctx) {}

  static bool classof(const KernelInfo *T) {
    return T->getFusionKind() == FusionKind::ShallowCV;
  }
};

class ShallowCVScheduler : public SchedulerBase {
public:
  explicit ShallowCVScheduler(func::FuncOp funcOpIn)
      : SchedulerBase(
            funcOpIn,
            std::make_unique<ShallowCVKernelInfo>(funcOpIn->getContext()),
            std::make_unique<TilingInfo>()){};

  TilingComputeFn calculateTilingImpl() {
    return [](KernelInfo *kernelInfo,
              ExprBuilder *opBuilder) -> TilingFnResultTy {
      return TilingFnResultTy{};
    };
  }
  
  LogicalResult createScheduleImpl(TilingKey key, OpBuilder &opBuilder) {
    return failure();                                  
  }
  
  LogicalResult runOnOperation(OpBuilder &opBuilder);
};

} // namespace mtfusion
} // namespace mlir

#endif // MTIR_DIALECT_MTFUSION_TRANSFORMS_AUTOSCHEDULE_SHALLOWCVSCHEDULE_H