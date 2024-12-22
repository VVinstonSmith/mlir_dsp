//===- MixCVSchedule.h -- Schedule for MixCV Op ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MTIR_DIALECT_MTFUSION_TRANSFORMS_AUTOSCHEDULE_MIXCVSCHEDULE_H
#define MTIR_DIALECT_MTFUSION_TRANSFORMS_AUTOSCHEDULE_MIXCVSCHEDULE_H

#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
struct LogicalResult;
class Location;
class OpBuilder;

namespace mtfusion {

class MixCVKernelInfo : public KernelInfo {
public:
  MixCVKernelInfo(MLIRContext *ctx)
      : KernelInfo(FusionKind::MixCV, ctx) {}

  static bool classof(const KernelInfo *T) {
    return T->getFusionKind() == FusionKind::MixCV;
  }
};

class MixCVScheduler : public SchedulerBase {
public:
  explicit MixCVScheduler(func::FuncOp funcOpIn)
      : SchedulerBase(
            funcOpIn,
            std::make_unique<MixCVKernelInfo>(funcOpIn->getContext()),
            std::make_unique<TilingInfo>()){};

  TilingComputeFn calculateTilingImpl() override;
  
  LogicalResult createScheduleImpl(TilingKey key,
                                   OpBuilder &opBuilder) override;

};

} // namespace mtfusion
} // namespace mlir

#endif // MTIR_DIALECT_MTFUSION_TRANSFORMS_AUTOSCHEDULE_MIXCVSCHEDULE_H