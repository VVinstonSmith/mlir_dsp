//===- PureElemwiseSchedule.h -- Schedule for Pure Elemwise Op ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MTIR_DIALECT_MTFUSION_TRANSFORMS_AUTOSCHEDULE_PUREELEMWISE_H
#define MTIR_DIALECT_MTFUSION_TRANSFORMS_AUTOSCHEDULE_PUREELEMWISE_H

#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
struct LogicalResult;
class Location;
class OpBuilder;

namespace mtfusion {

class PureElemwiseKernelInfo : public KernelInfo {
public:
  PureElemwiseKernelInfo(MLIRContext *ctx)
      : KernelInfo(FusionKind::PureElemwise, ctx) {}

  static bool classof(const KernelInfo *T) {
    return T->getFusionKind() == FusionKind::PureElemwise;
  }
};

class PureElemwiseScheduler : public SchedulerBase {
public:
  explicit PureElemwiseScheduler(func::FuncOp funcOpIn)
      : SchedulerBase(
            funcOpIn,
            std::make_unique<PureElemwiseKernelInfo>(funcOpIn->getContext()),
            std::make_unique<TilingInfo>()){};

  TilingComputeFn calculateTilingImpl() override;
  
  LogicalResult createScheduleImpl(TilingKey key,
                                   OpBuilder &opBuilder) override;
};

} // namespace mtfusion
} // namespace mlir

#endif // MTIR_DIALECT_MTFUSION_TRANSFORMS_AUTOSCHEDULE_PUREELEMWISE_H