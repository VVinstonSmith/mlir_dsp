//===- AnyPBSchedule.h -- Schedule for Any PB Op ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MTIR_DIALECT_MTFUSION_TRANSFORMS_AUTOSCHEDULE_ANYPBSCHEDULE_H
#define MTIR_DIALECT_MTFUSION_TRANSFORMS_AUTOSCHEDULE_ANYPBSCHEDULE_H

#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
struct LogicalResult;
class Location;
class OpBuilder;

namespace mtfusion {

class AnyPBKernelInfo : public KernelInfo {
public:
  AnyPBKernelInfo(MLIRContext *ctx)
      : KernelInfo(FusionKind::AnyPB, ctx) {}

  static bool classof(const KernelInfo *T) {
    return T->getFusionKind() == FusionKind::AnyPB;
  }

  /// The input value idx with the largest dimension size.
  size_t inputValueIdxWithHighestOrderDim{0};

  size_t tileableDimSize{0};
};

class AnyPBScheduler : public SchedulerBase {
public:
  explicit AnyPBScheduler(func::FuncOp funcOpIn)
      : SchedulerBase(
            funcOpIn,
            std::make_unique<AnyPBKernelInfo>(funcOpIn->getContext()),
            std::make_unique<TilingInfo>()){};

  /// Implementation of kernel analysis and verification.
  LogicalResult analyzeAndVerifyKernelImpl() override;

  TilingComputeFn calculateTilingImpl() override;
  
  LogicalResult createScheduleImpl(TilingKey key,
                                   OpBuilder &opBuilder) override;

private:
  ValueHandleFoldResults getTilingFactors(
    const AnyPBKernelInfo *anyPBInfo,
    const SmallVector<TilingData *> &tilingData,
    const ValueHandles &tilingDataHandles) const;
};

} // namespace mtfusion
} // namespace mlir

#endif // MTIR_DIALECT_MTFUSION_TRANSFORMS_AUTOSCHEDULE_ANYPBSCHEDULE_H