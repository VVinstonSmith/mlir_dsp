//===- ShallowCVSchedule.cpp -- Auto-schedule fused kernels -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements auto schedule policy for shallow cv kernels.
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/ShallowCVSchedule.h"
#include "mtir/Dialect/MtFusion/Transforms/AutoSchedule/AutoScheduleBase.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"
// #include "mtir/Dialect/MtFusion/Transforms/Transforms.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mtfusion-shallow-cv"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] [Shallow CV] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::mtfusion;

//===----------------------------------------------------------------------===//
// ShallowCVScheduler
//===----------------------------------------------------------------------===//

LogicalResult ShallowCVScheduler::runOnOperation(OpBuilder &opBuilder) {
  func::FuncOp shallowCVFunc = getOriginalKernel();
  // Step 1: Apply pure elementwise opfusion
  MtFusionOpFusionOptions options;
  options.fusionMode = FusionKind::LastAxisPBR;
  options.alwaysInline = true;
  FailureOr<SmallVector<func::FuncOp>> outlinedFuncs =
      applyOpFusionOutline(shallowCVFunc, options);
  if (failed(outlinedFuncs))
    return shallowCVFunc->emitError("Failed to apply LastAxisPBR fusion.");

  // Step 2: Apply Schedule
  for (auto funcOp : *outlinedFuncs) {
    LDBG("Scheduling outlined func: " << *funcOp);
    if (failed(applySchedule(funcOp, opBuilder)))
      return failure();
  }
  return success();
}