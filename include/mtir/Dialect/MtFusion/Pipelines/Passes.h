//===- Passes.h - MtFusion pipeline entry points -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes of all MtFusion pipelines.
//
//===----------------------------------------------------------------------===//
#ifndef MTIR_DIALECT_MTFUSION_PIPELINES_PASSES_H
#define MTIR_DIALECT_MTFUSION_PIPELINES_PASSES_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir {
namespace mtfusion {
struct MtFusionPipelineOptions
    : public mlir::PassPipelineOptions<MtFusionPipelineOptions> {
  bool enableTritonKernelCompile{false};

  /// options for performance improvement
  bool enableAutoMultiBuffer{false};
  bool enableOpsReorder{true};
  int32_t maxHorizontalFusionSize{-1};
  int64_t maxBufferCntTuning{0};

  /// TODO : remove it after add platform info
  unsigned blockDim{1};
};

void buildMtFusionPipelines(OpPassManager &pm,
                           const MtFusionPipelineOptions &options);
 
void registerLowerMtFusionPipelines();
} // namespace mtfusion
} // namespace mlir

#endif // MTIR_DIALECT_MTFUSION_PIPELINES_PASSES_H