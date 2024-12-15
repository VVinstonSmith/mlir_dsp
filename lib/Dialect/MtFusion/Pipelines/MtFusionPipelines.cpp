//===- MtFusionPipelines.cpp - MtFusion pipelines -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// #include "mtir/Conversion/MtFusionToMtIVM/MtFusionToMtIVM.h"
#include "mtir/Dialect/MtFusion/Pipelines/Passes.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/Passes.h"

#include <iostream>

namespace mlir {
namespace mtfusion {

static void canonicalizationPipeline(OpPassManager &pm) {
  pm.addPass(createCSEPass());
  CanonicalizerOptions options;
  options.enableExtendedPattern = true;
  std::vector<std::string> disabledPatterns{"FoldFillWithCopy"};
  options.disabledPatterns = disabledPatterns;
  pm.addPass(createCanonicalizerPass(options));
}

// static void preProcess(OpPassManager &pm,
//                        const MtFusionPipelineOptions &options) {
//   if (options.enableTritonKernelCompile) {
//     std::cout << "createArithToMtFusionConversionPass." << std::endl; 
//     pm.addPass(createArithToMtFusionConversionPass());
//   }
//   pm.addPass(createTensorToMtFusionConversionPass());
//   canonicalizationPipeline(pm);
//   pm.nest<func::FuncOp>().addPass(createDowngradeFP64CstOpPass());
//   pm.nest<func::FuncOp>().addPass(createGenericToNamedConversionPass());
//   pm.nest<func::FuncOp>().addPass(createMtFusionNormalizeOpsPass());
//   pm.nest<func::FuncOp>().addPass(createLegalizeBF16Pass());
//   pm.nest<func::FuncOp>().addPass(createLegalizeBoolPass());
//   pm.nest<func::FuncOp>().addPass(createSimplifyOpsPass());
//   pm.nest<func::FuncOp>().addPass(createMtFusionInlineBrcPass());
//   // normalize should be called after inline-brc pass:
//   //  a) convert scalar-vector ops to vector-scalar ops
//   pm.nest<func::FuncOp>().addPass(createMtFusionNormalizeOpsPass());
//   // tensor-results-to-out-params pass:
//   //  a) requires merge consecutive extract/insert slice patterns optimization
//   //  b) should be last in the preprocess pipeline because some pass might
//   //     modify IR related to the result value
// }

// static void flattenAndFold(OpPassManager &pm) {
//   pm.nest<func::FuncOp>().addPass(createBubbleUpExtractSlicePass());
//   canonicalizationPipeline(pm);
//   pm.nest<func::FuncOp>().addPass(tensor::createPropagateReshapePass());
//   pm.nest<func::FuncOp>().addPass(createSimplifyOpsPass());
//   canonicalizationPipeline(pm);
//   FlattenOpsOptions flattenOpsOpt;
//   flattenOpsOpt.flattenMode = mtfusion::FlattenMode::Tidy;
//   pm.nest<func::FuncOp>().addPass(createFlattenOpsPass(flattenOpsOpt));
//   pm.nest<func::FuncOp>().addPass(
//       tensor::createCanonicalizeTensorReshapePass());
//   canonicalizationPipeline(pm);
//   // Pass to fold `tensor.empty` ops.
//   pm.nest<func::FuncOp>().addPass(tensor::createFoldTensorEmptyPass());
//   canonicalizationPipeline(pm);
// }

static void inferAndOutlineOp(OpPassManager &pm,
                              const MtFusionPipelineOptions &options) {
  // pm.nest<func::FuncOp>().addPass(createInferFuncFusionKind());
  MtFusionOpFusionOptions opFusionPassOption;
  opFusionPassOption.alwaysInline = false;
  opFusionPassOption.moveOutToParam = true;
  opFusionPassOption.outputMode = OutputMode::Multiple;
  opFusionPassOption.fusionMode = FusionKind::PureElemwise;
  opFusionPassOption.maxHorizontalFusionSize = options.maxHorizontalFusionSize;
  pm.addPass(createMtFusionOpFusionPass(opFusionPassOption));
  canonicalizationPipeline(pm);
  pm.nest<func::FuncOp>().addPass(createSingleOpOutlinePass());
}

// static void mtfusionTilingOptimizationPipeline(OpPassManager &pm) {
//   pm.addPass(createConstantizeTilingDataPass());
//   return;
//   canonicalizationPipeline(pm);
//   pm.addPass(createPackTilingDataPass());
//   // after tiling is all constantized and packed, try to simplify loops
//   pm.addPass(createArithToAffineConversionPass());
//   canonicalizationPipeline(pm);
//   pm.addPass(createSCFForLoopCanonicalizationPass());
//   canonicalizationPipeline(pm);
// }

static void mtfusionAutoSchedulePipeline(OpPassManager &pm,
                                        const MtFusionPipelineOptions &options) {
  // pm.nest<func::FuncOp>().addPass(createComposeMultiReduce());
  if (options.enableOpsReorder)
    pm.nest<func::FuncOp>().addPass(createReorderOpsByBFS());
  canonicalizationPipeline(pm);
  pm.addPass(createTensorResToOutParamsPass());
  
  // BEGIN AUTO SCHEDULE
  AutoScheduleOptions autoScheduleOptions;
  autoScheduleOptions.blockDim = options.blockDim;
  autoScheduleOptions.enableAutoMultiBuffer = options.enableAutoMultiBuffer;
  autoScheduleOptions.maxBufferCntTuning = options.maxBufferCntTuning;
  pm.addPass(createMtFusionAutoSchedulePass(autoScheduleOptions));

  // END AUTO SCHEDULE
  // pm.nest<func::FuncOp>().addPass(createDecomposeMulti());
  
  // Auto Schedule might generated generic ops.
  // pm.nest<func::FuncOp>().addPass(createGenericToNamedConversionPass());
  // if (options.enableOpsReorder) {
  //   canonicalizationPipeline(pm);
  //   pm.nest<func::FuncOp>().addPass(createReorderOpsByBFS());
  // }

  // mtfusionTilingOptimizationPipeline(pm);
}

static void postProcess(OpPassManager &pm) {
  // pm.nest<func::FuncOp>().addPass(createMtFusionInlineBrcPass());
  // normalize should be called after inline-brc pass:
  //  a) convert scalar-vector ops to vector-scalar ops
  // pm.nest<func::FuncOp>().addPass(createMtFusionNormalizeOpsPass());
  // will only operate on functions with ShallowCV fusion kind
  // pm.addPass(createAddFFTSAddrPass());
}

void buildMtFusionPipelines(OpPassManager &pm,
                           const MtFusionPipelineOptions &options) {
  // preProcess(pm, options);
  canonicalizationPipeline(pm);
  if (!options.enableTritonKernelCompile) {
    // flattenAndFold(pm);
    inferAndOutlineOp(pm, options);
    mtfusionAutoSchedulePipeline(pm, options);
  } else {
    pm.nest<func::FuncOp>().addPass(
        tensor::createCanonicalizeTensorReshapePass());
  }
  canonicalizationPipeline(pm);
  // postProcess(pm);
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerLowerMtFusionPipelines() {
  PassPipelineRegistration<MtFusionPipelineOptions>(
      "lower-mtfusion-pipeline", "lower mtfusion pipeline",
      [](OpPassManager &pm, const MtFusionPipelineOptions &options) {
        buildMtFusionPipelines(pm, options);
      });
}

} // namespace mtfusion
} // namespace mlir
