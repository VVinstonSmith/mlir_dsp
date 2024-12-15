//===-- Passes.td - MtFusion dialect pass definition file ------*-tablegen-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MTIR_DIALECT_MTFUSION_TRANSFORMS_PASSES
#define MTIR_DIALECT_MTFUSION_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

// def MtFusionOpFusion : Pass<"mtfusion-fuse-ops", "ModuleOp"> {
//   let summary = "MtFusion Fuse operations on tensors";
//   let constructor = "mtfusion::createMtFusionOpFusionPass()";
//   let dependentDialects = [
//     "mtfusion::MtFusionDialect",
//   ];
//   let options = [
//     Option<"outputMode", "output-mode", "::mlir::mtfusion::OutputMode",
//     "mtfusion::OutputMode::Multiple",
//     "Outlined function output mode (default is multi, can also use single or single-aggr)",
//     [{::llvm::cl::values(
//         clEnumValN(mtfusion::OutputMode::Multiple, "multi",
//                    "Outlined function with multiple outputs"),
//         clEnumValN(mtfusion::OutputMode::Single, "single",
//                    "Outlined function with single output"),
//         clEnumValN(mtfusion::OutputMode::SingleAggressive, "single-aggr",
//                    "Outlined function with single output, fusing "
//                    "more aggressively by duplicating operations")
//     )}]>,
//     Option<"fusionMode", "fusion-mode", "::mlir::mtfusion::FusionKind",
//            "mtfusion::FusionKind::Unknown", "Fusion kind is determined by label">,
//     Option<"alwaysInline", "always-inline", "bool", /*default=*/"false",
//                "Enable always inline for the outline function.">,
//     Option<"moveOutToParam", "move-out-to-param", "bool", /* default */"true",
//            "Whether move the tensor out to params or not">,
//     Option<"maxHorizontalFusionSize", "max-horizontal-fusion-size", "int", /* default */"-1",
//            "Maximum horizontal (non-dependent) fusioning allowed, -1 for unlimited attempt"
//            "of horizontal fusion">
//   ];
// }

// def AutoSchedule : Pass<"mtfusion-auto-schedule", "ModuleOp"> {
//   let summary = "Auto schedule fused kernels.";
//   let constructor = "mlir::mtfusion::createMtFusionAutoSchedulePass()";
//   let dependentDialects = ["linalg::LinalgDialect",
//                            "mtfusion::MtFusionDialect",
//                            "transform::TransformDialect",
//                            "arith::ArithDialect"];
//   let options = [
//     Option<"blockDim", "block-dim", "unsigned", "1", "Number of blocks to use">,
//     Option<"enableAutoMultiBuffer", "enable-auto-multi-buffer", "bool", "false",
//            "Enable auto multi buffer">,
//     Option<"maxBufferCntTuning", "max-buffer-count-tuning", "int64_t", "0",
//            "allow maxBufferCnt tuning">,
//   ];
// }

// def RedundantCopyRemoval : Pass<"redundant-copy-removal", "func::FuncOp"> {
//   let summary = "Remove redundant copy operations";
//   let constructor = "mtfusion::createRedundantCopyRemovalPass()";
// }

// def AddFFTSAddr : Pass<"mtfusion-add-ffts-addr", "ModuleOp"> {
//   let summary = "Add ffts base address to func param and annotation";
//   let constructor = "mlir::mtfusion::createAddFFTSAddrPass()";
// }

// def GenericToNamedConversion : Pass<"convert-generic-to-named", "func::FuncOp"> {
//   let summary = "Convert linalg generic ops to linalg named ops and mtfusion named ops.";
//   let constructor = "mlir::mtfusion::createGenericToNamedConversionPass()";
//   let dependentDialects = ["linalg::LinalgDialect",
//                            "mtfusion::MtFusionDialect"];
// }

// def FlattenOps : Pass<"mtfusion-flatten-ops", "func::FuncOp"> {
//   let constructor = "mlir::mtfusion::createFlattenOpsPass()";
//   let summary = "Flatten linalg and mtfusion ops.";
//   let dependentDialects = ["linalg::LinalgDialect",
//                            "mtfusion::MtFusionDialect",
//                            "memref::MemRefDialect",
//                            "tensor::TensorDialect"];

//   let options = [
//     Option<"flattenMode", "flatten-mode", "mtfusion::FlattenMode",
//     "mtfusion::FlattenMode::Greedy",
//     "Flatten mode, tidy mode will do an analysis on the entire function",
//     [{::llvm::cl::values(
//         clEnumValN(mtfusion::FlattenMode::Greedy, "greedy",
//                    "Outlined function with greedy elemwise flatten"),
//         clEnumValN(mtfusion::FlattenMode::Tidy, "tidy",
//                    "Outlined function with analysis")
//     )}]>
//   ];
// }

// def TensorResToOutParams : Pass<"mtfusion-tensor-results-to-out-params", "ModuleOp"> {
//   let summary = "Move tensor results to function output parameters";
//   let constructor = "mtfusion::createTensorResToOutParamsPass()";
//   let options = [
//     ListOption<"includeSymbols", "include-symbols", "std::string",
//      "Comma separated list of symbols that should apply this transformation. "
//      "If empty, the default behavior is to apply transformation to all functions.">
//   ];
// }

// def MtFusionInlineBrc : Pass<"mtfusion-inline-brc", "func::FuncOp"> {
//   let summary = "Inline broadcast-like ops.";
//   let constructor = "mtfusion::createMtFusionInlineBrcPass()";
// }

// def MtFusionBufferize : Pass<"mtfusion-bufferize"> {
//   let summary = "Bufferize the mtfusion dialect";
//   let constructor = "mlir::mtfusion::createMtFusionBufferizePass()";
//   let dependentDialects = [
//     "affine::AffineDialect",
//     "bufferization::BufferizationDialect",
//     "mtfusion::MtFusionDialect",
//     "memref::MemRefDialect",
//   ];
// }

// def SingleOpOutline : Pass<"mtfusion-single-op-outline", "func::FuncOp"> {
//   let constructor = "mlir::mtfusion::createSingleOpOutlinePass()";
//   let summary = "Outline single linalg ops into kernels.";
//   let dependentDialects = ["mtfusion::MtFusionDialect"];
// }

// def SimplifyOps : Pass<"mtfusion-simplify-ops", "func::FuncOp"> {
//   let summary = "Simplify operations";
//   let constructor = "mlir::mtfusion::createSimplifyOpsPass()";
//   let dependentDialects = ["mtfusion::MtFusionDialect"];
// }

// def Normalize : Pass<"mtfusion-normalize-ops", "func::FuncOp"> {
//   let summary = "Normalize Hfusion";
//   let constructor = "mlir::mtfusion::createMtFusionNormalizeOpsPass()";
//   let dependentDialects = ["mtfusion::MtFusionDialect"];
// }

// def PackTilingData : Pass<"mtfusion-pack-tiling-data", "ModuleOp"> {
//   let summary = "Pack dynamic tiling information into a struct.";
//   let description = [{
//     Pack the tiling information into a struct.
//   }];
//   let constructor = "mlir::mtfusion::createPackTilingDataPass()";
//   let dependentDialects = [
//     "mlir::func::FuncDialect",
//     "mlir::memref::MemRefDialect",
//   ];
// }

// def ConstantizeTilingData : Pass<"mtfusion-constantize-tiling-data", "ModuleOp"> {
//   let summary = "Propagate constants between tiling and device function";
//   let description = [{
//     Propagate constants from calculate tiling to the device
//     function.
//   }];
//   let constructor = "mlir::mtfusion::createConstantizeTilingDataPass()";
//   let dependentDialects = [
//     "mlir::func::FuncDialect",
//     "mlir::memref::MemRefDialect",
//   ];
// }

// def InferFuncFusionKind : Pass<"mtfusion-infer-func-fusion-kind", "func::FuncOp"> {
//   let constructor = "mlir::mtfusion::createInferFuncFusionKind()";
//   let summary = "Infer function for fusion kind";
//   let dependentDialects = ["mtfusion::MtFusionDialect"];
// }

// def LegalizeBF16Pass : Pass<"legalize-bf16", "func::FuncOp"> {
//   let summary = "normalize BF16 to FP32.";
//   let constructor = "mlir::mtfusion::createLegalizeBF16Pass()";
//   let dependentDialects = ["linalg::LinalgDialect",
//                            "mtfusion::MtFusionDialect"];
// }

// def LegalizeBoolPass : Pass<"legalize-bool", "func::FuncOp"> {
//   let summary = "cast int8 to int1 for input and int1 to int8 for output.";
//   let constructor = "mlir::mtfusion::createLegalizeBoolPass()";
//   let dependentDialects = ["arith::ArithDialect",
//                            "mtfusion::MtFusionDialect"];
// }

def ReorderOpsByBFS : Pass<"mtfusion-reorder-ops", "func::FuncOp"> {
  let summary = "reorder the ops by bfs.";
  let constructor = "mlir::mtfusion::createReorderOpsByBFS()";
  let dependentDialects = ["linalg::LinalgDialect",
                           "mtfusion::MtFusionDialect",
                           "mlir::tensor::TensorDialect"];
}

// def DowngradeFP64CstOpPass : Pass<"downgrade-fp64", "func::FuncOp"> {
//   let summary = "downgrade fp64 constant to fp32";
//   let constructor = "mlir::mtfusion::createDowngradeFP64CstOpPass()";
//   let dependentDialects = ["arith::ArithDialect",
//                            "mtfusion::MtFusionDialect"];
// }

// def ComposeMultiReduce : Pass<"compose-multi-reduce", "func::FuncOp"> {
//   let summary = "Compose multi reduce optimization";
//   let constructor = "mlir::mtfusion::createComposeMultiReduce()";
//   let dependentDialects = [];
//   let options = [
//     Option<"maxCompose", "max-compose", "int",
//            "-1", "Maximum reduce composed into single operation, -1 is limitless">,
//     Option<"maxDistDiff", "max-dist-diff", "int",
//            "-1", "Maximum distance difference from common ancestor">,
//   ];
// }

// def DecomposeMulti : Pass<"decompose-multi", "func::FuncOp"> {
//   let summary = "Decompose multi ops into single ones";
//   let constructor = "mlir::mtfusion::createDecomposeMulti()";
//   let dependentDialects = [];
// }

// def CacheIO : Pass<"mtfusion-cache-io", "func::FuncOp"> {
//   let summary = "cache input and output argument";
//   let constructor = "::mlir::mtfusion::createCacheIO()";
//   let dependentDialects = ["linalg::LinalgDialect",
//                            "mtfusion::MtFusionDialect",
//                            "mlir::tensor::TensorDialect"];
// }

// def BubbleUpExtractSlice : Pass<"bubble-up-extract-slice", "func::FuncOp"> {
//   let summary = "Bubble up extract slice";
//   let description = [{
//     Bubble up extract slice
//   }];
//   let constructor = "::mlir::mtfusion::createBubbleUpExtractSlicePass()";
//   let dependentDialects = [];
// }

#endif // MTIR_DIALECT_MTFUSION_TRANSFORMS_PASSES