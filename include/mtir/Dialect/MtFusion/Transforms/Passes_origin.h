//===- Passes.h - MtFusion dialect pass entrypoints --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//
#ifndef MTIR_DIALECT_MtFUSION_TRANSFORMS_PASSES_H
#define MTIR_DIALECT_MtFUSION_TRANSFORMS_PASSES_H

#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace func {
class FuncOp;
} // namespace func

namespace mtfusion {
namespace opfusion {
class FusableHelper;
class FusableBlock;
using FusableBlocks = SmallVector<FusableBlock, 8>;
} // namespace opfusion

} // namespace mtfusion
} // namespace mlir

namespace mlir {

#define GEN_PASS_DECL
#include "mtir/Dialect/MtFusion/Transforms/Passes.h.inc"

namespace mtfusion {

/// Given a funcOp, try to outline fusable ops into functions with options and
/// return the outlined functions by vector
opfusion::FusableBlocks
getFusableBlocks(func::FuncOp func, opfusion::FusableHelper &fusableHelper);

/// Given a funcOp, try to outline fusable ops into functions with options and
/// return the outlined functions by vector
LogicalResult outlineFusedFuncs(func::FuncOp entryFunc,
                                const MtFusionOpFusionOptions &options,
                                SmallVector<func::FuncOp> &outlinedFuncs);

/// Create a pass to fuse operations into outlined functions.
std::unique_ptr<mlir::Pass>
createMtFusionOpFusionPass(const MtFusionOpFusionOptions &options = {});

/// Create a pass to auto schedule fused kernels.
std::unique_ptr<Pass>
createMtFusionAutoSchedulePass(const AutoScheduleOptions &options = {});

/// Create a pass to execute auto schedule sequence for the target kernel.
std::unique_ptr<Pass>
createAutoScheduleInterpreterPass(const std::string &kernelName);

/// Create a pass to erase auto schedule sequence for the target kernel.
std::unique_ptr<Pass>
createEraseAutoSchedulePass(const std::string &kernelName);

/// Create a pass to remove redundant copy
// std::unique_ptr<Pass> createRedundantCopyRemovalPass();

/// Create a pass to add ffts base address to func param and annotation
// std::unique_ptr<Pass> createAddFFTSAddrPass();

/// Create a pass to lianlg generic ops to named ops
// std::unique_ptr<Pass> createGenericToNamedConversionPass();

/// Create a pass to flatten linalg and MtFusion ops.
// std::unique_ptr<Pass>
// createFlattenOpsPass(const FlattenOpsOptions &options = {});

/// Create a pass to move output tensor results' tied init operand to function
/// parameters.
std::unique_ptr<mlir::Pass> createTensorResToOutParamsPass();

/// Create a pass to move output tensor results' tied init operand to function
/// parameters.
std::unique_ptr<Pass>
createTensorResToOutParamsPass(ArrayRef<std::string> includeSymbols);

/// Create a pass to bufferize MtFusion ops.
// std::unique_ptr<Pass> createMtFusionBufferizePass();

/// Create a pass to outline single linalg op.
std::unique_ptr<Pass> createSingleOpOutlinePass();

/// Create a pass to simplify operations.
// std::unique_ptr<Pass> createSimplifyOpsPass();

/// Create a pass to normalize operations.
// std::unique_ptr<Pass> createMtFusionNormalizeOpsPass();

/// Create a pass to inline broadcast-like op
// std::unique_ptr<Pass> createMtFusionInlineBrcPass();

/// Create a pass to pack tiling data.
std::unique_ptr<Pass> createPackTilingDataPass();

/// Create a pass to constantize tiling data.
std::unique_ptr<Pass> createConstantizeTilingDataPass();

/// Create a pass to infer func fusion kind
// std::unique_ptr<Pass> createInferFuncFusionKind();

/// Create a pass to legalize bf16 type
// std::unique_ptr<Pass> createLegalizeBF16Pass();

/// Create a pass to legalize bool
// std::unique_ptr<Pass> createLegalizeBoolPass();

/// create a pass to reorder MtFusion ops by bfs
std::unique_ptr<Pass> createReorderOpsByBFS();

/// Create a pass to downgrade FP64 constants to FP32
// std::unique_ptr<Pass> createDowngradeFP64CstOpPass();

// create compose and decompose multi reduce opt pass
// std::unique_ptr<Pass> createComposeMultiReduce();
// std::unique_ptr<Pass> createDecomposeMulti();

/// create a pass to cache io arguments
std::unique_ptr<Pass> createCacheIO();

/// Create a pass to bubble up extract slice
// std::unique_ptr<Pass> createBubbleUpExtractSlicePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mtir/Dialect/MtFusion/Transforms/Passes.h.inc"

/// Register a pass to execute auto schedule sequence for the target kernel.
void registerAutoScheduleInterpreterPass();

/// Register a pass to erase auto schedule sequence for the target kernel.
void registerEraseAutoSchedulePass();

} // namespace mtfusion
} // namespace mlir

#endif // MTIR_DIALECT_MTFUSION_TRANSFORMS_PASSES_H
