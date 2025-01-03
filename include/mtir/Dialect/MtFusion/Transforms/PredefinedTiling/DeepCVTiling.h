//===- DeepCVTiling.h -- Tiling for DeepCV Op ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MTIR_DIALECT_MTFUSION_TRANSFORMS_PREDEFINEDTILING_DEEPCVTILING_H
#define MTIR_DIALECT_MTFUSION_TRANSFORMS_PREDEFINEDTILING_DEEPCVTILING_H

#include "mtir/Dialect/MtFusion/Transforms/PredefinedTiling/TilingBase.h"
#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
struct LogicalResult;
class Location;
class OpBuilder;

namespace mtfusion {

class DeepCVKernelInfo : public KernelInfo {
public:
  DeepCVKernelInfo(MLIRContext *ctx)
      : KernelInfo(FusionKind::MixCV, ctx) {}
  // currently using MixCV as fusion kind.
  static bool classof(const KernelInfo *T) {
    return T->getFusionKind() == FusionKind::MixCV;
  }
};

class DeepCVTiler : public TilerBase {
public:
  explicit DeepCVTiler(func::FuncOp funcOpIn)
      : TilerBase(
            funcOpIn,
            std::make_unique<DeepCVKernelInfo>(funcOpIn->getContext()),
            std::make_unique<TilingInfo>()){};

  explicit DeepCVTiler(func::FuncOp funcOpIn, mtfusion::TilingSeq& tilingSeq)
      : TilerBase(
            funcOpIn, tilingSeq,
            std::make_unique<DeepCVKernelInfo>(funcOpIn->getContext()),
            std::make_unique<TilingInfo>()){};
  
  LogicalResult createTilingImpl(OpBuilder &opBuilder) override;

  ValueHandles matchMatmulOps(OpBuilder &opBuilder);

  void applyCanonicalization(OpBuilder &opBuilder);

};

} // namespace mtfusion
} // namespace mlir

#endif // MTIR_DIALECT_MTFUSION_TRANSFORMS_PREDEFINEDTILING_DEEPCVTILING_H