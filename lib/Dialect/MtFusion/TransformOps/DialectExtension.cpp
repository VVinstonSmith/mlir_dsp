//===- DialectExtension.cpp - MtFusion transform dialect extension ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mtir/Dialect/MtFusion/TransformOps/MtFusionTransformOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class MtFusionTransformDialectExtension
    : public transform::TransformDialectExtension<
          MtFusionTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareDependentDialect<mtfusion::MtFusionDialect>();
    declareDependentDialect<linalg::LinalgDialect>();
    declareDependentDialect<func::FuncDialect>();

    declareGeneratedDialect<affine::AffineDialect>();
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<index::IndexDialect>();
    declareGeneratedDialect<linalg::LinalgDialect>();
    declareGeneratedDialect<tensor::TensorDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "mtir/Dialect/MtFusion/TransformOps/MtFusionTransformOps.cpp.inc"
        >();
  }
};
} // namespace

void mlir::mtfusion::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<MtFusionTransformDialectExtension>();
}