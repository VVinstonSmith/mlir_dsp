//===- InitAllDialects.h - MLIR Dialects Registration -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all
// mtivm-specific dialects to the system.
//
//===----------------------------------------------------------------------===//

#ifndef MTIR_INITALLDIALECTS_H
#define MTIR_INITALLDIALECTS_H

#include "mtir/Dialect/MtFusion/IR/MtFusion.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#if MTIR_TORCH_CONVERSIONS_ENABLED
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#endif

namespace mtir {

/// Add all the mtivm-specific dialects to the provided registry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<mlir::mtfusion::MtFusionDialect>();
  // clang-format on

#if MTIR_TORCH_CONVERSIONS_ENABLED
  // clang-format off
  registry.insert<mlir::torch::Torch::TorchDialect,
                  mlir::torch::TorchConversion::TorchConversionDialect,
                  mlir::torch::TMTensor::TMTensorDialect>();
  // clang-format on
#endif
}

/// Append all the mtir-specific dialects to the registry contained in the
/// given context.
inline void registerAllDialects(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  mtir::registerAllDialects(registry);
  context.appendDialectRegistry(registry);
}

} // namespace mtir

#endif // MTIR_INITALLDIALECTS_H
