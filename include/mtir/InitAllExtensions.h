//===- InitAllExtensions.h - MLIR Extension Registration --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all mtir
// dialect extensions to the system.
//
//===----------------------------------------------------------------------===//

#ifndef MTIR_INITALLEXTENSIONS_H
#define MTIR_INITALLEXTENSIONS_H

#include "mtir/Dialect/MtFusion/TransformOps/MtFusionTransformOps.h"
#include "mlir/IR/DialectRegistry.h"

namespace mtir {

inline void registerAllExtensions(mlir::DialectRegistry &registry) {
  // Register all transform dialect extensions.
  mlir::mtfusion::registerTransformDialectExtension(registry);
}

} // namespace mtir

#endif // MTIR_INITALLEXTENSIONS_H