//===- MtFusionTransformOps.h - MtFusion transform ops -------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MTIR_DIALECT_MTFUSION_TRANSFORMOPS_MTFUSIONTRANSFORMOPS_H
#define MTIR_DIALECT_MTFUSION_TRANSFORMOPS_MTFUSIONTRANSFORMOPS_H

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

//===----------------------------------------------------------------------===//
// MtFusion Transform Operations
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/TransformOps/MtFusionTransformOpsEnums.h.inc"

#define GET_OP_CLASSES
#include "mtir/Dialect/MtFusion/TransformOps/MtFusionTransformOps.h.inc"

namespace mlir {
namespace mtfusion {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace mtfusion
} // namespace mlir

#endif // MTIR_DIALECT_MTFUSION_TRANSFORMOPS_MTFUSIONTRANSFORMOPS_H