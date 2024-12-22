//===- MtFusion.h - Mt3000 Fusion dialect -------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MTIR_DIALECT_MTFUSION_IR_MTFUSION_H
#define MTIR_DIALECT_MTFUSION_IR_MTFUSION_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace mlir {
namespace mtfusion {

class MtFusionOp;

std::string generateLibraryCallName(Operation *op);

} // namespace mtfusion
} // namespace mlir

//===----------------------------------------------------------------------===//
// MtFusion Dialect
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/IR/MtFusionOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// MtFusion Enums
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/IR/MtFusionEnums.h.inc"

//===----------------------------------------------------------------------===//
// MtFusion Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mtir/Dialect/MtFusion/IR/MtFusionAttrs.h.inc"

//===----------------------------------------------------------------------===//
// MtFusion Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mtir/Dialect/MtFusion/IR/MtFusionOps.h.inc"

#define GET_OP_CLASSES
#include "mtir/Dialect/MtFusion/IR/MtFusionStructuredOps.h.inc"

#endif // MTIR_DIALECT_MTFUSION_IR_MTFUSION_H

