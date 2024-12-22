//===- MtFusionDialect.cpp - Implementation of MtFusion dialect and types ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mtir/Dialect/MtFusion/IR/MtFusion.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::mtfusion;

#define GET_ATTRDEF_CLASSES
#include "mtir/Dialect/MtFusion/IR/MtFusionAttrs.cpp.inc"

void mlir::mtfusion::MtFusionDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "mtir/Dialect/MtFusion/IR/MtFusionOps.cpp.inc"
        >();

    addAttributes<
#define GET_ATTRDEF_LIST
#include "mtir/Dialect/MtFusion/IR/MtFusionAttrs.cpp.inc"
        >();
}

#include "mtir/Dialect/MtFusion/IR/MtFusionEnums.cpp.inc"

#include "mtir/Dialect/MtFusion/IR/MtFusionOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Device Mapping Attributes
//===----------------------------------------------------------------------===//

int64_t BlockMappingAttr::getMappingId() const {
  return static_cast<int64_t>(getBlock());
}

bool BlockMappingAttr::isLinearMapping() const {
  return getMappingId() >= static_cast<int64_t>(MappingId::LinearDim0);
}

int64_t BlockMappingAttr::getRelativeIndex() const {
  return isLinearMapping()
             ? getMappingId() - static_cast<int64_t>(MappingId::LinearDim0)
             : getMappingId();
}