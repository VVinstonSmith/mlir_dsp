//===- MtIVMPasses.td - Conversion pass definition file -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MTIR_CONVERSION_PASSES_H
#define MTIR_CONVERSION_PASSES_H

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// MtFusionToMtIVM
//===----------------------------------------------------------------------===//

// def ConvertMtFusionToMtIVM : Pass<"convert-mtfusion-to-mtivm"> {
//   let summary = "Convert MtFusion ops to MtIVM ops";
//   let constructor = "mlir::createMtFusionToMtIVMConversionPass()";
//   let dependentDialects = ["linalg::LinalgDialect", "mtivm::MtIVMDialect"];
// }