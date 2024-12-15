//===- MtFusionBase.td - Base definitions for hybrid fusion dialect ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MTFUSION_IR_MTFUSIONBASE
#define MTFUSION_IR_MTFUSIONBASE

include "mtir/Dialect/MtFusion/IR/MtFusionEnums.td"

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def MtFusion_Dialect : Dialect {
  let name = "mtfusion";
  let cppNamespace = "::mlir::mtfusion";
  let description = [{
    MtFusion (Mt3000 Fusion) dialect.
  }];
  let dependentDialects = [
    "linalg::LinalgDialect",
  ];
  let hasCanonicalizer = 1;
  let useDefaultAttributePrinterParser = 1;
}

// Define the function attribute enums matching the OpDSL functions.
def UnaryFnAttr : EnumAttr<MtFusion_Dialect, UnaryFn, "unary_fn"> {
  let assemblyFormat = "`<` $value `>`";
}
def BinaryFnAttr : EnumAttr<MtFusion_Dialect, BinaryFn, "binary_fn"> {
  let assemblyFormat = "`<` $value `>`";
}
def CompareFnAttr : EnumAttr<MtFusion_Dialect, CompareFn, "compare_fn"> {
  let assemblyFormat = "`<` $value `>`";
}
def TernaryFnAttr : EnumAttr<MtFusion_Dialect, TernaryFn, "ternary_fn"> {
  let assemblyFormat = "`<` $value `>`";
}
def TypeFnAttr : EnumAttr<MtFusion_Dialect, TypeFn, "type_fn"> {
  let assemblyFormat = "`<` $value `>`";
}
def RoundModeAttr : EnumAttr<MtFusion_Dialect, RoundMode, "round_mode"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // MTFUSION_IR_MTFUSIONBASE
