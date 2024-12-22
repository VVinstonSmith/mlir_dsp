//===- MtFusionEnums.td - MtFusion dialect base support ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for enums used in mtfusion operations.
//
//===----------------------------------------------------------------------===//

#ifndef MTFUSION_IR_ENUMS
#define MTFUSION_IR_ENUMS

include "mlir/IR/EnumAttr.td"

// Define the function attribute enums matching the OpDSL functions.
def UnaryFn : I32EnumAttr<"UnaryFn", "", [
  I32EnumAttrCase<"relu", 0>,
  I32EnumAttrCase<"sqrt", 1>,
  I32EnumAttrCase<"rsqrt", 2>,
  I32EnumAttrCase<"rec", 3>,
  I32EnumAttrCase<"vnot", 4>,
  I32EnumAttrCase<"tanh", 5>,
  I32EnumAttrCase<"sin", 6>,
  I32EnumAttrCase<"cos", 7>,
  I32EnumAttrCase<"absi", 8>,
  I32EnumAttrCase<"erf", 9>,
  I32EnumAttrCase<"log2", 10>,
  I32EnumAttrCase<"log10", 11>,
  I32EnumAttrCase<"log1p", 12>,
  I32EnumAttrCase<"exp2", 13>,
  I32EnumAttrCase<"expm1", 14>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mtfusion";
}
def BinaryFn : I32EnumAttr<"BinaryFn", "", [
  I32EnumAttrCase<"vor", 0>,
  I32EnumAttrCase<"vand", 1>,
  I32EnumAttrCase<"vxor", 2>,
  I32EnumAttrCase<"minf", 3>,
  I32EnumAttrCase<"maxf", 4>,
  I32EnumAttrCase<"powf", 5>,
  I32EnumAttrCase<"mod",  6>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mtfusion";
}
def CompareFn : I32EnumAttr<"CompareFn", "", [
  I32EnumAttrCase<"veq", 0>,
  I32EnumAttrCase<"vne", 1>,
  I32EnumAttrCase<"vle", 2>,
  I32EnumAttrCase<"vlt", 3>,
  I32EnumAttrCase<"vge", 4>,
  I32EnumAttrCase<"vgt", 5>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mtfusion";
}
def TernaryFn : I32EnumAttr<"TernaryFn", "", [
  I32EnumAttrCase<"select", 0>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mtfusion";
}
def TypeFn : I32EnumAttr<"TypeFn", "", [
  I32EnumAttrCase<"cast_signed", 0>,
  I32EnumAttrCase<"cast_unsigned", 1>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mtfusion";
}
def RoundMode : I32EnumAttr<"RoundMode", "", [
  I32EnumAttrCase<"NORMAL", 0, "normal">,
  I32EnumAttrCase<"RINT", 1, "rint">,
  I32EnumAttrCase<"ROUND", 2, "round">,
  I32EnumAttrCase<"FLOOR", 3, "floor">,
  I32EnumAttrCase<"CEIL", 4, "ceil">,
  I32EnumAttrCase<"TRUNC", 5, "trunc">,
  I32EnumAttrCase<"ROUNDING", 6, "rounding">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mtfusion";
}

//===----------------------------------------------------------------------===//
// OpFusion Related Enums.
//===----------------------------------------------------------------------===//

class MtFusion_I32Enum<string name, string description,
                      list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mtfusion";
}

def MTFUSION_FUSION_KIND_PURE_ELEMWISE : I32EnumAttrCase<"PureElemwise", 1, "PURE_ELEMWISE">;
def MTFUSION_FUSION_KIND_ANY_PB : I32EnumAttrCase<"AnyPB", 2, "ANY_PB">;
def MTFUSION_FUSION_KIND_LAST_AXIS_PBR : I32EnumAttrCase<"LastAxisPBR", 3, "LAST_AXIS_PBR">;
def MTFUSION_FUSION_KIND_MIX_CV : I32EnumAttrCase<"MixCV", 4, "MIX_CV">;
def MTFUSION_FUSION_KIND_SHALLOW_CV : I32EnumAttrCase<"ShallowCV", 5, "SHALLOW_CV">;
def MTFUSION_FUSION_KIND_UNKNOWN : I32EnumAttrCase<"Unknown", 6, "UNKNOWN">;

def MtFusion_FusionKindEnum : 
  MtFusion_I32Enum<"FusionKind", "MtFusion fused kernel kind", [
  MTFUSION_FUSION_KIND_PURE_ELEMWISE,
  MTFUSION_FUSION_KIND_ANY_PB,
  MTFUSION_FUSION_KIND_LAST_AXIS_PBR,
  MTFUSION_FUSION_KIND_SHALLOW_CV,
  MTFUSION_FUSION_KIND_MIX_CV,
  MTFUSION_FUSION_KIND_UNKNOWN,
]> {}

def MTFUSION_FLATTEN_MODE_GREEDY : I32EnumAttrCase<"Greedy", 1>;
def MTFUSION_FLATTEN_MODE_TIDY : I32EnumAttrCase<"Tidy", 2>;

def MtFusion_FlattenModeEnum : 
  MtFusion_I32Enum<"FlattenMode", "MtFusion flatten mode", [
  MTFUSION_FLATTEN_MODE_GREEDY,
  MTFUSION_FLATTEN_MODE_TIDY,
]> {}

def MTFUSION_OUTPUT_MODE_MULTIPLE : I32EnumAttrCase<"Multiple", 1>;
def MTFUSION_OUTPUT_MODE_SINGLE : I32EnumAttrCase<"Single", 2>;
def MTFUSION_OUTPUT_MODE_SINGLE_AGGR : I32EnumAttrCase<"SingleAggressive", 3>;

def MtFusion_OutputModeEnum : 
  MtFusion_I32Enum<"OutputMode", "MtFusion Output mode", [
  MTFUSION_OUTPUT_MODE_MULTIPLE,
  MTFUSION_OUTPUT_MODE_SINGLE,
  MTFUSION_OUTPUT_MODE_SINGLE_AGGR,
]> {}

// Enums originate from HACC.

def MTFUSION_FUNC_KIND_HOST : I32EnumAttrCase<"Host", 1>;
def MTFUSION_FUNC_KIND_DEVICE : I32EnumAttrCase<"Device", 2>;

def MtFusion_FuncKindEnum : 
  MtFusion_I32Enum<"FuncKind", "MtFusion function kind", [
  MTFUSION_FUNC_KIND_HOST,
  MTFUSION_FUNC_KIND_DEVICE,
]> {}

// Enums originate from HIVM.

def DimX : I64EnumAttrCase<"DimX", 0, "x">;
def DimY : I64EnumAttrCase<"DimY", 1, "y">;
def DimZ : I64EnumAttrCase<"DimZ", 2, "z">;
def LinearDim0 : I64EnumAttrCase<"LinearDim0", 3, "linear_dim_0">;
def LinearDim1 : I64EnumAttrCase<"LinearDim1", 4, "linear_dim_1">;
def LinearDim2 : I64EnumAttrCase<"LinearDim2", 5, "linear_dim_2">;

def MappingIdEnum : I64EnumAttr<"MappingId", "Mapping ids for loop mapping", [
    DimX, DimY, DimZ, LinearDim0, LinearDim1, LinearDim2]> {
  let cppNamespace = "::mlir::mtfusion";
}


#endif // MTFUSION_IR_ENUMS