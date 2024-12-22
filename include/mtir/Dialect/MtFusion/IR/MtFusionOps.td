//===-------- MtFusionOps.td - MtFusion op definitions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MTFUSION_IR_MTFUSIONOPS
#define MTFUSION_IR_MTFUSIONOPS

include "mtir/Dialect/MtFusion/IR/MtFusionBase.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ShapedOpInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

// Base class for MtFusion dialect ops that do not correspond to library calls.
class MtFusion_Op<string mnemonic, list<Trait> traits = []> :
    Op<MtFusion_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

def MtFusion_CastOp : MtFusion_Op<"cast", [
    DestinationStyleOpInterface,
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<CastOpInterface>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    Pure
  ]> {
  let summary = "tensor cast operation";
  let arguments = (ins 
    Variadic<AnyTensor>:$inputs, 
    Variadic<AnyTensor>:$outputs);
  let results = (outs Variadic<AnyTensor>:$results);
  let assemblyFormat = [{
    attr-dict (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    (`outs` `(` $outputs^  `:` type($outputs) `)`)?
    (`->` type($results)^)?
  }];
  let hasCanonicalizer = 1;

  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes,
      "ValueRange":$inputs, "ValueRange":$outputs,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs), [{
        SmallVector<Value, 2> operands = {inputs.begin(), inputs.end()};
        operands.append(outputs.begin(), outputs.end());
        build($_builder, $_state, resultTypes, operands, attrs);
      }]>
  ];

  let extraClassDeclaration = [{
    mlir::MutableOperandRange getDpsInitsMutable() {
      return getOutputsMutable();
    }
    mtfusion::RoundMode getRoundMode() {
      return (*this)->getAttrOfType<RoundModeAttr>(RoundModeAttr::name).getValue();
      // for(auto &attr : getAttributes()){
      //   if(auto namedAttr = dyn_cast<NamedAttribute>(attr)){
      //     if(namedAttr.getName() == "RoundMode")
      //       return namedAttr;
      //   }
      // }
    }
  }];
}

#endif // MTFUSION_IR_MTFUSIONOPS