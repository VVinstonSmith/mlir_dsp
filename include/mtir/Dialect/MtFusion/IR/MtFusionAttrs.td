//===- MtFusionAttrs.td - MtFusion dialect base support ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for enumeration and attributes used in mt3000
// fusion operations.
//
//===----------------------------------------------------------------------===//

#ifndef MTFUSION_IR_MTFUSIONATTRS
#define MTFUSION_IR_MTFUSIONATTRS

include "mtir/Dialect/MtFusion/IR/MtFusionBase.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// MtFusion Attributes.
//===----------------------------------------------------------------------===//

class MtFusion_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<MtFusion_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

class MtFusion_I32EnumAttr<string mnemonic, MtFusion_I32Enum enumInfo>
    : EnumAttr<MtFusion_Dialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

def MtFusion_FusionKindAttr : MtFusion_Attr<"FusionKind", "fusion_kind"> {
  let parameters = (ins
    EnumParameter<MtFusion_FusionKindEnum>:$fusion_kind
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    MtFusion fused kernel kind.
  }];
}

def MtFusion_AlwaysInlineAttr : MtFusion_Attr<"AlwaysInline", "always_inline"> {
  let description = [{
    MtFusion always inline attribute for function.
  }];
}

def MtFusion_StorageAlignDimsAttr : MtFusion_Attr<"StorageAlignDims", "storage_align_dims"> {
  let description = [{
    MtFusion storage align dims.
  }];
}

def MtFusion_StorageAlignValueInByteAttr : MtFusion_Attr<"StorageAlignValueInByte", "storage_align_value_in_byte"> {
  let description = [{
    MtFusion storage align value in byte.
  }];
}

def MtFusion_MultiBufferAttr : MtFusion_Attr<"MultiBuffer", "multi_buffer"> {
  let description = [{
    MtFusion multi buffer attribute for target op.
  }];
}

def MtFusion_BlockDimAttr : MtFusion_Attr<"BlockDim", "block_dim"> {
  let description = [{
    MtFusion block dimension attribute for function.
  }];
}

def MtFusion_FFTSBaseAddressAttr : MtFusion_Attr<"FFTSBaseAddress", "ffts_base_address"> {
  let description = [{
    MtFusion ffts base address attribute for function parameter.
  }];
}

def MtFusion_FuncArgForVOpAttr : MtFusion_Attr<"FuncArgForVOp", "func_arg_for_v"> {
  let description = [{
    MtFusion Func Args For element wise ops.
  }];
}

def MtFusion_ReduceComposeAttr : MtFusion_Attr<"ReduceCompose", "reduce_composed"> {
  let description = [{
    MtFusion reduced composed.
  }];
}

// Attrs originate from HACC.

def MtFusion_FuncKindAttr : MtFusion_Attr<"FuncKind", "function_kind"> {
  let parameters = (ins
    EnumParameter<MtFusion_FuncKindEnum>:$function_kind
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    MtFusion function kind.
  }];
}

def MtFusion_EntryAttr : MtFusion_Attr<"Entry", "entry"> {
  let description = [{
    MtFusion entry attribute for function.
  }];
}

def MtFusion_TilingFuncAttr : MtFusion_Attr<"TilingFunc", "tiling_func"> {
  let description = [{
    MtFusion tiling attribute for function.
  }];
}

def MtFusion_TilingDataAttr : MtFusion_Attr<"TilingData", "tiling_data"> {
  let description = [{
    This argument serves as a tiling parameter.
  }];
}

def MtFusion_TilingKeyAttr : MtFusion_Attr<"TilingKey", "tiling_key"> {
  let description = [{
    This argument serves as the tiling key.
  }];
}

def MtFusion_TilingStructAttr : MtFusion_Attr<"TilingStruct", "tiling_struct"> {
  let description = [{
    This argument serves as a tiling struct.
  }];
}

def MtFusion_TilingAxisAttr : MtFusion_Attr<"TilingAxis", "tiling_axis"> {
  let parameters = (ins
    EnumParameter<MtFusion_AxisEnum>:$axis
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    MtFusion tiling axis.
  }];
}

def MtFusion_NthreadsAttr : MtFusion_Attr<"Nthreads", "nthreads"> {
  let parameters = (ins 
    OptionalParameter<"int64_t">:$nthreads
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    MtFusion number of threads.
  }];
}

def MtFusion_CopyMatAttr : MtFusion_Attr<"CopyMat", "copy_mat"> {
  let parameters = (ins 
    OptionalArrayRefParameter<"mlir::mtfusion::Matrix">:$matrices
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Which matrix to be copied during tiling.
  }];
}

def MtFusion_CacheAttr : MtFusion_Attr<"CopyDst", "copy_dst"> {
  let parameters = (ins
    OptionalArrayRefParameter<"mlir::mtfusion::Cache">:$caches
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Which cache space receives the copied matrix during tiling.
  }];
}

// Attrs originate from HIVM.

def MtFusion_BlockMappingAttr : MtFusion_Attr<"BlockMapping", "block_mapping", [
    DeclareAttrInterfaceMethods<DeviceMappingAttrInterface> ] > {
  let parameters = (ins
    EnumParameter<MappingIdEnum>:$block
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    MtFusion mapping blocks to cores.
  }];
}

#endif // MTFUSION_IR_MTFUSIONATTRS