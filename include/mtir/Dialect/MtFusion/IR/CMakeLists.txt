
add_mlir_dialect(MtFusionOps mtfusion)
add_mlir_doc(MtFusionOps MtFusionOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS MtFusionEnums.td)
mlir_tablegen(MtFusionEnums.h.inc -gen-enum-decls)
mlir_tablegen(MtFusionEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRMtFusionEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS MtFusionAttrs.td)
mlir_tablegen(MtFusionAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(MtFusionAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRMtFusionAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS MtFusionStructuredOps.td)
mlir_tablegen(MtFusionStructuredOps.h.inc -gen-op-decls)
mlir_tablegen(MtFusionStructuredOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRMtFusionStructuredOpsIncGen)



