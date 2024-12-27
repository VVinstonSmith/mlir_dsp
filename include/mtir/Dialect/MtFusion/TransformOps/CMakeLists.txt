set(LLVM_TARGET_DEFINITIONS MtFusionTransformOps.td)
mlir_tablegen(MtFusionTransformOps.h.inc -gen-op-decls)
mlir_tablegen(MtFusionTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRMtFusionTransformOpsIncGen)

set(LLVM_TARGET_DEFINITIONS MtFusionTransformEnums.td)
mlir_tablegen(MtFusionTransformOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(MtFusionTransformOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRMtFusionTransformEnumsIncGen)

add_mlir_doc(MtFusionTransformOps MtFusionStructuredTransformOps Dialects/ -gen-op-doc)
