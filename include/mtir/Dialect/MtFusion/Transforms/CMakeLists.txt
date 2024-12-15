set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name MtFusion)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix MtFusion)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix MtFusion)
add_public_tablegen_target(MLIRMtFusionPassIncGen)

add_mlir_doc(Passes MtFusionPasses ./ -gen-pass-doc)