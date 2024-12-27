add_mlir_dialect_library(MtFusionTransformOps
    DialectExtension.cpp
    MtFusionTransformOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${MTIR_SOURCE_DIR}/include/mtir/Dialect/MtFusion/TransformOps

    DEPENDS
    MLIRMtFusionTransformOpsIncGen
    MLIRMtFusionTransformEnumsIncGen

    # LINK_LIBS PUBLIC
)