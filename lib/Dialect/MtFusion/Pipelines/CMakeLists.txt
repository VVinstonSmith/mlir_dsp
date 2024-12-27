add_mlir_dialect_library(MLIRMtFusionPipelines
    MtFusionPipelines.cpp

    ADDITIONAL_HEADER_DIRS
    ${MTIR_SOURCE_DIR}/mtir/Dialect/MtFusion

    # LINK_LIBS PUBLIC
)
