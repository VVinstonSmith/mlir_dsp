add_mlir_dialect_library(MLIRMtFusionTransforms
    ReorderOpsByBFS.cpp
    # TensorResToOutParams.cpp

    ADDITIONAL_HEADER_DIRS
    ${MTIR_SOURCE_DIR}/include/mtir/Dialect/MtFusion

    DEPENDS
    MLIRMtFusionPassIncGen

    # LINK_LIBS PUBLIC
    # MLIRIR
    # MLIRPass
    # MLIRTransforms
    # MLIRSupport
)