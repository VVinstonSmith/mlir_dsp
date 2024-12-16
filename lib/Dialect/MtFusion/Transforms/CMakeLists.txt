add_mlir_dialect_library(MLIRMtFusionTransforms
    ReorderOpsByBFS.cpp
    TensorResToOutParams.cpp
    OpFusion/FusableBlock.cpp
    OpFusion/FusableBlockAnalyzer.cpp
    OpFusion/FusableBlockOutliner.cpp
    OpFusion/FusableHelper.cpp
    InferFuncFuseKind.cpp
    SingleOpOutline.cpp

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