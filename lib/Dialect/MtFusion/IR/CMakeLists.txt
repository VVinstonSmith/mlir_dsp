add_mlir_dialect_library(MLIRMtFusionDialect
    MtFusionOps.cpp
    MtFusionDialect.cpp

    ADDITIONAL_HEADER_DIRS
    ${MTIR_SOURCE_DIR}/include/mtir/Dialect/MtFusion

    DEPENDS
    MLIRMtFusionOpsIncGen
    MLIRMtFusionEnumsIncGen
    MLIRMtFusionAttrsIncGen
    MLIRMtFusionStructuredOpsIncGen

    LINK_COMPONENTS
    Core

    # LINK_LIBS PUBLIC   
    # MLIRAffineDialect
    # MLIRArithDialect
    # MLIRArithUtils
    # MLIRCastInterfaces
    # MLIRComplexDialect
    # MLIRDestinationStyleOpInterface
    # MLIRDialectUtils
    # MLIRIR
    # MLIRInferTypeOpInterface
    # MLIRParallelCombiningOpInterface
    # MLIRShapedOpInterfaces
    # MLIRSideEffectInterfaces
    # MLIRSupport
    # MLIRValueBoundsOpInterface
    # MLIRViewLikeInterface
)