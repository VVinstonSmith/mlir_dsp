add_mlir_dialect_library(MLIRMtFusionUtils
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MTIR_SOURCE_DIR}/mtir/Dialect/MtFusion

#   LINK_LIBS PUBLIC
#   MLIRAffineDialect
#   MLIRArithDialect
#   MLIRArithUtils
#   MLIRIR
#   MLIRTensorDialect
#   MLIRValueBoundsOpInterface
)
