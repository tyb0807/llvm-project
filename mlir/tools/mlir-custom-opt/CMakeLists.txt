set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

set(LIBS
  # Core MLIR libraries
  MLIROptLib

  # Required dialects
  MLIRAMDGPUDialect
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect

  # Required transforms/passes
  MLIRAffineToStandard
  MLIRArithTransforms
  MLIRMemRefTransforms
  MLIRTransforms

  # Core infrastructure
  MLIRIR
  MLIRPass
  MLIRSupport
  )

add_mlir_tool(mlir-custom-opt
  mlir-custom-opt.cpp

  )
mlir_target_link_libraries(mlir-custom-opt PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-custom-opt)
