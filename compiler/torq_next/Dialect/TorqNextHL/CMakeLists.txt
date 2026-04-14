iree_tablegen_library(
  NAME
    TorqNextHLDialectGen
  TD_FILE
    "TorqNextHLBase.td"
  OUTS
    --gen-dialect-decls TorqNextHLDialect.h.inc
    --gen-dialect-defs TorqNextHLDialect.cpp.inc
)

iree_tablegen_library(
  NAME
    TorqNextHLOpsGen
  TD_FILE
    "TorqNextHLOps.td"
  OUTS
    --gen-op-decls TorqNextHLOps.h.inc
    --gen-op-defs TorqNextHLOps.cpp.inc
)

iree_cc_library(
  NAME
    Dialect
  HDRS
    TorqNextHLDialect.h
    TorqNextHLOps.h
    BufferizationInterfaces.h
  SRCS
    TorqNextHLDialect.cpp
    TorqNextHLOps.cpp
    BufferizationInterfaces.cpp
  DEPS
    ::TorqNextHLDialectGen
    ::TorqNextHLOpsGen
    LLVMSupport
    MLIRIR
    MLIRArithDialect
    MLIRLinalgDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTensorDialect
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
  PUBLIC
)
