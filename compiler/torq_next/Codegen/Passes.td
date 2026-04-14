// TorqNext Codegen pass definitions

#ifndef TORQ_NEXT_CODEGEN_PASSES
#define TORQ_NEXT_CODEGEN_PASSES

include "mlir/Pass/PassBase.td"

def TorqNextLowerExecutableTarget
    : Pass<"iree-torq-next-lower-executable-target-pass", "ModuleOp"> {

    let summary = "Lower the executable target to TorqNext";
    let constructor = "mlir::syna::torq_next::createTorqNextLowerExecutableTargetPass()";

    let dependentDialects = [
        "syna::torq_next_hl::TorqNextHLDialect",
        "::mlir::linalg::LinalgDialect",
        "::mlir::scf::SCFDialect",
        "::mlir::tensor::TensorDialect",
        "::mlir::memref::MemRefDialect",
        "::mlir::bufferization::BufferizationDialect",
        "::mlir::arith::ArithDialect"
    ];
}

def TorqNextTileMatMul
    : InterfacePass<"torq-next-tile-matmul", "mlir::FunctionOpInterface"> {

    let summary = "Tile torq_next_hl.matmul for SRAM budget";
    let description = [{
        Tiles matmul operations so that the working set (input tiles + output tile)
        fits within the available SRAM budget (400KB of the SL2610's 512KB SRAM).
    }];
    let constructor = "mlir::syna::torq_next::createTileMatMulPass()";

    let dependentDialects = [
        "syna::torq_next_hl::TorqNextHLDialect",
        "::mlir::scf::SCFDialect",
        "::mlir::tensor::TensorDialect",
        "::mlir::arith::ArithDialect"
    ];
}

def TorqNextInsertDMA
    : InterfacePass<"torq-next-insert-dma", "mlir::FunctionOpInterface"> {

    let summary = "Insert DMA operations for DRAM<->SRAM transfers";
    let description = [{
        After bufferization, inserts DMA operations to move data between
        DRAM and local SRAM for each tiled matmul computation.
    }];
    let constructor = "mlir::syna::torq_next::createInsertDMAPass()";

    let dependentDialects = [
        "syna::torq_next_hl::TorqNextHLDialect",
        "::mlir::memref::MemRefDialect"
    ];
}

#endif // TORQ_NEXT_CODEGEN_PASSES
