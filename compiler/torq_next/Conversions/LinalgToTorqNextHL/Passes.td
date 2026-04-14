// LinalgToTorqNextHL pass definitions

#ifndef TORQ_NEXT_LINALG_TO_TORQ_NEXT_HL_PASSES
#define TORQ_NEXT_LINALG_TO_TORQ_NEXT_HL_PASSES

include "mlir/Pass/PassBase.td"

def LinalgToTorqNextHLConversion :
    InterfacePass<"torq-next-linalg-to-torq-next-hl",
                  "mlir::FunctionOpInterface"> {

    let summary = "Convert linalg.matmul to torq_next_hl.matmul";

    let description = [{
        Pattern-matches linalg.matmul and linalg.batch_matmul operations
        and converts them to torq_next_hl.matmul operations.
    }];

    let constructor =
        "mlir::syna::torq_next::createLinalgToTorqNextHLConversionPass()";

    let dependentDialects = [
        "syna::torq_next_hl::TorqNextHLDialect",
        "::mlir::linalg::LinalgDialect"
    ];
}

#endif // TORQ_NEXT_LINALG_TO_TORQ_NEXT_HL_PASSES
