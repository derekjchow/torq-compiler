// TorqNext pipeline implementation

#include "Pipelines.h"

#include "mlir/Transforms/Passes.h"

namespace mlir::syna::torq_next {

void buildLinalgToTorqNextInputConversionPassPipeline(
    OpPassManager &passManager) {
    // The input conversion for torq_next is minimal - just canonicalize.
    // The actual linalg.matmul -> torq_next_hl.matmul conversion happens
    // in the codegen pipeline (TorqNextLowerExecutableTargetPass).
    passManager.addPass(createCanonicalizerPass());
}

} // namespace mlir::syna::torq_next
