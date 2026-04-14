// TorqNext pipeline registration header

#pragma once

#include "mlir/Pass/PassManager.h"

namespace mlir::syna::torq_next {

/// Build the input conversion pipeline for linalg -> torq_next
void buildLinalgToTorqNextInputConversionPassPipeline(
    OpPassManager &passManager);

} // namespace mlir::syna::torq_next
