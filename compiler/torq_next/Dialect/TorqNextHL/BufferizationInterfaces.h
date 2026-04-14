// TorqNextHL Bufferization Interfaces header

#pragma once

#include "mlir/IR/DialectRegistry.h"

namespace mlir::syna::torq_next_hl {

void registerBufferizationInterfaceExternalModels(DialectRegistry &registry);

} // namespace mlir::syna::torq_next_hl
