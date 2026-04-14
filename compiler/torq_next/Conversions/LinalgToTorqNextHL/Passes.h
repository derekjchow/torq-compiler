// LinalgToTorqNextHL passes header

#pragma once

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::syna::torq_next {

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLinalgToTorqNextHLConversionPass();

void registerLinalgToTorqNextHLPasses();

} // namespace mlir::syna::torq_next
