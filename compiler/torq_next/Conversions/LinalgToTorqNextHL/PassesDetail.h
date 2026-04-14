// LinalgToTorqNextHL passes detail header

#pragma once

#include "Passes.h"

#include "torq_next/Dialect/TorqNextHL/TorqNextHLDialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::syna::torq_next {

#define GEN_PASS_CLASSES
#include "torq_next/Conversions/LinalgToTorqNextHL/Passes.h.inc"

} // namespace mlir::syna::torq_next
