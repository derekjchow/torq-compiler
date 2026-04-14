// TorqNext Codegen passes detail header

#pragma once

#include "Passes.h"

#include "torq_next/Dialect/TorqNextHL/TorqNextHLDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::syna::torq_next {

#define GEN_PASS_CLASSES
#include "torq_next/Codegen/Passes.h.inc"

} // namespace mlir::syna::torq_next
