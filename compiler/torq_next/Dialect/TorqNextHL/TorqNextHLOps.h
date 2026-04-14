// TorqNextHL Ops header

#pragma once

#include "torq_next/Dialect/TorqNextHL/TorqNextHLDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

#define GET_OP_CLASSES
#include "torq_next/Dialect/TorqNextHL/TorqNextHLOps.h.inc"
