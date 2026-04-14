// TorqNextHL Dialect implementation

#include "torq_next/Dialect/TorqNextHL/TorqNextHLDialect.h"
#include "torq_next/Dialect/TorqNextHL/TorqNextHLOps.h"

#include "torq_next/Dialect/TorqNextHL/TorqNextHLDialect.cpp.inc"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir::syna::torq_next_hl {

namespace {
struct TorqNextHLInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;
    bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final { return true; }
};
} // namespace

void TorqNextHLDialect::initialize() {
    addInterfaces<TorqNextHLInlinerInterface>();

    addOperations<
#define GET_OP_LIST
#include "torq_next/Dialect/TorqNextHL/TorqNextHLOps.cpp.inc"
        >();

    declarePromisedInterfaces<bufferization::BufferizableOpInterface, MatMulOp>();
}

} // namespace mlir::syna::torq_next_hl
