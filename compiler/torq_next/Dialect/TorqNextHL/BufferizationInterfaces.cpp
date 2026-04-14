// TorqNextHL Bufferization Interfaces implementation

#include "BufferizationInterfaces.h"

#include "torq_next/Dialect/TorqNextHL/TorqNextHLDialect.h"
#include "torq_next/Dialect/TorqNextHL/TorqNextHLOps.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using mlir::bufferization::AnalysisState;
using mlir::bufferization::BufferizableOpInterface;
using mlir::bufferization::BufferizationOptions;
using mlir::bufferization::replaceOpWithBufferizedValues;

namespace mlir::syna::torq_next_hl {

namespace {

template <typename OpT>
static LogicalResult bufferizeOp(
    Operation *op, RewriterBase &rewriter,
    const bufferization::BufferizationOptions &options) {

    if (!bufferization::hasTensorSemantics(op))
        return op->emitError("op has not tensor semantics");

    DestinationStyleOpInterface dstOp = cast<DestinationStyleOpInterface>(op);

    SmallVector<Value> newOperands;
    SmallVector<Value> newValues;

    for (auto &operand : op->getOpOperands()) {
        if (isa<TensorType>(operand.get().getType())) {
            FailureOr<Value> maybeBuffer =
                getBuffer(rewriter, operand.get(), options);
            if (failed(maybeBuffer))
                return op->emitError("unable to bufferize operand");

            if (dstOp.isDpsInit(&operand))
                newValues.push_back(*maybeBuffer);

            newOperands.push_back(*maybeBuffer);
        } else {
            newOperands.push_back(operand.get());
        }
    }

    rewriter.create<OpT>(op->getLoc(), TypeRange{}, newOperands, op->getAttrs());
    bufferization::replaceOpWithBufferizedValues(rewriter, op, newValues);

    return success();
}

struct MatMulOpBufferizableInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          MatMulOpBufferizableInterface, MatMulOp> {

    bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                                const AnalysisState &state) const {
        // lhs and rhs are read; init is also read (accumulation)
        return true;
    }

    bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                                 const AnalysisState &state) const {
        return cast<DestinationStyleOpInterface>(op).isDpsInit(&opOperand);
    }

    LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                            const bufferization::BufferizationOptions &options) const {
        return bufferizeOp<MatMulOp>(op, rewriter, options);
    }
};

} // namespace

void registerBufferizationInterfaceExternalModels(DialectRegistry &registry) {
    registry.addExtension(+[](MLIRContext *context, TorqNextHLDialect *dialect) {
        MatMulOp::attachInterface<MatMulOpBufferizableInterface>(*context);
    });
}

} // namespace mlir::syna::torq_next_hl
