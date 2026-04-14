// Insert DMA operations for DRAM<->SRAM transfers

#include "PassesDetail.h"

#include "torq_next/Dialect/TorqNextHL/TorqNextHLOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-next-insert-dma"

using namespace mlir;
using namespace mlir::syna::torq_next;
namespace tnhl = mlir::syna::torq_next_hl;

namespace {

/// Memory space constants
static constexpr int DRAM_MEMORY_SPACE = 0;
static constexpr int SRAM_MEMORY_SPACE = 1;

/// Create a contiguous memref type in the given memory space.
/// Uses default (dense, row-major) layout regardless of source layout.
static MemRefType getMemRefInSpace(MemRefType origType, int memorySpace) {
    return MemRefType::get(origType.getShape(), origType.getElementType(),
                           MemRefLayoutAttrInterface{},
                           IntegerAttr::get(
                               IntegerType::get(origType.getContext(), 64),
                               memorySpace));
}

/// Compute the size in bytes of a memref.
static int64_t getMemRefSizeBytes(MemRefType type) {
    int64_t numElements = 1;
    for (int64_t dim : type.getShape()) {
        if (ShapedType::isDynamic(dim))
            return -1;
        numElements *= dim;
    }
    return numElements * (type.getElementTypeBitWidth() / 8);
}

class InsertDMAPass : public TorqNextInsertDMABase<InsertDMAPass> {
public:
    void runOnOperation() override {
        auto funcOp = getOperation();

        // Walk all matmul ops that have been bufferized (memref semantics).
        SmallVector<tnhl::MatMulOp> matmulOps;
        funcOp->walk([&](tnhl::MatMulOp op) {
            if (!op.hasPureTensorSemantics())
                matmulOps.push_back(op);
        });

        for (auto matmulOp : matmulOps) {
            OpBuilder b(matmulOp);
            Location loc = matmulOp.getLoc();

            auto lhsType = cast<MemRefType>(matmulOp.getLhs().getType());
            auto rhsType = cast<MemRefType>(matmulOp.getRhs().getType());
            auto initType = cast<MemRefType>(matmulOp.getInit().getType());

            // Allocate SRAM buffers for each operand.
            auto sramLhsType = getMemRefInSpace(lhsType, SRAM_MEMORY_SPACE);
            auto sramRhsType = getMemRefInSpace(rhsType, SRAM_MEMORY_SPACE);
            auto sramInitType = getMemRefInSpace(initType, SRAM_MEMORY_SPACE);

            auto sramLhs = b.create<memref::AllocOp>(loc, sramLhsType);
            auto sramRhs = b.create<memref::AllocOp>(loc, sramRhsType);
            auto sramInit = b.create<memref::AllocOp>(loc, sramInitType);

            int64_t lhsSize = getMemRefSizeBytes(lhsType);
            int64_t rhsSize = getMemRefSizeBytes(rhsType);
            int64_t initSize = getMemRefSizeBytes(initType);

            // DMA: DRAM -> SRAM for inputs
            b.create<tnhl::DMAOp>(
                loc, matmulOp.getLhs(), sramLhs,
                b.getI64IntegerAttr(lhsSize > 0 ? lhsSize : 0));
            b.create<tnhl::DMAOp>(
                loc, matmulOp.getRhs(), sramRhs,
                b.getI64IntegerAttr(rhsSize > 0 ? rhsSize : 0));
            // DMA: load init (for accumulation)
            b.create<tnhl::DMAOp>(
                loc, matmulOp.getInit(), sramInit,
                b.getI64IntegerAttr(initSize > 0 ? initSize : 0));

            // Replace matmul operands with SRAM versions.
            matmulOp.getLhsMutable().assign(sramLhs);
            matmulOp.getRhsMutable().assign(sramRhs);
            matmulOp.getInitMutable().assign(sramInit);

            // After matmul, DMA: SRAM -> DRAM for output
            b.setInsertionPointAfter(matmulOp);
            b.create<tnhl::DMAOp>(
                loc, sramInit, matmulOp.getInit(),
                b.getI64IntegerAttr(initSize > 0 ? initSize : 0));

            // Dealloc SRAM buffers
            b.create<memref::DeallocOp>(loc, sramLhs);
            b.create<memref::DeallocOp>(loc, sramRhs);
            b.create<memref::DeallocOp>(loc, sramInit);
        }
    }
};

} // namespace

namespace mlir::syna::torq_next {

std::unique_ptr<InterfacePass<FunctionOpInterface>> createInsertDMAPass() {
    return std::make_unique<InsertDMAPass>();
}

} // namespace mlir::syna::torq_next
