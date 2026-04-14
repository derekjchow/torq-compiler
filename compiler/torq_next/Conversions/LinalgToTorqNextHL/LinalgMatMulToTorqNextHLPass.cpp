// Convert linalg matmul patterns -> torq_next_hl.matmul

#include "PassesDetail.h"

#include "torq_next/Dialect/TorqNextHL/TorqNextHLOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::syna::torq_next;

namespace tnhl = mlir::syna::torq_next_hl;

namespace {

/// Check if a linalg.generic body represents a matmul: mulf + addf.
static bool isMatMulBody(Region &region) {
    if (region.getBlocks().size() != 1)
        return false;
    auto &block = region.front();
    // Expected body: %mulf = arith.mulf %arg0, %arg1; %addf = arith.addf %arg2, %mulf; yield %addf
    // 3 block args (lhs elem, rhs elem, accumulator)
    if (block.getNumArguments() != 3)
        return false;

    auto ops = block.without_terminator();
    SmallVector<Operation *> bodyOps;
    for (auto &op : ops)
        bodyOps.push_back(&op);
    if (bodyOps.size() != 2)
        return false;

    if (!isa<arith::MulFOp, arith::MulIOp>(bodyOps[0]))
        return false;
    if (!isa<arith::AddFOp, arith::AddIOp>(bodyOps[1]))
        return false;

    return true;
}

/// Check if the iterator types represent a matmul-like contraction:
/// all parallel except the last which is reduction.
static bool isMatMulIteratorTypes(ArrayRef<utils::IteratorType> iterTypes) {
    if (iterTypes.empty())
        return false;
    for (size_t i = 0; i < iterTypes.size() - 1; ++i) {
        if (iterTypes[i] != utils::IteratorType::parallel)
            return false;
    }
    return iterTypes.back() == utils::IteratorType::reduction;
}

/// Pattern to convert linalg.matmul to torq_next_hl.matmul.
struct LinalgMatMulToTorqNextHL : public OpRewritePattern<linalg::MatmulOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                  PatternRewriter &rewriter) const override {
        Value lhs = op.getDpsInputs()[0];
        Value rhs = op.getDpsInputs()[1];
        Value init = op.getDpsInits()[0];
        auto resultType = op.getResult(0).getType();

        auto matmulOp = rewriter.create<tnhl::MatMulOp>(
            op.getLoc(), resultType, lhs, rhs, init);
        rewriter.replaceOp(op, matmulOp.getResult());
        return success();
    }
};

/// Pattern to convert linalg.batch_matmul to torq_next_hl.matmul.
struct LinalgBatchMatMulToTorqNextHL
    : public OpRewritePattern<linalg::BatchMatmulOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::BatchMatmulOp op,
                                  PatternRewriter &rewriter) const override {
        Value lhs = op.getDpsInputs()[0];
        Value rhs = op.getDpsInputs()[1];
        Value init = op.getDpsInits()[0];
        auto resultType = op.getResult(0).getType();

        auto matmulOp = rewriter.create<tnhl::MatMulOp>(
            op.getLoc(), resultType, lhs, rhs, init);
        rewriter.replaceOp(op, matmulOp.getResult());
        return success();
    }
};

/// Pattern to convert linalg.generic with matmul-like semantics.
/// Matches the pattern produced by StableHLO dot_general lowering:
///   linalg.generic {
///     indexing_maps = [affine_map<(b,m,n,k) -> (b,m,k)>,
///                      affine_map<(b,m,n,k) -> (k,n)>,
///                      affine_map<(b,m,n,k) -> (b,m,n)>],
///     iterator_types = ["parallel","parallel","parallel","reduction"]
///   } { mulf + addf }
struct LinalgGenericMatMulToTorqNextHL
    : public OpRewritePattern<linalg::GenericOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::GenericOp op,
                                  PatternRewriter &rewriter) const override {
        // Must have exactly 2 inputs and 1 output.
        if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
            return failure();

        // Check iterator types: all parallel except last is reduction.
        auto iterTypes = op.getIteratorTypesArray();
        if (!isMatMulIteratorTypes(iterTypes))
            return failure();

        // Check body is mulf + addf (matmul accumulation).
        if (!isMatMulBody(op.getRegion()))
            return failure();

        Value lhs = op.getDpsInputs()[0];
        Value rhs = op.getDpsInputs()[1];
        Value init = op.getDpsInits()[0];
        auto resultType = op.getResult(0).getType();

        auto matmulOp = rewriter.create<tnhl::MatMulOp>(
            op.getLoc(), resultType, lhs, rhs, init);
        rewriter.replaceOp(op, matmulOp.getResult());
        return success();
    }
};

class LinalgToTorqNextHLConversionPass
    : public LinalgToTorqNextHLConversionBase<LinalgToTorqNextHLConversionPass> {
public:
    void runOnOperation() override {
        auto funcOp = getOperation();
        MLIRContext *ctx = funcOp->getContext();

        RewritePatternSet patterns(ctx);
        patterns.add<LinalgMatMulToTorqNextHL>(ctx);
        patterns.add<LinalgBatchMatMulToTorqNextHL>(ctx);
        patterns.add<LinalgGenericMatMulToTorqNextHL>(ctx);

        if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
            return signalPassFailure();
    }
};

} // namespace

namespace mlir::syna::torq_next {

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLinalgToTorqNextHLConversionPass() {
    return std::make_unique<LinalgToTorqNextHLConversionPass>();
}

} // namespace mlir::syna::torq_next

namespace {
#define GEN_PASS_REGISTRATION
#include "torq_next/Conversions/LinalgToTorqNextHL/Passes.h.inc"
} // namespace

void mlir::syna::torq_next::registerLinalgToTorqNextHLPasses() {
    registerLinalgToTorqNextHLConversion();
}
