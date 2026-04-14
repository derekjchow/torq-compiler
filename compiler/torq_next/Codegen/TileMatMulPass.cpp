// Tile torq_next_hl.matmul for SRAM budget

#include "PassesDetail.h"

#include "torq_next/Dialect/TorqNextHL/TorqNextHLOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-next-tile-matmul"

using namespace mlir;
using namespace mlir::syna::torq_next;
namespace tnhl = mlir::syna::torq_next_hl;

namespace {

// SRAM budget: 400KB = 409600 bytes
static constexpr int64_t SRAM_BUDGET_BYTES = 400 * 1024;

/// Compute tile sizes for a matmul [M, K] x [K, N] -> [M, N] in the given
/// element type, subject to the SRAM budget.
///
/// We need: M_tile * K_tile * elemSize (A) +
///          K_tile * N_tile * elemSize (B) +
///          M_tile * N_tile * elemSize (C) <= budget
///
/// Strategy: keep M full, binary-search for K_tile and N_tile.
static void computeTileSizes(int64_t M, int64_t K, int64_t N,
                             int64_t elemSizeBytes,
                             int64_t &mTile, int64_t &kTile, int64_t &nTile) {
    mTile = M; // Keep M dimension full (typically small, e.g. 64)

    // Try powers of 2 for K and N, starting large
    for (int64_t kt = std::min(K, (int64_t)256); kt >= 32; kt /= 2) {
        for (int64_t nt = std::min(N, (int64_t)256); nt >= 32; nt /= 2) {
            int64_t aMem = mTile * kt * elemSizeBytes;
            int64_t bMem = kt * nt * elemSizeBytes;
            int64_t cMem = mTile * nt * elemSizeBytes;
            int64_t total = aMem + bMem + cMem;
            if (total <= SRAM_BUDGET_BYTES) {
                kTile = kt;
                nTile = nt;
                LLVM_DEBUG({
                    llvm::dbgs() << "Tile sizes: M=" << mTile << " K=" << kTile
                                 << " N=" << nTile << " total=" << total
                                 << " bytes\n";
                });
                return;
            }
        }
    }

    // Fallback: very small tiles
    kTile = 32;
    nTile = 32;
}

class TileMatMulPass : public TorqNextTileMatMulBase<TileMatMulPass> {
public:
    void runOnOperation() override {
        auto funcOp = getOperation();
        IRRewriter rewriter(funcOp->getContext());

        // Collect all matmul ops first to avoid invalidation issues.
        SmallVector<tnhl::MatMulOp> matmulOps;
        funcOp->walk([&](tnhl::MatMulOp op) {
            matmulOps.push_back(op);
        });

        for (auto matmulOp : matmulOps) {
            if (!matmulOp.hasPureTensorSemantics())
                continue;

            auto lhsType = cast<ShapedType>(matmulOp.getLhs().getType());
            auto initType = cast<ShapedType>(matmulOp.getInit().getType());
            int64_t initRank = initType.getRank();

            int64_t elemSize =
                lhsType.getElementType().getIntOrFloatBitWidth() / 8;

            // Get M, K, N from shapes.
            // M is second-to-last of init, N is last of init, K is last of lhs.
            int64_t M = (initRank >= 2) ? initType.getDimSize(initRank - 2) : 1;
            int64_t K = lhsType.getDimSize(lhsType.getRank() - 1);
            int64_t N = initType.getDimSize(initRank - 1);

            // Skip if all dimensions are dynamic
            if (ShapedType::isDynamic(M) || ShapedType::isDynamic(K) ||
                ShapedType::isDynamic(N))
                continue;

            // Check if the full matmul already fits
            int64_t totalBytes = (M * K + K * N + M * N) * elemSize;
            if (totalBytes <= SRAM_BUDGET_BYTES)
                continue;

            int64_t mTile, kTile, nTile;
            computeTileSizes(M, K, N, elemSize, mTile, kTile, nTile);

            // Build tile sizes for the iteration domain.
            // Iteration domain: [<output_dims...>, K]
            // For output [B,M,N]: tile sizes [0, mTile, nTile, kTile]
            // For output [M,N]: tile sizes [mTile, nTile, kTile]
            SmallVector<OpFoldResult> tileSizes;
            OpBuilder b(matmulOp);
            // Batch dims (all except last two output dims): don't tile
            for (int64_t i = 0; i < initRank - 2; ++i)
                tileSizes.push_back(b.getIndexAttr(0));
            tileSizes.push_back(b.getIndexAttr(mTile));
            tileSizes.push_back(b.getIndexAttr(nTile));
            tileSizes.push_back(b.getIndexAttr(kTile));

            // Use SCF tiling via TilingInterface.
            scf::SCFTilingOptions tilingOptions;
            tilingOptions.setTileSizes(tileSizes);

            rewriter.setInsertionPoint(matmulOp);
            FailureOr<scf::SCFTilingResult> tilingResult =
                scf::tileUsingSCF(rewriter, cast<TilingInterface>(matmulOp.getOperation()),
                                  tilingOptions);

            if (failed(tilingResult)) {
                matmulOp.emitWarning("failed to tile matmul");
                continue;
            }

            // Replace the original op with the tiled result.
            rewriter.replaceOp(matmulOp, tilingResult->replacements);
        }
    }
};

} // namespace

namespace mlir::syna::torq_next {

std::unique_ptr<InterfacePass<FunctionOpInterface>> createTileMatMulPass() {
    return std::make_unique<TileMatMulPass>();
}

} // namespace mlir::syna::torq_next
