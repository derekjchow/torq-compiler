// TorqNextHL Ops implementation with TilingInterface

#include "torq_next/Dialect/TorqNextHL/TorqNextHLOps.h"
#include "torq_next/Dialect/TorqNextHL/TorqNextHLDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/TilingInterface.h"

using namespace mlir;
using namespace mlir::syna::torq_next_hl;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static Value getDimValue(OpBuilder &b, Location loc, Value v, int64_t dim) {
    auto shapedTy = cast<ShapedType>(v.getType());
    if (!shapedTy.isDynamicDim(dim))
        return b.create<arith::ConstantIndexOp>(loc, shapedTy.getDimSize(dim));
    if (isa<RankedTensorType>(v.getType()))
        return b.create<tensor::DimOp>(loc, v, dim);
    return b.create<memref::DimOp>(loc, v, dim);
}

static Range makeRange(OpBuilder &b, Location loc, Value v, int64_t dim) {
    return Range{b.getIndexAttr(0), getDimValue(b, loc, v, dim), b.getIndexAttr(1)};
}

/// Extract a slice from a tensor or memref value.
static Value extractSlice(OpBuilder &b, Location loc, Value v,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes,
                          ArrayRef<OpFoldResult> strides) {
    if (isa<RankedTensorType>(v.getType()))
        return b.create<tensor::ExtractSliceOp>(loc, v, offsets, sizes, strides);
    return b.create<memref::SubViewOp>(loc, cast<MemRefType>(v.getType()),
                                        v, offsets, sizes, strides);
}

//===----------------------------------------------------------------------===//
// MatMulOp TilingInterface implementation
//
// Supports arbitrary lhs/rhs/init ranks as produced by StableHLO dot_general.
// The iteration domain is built from the output shape + the contracting dim.
//
// For lhs=[B,M,K] rhs=[K,N] init=[B,M,N]:
//   Iteration domain: [B, M, N, K] (K is reduction)
//   lhs indexing:  (B, M, N, K) -> (B, M, K)  -- uses dims {0,1,3}
//   rhs indexing:  (B, M, N, K) -> (K, N)      -- uses dims {3,2}
//   init indexing: (B, M, N, K) -> (B, M, N)   -- uses dims {0,1,2}
//
// For lhs=[M,K] rhs=[K,N] init=[M,N]:
//   Iteration domain: [M, N, K]
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> MatMulOp::getLoopIteratorTypes() {
    auto initType = cast<ShapedType>(getInit().getType());
    int64_t initRank = initType.getRank();
    SmallVector<utils::IteratorType> iteratorTypes(initRank,
                                                    utils::IteratorType::parallel);
    iteratorTypes.push_back(utils::IteratorType::reduction); // K
    return iteratorTypes;
}

SmallVector<Range> MatMulOp::getIterationDomain(OpBuilder &b) {
    Location loc = getLoc();
    auto initType = cast<ShapedType>(getInit().getType());
    auto lhsType = cast<ShapedType>(getLhs().getType());
    int64_t initRank = initType.getRank();

    SmallVector<Range> domain;
    // First initRank dims are parallel (output shape)
    for (int64_t i = 0; i < initRank; ++i)
        domain.push_back(makeRange(b, loc, getInit(), i));
    // Last dim is K (reduction) from the last dim of lhs
    domain.push_back(makeRange(b, loc, getLhs(), lhsType.getRank() - 1));
    return domain;
}

FailureOr<TilingResult>
MatMulOp::getTiledImplementation(OpBuilder &b,
                                 ArrayRef<OpFoldResult> offsets,
                                 ArrayRef<OpFoldResult> sizes) {
    Location loc = getLoc();
    auto lhsType = cast<ShapedType>(getLhs().getType());
    auto rhsType = cast<ShapedType>(getRhs().getType());
    auto initType = cast<ShapedType>(getInit().getType());
    int64_t initRank = initType.getRank();
    int64_t lhsRank = lhsType.getRank();
    int64_t rhsRank = rhsType.getRank();

    // offsets/sizes: [<output_dims...>, K]
    // K is the last element
    OpFoldResult kOff = offsets.back();
    OpFoldResult kSz = sizes.back();
    // N is the last output dim
    int64_t nIdx = initRank - 1;
    auto one = b.getIndexAttr(1);

    // Build lhs slice offsets/sizes.
    // lhs shape: [...batch_dims, M, K]
    // For a 3D lhs [B,M,K] with output [B,M,N]: lhs uses output dims [0..initRank-2] + K
    SmallVector<OpFoldResult> lhsOff, lhsSz, lhsStr;
    for (int64_t i = 0; i < lhsRank - 1; ++i) {
        // Map lhs dims to output dims (all except N)
        // For 3D lhs [B,M,K], output [B,M,N]: lhs dim 0->out dim 0, lhs dim 1->out dim 1
        // For 2D lhs [M,K], output [M,N]: lhs dim 0->out dim 0
        lhsOff.push_back(offsets[i]);
        lhsSz.push_back(sizes[i]);
        lhsStr.push_back(one);
    }
    // Last lhs dim is K
    lhsOff.push_back(kOff);
    lhsSz.push_back(kSz);
    lhsStr.push_back(one);

    // Build rhs slice offsets/sizes.
    // rhs shape: [...optional_batch, K, N]
    SmallVector<OpFoldResult> rhsOff, rhsSz, rhsStr;
    if (rhsRank > 2) {
        // Batch dims: rhs batch dims map to output batch dims
        for (int64_t i = 0; i < rhsRank - 2; ++i) {
            rhsOff.push_back(offsets[i]);
            rhsSz.push_back(sizes[i]);
            rhsStr.push_back(one);
        }
    }
    // K dim
    rhsOff.push_back(kOff);
    rhsSz.push_back(kSz);
    rhsStr.push_back(one);
    // N dim
    rhsOff.push_back(offsets[nIdx]);
    rhsSz.push_back(sizes[nIdx]);
    rhsStr.push_back(one);

    // Build init slice: just the output dims (no K)
    SmallVector<OpFoldResult> initOff, initSz, initStr;
    for (int64_t i = 0; i < initRank; ++i) {
        initOff.push_back(offsets[i]);
        initSz.push_back(sizes[i]);
        initStr.push_back(one);
    }

    Value tiledLhs = extractSlice(b, loc, getLhs(), lhsOff, lhsSz, lhsStr);
    Value tiledRhs = extractSlice(b, loc, getRhs(), rhsOff, rhsSz, rhsStr);
    Value tiledInit = extractSlice(b, loc, getInit(), initOff, initSz, initStr);

    SmallVector<Type> resultTypes;
    if (hasPureTensorSemantics())
        resultTypes.push_back(tiledInit.getType());

    auto tiledOp = b.create<MatMulOp>(loc, resultTypes, tiledLhs, tiledRhs, tiledInit);

    return TilingResult{{tiledOp.getOperation()},
                        SmallVector<Value>(tiledOp->getResults())};
}

LogicalResult
MatMulOp::getResultTilePosition(OpBuilder &b, unsigned resultNumber,
                                ArrayRef<OpFoldResult> offsets,
                                ArrayRef<OpFoldResult> sizes,
                                SmallVector<OpFoldResult> &resultOffsets,
                                SmallVector<OpFoldResult> &resultSizes) {
    auto initType = cast<ShapedType>(getInit().getType());
    int64_t initRank = initType.getRank();

    // Result maps to the first initRank dims of the iteration domain
    for (int64_t i = 0; i < initRank; ++i) {
        resultOffsets.push_back(offsets[i]);
        resultSizes.push_back(sizes[i]);
    }
    return success();
}

//===----------------------------------------------------------------------===//
// DMAOp TilingInterface implementation (minimal)
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> DMAOp::getLoopIteratorTypes() {
    return {utils::IteratorType::parallel};
}

SmallVector<Range> DMAOp::getIterationDomain(OpBuilder &b) {
    Location loc = getLoc();
    int64_t transferSize = getTransferSizeBytes();
    Value size = b.create<arith::ConstantIndexOp>(loc, transferSize);
    return {Range{b.getIndexAttr(0), size, b.getIndexAttr(1)}};
}

FailureOr<TilingResult>
DMAOp::getTiledImplementation(OpBuilder &b,
                              ArrayRef<OpFoldResult> offsets,
                              ArrayRef<OpFoldResult> sizes) {
    auto tiledOp = b.create<DMAOp>(getLoc(), getSrc(), getDst(), getTransferSizeBytesAttr());
    return TilingResult{{tiledOp.getOperation()}, SmallVector<Value>{}};
}

LogicalResult
DMAOp::getResultTilePosition(OpBuilder &b, unsigned resultNumber,
                             ArrayRef<OpFoldResult> offsets,
                             ArrayRef<OpFoldResult> sizes,
                             SmallVector<OpFoldResult> &resultOffsets,
                             SmallVector<OpFoldResult> &resultSizes) {
    return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "torq_next/Dialect/TorqNextHL/TorqNextHLOps.cpp.inc"
