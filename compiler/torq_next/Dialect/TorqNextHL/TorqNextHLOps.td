// TorqNextHL operation definitions

#ifndef TORQ_NEXT_HL_DIALECT_IR_OPS
#define TORQ_NEXT_HL_DIALECT_IR_OPS

include "torq_next/Dialect/TorqNextHL/TorqNextHLBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===----------------------------------------------------------------------===//
// MatMul Op
//===----------------------------------------------------------------------===//

def TorqNextHL_MatMulOp : TorqNextHL_Op<"matmul", [
    DeclareOpInterfaceMethods<TilingInterface, [
      "getIterationDomain",
      "getLoopIteratorTypes",
      "getTiledImplementation",
      "getResultTilePosition"
    ]>,
    DestinationStyleOpInterface,
    MemoryEffectsOpInterface
  ]> {
  let summary = "Matrix multiplication operation";
  let description = [{
    Computes C = A @ B. Supports 2D matmul [M,K] x [K,N] -> [M,N]
    and batched 3D matmul [B,M,K] x [B,K,N] -> [B,M,N].
    Uses destination-passing style with the $init operand as the output buffer.
  }];

  let arguments = (ins
    TensorOrMemref:$lhs,
    TensorOrMemref:$rhs,
    TensorOrMemref:$init
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }

    void getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
      if (hasPureTensorSemantics()) return;
      effects.emplace_back(MemoryEffects::Read::get(), &getLhsMutable(),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getRhsMutable(),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Read::get(), &getInitMutable(),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getInitMutable(),
                           SideEffects::DefaultResource::get());
    }

    // Returns true if operating on tensors (pre-bufferization).
    bool hasPureTensorSemantics() {
      return isa<RankedTensorType>(getLhs().getType());
    }

    // Get the rank of the matmul (2 for 2D, 3 for batched).
    int64_t getRank() {
      return cast<ShapedType>(getLhs().getType()).getRank();
    }
  }];
}

//===----------------------------------------------------------------------===//
// DMA Op
//===----------------------------------------------------------------------===//

def TorqNextHL_DMAOp : TorqNextHL_Op<"dma", [
    DeclareOpInterfaceMethods<TilingInterface, [
      "getIterationDomain",
      "getLoopIteratorTypes",
      "getTiledImplementation",
      "getResultTilePosition"
    ]>,
    MemoryEffectsOpInterface
  ]> {
  let summary = "DMA data movement between DRAM and SRAM";
  let description = [{
    Transfers data between DRAM and local SRAM via DMA.
    The source and destination are memrefs in different memory spaces.
  }];

  let arguments = (ins
    AnyMemRef:$src,
    AnyMemRef:$dst,
    I64Attr:$transfer_size_bytes
  );

  let extraClassDeclaration = [{
    void getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
      effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                           SideEffects::DefaultResource::get());
    }
  }];
}

#endif // TORQ_NEXT_HL_DIALECT_IR_OPS
