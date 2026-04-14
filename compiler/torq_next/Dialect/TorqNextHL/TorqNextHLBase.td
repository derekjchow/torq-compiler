// TorqNextHL dialect definition

#ifndef TORQ_NEXT_HL_DIALECT_IR_BASE
#define TORQ_NEXT_HL_DIALECT_IR_BASE

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"

def TorqNextHL_Dialect : Dialect {
  let name = "torq_next_hl";
  let cppNamespace = "::mlir::syna::torq_next_hl";
  let summary = "The TorqNext high-level operator dialect.";
  let description = [{
    A dialect for expressing high-level operations targeting the torq_next backend.
    Contains MatMul and DMA operations with TilingInterface and bufferization support.
  }];
}

class TorqNextHL_Op<string mnemonic, list<Trait> traits = []> :
    Op<TorqNextHL_Dialect, mnemonic, traits>;

def TensorOrMemref :
  AnyTypeOf<[AnyMemRef, AnyRankedTensor], "", "::mlir::ShapedType">;

#endif // TORQ_NEXT_HL_DIALECT_IR_BASE
