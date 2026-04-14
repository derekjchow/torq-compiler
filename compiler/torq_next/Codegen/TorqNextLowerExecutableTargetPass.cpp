// TorqNext main lowering pass - orchestrates the codegen pipeline

#include "PassesDetail.h"

#include "torq_next/Conversions/LinalgToTorqNextHL/Passes.h"

#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-next-lower-executable-target"

using namespace mlir;
using namespace mlir::syna::torq_next;

namespace {

class TorqNextLowerExecutableTargetPass
    : public TorqNextLowerExecutableTargetBase<TorqNextLowerExecutableTargetPass> {
public:
    void runOnOperation() override {
        auto moduleOp = getOperation();

        auto pipeline = OpPassManager(moduleOp.getOperationName());

        // 1. Convert linalg.matmul -> torq_next_hl.matmul
        pipeline.addNestedPass<func::FuncOp>(
            createLinalgToTorqNextHLConversionPass());
        pipeline.addPass(createCanonicalizerPass());

        // 2. Tile matmul for SRAM budget
        pipeline.addNestedPass<func::FuncOp>(createTileMatMulPass());
        pipeline.addPass(createCanonicalizerPass());

        // 3. Bufferize (tensor -> memref)
        bufferization::OneShotBufferizationOptions bufOptions;
        bufOptions.bufferizeFunctionBoundaries = true;
        pipeline.addPass(
            bufferization::createOneShotBufferizePass(bufOptions));
        pipeline.addPass(createCanonicalizerPass());

        // 4. Insert DMA operations
        pipeline.addNestedPass<func::FuncOp>(createInsertDMAPass());
        pipeline.addPass(createCanonicalizerPass());

        if (failed(runPipeline(pipeline, moduleOp)))
            return signalPassFailure();
    }
};

} // namespace

namespace mlir::syna::torq_next {

std::unique_ptr<OperationPass<ModuleOp>>
createTorqNextLowerExecutableTargetPass() {
    return std::make_unique<TorqNextLowerExecutableTargetPass>();
}

void buildTorqNextCodegenPassPipeline(OpPassManager &variantPassManager) {
    // Resolve workgroup counts first — replaces flow.dispatch.workgroup_count_from_slice
    // with constant (1,1,1) since torq_next handles all tiling internally.
    // This pass operates on the variant which contains both the export op
    // (with the workgroup count region) and the inner module.
    variantPassManager.addPass(createResolveWorkgroupCountPass());

    OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
    modulePassManager.addPass(createTorqNextLowerExecutableTargetPass());
}

} // namespace mlir::syna::torq_next

namespace {
#define GEN_PASS_REGISTRATION
#include "torq_next/Codegen/Passes.h.inc"
} // namespace

void mlir::syna::torq_next::registerCodegenTorqNextPasses() {
    registerTorqNextLowerExecutableTarget();
    registerTorqNextTileMatMul();
    registerTorqNextInsertDMA();
}
