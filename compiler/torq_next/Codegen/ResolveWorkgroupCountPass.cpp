// Replace flow.dispatch.workgroup_count_from_slice with constant (1,1,1).
//
// The torq_next backend handles all tiling internally within each dispatch,
// so only a single workgroup is needed. This pass resolves the placeholder
// workgroup count op that IREE inserts during dispatch formation.

#include "torq_next/Codegen/Passes.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "torq-next-resolve-workgroup-count"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

class ResolveWorkgroupCountPass
    : public PassWrapper<ResolveWorkgroupCountPass, OperationPass<>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResolveWorkgroupCountPass)

    StringRef getArgument() const override {
        return "torq-next-resolve-workgroup-count";
    }
    StringRef getDescription() const override {
        return "Replace workgroup count placeholders with constant (1,1,1)";
    }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<arith::ArithDialect>();
    }

    void runOnOperation() override {
        getOperation()->walk([&](IREE::Flow::DispatchWorkgroupCountFromSliceOp op) {
            OpBuilder b(op);
            Location loc = op.getLoc();
            Value one = b.create<arith::ConstantIndexOp>(loc, 1);
            op.replaceAllUsesWith(ValueRange{one, one, one});
            op.erase();
        });
    }
};

} // namespace

namespace mlir::syna::torq_next {

std::unique_ptr<OperationPass<>> createResolveWorkgroupCountPass() {
    return std::make_unique<ResolveWorkgroupCountPass>();
}

} // namespace mlir::syna::torq_next
