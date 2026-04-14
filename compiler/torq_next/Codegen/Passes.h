// TorqNext Codegen passes header

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::syna::torq_next {

std::unique_ptr<OperationPass<ModuleOp>> createTorqNextLowerExecutableTargetPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createTileMatMulPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createInsertDMAPass();

std::unique_ptr<OperationPass<>> createResolveWorkgroupCountPass();

void registerCodegenTorqNextPasses();

void buildTorqNextCodegenPassPipeline(OpPassManager &variantPassManager);

} // namespace mlir::syna::torq_next
