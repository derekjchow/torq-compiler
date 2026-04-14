// TorqNext HAL Target header

#pragma once

#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetDevice.h"
#include "iree/compiler/Utils/OptionUtils.h"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq_next {

struct TorqNextTargetOptions {
    void bindOptions(OptionsBinder &binder) {}
};

std::shared_ptr<IREE::HAL::TargetDevice>
createTarget(const TorqNextTargetOptions &options);

std::shared_ptr<IREE::HAL::TargetBackend>
createBackend(const TorqNextTargetOptions &options);

} // namespace mlir::syna::torq_next
