// TorqNext IREE Plugin Registration

#include "torq_next/Codegen/Passes.h"
#include "torq_next/Conversions/LinalgToTorqNextHL/Passes.h"
#include "torq_next/Dialect/TorqNextHL/BufferizationInterfaces.h"
#include "torq_next/Dialect/TorqNextHL/TorqNextHLDialect.h"
#include "torq_next/Pipelines/Pipelines.h"
#include "torq_next/Target/TorqNextTarget.h"

#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq_next {
namespace {

struct TORQNextSession
    : public PluginSession<TORQNextSession, TorqNextTargetOptions,
                           PluginActivationPolicy::DefaultActivated> {

    static void registerPasses() {
        registerCodegenTorqNextPasses();
        registerLinalgToTorqNextHLPasses();
    }

    void onRegisterDialects(DialectRegistry &registry) override {
        registry.insert<torq_next_hl::TorqNextHLDialect>();
        torq_next_hl::registerBufferizationInterfaceExternalModels(registry);
    }

    bool extendCustomInputConversionPassPipeline(
        OpPassManager &passManager, std::string_view typeMnemonic) override {

        if (typeMnemonic == "linalg-torq-next") {
            buildLinalgToTorqNextInputConversionPassPipeline(passManager);
            return true;
        }

        return false;
    }

    void populateCustomInputConversionTypes(
        StringSet<> &typeMnemonics) override {
        typeMnemonics.insert("linalg-torq-next");
    }

    void populateDetectedCustomInputConversionTypes(
        ModuleOp &module, StringSet<> &typeMnemonics) override {
        // Only activate when the target device is torq_next.
        auto targetsAttr =
            module->getAttrOfType<ArrayAttr>("hal.device.targets");
        if (!targetsAttr)
            return;

        bool hasTorqNextTarget = false;
        for (auto attr : targetsAttr) {
            if (auto deviceTarget =
                    dyn_cast<IREE::HAL::DeviceTargetAttr>(attr)) {
                if (deviceTarget.getDeviceID().getValue() == "torq_next") {
                    hasTorqNextTarget = true;
                    break;
                }
            }
        }
        if (!hasTorqNextTarget)
            return;

        auto *ctx = module.getContext();
        const Dialect *linalgDialect = ctx->getLoadedDialect("linalg");

        if (!linalgDialect)
            return;

        bool hasLinalg = false;
        module.walk([&](Operation *op) {
            if (op->getDialect() == linalgDialect)
                hasLinalg = true;
            return WalkResult::advance();
        });

        if (hasLinalg)
            typeMnemonics.insert("linalg-torq-next");
    }

    void resolveDetectedCustomInputConversionTypes(
        StringSet<> &typeMnemonics) override {
        // When torq_next is active, remove linalg-torq to avoid the torq
        // backend trying to process the same input.
        if (typeMnemonics.contains("linalg-torq-next")) {
            typeMnemonics.erase("linalg-torq");
        }
    }

    void populateHALTargetDevices(
        IREE::HAL::TargetDeviceList &targets) override {
        targets.add("torq_next",
                     [&]() { return createTarget(options); });
    }

    void populateHALTargetBackends(
        IREE::HAL::TargetBackendList &targets) override {
        targets.add("torq_next",
                     [=]() { return createBackend(options); });
    }
};

} // namespace
} // namespace mlir::syna::torq_next

IREE_DEFINE_COMPILER_OPTION_FLAGS(::mlir::syna::torq_next::TorqNextTargetOptions);

extern "C" bool iree_register_compiler_plugin_torq_next(
    mlir::iree_compiler::PluginRegistrar *registrar) {
    registrar->registerPlugin<::mlir::syna::torq_next::TORQNextSession>(
        "torq_next");
    return true;
}
