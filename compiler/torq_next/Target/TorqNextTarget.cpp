// TorqNext HAL Target implementation

#include "TorqNextTarget.h"

#include "torq_next/Codegen/Passes.h"
#include "torq_next/Dialect/TorqNextHL/TorqNextHLDialect.h"
#include "torq_next/Dialect/TorqNextHL/TorqNextHLOps.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/compiler/Utils/ModuleUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "iree-torq-next-target"

namespace mlir::syna::torq_next {

class TorqNextTargetDevice : public iree_compiler::IREE::HAL::TargetDevice {
public:
    TorqNextTargetDevice(const TorqNextTargetOptions &options) : options_(options) {}

    IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(
        MLIRContext *context,
        const IREE::HAL::TargetRegistry &targetRegistry) const override {
        Builder b(context);
        SmallVector<NamedAttribute> configItems;
        auto configAttr = b.getDictionaryAttr(configItems);

        SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
        targetRegistry.getTargetBackend("torq_next")
            ->getDefaultExecutableTargets(context, "torq_next", configAttr,
                                          executableTargetAttrs);

        return IREE::HAL::DeviceTargetAttr::get(
            context, b.getStringAttr("torq_next"), configAttr,
            executableTargetAttrs);
    }

private:
    const TorqNextTargetOptions &options_;
};

class TorqNextTargetBackend : public IREE::HAL::TargetBackend {
public:
    TorqNextTargetBackend(const TorqNextTargetOptions &options) : options_(options) {}

    std::string getLegacyDefaultDeviceID() const override { return "torq_next"; }

    void getDefaultExecutableTargets(
        MLIRContext *context, StringRef deviceID,
        DictionaryAttr deviceConfigAttr,
        SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
        const override {
        executableTargetAttrs.push_back(getExecutableTarget(context));
    }

    IREE::HAL::ExecutableTargetAttr
    getExecutableTarget(MLIRContext *context) const {
        Builder b(context);
        SmallVector<NamedAttribute> configItems;
        return IREE::HAL::ExecutableTargetAttr::get(
            context, b.getStringAttr("torq_next"),
            b.getStringAttr("torq-next-fb"),
            b.getDictionaryAttr(configItems));
    }

    void getDependentDialects(DialectRegistry &registry) const override {
        auto loweringPass = createTorqNextLowerExecutableTargetPass();
        loweringPass->getDependentDialects(registry);
        registry.insert<IREE::Codegen::IREECodegenDialect>();
    }

    void buildConfigurationPassPipeline(
        IREE::HAL::ExecutableTargetAttr targetAttr,
        OpPassManager &passManager) override {}

    void buildTranslationPassPipeline(
        IREE::HAL::ExecutableTargetAttr targetAttr,
        OpPassManager &passManager) override {
        buildTorqNextCodegenPassPipeline(passManager);
    }

    void buildLinkingPassPipeline(OpPassManager &passManager) override {}

    LogicalResult serializeExecutable(
        const SerializationOptions &options,
        IREE::HAL::ExecutableVariantOp variantOp,
        OpBuilder &executableBuilder) override {

        SmallVector<IREE::HAL::ExecutableExportOp> exportOps =
            llvm::to_vector(variantOp.getOps<IREE::HAL::ExecutableExportOp>());

        if (exportOps.empty())
            return variantOp.emitError()
                   << "at least one hal.executable.export op is required";

        ModuleOp innerModuleOp = variantOp.getInnerModule();
        if (!innerModuleOp)
            return innerModuleOp.emitError("expected a non-empty inner module");

        // Serialize the module IR as a binary blob.
        // This is a minimal serialization - the IR itself is the "executable".
        std::string irStr;
        llvm::raw_string_ostream os(irStr);
        innerModuleOp.print(os);
        os.flush();

        // Create a binary attribute from the IR string as i8 data.
        SmallVector<int8_t> binaryData(irStr.size());
        std::memcpy(binaryData.data(), irStr.data(), irStr.size());

        auto binaryAttr = DenseIntElementsAttr::get(
            RankedTensorType::get({static_cast<int64_t>(binaryData.size())},
                                  executableBuilder.getIntegerType(8)),
            binaryData);

        auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
            variantOp.getLoc(), variantOp.getSymName(),
            variantOp.getTarget().getFormat(), binaryAttr);

        binaryOp.setMimeTypeAttr(
            executableBuilder.getStringAttr("application/x-flatbuffers"));

        return success();
    }

private:
    const TorqNextTargetOptions &options_;
};

std::shared_ptr<IREE::HAL::TargetDevice>
createTarget(const TorqNextTargetOptions &options) {
    return std::make_shared<TorqNextTargetDevice>(options);
}

std::shared_ptr<IREE::HAL::TargetBackend>
createBackend(const TorqNextTargetOptions &options) {
    return std::make_shared<TorqNextTargetBackend>(options);
}

} // namespace mlir::syna::torq_next
