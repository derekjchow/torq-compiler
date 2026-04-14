set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree::target::torq_next")

# Add include paths so that #include "torq_next/..." resolves correctly.
# The parent compiler/ directory is the include root (same as torq).
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  "${CMAKE_CURRENT_BINARY_DIR}/.."
)

add_subdirectory(Dialect/TorqNextHL)
add_subdirectory(Conversions/LinalgToTorqNextHL)
add_subdirectory(Codegen)
add_subdirectory(Target)
add_subdirectory(Pipelines)

iree_cc_library(
  NAME
    registration
  SRCS
    "PluginRegistration.cpp"
  DEPS
    iree::compiler::Dialect::HAL::Target
    iree::compiler::PluginAPI
    iree::target::torq_next::Target
    iree::target::torq_next::Dialect::TorqNextHL::Dialect
    iree::target::torq_next::Codegen
    iree::target::torq_next::Conversions::LinalgToTorqNextHL
    iree::target::torq_next::Pipelines
  PUBLIC
)

iree_compiler_register_plugin(
  PLUGIN_ID
    torq_next
  TARGET
    ::registration
)
