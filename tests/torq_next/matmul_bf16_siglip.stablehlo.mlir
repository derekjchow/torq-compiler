// Test: SigLIP Q-projection matmul in BF16
// [1x64x768] @ [768x768] -> [1x64x768]
// This matches the Q/K/V projection matmul from the SigLIP vision encoder
// targeting the SL2610 NPU via torq_next backend.
//
// Compile:
//   iree-opt --iree-stablehlo-to-iree-input matmul_bf16_siglip.stablehlo.mlir -o matmul.linalg.mlir
//   torq-compile matmul.linalg.mlir --iree-input-type=linalg-torq-next --iree-hal-target-device=torq_next -o matmul.vmfb

module @siglip_q_projection {
  func.func @main(%arg0: tensor<1x64x768xbf16>, %arg1: tensor<768x768xbf16>) -> tensor<1x64x768xbf16> {
    %0 = "stablehlo.dot_general"(%arg0, %arg1) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [2],
        rhs_contracting_dimensions = [0]>,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<1x64x768xbf16>, tensor<768x768xbf16>) -> tensor<1x64x768xbf16>
    return %0 : tensor<1x64x768xbf16>
  }
}
