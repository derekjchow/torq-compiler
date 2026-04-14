// Pre-lowered test: SigLIP Q-projection matmul in BF16 targeting torq_next
// Generated from matmul_bf16_siglip.stablehlo.mlir via:
//   iree-opt --iree-stablehlo-to-iree-input matmul_bf16_siglip.stablehlo.mlir
//
// Compile:
//   torq-compile matmul_bf16_siglip.linalg.mlir \
//     --iree-input-type=linalg-torq-next \
//     --iree-hal-target-backends=torq_next \
//     --compile-to=hal \
//     -o matmul_bf16.hal.mlir

#executable_target_torq_next_fb = #hal.executable.target<"torq_next", "torq-next-fb">
#device_target_torq_next = #hal.device.target<"torq_next", [#executable_target_torq_next_fb]>
module @siglip_q_projection attributes {hal.device.targets = [#device_target_torq_next]} {
  func.func @main(%arg0: tensor<1x64x768xbf16>, %arg1: tensor<768x768xbf16>) -> tensor<1x64x768xbf16> {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<1x64x768xbf16>
    %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<1x64x768xbf16>) -> tensor<1x64x768xbf16>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                                           affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
                                           affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
                          iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%arg0, %arg1 : tensor<1x64x768xbf16>, tensor<768x768xbf16>)
      outs(%1 : tensor<1x64x768xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %3 = arith.mulf %in, %in_0 : bf16
      %4 = arith.addf %out, %3 : bf16
      linalg.yield %4 : bf16
    } -> tensor<1x64x768xbf16>
    return %2 : tensor<1x64x768xbf16>
  }
}
