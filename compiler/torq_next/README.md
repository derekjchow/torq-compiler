# torq_next Backend

IREE compiler backend targeting the SL2610 NPU with a `TorqNextHL` dialect (MatMul + DMA ops) and a tiling strategy fitting 400 KB of the 512 KB on-chip SRAM.

## Build

```bash
mkdir build && cd build
cmake ../torq-compiler -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
ninja torq-compile iree-opt
```

## Compile the test matmul

SigLIP Q-projection: `[1x64x768] @ [768x768]` in BF16.

```bash
IREE_OPT=./build/third_party/iree/tools/iree-opt
TORQ=./build/third_party/iree/tools/torq-compile

# StableHLO -> Linalg
$IREE_OPT --iree-stablehlo-to-iree-input \
  tests/torq_next/matmul_bf16_siglip.stablehlo.mlir \
  -o matmul.linalg.mlir

# Linalg -> vmfb (use the pre-targeted file that has hal.device.targets set)
$TORQ tests/torq_next/matmul_bf16_siglip.linalg.mlir \
  --iree-input-type=linalg-torq-next \
  --iree-hal-target-backends=torq_next \
  --mlir-disable-threading \
  -o matmul.vmfb
```
