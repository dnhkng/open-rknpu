# Independent elementwise Mul — 2026-09-08

12 fresh graphs, 384 inferences, 73,728 exact output bytes passed on RV1103.
The compiler generates weights and commands without RKNN or captured programs.
The existing libc-only runtime submits two Conv tasks and one 78-word elementwise
task (enable 24, interrupt mask 768), using sequence format version 3.

Supported graph: two independent 1x1 Conv branches sharing one external input,
then Mul. All tensors are fixed float32 ONNX `[1,3,8,8]`; weights and biases are
constants. Broadcasting, two external inputs, extra activations, other shapes
and arbitrary quantization at the Mul boundary are not supported.

Both branch outputs have the same conservative scale s and zero point 0.
Output scale is float32(128*s*s), zero point 0:

`output = clip(round_to_nearest_even(int32(A)*int32(B)/128), -128, 127)`

The suite covers signed, positive-only, negative-only and unequal branch weights,
constant/ramp/random inputs, and input scale/zero-point pairs (1,0), (.25,128),
(.5,255). Host boundary tests cover signed ties and saturation. This verifies
quantized arithmetic, not float-model accuracy or calibration quality.

Mesa Rocket and RK3588 TRM chapter 36 guided EW_OP_TYPE (0x4070 bit2) and
OD_BYPASS (0x4050 bit1). RV1103-specific packed fields remain intact. The full
Mul profile passes hardware tests; individual changed bits have not all been
isolated. See ../hardware_refs/README.md for sources and incompatibilities.

Generate the suite from repository root:

```sh
PYTHONPATH=src python research/build_add_suite.py --op Mul
open-rknpu compile research/mul_suite/model000.onnx --sequence -o model.bin
```

Use tests/board_api.c with the current runtime, as documented in ../add_suite/README.md.
Because board flash is limited, each source modelNNN.bin/inputNNN.u8/expectedNNN.i8
was streamed in turn to the board mul_suite directory under the names model000.bin,
input000.u8, expected000.i8, then board_api_test ran that directory with count 1.
The retained board files therefore correspond to host model011. Full results are
in ../mul_suite.log, with each original model index recorded. All source artifacts
remain on the host. All 37 host tests pass; the runtime needed no changes for Mul.
