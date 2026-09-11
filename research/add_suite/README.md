# Independent elementwise Add — 2026-09-08

`compile --sequence` now accepts two independent 1x1 Conv branches from the same
input, followed by elementwise Add. All tensors are fixed `[1,3,8,8]` in ONNX;
weights/biases are constant float32. No broadcasting, additional activation,
different shapes or two external inputs are supported by this first profile.

**12 fresh graphs, 384 inferences, 73,728 exact output bytes pass on RV1103.**
No vendor toolkit, RKNN model or captured commands are used to compile them.
See `../add_suite.log` and the suite manifest for hardware evidence and parameters.

The compiler emits two Conv tasks plus a separate 78-word Add task in registers
0x4000/0x5000/0x8000, with enable=24 and interrupt mask=768. All three run serially.
Runtime and Python sequence validation now accept this exact additional descriptor
combination. The container version remains 3; older runtimes correctly reject it,
so rebuild the runtime before running newly compiled Add graphs.

## Quantization and arithmetic

Each Conv retains independent weights and biases, but both outputs are quantized
to the same scale `s` and zero point 0. The compiler derives `s` from conservative
branch interval estimates; this is not calibration or accuracy tuning. The Add
output uses scale `2*s` and zero point 0, so every possible int8 sum fits:

`output_byte = round_to_nearest_even((branch_a_byte + branch_b_byte) / 2)`

Rounding differs from the Conv path's final half-up conversion. The first probe
exposed this at an exact halfway value; the corrected reference subsequently
matched all 384 runs, including signed, positive-only, negative-only and strongly
unequal branch weights. Inputs include constants, ramps and random values with
scale/zero-point pairs `(1,0)`, `(0.25,128)`, `(0.5,255)`.

This profile does not yet implement arbitrary unequal input scales/zero points
at the Add boundary. The compiler explicitly makes branch scales equal. It also
does not establish float-model accuracy or performance improvements.

## Reproduction

From repository root:

```bash
PYTHONPATH=src python research/build_add_suite.py
open-rknpu compile \
  research/add_suite/model000.onnx --sequence -o model.bin
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime \
  tests/board_api.c runtime/open_rknpu.c -o research/board_api_test
```

Push the rebuilt test executable and only the suite's `modelNNN.bin`,
`inputNNN.u8`, and `expectedNNN.i8` files (198,336 bytes total, plus flash overhead)
to `/userdata/open-npu-research/add_suite`. Run:

```bash
adb shell '/userdata/open-npu-research/board_api_test /userdata/open-npu-research/add_suite 12'
```

Use the installed ADB path from investigation notes if needed. Check free flash
space first; never use RAM-backed `/tmp`. After this run, cases 001–011 were
removed from the board to recover scarce flash space; all cases remain on the host
and model000 remains staged. The updated `open-rknpu-run` is also on the board.

All 35 host tests pass. Python and C agree on acceptance of the new descriptor
and rejection of wrong masks/register counts. CLI output matches the verified
model000 binary. The rebuilt runtime also reran all 192 depthwise cases exactly.
Camera PID 283 remained running, with approximately 6 MiB RAM available.

Next operation: Mul on this same elementwise path, with its own integer arithmetic
verification rather than assuming Add's rounding/scaling behavior carries over.

## Related-hardware cross-check

Mesa Rocket and RK3588 TRM chapter 36 now guide this emitter. The Add ALU selector
and output-rounding bit were isolated in additional independently generated RV1103
experiments: 64 runs, 12,288 exact bytes. Signed ties-away-from-zero with rounding
bit 30 set and Max with ALU selector 0 both pass. Existing Add bytecode is unchanged.
See [crosswalk, sources and incompatible fields](../hardware_refs/README.md).
