# Independent depthwise convolution — 2026-09-08

The public `compile --sequence` path now accepts this exact graph:

`Conv(1x1 or 3x3, stride1, same padding) -> [Relu] -> DepthwiseConv(3x3, stride1, pad1)`

External input, intermediate and output are `[1,3,8,8]` NCHW in ONNX. Both Conv
operators require constant float32 weights and biases. Depthwise group=3 and
channel multiplier=1. Inputs use affine UINT8; outputs are affine INT8. Depthwise
weights use symmetric per-channel quantization; no calibration or tuning is done.
Other shapes, channels, strides and a trailing activation are rejected.

**12 fresh models, 192 inferences, 36,864 exact output bytes pass on RV1103.**
Models include signed, positive-only, negative-only and zero-channel depthwise
weights, nonzero biases, first-layer ReLU on/off, 1x1/3x3 stems, and input scales/
zero points `(1,0)`, `(0.25,128)`, `(0.5,255)`. Inputs include constants, a corner
impulse, a ramp and deterministic random data. These graphs and commands were
never compiled by the vendor toolkit. The C API rejects wrong buffer sizes.
Evidence: `../depthwise_suite.log` and `manifest.json`.

## Lowering

The emitter in `src/open_rknpu/depthwise.py` synthesizes both programs and all
constants. It reads no captures, RKNN files or proprietary libraries. The existing
libc/direct-ioctl runtime executes two serial tasks; no runtime or format change
was needed.

For this profile, depthwise weights are tap-major, 32 bytes per spatial tap, with
each channel occupying a two-byte pair. We write `(signed_weight, 0)` using zero
weight zero-points; nonzero zero-point pair semantics remain unverified. Biases
are INT32 at table offset 0; UINT16 channel multipliers begin at offset 16, unlike
the ordinary Conv table where zero-point corrections occupy offset 16 and scales
begin at offset 24. The reference computes independent per-channel spatial sums
with input-zero-point padding, followed by the established two-stage rounding.

The captured depthwise native output allocation describes two 16-channel planes.
For our three logical channels, values occupy the first three lanes per pixel in
the first plane; the remaining allocation is padding. We reserve enough arena
space for both planes and expose only logical channels through the existing API.
Do not extrapolate this layout to wider channel counts.

## Reproduce

From repository root:

```bash
PYTHONPATH=src python research/build_depthwise_suite.py
open-rknpu compile \
  research/depthwise_suite/model000.onnx --sequence -o model.bin
PYTHONPATH=src python -m unittest discover -s tests -q
```

Transfer only `modelNNN.bin`, `inputNNN.u8`, and `expectedNNN.i8` to the board's
`/userdata/open-npu-research/depthwise_suite/` (124,416 bytes total, plus flash
allocation overhead). Do not transfer the manifest or ONNX files. Run the existing
`board_api_test /userdata/open-npu-research/depthwise_suite 12` to repeat the
hardware check. Use the full ADB path from `docs/board-access.md`/investigation
notes if necessary. Camera PID 283 remained running during the test.

All 32 host tests pass. New tests cover rejection of unsupported graphs, symmetric
weight edge cases and grouped-versus-diagonal-dense reference equivalence.
CLI output also matches the independently verified model000 executable exactly.

This is the first independently generated depthwise profile, not full depthwise
coverage. Next primitive work: the standalone elementwise Add/Mul/Sub/Max path.
