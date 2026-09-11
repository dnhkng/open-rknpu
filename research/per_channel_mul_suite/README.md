# Per-channel Mul quantization (accepted)

`compile --sequence --per-channel-mul` compiles

```
Mul(input, factor)   input: [1,3,H,W] uint8-range, factor: [3,1,1] float32
```

onto the verified 1x1 depthwise profile instead of the elementwise Mul task.

## Why the lowering is needed

The verified elementwise constant Mul stores one operand scale for the whole
constant (`BS_OW_CFG` conversion at `0x4084`/`0x4088`, operand stream at
`0x5038` with stride `0x5040`). A constant such as `[.02,.35,1.9]` therefore
becomes operand bytes `[1,23,127]`: the small channel keeps only 1 of 127 levels.
The verified depthwise profile stores a native per-output-channel weight scale
(`channel_multipliers` in the bias block) plus one output conversion, so the same
product keeps `[127,127,127]` operand bytes while each channel is quantized on its
own grid. Bias is zero, so the arithmetic is unchanged.

The consumer of a per-channel Mul **output** conversion (`BS_OW_CFG.OW_SRC=1` plus
a `0x5020` operand block) was probed first and hangs the NPU; a corrected re-probe
(2026-09-11) with the block at the address `0x5020` names, in the decoded Conv block
format, still hangs, and with `OD_BYPASS=1` the elementwise task ignores the block
completely. The retained experiment is
[`../mul_per_channel_ow_suite/`](../mul_per_channel_ow_suite/). The accepted path
therefore keeps one per-tensor output scale.

## Generation

```
PYTHONPATH=src python research/build_per_channel_mul_suite.py
```

`compile_sequence(path, per_channel_mul=True)` lowers each model to one 1x1 Conv
stem plus one 1x1 depthwise task; expected bytes come from
`depthwise_reference(reference(x, q1), q2, q1.output_zero_point)` over the
identity stem. Every model is generated independently (jittered constant,
geometry varied); no captured payload or vendor artifact is read.

## Board result — 12/12 models, 192/192 exact inferences, 23,232 bytes

```
PYTHONPATH=src python research/run_profile_suite.py per_channel_mul_suite
```

`board_results_0.json` records the per-model harness output; each line is
`PASS: 1 models, 16 inferences, <bytes> output bytes; invalid buffer sizes rejected`.

| Model | Pattern | HxW | Constant | shared-scale max err | per-channel max err |
| --- | --- | --- | --- | --- | --- |
| 000 | mild | 5x5 | [1.186, 1.4887, 1.4777] | 2.600 | 0.746 |
| 001 | wide | 5x8 | [0.0219, 0.3751, 1.7793] | 2.989 | 0.890 |
| 002 | extreme | 6x7 | [0.0047, 0.2985, 2.2200] | 3.365 | 1.111 |
| 003 | mixed_sign | 8x8 | [-0.3785, 0.8418, -2.2518] | 4.250 | 1.542 |
| 004 | small_all | 7x5 | [0.0101, 0.0183, 0.0327] | 0.051 | 0.016 |
| 005 | zero_channel | 6x6 | [0.0, 0.5382, -1.5653] | 2.861 | 1.046 |
| 006 | mild | 5x5 | [1.2782, 1.4195, 1.5868] | 3.875 | 0.801 |
| 007 | wide | 5x8 | [0.0216, 0.3670, 1.9135] | 3.263 | 0.961 |
| 008 | extreme | 6x7 | [0.0054, 0.3198, 2.4692] | 4.549 | 1.244 |
| 009 | mixed_sign | 8x8 | [-0.3833, 0.8305, -2.2369] | 3.391 | 1.541 |
| 010 | small_all | 7x5 | [0.0092, 0.0190, 0.0309] | 0.056 | 0.016 |
| 011 | zero_channel | 6x6 | [0.0, 0.5003, -1.4006] | 2.356 | 0.912 |

Errors are float-domain maxima over the 16 board inputs: the exact product
`uint8 * factor` compared with each path's dequantized INT8 output. The
per-channel lowering wins on every model by 2.2x-4.7x (worst case 4.549 -> 1.244).
The residual is the single per-tensor output grid, not the constant.

## Bounds

* Single batch, C=3 input, H/W 5..8 (the verified depthwise RGB window).
* Constant must be spatially invariant with at least two distinct channel values;
  scalar, spatial and out-of-window constants raise `ValueError`.
* Output grid stays one per-tensor scale; per-channel *output* conversion remains
  an evidenced blocker.

## Host regression

`tests/test_per_channel_mul.py` (4 tests) pins the profile tag, the per-channel
weight scales, the error win, the geometry/magnitude variants, the unchanged
default EW path and the rejected profiles.
