# Runtime per-channel scale Mul — unequal external input shapes

`Mul(image[1,3,H,W], scale[1,3,1,1])` with **two external inputs of different
shapes**. The per-channel operand is exposed as a named external tensor of shape
`(1,1,1,3)`: the runtime packs three bytes into the 16-byte operand row that the
verified elementwise per-channel mode already reads, so no new task machinery is
needed. This retires the "unequal logical inputs" and "complementary singleton
axes with two runtime operands" blockers from the plan.

**Verified on RV1103: 16 models, 256 inferences, 32,832 exact output bytes**
(`board_results_0.json`, `board_summary.txt`).

| Models | Geometry | Operand codes |
| --- | --- | --- |
| 000–003 | 8×8 | `[127,127,127]`, `[64,0,-64]`, `[-128,127,32]`, `[1,2,3]` |
| 004–007 | 5×5 | same four patterns |
| 008–011 | 6×7 | same four patterns |
| 012–015 | 5×8 | same four patterns |

## Contract

* The image input keeps the established packed U8 layout (row stride 16) with
  `input_scale`/`input_zero_point` as usual.
* The scale input is three signed INT8 operand codes, one per channel. The caller
  passes `code + 128` bytes because the runtime's native16 packer subtracts 128.
* `operand_scale` (default `1/127`) is the dequantization step of those codes, so
  a float factor `f_c` becomes `code_c = clip(rint(f_c / operand_scale), -128, 127)`
  and the composed float function is `image * f`.
* The output grid comes from `mul_output_conversion(branch_scale * operand_scale,
  output_range)`, exactly as the constant Mul path does.

## Container

Payload 4,096 bytes (the verified two-identity-branch elementwise program), then:

| Tensor | Role | Layout | Shape | Offset | Bytes |
| --- | --- | --- | --- | --- | --- |
| `image` | input 0 | packed U8 | 1×H×W×3 | 8192 | H·16·3 |
| `scale` | input 1 | native16 | 1×1×1×3 | 4096 | 64 |
| `converted` | internal | native16 | 1×H×W×3 | 12288 | 1024 |
| `output` | output 0 | native16 | 1×H×W×3 | 16384 | 1024 |

The elementwise task reads the operand with `0x5038 = scale offset`,
`0x5034 = 4` (per-channel mode) and `0x5040 = 16` (16-byte row).

## Reproduction

```sh
PYTHONPATH=src python research/build_runtime_scale_suite.py
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -fno-use-linker-plugin \
  -Iruntime tests/board_io.c runtime/open_rknpu.c -o /tmp/board_io
# stage the suite and record board evidence (pushes /tmp/board_io and every file)
PYTHONPATH=src python research/run_v5_suite.py runtime_scale_suite
```

`run_v5_suite.py` writes `board_results_0.json` and `board_summary.txt`; the run
above is the one they record. To drive the board by hand instead, push the suite
to `/userdata/open-npu-research/runtime_scale` and run `./board_io . 16`.

`tests/board_io.c` concatenates external inputs in tensor-index order, so each
input record is `image_bytes ++ 3 scale bytes`; `manifest.json` records the
geometry, codes and quantization of every model, and `tests/test_runtime_scale.py`
checks the container, the operand wiring, the board-verified bytes and the
rejections.
