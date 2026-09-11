# Depthwise-branch join — dense and depthwise branches from one stem

One shared 1x1 Conv stem feeds **two branches**: a dense 1x1/3x3 Conv and a
group-3 depthwise 1x1/3x3/5x5 Conv. A single elementwise join folds the two
8x8/C3 grids. This retires the plan's "multi-task pool/depthwise branches into
Mul" blocker for the depthwise half (a pooling branch still needs its own
relocation work).

**Verified on RV1103: 12 models, 384 inferences, 73,728 exact output bytes**
(`board_results_0.json`, `board_summary.txt`).

| Models | Join | Dense kernel | Depthwise kernel | Stem Relu | Asymmetric depthwise |
| --- | --- | --- | --- | --- | --- |
| 000 | Add | 1×1 | 3×3 | yes | no |
| 001 | Add | 3×3 | 3×3 | yes | no |
| 002 | Mul | 1×1 | 3×3 | yes | no |
| 003 | Mul | 3×3 | 3×3 | yes | no |
| 004 | Sub | 1×1 | 5×5 | yes | no |
| 005 | Sub | 3×3 | 5×5 | yes | no |
| 006 | Max | 1×1 | 1×1 | yes | no |
| 007 | Max | 3×3 | 1×1 | yes | no |
| 008 | Add | 3×3 | 5×5 | no | no |
| 009 | Mul | 1×1 | 5×5 | yes | yes |
| 010 | Add | 3×3 | 1×1 | yes | yes |
| 011 | Max | 3×3 | 3×3 | yes | no |

Every model runs 32 input cases including all-zero, all-255 and 128. At least 92%
of every model's expected outputs are nonzero, which `tests/test_depthwise_join.py`
asserts so a passing board run cannot be vacuous.

## Composition

The depthwise task program is not re-derived: it comes from the verified
standalone emitter (`open_rknpu.depthwise.compile_depthwise`) and this container
**relocates its four address registers** (`0x1070` input, `0x4020` output,
`0x1110` weights, `0x5020` bias) plus the input zero point `0x1184`, copying its
weight and bias blocks verbatim. The dense branch is emitted with the shared
native field builder. The optional asymmetric depthwise pair (per-channel weight
zero points) is preserved by the standalone compile and exercised by models
009/010.

## Quantization

Both branch grids use zero point zero. A **Mul** join folds the two free operand
scales (`out = rint(a*b/128)`, output scale `128·sa·sb`); **Add/Sub/Max** need one
shared scale, so both branches are re-quantized onto `max(adjusted)` and the join
output scale is twice it. The per-branch scale is the natural scale adjusted by
`max(128+zp, 127-zp)/127`, exactly as the dense diamond does.

## Arena layout

`open_rknpu.liveness` orders the four tasks and places `stem`, `dense` and
`depthwise`; the stem is live until both branches have read it, so its bytes are
reused only where the intervals are disjoint. External tensors are placed after
all internals so the version-5 no-overlap rule holds.

## Reproduction

```sh
PYTHONPATH=src python research/build_depthwise_join_suite.py
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -fno-use-linker-plugin \
  -Iruntime tests/board_io.c runtime/open_rknpu.c -o /tmp/board_io
PYTHONPATH=src python research/run_v5_suite.py depthwise_join_suite
```

`tests/test_depthwise_join.py` recompiles every model byte-identically,
reproduces the expected bytes from
`open_rknpu.depthwise_join.depthwise_join_reference`, and checks the rejections
(output override on a non-Mul join, operand zero points, a group that is not the
branch width, a hidden stem wider than three channels, an unsupported depthwise
kernel, non-default input quantization, calibration) plus that a two-dense-head
diamond still takes the diamond path.
