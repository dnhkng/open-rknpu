# Pool-join — two Conv+pool branches from one stem

One shared 1×1 Conv stem feeds two dense 1×1/3×3 Conv branches, each followed by a
**2×2 stride-2 MaxPool or AveragePool**, and one elementwise join folds the two
4×4/C3 pooled grids. This closes the plan's "multi-task pool/depthwise branches
into Mul" blocker for the pooling half, alongside
[`depthwise_join_suite`](../depthwise_join_suite/).

**Verified on RV1103: 12 models, 384 inferences, 18,432 exact output bytes**
(`board_results_0.json`, `board_summary.txt`).

| Models | Join | Pool | Head kernels |
| --- | --- | --- | --- |
| 000 | Add | MaxPool | 1×1, 1×1 |
| 001 | Add | AveragePool | 3×3, 1×1 |
| 002 | Mul | MaxPool | 1×1, 3×3 |
| 003 | Mul | AveragePool | 3×3, 3×3 |
| 004 | Sub | MaxPool | 1×1, 1×1 |
| 005 | Sub | AveragePool | 3×3, 1×1 |
| 006 | Max | MaxPool | 1×1, 3×3 |
| 007 | Max | AveragePool | 3×3, 3×3 |
| 008 | Add | MaxPool | 3×3, 3×3 (no stem Relu) |
| 009 | Mul | AveragePool | 1×1, 1×1 (hidden 16) |
| 010 | Sub | MaxPool | 3×3, 1×1 (hidden 16) |
| 011 | Max | AveragePool | 1×1, 3×3 (hidden 3) |

Every model runs 32 input cases including all-zero, all-255 and 128. At least 96%
of every model's expected outputs are nonzero, which `tests/test_pool_join.py`
asserts so a passing board run cannot be vacuous.

## Composition and quantization

Each branch is emitted with the shared native Conv field builder (input zero point
declared in `0x1184`), and each pool task is built by
`open_rknpu.pooling.pool_registers` — the *same* builder the sequence lowering
uses — so the 37-word pool program has one definition. Only the two addresses
(`0x701c` input, `0x6070` output) and the geometry fields are substituted.

Pooling preserves the grid scale, so a Mul join folds the two free branch scales
(`out = rint(a*b/128)`) while Add/Sub/Max re-quantize both branches onto one
shared scale and produce `rint((a+b)/2)`. The join task runs at 4×4 with a 256-byte
plane stride. `open_rknpu.liveness` orders the six tasks and places the stem, both
branch grids and both pooled grids; the external input and output are placed after
every internal tensor so the version-5 no-overlap rule holds.

## Reproduction

```sh
PYTHONPATH=src python research/build_pool_join_suite.py
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -fno-use-linker-plugin \
  -Iruntime tests/board_io.c runtime/open_rknpu.c -o /tmp/board_io
PYTHONPATH=src python research/run_v5_suite.py pool_join_suite
```

The runner stages the suite at `/userdata/open-npu-research/pool_join_suite`, runs
`./board_io . 12` and writes both evidence files. `tests/test_pool_join.py`
recompiles every model byte-identically, reproduces the expected bytes from
`open_rknpu.pool_join.pool_join_reference`, and checks the rejections (output
override on a non-Mul join, operand zero points, mismatched pool kinds, a 3×3
pool kernel, an out-of-range stem width, non-default input quantization,
calibration) plus that a two-dense-head diamond still takes the diamond path.
