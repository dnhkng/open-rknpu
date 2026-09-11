# Pooled-branches suite — multi-layer branch chains that pool before the joins

Every branch of a pooled join is a **Conv chain of one to three layers** (a depthwise
layer inside a chain is expanded to an exact block-diagonal dense kernel), followed by a
2×2 stride-2 **MaxPool or AveragePool**, so each join operand is a 4×4/C3 grid. One or
two elementwise joins fold two or three branches. This extends
[`pool_join_suite`](../pool_join_suite/) from two single-Conv branches to deeper,
wider backbones.

**Verified on RV1103: 13 models, 416 inferences, 19,968 exact output bytes**
(`board_results_0.json`, `board_summary.txt`).

| Models | Branches (depth) | Pool | Joins |
| --- | --- | --- | --- |
| 000 | 2, 1 | MaxPool | Add |
| 001 | 1, 2 | AveragePool | Sub |
| 002 | 3 (depthwise), 1 | MaxPool | Mul |
| 003 | 3 (depthwise), 2 | AveragePool | Max |
| 004 | 2, 1, 1 | MaxPool | Add, Add |
| 005 | 1, 3 (depthwise), 1 | AveragePool | Mul, Add |
| 006 | 2, 2, 1 | MaxPool | Sub, Max |
| 007 | 2, 1, 3 (depthwise) | AveragePool | Add, Mul |
| 008 | 3 (16-channel middle), 1 | MaxPool | Add |
| 009 | 3 (depthwise), 3 (depthwise), 3 (depthwise) | AveragePool | Add, Sub |
| 010 | 3 (depthwise), 1 | MaxPool | Mul |
| 011 | 1, 1, 3 | AveragePool | Mul, Mul |
| 012 | 3, 1, 1 | MaxPool | Add, Mul |

Every model runs 32 input cases including all-zero, all-255 and 128; at least 62% of
every model's expected outputs are nonzero, asserted by
`tests/test_pooled_branches.py`.

## Implementation notes

* **Band propagation runs over the branch finals.** Pooling preserves the grid band, so
  the join scale rules (Mul folds two free scales; Add/Sub/Max need one shared band and
  re-quantize an uncommitted branch output) are applied with the branch finals standing
  in for the pooled grids, then the pooled tensors inherit those bands.
* **The pool task** comes from the shared `open_rknpu.pooling.pool_registers` builder
  (`8x8 -> 4x4`, 37 words, enable 96), and each join runs at 4×4 with a 256-byte plane
  stride.
* **Fresh slot per internal**, as in the other join profiles.
* **Dispatch order matters.** This matcher is checked *after* `parse_pool_join`, so the
  earlier two-single-Conv containers keep their emitter and stay byte-identical; a first
  attempt that routed them here was caught by their byte-identity test.

## Reproduction

```sh
PYTHONPATH=src python research/build_pooled_branches_suite.py
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -fno-use-linker-plugin \
  -Iruntime tests/board_io.c runtime/open_rknpu.c -o /tmp/board_io
PYTHONPATH=src python research/run_v5_suite.py pooled_branches_suite
```

`tests/test_pooled_branches.py` recompiles every model byte-identically, reproduces the
expected bytes from `open_rknpu.pooled_branches.pooled_branches_reference`, and checks
the rejections (a non-2×2 pool kernel, four branches, a chain that does not end in three
channels, calibration, and Mul operand zero points).
