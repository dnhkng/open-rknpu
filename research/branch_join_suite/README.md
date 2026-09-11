# Multi-layer branch join — residual-style branch chains folded by joins

One shared 1×1 stem feeds three or four **branches**, each of which is one to three
dense Conv layers chained off the stem (a depthwise branch is a single layer). Two or
three elementwise joins combine any two previously produced tensors, so a branch's
intermediate layer, its final layer, or an earlier join result can all feed a join.
This is the residual-block form: a branch with its own depth, not just a single Conv
head.

**Verified on RV1103: 12 models, 384 inferences, 73,728 exact output bytes**
(`board_results_0.json`, `board_summary.txt`).

| Models | Branch depths | Joins | Notes |
| --- | --- | --- | --- |
| 000 | 2, 1, 1 | Add, Mul | join reads an intermediate layer |
| 001 | 2, 1, 1 | Add, Sub | 8-channel intermediate |
| 002 | 3, 1, 1 | Add, Mul | 8-channel intermediate |
| 003 | 3, 2, 1 | Add, Sub, Mul | 8-channel intermediates |
| 004 | 1, 1, 1, 1 | Add, Mul, Mul | reuse of a final layer |
| 005 | 2, 1, 1 | Add, Mul | depthwise branch |
| 006 | 2, 1, 1 | Add, Sub | 16-channel intermediate |
| 007 | 3, 1, 1 | Sub, Add, Mul | |
| 008 | 3, 3, 1 | Mul, Add, Mul | 8-channel intermediates |
| 009 | 1, 2, 1 | Add, Mul | depthwise branch, dead branch retained |
| 010 | 2, 1, 1, 1 | Add, Mul, Mul | four branches |
| 011 | 2, 2, 1 | Add, Mul, Mul | |

Every model runs 32 input cases including all-zero, all-255 and 128; at least 75% of
every model's expected outputs are nonzero, asserted by
`tests/test_branch_join.py`.

## What the general form needed

* **Per-layer channel counts.** An intermediate layer may output 1..16 channels, so
  the native Conv fields and the weight/bias packing are emitted for that layer's
  own output count (`0x403c`, `0x1030`, `0x1038` and the 32-byte bias groups), not
  the three-channel shape the single-layer profiles use. Only a branch's final layer
  must have three channels, since that is what a join consumes.
* **Per-layer input bands.** A layer reads the previous layer's grid, so its
  `0x1184` border/zero-point field and its weight quantization use that layer's band;
  intermediates keep their natural quantization while finals are zero-centered (and
  re-quantized when an Add/Sub/Max join demands a shared band).
* **Conservative placement.** Every internal gets a fresh arena slot. With arena
  reuse the board returned stale data for model 009 (a slot written by two different
  tasks of a mixed dense/depthwise family); regenerating the containers with fresh
  slots fixed it, which is the same hazard first recorded for the runtime-scale
  profile.

## Reproduction

```sh
PYTHONPATH=src python research/build_branch_join_suite.py
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -fno-use-linker-plugin \
  -Iruntime tests/board_io.c runtime/open_rknpu.c -o /tmp/board_io
PYTHONPATH=src python research/run_v5_suite.py branch_join_suite
```

`tests/test_branch_join.py` recompiles every model byte-identically, reproduces the
expected bytes from `open_rknpu.join_dag.join_dag_reference` with the recorded branch
chains, checks the intermediate-operand rule, and checks the rejections (an
intermediate with more than three channels feeding a join, operand zero points,
calibration, a depthwise branch on a wider stem, an unsupported depthwise kernel, and
a four-layer branch).
