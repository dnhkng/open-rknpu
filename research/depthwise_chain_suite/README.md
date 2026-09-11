# Depthwise-chain suite — depthwise-separable blocks inside DAG branches

A branch of the general join DAG may be a chain of one to three Conv layers, and a
**depthwise layer may sit inside that chain** (`group` equal to its input channel
count). The dedicated depthwise emitter models a branch that reads the stem directly,
so a chained depthwise layer is expanded to an equivalent **block-diagonal dense
kernel** and emitted through the per-layer dense path. Joins then fold branch finals
exactly as in the other join-DAG profiles.

**Verified on RV1103: 12 models, 384 inferences, 73,728 exact output bytes**
(`board_results_0.json`, `board_summary.txt`).

| Models | Depthwise layers | Branch shapes | Joins |
| --- | --- | --- | --- |
| 000 | 1 | 3→8, DW8, →3 | Add, Mul |
| 001 | 1 | DW3, →3 | Sub, Add |
| 002 | 1 | 3→16, DW16, →3 | Add, Mul |
| 003 | 1 | 3→8 (k3), DW8 (k1), →3 (k3) | Add, Sub |
| 004 | 2 | two 3→8, DW8, →3 chains | Add, Mul |
| 005 | 1 | 3→4, DW4, →3 with four branches | Add, Mul, Mul |
| 006 | 1 | depthwise chain in the second branch | Add, Mul |
| 007 | 2 | depthwise chain plus a single-layer depthwise branch | Add, Sub |
| 008 | 1 | 3→16 (k3), DW16, →3 (k3) | Mul, Add |
| 009 | 3 | three 3→8, DW8, →3 chains | Add, Sub |
| 010 | 2 | depthwise chain in the second of four branches | Add, Mul, Mul |
| 011 | 2 | 3→8 (k3), DW8, →3 plus a depthwise branch | Add, Mul |

Every model runs 32 input cases including all-zero, all-255 and 128; at least 78% of
every model's expected outputs are nonzero, asserted by
`tests/test_depthwise_chain.py`.

## Why the block-diagonal expansion is exact

For `group == C` each output channel connects to exactly one input channel, so a
`(C,1,k,k)` depthwise kernel is the same operator as a `(C,C,k,k)` kernel whose only
nonzeros are `expanded[c,c] = w[c,0]`. In the quantized domain every added tap has
weight zero, so it contributes nothing regardless of the input grid's zero point; the
per-output-channel weight scales and biases are then quantized by the existing dense
path (`native_quantize` over the expanded kernel). The dedicated depthwise emitter is
still used for single-layer depthwise branches, so the earlier `join_dag_suite`
containers are unchanged byte-for-byte.

## Reproduction

```sh
PYTHONPATH=src python research/build_depthwise_chain_suite.py
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -fno-use-linker-plugin \
  -Iruntime tests/board_io.c runtime/open_rknpu.c -o /tmp/board_io
PYTHONPATH=src python research/run_v5_suite.py depthwise_chain_suite
```

`tests/test_depthwise_chain.py` recompiles every model byte-identically, reproduces
the expected bytes from `open_rknpu.join_dag.join_dag_reference` with the recorded
branch chains, checks that a chained depthwise layer is emitted on the dense path, and
checks the rejections (a join operand with more than three channels, a chained
depthwise 5×5 kernel outside the dense expansion, a single-layer depthwise branch on a
wider stem, and calibration).
