# Mixed-head fan-out — dense and depthwise heads folded by chained joins

One shared 1×1 Conv stem feeds **three to five heads of two different families** —
dense 1×1/3×3 Conv and group-3 depthwise 1×1/3×3/5×5 Conv — and `n-1`
independently generated joins fold the head grids left to right with mixed
`Add`/`Sub`/`Max`/`Mul` kinds, before an optional `[Conv, Relu]* Conv` tail.

This is the P1 scheduler step the plan asks for: two emitter families are composed
**by tensor name** in one fan-out. Dense heads are emitted with the shared native
Conv field builder; each depthwise head relocates the verified standalone depthwise
program (`open_rknpu.depthwise.compile_depthwise`) into this container, with its
four addresses and `0x1184` rewritten and its weight/bias blocks copied verbatim.
The task order and arena come from `open_rknpu.liveness`.

**Verified on RV1103: 12 models, 384 inferences, 73,728 exact output bytes**
(`board_results_0.json`, `board_summary.txt`).

| Models | Heads | Kernels | Joins | Tail |
| --- | --- | --- | --- | --- |
| 000 | dense, dense, depthwise | 1, 3, 1 | Add, Mul | — |
| 001 | dense, depthwise, dense | 3, 1, 3 | Mul, Add | 1×1 |
| 002 | depthwise, dense, depthwise | 3, 1, 3 | Add, Mul | — (asymmetric) |
| 003 | dense, depthwise, dense, depthwise | 1, 3, 1, 5 | Add, Sub, Add | — |
| 004 | dense, depthwise, depthwise | 1, 3, 5 | Mul, Mul | — (no stem Relu) |
| 005 | depthwise, depthwise, dense | 3, 1, 1 | Add, Add | — (asymmetric) |
| 006 | dense, dense, dense, depthwise | 3, 3, 1, 3 | Add, Mul, Add | 3×3 |
| 007 | depthwise, dense, dense, dense | 1, 3, 1, 3 | Mul, Add, Mul | — (asymmetric) |
| 008 | dense, depthwise, dense, dense | 3, 1, 3, 3 | Sub, Add, Max | — |
| 009 | dense, depthwise, dense, depthwise, dense | 1, 3, 1, 5, 1 | Add, Max, Add, Sub | — |
| 010 | depthwise, dense, depthwise, dense, depthwise | 1, 1, 3, 3, 5 | Add, Add, Mul, Add | — (asymmetric) |
| 011 | dense, dense, dense (hidden 8) | 1, 3, 1 | Add, Mul | — |

Every model runs 32 input cases including all-zero, all-255 and 128. At least 70%
of every model's expected outputs are nonzero (mixed-sign asymmetric depthwise
weights are the lowest), which `tests/test_mixed_heads.py` asserts so a passing
board run cannot be vacuous.

## Arithmetic

Both families share the join rules: a Mul join folds two free operand scales
(`out = rint(a*b/128)`), Add/Sub/Max re-quantize the next head onto the running
result and produce `rint((a+b)/2)`. Depthwise heads require a **three-channel
stem** (the depthwise task's group is its channel count), which is why every mixed
model except the all-dense control uses hidden 3; the control keeps hidden 8 to
show the dense path is unchanged.

## Reproduction

```sh
PYTHONPATH=src python research/build_mixed_head_suite.py
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -fno-use-linker-plugin \
  -Iruntime tests/board_io.c runtime/open_rknpu.c -o /tmp/board_io
PYTHONPATH=src python research/run_v5_suite.py mixed_head_suite
```

`tests/test_mixed_heads.py` recompiles every model byte-identically, reproduces
the expected bytes from `open_rknpu.graph.diamond_reference` with `head_kinds` and
`depthwise_quantizations`, checks that per-head quantization metadata is tracked per
family, and checks the rejections (a depthwise head on a wider stem, a group that is
not three, a 7×7 depthwise kernel, operand zero points, calibration). The existing
all-dense `join_chain_suite` still recompiles byte-identically, which is what proves
the generalization did not disturb the verified path.
