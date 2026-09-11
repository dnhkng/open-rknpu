# Join-chain fan-out — 3..5 heads folded by mixed elementwise joins

One shared 1x1 Conv stem feeds **three to five dense heads**, and `n-1`
independently generated elementwise joins fold the head grids left to right
(`Add`, `Sub`, `Max` or `Mul` at every position) before an optional
`[Conv, Relu]* Conv` tail. This is the scheduler's first variable fan-out: the
task order and the arena come from `open_rknpu.liveness`, so a buffer whose
lifetime has ended is reused by a later join.

**Verified on RV1103: 12 models, 384 inferences, 73,728 exact output bytes**
(`board_results_0.json`, `board_summary.txt`).

| Models | Join kinds | Heads | Tail |
| --- | --- | --- | --- |
| 000 | Add, Add | 3 | — |
| 001 | Sub, Add | 3 | — |
| 002 | Max, Add | 3 | — |
| 003 | Mul, Add | 3 | — |
| 004 | Mul, Mul | 3 | — |
| 005 | Add, Add, Add | 4 | 1×1 |
| 006 | Sub, Max, Add | 4 | 3×3 |
| 007 | Mul, Add, Add | 4 | 1×1 |
| 008 | Add, Sub, Max, Add | 5 | — |
| 009 | Max, Add, Mul, Add | 5 | — |
| 010 | Add, Add (no stem Relu) | 3 | — |
| 011 | Mul, Add, Add | 4 | 3×3 |

Head kernels cycle through 1×1/3×3, hidden channels are 8 or 16, and every model
runs 32 input cases including all-zero, all-255 and 128. Weights are small and
positive so the head grids use the int8 range: **at least 95% of every model's
expected outputs are nonzero**, which `tests/test_join_chain.py` asserts so a
passing board run cannot be vacuous.

## Join arithmetic

* A **Mul** join folds two free operand scales into its own conversion:
  `out = rint(a*b/128)` on output scale `128·sa·sb`.
* **Add/Sub/Max** joins require both operands on one shared scale, so the next
  head is re-quantized onto the running result: `out = rint((a+b)/2)` on output
  scale `2·s`.
* Every operand uses zero point zero, and each head's task declares the stem grid
  zero point it actually reads in `0x1184` (the CNA border path injects it).

## Arena layout

`open_rknpu.liveness` orders the `1 + n + (n-1) + tail` tasks and places every
internal tensor. Because a head is dead once its join has read it, the stem slot
is reused by the first join and later joins reuse earlier buffers where the
intervals are disjoint; `manifest.json` records `schedule` and
`tensor_offsets` for each model. External tensors are placed after all internals
so the version-5 no-overlap rule holds.

## Reproduction

```sh
PYTHONPATH=src python research/build_join_chain_suite.py
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -fno-use-linker-plugin \
  -Iruntime tests/board_io.c runtime/open_rknpu.c -o /tmp/board_io
PYTHONPATH=src python research/run_v5_suite.py join_chain_suite
```

The runner stages the suite at `/userdata/open-npu-research/join_chain_suite`,
runs `./board_io . 12` (the model count; each model's 32 cases come from its input
file) and writes both evidence files.
`tests/test_join_chain.py` recompiles every model byte-identically, reproduces
the expected bytes from `open_rknpu.graph.diamond_reference`, and checks the
rejections (operand zero points, an output override on a non-Mul last join,
a reused head input, nine heads, non-default input quantization, calibration).
