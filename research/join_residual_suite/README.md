# Join-chain runtime residual — an external feature map on a DAG result

A shared 1×1 stem fans out to 3..5 dense/depthwise heads, chained joins fold them,
and a final `Add/Sub/Max(result, residual)` combines the result with a **second
external input** of shape `[1,3,8,8]`. Where the sibling
[`join_scale_suite`](../join_scale_suite/) supplies a per-channel *gain*, this profile
supplies a full **feature map** — a runtime residual connection into a computed DAG.

**Verified on RV1103: 12 models, 384 inferences, 73,728 exact output bytes**
(`board_results_0.json`, `board_summary.txt`).

| Models | Heads | Kernels | Joins | Tail |
| --- | --- | --- | --- | --- |
| 000 | dense ×3 | 1,3,1 | Add, Mul | Add |
| 001 | dense, depthwise, dense | 3,1,3 | Mul, Add | Sub |
| 002 | depthwise, dense, depthwise | 3,1,3 | Add, Mul | Max (asymmetric) |
| 003 | dense, depthwise, dense, depthwise | 1,3,1,5 | Add, Sub, Add | Add |
| 004 | dense ×4 | 3,3,1,3 | Mul, Mul, Add | Sub |
| 005 | depthwise, depthwise, dense, dense | 3,1,1,3 | Add, Add, Mul | Max (asymmetric) |
| 006 | dense, depthwise, dense, dense, dense | 1,3,1,3,1 | Max, Add, Mul, Sub | Add |
| 007 | dense ×3 (no stem Relu) | 1,1,1 | Mul, Mul | Sub |
| 008 | dense, depthwise, dense | 1,5,3 | Add, Add | Max (asymmetric) |
| 009 | dense ×5 | 1,3,1,3,1 | Add, Sub, Max, Mul | Add |
| 010 | dense, depthwise, depthwise, dense | 3,3,5,1 | Mul, Add, Add | Sub |
| 011 | dense ×3 | 3,3,3 | Sub, Sub | Max |

Every model runs 32 input cases (all-zero, all-255, 128 and random images) with
residual maps that include zero, +127, −128 and random values in −60..60; at least
65% of every model's expected outputs are nonzero, asserted by
`tests/test_join_residual.py`.

## Contract and emission

* The residual input is a `[1,3,8,8]` external tensor in the v5 native16 layout. Its
  values are **zero-centered INT8 on the join result's scale**; the caller passes
  `value + 128` bytes because the loader's native16 packer subtracts 128 (the same
  convention as the per-channel scale codes).
* Both operands therefore share one scale, which is exactly what the verified
  Add/Sub/Max join needs: `out = rint((a + b)/2)` on output scale `2·s`. The task is
  the verified join emitter with the external grid as its ERDMA secondary at the
  per-pixel plane stride (`0x5034 = 0x40000004`, `0x5040 = surface`).
* As in the runtime-scale profile, every internal gets a **fresh slot**: a join
  primary whose arena slot had been reused by an earlier Conv task returned stale
  data on the board. `manifest.json` records `schedule` and `tensor_offsets`.

## Reproduction

```sh
PYTHONPATH=src python research/build_join_residual_suite.py
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -fno-use-linker-plugin \
  -Iruntime tests/board_io.c runtime/open_rknpu.c -o /tmp/board_io
PYTHONPATH=src python research/run_v5_suite.py join_residual_suite
```

Each input record is `image (192 B) ++ residual + 128 (192 B)`.
`tests/test_join_residual.py` recompiles every model byte-identically, reproduces
the expected bytes from `open_rknpu.graph.join_chain_residual_reference`, checks
that the residual primary owns a fresh slot, and checks the rejections (a
per-channel operand shape, an output override on a non-Mul tail join, operand zero
points, a depthwise head on a wider stem, calibration).
