# Pooled-DAG suite — a join expression reduced by a terminal pool

One shared 1×1 stem feeds three or four branches (each one to three Conv layers, with a
depthwise layer allowed inside a chain), two or three joins fold any two previously
produced tensors, and a terminal **2×2 stride-2 MaxPool or AveragePool** reduces the
joined 8×8/C3 result to the 4×4/C3 graph output. This composes the general join DAG
with the verified pooling stage: the pool task is built by the shared
`open_rknpu.pooling.pool_registers` builder and preserves the grid band, so the header
keeps the join's output quantization and only the geometry changes.

**Verified on RV1103: 12 models, 384 inferences, 18,432 exact output bytes**
(`board_results_0.json`, `board_summary.txt`).

| Models | Pool | Branches | Joins |
| --- | --- | --- | --- |
| 000 | MaxPool | 2, 1, 1 | Add, Mul |
| 001 | AveragePool | 2, 1, 1 | Add, Sub |
| 002 | MaxPool | 3, 1, 1 | Add, Mul |
| 003 | AveragePool | 3, 2, 1 | Add, Sub, Mul |
| 004 | MaxPool | 1, 1, 1, 1 | Add, Mul, Mul |
| 005 | AveragePool | 3 (depthwise chain), 1, 1 | Add, Mul |
| 006 | MaxPool | 2 (16-channel intermediate), 1, 1 | Add, Sub |
| 007 | AveragePool | 3, 1, 1 | Sub, Add, Mul |
| 008 | MaxPool | 3, 3, 1 | Mul, Add, Mul |
| 009 | AveragePool | 1, 3 (depthwise chain), 1 | Add, Mul |
| 010 | MaxPool | 2, 1, 1, 1 | Add, Mul, Mul |
| 011 | AveragePool | 2, 2, 1 | Add, Mul, Mul |

Every model runs 32 input cases including all-zero, all-255 and 128; at least 62% of
every model's expected outputs are nonzero, asserted by `tests/test_pooled_dag.py`.

## Reproduction

```sh
PYTHONPATH=src python research/build_pooled_dag_suite.py
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -fno-use-linker-plugin \
  -Iruntime tests/board_io.c runtime/open_rknpu.c -o /tmp/board_io
PYTHONPATH=src python research/run_v5_suite.py pooled_dag_suite
```

`tests/test_pooled_dag.py` recompiles every model byte-identically, reproduces the
expected bytes from `open_rknpu.join_dag.join_dag_reference` with the recorded pool,
checks that every internal owns a fresh arena slot and that the pool is recorded in the
metadata, and checks the rejections (a non-2×2 pool kernel, a join operand with more
than three channels, calibration, and Mul operand zero points). The earlier
`join_dag_suite` (no pool) and `branch_join_suite` still recompile byte-identically.
