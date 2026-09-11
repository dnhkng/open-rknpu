# Join-expression DAG — internal grids reused across joins

One shared 1×1 stem feeds 3–4 dense/depthwise heads, and two or three elementwise
joins combine **any two previously produced tensors**, so a head or an earlier join
result can feed more than one consumer. The left-fold chain keeps its own emitter
(`join_chain_suite`); this profile is the general expression form that the chain
matcher declines.

**Verified on RV1103: 12 models, 384 inferences, 73,728 exact output bytes**
(`board_results_0.json`, `board_summary.txt`).

| Models | Heads | Expression |
| --- | --- | --- |
| 000 | dense ×3 (1,3,1) | `Add(h0,h1)`, `Mul(j0,h0)` |
| 001 | dense ×3 (3,1,3) | `Add(h0,h1)`, `Mul(j0,h1)` |
| 002 | dense ×3 (1,1,1) | `Mul(h0,h1)`, `Mul(j0,h0)` |
| 003 | dense ×3 (3,3,3) | `Sub(h1,h0)`, `Add(j0,h2)` |
| 004 | dense ×3 (1,3,1) | `Max(h0,h1)`, `Sub(h2,j0)`, `Mul(j1,h0)` |
| 005 | dense ×3 (3,1,3) | `Add(h0,h1)`, `Add(j0,h2)`, `Mul(j1,h1)` |
| 006 | dense ×3 (1,3,3) | `Mul(h0,h1)`, `Sub(j0,h2)`, `Mul(j1,h0)` |
| 007 | dense ×3 (3,3,1) | `Add(h0,h1)`, `Sub(j0,h2)`, `Mul(j1,h2)` |
| 008 | dense ×3 (1,1,3) | `Mul(h0,h1)`, `Mul(h1,h2)`, `Mul(j0,j1)` |
| 009 | dense ×3 (3,1,1) | `Add(h0,h1)`, `Mul(j0,h2)`, `Mul(j1,h0)` |
| 010 | depthwise,dense,dense (3,1,3) | `Add(h0,h1)`, `Mul(j0,h0)` |
| 011 | dense,depthwise,depthwise (1,3,1) | `Add(h0,h1)`, `Sub(j0,h2)`, `Mul(j1,h0)` |

Every model runs 32 input cases including all-zero, all-255 and 128; at least 35% of
every model's expected outputs are nonzero, asserted by `tests/test_join_dag.py`.

## Scale propagation

Each tensor carries one INT8 scale, assigned in topological node order:

* A **Mul** join folds two free operand scales (`128·sa·sb`) and commits both.
* **Add/Sub/Max** need both operands on one band: an uncommitted head operand is
  re-quantized onto the other operand's scale, and two operands that are already
  committed must agree. A tensor that a later join wants on a *different* band is
  rejected with a specific message (`join DAG cannot re-quantize a join result onto
  another band` / `...reused tensors fixed different bands`).

The task order and the per-tensor live intervals come from
`open_rknpu.liveness`; a reused grid stays live across all of its consumers, which
`tests/test_join_dag.py` asserts by checking that `h0`'s interval still extends to
the second join. **Placement is conservative**: every internal gets a fresh arena
slot. The board returned stale data when one slot was written by two different tasks
of a mixed task family (first seen in the multi-layer branch work below), so this
profile does not rely on arena reuse. The containers were regenerated under that rule
and re-verified on the board.

## Reproduction

```sh
PYTHONPATH=src python research/build_join_dag_suite.py
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -fno-use-linker-plugin \
  -Iruntime tests/board_io.c runtime/open_rknpu.c -o /tmp/board_io
PYTHONPATH=src python research/run_v5_suite.py join_dag_suite
```

`tests/test_join_dag.py` recompiles every model byte-identically, reproduces the
expected bytes from `open_rknpu.join_dag.join_dag_reference`, checks that the
left-fold chain still routes to the chain emitter, and checks the rejections
(operand zero points, calibration, a depthwise head on a wider stem, a join reading
the same tensor twice, and a reused tensor demanded on two bands).
