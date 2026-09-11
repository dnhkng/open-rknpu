# Chain walk: pooling anywhere in a Conv chain (op-level walk)

`open_rknpu.walk` is the op-level walk path of the scheduler: it parses a linear graph,
walks its nodes, builds one `open_rknpu.compose` stage per op with the verified per-op
builders, and tracks each tensor's quantization band as it goes. It is the first path
where a **pool may sit inside a Conv chain** - `Conv -> pool -> Conv`, repeated pools
and a pool before the output all lower to the same composed container.

**Verified on RV1103: 12 models, 192 inferences, 6,368 exact output bytes**
(`run_v5_suite.py walk_chain_suite`). No profile above the walk matches a graph whose
pool is followed by another node, so the walk cannot hijack existing evidence
(`tests/test_walk.py` checks the dispatch boundary).

| model | graph | output |
| --- | --- | --- |
| 0 | Conv3x3+Relu -> MaxPool -> Conv3x3 | 3x4x4 |
| 1 | Conv1x1 -> AveragePool -> Conv3x3+Relu | 3x4x4 |
| 2 | Conv3x3+Relu -> MaxPool -> Conv3x3+Relu -> MaxPool -> Conv1x1 | 3x2x2 |
| 3 | Conv3x3+Relu (16 hidden) -> MaxPool -> Conv3x3 | 3x4x4 |
| 4 | Conv3x3+Relu -> MaxPool -> Conv3x3+Relu -> Conv3x3 | 3x4x4 |
| 5 | Conv3x3+Relu -> AveragePool -> Conv1x1+Relu -> MaxPool -> Conv3x3 | 3x2x2 |
| 6-8 | the same shapes on 8x6, 6x8 and 6x6 grids | 3x4x3 etc. |
| 9 | Conv3x3+Relu -> MaxPool -> Conv3x3+Relu -> AveragePool -> Conv3x3+Relu -> MaxPool -> Conv1x1 | 3x1x1 |
| 10 | Conv3x3+Relu -> MaxPool -> Conv3x3+Relu -> MaxPool (pooled output) | 8x2x2 |
| 11 | Conv3x3+Relu (3 hidden) -> MaxPool -> Conv3x3 | 3x4x4 |

The narrow and wide grids exist because the geometry-dependent registers were the last
bug in this path; they now exercise the same code on 8x6, 6x8 and 6x6 surfaces.

## How a stage is built

* **First Conv** (reads the packed UINT8 image): the layer is compiled with the
  established single-Conv emitter and its whole register set, weight/bias blocks and
  band are copied, exactly like the chain family's first stage.
* **Later Conv** (reads a native16 grid): `native_fields` with the *actual* geometry of
  the surface it reads and writes, plus the chain family's `native_quantize` band and
  Relu flag. Using the 8x8-derived register defaults here was the last bug: a pooled
  4x4 grid needs `0x107c/0x1080/0x118c/0x3014/0x4030/0x4034/0x405c/0x500c/0x5010` from
  its own geometry.
* **Conv feeding a pool**: re-quantized onto a zero-point-0 grid with the scale widened
  by `max(128+zp,127-zp)/127`, the same policy `pooled_branches` uses, because the
  verified DPU pool programs assume a zero-point-0 band.
* **Pool**: `pool_registers` with the input/output geometry; the task's registers are
  identical to the board-verified `pooled_branches` pool apart from its two addresses.
* Every internal gets a fresh arena slot (`reuse=False`): a slot written by two
  different task families returned stale data on the board.

## Not yet walked

Fan-in joins are still profile-matched: `diamond`/`join_dag`/`pooled_branches` declare
their own stages rather than the walk dispatching a join op. A first attempt at a join
walk was written and withdrawn in this round (unfinished band re-quantization for the
two heads); the next step is to lift the join band rules (`_join_fields` plus the
shared-scale re-quantization `compile_diamond` uses) into the walk.

## Reproduction

```sh
PYTHONPATH=src python3 research/build_walk_chain_suite.py
PYTHONPATH=src python3 research/run_v5_suite.py walk_chain_suite
PYTHONPATH=src python3 -m pytest tests/test_walk.py -q
```

The suite is independently generated (no vendor artifact); expected bytes come from
`chain_walk_reference`, which composes the established image reference for the first
Conv, the native grid reference for later Convs and the 2x2 block max/mean for a pool.
