# Mixed-engine DAG in one job (published mode)

`compile_sequence(model, submission='batched')` emits a **single linked job**: each tail
carries the successor program's fetch amount (`docs/plans/pipelining-plan.md` S8/S10,
`../grouped_probe/`), which is what the front end needs to load the next program, so a
whole DAG - mixed engines included - runs in one ioctl. The earlier "engine hand-off"
reading of that word, and the run splitting that followed from it, are withdrawn: see
`../grouped_probe/README.md`.

## Containers

`run_mixed_batched_probe.py` rebuilds four containers from the suites' own ONNX models
and checks, before touching the board, that the **serial** compile still reproduces the
retained board-verified container byte for byte (so the only difference is the linked
tail words and the submission flag).

| container | tasks | engines |
| --- | --- | --- |
| `pool_join_suite` model000 | 6 | 3 CNA, 3 DPU |
| `pool_join_suite` model001 | 6 | 3 CNA, 3 DPU |
| `pooled_branches_suite` model000 | 7 | 4 CNA, 3 DPU |
| `pooled_branches_suite` model003 | 9 | 6 CNA, 3 DPU |

## Board evidence

`tests/board_bench.c`, 4 cases per mode, 16 runs per case, expected bytes are the
suites' own board-verified files:

| container | mode | tasks | min | median | exact |
| --- | --- | --- | --- | --- | --- |
| `pool_join` 000 | serial | 6 | 73.5 us | 114.9 us | yes |
| `pool_join` 000 | **one job** | 6 | 32.4 us | 133.0 us | yes |
| `pool_join` 001 | serial | 6 | 77.3 us | 167.4 us | yes |
| `pool_join` 001 | **one job** | 6 | 34.7 us | 87.8 us | yes |
| `pooled_branches` 000 | serial | 7 | 81.4 us | 130.4 us | yes |
| `pooled_branches` 000 | **one job** | 7 | 36.5 us | 256.1 us | yes |
| `pooled_branches` 003 | serial | 9 | 109.7 us | 224.6 us | yes |
| `pooled_branches` 003 | **one job** | 9 | 44.3 us | 91.3 us | yes |

(The first `pool_join` job median is board-load noise; the minimum, which is the floor,
is the number the tests compare.)

## Bounds

* one job per container: every transition must carry a measured control. `0x14` is the
  Conv->Pool hand-off, so the graphs here (Conv runs then pool runs then one elementwise
  join, i.e. `29->96->24`) chain into one job; a graph with an unmeasured step
  (Conv->elementwise or anything back into a Conv) splits there instead
  (`../grouped_probe/`);
* emitters that build their containers without the composer (`depthwise_join`,
  `pooled_dag`, `join_dag`, ...) ignore the `submission` request; `compile_sequence`
  rejects the request with "the container is serial" instead of emitting a wrong mode;
* depth is bounded by the loader's 64-task table.

Reproduction:

```sh
PYTHONPATH=src python3 research/run_mixed_batched_probe.py --iterations 16
```
