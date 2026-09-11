# The tail control is the successor's fetch amount: every DAG is one job

A non-serial container links every task, and each tail's second word (register `0x14`) is
the value the driver would program into `PC_DATA_AMOUNT` for the task it links to:

```
control = (successor_regcfg_amount + RKNPU_PC_DATA_EXTRA_AMOUNT + scale - 1) / scale - 1
        = (words + 4 + 2 - 1) / 2 - 1        # RV1106: extra 4, scale 2
```

| successor program | words | control |
| --- | --- | --- |
| Conv | 126 | `0x40` (64) |
| pool | 37 | `0x14` (20) |
| elementwise / join | 78 | `0x28` (40) |
| LUT setup | 1106 | `0x22A` (554) |

The vendor captures write exactly these values (`0x40` before a Conv in `capture_chain4`,
`0x14` before a pool in `capture_pool_max`, `0x28` before the terminal task), which is
what made an earlier probe read them as "engine hand-off codes" and conclude that some
transitions had no encoding. `docs/plans/pipelining-plan.md` S8/S10; the withdrawn table and the
reason its sweeps failed are kept in `../job_field_probe/README.md`.

Because the value is a fetch size and not a routing code, it is recomputed whenever the
successor's register count changes: `open_rknpu.compose.amount_control` derives it and
`batched_layout` validates it. A terminal tail keeps the `0x28` sentinel (the link is
zero, so the front end stops).

## Discovery: one transition at a time (clean boot per case)

Container surgery on verified serial containers (`kind="discovery"` in
`board_results.json`), 16 runs each:

| case | container | result |
| --- | --- | --- |
| Conv->elementwise, `0x28` = amount(78) | `mixed_head_suite/model000`, links 0..3 | **exact**, 2 ioctls |
| Conv->elementwise, `0x14` (counter-example) | the same container | timeout, retained |
| elementwise->elementwise, `0x28` | the same container, links 0..4 | **exact**, 1 ioctl |
| Conv->Pool, `0x14` = amount(37) | `mnist_pool_suite/model000` | **exact**, 1 ioctl |

The counter-example is what the old table implied: with a value that is not the
successor's amount the run does not complete. With the amount it does, and the whole
graph becomes one job.

## Published: every emitter that honours `submission='batched'`

`compile_sequence(model, submission='batched')` builds each container (after checking that
the serial compile still reproduces the suite's retained, board-verified container), then
both modes run over 4 cases with 16 runs each. `engine_runs` is what the runtime reports,
so "one job" is measured:

| suite | model | tasks | serial ioctls | one-job ioctls | serial min | one-job min | exact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `diamond_tail_suite` | 000 | 5 | 5 | **1** | 73.8 us | 45.8 us | yes |
| `depthwise_join_suite` | 000 | 4 | 4 | **1** | 63.6 us | 54.3 us | yes |
| `pooled_dag_suite` | 000 | 8 | 8 | **1** | 170.6 us | 51.0 us | yes |
| `join_dag_suite` | 000 | 6 | 6 | **1** | 98.0 us | 45.2 us | yes |
| `pool_join_suite` | 000 | 6 | 6 | **1** | 109.7 us | 46.4 us | yes |
| `pooled_branches_suite` | 000 | 7 | 7 | **1** | 87.2 us | 42.0 us | yes |
| `mixed_head_suite` | 001 | 7 | 7 | **1** | 101.8 us | 50.8 us | yes |
| `join_chain_suite` | 005 | 9 | 9 | **1** | 141.5 us | 55.4 us | yes |

The four DAG emitters threaded for this pass are `graph.compile_diamond` (diamond /
diamond-tail), `depthwise_join.compile_depthwise_join` and `join_dag.compile_join_dag`
(join-DAG and pooled-DAG, which also picks up a 37-word pool task as its last step).
`compose`-based profiles (`pool_join`, `pooled_branches`, the join-chain/mixed-head
emitter) were already threaded and simply moved from "one job where the transitions were
already correct" to "one job for every graph".

## Bounds

* the control is derived from the successor's register count, so it must be recomputed if
  an emitter changes a program's word count after writing tails (all emitters call
  `handoff_tails` right before encoding);
* a container whose tails are all terminal is still one run per task (the runtime derives
  runs from the link words), which is what the flag-flipped probe twins do;
* depth is bounded by the loader's 64-task table;
* emitters that ignore the `submission` request (`add_suite`, most single-op suites) are
  covered by the scheduler's post-pass (`sequence.relink_for_batched`), which relinks the
  container they produced - see `../batched_all_probe/`.

Reproduction:

```sh
PYTHONPATH=src python3 research/run_grouped_probe.py --iterations 16
```
