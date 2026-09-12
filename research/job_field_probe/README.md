# Job fields and the engine hand-off control (S8)

Two questions from `docs/plans/pipelining-plan.md` S8:

1. what do the driver's `core_mask` / `subcore_task[]` submission fields expect on this
   SoC, and
2. can a mixed-engine task list run in **one** job, the shape every DAG profile submits
   serially?

## Static decode

the upstream `rknpu_job.c` ([provenance](../vendor/README.md) from `research/`), `rknpu_ioctl.h`, `rknpu_drv.c`:

* `rv1106_rknpu_config` uses the single-entry `rknpu_irqs` (`num_irqs = 1`);
  `rknpu_job_alloc` therefore overwrites `args->core_mask` with `RKNPU_CORE0_MASK`, and
  `rknpu_job_subcore_commit_pc` reads `subcore_task[]` only under `num_irqs > 1`. Both
  fields are inert here.
* `REG_WRITE(args->task_base_addr, PC_DMA_BASE_ADDR)` is the only writer of the
  front end's base register; the runtime passes the payload DMA base, which is what makes
  the next-command link (`reg 0x10`) payload-relative.
* The tail's second word (register `0x14`) is the control. Vendor captures decode it as
  `0x40` in front of another Conv task (`capture_chain4`), `0x14` in front of a pool task
  (`capture_pool_max`, `capture_identity`) and `0x28` on the terminal task.

## Measured

`core_mask`/`subcore_task[]` (`tests/board_core.c`, `core_fields.txt`, 32 runs per
configuration, `mnist_pool_suite/model000`, 2 tasks CNA->DPU):

| configuration | output bytes | mismatches |
| --- | --- | --- |
| `core=0x0` (baseline) | baseline | 0 |
| `core=0x7` | unchanged | 0 |
| `core=0x7` + five windows `{0,1},{1,1},{2,0},{3,0},{4,0}` | unchanged | 0 |
| `core=0x0` + the same windows | unchanged | 0 |
| `core=0x1` + window `{0,1}` | unchanged | 0 |

The runtime hook is `ornpu_set_submit_core(model, core_mask, windows, pairs)`.

Engine hand-off (`run_job_field_probe.py`, one clean boot per attempt, 18 attempts,
16 runs each; the containers are the suites' verified serial ones with only the
submission flag and the two link words per task changed):

| attempt | tasks | result |
| --- | --- | --- |
| `pair` serial control (`mnist_pool_suite`, CNA->DPU) | 2 | exact |
| `pair` batched, CNA->DPU control **0x14** | 2 | **exact** |
| `join6` serial control (`pool_join_suite`, 3 CNA + 3 DPU) | 6 | exact |
| `join6` batched, DPU->DPU control 0x14 | 6 | timeout |
| `join6` batched, DPU->DPU control **0x40** | 6 | **exact** |
| `mixedhead7` serial control (`mixed_head_suite`, 4 CNA + 2 DPU + CNA) | 7 | exact |
| `mixedhead7` batched, DPU->CNA control `0x14/0x40/0x28/0x18/0x1C/0x2C/0x48/0x30/0x24/0x44/0x0C/0x20` | 7 | timeout, all 12 |

## WITHDRAWN: the engine hand-off table

Everything below this line is retained as the history of a wrong conclusion, not as a
result. The tail control is **not** an engine hand-off code: it is the successor
program's fetch amount, `PC_DATA_AMOUNT = (regcfg_amount + 4 + 2 - 1)/2 - 1`, which is
why the values coincided with `0x40` (126-word Conv), `0x14` (37-word pool) and `0x28`
(78-word elementwise). The table and the "no encoding" rows that follow were withdrawn
on 2026-09-11 (`docs/plans/pipelining-plan.md` S8/S10, `research/grouped_probe/`):

* the `mixedhead7` DPU->CNA sweep was **confounded**: the prefix of the run it probed
  carried `0x14` at a Conv->elementwise step, which is not that transition's amount, so
  the run failed before reaching the transition under test;
* the "Conv->elementwise has no encoding" sweep (17 candidates) set *both* the
  Conv->elementwise and the elementwise->elementwise transitions to the same value and
  never tried `0x28`;
* with the successor's amount both transitions pass in a single-transition experiment
  (`research/grouped_probe/board_results.json`, `kind="discovery"`).

The raw records in `summary.json` and `board_results.json` are kept: `cna_dpu=0x14`,
`dpu_dpu=0x40` and `dpu_cna=null` are what the probe measured, and `0x14`/`0x40` are
indeed the amounts of the programs those cases linked to. The conclusions drawn from
them are what is withdrawn.

## Original (withdrawn) conclusion

So the control word names the transition, not the successor alone - and the transition
is keyed by the **task family** (the enable value), because the front end tells a pool
task (`96`) apart from an elementwise task (`24`):

| control | transition | evidence |
| --- | --- | --- |
| `0x40` | Conv->Conv, Pool->Pool | vendor `capture_chain4`; this probe's `join6` |
| `0x14` | Conv -> Pool | vendor `capture_pool_max`/`capture_identity`; this probe's `pair` |
| `0x40` | Pool -> elementwise | `../mixed_batched_probe/` (`pool_join`) |
| `0x28` | terminal | both captures |
| none | Conv->elementwise, elementwise->elementwise, elementwise->Conv, Pool->Conv | swept later: 17, 6, 12 candidates respectively, all time out |

## What it unlocks

`open_rknpu.compose.amount_control` writes the successor's fetch amount at every
transition, so `compile_sequence(..., submission='batched')` emits a **one-job** container
for any graph - mixed engines included (`research/mixed_batched_probe/`,
`research/grouped_probe/`: 4-9-task containers, all exact, about half the serial minimum
latency). A profile whose emitter ignores the request is refused with "the container is
serial" instead of being emitted in the wrong mode.

Reproduction:

```sh
PYTHONPATH=src python3 research/run_job_field_probe.py --iterations 16   # slow: reboots
PYTHONPATH=src python3 research/run_job_field_core.py --iterations 32
PYTHONPATH=src python3 research/run_mixed_batched_probe.py --iterations 16
```
