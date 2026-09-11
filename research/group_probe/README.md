# Job shape on RV1103: links, engines and depth

Follow-up to `../submission_probe/` (S1) after a tooling bug there was found and fixed:
the next-command link is **payload-relative** (the runtime programs `PC_DMA_BASE_ADDR`
with the payload base), so a chain from program `0x0` to program `0x440` writes
`0x10 = 0x440`. The first probe wrote `payload + 0x440`, which made every linked case
jump past its program and time out. The conclusions recorded in `submission_probe`
about "links do not help" and "a job is limited to two tasks" were therefore wrong;
this directory holds the corrected measurements. Programs, tensor tables and arenas are
untouched throughout: the probes only rewrite task tails, the descriptor table and the
submission flag, so any difference is caused by the job shape.

## Rule (measured, clean boot per case)

| case | shape | result |
| --- | --- | --- |
| `C1` | 2 CNA, terminal tails, batched | timeout (-110) |
| `C3` | 3 CNA, linked, batched | **PASS**, 32/32 exact |
| `C4` | 4 descriptors over a 2-program loop, linked | **PASS**, 32/32 exact |
| `C6` | v5 6 tasks CNA+DPU, linked, per-task masks | timeout (-110) |
| `C5` | v5 6 tasks CNA+DPU, linked, **union mask** | timeout (-110) |
| `depth` | 8 / 32 / 64 tasks, linked loop job | **PASS** at every depth |

So a job submitted in one ioctl runs its whole list when

1. every task links to the next program (`0x10` = next payload-relative program
   offset, control `0x14 = 0x40`) and the last task is terminal;
2. every transition carries the control word measured for its engine hand-off
   (S8, `../job_field_probe/`): `0x40` inside a CNA or DPU run, `0x14` from CNA to
   DPU, `0x28` terminal. This entry originally concluded "one engine per job" because
   the probe wrote `0x40` at every transition, which times out at a CNA->DPU one;
3. the list fits the loader's 64-task table; depth is otherwise not limited (the
   driver's `max_submit_number` is 65535).

`research/run_job_probe.py` reproduces the rule cases, the depth probe builds
`duplicate_table(pair, N/2)` plus loop links, and `research/run_submission_rule.py`
still holds the original (unlinked) cases as the retained counter-example.

## What it is worth

`crossover.json` compares, on the same graph, N serial submissions against one linked
batched job (128 runs each, medians):

| tasks | serial | one batched job | speedup |
| --- | --- | --- | --- |
| 8 (real 8-layer chain, `deep_chain_suite`) | 205 us | 111 us | 1.8x |
| 12 (real 12-layer chain) | 666 us | 282 us | 2.4x |
| 16 (real 16-layer chain) | 1513 us | 421 us | 3.6x |
| 32 (loop job) | 3172 us | 229 us | 13.8x |
| 64 (loop job) | 1528 us | 449 us | 3.4x |

A *short* run is the other way round: on a three-conv graph, serial measured 67 us
against 116 us for one linked job. The model is roughly `serial ~ 22 us/task` plus a
per-ioctl cost versus `batched ~ 100 us + 8 us/task`, i.e. a crossover near eight
tasks per run. Serial therefore stays the default; a profile with a long run can
request `submission='batched'` from `open_rknpu.compose` or the N-layer chain emitter,
which write the measured links and controls for exactly that shape. A DPU->CNA
hand-off has no measured encoding, so a graph that schedules one (mixed heads) is
still refused.
