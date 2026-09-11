# Per-family task cost: attempt, and why the differential is not resolvable this way

The plan's S5 residual wanted a cost per task family (CNA conv, DPU pool, DPU
elementwise). The verified tree has no comparable single-family containers, so this
probe builds a matched set at one geometry and differences it:

| model | graph | tasks | expected |
| --- | --- | --- | --- |
| `conv` | one 1x1 Conv C3->C3 at 8x8 | 1 CNA | conv requantisation |
| `conv_pool` | the **same** Conv weights and bias + 2x2 stride-2 MaxPool | 2 (CNA + DPU pool) | conv then 2x2 max |
| `ew2` | copy of `runtime_scale_suite/model000` (1x1 Conv + per-channel Mul) | 2 (CNA + DPU elementwise) | the verified suite's bytes |

`research/run_family_cost_probe.py` benches all three with `tests/board_bench.c`
(128 runs each, one model per clean boot). All three are **exact**; the measured
medians/mins are in `family_cost.json`:

| model | tasks | median | min |
| --- | --- | --- | --- |
| `conv` | 1 | 402.8 us | 45.5 us |
| `conv_pool` | 2 | 89.3 us | 30.3 us |
| `ew2` | 2 | 65.9 us | 38.8 us |

## Conclusion: below the noise floor, not a hardware fact

Differencing against the single-task container gives **negative** marginals
(pool -15 us, elementwise -7 us on the minima). A two-task container cannot be
cheaper than a one-task container, so the difference is measurement structure, not
hardware behaviour:

* the per-inference fixed cost (submit + wait + input pack + output unpack) is
  25-45 us at this size, while a task marginal is a few microseconds;
* serial medians on this board swing by up to 9x with CPU load (the same 1-task
  container shows 45 us min against 403 us median), i.e. jitter of +-15 us dwarfs the
  per-task cost;
* the only reliable per-task figures remain the *batched* ones from job lists
  (`research/group_probe/`): 3-7 us per task in one job, and about 14 us per task when
  submitted serially, both for the CNA family.

## Prerequisite for a real per-family table

A single-family **multi-task** profile - for example N independent pool tasks in one
container - so the cost can be fitted as a slope over N instead of differenced between
two containers. Pool and elementwise task lists need an emitter addition (the pool
register builder currently only appears inside Conv+pool profiles), or the batched
loop-job construction used for CNA. Until then the per-family numbers stay recorded as
unresolved rather than guessed.

Reproduction:

```sh
PYTHONPATH=src python3 research/build_family_cost_probe.py
PYTHONPATH=src python3 research/run_family_cost_probe.py
```
