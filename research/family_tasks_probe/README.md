# Per-family task cost by duplicated-task fitting (S5)

Differencing two different containers was below the board's noise floor
(`../family_cost_probe/`). This probe repeats a **verified task descriptor** N times
inside one container instead: every copy reads and writes the same addresses, so the
extra runs are idempotent and the expected bytes stay valid, while the per-inference
cost becomes `C + N * task`. Fitting that line over N is robust to load jitter because
every point moves together.

Templates (8x8/C3, serial submission; `build_family_tasks_probe.py`):

* `conv` - one CNA Conv task (duplicated N times);
* `pool` - Conv + DPU pool, duplicating the pool descriptor;
* `elem` - Conv + DPU elementwise Mul (runtime-scale profile), duplicating the Mul.

All 18 variants are **exact** on the board over 64 runs each (`family_tasks.json`).
Least-squares fits (min-based, which is the board floor; the median fits are
load-dominated and shown only for contrast):

| family | intercept | cost per task | R2 |
| --- | --- | --- | --- |
| `conv` (CNA) | 6.7 us | **12.2 us** | 1.00 |
| `pool` (DPU) | -4.5 us | **15.4 us** | 0.98 |
| `elem` (DPU) | -22.2 us | **20.3 us** | 0.97 |

The near-zero intercepts say the per-inference fixed cost is a few microseconds once
the tasks are already in flight serially; the ordering conv < pool < elementwise
matches the program sizes (126 / 37 / 78 register words) and the DPU's extra store
work. These are the numbers the plan's S5 table asked for; the earlier container
differencing failed because a 25-45 us fixed cost and +-15 us jitter swamped a
few-microsecond task marginal.

Median fits for the same data: conv `35 + 17.8/task`, pool `-311 + 124/task`,
elem `-705 + 220/task` - negative intercepts, i.e. medians carry up to 9x load noise
and are not the right estimator here (a second 64-run measurement over 23 held-out
containers repeats the minimum within 1.01x median while the median is 1.24-6.23x it,
`../family_cost_crosscheck/`).

**Scope (cross-checked 2026-09-11):** the CNA row is the one that generalizes to real
dependency chains (13.4-13.9 us/task measured against 12.2 here). The `elem` row is
specific to its runtime-scale template: the two-surface elementwise tasks the DAG
suites run cost 11.8-14.5 us/task, and the `pool` row is not identifiable from real
containers. See `../family_cost_crosscheck/`.

Reproduction:

```sh
PYTHONPATH=src python3 research/build_family_tasks_probe.py
PYTHONPATH=src python3 research/run_family_tasks_probe.py --iterations 64
PYTHONPATH=src python3 -m pytest tests/test_family_cost.py -q
```
