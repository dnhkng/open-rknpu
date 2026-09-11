# Per-family cost table cross-check on held-out containers (S5)

`../family_tasks_probe/` fitted the per-family cost by repeating one idempotent task
descriptor N times inside a container, which is robust to load jitter but measures
*independent, overlapping* tasks. This probe asks the independent question: does that
table predict real dependency chains whose task mix it never saw?

`build_family_cost_crosscheck.py` selects **23 already board-verified containers** from
eight suites purely by their (conv, pool, elementwise) task counts - read from the task
enable masks - covering the mixes `(1,0,1)`, `(1,1,0)`, `(1,2,0)`, `(2,0,2..4)`,
`(3,0,1)`, `(3,0,3)`, `(4,0,2..3)`, `(5,1,2)`, `(5,3,2)`, `(6,1,2..3)`, `(8,0,8)`,
`(8,1,3)`, `(10,3,2)`, `(16,0,16)` and the pure-CNA chains 3/4/8/12/16. Each is then
benched with `board_bench` (64 runs, serial, min/median/mean/max, exactness against
its retained expected file). Two independent runs are retained
(`crosscheck_run1.json`, `crosscheck_run2.json`); **all 23 containers are exact in
both**.

## Result

| | table (duplicated idempotent tasks) | held-out fit, run 1 | held-out fit, run 2 |
| --- | --- | --- | --- |
| intercept | 6.7 us (mean) | 6.3 us | 15.9 us |
| conv (CNA) | **12.2** | 13.9 +- 1.0 | 13.4 +- 0.9 |
| pool (DPU) | 15.4 | 6.6 +- 4.5 | 4.7 +- 3.9 |
| elementwise (DPU) | 20.3 | 14.5 +- 1.3 | 11.8 +- 1.2 |
| R2 | - | 0.964 | 0.966 |

Least squares over the held-out set alone (one intercept plus three per-task costs):

* **the CNA row cross-checks.** Real chains give 13.4-13.9 us/task against the table's
  12.2 us - within the run-to-run spread of the floor statistic (below), and the only
  family whose table value transfers;
* **the elementwise row is profile-specific, not family-wide.** Real two-surface EW
  tasks cost 11.8-14.5 us/task against the table's 20.3 us. The family key (enable 24,
  78 words) is the same, but the operands are not: the table's template is the
  *runtime-scale* Mul (`0x5018=0`, `0x5034=1`, i.e. a 16-byte per-channel operand),
  while the DAG suites run two-surface EW tasks (`0x5018=0x3000`,
  `0x5034=0x40000004`, `0x5038`/`0x5040` naming a second surface). The register-count
  key does not determine the cost;
* **the pool row stays unidentifiable** from real containers: pool appears only with
  conv 1/5-10 and small counts, so its coefficient carries a +-4 us error.

Slope-only table prediction (no intercept) against the measured minimum: median
**-4.4%** (run 1) / **-6.8%** (run 2), mean +7.5% / +9.9%, with the worst cases being
the short pure-CNA chains (`chain_multi_suite`, +124% / +132%): real short chains pay a
per-container fixed cost of roughly 15-35 us that the duplication probe cannot see,
because its copies are independent and overlap in the hardware pipeline, while a real
chain serializes on each hand-off.

## Noise floor

The same 23 containers were measured twice. The **minimum repeats within 0.90-1.40x
(median 1.01x)**, while the **median is 1.24-6.23x the minimum** (median 1.68x) at 64
runs. That is the quantified reason the table is fitted on minima: the median is a
board-load statistic, not a floor. `tests/test_family_cost.py` pins the two runs, the
exactness, the repeatability window, the table-vs-held-out comparison above, and that
the recorded predictions are recomputed from `family_tasks_probe/family_tasks.json`.

## Reproduction

```sh
PYTHONPATH=src python3 research/build_family_cost_crosscheck.py
PYTHONPATH=src python3 research/run_family_cost_crosscheck.py --iterations 64 \
    --output crosscheck_run1.json   # repeat with run2
PYTHONPATH=src python3 -m pytest tests/test_family_cost.py -q
```

The probe contains no vendor artifacts; every container it benches is the retained
board-verified model of the suite it names.
