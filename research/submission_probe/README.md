# Batched task submission on RV1103: measured rule and evidence

> **Corrected (2026-09-10).** The link values in this probe were written as
> `payload + offset` instead of `offset`, so every linked case jumped past its
> program. The conclusions below that "links do not help" and that a batched job is
> limited to two tasks are **wrong**; see [`../group_probe/`](../group_probe/README.md)
> for the corrected measurements (links are required, single engine, depth up to the
> loader's 64 tasks). This directory is retained as the *unlinked* counter-example:
> a batched list whose tasks are terminal runs its first task and times out.

The NPU front end accepts a whole task list per ioctl (`PC_TASK_CONTROL =
(0x6 | task_pp_en) << pc_task_number_bits | task_number`, ping-pong from
`JOB_PINGPONG`), and `runtime/open_rknpu.c` already submits that way whenever the
container header says `serial=0`. Almost every verified container is serial, so this
probe asked what a batched job actually requires.

## Method

Every case is a **verified container** with its submission flag flipped. The mode is
header bit 0 of byte 84, so flipping it and re-checksumming leaves the programs,
tensor table and arena byte-identical: an output or timing difference is caused by
the submission mode alone.

* `research/build_submission_probe.py` builds the twins and `manifest.json`.
* `tests/board_bench.c` (`board_bench <model> <input> <iters> <expected>`) repeats the
  inference and reports min/median/mean/max plus `unstable` (runs differing from the
  first run) and `mismatches` (runs differing from the integer reference).
* `research/run_submission_rule.py` runs every case after a **clean boot**, because a
  timed-out job leaves the NPU wedged until reboot, and writes `rule_results.json`.

## Result

| case | tasks / engines | submission | result |
| --- | --- | --- | --- |
| `chain_output_quantization_suite` | 2 / CNA+CNA | batched (committed, linked) | **PASS**, 16/16 exact and stable |
| `mnist_pool_suite` | 2 / DPU+CNA | batched flag twin | timeout (-110) |
| `mul_broadcast_mode_suite` | 2 / DPU+CNA | batched flag twin | timeout (-110) |
| `runtime_scale_suite` (v5) | 2 / CNA+DPU | batched flag twin | timeout (-110) |
| `native_chain_suite` | 3 / CNA+CNA+CNA | batched + next-command links | timeout (-110) |
| `pool_join_suite` (v5) | 6 / CNA+DPU | batched flag twin | timeout (-110) |
| each of the above | - | committed serial | PASS |

`rule_results.json` holds the per-case bench lines. The driver log for a mixed case
reads `task counter: 1` with `require mask: 0x300`: the front end completes the first
task and never signals the rest.

## What was ruled out as the cause

* **Program layout** - padding the programs onto the vendor's fixed 0x440 grid still
  times out, so the front end is not walking a fixed stride.
* **Next-command links** - setting `0x10 = next program` with control `0x40` (exactly
  the layout `graph.py` uses for its two-layer chain) does not help a three-task list.
* **Interrupt mask** - per-task masks and the job-union mask `0xF00` both time out.
  The driver's `rknpu_fuzz_status` collapses the 0x300 (CNA) and 0xC00 (DPU) status
  groups, so one job mask cannot cover both engines; the union value is also rejected
  by the strict loaders, and relaxing them changed nothing.

## Consequences

* `open_rknpu.scheduler.batched_supported` encodes the rule: batched submission is
  accepted only for a two-task single-engine list. **[Superseded 2026-09-11 (S8):** the
  limit was the tail control word, not the engine count; see `../job_field_probe/`.
  The 67 verified CNA+CNA containers
  keep working; serial emission is byte-identical across the campaign.
* `compile_sequence(..., submission='batched')` refuses anything else with that
  reason, so no profile can ship a container that hangs the board.
* No latency win was measured for the accepted case at 8x8 (median 488 us batched vs
  49 us serial over 16 runs, both dominated by scheduling noise on a shared core), so
  the value of task lists would come from engine-group splitting at larger sizes
  (`docs/plans/pipelining-plan.md` S2), not from this two-task case.

## Follow-ups (tracked in docs/plans/pipelining-plan.md)

1. Decode what `PC_DMA_BASE_ADDR` / `subcore_task[]` expect for 3+ task lists - the
   vendor user space fills per-engine subcore lists that our runtime leaves zeroed.
2. Split a graph at engine boundaries and submit one same-engine job per group, with
   the existing synchronous wait between groups; measure the ioctl count and latency.
