# NPU utilisation plan: task lists, overlap and tiling

Status: 2026-09-11. This plan is the ordered queue for using the parts of the NPU
that the verified profiles currently leave idle. It follows
[completion-plan](completion-plan.md) (P0–P10 complete) and uses the same
evidence rule: an accepted mode needs independently generated commands, a host
regression test and an exact RV1103 board run; a rejected mode needs a retained
failing experiment or a documented prerequisite. S0–S8 are done; S9 (the chain
family's dropped hidden Relu, found while doing S7) is an accepted semantics change
whose board runs still have to be re-issued.

## 1. What is idle today

The NPU is a **task queue feeding fixed-function engines** (CNA convolution, DPU
pool/elementwise/activation, DMA), not programmable cores. A task is a block of
register values in DDR (`struct rknpu_task { op_idx, enable_mask, int_mask,
int_clear, regcfg_addr, regcfg_amount }`); the driver writes `PC_DATA_ADDR`,
`PC_DATA_AMOUNT`, `INT_MASK`/`INT_CLEAR` and `PC_TASK_CONTROL =
((0x6 | task_pp_en) << pc_task_number_bits) | task_number`, then `PC_OP_EN`
(`research/vendor/rknpu_job.c`). One register write starts up to 2^16−1 tasks on
RV1106, and `task_pp_en` (job flag `JOB_PINGPONG`) double-buffers the register
command fetch against execution.

Measured state of our containers:

| container kind | count | submission |
| --- | --- | --- |
| `ORNPUBIN` v1/v2 (single program, no submission flag) | 377 | n/a |
| `ORNPUSEQ` v3, 1 task | 836 | serial |
| `ORNPUSEQ` v3, 2 tasks | 429 | **67 batched** (all `chain_output_quantization_suite`, CNA+CNA) / 362 serial |
| `ORNPUSEQ` v3, 3 tasks | 444 | serial |
| `ORNPUSEQ` v3, 4–6 tasks | 51 | serial |
| `ORNPUSEQ` v3, 12–48 tasks | 7 | serial |
| `ORNPUSEQ` v4 | 2 | serial |
| `ORNPUSEQ` v5 named-tensor graphs | 209 | **all serial** |

So a batched job was only ever emitted for **67 two-task, single-engine (CNA+CNA)
containers**. What is *not* proven: batched submission of v5 named-tensor graphs,
mixed-engine lists, deep task lists, any form of **surface tiling**, and **cross-job
pipelining** (`JOB_FENCE_IN/OUT`, `JOB_NONBLOCK`). The runtime supports all of these
(`submit_tasks()` submits `number = task_count` in one ioctl when `serial=0`); the
compiler opts out for v5 and never tiles. S1 measured what the hardware actually
allows (see below).

Known hazard to respect: two independent board sightings of **stale data** when one
arena slot was written by two different task families (`docs/investigation-log.md`).
That is consistent with real inter-task overlap, and it is why the newest profiles
give every internal a fresh slot. Understanding that rule is phase S1.

## 2. Phases

### S0 — Instrumentation (no new hardware behaviour)

*You cannot tune what you cannot measure.*

- `tests/board_bench.c`: `board_bench <model> <input> <iters>` runs `ornpu_run`
  in a loop with `clock_gettime(CLOCK_MONOTONIC)`, reports min/median/mean µs and
  the inferred per-task cost; optionally pins to one core if the libc allows.
- Record baseline latency for a serial v5 model, a batched legacy 2-task model and
  a batched legacy 5-task model, with repeat stability.
- Deliverable: `tests/board_bench.c` (kept, used by every later phase), with
  `ornpu_info.task_count` / `submission_serial` added to the C API.
- Baseline (8x8/C3 profiles, median of 16-32 runs, board otherwise idle): a
  two-task legacy conv job completes in ~30-60 us; a three-task conv chain in
  ~78 us; the six-task v5 pool join in ~125 us; the seven-task v5 pooled-branches
  graph in ~120-140 us. Individual runs scatter (max 1-13 ms) because the CPU and
  the camera service share the core, so later phases compare medians of >= 32 runs.
- Gate: builds with `-Werror`, repeated runs agree within a few percent.

### S1/S2 — Batched submission: corrected rule and measured value (done 2026-09-10)

*Result: one ioctl can carry a whole same-engine task list, provided every task links
to the next program; it is much faster than serial for deep runs (up to 11x on a
12-layer chain) and slower for very short ones. Serial stays the default.*

Tooling:

* `tests/board_bench.c` — repeats one inference, reports min/median/mean/max, compares
  every run against the expected bytes **and** against the first run (`unstable`
  catches ordering hazards), and prints the task count and submission mode.
  `ornpu_info` gained `task_count` and `submission_serial`.
* `research/submission_probe/` — the first probe: submission-flag twins whose
  programs, tensor table and arena stay byte-identical.
* `research/group_probe/` — the corrected probe: job-shape cases, the depth probe and
  the serial-vs-batched crossover.
* `research/deep_chain_suite/` + `research/run_deep_chain_compare.py` — a real,
  independently generated 8/12/16-layer Conv chain emitted both ways.

**Correction.** The first probe wrote the next-command link as `payload + offset`
instead of `offset`. The link is payload-relative (the runtime programs
`PC_DMA_BASE_ADDR` with the payload base), so every linked case jumped past its
program and timed out; the first conclusions - "links do not help" and "a job is
limited to two tasks" - were wrong and are superseded here. The wrong cases are kept
in `submission_probe/` as the *unlinked* counter-example.

**Rule (measured, clean boot per case).**

| case | shape | result |
| --- | --- | --- |
| 2 CNA, terminal tails | batched | timeout (-110) |
| 3 CNA, correctly linked | batched | **PASS**, 32/32 exact |
| 4 tasks over a program loop, linked | batched | **PASS**, 32/32 exact |
| 8 / 32 / 64 tasks, linked loop | batched | **PASS** at every depth |
| v5 6 tasks CNA+DPU, linked, per-task masks | batched | timeout (-110) |
| v5 6 tasks CNA+DPU, linked, union mask `0xF00` | batched | timeout (-110) |

So a batched job runs its whole list when every task links to the next program
(`0x10` = next payload-relative program offset) and every transition carries its engine
hand-off control; depth is bounded only by the loader's 64-task table (the driver's
`max_submit_number` is 65535). **S8 corrected this section's mixed-engine conclusion:**
the six-task CNA+DPU cases above wrote `0x40` at the CNA->DPU transition, which times
out; `0x14` is the hand-off, so a CNA-run-then-DPU-run list is a valid one-job
submission (see S8 below).

**Value.**

| graph | serial | one batched job | speedup |
| --- | --- | --- | --- |
| 3-conv chain (3 tasks) | 67 us | 116 us | 0.6x |
| 8-layer chain (`deep_chain_suite`) | 1325 us | 194 us | 6.8x |
| 12-layer chain | 1932 us | 176 us | 11.0x |
| 16-layer chain | 345 us | 163 us | 2.1x |
| 32-task loop job | 3172 us | 229 us | 13.8x |
| 64-task loop job | 1528 us | 449 us | 3.4x |

The cost model is roughly `serial ~ 22 us/task` plus a per-ioctl cost against
`batched ~ 100 us + 8 us/task`, so the crossover is near eight tasks in one
same-engine run. The deep chains were verified exact in **both** modes over 128 runs.
Serial therefore remains the default; a profile with a long run requests
`submission='batched'`, and `open_rknpu.compose` and the N-layer chain emitter write the
links and the measured hand-off control for exactly that shape, refusing a graph whose
schedule needs an unmeasured hand-off (DPU->CNA) with a specific error
(`open_rknpu.compose.batched_layout` checks the emitted container either way).

**Deep task-list limits.** Depth was verified to 64 tasks (the loader's `MAX_TASKS`);
a 64-task job completes in 449 us, so the practical bound is the container table, not
the hardware. `PC_TASK_CONTROL` carries the task number in 16 bits.

### S3 — Double buffering and height-strip tiling (done 2026-09-10)

*Result: double-buffered intermediates (two surfaces per chain instead of one per
layer) are verified exact on the board and roughly halve deep-chain latency while
flattening the arena. Height-strip tiling is not implementable in the current chain
emitter and would not overlap on this IP; the prerequisite is stated below.*

**Double buffering (done).** The N-layer chain emitter supports
`reuse_intermediates=True`: layer `L` writes buffer `(L-1)%2`, so only two
intermediate surfaces are ever live. New twins in `research/deep_chain_suite/`
(`reuse/`, `reuse_batched/`) cover the 8/12/16-layer chains, verified on the board
with `tests/board_bench.c`, 128 runs each:

| layers | arena untiled | arena double-buffered | serial | double-buffered serial | one batched job | reuse + batched |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 28,672 B | 20,480 B (71%) | 369 us | **184 us** | 93 us | 94 us |
| 12 | 36,864 B | 24,576 B (67%) | 1537 us | **721 us** | 240 us | 218 us |
| 16 | 40,960 B | 24,576 B (60%) | 1120 us | **645 us** | 256 us | **186 us** |

All four variants are byte-exact against the same expected outputs, so the win is
memory behaviour: with two 1 KB surfaces live instead of one per layer the 8x8 chain
stops streaming 16 distinct buffers through DDR. The arena stops growing with depth
(24,576 B for 12 and 16 layers), which is what the 4 MiB arena bound needs for real
models.

**Height-strip tiling (implemented and verified).** `open_rknpu.tiled_chain.
compile_tiled_chain(model, tiles=T)` splits the 8x8 height into `T` strips: one native16
program per (layer, strip) built through the extracted `native.native_fields(...)` with
strip geometry, reading the strip's rows of the previous surface and writing its rows of
the next, with one double-buffered pair of strip surfaces per strip. Quantization,
weights and bias come from the untiled `chain_n` emitter, so both containers share one
reference and one expected file. `open-rknpu compile --sequence --tiles N` and
`compile_sequence(..., tiles=N)` expose it.

Board evidence (`research/tiled_chain_probe/`, 16 cases each, `tests/board_bench.c`,
identical input/expected bytes as the untiled 16-layer container):

| tiles | tasks | strip rows | min | median | mismatches |
| --- | --- | --- | --- | --- | --- |
| 1 | 16 | 8 | 218.5 us | 287.6 us | 0 |
| 2 | 32 | 4 | 432.8 us | 552.4 us | 0 |
| 4 | 64 | 2 | 831.0 us | 1210.4 us | 0 |

All three reproduce the untiled expected bytes exactly, so the strip geometry, the
per-strip double buffering, the 16-lane weight/bias packing and the quantization chain
are correct. The measured trade at 8x8: latency grows roughly linearly with the task
count (288 -> 552 -> 1210 us) and the arena grows too (28,672 / 45,056 / 81,920 B
against 40,960 B untiled), because a task program is 1088 B and there is no engine to
overlap with. Tiling is therefore an available option, not a default; its memory win
needs maps where surface bytes dominate program bytes.

Remaining: `K > 1` strips need the halo rows copied into each strip's input surface (the
emitter refuses non-1x1 kernels), and a strip pipeline cannot overlap on this IP for the
reason above.

### S4 — Cross-job pipelining: non-blocking submission verified; fences blocked on kernel config (done 2026-09-10)

*Result: `JOB_NONBLOCK` is accepted and a queue-then-drain pipeline is 1.5-2.25x
faster than synchronous submission with exact outputs. `JOB_FENCE_OUT`/`FENCE_IN`
return -EINVAL, so the running kernel lacks `CONFIG_ROCKCHIP_RKNPU_FENCE`; that is the
documented prerequisite for a single-instance stream that reads each output as it
completes.*

The remaining "tune the pipeline with the clock" idea is closed by measurement: the
driver ACTION surface is audited in `research/action_probe/README.md` - `GET_FREQ`
reports 420 MHz, `SET_FREQ` has an empty body in driver v0.8.2, the bandwidth-policy
actions return `-EINVAL`, there is no SRAM pool and `GET_VOLT` oopses on this board
(no regulator in the device tree), so frequency/voltage experiments are not available.

Primitives added to `runtime/open_rknpu.h` (experimental, no change to the
synchronous path): `ornpu_submit_flags(model, extra_flags, *fence_fd)` (`0x2`
NONBLOCK, `0x8` FENCE_IN, `0x10` FENCE_OUT), `ornpu_wait_fence(fd, timeout_ms)`
(`poll`), `ornpu_sync_outputs(model)`, and `ornpu_set_input(model, index, data, size)`
so the next input can be packed while the NPU works.

Measurement (`tests/board_async.c`, 48 interleaved rounds, one round = two
inferences, median of paired rounds so board-load drift cancels):

| container | synchronous pair | pipelined pair | ratio | outputs |
| --- | --- | --- | --- | --- |
| 3-conv chain, one batched job | 195 us | 87 us | **2.25x** | exact |
| 3-conv chain, three serial jobs | 220 us | 148 us | **1.49x** | exact |

The pipelined round queues inference A with `NONBLOCK`, runs inference B blocking -
whose completion implies A's, because the driver runs one job at a time per core in
order - and then reads A's outputs. `NONBLOCK` returned 0 and the drained output
matched both the reference and the expected bytes for every container tested.

**Fence-free completion: the barrier job (done 2026-09-11).** `JOB_FENCE_OUT` and
`JOB_FENCE_IN` still return -EINVAL (`rc=-22`, re-checked): the kernel was built without
`CONFIG_ROCKCHIP_RKNPU_FENCE`. A *pollable completion fd* therefore does not exist, but
the functional requirement - a single-instance stream that reads each output as soon as
it is produced - is met without it, because jobs run in order per core: queue the
inference with `ORNPU_JOB_NONBLOCK`, then run a small **barrier** container blocking and
sync the queued model's outputs. The barrier completes only after the queued work, so
the lag is zero at the cost of one small job per inference
(`runtime/open_rknpu.h`, `research/barrier_probe/`, `tests/board_barrier.c`):

| container | tasks | synchronous | barrier-completed | delta |
| --- | --- | --- | --- | --- |
| 7-task serial mixed-engine graph | 7 | 183.5 us | **134.5 us** | **-27%** |
| 16-task one-job chain | 16 | 107.3 us | 134.5 us | +25% (the 24 us barrier) |

For a serial container the pattern is *faster* as well as lag-0 (one barrier wait
replaces N blocking ioctl round trips); for an already-batched container it is pure
overhead, and the lag-1 two-instance drain pipeline (1.13x here, 1.5-2.25x in
`research/async_probe/`) stays the better throughput choice. What the barrier cannot
provide - sharing the NPU between processes and dma-buf style producer/consumer sync -
still needs the kernel config, which remains a deployment decision.

### S5 — Engine cost model (done: fixed/marginal for CNA and a per-family table)

*Delivered: a fixed/marginal model for a homogeneous CNA run, and a measured answer for
the per-family table: the differential is below this board's noise floor and needs
single-family multi-task containers.*

Measured with `tests/board_bench.c` (128 runs per case):

* one batched job of N CNA tasks: min 30.6 us (N=3), 47.5 us (N=8), 111 us (N=32),
  206 us (N=64) - roughly a 30-90 us fixed cost plus **3-7 us per task**;
* serial submission: min 46 us (N=3), 108 us (N=8), 454 us (N=32), 944 us (N=64) -
  about **14 us per task** plus the per-ioctl round trip, and medians swing up to an
  order of magnitude because N blocking submissions interleave with CPU work;
* crossover near **four to eight tasks** per same-engine run, matching the paired
  deep-chain results (3 tasks: batched loses; 8+ layers: batched wins 2-6x).

**Per-family table (done).** `research/family_tasks_probe/` repeats a verified task
descriptor N times inside one container (idempotent, so exactness is preserved) and
fits `cost(N) = C + N * task` over N = 1..32. All 18 variants are exact over 64 runs
each; min-based fits, which are the board floor:

| family | intercept | per task | R2 |
| --- | --- | --- | --- |
| conv (CNA) | 6.7 us | **12.2 us** | 1.00 |
| pool (DPU) | -4.5 us | **15.4 us** | 0.98 |
| elementwise (DPU) | -22.2 us | **20.3 us** | 0.97 |

Median fits for the same data carry up to 9x load noise (negative intercepts), which
is why the earlier container-differencing attempt (`research/family_cost_probe/`)
failed; the slope method is the one that resolves it.

**Cross-check (done 2026-09-11).** `research/family_cost_crosscheck/` benches 23
*held-out* real containers selected only by their (conv, pool, elementwise) task counts
(two independent 64-run board runs, all exact). The CNA row transfers: real chains give
**13.4-13.9 us/task** against the table's 12.2, inside the run-to-run spread of the
minimum (0.90-1.40x over the same containers). The elementwise row does **not**: real
two-surface EW tasks cost 11.8-14.5 us against the 20.3 us runtime-scale template, so
the row is profile-specific even though the enable/register key matches. The pool row
stays unidentifiable (+-4 us) because pool only ever appears with one to three copies.
Slope-only predictions land within a median -4 to -7% (mean +8 to +10%) overall, and
short pure-CNA chains cost up to 2.3x the prediction: real chains pay a per-container
fixed cost that idempotent, overlapping duplicates hide. At 64 runs the minimum repeats
within 1.01x (median) while the median is 1.24-6.23x the minimum - the measured reason
the table is fitted on minima.

### S6 — Integration (done 2026-09-10)

* `open-rknpu compile --sequence --submission serial|batched` (serial default): the
  batched value reaches every emitter, and the scheduler relinks whatever container it
  produced (`sequence.relink_for_batched`), so a mixed-engine DAG is one job and the mode
  is never refused for a compilable profile (`tests/test_cli.py`).
* The ledger carries a `deep_chain` row (3 models / 48 inferences / 9,216 exact bytes,
  `board_results_0.json`), and the totals in `README.md`, `docs/plans/completion-plan.md`,
  `docs/plans/project-goals.md` and the investigation log agree with the row sum
  checked by `tests/test_ledger.py` (115 rows).
* `runtime/sequence_format.md` documents the three submission shapes now measured:
  serial per task (default), one linked same-engine job, and non-blocking
  queue-then-drain, including the engine/link/depth rules and the fence prerequisite
  (`CONFIG_ROCKCHIP_RKNPU_FENCE` absent on the attached kernel).
* Host regression pins: `tests/test_submission.py` (rule, emitter, deep-chain twins and
  double buffering, retained evidence), `tests/test_async.py` (primitives and S4
  evidence), `tests/test_cli.py` (CLI), plus the existing suite/ledger/binding tests.
* Packaging rebuilt with the new CLI and runtime sources.

### S7 — `K > 1` strip tiling with halo rows (done 2026-09-11)

*Result: `compile_tiled_chain` covers the whole `chain_n` family (padded 1x1 and 3x3) and
is byte-exact against the untiled chain on the board, serially and as one linked job.*

The K=1 path needs no halo: a strip's input rows are contiguous in its own strip buffer.
For `k = 3` a strip's output rows `[s*r, (s+1)*r)` need input rows
`[max(0, s*r - h), min(8, (s+1)*r + h))` with `h = (k-1)//2`, and those halo rows are
produced by the **neighbouring strips**, so isolated per-strip buffers cannot hold them.
The K>1 path therefore gives the layer one shared double-buffered native16 intermediate
(`2 * 8*8*16 B`); each strip writes only its own rows and reads its window straight out of
that surface, with `tpt`/`pl` supplying the zero-point rows at the image edge - the same
geometry `native.compile_native_input` already uses for its 6144-atom height tiles. No
copy task and no extra surface bytes are needed at 8x8 (two full surfaces either way);
the value is a bounded per-task input fetch and a general emitter. K=1 keeps the isolated
strip layout, so the earlier S3 containers and their comparison remain valid.

Delivered:

1. Weight/bias packing generalized to `k*k` taps (`(t*oc + o)*16 + c`, size
   `align(oc*16*k*k)`) - byte-identical at `k = 1`.
2. Per (layer, strip) fields through `native_fields(...)` with the strip geometry above,
   `pl = h`, `scan_flags = 4 if k == 1 else 8`, reading the shared previous surface.
3. Bounds: `chain_n`'s 1x1/3x3 padded kernels, hidden 3..16, `tiles` in {1,2,4,8}, and the
   loader's 64-task table (16 layers x 8 strips is refused with that reason).
4. Batched tiling writes the next-command link per task (control `0x40`, all CNA, last
   terminal), so a tiled container is a valid one-job submission
   (`compose.batched_layout` accepts it).
5. Board evidence: `research/tiled_k3_probe/` (16-layer 3x3 chain and a mixed 1x1/3x3
   chain, tiles 1/2/4 serial and batched, 4 input cases each, all exact) and the S3 suite
   re-run over all 16 input cases (`research/tiled_chain_probe/`).

| chain | tasks (tiles=4) | serial min | one linked job min | exact |
| --- | --- | --- | --- | --- |
| 16-layer 3x3 (`k3`) | 64 | 919 us | 266 us | yes |
| 12-layer mixed K1/K3 (`mixed`) | 48 | 692 us | 169 us | yes |
| 16-layer 1x1 (S3 re-run) | 64 | 848 us | 151 us | yes |

Two board-found fixes are pinned by `tests/test_tiled_chain.py`:

* hidden native surfaces store the 128-shifted INT8 value, so the engine input zero point
  is 0 in that domain: register `0x1184 = 0xff80`, the untiled chain's own default. The
  first version passed the previous layer's `output_zero_point`, which only shifts the
  padded rows of a 3x3 strip - and made every tiled case mismatch.
* the activation registers must follow the layer's quantization metadata (see S9).

### S8 — Job fields: `subcore_task[]`, `core_mask`, and the tail control word (done 2026-09-11)

*Result: the task-tail control word encodes the engine group transition; with the right
control a mixed-engine list runs in one job. `subcore_task[]`/`core_mask` are inert on
this SoC, as the driver source says.*

Static decode (`research/vendor/rknpu_job.c`, `rknpu_ioctl.h`):

* `subcore_task[]` is read only under `config->num_irqs > 1`, and `rknpu_job_alloc`
  forces `core_mask = CORE0` when `num_irqs == 1`. RV1106 uses the single-entry
  `rknpu_irqs`, so sub-core routing is not requestable on this SoC.
* `PC_DMA_BASE_ADDR` is written from `args->task_base_addr`; our runtime passes the
  payload DMA base, which is what makes the next-command link payload-relative.

Measured on the board (`research/job_field_probe/`, one clean boot per attempt, verified
serial containers changed only in the submission flag and their tail words):

* `core_mask`/`subcore_task[]` inertness (`tests/board_core.c`,
  `ornpu_set_submit_core`): `core=0x7`, five nonzero windows and `core=0x1` all reproduce
  the baseline output bytes;
* the tail control word: it is the **successor program's fetch amount**, not an engine
  hand-off code - `PC_DATA_AMOUNT = (regcfg_amount + 4 + 2 - 1)/2 - 1`, i.e. `0x40` before
  a 126-word Conv, `0x14` before a 37-word pool, `0x28` before a 78-word elementwise task
  (`compose.amount_control`, matching the vendor captures). The first version of this
  section read those coincidences as an engine table and claimed Conv->elementwise,
  elementwise->elementwise, elementwise->Conv and Pool->Conv had no encoding; the sweeps
  behind that were confounded (they set both transitions of the probed run to the same
  value, never tried `0x28`, and probed one transition on top of a run whose prefix was
  already wrong). Single-transition experiments with the successor's amount pass every
  one of them (`research/grouped_probe/`, `kind="discovery"`); the withdrawn table is
  kept in `research/job_field_probe/README.md`.

Because the control is a fetch size, every transition links: `compose` and the DAG
emitters write the successor's amount and the entire list is **one job**, which is what
`research/mixed_batched_probe/` (composer profiles) and `research/grouped_probe/`
(published section: 8 containers, 4-9 tasks) verify on the board.

### S9 — Hidden activation dropped by the chain family (found by S7, fixed 2026-09-11)

*Finding: `chain_n` emitted its native hidden layers with the activation registers off
(`0x4060 = 0x13`, `0x406c = 0x40e0 = 0x80000000`) and `chain_n_reference_layers` used a
reference that ignored the Relu flag, so a `Relu` after a native layer was dropped by
both the container and its expected bytes. The legacy first layer did apply its Relu.*

Measured before the fix:

* `capture_native_h64k5` shows the vendor compiler writing the activation-on form
  (`0x4060 = 0x12`, `0x406c = 0x40e0 = 0`), so the encoding is right and native16 Relu is
  a supported hardware path.
* A/B on the board, patching only the activation registers of an otherwise exact tiled
  3x3 chain: activation off = exact, activation on = mismatch against the untiled
  expected bytes. For the same A/B on a 1x1 tiled chain both are exact: on this IP the
  single-row (K=1) scan path ignores the clamp registers, so a K=1 hidden Relu is a
  no-op either way and only K>1 hidden Relus were silently dropped.
* Patching the affected containers changed exactly three registers per hidden layer
  (`0x4060`, `0x406c`, `0x40e0`) in exactly four suites; `chain_output_quantization`
  keeps its last layer activation off, as it should.

Fixed (the semantics change the finding called for):

1. `chain_n` sets each native layer's `q.relu = index < count - 1` (the graph is
   `[Conv, Relu]*(N-1) + [Conv]`) and writes the activation registers for the layers that
   carry one; the first layer keeps the legacy emitter's Relu.
2. `chain.native_reference` clamps the accumulator (`if q.relu: acc = maximum(acc, 0)`)
   before the output conversion, which is where the hardware clamps (measured), so the
   composed reference and the container stay together. No other caller passes a
   relu-flagged native quantization, and the serial emission of every other suite is
   byte-identical.
3. `tiled_chain` already took each layer's activation from its quantization metadata, so
   the tiled containers now apply the hidden Relu too and remain a drop-in tiling.

Board re-runs (all four suites rebuilt, 16 (8 for `chain_multi`) inputs per model,
`tests/board_api.c` / `tests/board_io.c`): `native_chain` 5 models/80 inferences/15,360
exact bytes, `chain_reuse` 5/80/15,360, `deep_chain` 3/48/9,216, `chain_multi` 4/32/40,448
- the same counts as the ledger rows, now with the hidden Relu applied - plus the deep
chain's serial/batched/reuse twins (`deep_chain_suite/batched_results.json`) and both
tiled probes (`research/tiled_chain_probe/`, `research/tiled_k3_probe/`), all exact.

### S10 — Threading grouped submission through the DAG emitters (done 2026-09-11)

*Result: every DAG emitter now honours `submission='batched'`, and because the tail
control is the successor's fetch amount (S8, corrected) every one of those graphs is a
**single job** - the runtime derives runs from the link words, and all links are present.*

What changed:

1. `compose` links every stage and writes `amount_control(FAMILIES[successor].words)`;
   `batched_layout` validates against the successor's register count instead of a
   transition table, and `meta['engine_runs']` reports the run structure.
2. `graph.handoff_tails(data, tasks, serial)` is the one tail writer for hand-built
   containers: it links each task to the next program with the successor's amount. It is
   used by the join-chain/mixed-head emitter and by the three emitters threaded here -
   `graph.compile_diamond` (diamond / diamond-tail),
   `depthwise_join.compile_depthwise_join`, and `join_dag.compile_join_dag` (join-DAG and
   pooled-DAG) - each of which gained a `serial` parameter that the scheduler passes from
   `submission`.
3. Serial emission is byte-identical everywhere: every threaded emitter's serial compile
   reproduces its suite's retained container, and the campaign sweep is unchanged.

Board evidence (`research/grouped_probe/`, 4 cases x 16 runs per mode, clean boot per
model; `ioctls` is `ornpu_info.engine_runs`):

| suite | tasks | serial ioctls | one-job ioctls | serial min | one-job min | exact |
| --- | --- | --- | --- | --- | --- | --- |
| `diamond_tail_suite` 000 | 5 | 5 | **1** | 73.8 us | 45.8 us | yes |
| `depthwise_join_suite` 000 | 4 | 4 | **1** | 63.6 us | 54.3 us | yes |
| `pooled_dag_suite` 000 | 8 | 8 | **1** | 170.6 us | 51.0 us | yes |
| `join_dag_suite` 000 | 6 | 6 | **1** | 98.0 us | 45.2 us | yes |
| `pool_join_suite` 000 | 6 | 6 | **1** | 109.7 us | 46.4 us | yes |
| `pooled_branches_suite` 000 | 7 | 7 | **1** | 87.2 us | 42.0 us | yes |
| `mixed_head_suite` 001 | 7 | 7 | **1** | 101.8 us | 50.8 us | yes |
| `join_chain_suite` 005 | 9 | 9 | **1** | 141.5 us | 55.4 us | yes |

The same run's discovery cases (`kind="discovery"`) are the amount-rule experiment: a
single Conv->elementwise transition with `0x28` is exact, the same transition with `0x14`
is the retained timeout counter-example, elementwise->elementwise with `0x28` is exact,
and the vendor's Conv->Pool `0x14` is exact.

The runtime's per-run submission (`compute_runs`) stays: it is what derives the runs from
the tails, and it keeps a container whose tails are all terminal safe (one job per task
instead of the old hang).

**Every remaining emitter (`done 2026-09-11`).** The emitters that build containers by
hand or one task at a time never learned about the mode, so rather than thread `serial`
through each of them the scheduler relinks *whatever* an emitter produced:
`sequence.relink_for_batched(data)` rewrites the tails (next-program link plus the
successor's fetch amount) and header flag bit 0, leaving the programs untouched; a legacy
`ORNPUBIN` container (no task table, no submission flag) is a no-op, which is correct
because one program is one ioctl either way. `research/batched_all_probe/` verifies this
on retained containers whose emitters ignore the request: 3 to 48 tasks, both modes
exact, one job each (`add_suite` 3 tasks 64.8 -> 29.5 us; `mul_batch_suite` 48 tasks
889.0 -> 445.4 us; `mul_batch_broadcast_suite` 32 alternating-engine tasks
409.8 -> 128.6 us), and a sweep of every `model000.onnx` shows 156 of 161 compile with
the mode - the other five fail identically without it, i.e. for profile reasons.
`tests/test_submission.py` pins the post-pass (idempotent, emitter-equivalent, one job)
and the evidence, and `tests/test_cli.py` checks a hand-built profile through the CLI.

**Residual (documented, not claimed):** `CONFIG_ROCKCHIP_RKNPU_FENCE` remains the only
way to get a driver-level completion fd (multi-process sharing and dma-buf sync). The
per-family cost table is cross-checked and scoped: the CNA row transfers to real
chains, the elementwise row is template-specific and the pool row is not identifiable
from real containers (`research/family_cost_crosscheck/`). The utilisation plan has no other open phase: S10's post-pass covers
every container the compiler can emit, and the remaining project residuals live in
`docs/plans/completion-plan.md` - non-power-of-two/mixed LUT bands (index residual measured, no
affine rule), and a single op-level walk that dispatches every normalized op (the composer now
assembles every DAG emitter byte-identically, and `open_rknpu.walk` lowers a linear
Conv/Relu chain with a pool in any position, board-verified over 6 models - while
fan-out/join and elementwise graphs are still profile-matched rather than walked). A vendor-zoo detector, a distinct RV1106 SoC and publication stay user/hardware
decisions.
