# Performance guide

How to measure an `open-rknpu` model, what has actually been measured, how to size a
model against the runtime's memory, and what the attached board cannot do. Every number
on this page comes from a file in this repository, cited inline; nothing is estimated
from a datasheet.

## The reference platform, and what these numbers are worth

All measurements come from one Luckfox Pico Mini B (RV1103 SoC), driver RKNPU v0.8.2,
NPU clock 420 MHz, ARMv7 Cortex-A7 / uClibc, **with the camera service `rkipc` running**
([docs/board.md](board.md)). The CPU is shared with `rkipc`, so absolute microseconds
move between runs and even within a run.

Treat every figure below as a **single-board observation**, not a portable constant:

* the minimum of a 64-run bench repeats within 0.90–1.40× (median 1.01×), but the
  **median is 1.24–6.23× the minimum** (median 1.68×) — the median is a board-load
  statistic, which is why the per-family table is fitted on minima
  ([research/family_cost_crosscheck/README.md](../research/family_cost_crosscheck/README.md));
* the same one-task container showed 45 µs minimum against 403 µs median
  ([research/family_cost_probe/README.md](../research/family_cost_probe/README.md));
* results are only comparable if the hardware, driver and `rkipc` state are unchanged —
  the identity fields are part of the evidence ([docs/board.md](board.md)).

## How a measurement is taken

The runtime exposes no cycle counter, and the driver's bandwidth counters
(`GET_DT_WR_AMOUNT`, `GET_WT_RD_AMOUNT`, `GET_TOTAL_RW_AMOUNT`) are free-running byte
counts, not time ([research/action_probe/README.md](../research/action_probe/README.md)).
Every latency number in this repository is therefore **host-side wall time**:
`clock_gettime(CLOCK_MONOTONIC)` sampled immediately before and after `ornpu_run`
([tests/board_bench.c](../tests/board_bench.c), `now_us()`).

The rules that make repeated measurements comparable:

* **One round is one inference.** `board_bench` times a single `ornpu_run` (or
  `ornpu_run_io`) per iteration; `board_async` deliberately makes one round *two*
  inferences, because it measures a pipeline of two ([tests/board_async.c](../tests/board_async.c)).
* **Warm up first, and report the first sweep separately.** The first inference after
  `ornpu_open` includes cache/DMA warm-up and NPU bring-up: the mel-CNN takes 3.47 ms on
  the first sweep against 2.54 ms averaged over five sweeps
  ([examples/mel-kws/README.md](../examples/mel-kws/README.md)); the Fashion-MNIST
  prefix costs ~14 ms/image on its first run and ~0.28 ms/image once warm
  ([examples/fashion/README.md](../examples/fashion/README.md)). `board_bench` warms up
  `min(iterations/4, 8)` times before it starts the clock.
* **Report min/median/mean/max, not a single number.** `board_bench` prints all four,
  and also prints `unstable=` (runs whose output differs from the first run) and
  `mismatches=` (runs that differ from the expected bytes).
* **Compare paired, interleaved medians to cancel board drift.** `board_async` alternates
  a synchronous round with a pipelined round so that load from `rkipc` hits both modes
  equally, then compares the medians ([tests/board_async.c](../tests/board_async.c)).
* **Fit cost models on minima, compare systems on medians.** The per-family coefficients
  are slope fits over repeated idempotent tasks ([docs/plans/pipelining-plan.md](plans/pipelining-plan.md) S5);
  the shipped recommendations are checked against held-out containers
  ([research/family_cost_crosscheck/README.md](../research/family_cost_crosscheck/README.md)).

| Harness | What it measures | Where |
| --- | --- | --- |
| [`tests/board_bench.c`](../tests/board_bench.c) | one model, N timed inferences, min/median/mean/max, exactness and stability | latency |
| [`tests/board_async.c`](../tests/board_async.c) | synchronous pair vs `JOB_NONBLOCK` pipelined pair, interleaved medians; fence probe | overlap |
| [`tests/board_barrier.c`](../tests/board_barrier.c) | synchronous vs queue-with-`NONBLOCK`-then-barrier, per inference | overlap, lag 0 |
| [`tests/board_io.c`](../tests/board_io.c), [`tests/board_api.c`](../tests/board_api.c) | whole suites: every inference compared byte-for-byte with `expectedNNN.i8` | correctness, not time |
| [`research/run_v5_suite.py`](../research/run_v5_suite.py) | stages a v5 suite, runs `board_io`, writes `board_results_*.json` + `board_summary.txt` | evidence capture |

## Recorded results

### End-to-end model latency

| Model | Configuration | Latency | Source |
| --- | --- | --- | --- |
| mel-CNN, 7 tasks, 3×32×32 in / 8×8×10 out | serial (one ioctl per task) | **3.47 ms first sweep, 2.54 ms mean of 5 sweeps** (1,500 inferences) | [examples/mel-kws/README.md](../examples/mel-kws/README.md), [research/mel_kws_suite/README.md](../research/mel_kws_suite/README.md) |
| MNIST prefix, both Conv on NPU, calibrated | 1 task + CPU suffix | 2.383 ms NPU (mean of 100 images) | [examples/mnist/README.md](../examples/mnist/README.md) |
| MNIST prefix, hybrid (Conv2 on CPU) | 1 task + CPU suffix | 9.740 ms NPU + 9.477 ms CPU | [examples/mnist/README.md](../examples/mnist/README.md) |
| Fashion-MNIST both-Conv, calibrated, 1,000 images | streamed, one process | 1.448 ms NPU + 0.089 ms CPU per image (1.54 ms/image wall) | [examples/fashion/README.md](../examples/fashion/README.md) |
| Fashion-MNIST hybrid, 1,000 images | Conv1 only on NPU | 8.468 ms NPU + 9.544 ms CPU per image | [examples/fashion/README.md](../examples/fashion/README.md) |

The two trained classifiers show the shape of the trade: the NPU prefix is fast, but a
CPU suffix written in scalar C dominates — the Fashion-MNIST hybrid variant is CPU-bound,
its Conv2/ReLU/pool suffix needing ~7.7 ms/image of scalar C, "an order of magnitude more
than the NPU spends on Conv1" ([examples/fashion/README.md](../examples/fashion/README.md)).

### Batched submission versus serial

One ioctl can carry a whole linked task list (the rules and the tail control word are in
[runtime/sequence_format.md](../runtime/sequence_format.md) and
[docs/plans/pipelining-plan.md](plans/pipelining-plan.md) S1/S2/S8/S10). What it is worth,
from the S1 comparison (medians, 128 runs per mode where noted):

| Graph | Serial | One batched job | Speedup | Source |
| --- | ---: | ---: | ---: | --- |
| 3-conv chain (3 tasks) | 67 µs | 116 µs | **0.6× (serial wins)** | [docs/plans/pipelining-plan.md](plans/pipelining-plan.md) |
| 8-layer chain | 1325 µs | 194 µs | 6.8× | [docs/plans/pipelining-plan.md](plans/pipelining-plan.md) |
| 12-layer chain | 1932 µs | 176 µs | **11.0×** | [docs/plans/pipelining-plan.md](plans/pipelining-plan.md) |
| 16-layer chain | 345 µs | 163 µs | 2.1× | [docs/plans/pipelining-plan.md](plans/pipelining-plan.md) |
| 32-task loop job | 3172 µs | 229 µs | 13.8× | [docs/plans/pipelining-plan.md](plans/pipelining-plan.md) |
| 64-task loop job | 1528 µs | 449 µs | 3.4× | [docs/plans/pipelining-plan.md](plans/pipelining-plan.md) |

An independent run of the same 8/12/16-layer chains measured 205/111 µs (1.8×),
666/282 µs (2.4×) and 1513/421 µs (3.6×) — the absolute times differ by up to ~2×
between sessions, which is exactly the board-load spread described above
([research/group_probe/README.md](../research/group_probe/README.md)).

Every DAG emitter can now request `submission='batched'`; the measured minimum per suite
(S10, 16 runs per mode) shows a mixed-engine graph collapsing to **one** ioctl:

| Suite (model) | Tasks | Serial ioctls | One-job ioctls | Serial min | One-job min | Source |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `diamond_tail_suite/000` | 5 | 5 | 1 | 73.8 µs | 45.8 µs | [docs/plans/pipelining-plan.md](plans/pipelining-plan.md) |
| `depthwise_join_suite/000` | 4 | 4 | 1 | 63.6 µs | 54.3 µs | [docs/plans/pipelining-plan.md](plans/pipelining-plan.md) |
| `pooled_dag_suite/000` | 8 | 8 | 1 | 170.6 µs | 51.0 µs | [docs/plans/pipelining-plan.md](plans/pipelining-plan.md) |
| `join_dag_suite/000` | 6 | 6 | 1 | 98.0 µs | 45.2 µs | [docs/plans/pipelining-plan.md](plans/pipelining-plan.md) |
| `pool_join_suite/000` | 6 | 6 | 1 | 109.7 µs | 46.4 µs | [docs/plans/pipelining-plan.md](plans/pipelining-plan.md) |
| `pooled_branches_suite/000` | 7 | 7 | 1 | 87.2 µs | 42.0 µs | [docs/plans/pipelining-plan.md](plans/pipelining-plan.md) |
| `mixed_head_suite/001` | 7 | 7 | 1 | 101.8 µs | 50.8 µs | [docs/plans/pipelining-plan.md](plans/pipelining-plan.md) |
| `join_chain_suite/005` | 9 | 9 | 1 | 141.5 µs | 55.4 µs | [docs/plans/pipelining-plan.md](plans/pipelining-plan.md) |

### Cross-job pipelining and the fence-free barrier

`JOB_NONBLOCK` lets the CPU prepare the next input while the NPU works. The measured
result is a **1.5–2.25×** throughput gain over synchronous submission, with exact outputs
(48 interleaved rounds; one round is two inferences):

| Container | Synchronous pair | Pipelined pair | Ratio | Source |
| --- | ---: | ---: | ---: | --- |
| 3-conv chain, one batched job | 195 µs | 87 µs | **2.25×** | [docs/plans/pipelining-plan.md](plans/pipelining-plan.md), [research/async_probe/README.md](../research/async_probe/README.md) |
| 3-conv chain, three serial jobs | 220 µs | 148 µs | **1.49×** | [docs/plans/pipelining-plan.md](plans/pipelining-plan.md), [research/async_probe/README.md](../research/async_probe/README.md) |

There is **no completion fence** on this kernel (`JOB_FENCE_OUT`/`FENCE_IN` return
`-EINVAL`; `CONFIG_ROCKCHIP_RKNPU_FENCE` is absent), so the lag-1 pipeline drains a queued
job with the next real inference. For a single-instance stream that wants lag 0, queue the
inference with `ORNPU_JOB_NONBLOCK` and run a small blocking **barrier** container; jobs run
in order per core, so the barrier completes only after the queued job
([runtime/open_rknpu.h](../runtime/open_rknpu.h), [research/barrier_probe/README.md](../research/barrier_probe/README.md)):

| Container | Tasks | Synchronous | Barrier-completed | Barrier job | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mixed_head_suite/model001` (serial, mixed engines) | 7 | 183.5 µs | **134.5 µs** | 23.9 µs | **−27%** |
| `deep_chain_suite/batched/model002` (one job) | 16 | 107.3 µs | 134.5 µs | 24.8 µs | +25% |

So the barrier is a win **only** for a serial container (it replaces N blocking ioctl
round trips with one wait); for an already-batched container it is pure overhead and the
lag-1 drain pipeline is the better throughput choice
([research/barrier_probe/README.md](../research/barrier_probe/README.md)).

### Memory behaviour: double-buffered intermediates and tiling

The N-layer chain emitter's `reuse_intermediates=True` keeps two intermediate surfaces
live instead of one per layer, which roughly halves deep-chain latency and stops the arena
growing with depth (serial medians shown; all variants byte-exact over 128 runs,
[research/deep_chain_suite/README.md](../research/deep_chain_suite/README.md)):

| Layers | Arena untiled | Arena double-buffered | Serial | Double-buffered serial | One batched job | Reuse + batched |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 28,672 B | 20,480 B (71%) | 369 µs | **184 µs** | 93 µs | 94 µs |
| 12 | 36,864 B | 24,576 B (67%) | 1537 µs | **721 µs** | 240 µs | 218 µs |
| 16 | 40,960 B | 24,576 B (60%) | 1120 µs | **645 µs** | 256 µs | **186 µs** |

Height-strip tiling is available but is **not** a latency win on this IP: at 8×8 with 16
layers it grows the task count (16 → 32 → 64) and the arena (28,672 → 45,056 → 81,920 B,
against 40,960 B for the untiled 16-layer chain) and the median latency with it
(287.6 → 552.4 → 1210.4 µs, 16 cases each, all exact,
[docs/plans/pipelining-plan.md](plans/pipelining-plan.md) S3). Use it when surface bytes
dominate program bytes, not for speed.

## The per-family cost model

Two independent fits describe a job. For a homogeneous CNA (conv) run:

```
serial  ≈ 22 µs per task + a per-ioctl round trip
batched ≈ 100 µs fixed + 8 µs per task
```

so a **crossover near eight tasks per same-engine run**
([docs/plans/pipelining-plan.md](plans/pipelining-plan.md) S1/S5,
[research/group_probe/README.md](../research/group_probe/README.md)). A deeper look fits
`cost(N) = intercept + N·task` on repeated idempotent tasks (min-based, 1..32 copies, 64
runs each, all exact):

| Family (enable/mask) | Intercept | Per task | R² | Source |
| --- | ---: | ---: | ---: | --- |
| conv, CNA (29/768) | 6.7 µs | **12.2 µs** | 1.00 | [docs/plans/pipelining-plan.md](plans/pipelining-plan.md) S5 |
| pool, DPU (96/3072) | −4.5 µs | **15.4 µs** | 0.98 | [docs/plans/pipelining-plan.md](plans/pipelining-plan.md) S5 |
| elementwise, DPU (24/768) | −22.2 µs | **20.3 µs** | 0.97 | [docs/plans/pipelining-plan.md](plans/pipelining-plan.md) S5 |

**Use the rows with their measured scope, not as universal constants**
([research/family_cost_crosscheck/README.md](../research/family_cost_crosscheck/README.md)):

* the **CNA row transfers** to real chains: 23 held-out containers give 13.4–13.9 µs/task
  against the table's 12.2 µs;
* the **elementwise row is profile-specific**, not family-wide: real two-surface
  elementwise tasks cost 11.8–14.5 µs/task, because the register key is the same but the
  operands differ;
* the **pool row is not identifiable** from real containers (pool only appears with 1–3
  copies), so its coefficient carries ±4 µs;
* slope-only prediction lands within a median −4 to −7% (mean +8 to +10%) overall, but
  **short pure-CNA chains cost up to 2.3× the prediction**: the duplication probe's copies
  overlap in the pipeline, while a real chain serializes on each hand-off.

### Predicting a new model's latency

1. Count tasks per family from the container (enable/mask; `open-rknpu inspect` prints the
   task table, or read `ornpu_info.task_count` / `submission_serial` at runtime).
2. If the model is one linked run and it is deep (roughly ≥ 8 tasks), predict
   `~100 µs + N × per-task` with the CNA row; for short runs add the measured
   per-container fixed cost rather than trusting the intercept.
3. If the model is serial, predict `~22 µs/task` plus the per-ioctl round trip, then
   expect the median to be noisy.
4. Treat both as an order-of-magnitude check, and confirm with `board_bench` on your own
   container. Worked example: the 7-task mel-CNN serial container is predicted at
   hundreds of microseconds by a per-task model, but measures **2.54 ms** — it is a real
   trained model with large surfaces and mixed engines, so the simple task-count model is
   not the right tool at that size ([examples/mel-kws/README.md](../examples/mel-kws/README.md)).

## Memory sizing

The runtime allocates two device buffers per model and never grows them
([runtime/open_rknpu.c](../runtime/open_rknpu.c), `ornpu_open`):

| Allocation | Size | Notes |
| --- | --- | --- |
| task buffer | **4,096 B fixed** | one `struct rknpu_task` (40 B) per task, max 64 tasks |
| payload + arena buffer | `header.arena_size` | the payload is copied to arena offset 0; tensor offsets start at/after the payload |

Per-model sizes come from `ornpu_info` / `ornpu_tensor_info`:

* `input_bytes`, `output_bytes` are the **flat packed NHWC API buffer sizes**:
  `batch × H × W × C`. This is the memory *you* allocate for `ornpu_run` /
  `ornpu_run_io` ([runtime/open_rknpu.h](../runtime/open_rknpu.h)).
* `ornpu_tensor_info.arena_bytes` is the **arena storage** for one tensor; it is larger
  than `api_bytes` whenever a layout pads. `arena_offset` is the tensor's offset inside
  the arena and `api_offset`/`api_bytes` place it in a flat per-role API buffer
  ([runtime/open_rknpu.c](../runtime/open_rknpu.c), `describe_tensor`).

Arena bytes per tensor ([runtime/sequence_format.md](../runtime/sequence_format.md),
"Version 5 named tensors"; [src/open_rknpu/sequence.py](../src/open_rknpu/sequence.py),
`tensor_native_bytes`):

| Layout | Arena bytes |
| --- | --- |
| packed UINT8 / packed INT8 | `batch × H × ceil(W/16)×16 × C` |
| native16 INT8 planes | `batch × ceil(H·W/4) × 64 × ceil(C/16)` |

Worked examples from published containers:

* `research/walk_chain_suite/model000.bin`: payload 3,456 B, arena 8,192 B; the native16
  8×8/C8 internal is 1,024 B and each 4×4/C3 surface is 256 B, while the packed 8×8/C3
  input occupies 384 B of arena but only **192 B** of API buffer.
* `research/mel_kws_suite/model000.bin` (the 7-task mel-CNN): payload 16,256 B, arena
  **81,920 B**; the native16 32×32/C3 input needs 16,384 B of arena but the API buffer is
  the flat 3,072 B, and the 8×8/C10 output needs 1,024 B of arena against 640 B of API
  buffer.
* `deep_chain_suite` 8-layer chain: payload 13,952 B, arena 28,672 B untiled / 20,480 B
  double-buffered ([research/deep_chain_suite/README.md](../research/deep_chain_suite/README.md)).

Process-level RSS from the examples (measured on the board, `rkipc` running): **856 KiB**
peak for the MNIST runner (arena 24 KiB + task storage 4 KiB, plus the CPU suffix's
7,296 B of float scratch) and **536 KiB** for the Fashion-MNIST runner (arena 2,496 B
hybrid / 12,432 B both-Conv). These RSS figures exclude kernel/DMA memory
([examples/mnist/README.md](../examples/mnist/README.md),
[examples/fashion/README.md](../examples/fashion/README.md)).

Hard loader bounds to size against ([runtime/sequence_format.md](../runtime/sequence_format.md)
"Limits", [runtime/open_rknpu.c](../runtime/open_rknpu.c)):

| Quantity | Bound |
| --- | --- |
| payload | ≤ 1 MiB, multiple of 64 |
| arena | ≤ 4 MiB, multiple of 4096 |
| task table | 64 tasks |
| tensor table | 64 tensors; ≤ 8 external inputs and ≤ 8 external outputs |
| batch | 1..16 (v5) |
| H, W | 1..1024 |
| channels | 1..128 (packed input 1 or 3) |
| program slot | `(register_words + 4) × 8` bytes, aligned to 64 ([src/open_rknpu/compose.py](../src/open_rknpu/compose.py)) |

A practical sizing pass: sum `arena_bytes` over all tensors (they may share bytes only for
internals that are never live together, via the liveness allocator), add the payload, round
the arena up to 4096 and confirm it is ≤ 4 MiB; then allocate `input_bytes` + `output_bytes`
for the API buffers. The compiler already reports this as `meta['arena_bytes']` and
`meta['allocated_bytes']` ([docs/api.md](api.md)).

## Benchmarking your own model

1. Compile it, then inspect it: `open-rknpu inspect model.bin` prints the header, tasks and
   tensor table as JSON; note `task_count` and whether the container is serial or linked.
2. Have an integer reference to compare against: the suite generators write
   `inputNNN.u8` + `expectedNNN.i8` for every model, and `tests/board_io.c` streams them
   and compares every output byte.
3. Time one model with `board_bench`:

```sh
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime tests/board_bench.c \
  runtime/open_rknpu.c -o /tmp/board_bench
adb push /tmp/board_bench /userdata/open-npu-research/bench/
adb shell /userdata/open-npu-research/bench/board_bench model.bin input000.u8 64 expected000.i8
```

   The output line carries `min/median/mean/max`, `unstable` and `mismatches`; run it more
   than once, on a quiet board, and compare minima.
4. Measure overlap with `board_async` (two instances, paired interleaved rounds) or
   `board_barrier` (single instance, queue-then-barrier), both built the same way from
   `tests/board_async.c` / `tests/board_barrier.c`.
5. Re-run a whole published suite with
   `PYTHONPATH=src python research/run_v5_suite.py <suite> --binary /tmp/board_io` for v5
   named-tensor suites, or `research/run_profile_suite.py <suite>` for v3/v4; each records
   `board_results_*.json` and `board_summary.txt` next to the suite
   ([research/run_v5_suite.py](../research/run_v5_suite.py),
   [docs/verification.md](verification.md)).

## What the board cannot do — do not build a plan on it

Measured with the driver's ACTION probe on 2026-09-11
([research/action_probe/README.md](../research/action_probe/README.md)):

| Capability | Status | Consequence |
| --- | --- | --- |
| clock scaling (`SET_FREQ`) | accepted but empty in driver v0.8.2 | no frequency/voltage tuning experiment |
| bandwidth policy (`GET_BW_*`) | `-EINVAL` | no devfreq/OPP control |
| SRAM (`GET_*_SRAM_SIZE`) | 0 (no pool) | no SRAM residency; everything streams through DDR |
| IOMMU | absent (non-IOMMU mode) | buffers are contiguous device allocations |
| `JOB_FENCE_IN` / `JOB_FENCE_OUT` | `-EINVAL` (kernel built without `CONFIG_ROCKCHIP_RKNPU_FENCE`) | no pollable completion fd; use the barrier pattern |
| `GET_VOLT` | oopses the caller (no regulator in the device tree) | never issue it |
| dma-buf zero-copy from the ISP | not implemented | inputs are staged through the runtime's buffers |

Any performance plan that assumes clock scaling, SRAM residency, a completion fence or
process-shared NPU work is outside what this board can deliver
([docs/board.md](board.md), [docs/plans/pipelining-plan.md](plans/pipelining-plan.md) S4).
