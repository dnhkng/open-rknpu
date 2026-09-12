# Benchmark harness

`examples/benchmark/bench.py` turns a list of containers (or the ONNX models they were
compiled from) into a latency/throughput table, so a user can size their **own** model
instead of trusting the numbers shipped in `docs/performance.md`. It reuses the fixtures
and the board runners this repository already publishes: the same
`research/<suite>/modelNNN.bin` + `inputNNN.u8` + `expectedNNN.i8` layout that
`tests/board_io.c` streams and `research/run_v5_suite.py` stages.

Run everything from the repository root. All commands below are host-runnable except the
ones marked `# board`, which need the RV1103 board on `adb` and the vendor cross-compiler.

## What it measures — and what it does not

**Host mode (default, no board).**

| Column | Meaning |
| --- | --- |
| `bytes` | container file size |
| `task_count` | engine tasks in the container's task table (`ornpu_info.task_count`) |
| `engine_runs` | the ioctls the runtime submits the container as, derived from the task tails (`open_rknpu.compose.engine_runs`): equal to `task_count` for a serial container, fewer for a linked/batched one. `-` for a legacy `ORNPUBIN` container, which has one program and no task table. |
| `compile_ms` | wall-clock of `open_rknpu.scheduler.compile_sequence` on the sibling `modelNNN.onnx`, on the host that runs the harness. `-` when no sibling exists or when it needs profile options the default entry point does not supply. |

Host mode **does not** measure the board. `compile_ms` is host compile time, not board
inference time, and it changes with the host and its load. `bytes`, `task_count` and
`engine_runs` are fixed by the compiler and are reproducible.

**Board mode (`--board`).** One row per model:

| Column | Meaning |
| --- | --- |
| `inferences` | timed inferences behind the row: one submission's inferences × `--repeats` |
| `exact bytes` | output bytes the board runner compared byte-for-byte with `expectedNNN.i8` over those inferences. `-` when there is no expected file or a mismatch was seen. |
| `ms/run` | wall-clock milliseconds per inference, from the chosen `--stat` (below) |
| `runs/s` | `1000 / ms/run` |

It **does not** invoke the vendor toolkit: the harness only pushes a binary the user has
already cross-compiled, exactly as `docs/performance.md` shows. It does not measure band-
width, energy, device memory, multiple cores or multiple boards, and it never estimates:
`-` means *not measured*.

The board itself cannot make these numbers portable or fine-grained
([`docs/performance.md`](../../docs/performance.md) "What the board cannot do"):

* there is **no cycle counter**; the only clock is host-side
  `clock_gettime(CLOCK_MONOTONIC)` (`tests/board_bench.c`, `now_us()`);
* there is **no completion fence** (`JOB_FENCE_IN`/`FENCE_OUT` return `-EINVAL`), so every
  run's outputs must be synced before they are read and the sync is inside the measured
  region — it cannot be overlapped away;
* **clock scaling** (`SET_FREQ`) is accepted but empty in driver v0.8.2, so there is no
  frequency experiment to normalise against;
* the CPU is shared with the camera service `rkipc`, so absolute microseconds move between
  runs. Treat each row as a **single-board observation** and compare minima or paired
  medians, never one median against another ([`docs/performance.md`](../../docs/performance.md)).

## The measurement protocol

Two runners share the table.

`--runner io` (the default) drives `tests/board_io.c`, the streaming protocol
`research/run_v5_suite.py` stages: each model's recorded cases go to `input000.u8` /
`expected000.i8` in their own remote directory and one `./board_io . 1` submission is
wall-clocked from the host with every output byte compared. It needs a **v5**
named-tensor container; legacy/v3/v4 containers are skipped with a note.

`--runner bench` drives `tests/board_bench.c`, which samples
`clock_gettime(CLOCK_MONOTONIC)` immediately before and after one `ornpu_run_io` per round
and prints `min/median/mean/max` plus `unstable=` and `mismatches=`. This is the in-process
measurement [`docs/performance.md`](../../docs/performance.md) tabulates; it also accepts
the legacy containers.

| | `--runner io` (default) | `--runner bench` |
| --- | --- | --- |
| one submission | one `./board_io . 1`, all recorded cases | one invocation, `--iterations` rounds |
| warm-up | one untimed submission, discarded | one untimed invocation, discarded, **plus** `board_bench`'s own `min(iterations/4, 8)` warm-up rounds |
| `--repeats R` | `R` timed submissions | `R` timed invocations |
| one sample | host wall clock of the submission ÷ its reported inferences | the invocation's `min`, `median` and `mean` |
| `--stat` | `min`/`median`/`mean`/`p90` of the `R` samples | `min` = smallest invocation minimum; `mean` = mean of invocation means; `median`/`p90` over invocation medians |

* **`sync` per run.** `adb shell sync` is issued before every timed submission/invocation so
  dirty page writeback from the previous one is not charged to it; the runtime's own
  input/output cache sync is part of each measured run.
* **repetitions.** `--repeats R` (default 5). `--iterations N` sizes a `bench` invocation and
  defaults to the number of recorded cases in `inputNNN.u8`; `io` always runs every case.
* **per-percentile numbers.** `p90` is the nearest-rank percentile of the `R` samples,
  `index = round(0.90 × (R − 1))`. The sample size is only `--repeats`, so `p90` of the
  default five is the maximum; raise `--repeats` for a meaningful tail.

`io`'s `ms/run` is a conservative end-to-end bound — it includes the `adb` round trip,
`ornpu_open` and fixture I/O — so use it for exactness and throughput floors; use `bench`
for per-inference latency.

## Recorded table

Measured **2026-09-12** by the maintainer; every cell in this section that is not `-` is a
recorded number, reproduced here rather than invented. The structural columns come from
`bench.py`'s host mode and are reproducible on any host; `compile_ms` is not. The board
cells are `-` unless `docs/performance.md` records that exact measurement for that
container, in which case the row says so — **the maintainer fills the remaining `-` cells
from a board run** with the commands at the bottom of this page.

### Host structure

<!-- Generated by: PYTHONPATH=src python examples/benchmark/bench.py <14 paths> --markdown -->

| model | bytes | task_count | engine_runs | compile_ms |
| --- | ---: | ---: | ---: | ---: |
| `research/chain_suite/model000.bin` | 8288 | 2 | - | 19.2 |
| `research/deep_chain_suite/batched/model002.bin` | 19808 | 16 | 1 | - |
| `research/depthwise_join_suite/model000.bin` | 5104 | 4 | 4 | 6.0 |
| `research/diamond_tail_suite/model000.bin` | 6592 | 5 | 5 | 4.3 |
| `research/join_chain_suite/model005.bin` | 11072 | 9 | 9 | 4.7 |
| `research/join_dag_suite/model000.bin` | 7312 | 6 | 6 | 3.5 |
| `research/mel_kws_suite/model000.bin` | 16992 | 7 | 7 | - |
| `research/mixed_head_suite/model001.bin` | 8992 | 7 | 7 | 7.2 |
| `research/pool_join_suite/model000.bin` | 5776 | 6 | 6 | 2.8 |
| `research/pooled_branches_suite/model000.bin` | 7520 | 7 | 7 | 3.5 |
| `research/pooled_dag_suite/model000.bin` | 9072 | 8 | 8 | 4.0 |
| `research/sequence_suite/model000.bin` | 16512 | 2 | 1 | - |
| `research/two_head_suite/model000.bin` | 4448 | 3 | 3 | 2.2 |
| `research/walk_chain_suite/model000.bin` | 3872 | 3 | 3 | 2.7 |

`chain_suite/model000.bin` is the legacy `ORNPUBIN` format, so its `engine_runs` is `-`;
`sequence_suite/model000.bin` is a v3 container submitted as **one** linked run;
`deep_chain_suite/batched/model002.bin` is a v3 16-task chain linked into one run.
`mel_kws_suite/` and `sequence_suite/` ship no ONNX model, and
`deep_chain_suite/batched/` has none, so their `compile_ms` is `-`.

### Board latency/throughput

| model | inferences | exact bytes | ms/run | runs/s | source |
| --- | ---: | ---: | ---: | ---: | --- |
| `research/chain_suite/model000.bin` | 160 | 30,720 | 0.026 | 38,461.5 | measured 2026-09-12 by this harness, min of 5 |
| `research/deep_chain_suite/batched/model002.bin` | - | - | - | - | - |
| `research/depthwise_join_suite/model000.bin` | - | - | 0.064 | 15723.3 | recorded 2026-09-12, see `docs/performance.md` |
| `research/diamond_tail_suite/model000.bin` | - | - | 0.074 | 13550.1 | recorded 2026-09-12, see `docs/performance.md` |
| `research/join_chain_suite/model005.bin` | - | - | 0.142 | 7067.1 | recorded 2026-09-12, see `docs/performance.md` |
| `research/join_dag_suite/model000.bin` | - | - | 0.098 | 10204.1 | recorded 2026-09-12, see `docs/performance.md` |
| `research/mel_kws_suite/model000.bin` | - | - | 2.540 | 393.7 | recorded 2026-09-12, see `docs/performance.md` |
| `research/mixed_head_suite/model001.bin` | - | - | 0.102 | 9823.2 | recorded 2026-09-12, see `docs/performance.md` |
| `research/pool_join_suite/model000.bin` | - | - | 0.110 | 9115.8 | recorded 2026-09-12, see `docs/performance.md` |
| `research/pooled_branches_suite/model000.bin` | - | - | 0.087 | 11467.9 | recorded 2026-09-12, see `docs/performance.md` |
| `research/pooled_dag_suite/model000.bin` | - | - | 0.171 | 5861.7 | recorded 2026-09-12, see `docs/performance.md` |
| `research/sequence_suite/model000.bin` | 180 | 564,480 | 0.078 | 12,787.7 | measured 2026-09-12 by this harness, min of 5 |
| `research/two_head_suite/model000.bin` | 160 | 61,440 | 0.058 | 17,301.0 | measured 2026-09-12 by this harness, min of 5 |
| `research/walk_chain_suite/model000.bin` | 80 | 3,840 | 0.044 | 22,573.4 | measured 2026-09-12 by this harness, min of 5 |

The rows marked "measured … by this harness" are fresh output of the board command below
(`--runner bench --stat min --repeats 5`). Two runs agreed to the last quoted digit for
chain/sequence/two_head and differed by 0.001 ms (walk_chain 0.043 vs 0.044; 17,574 vs
22,573 runs/s), which is the resolution to expect from a shared board. The
remaining filled `ms/run` cells are the **serial minima** of the S10 table (16 runs per mode) in
[`docs/performance.md`](../../docs/performance.md) — the same `min` statistic the
`--runner bench --stat min` invocation below reports; the `mel_kws_suite/model000.bin` cell
is that page's **2.54 ms mean of 5 sweeps** (1,500 inferences), reported with `--stat mean`.
Every other
board cell is intentionally `-`: it has not been measured with this harness, and no number
here is extrapolated. `research/deep_chain_suite/batched/model002.bin` is skipped by the
harness (`no input002.u8 beside it`; the case files live one directory up), which is the
honest behaviour for a model that is not in the suite layout. `research/<suite>/board_results_0.json` records a past correctness
pass (inferences and exact output bytes per model) that `--runner io` reproduces, but those
aggregates are not timing and are not copied into the table.

## Commands

### Host (no board)

```sh
PYTHONPATH=src python examples/benchmark/bench.py --suite two_head_suite
PYTHONPATH=src python examples/benchmark/bench.py --suite diamond_tail_suite --suite pool_join_suite --markdown
PYTHONPATH=src python examples/benchmark/bench.py research/mel_kws_suite/model000.bin --no-compile
PYTHONPATH=src python examples/benchmark/bench.py research/two_head_suite/model000.onnx
PYTHONPATH=src python examples/benchmark/bench.py --glob 'research/*_suite/model000.bin'
```

`--glob` takes a repository-relative glob; quote it so the shell does not expand it.
`--output PATH` also writes the table to `PATH`. `--no-compile` skips the sibling-ONNX
timing for a quick structural sweep.

The host table above is the `--markdown` output of:

```sh
PYTHONPATH=src python examples/benchmark/bench.py \
  research/sequence_suite/model000.bin \
  research/chain_suite/model000.bin \
  research/walk_chain_suite/model000.bin \
  research/two_head_suite/model000.bin \
  research/deep_chain_suite/batched/model002.bin \
  research/diamond_tail_suite/model000.bin \
  research/depthwise_join_suite/model000.bin \
  research/pooled_dag_suite/model000.bin \
  research/join_dag_suite/model000.bin \
  research/pool_join_suite/model000.bin \
  research/pooled_branches_suite/model000.bin \
  research/mixed_head_suite/model001.bin \
  research/join_chain_suite/model005.bin \
  research/mel_kws_suite/model000.bin \
  --markdown
```

### Board

Cross-compile the two runners first (the vendor toolchain is fetched separately; this is
the recipe from [`docs/performance.md`](../../docs/performance.md)):

```sh
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime tests/board_bench.c \
  runtime/open_rknpu.c -o /tmp/board_bench   # board
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime tests/board_io.c \
  runtime/open_rknpu.c -o /tmp/board_io   # board
```

Fill the board table with the default `board_io` runner (`--runner io`), which is the
protocol `research/run_v5_suite.py` stages:

```sh
PYTHONPATH=src python examples/benchmark/bench.py --board \
  --suite depthwise_join_suite --suite diamond_tail_suite --suite join_chain_suite \
  --suite join_dag_suite --suite mel_kws_suite --suite mixed_head_suite \
  --suite pool_join_suite --suite pooled_branches_suite --suite pooled_dag_suite \
  --binary /tmp/board_io --repeats 5 --markdown   # board
PYTHONPATH=src python research/run_v5_suite.py two_head_suite --binary /tmp/board_io   # board
```

For the in-process latency that the recorded `ms/run` cells come from, use
`--runner bench`; `--stat min` records the same minimum `docs/performance.md` tabulates:

```sh
PYTHONPATH=src python examples/benchmark/bench.py --board --runner bench --stat min \
  --suite depthwise_join_suite --suite diamond_tail_suite --suite join_chain_suite \
  --suite join_dag_suite --suite mel_kws_suite --suite mixed_head_suite \
  --suite pool_join_suite --suite pooled_branches_suite --suite pooled_dag_suite \
  --binary /tmp/board_bench --repeats 5 --markdown   # board
```

`--adb PATH` and `--serial SERIAL` select the transport and device (`ADB` and `ADB_SERIAL`
are honoured as environment defaults); `--remote DIR` overrides the
`/userdata/open-npu-research/bench` staging directory; `--binary PATH` is the already
cross-compiled runner to push.
