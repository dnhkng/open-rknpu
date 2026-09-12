# Board runbook

Operational guide for running containers on the reference board. It assumes you already
have a compiled container and need to stage it, run a suite, read the evidence, and get
the board back when something wedges. The hardware/format background is in
[board.md](board.md); the compiler-side symptom→fix guide is
[troubleshooting.md](troubleshooting.md) — this page does not repeat it.

> Board evidence is only meaningful if the hardware and driver are unchanged. The identity
> fields below are part of the evidence: every `board_results_*.json` in this repository was
> produced on this board with `rkipc` alive ([board.md](board.md)).

## Reference hardware and driver

| Item | Value | Source |
| --- | --- | --- |
| Board | Luckfox Pico Mini B (RV1103 SoC) | [board.md](board.md), [research/hardware_identity.md](../research/hardware_identity.md) |
| SoC `compatible` | `rockchip,rv1103g-38x38-ipc-v10`, `rockchip,rv1103` | [research/hardware_identity.md](../research/hardware_identity.md) |
| NPU node | `/proc/device-tree/npu@ff660000`, compatible `rockchip,rv1106-rknpu` | [board.md](board.md), [research/hardware_identity.md](../research/hardware_identity.md) |
| Driver | RKNPU v0.8.2, non-IOMMU mode (`rknpu iommu device-tree entry not found`) | [board.md](board.md) |
| NPU clock | 420 MHz (`GET_FREQ`), no regulator, no SRAM pool | [board.md](board.md), [research/action_probe/README.md](../research/action_probe/README.md) |
| CPU | ARMv7 Cortex-A7, ARM32 uClibc | [board.md](board.md), [research/hardware_identity.md](../research/hardware_identity.md) |
| Device | `/dev/rknpu` (char device; `SUBMIT`/`MEM_CREATE`/`MEM_MAP`/`MEM_DESTROY`/`MEM_SYNC`) | [board.md](board.md) |
| Serial | `498063e3262e55b7` | [research/hardware_identity.md](../research/hardware_identity.md) |
| Camera service | `rkipc` **must stay alive** — the board is also the camera appliance | [board.md](board.md) |

The accelerator is the shared **RV1106 NPU IP**, so NPU-level results are RV1106-NPU-IP
results; SoC-level conclusions (clock tree, memory map, peripherals) are specific to this
RV1103 die and do not transfer to a distinct RV1106 SoC
([research/hardware_identity.md](../research/hardware_identity.md), [board.md](board.md)).

Reproduce the identity over adb before trusting a comparison
([research/hardware_identity.md](../research/hardware_identity.md)):

```sh
adb shell "cat /proc/device-tree/compatible | tr '\0' '\n'"
adb shell "cat /proc/device-tree/model"
adb shell "cat /proc/device-tree/npu@ff660000/compatible | tr '\0' '\n'"
adb shell "dmesg | grep -i rknpu"
```

## Getting a shell

`adb` is the reliable path: it works over raw USB regardless of the board's IP state. The
board is USB-only (no wifi/ethernet configured), extremely RAM-constrained (~33 MB total)
and runs the stock camera/NPU app ([board-access.md](board-access.md)).

```sh
lsusb -d 2207:0019                  # Rockchip rk3xxx, present if powered/connected
adb devices -l                      # should show one device
adb shell 'pidof rkipc; free -m; df -h /userdata'
```

Two things to expect ([board-access.md](board-access.md)):

* The adb device is labelled `product:occam model:Nexus_4 device:mako` — that is this
  board, not an Android phone; the adb serial `498063e3262e55b7` matches the USB device's
  `iSerial`. Serial `498063e3262e55b7` is also what `research/run_v5_suite.py` uses.
* SSH (`ssh root@172.32.0.93`) is a convenience over a USB-RNDIS network that needs fixing
  on **both** sides after every reboot/replug: the host udev rule
  (`/etc/udev/rules.d/70-luckfox-rndis.rules`) commonly misses the first plug, and the
  board's `usb0` can miss its IP. The full two-part fix is in
  [board-access.md](board-access.md) — use adb to repair the board side.

**Point the scripts at your adb.** The suite runners invoke the platform tool by name and
honour the `ADB` environment variable, so either put `adb` on `PATH` or set it explicitly;
the v5 runner also passes `-s 498063e3262e55b7` ([board.md](board.md),
[research/run_v5_suite.py](../research/run_v5_suite.py)):

```sh
export ADB=/path/to/adb
```

## The `rkipc` rule

Keep the camera service running. The board is a camera appliance first, and every board
result in this repository was produced with `rkipc` alive ([board.md](board.md)). Do not
stop it to free memory or space, and do not attach `gdb` to it — `rkipc` runs ~30 threads
and even a single breakpoint can OOM-kill it on this board
([board-access.md](board-access.md)).

Check it, and restart it only if you actually killed it
([board-access.md](board-access.md)):

```sh
adb shell "pidof rkipc"     # should return a pid
adb shell "killall rkipc"; sleep 3
adb shell "export LD_LIBRARY_PATH=/oem/usr/lib:\$LD_LIBRARY_PATH; cd /oem/usr/bin && nohup ./rkipc -a /oem/usr/share/iqfiles > /tmp/rkipc.log 2>&1 &"
adb shell "pidof rkipc; free -m"
```

If a restart fails with `failed to allocate model memory!` / `ENOMEM`, the CMA pool is
fragmented and a plain restart will not fix it — reboot
([board-access.md](board-access.md)).

## `/userdata` budget and staging

`/userdata` is a small flash-backed UBI volume (≈4.5 MB) and the existing staging directory
already uses part of it. **Never write to `/tmp` for anything you intend to keep**: `/tmp`
is a 16.4 MB RAM-backed tmpfs, and overflowing it can hang the board's entire USB stack
until a physical power cycle ([board.md](board.md), [board-access.md](board-access.md)).

Stage under `/userdata/open-npu-research/<name>/` and delete what you no longer need
([board.md](board.md)). `research/run_v5_suite.py` does exactly this: it clears the remote
staging directory for the suite, pushes the runner and that suite's models/inputs/expected
outputs, and runs the file runner once
([research/run_v5_suite.py](../research/run_v5_suite.py)).

Check headroom before a big push:

```sh
adb shell 'df -h /userdata'
adb shell 'ls -la /userdata/open-npu-research'
```

Two Make targets cover the common path ([Makefile](../Makefile)):

```sh
make board-io                        # cross-compile the v5 runner to /tmp/board_io
make board-suite SUITE=walk_chain_suite   # board-io + run one published suite
```

`board-io` cross-compiles `tests/board_io.c` with the fetched ARM uClibc toolchain and the
board sysroot to `/tmp/board_io`; `board-suite` depends on it and then runs
`research/run_v5_suite.py $(SUITE) --binary /tmp/board_io`
([Makefile](../Makefile)). The toolchain is **not** in the repository — `research/toolchain/`
is fetched by `research/fetch_toolchain.py`, and any ARM uClibc cross compiler with the
board's sysroot will do ([board.md](board.md)). `--remote <dir>` renames the staging
directory on the board ([board.md](board.md)).

## Running a suite and reading the evidence

The runner depends on the container family ([research/run_v5_suite.py](../research/run_v5_suite.py),
[research/run_profile_suite.py](../research/run_profile_suite.py)):

| Container family | Runner (host script) | Binary on the board |
| --- | --- | --- |
| v5 named tensors | `research/run_v5_suite.py <suite> --binary /tmp/board_io` | `/tmp/board_io` (pushed into the staging dir) |
| v3/v4 task tables / legacy profiles | `research/run_profile_suite.py <suite>` | `/userdata/open-npu-research/board_api_test` |

Both write the same two evidence files next to the suite:

* `board_results_<start>.json` — one object per model with `model`, `passed`,
  `inferences`, `bytes` and the runner's raw `output` line. The v5 runner derives
  `passed` from the `model N: … inputs passed` line and computes
  `bytes = inferences × output_bytes` from the suite's `manifest.json`
  ([research/run_v5_suite.py](../research/run_v5_suite.py)); the v3/v4 runner matches the
  runner's `PASS: 1 models, N inferences, B output bytes` text
  ([research/run_profile_suite.py](../research/run_profile_suite.py)).
* `board_summary.txt` — the single `PASS:`/`FAIL:` line, e.g.
  `PASS: 4 models, 76 inferences, 14592 exact output bytes (board_api_test)` for the
  percentile calibration suite
  ([research/percentile_calibration_suite/board_summary.txt](../research/percentile_calibration_suite/board_summary.txt)).

The v5 runner (`tests/board_io.c`) is worth knowing before you read a failure: it inspects
the tensor table, streams concatenated cases from `inputNNN.u8`, compares every external
output byte against `expectedNNN.i8`, and prints
`PASS: N v5 models, M inferences, B exact output bytes; invalid IO rejected`
([tests/board_io.c](../tests/board_io.c)). A per-byte mismatch prints
`model N run R output J: got X expected Y` and exits with `-ERANGE`
([tests/board_io.c](../tests/board_io.c)).

For a one-off container, `runtime/main.c` is the small runner: push it and use
`--inspect`/one inference from a file ([c-api.md](c-api.md),
[runtime/main.c](../runtime/main.c)).

### Timing

Time from the host with `clock_gettime(CLOCK_MONOTONIC)` around `ornpu_run`, and report
both the first sweep and the mean of several sweeps: the first inference after
`ornpu_open` includes cache/DMA warm-up. The board has no userspace cycle counter; the
driver's bandwidth counters (`GET_DT_WR_AMOUNT` etc.) are cumulative byte counts, not time
([board.md](board.md)). Reference numbers on this board with `rkipc` running: a 7-task
mel-CNN container takes **2.54 ms** mean per inference; a 3-conv batched chain takes 195 µs
synchronous versus 87 µs pipelined for a pair (2.25×); deep 8- and 12-layer chains gain
6.8× and 11× from batched submission ([board.md](board.md)).

## Recovering a wedged NPU

A malformed or hand-built register program can hang the NPU, and it can also hang the
driver (`failed to wait job` / `job timeout` / `soft reset`). The recovery is a reboot
([board.md](board.md), [troubleshooting.md](troubleshooting.md#the-board-hangs)):

```sh
adb shell reboot
```

Wait ~15–20 s, then re-poll `lsusb -d 2207:0019`; a fresh boot re-triggers the host RNDIS
and board `usb0` races, so redo the two-part SSH fix if you need SSH
([board-access.md](board-access.md)). If adb itself is unresponsive (for example after
overflowing `/tmp`), a **physical power cycle** is the reliable recovery — the board will
not accept a software reboot command in that state ([board-access.md](board-access.md)).

Prefer compiler output over patching register words by hand; the runtime validates the
container (magic, checksum, geometry, tensor indices) before submitting, which is what
keeps a bad file from reaching `/dev/rknpu` ([board.md](board.md),
[container-format.md](container-format.md)).

## What the board cannot do

Measured with `research/action_probe.c` on 2026-09-11, driver v0.8.2, `rkipc` running,
non-IOMMU mode ([research/action_probe/README.md](../research/action_probe/README.md),
[board.md](board.md)):

| Capability | Status |
| --- | --- |
| `GET_FREQ` / `GET_DRV_VERSION` / `GET_HW_VERSION` | available (420 MHz, v0.8.2, `0x54524548`) |
| bandwidth counters `GET_DT_*`, `GET_WT_RD_AMOUNT`, `GET_TOTAL_RW_AMOUNT` | available, cumulative since boot, not a cycle counter |
| `SET_FREQ` | accepted but an **empty body** in this driver: **no clock scaling** |
| `GET_BW_PRIORITY` / `GET_BW_EXPECT` / `GET_BW_TW` | `-EINVAL` (no devfreq/OPP policy) |
| SRAM (`GET_TOTAL_SRAM_SIZE` / `GET_FREE_SRAM_SIZE`) | 0 (no SRAM pool configured) |
| IOMMU (`GET_IOMMU_EN`) | 0 / not present (non-IOMMU mode) |
| `JOB_FENCE_IN` / `JOB_FENCE_OUT` | `-EINVAL`: the kernel was built without `CONFIG_ROCKCHIP_RKNPU_FENCE` — **no pollable completion fd** |
| `GET_VOLT` | **oopses the caller** (no rknpu regulator in the device tree); the board survives, but never issue it |
| dma-buf zero-copy from the ISP | not implemented; inputs are staged through the runtime's buffers |

Because there is no fence, lag-0 completion uses the **barrier-job pattern**: submit the
model non-blocking (`ORNPU_JOB_NONBLOCK`, flag `0x2`), then submit a small blocking
**barrier** container; jobs run in order per core, so the barrier completes only after the
queued job, and `ornpu_sync_outputs(model)` makes the outputs readable
([c-api.md](c-api.md), [research/barrier_probe/](../research/barrier_probe/),
[board.md](board.md)). Read the observed counters with
`adb shell 'mount -t debugfs none /sys/kernel/debug 2>/dev/null; cat /sys/kernel/debug/rknpu/load'`
(`version`, `freq`, `load`, `power` are safe; **never** read `.../rknpu/volt`)
([board-access.md](board-access.md)).

Those absences are why clock scaling is closed as a measured negative rather than
implemented ([board.md](board.md), [docs/plans/pipelining-plan.md](plans/pipelining-plan.md)).

## Troubleshooting table

The compiler-side failures (rejections, accuracy, container parse errors) are in
[troubleshooting.md](troubleshooting.md); this table is only the board operations.

| Symptom | Likely cause | Recovery |
| --- | --- | --- |
| `adb devices` shows `offline` / `device offline`, or a runner retries and aborts | USB gadget re-enumeration or a host-side adb hiccup; the probe scripts treat `device offline` as retryable | re-plug USB, `adb kill-server && adb start-server`, re-check `lsusb -d 2207:0019` and `adb devices -l`; if the whole USB stack is hung (see `/tmp`), **physical power cycle** ([board-access.md](board-access.md), [research/run_batched_all_probe.py](../research/run_batched_all_probe.py)) |
| `adb push` fails with `No space left on device`; `df -h /userdata` shows the volume full | `/userdata` is ≈4.5 MB and holds the previous suite; the board's RAM is only ~33 MB | `adb shell 'df -h /userdata'`, list and delete old suites under `/userdata/open-npu-research/`, then re-push; `research/run_v5_suite.py` stages one suite and clears its own staging dir ([troubleshooting.md](troubleshooting.md#i-ran-out-of-space-on-userdata), [board.md](board.md)) |
| Runner aborts with `model N is not a named-tensor executable` | a v3/v4 (or legacy) container was fed to the **v5** runner | run v3/v4 suites through `research/run_profile_suite.py` (which uses `board_api_test`), not `run_v5_suite.py` ([tests/board_io.c](../tests/board_io.c), [research/run_profile_suite.py](../research/run_profile_suite.py)) |
| Runner aborts on a container that used to open | a **stale runtime/runner** from before the descriptor family was added (elementwise/LUT tasks, native16, v5 tables) | rebuild `runtime/open_rknpu.c` / `tests/board_io.c` from the same commit and re-push; old runtimes reject the new descriptors and flags ([runtime/sequence_format.md](../runtime/sequence_format.md)) |
| The suite's `board_summary.txt` is missing and the runner prints `model N accepted a missing output` / `accepted too many inputs` | the runtime did **not** reject a wrong external-buffer count — a loader regression, not a data problem | the runner asserts those two negative cases before any submit; rebuild runtime + runner together from the same commit ([tests/board_io.c](../tests/board_io.c), [c-api.md](c-api.md)) |
| `board_summary.txt` says `... invalid IO rejected` / the board exits with `-EINVAL` before running | **expected**: the runner's negative IO tests passed and a wrong buffer size was refused | nothing to fix; treat `PASS: … invalid IO rejected` as success ([tests/board_io.c](../tests/board_io.c), [c-api.md](c-api.md)) |
| `pidof rkipc` is empty, or a model fails after the camera died | `rkipc` was stopped (or OOM-killed) — the board is a camera appliance first | restart it with the exact `LD_LIBRARY_PATH`/IQ-file command above; if restart fails with `ENOMEM`, reboot ([board-access.md](board-access.md), [board.md](board.md)) |

## See also

* [board.md](board.md) — the hardware, `/dev/rknpu`, the reference numbers and the safety notes.
* [board-access.md](board-access.md) — SSH/RNDIS repair, `/tmp`, `rkipc` restart, the full action list.
* [troubleshooting.md](troubleshooting.md) — compiler and container symptom→cause→fix.
* [container-format.md](container-format.md) and [c-api.md](c-api.md) — what is in the file and how the runner reads it.
* [verification.md](verification.md) — what a `board_results_*.json` counts and how the ledger is maintained.
