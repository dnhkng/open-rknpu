# Board workflow

## The reference hardware

| Item | Value |
| --- | --- |
| Board | Luckfox Pico Mini B (RV1103 SoC) |
| NPU node | `/proc/device-tree/npu@ff660000`, compatible `rockchip,rv1106-rknpu` |
| Driver | RKNPU v0.8.2, non-IOMMU mode (`rknpu iommu device-tree entry not found`) |
| NPU clock | 420 MHz (`GET_FREQ`), no regulator, no SRAM pool |
| CPU | ARMv7 Cortex-A7, ARM32 uClibc |
| Device | `/dev/rknpu` (char device, `SUBMIT`/`MEM_CREATE`/`MEM_MAP`/`MEM_DESTROY`/`MEM_SYNC`) |
| Camera service | `rkipc` **must stay alive** — the board is also the camera appliance |

NPU-level results are RV1106-NPU-IP results; SoC-level conclusions (clock tree, memory map,
peripherals) are specific to this die. See
[research/hardware_identity.md](../research/hardware_identity.md).

## Access

```sh
adb devices                       # the board shows as a USB device, e.g. 498063e3262e55b7
adb shell 'pidof rkipc; free -m; df -h /userdata'
```

`/userdata` is a small UBI volume (≈4.5 MB, and the existing staging directory already uses
part of it). Stage under `/userdata/open-npu-research/<name>/` and delete what you no longer
need; never write to `/tmp` for anything you intend to keep, and never stop `rkipc`.

If a job ever wedges the NPU (a bad register program can), the recovery is a reboot:
`adb shell reboot`.

## Build the runtime

```sh
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime tests/board_io.c runtime/open_rknpu.c \
  -o /tmp/board_io
```

The toolchain is **not** in this repository (`research/toolchain/` is fetched by
`research/fetch_toolchain.py`); any ARM uClibc cross compiler with the board's sysroot will
do. The runtime uses only libc, `sys/ioctl.h` and `mmap`.

## Run a model

The C API is three calls:

```c
ornpu_model *model;
ornpu_info info;
ornpu_inspect("model.bin", &info);       /* reads and validates the whole file */
ornpu_open("model.bin", &model);
ornpu_run(model, input_u8, info.input_bytes, output_i8, info.output_bytes);
ornpu_close(model);
```

`tests/board_io.c` is the reference runner for v5 containers: it inspects the tensor table,
streams concatenated cases from `inputNNN.u8`, compares every byte with `expectedNNN.i8`,
exercises the invalid-IO paths, and prints `PASS: N models, M inferences, B exact output
bytes`. `research/run_v5_suite.py` stages a suite and its manifest and records
`board_results_*.json` + `board_summary.txt`.

```sh
PYTHONPATH=src python research/run_v5_suite.py mel_kws_suite --binary /tmp/board_io
```

The scripts invoke the Android platform tool by name; put it on `PATH` or point `ADB` at it
(`export ADB=/path/to/adb`), and pass the host-built runner with `--binary`. `make
board-suite SUITE=<name>` does the cross-compile and the run in one step, writing the
evidence into the suite directory (`--remote` renames the staging directory on the board).

## Timing

Time from the host with `clock_gettime(CLOCK_MONOTONIC)` around `ornpu_run`, and report both
the first sweep and the mean of several sweeps: the first inference after `ornpu_open`
includes cache/DMA warm-up. The board has no cycle counter for userspace; the driver's
bandwidth counters (`GET_DT_WR_AMOUNT` etc.) are cumulative byte counts, not time.

Reference numbers (same board, `rkipc` running): a 7-task mel-CNN container takes 2.54 ms
mean per inference; a 3-conv batched chain takes 195 µs synchronous versus 87 µs pipelined
for a *pair* of inferences (2.25×); deep 8- and 12-layer chains gain 6.8× and 11× from
batched submission.

## What the board cannot do

Measured with [`research/action_probe.c`](../research/action_probe/README.md) (2026-09-11):

| Capability | Status |
| --- | --- |
| `GET_FREQ` / `GET_DRV_VERSION` / `GET_HW_VERSION` | available (420 MHz, v0.8.2, `0x54524548`) |
| bandwidth counters `GET_DT_*`, `GET_WT_RD_AMOUNT`, `GET_TOTAL_RW_AMOUNT` | available, cumulative |
| `SET_FREQ` | accepted but an **empty body** in this driver: no clock scaling |
| `GET_BW_PRIORITY/EXPECT/TW` | `-EINVAL` (no devfreq/OPP policy) |
| SRAM (`GET_TOTAL/FREE_SRAM_SIZE`) | 0 (no SRAM pool configured) |
| IOMMU | not present (non-IOMMU mode) |
| `JOB_FENCE_IN`/`FENCE_OUT` | `-EINVAL`: the kernel was built without `CONFIG_ROCKCHIP_RKNPU_FENCE` — use the barrier-job pattern for fence-free completion |
| `GET_VOLT` | **oopses the caller** on this board (no regulator in the device tree); do not issue it |
| dma-buf zero-copy from the ISP | **supported for packed input layouts**: the driver imports a dma-buf (`CREATE` flag `0x80`), `ornpu_open_shared` allocates the arena from the CMA heap and returns the fd, and `ornpu_run_prefilled` runs without copying the input (32 models / 64 inferences / 19,968 exact bytes on `add_geometry_suite`). A `native16` input still needs the runtime's packing, so that layout is refused with `-ENOTSUP` |

Those absences are why `docs/plans/pipelining-plan.md` S4 is closed as "measured negative" rather than
completed with a clock-scaling experiment.

## Safety notes

* Keep the camera service running; the board is a camera appliance first.
* A malformed register program can hang the NPU; the recovery is a reboot, and the runtime
  validates the container (magic, checksum, geometry, tensor indices) before submitting.
* Board evidence is only meaningful if the hardware and driver are unchanged. The identity
  fields above are part of the evidence: every `board_results_*.json` in this repository was
  produced on this board with `rkipc` alive.
