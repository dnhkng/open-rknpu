# Board quickstart (RV1103)

The shortest path from "board on the desk" to "a compiled container ran on the NPU and
matched its reference byte for byte". It is deliberately short: the deep pages are
[board.md](board.md) (hardware and the runtime), [board-access.md](board-access.md)
(USB/SSH recovery) and [board-runbook.md](board-runbook.md) (running suites and capturing
evidence).

> Commands that need the board — they run on it or go through `adb` — are marked `# board`
> throughout this repository's documentation; host-only commands are unmarked.

## What you need

| Item | Notes |
| --- | --- |
| Luckfox Pico Mini B (RV1103) | the reference board; the accelerator is the shared RV1106 NPU IP |
| USB data cable | the board is USB-only here (no Wi-Fi/Ethernet configured) |
| `adb` (Android platform-tools) | the reliable way in; it works over raw USB and needs no IP setup |
| this repository | the cross toolchain is *not* committed: `research/fetch_toolchain.py` downloads it into `research/toolchain/` |

If `adb` is not on your `PATH`, use the absolute path (the board scripts also honour an
`ADB` environment variable):

```bash
export ADB=/path/to/platform-tools/adb   # host, once per shell
```

## 1. Plug in and find the board

```bash
lsusb -d 2207:0019          # host: the Rockchip USB device must appear
adb devices -l              # host: one device, serial 498063e3262e55b7   # board
```

Do not be alarmed by the label: the device reports itself as `product:occam model:Nexus_4`.
That is the board's USB gadget firmware, not a phone. The serial is the real identifier.

**Using a different board?** `research/run_v5_suite.py` pins the reference serial
(`498063e3262e55b7`) and passes `adb -s`; edit that constant (or run the runner's commands by
hand) for another board.

## 2. First look at the board

```bash
adb shell "pidof rkipc; free -m; df -h /userdata"   # board
```

Healthy output: `rkipc` has a pid, there are a few MB of free RAM, and `/userdata` shows
about 4.5 MB total. The board is a camera appliance first — `rkipc` is expected to be
running.

## 3. Four rules that keep the board alive

1. **Never stop or attach a debugger to `rkipc`.** It runs ~30 threads; stopping it drops
   the camera pipeline and a `gdb` attach can OOM-kill it. Restart instructions (and the
   `ENOMEM` caveat) are in [board-access.md](board-access.md).
2. **Never write to `/tmp`.** It is a 16.4 MB RAM-backed tmpfs; overflowing it can wedge the
   board's whole USB stack, which then needs a **physical power cycle** to recover.
3. **Keep `/userdata` small.** It is a ~4.5 MB flash volume shared with the running system.
   Stage under `/userdata/open-npu-research/<name>/` and delete it when you are done.
4. **Never read `/sys/kernel/debug/rknpu/volt`.** The driver does not null-check the missing
   regulator and the read segfaults the caller. `version`, `freq`, `load` and `power` are safe.

## 4. Your first inference

Build the file runner and replay a small, already-verified suite. `make board-io` cross
compiles the runner; `make board-suite` stages one suite, runs it on the NPU and compares
every output byte with the checked-in reference:

```bash
python3 research/fetch_toolchain.py   # host, once: downloads the 81 MB toolchain (skipped if research/toolchain/ exists)
make board-io                         # cross-compile tests/board_io.c -> /tmp/board_io   # board
make board-suite SUITE=walk_chain_suite   # stage 12 models, run them, compare every byte   # board
```

Success looks like this (the numbers are the published evidence for that suite):

```text
model 0: 16 inputs passed (1 outputs)
...
PASS: 12 v5 models, 192 inferences, 6368 exact output bytes; invalid IO rejected
```

If you see `PASS`, the board, the driver, the runtime and the compiled containers all agree
with the reference. The runner stages into `/userdata/open-npu-research/<suite>/`; remove it
afterwards if space is tight:

```bash
adb shell "rm -rf /userdata/open-npu-research/walk_chain_suite"   # board
```

To try a real application next, [`examples/camera/`](../examples/camera/README.md) converts a
recorded 64x48 frame and runs a chain-walk container on the board (recorded replay: 8
inferences, 8,192 exact bytes, 0 mismatches).

## 5. When it goes wrong

| Symptom | What to do |
| --- | --- |
| `adb devices` is empty | re-seat the USB cable; confirm `lsusb -d 2207:0019`; if the board stopped enumerating after a large push to `/tmp`, power-cycle it |
| Board stops responding to USB | **physical power cycle** (not `adb reboot` — adb is what is broken) |
| A run hangs or returns `-22` after ~1 s | the program wedged the NPU; `adb shell reboot` and re-check `pidof rkipc` |
| SSH will not connect | skip it: use `adb` — it does not need the network (see the two-part fix in [board-access.md](board-access.md)) |
| `failed to allocate model memory` on a restart | reboot; CMA fragmentation is not fixed by restarting processes |
| Output bytes differ from `expected*.i8` | that is a compiler/board finding, not an access problem: keep the command, the suite and the summary line and open an issue |

## Read next

* [board.md](board.md) — the hardware identity, `/dev/rknpu`, building the runtime.
* [board-runbook.md](board-runbook.md) — staging, running suites, capturing evidence.
* [board-access.md](board-access.md) — USB/SSH recovery and the full troubleshooting notes.
* [verification.md](verification.md) — why a byte-exact comparison is the only accepted claim.
