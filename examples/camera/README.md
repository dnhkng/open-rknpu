# Camera/V4L2 -> NPU (checklist row E5)

The board is a camera SoC, and this is the pipeline it exists for: a V4L2 capture node
hands over a YUV frame, the frame becomes the packed RGB image a small CNN expects, and the
NPU runs the CNN. This example builds the model, synthesizes a deterministic frame in each
capture format, converts the frame with a documented integer rule, publishes the profile's
own integer reference, and ships a libc-only board program that captures or replays a frame
and compares every output byte.

The one honest caveat is up front: **the live sensor path was probed but not demonstrated on
this board.** `rkisp_mainpath` is held by `rkipc` and a second streaming client is refused,
so there is no board run of `--device` in this repository. The recorded-frame path
(`--frame`), the conversion contract, and the capture code itself are host-verified; the
exact probe result is in [The `rkipc` constraint](#the-rkipc-constraint) and the board table
below keeps its cells at `-`.

## Pipeline

```
V4L2 /dev/videoN                                              (--device)
  YUYV 4:2:2 or NV12 4:2:0, mmap buffers, one frame
        |
        |  or a recorded buffer                             (--frame)
        v
convert: decode YUV -> RGB888 -> nearest downsample -> packed NHWC UINT8
        |  examples/camera/convert.py == examples/camera/camera_npu.c
        v
NPU: chain-walk container, 32x32x3 -> 16x16x4
        |  shared dma-buf arena (ornpu_open_shared / ornpu_run_prefilled) on --device,
        |  ornpu_run's copying path on --frame
        v
SUMMARY ... exact_bytes=.. mismatches=.. result=PASS
```

## What compiled, and inside which bounds

The graph is a 32x32 RGB stem with a pooled 1x1 head:

| # | Op | Shape out | Notes |
| --- | --- | --- | --- |
| 0 | input | `[1,3,32,32]` UINT8 | RGB, scale 1.0 / zero point 0 |
| 1 | `Conv` 3x3, 3->8, pads 1, stride 1 | `[1,8,32,32]` | first stage, native16 image input |
| 2 | `Relu` | `[1,8,32,32]` | folded into the Conv task |
| 3 | `MaxPool` 2x2, stride 2 | `[1,8,16,16]` | the only verified pool geometry |
| 4 | `Conv` 1x1, 8->4 | `[1,4,16,16]` | classifier head |

`compile_sequence` lands on the **`chain-walk`** profile (`src/open_rknpu/walk.py`), three
tasks, container format v5, 4,320 bytes. The 32x32 image is larger than the legacy 5..8
single-Conv image profile, so `parse_chain` marks the first Conv `native_input` and the walk
lowers it through the board-verified native16 image emitter; every later Conv reads a
native16 grid and the pool is interior to the chain, so the head keeps the graph inside the
op-level walk rather than the terminal-pool profile. Bounds are the native rows of
[docs/support-matrix.md](../../docs/support-matrix.md#1-dense-convolution-conv): input C1..128,
output C1..128, H/W 1..128, odd K1..31, stride 1..4, and a 2x2 stride-2 pool with zero
padding; the 1,024-atom image is well under the 6,144-atom height-tiling bound.

The bands the emitter recorded (deterministic compiler output, in `build/report.json`):

| Task | Op | Kernel | Out C | Input scale / zp | Output scale / zp | Relu |
| --- | --- | --- | ---: | --- | --- | --- |
| 0 | `Conv` | 3x3 | 8 | 1.0 / 0 | 5.469708 / 0 | yes |
| 1 | `MaxPool` | 2x2 | 8 | preserves the grid | preserves the grid | - |
| 2 | `Conv` | 1x1 | 4 | 1.0 / 0 | 6.316686 / -1 | no |

## The conversion rule

`examples/camera/convert.py` is the definition and `examples/camera/camera_npu.c` implements
exactly the same rule; `tests/test_example_camera.py` compiles the C harness on the host,
runs its `--emit-input` mode on the synthetic frames and asserts byte equality with the
Python bytes. Three integer steps, no floating point and no filtering:

1. **Decode the capture frame to RGB888** using full-range BT.601 with Q8 integer
   coefficients; every shift is a floor division by 256, and the C helper floors negative
   values too:
   `R = Y + (359*(V-128) >> 8)`, `G = Y - ((88*(U-128) + 183*(V-128)) >> 8)`,
   `B = Y + (454*(U-128) >> 8)`, each clamped to `0..255`.
   * `yuyv` — 4:2:2, bytes `Y0 U Y1 V` per two pixels; the pair shares `U` and `V`.
   * `nv12` — 4:2:0, a `width*height` `Y` plane then interleaved `U`,`V` at
     `width/2 x height/2`; each 2x2 luma block shares one pair.
   * `rgb` — RGB888, used unchanged.
2. **Downsample to the model input** by nearest neighbour:
   `src_x = x * width // out_width`, `src_y = y * height // out_height`.
3. **Emit packed NHWC UINT8**: `out_height` rows of `out_width*3` bytes, no row padding.
   The runtime applies its own 16-pixel row stride inside the arena; the API buffer is
   exactly `32*32*3 = 3,072` bytes.

Why nearest and integer-only: a float `round()` differs between Python's banker's rounding
and C's `nearbyint`, and a box filter divides by a geometry-dependent count. Neither is
needed for the plumbing this example proves, and byte equality between the two
implementations is a much stronger statement when the rule is integer-only.

## Host build (deterministic, no board, no network)

From the repository root:

```bash
PYTHONPATH=src python examples/camera/build.py
PYTHONPATH=src python examples/camera/convert.py --frame examples/camera/build/frame.nv12 --format nv12 --width 64 --height 48 --out-width 32 --out-height 32 --output /tmp/camera-input.u8
cc -O2 -std=gnu99 -Wall -Wextra -Werror -D_GNU_SOURCE -Iruntime examples/camera/camera_npu.c runtime/open_rknpu.c -o examples/camera/build/camera_npu
examples/camera/build/camera_npu --model examples/camera/build/model.bin --frame examples/camera/build/frame.yuyv --format yuyv --width 64 --height 48 --emit-input /tmp/camera-yuyv.u8
PYTHONPATH=src python -m unittest tests.test_example_camera
```

`build.py` writes `build/model.bin`, `build/frame.yuyv`, `build/frame.nv12`,
`build/frame.rgb`, `build/input.u8`, `build/expected.i8` and `build/report.json`.
`input.u8` is the `convert.py` conversion of **`frame.nv12`** (NV12 is what
`rkisp_mainpath` produces on this board), and `expected.i8` is the profile's own integer
reference — `open_rknpu.walk.chain_walk_reference` over the emitter's recorded bands and the
`parse_chain` op list of the *normalized* graph — applied to exactly those bytes.

The frame is a deterministic 64x48 test chart (R horizontal ramp, G vertical ramp, B
diagonal ramp, plus an eight-column saturated colour-bar strip in the bottom eight rows),
stored in each format so the downsample from 64x48 to 32x32 is exercised in both axes.

### Measured host result

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `build/model.bin` | 4,320 | `861c9a512ca9df2f9b8cb4c2a9a9a77df63025e3c44e7178b0668b6946cf6dfe` |
| `build/frame.yuyv` | 6,144 | `343d66a5e4da1f611f1bad494570935eb920a0431f803bc9971ed5d61d1e4cef` |
| `build/frame.nv12` | 4,608 | `d6e095775c9e47350b47fb39a7f81f8b350b91b7379dbda1563921f0941d30a2` |
| `build/frame.rgb` | 9,216 | `036cc69d846e760b18f324b33b8645b06ba34fedbcff70191d4c2494f702f77c` |
| `build/input.u8` | 3,072 | `b30fd55e3d7bcafdda2a56deab76fe34e41322b43748ac1a16b49b22852017ff` |
| `build/expected.i8` | 1,024 | `c49ab1bd5773c963c3a9b359b533d397ccaf3781de013b40f04cc14edc36fcbc` |

The C harness's `--emit-input` output equals the `convert.py` bytes exactly for all three
formats — that equality is asserted by the test, not just observed.

## The `rkipc` constraint

`rkisp_mainpath` (`/dev/video11`) is the ISP's 2304x1296 NV12 stream, and `rkipc` (the
board's camera appliance) holds it open. A second streaming client is refused by the
driver with this exact probe result:

```text
$ v4l2-ctl -d /dev/video11 --stream-mmap --stream-count=1 --stream-to=/tmp/isp.raw
VIDIOC_REQBUFS returned -1 (Device or resource busy)
```

The C harness reproduces that error from its own `VIDIOC_REQBUFS` call and names the errno:

```text
camera_npu: /dev/video11: VIDIOC_REQBUFS failed: Device or resource busy (errno=16 EBUSY)
```

`rkipc` must stay alive (it is also the camera appliance, and every board result in this
repository was produced with it running), so the example does **not** stop it and does not
claim a live capture. The free CIF/scale nodes never deliver a frame while nothing streams
them (`rkcif_scale_ch2` blocked waiting for a frame in the probe), which is why `--device`
waits at most 5 s and then reports a timeout instead of hanging. The capture node is always
opened *before* the NPU, so a busy node is reported as the node's errno rather than a
misleading `ornpu_open` failure.

## Board

Every line below is board work (`# board`); the host commands above produce the artifacts
first. The harness cross-compiles with the pinned toolchain and the same source that the
host test compiles.

```bash
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" -O2 -std=gnu99 -Wall -Wextra -Werror -D_GNU_SOURCE -Iruntime examples/camera/camera_npu.c runtime/open_rknpu.c -o examples/camera/build/camera_npu  # board
adb shell mkdir -p /userdata/open-npu-research/camera  # board
adb push examples/camera/build/camera_npu examples/camera/build/model.bin examples/camera/build/frame.yuyv examples/camera/build/frame.nv12 examples/camera/build/frame.rgb examples/camera/build/expected.i8 /userdata/open-npu-research/camera/  # board
adb shell 'cd /userdata/open-npu-research/camera && chmod +x camera_npu && ./camera_npu --model model.bin --frame frame.nv12 --format nv12 --width 64 --height 48 --expected expected.i8 --runs 8 | tee recorded.log'  # board
adb shell 'cd /userdata/open-npu-research/camera && ./camera_npu --model model.bin --device /dev/video11 --format nv12 --width 2304 --height 1296'  # board
adb pull /userdata/open-npu-research/camera/recorded.log examples/camera/build/recorded.log  # board
adb shell rm -rf /userdata/open-npu-research/camera  # board
```

`--frame` runs the recorded replay through `ornpu_run`'s copying path and prints one
machine-readable line:

```text
SUMMARY model=model.bin format=nv12 input_bytes=3072 output_bytes=1024 runs=8 path=copy inferences=8 exact_bytes=8192 mismatches=0 result=PASS
```

`--device` tries the zero-copy hand-off first: when the input layout is packed and the CMA
heap exists it prints `ZEROCOPY arena_fd=.. offset=.. row_stride=..`, maps the shared dma-buf
arena and writes the converted rows there before `ornpu_run_prefilled`; otherwise it prints
`ZEROCOPY unavailable (..)` and falls back to `ornpu_run`. On this board the *live* mode
cannot reach the NPU: every capture node the driver exposes is multi-planar, so the
single-planar capture path refuses it, and the one node that carries the sensor
(`rkisp_mainpath`) is held by `rkipc` anyway. Both facts are measured below.

### Measured board result

Run on the reference board (Luckfox Pico Mini B, driver 0.8.2, `rkipc` alive):

| Run | Frames | Inferences | Exact output bytes | Mismatches | ms/inference | Path |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `--frame frame.nv12` (recorded replay) | 1 | 8 | 8,192 | 0 | 1.5 | copy |
| `--frame frame.nv12 --runs 512` (timing run) | 1 | 512 | 524,288 | 0 | 1.5 | copy |
| `--device /dev/video11` (live ISP, NV12) | 0 | 0 | 0 | n/a | n/a | refused: multi-planar node |
| `--device /dev/video11` while `rkipc` streams (raw `v4l2-ctl` probe) | 0 | 0 | 0 | n/a | n/a | `EBUSY` |

```text
SUMMARY model=model.bin format=nv12 input_bytes=3072 output_bytes=1024 runs=8 path=copy inferences=8 exact_bytes=8192 mismatches=0 result=PASS
SUMMARY model=model.bin format=nv12 input_bytes=3072 output_bytes=1024 runs=512 path=copy inferences=512 exact_bytes=524288 mismatches=0 result=PASS
camera_npu: /dev/video11 is multi-planar (device_caps=0x04201000 has V4L2_CAP_VIDEO_CAPTURE_MPLANE); this harness captures single-planar
```

The timing run is wall clock for the whole process (`time`, 0.77 s for 512 runs), so the
per-run figure includes the input copy the runtime performs inside every `ornpu_run`, not just
the NPU submit. The conversion itself is board-verified against the host reference:
`--emit-input` produces the same 3,072 bytes as `convert.py` for all three formats
(`yuyv` `37bb76c5…`, `nv12` `b30fd55e…`, `rgb` `20e1f327…`).

**Not shown on this board:** a live capture through `--device`. The kernel reports
`device_caps=0x04201000` (`V4L2_CAP_VIDEO_CAPTURE_MPLANE | V4L2_CAP_STREAMING`) for
`rkisp_mainpath`, the CIF channels and the scale channels, so there is no single-planar
capture node to negotiate; and the earlier raw probe of `rkisp_mainpath` returned
`VIDIOC_REQBUFS ... Device or resource busy` while `rkipc` streamed it (the free nodes never
delivered a frame with nothing streaming the sensor). The live sensor path is therefore
compiled and documented but not demonstrated here. Nothing in this README claims otherwise.

## Files

* `build.py` — deterministic graph builder, compiler driver, synthetic frames, integer reference.
* `convert.py` — the reference conversion (the rule above) with a host CLI.
* `camera_npu.c` — libc-only harness: `--frame`, `--emit-input`, `--device`.
* `build/` — generated `model.bin`, `frame.{yuyv,nv12,rgb}`, `input.u8`, `expected.i8`, `report.json`.
* `../../tests/test_example_camera.py` — repeatability, decode, reference, C-vs-Python bytes and device errors.
* `../../docs/investigation-log.md` — the `rkipc` probe and the F8 dma-buf path this reuses.
* `../../docs/support-matrix.md`, `../../docs/primitives.md` — the bounds the graph stays inside.
