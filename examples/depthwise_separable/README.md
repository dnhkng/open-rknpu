# Depthwise-separable classifier block (checklist row E7)

The depthwise-separable block is the most common mobile building block: a
**depthwise** spatial Conv, a **pointwise** 1×1 Conv that mixes the channels, and
a 2×2 stride-2 **pool** that halves the grid. This example builds that block on
an 8×8 RGB image, adds a 1×1 classifier head, compiles it with
`open_rknpu.scheduler.compile_sequence`, writes the profile's own integer
reference as `build/expected.i8`, and ships a libc-only board harness that
compares every output byte on the RV1103.

## Graph

The emitted graph is

| # | Op | Shape out | Notes |
| --- | --- | --- | --- |
| 0 | input | `[1,3,8,8]` NCHW, UINT8 | RGB image, scale 1.0 / zero point 0 |
| 1 | `Conv` 3×3, `group=3`, pads 1, stride 1 | `[1,3,8,8]` | depthwise, multiplier 1 |
| 2 | `Relu` | `[1,3,8,8]` | folded into the depthwise task |
| 3 | `Conv` 1×1, 3→8 | `[1,8,8,8]` | pointwise (channel mix) |
| 4 | `Relu` | `[1,8,8,8]` | folded into the pointwise task |
| 5 | `MaxPool` 2×2, stride 2 | `[1,8,4,4]` | the block's downsample |
| 6 | `Conv` 1×1, 8→4 | `[1,4,4,4]` | classifier head |

This is MobileNetV1's first separable layer (`depthwise` on RGB, then a 1×1
expansion) followed by the pool and a small head.

## What compiled it, and what did not

The container was compiled by the **`chain-walk`** profile
(`src/open_rknpu/walk.py`), four tasks, format v5, 4,720 bytes — one task for
each Conv and the pool.

The dedicated native depthwise profile (`src/open_rknpu/depthwise.py`) matches a
whole graph shape: `Conv[/Relu] → depthwise Conv`, optionally
`→ pointwise Conv`. It has no pool task, and the terminal-pool profile accepts a
single legacy Conv before the pool. Every depthwise graph that ends in a pool is
therefore refused, with the exact error recorded in `build/report.json`
(`rejected_attempts`):

```
ValueError: depthwise sequence requires Conv[/Relu] -> depthwise Conv
```

- `stem → depthwise → pointwise → MaxPool` — rejected (native profile);
- `depthwise → pointwise → MaxPool` (terminal pool) — rejected (same message).

What is inside the verified envelope is a depthwise Conv that reads the graph
input: the front end's image-input depthwise rewrite (`normalize.py`, board
evidence `research/depthwise_rewrite_expansion_suite/`) lowers `group=3` to a
zero-filled dense kernel, which enters the board-verified op-level chain walk
(`research/walk_chain_suite/`, 12 models / 384 board inferences). The
classifier head after the pool is what keeps the pool interior to the chain, so
the delivered graph is a real depthwise-separable classifier that runs end to end
on the NPU. No compiler behaviour was changed to make the graph compile.

## Build and test on the host

Deterministic, host-only, no board and no network. From the repository root:

```bash
PYTHONPATH=src python examples/depthwise_separable/build.py
PYTHONPATH=src python -m unittest tests.test_example_depthwise_separable
```

`build.py` writes `build/prefix.bin`, `build/inputs.u8`, `build/expected.i8` and
`build/report.json`. `expected.i8` is
`open_rknpu.walk.chain_walk_reference` over the emitter's recorded
quantizations and the `parse_chain` op list of the *normalized* graph — the same
objects `compile_chain_walk` consumes. The test runs the build twice under a
temporary working directory and asserts the container decodes, both runs agree
byte for byte, and the written reference equals the recomputed one.

## Measured host result

Eight deterministic cases (all-zero, all-255 and 128 corners plus five seeded
random images); one packed `HWC` UINT8 case is 192 bytes, one INT8 output is
64 bytes.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `build/prefix.bin` | 4,720 | `1c5872c65dbcbd5a91d36cd18f7e88141c67318f022ffc71d09b10f3df3953d1` |
| `build/inputs.u8` | 1,536 | `b2ff01df16cb9fcbc14f035b1cf70ac5718111017aae69d9f4ef43e4fc429491` |
| `build/expected.i8` | 512 | `c4f7b0584d3e13cbf7dc02b23a57f0fc68c48ae49bf3a19bec267d51bd485017` |

The reference is exact by construction for the profile that compiled the
container (byte equality is the correctness criterion, not a tolerance). Sample
bytes:

| Case | Input | Output (`[4,4,4]`, first row shown) |
| --- | --- | --- |
| 0 | all zero | `-1 -1 -1 -1` |
| 1 | all 255 | `14 2 7 -6` |
| 2 | all 128 | `6 1 3 -4` |

Per-case output ranges: case 0 `[-1,-1]`, case 1 `[-7,14]`, case 2 `[-4,6]`,
case 3 `[-8,15]`, cases 4–7 within `[-7,13]`. The dequantized reference tracks
the ONNX float model within 0.63 output LSB, so the band is not degenerate.

## Board

Cross-compile the harness with the pinned toolchain, stage the four files and
run all eight cases in one session. Every line is board work (`# board`); none of
them runs on the host.

```bash
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime examples/depthwise_separable/board.c runtime/open_rknpu.c -o examples/depthwise_separable/build/board  # board
adb shell mkdir -p /userdata/open-npu-research/depthwise_separable  # board
adb push examples/depthwise_separable/build/board examples/depthwise_separable/build/prefix.bin examples/depthwise_separable/build/inputs.u8 examples/depthwise_separable/build/expected.i8 /userdata/open-npu-research/depthwise_separable/  # board
adb shell 'cd /userdata/open-npu-research/depthwise_separable && chmod +x board && ./board prefix.bin inputs.u8 expected.i8 | tee board.log'  # board
adb pull /userdata/open-npu-research/depthwise_separable/board.log examples/depthwise_separable/build/board.log  # board
adb shell rm -rf /userdata/open-npu-research/depthwise_separable  # board
```

`board.c` opens the prefix once, runs the eight cases back to back, prints one
`case <n> bytes=64 mismatches=<m> ms=<t>` line per inference and a final
`SUMMARY ... exact_bytes=... mismatches=... inferences=... ms_per_inference=...
result=PASS` line. `mismatches` must be 0 and `exact_bytes` must be 512 for the
run to pass.

Measured on the reference board (Luckfox Pico Mini B, RV1103, driver v0.8.2) by
the block above:

| Prefix | Cases | Exact output bytes | Mismatches | Inferences | ms/inference | Top-1 accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `prefix.bin` (`chain-walk`) | 8 | 512 | 0 | 8 | 1.063 | - |

`SUMMARY prefix=prefix.bin cases=8 inferences=8 exact_bytes=512 mismatches=0
total_ms=8.506 ms_per_inference=1.0633 worst_ms=4.566 result=PASS`. A second run
on a busy host measured 3.27 ms/inference (worst 23.6 ms), so treat the mean as a
range that depends on board and host load, not a constant; the first inference
after `ornpu_open` is always the worst (cold caches), the fastest measured was
0.076 ms.
The container decodes as `8x8x3 -> 4x4x4` with 4 tasks.

The classifier head is a 1×1 Conv, but no labelled dataset is checked into this
repository for the RGB block, so top-1 accuracy is not claimed here; leave the
cell `-` unless a labelled set is supplied for the run.

## Files

* `build.py` — deterministic graph builder, compiler driver, integer reference.
* `board.c` — libc-only board harness (`ornpu_open` / `ornpu_run` / `ornpu_close`).
* `build/` — generated `prefix.bin`, `inputs.u8`, `expected.i8`, `report.json`.
* `../../tests/test_example_depthwise_separable.py` — repeatability and reference test.
* `../../tests/board_io.c` — the v5 suite runner this harness is modelled on.
* `../../docs/primitives.md`, `../../docs/support-matrix.md` — the primitive
  bounds the graph stays inside.
