<!-- SPDX-License-Identifier: MIT -->

# Multi-model lifecycle: two containers, one process

E13. This example answers a runtime question, not a compiler one: what happens
when one process needs **two** compiled models? `build.py` compiles two small but
differently-shaped prefixes, and `board.c` loads both with two `ornpu_open`
calls, alternates inferences between them, and closes each handle once.

The practical lesson is that `ornpu_open`/`ornpu_close` own everything
device-side for one model:

* `ornpu_open` opens `/dev/rknpu`, allocates a 4 KiB task buffer plus one dma
  arena of the container's `arena_size`, and copies the 8 KiB compiled payload
  (the register program that carries the packed weights) into that arena. The
  handle keeps both allocations until `ornpu_close`.
* **Open each model once and reuse the handle for every inference.** A script
  that calls `ornpu_open` per inference re-creates the dma allocations and
  re-uploads the payload each time (thrash); one that forgets `ornpu_close`
  leaks the arena and the task buffer on every call.
* **The two models do not share an arena.** There is no process-global device
  context: each `ornpu_model` owns its own allocation, so a two-model process
  reserves `arena(a) + arena(b) + 2 * 4096` bytes, not `max(...)`. That is the
  memory cost of scheduling two models; it is paid once by the one-open-per-model
  pattern.

## The two models

| Container | Prefix | Profile | Tasks | Arena |
| --- | --- | --- | --- | --- |
| `model_a.bin` | `Conv(3x3, RGB->16) -> Relu` on 16x16 | `native16-input` | 1 | 12288 B |
| `model_b.bin` | `Conv(1x1) -> depthwise Conv(3x3) -> pointwise Conv(1x1)` on 8x8 | `8x8-c8-k3-s1-pad1` | 3 | 24576 B |

Both containers are re-framed as the v5 named-tensor format `board.c` (and
`tests/board_io.c`) loads, and each `expected_*.i8` is asserted byte-equal to the
profile's Python integer reference: `native.native_input_reference` for `a`, and
`quantization.reference` -> `depthwise.depthwise_reference` ->
`chain.native_reference` for `b`. The ONNX float error printed next to it is a
secondary quantization-quality number, never the correctness claim.

## Host commands

```sh
PYTHONPATH=src python examples/multi_model/build.py
PYTHONPATH=src python examples/multi_model/run_host.py
ruff check examples/multi_model
PYTHONPATH=src python tests/test_example_multi_model.py
```

`build.py` is deterministic and writes `examples/multi_model/build/`
(git-ignored) with `model_a.bin`, `model_b.bin`, `model_a.onnx`, `model_b.onnx`,
`input_a.u8`, `input_b.u8`, `expected_a.i8`, `expected_b.i8` and `report.json`.
`run_host.py` needs that directory and prints the lifecycle arithmetic plus a
replay of both integer references over the published fixtures. The test runs the
build in temporary directories, so the plain command above is the only one that
writes into the checkout (under the git-ignored `build/`).

## Measured on the host

Artifacts as `build.py` produces them (deterministic; the test rebuilds and
compares every byte):

| Model | Profile | Tasks | Container | v5 suite | Cases | Input/case | Output/case | `expected*.i8` | Reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `a` | `native16-input` | 1 | 3632 B | 3776 B | 4 | 768 B | 4096 B | 16384 B | exact |
| `b` | `8x8-c8-k3-s1-pad1` | 3 | 5520 B | 5664 B | 4 | 192 B | 512 B | 2048 B | exact |

The arena arithmetic, from `report.json` and the decoded containers:

| What | Bytes |
| --- | --- |
| `model_a` arena | 12288 |
| `model_b` arena | 24576 |
| combined arena (independent allocations) | 36864 |
| task buffers (2 x 4096) | 8192 |
| both models open | 45056 |

What an 8-alternation schedule costs under each pattern (`run_host.py` prints
the same numbers):

| Pattern | Device bytes | Payload re-uploaded |
| --- | --- | --- |
| open once per model, alternate | 45056 held for all 8 inferences | 16384, once |
| open + close per inference | 180224 allocated and freed | 65536 |
| open per inference, `close` forgotten | 180224 leaked | 65536 |

The open-per-inference numbers count the four 4 KiB + arena allocations each
model would create over the eight alternations and the 8 KiB payload copy each
`ornpu_open` performs; the one-open pattern creates two allocations and copies
each payload once.

## Board

Cross-compile with the vendor toolchain exactly like `tests/board_io.c`, stage
both containers and the harness, and alternate 8 inferences (A, B, A, B, ...).
Every board line is marked `# board`, which `tests/test_docs_commands.py`
records as a hardware skip:

```sh
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime examples/multi_model/board.c runtime/open_rknpu.c -o examples/multi_model/build/board_multi  # board
adb shell mkdir -p /userdata/open-npu-research/multi_model  # board
adb push examples/multi_model/build/board_multi examples/multi_model/build/model_a.bin examples/multi_model/build/model_b.bin examples/multi_model/build/input_a.u8 examples/multi_model/build/input_b.u8 examples/multi_model/build/expected_a.i8 examples/multi_model/build/expected_b.i8 /userdata/open-npu-research/multi_model/  # board
adb shell 'cd /userdata/open-npu-research/multi_model && ./board_multi model_a.bin model_b.bin input_a.u8 input_b.u8 expected_a.i8 expected_b.i8 8'  # board
```

`board_multi` opens each container exactly once, alternates the two handles (run
`i` uses `model_a` for even `i` and `model_b` for odd `i`, cycling each model's
four cases), compares every output byte, prints one line per run

```text
model=A bytes=4096 exact=1
```

and ends with the three numbers this table needs, per model and in total:

```text
summary model=A inferences=4 exact_bytes=16384 ms_per_inference=...
summary model=B inferences=4 exact_bytes=2048 ms_per_inference=...
summary total inferences=8 exact_bytes=18432 ms_per_inference=...
PASS: 2 models, 8 inferences, 18432 exact bytes, one open per model
```

### Board results

Measured on the reference board (Luckfox Pico Mini B, RV1103, driver v0.8.2) with
800 alternations - 400 inferences per model - by the command block above:

| model | inferences | exact bytes | ms/inference |
| --- | --- | --- | --- |
| `model_a.bin` (`native16-input`) | 400 | 1,638,400 | 2.080 |
| `model_b.bin` (`8x8-c8-k3-s1-pad1`) | 400 | 204,800 | 4.987 |
| total | 800 | 1,843,200 | 3.534 |

`PASS: 2 models, 800 inferences, 1843200 exact bytes, one open per model`. The
`ms/inference` includes the `ornpu_run` + wait; the two models differ because
`model_b` is a three-task sequence against `model_a`'s single native task.

The final argument (`8` above) is the number of alternations: eight means four
inferences per model, exactly one pass over each model's four recorded cases.
Increase it (for example to `800`) for a steadier `ms/inference`; `exact bytes`
scales with it and must equal `inferences * output bytes per case` for each
model.

The board run also demonstrates the point of the example: both containers stay
open for the whole run, one open (and one arena) per model, so the summary's
`PASS` line means two independent arenas coexisted on one board session without
a reload.
