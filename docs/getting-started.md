# Getting started

## Install

```sh
git clone https://github.com/<you>/open-rknpu && cd open-rknpu
python -m pip install -e .          # requires Python >= 3.10, NumPy and ONNX
open-rknpu --version
```

The wheel ships the compiler plus the libc-only runtime sources
(`share/open-rknpu/runtime/`) so a board runtime can be built without a checkout. PyTorch
is only needed to train the examples (`examples/mel-kws/train.py`) and is never imported by
the compiler or the runtime.

## Compile a graph

The compiler accepts a *bounded* class of static ONNX graphs. Start with a single
convolution:

```sh
open-rknpu compile model.onnx -o model.bin --sequence
open-rknpu inspect model.bin        # JSON: header, tasks, constants, tensor table
```

`--sequence` selects the task-table profiles (everything except the legacy single-Conv
container). The default `compile` path emits the legacy v1/v2 container used by the first
milestone; modern graphs use `--sequence`, which is what every example in this repository
does.

The Python API is the same code the CLI calls:

```python
from open_rknpu.scheduler import compile_sequence

binary, meta = compile_sequence(
    "model.onnx",
    input_scale=1 / 255,     # real value = (byte - input_zero_point) * input_scale
    input_zero_point=0,
)
print(meta["profile"])       # e.g. "native16-input", "chain-walk", "diamond-tail"
print(meta["output_scale"], meta["output_zero_point"])
open("model.bin", "wb").write(binary)
```

The scheduler tries every profile in a fixed order and returns the first that accepts the
graph; `meta["profile"]` names it. If nothing accepts the graph it raises `ValueError` with
a specific reason (`native Conv requires static batch1..16, H/W1..128, ...`), never a
silent fallback. The dispatch order and the profile list are in
[architecture.md](architecture.md).

Useful compile options:

| Option | Effect |
| --- | --- |
| `input_scale`, `input_zero_point` | quantization of the UINT8 input image |
| `output_range={"scale":…, "zero_point":…}` | force the output band of the last Conv |
| `calibration_ranges=report["ranges"]` | use `open_rknpu.calibration.measure` bands per Conv (required for trained networks) |
| `submission="batched"` | link the tasks of each engine run into a single ioctl |
| `tiles=N` | split an 8×8 1×1 Conv chain into N height strips |
| `per_channel_mul=True`, `asymmetric_depthwise=True` | opt into the per-channel Mul and asymmetric depthwise profiles |

## Verify before you deploy

Compilation alone proves nothing. Every profile has a Python integer reference that
replays the hardware arithmetic (accumulator, multiplier/shift requantization, clipping)
for the *same* quantization parameters. Compare it with the compiled container:

```python
from open_rknpu.walk import chain_walk_reference, load_quantizations, parse_chain
from open_rknpu.normalize import normalize_model
import numpy as np, onnx

plan = parse_chain(normalize_model(onnx.load("model.onnx")).graph)
grid = chain_walk_reference(packed_uint8_image, load_quantizations(meta), plan["ops"], 0)
```

Each `examples/primitives/*.py` script does exactly this and prints the result, so the
primitive catalog doubles as a reference of "what should I compare against". For a model
with several inputs/outputs, use `ornpu_run_io` on the board and compare per tensor.

## Run it on the board

The short version (details in [board.md](board.md)):

```sh
# 1. cross-compile the runtime with the board toolchain
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime -Itests tests/board_io.c \
  runtime/open_rknpu.c -o /tmp/board_io

# 2. stage a suite and run every model, comparing each output byte
adb push /tmp/board_io /userdata/open-npu-research/suite/board_io
PYTHONPATH=src python research/run_v5_suite.py <suite> --binary /tmp/board_io
```

The repository's own suites are already published under `research/*_suite/` with their
`board_results_*.json`; `research/verify_suites.py` recompiles all of them against the
checked-in baseline without touching hardware.

## A first real model

* `examples/mnist/` and `examples/fashion/` compile a pretrained classifier's first layers
  to the NPU and finish the graph on the CPU (the hybrid pattern).
* `examples/mel-kws/` is the opposite: a trained audio CNN whose *whole* graph runs on the
  NPU, including calibration, at 98.00% INT8 accuracy.
* `examples/primitives/11_mnist_digits.py` walks through the MNIST first stage op by op.

## Where to look when something is rejected

1. The error message names the profile and the exact bound that failed.
2. [primitives.md](primitives.md) lists the same bounds with the emitter module.
3. `research/COVERAGE_EXPANSION_RESULTS.md` records the board evidence behind each bound
   and the failed hypotheses that produced the conservative limits.
4. [roadmap.md](roadmap.md) says whether the missing case is planned, blocked on hardware,
   or deliberately out of scope.
