# RV1103 primitive execution survey — 2026-09-08

**26 of 28 probe graphs replay successfully without a vendor runtime:** 104
inferences, 135,552 exact raw output bytes compared with synchronized vendor
readback. This establishes observed hardware execution for these small shapes.
It does not establish independent command generation, arbitrary-shape support,
full operator coverage, or numerical accuracy against float ONNX.

The compiler oracle identifies itself as **RKNN Toolkit2 2.3.0**. The board uses
the isolated research runtime 2.3.2 and stock RV1103 driver. Results are scoped to
this combination and do not assert impossibility on another compiler/version.

## What the probes establish

All probes begin with the same 8x8, three-channel Conv stem. Binary operations
use two nonconstant sigmoid branches so they cannot be replaced by a constant
or simple weight folding. Counts include the stem/branches.

| Probe(s) | Captured tasks | Evidence / classification |
|---|---:|---|
| Base Conv | 1 | Existing convolution path, enable 29, 126 register words |
| Depthwise 3x3 | 2 | Stem plus specialized convolution configuration; same enable 29 |
| ReLU, Clip/ReLU6, LeakyReLU, PReLU | 1 each | Fused register configuration; PReLU adds coefficient storage |
| Sigmoid, Tanh, HardSigmoid, HardSwish, Softplus, ELU, Swish, Mish | 2 each | Setup task (enable 24, 1,106 words) then fused convolution |
| Add, Mul, Sub, elementwise Max | 5 each | Four branch tasks plus one 78-word elementwise task, enable 24 |
| Div, int8 inputs | 4 | Only branches on NPU; compiler explicitly places Div on CPU; raw replay fails |
| Min, int8 inputs | None executed | Compiler assigns CPU; board runtime rejects CPU Min during initialization |
| MatMul with constant weights | 5 | Compiler turns MatMul into Conv plus layout operations |
| Softmax, channel axis | 76 | Compiler inserts INT8-to-FP16 conversion and a long mixed-task sequence |
| Concat, channel axis | 7 | Layout operations plus a convolution-related task after the two branches |
| Transpose, swap H/W | 2 | Stem plus 78-word task, enable 24 |
| Reshape, 8x8 → 4x16 | 5 | Actual layout work in this native tensor format; not eliminated |
| Nearest Resize, 2x | 2 | Compiler replaces Resize with ConvTranspose |
| GlobalAveragePool, 8x8 | 3 | Compiler decomposes it into two depthwise-style Conv stages after stem |
| BatchNormalization | 1 | Compiler folds it into Conv; no BatchNorm execution task remains |

### Activation table evidence

The 1,106-word setup task writes register **0x4104 1,027 times**, with three writes
to 0x4100. The following Conv changes the 0x4108–0x411c configuration as well as
activation/scaling fields. This is strong evidence for a programmable lookup-table
activation path, rather than separate dedicated hardware for every activation.
Exact table encoding and generation formulas remain to be recovered.

ReLU changes 0x4060/0x406c/0x40e0; Clip also sets upper bounds at 0x4028/0x40e4.
Leaky/PReLU changes 0x4040/0x4060/0x4068; PReLU additionally uses 0x5028/0x502c.
Quantization-related registers change too: these diffs alone are not sufficient
to implement a correct general emitter.

### Submission details that matter

The activation setup captures use submission flags **1**, unlike the existing
simple Conv path's **5**. Their setup/compute descriptors can share an operator
index. A first replay attempt that forced flags 5 and sequential indices timed
out; preserving both captured fields made all activation replays pass. This
isolates a harness mismatch, but does not determine the effect of each field
individually. The camera remained running; no restart or driver change was made.

At the time of this survey (2026-09-08) the public sequence container accepted at
most 64 tasks, at most 256 register words per task, and only the Conv/pooling
enable masks; it **now** also accepts the elementwise `24/768` descriptor (78
words) and the 1106-word LUT setup, and the LUT path is public (see
`runtime/sequence_format.md`). These probes demonstrate why it could not then
represent the whole inventory: LUT setup exceeds 256 words, Softmax has 76 tasks,
and enable masks 24/9 are additional paths.
The survey harness supports up to 100 tasks in a 4 KiB task buffer and a bounded
256 KiB arena. It is for trusted captured code, not a production file format.

## Reproduction and artifacts

- `../probe_primitives.py`: deterministic ONNX probes and vendor-only oracle builds.
- `*_build.log`: compiler placement tables, fusion/decomposition decisions.
- `../../research/fixtures/primitive_*/model.onnx`: original small ONNX graphs.
- `../primitive_*_capture.log`, `../capture_primitive_*`: allocations, commands,
  task descriptors and synchronized vendor output for zero/constant/ramp/impulse inputs.
- `../summarize_primitives.py`: generates `tasks.json` and `.replay` bundles.
- `../replay_survey.c`: libc/direct-ioctl replay, no RKNN imports or linking.
- `../run_primitive_replays.py`: bounded sequential board replay.
- `replay_results.json`, `*_replay.log`: authoritative pass/fail results.

The replay reconstructs relative allocations in a new DMA arena, preserves task
configuration, clears the final output buffer, and compares the complete output
bytes after cache synchronization. Comparison includes native padding. Div's
missing CPU step produces a mismatch on the first case, as expected. This checks
that captured NPU work is sufficient for the passing probes; it does not turn
captured constants or command templates into an independent compiler.

From repository root, following `docs/board-access.md`:

```bash
python3 research/probe_primitives.py
# Capture one at a time (replace NAME); Min is an expected initialization failure:
python3 research/run_oracle.py primitive_NAME
python3 research/summarize_primitives.py
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu11 -Wall -Wextra -Werror research/replay_survey.c \
  -o research/primitive_survey/replay-survey
python3 research/run_primitive_replays.py
```

Read board free flash space before capture/replay. Transfers use `/userdata`,
never `/tmp`. Camera PID 283 remained running and available RAM was about 6 MiB
at completion. Existing 28 host tests still pass; public runtime/compiler code
was not broadened by this survey.

## Implementation order

1. Independent depthwise command/weight generation, including its native layout.
2. Independent elementwise 78-word path and arithmetic/quantization semantics.
3. Clip/Leaky/PReLU configurations, then independently generated activation tables.
4. Layout operations and MatMul lowering using those established components.
5. Extend the public format/runtime with tested masks, task limits and submission
   modes, then integrate supported graph patterns.

Pooling was already independently verified before this survey. Additional ops,
dynamic-weight MatMul, broadcasts, other shapes/dtypes, and more complicated
composites remain untested. This is a primitive map and replay milestone, not
completion of all primitives in the open compiler.

### Follow-up: depthwise emitter completed

The first fixed depthwise profile is now independently generated through the
public compiler and verified on 12 fresh models (192 inferences). See
[depthwise milestone](../depthwise_suite/README.md). The replay-only status above
is historical for that profile; other newly surveyed primitives remain replay-only.

Independent Mul now passes 12 models / 384 runs / 73,728 exact bytes. See
[Mul implementation scope](../mul_suite/README.md). Sub is next.
