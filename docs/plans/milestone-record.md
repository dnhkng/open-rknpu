# Milestone record (the original project README)

Kept verbatim as the chronological project narrative: what was built, what was measured on
the board, and which limits were hit at each stage. Current documentation starts at
[../README.md](../README.md); the day-by-day register-level record is
[../investigation-log.md](../investigation-log.md).

---

# open-rknpu

An experimental open compiler and C runtime for the Rockchip RV1103 NPU.
The compiler takes supported ONNX graphs and emits register commands, packed
weights, and quantization parameters without Rockchip's compiler. The runtime
links only to libc and talks directly to the existing `/dev/rknpu` kernel driver.

**This is an early implementation, not a general model framework yet.**
Use `compile --sequence` for the expanded profiles:

* Single Conv[/ReLU]: static batch 1–16, **input channels 1–128**, output channels
  1–128, H/W 1–128, odd kernels through 31, strides 1–4, dilation through 17,
  explicit side padding and VALID/SAME auto-padding. Large planes use serial
  height tiles. A vendor C128 Conv is one CNA task, so no channel-split
  accumulation is involved (`research/native_c65_suite/`); kernel 33 stays
  rejected.
* Constant grouped/dilated/even/rectangular kernels can lower through bounded
  zero-filled dense rewrites. These do not imply native group/dilation modes.
* Direct grouped/depthwise Conv lowers to verified dense kernels through group32
  and outputC128. Chained learned ConvTranspose covers depthwise C1..16 K1/K2/K3 and small rectangular per-axis stride1/2 profiles, plus dense/grouped C1..16 K3 with stride1/2 and asymmetric padding,
  LeakyReLU, standalone Mul and two-branch Add/Mul/Sub/Max have bounded profiles.
  Immutable RGB Mul constants use per-channel operands or a materialized native
  plane; the ERDMA reads a compact secondary operand strictly linearly, so no native
  spatial broadcast exists (`research/mul_broadcast_notch_suite/`).
* A single constant-parameter `QLinearConv` can be imported for UINT8 input and
  INT8 output. It reconstructs float constants and requantizes them; it does not
  preserve the source weight bytes or yet accept general Q/DQ graphs.
* Native Conv can fuse Relu or constant Clip[0,6]/ReLU6. Bounded pointwise Conv
  also supports scalar LeakyReLU and scalar/per-channel PReLU slopes; Mul can fuse a
  terminal Relu while preserving independent operand and output zero points.

See [exact support, test counts and remaining work](../../research/COVERAGE_EXPANSION_RESULTS.md),
the [full checklist](coverage-matrix.md), and the
[completion plan](completion-plan.md) and
[NPU utilisation plan](pipelining-plan.md). Sigmoid/Tanh have a narrowly
bounded LUT profile for a diagonal C3 1x1 stem with one scalar gain per channel (sign-split table domain measured, `research/lut_domain_suite/`); mixed-input and non-power-of-two stems remain open, with the residual measured at the table index in `research/lut_index_probe/`.
Rebuild the C runtime for native16 and long-LUT executables.

Application inputs are packed NHWC UINT8 and outputs packed NHWC INT8 with
embedded scale/zero point. Two-input profiles concatenate tensors in graph input
order; the generated metadata records logical shapes and byte offsets. All-zero
Conv weights and constant bias-only outputs are supported. Accuracy tuning is
separate from exact integer-reference verification.

The C API reports `input_tensor_count` and exposes each current logical input
through `ornpu_get_input_tensor()`, including batch, shape, byte offset and size.
Two-input executables retain one flat buffer in graph-input order.

Without `--sequence`, the legacy compiler retains grayscale H/W5–32 and RGB
H/W5–8, channels1/3→1–16, K1/3/5, stride1 and symmetric same padding.
The following older graph profiles also remain available:

The public compiler also accepts **Conv → ReLU → Conv**, with either kernel independently 1x1 or padded 3x3,
input/output `[1,3,8,8]`, and 3–16 hidden channels. `compile --sequence`
generalises this to N-layer dense chains (`[Conv, ReLU] * (N-1) + [Conv]`, N=3/4)
sharing one arena; 5 models and 80 board inferences passed, and every layer
output can be exposed as a named v5 output (`--expose-intermediates`)
([evidence](../../research/native_chain_suite/README.md)). Both convolutions require
constant float32 bias. This two-task profile keeps intermediate INT8 data on the
NPU and uses analytic quantization. The legacy (non-`--sequence`) path rejects
output overrides for it; `compile --sequence` supports the final output override,
verified for zero points -128/127/-43/79. The same compile command and C API
handle either profile.

**Conv[/ReLU] → MaxPool or AveragePool** is also supported: 1x1 convolution,
three channels, 8×8 input, and 2×2 stride-2 pooling producing 4×4 output.
Pooling preserves the convolution's INT8 scale and zero point. Average pooling
rounds the four-value mean to even. Global pooling is not yet supported.
The same graph may contain three explicit 2×2 pooling stages, producing
8×8 → 4×4 → 2×2 → 1×1 with four NPU tasks total. All three pools must use the
same operator. Each average-pool stage rounds separately; this is not a
GlobalAveragePool implementation.

Graphs outside the documented profiles, dynamic shapes and general graph
scheduling are not implemented. A distinct RV1106 SoC has not been tested; the
attached board's NPU is the shared `rockchip,rv1106-rknpu` IP that the ledger
exercises. Unsupported graphs
fail compilation; there is no silent CPU fallback. PyTorch models must first be
exported to ONNX within the supported subset.

## Compile and inspect

From this checkout, install the compiler in a host Python environment:

```sh
pip install .
open-rknpu compile model.onnx --sequence --target rv1103 --quantize int8 -o model.bin
open-rknpu inspect model.bin
```

The default output range is an analytic bound for independently varying UINT8
inputs. To measure per-convolution ranges from representative data, use:

```sh
open-rknpu compile model.onnx --calibration calibration/ -o model.bin
```

The directory contains `.npy` batches shaped `[N,3,H,W]`, dtype UINT8 or float32,
with finite input values in `[0,255]`. Calibration streams samples through the
ONNX host reference and writes `model.calibration.json` with sample counts and
measured ranges. It applies to the legacy compiler profiles, including both
convolutions in a chain, and now also to `--sequence` for profiles with an
output-range contract: single Conv[/Relu/Clip] through the native and scheduled
paths, the Conv-ReLU-Conv chain (per-layer ranges), and the elementwise/Mul
profiles. Profiles that reject output overrides (LUT, LeakyReLU/PReLU, terminal
Reshape, strided, two-head) also reject calibration. Explicit output overrides
are supported for native Conv and standalone/immutable-broadcast Mul; other
sequence profiles reject them. Use separate evaluation data: values outside the
measured ranges can clip, so calibration can improve or worsen accuracy. On the
board-verified `sequence_calibration` suite, calibration lowered mean error on all
three graphs but raised maximum error on two. See
[evidence](../../research/sequence_calibration_suite/README.md).

Alternatively, override the output quantization with both `--output-scale` and
`--output-zero-point` when needed. The compiler checks accumulator limits and the
currently supported scale range. Output floats are recovered with
`(int8_output - output_zero_point) * output_scale`.
Explicit overrides cannot be combined with `--calibration`.
For standalone or two-branch Mul, `--mul-a-zero-point` and
`--mul-b-zero-point` select signed INT8 boundary zero points. They apply in graph
input/branch order and are centered by independently verified BS/EW offset paths.

## Build and use the runtime

Build `runtime/` with an ARM32 uClibc cross compiler compatible with the board.
For the development toolchain fetched during this investigation:

```sh
make -C runtime open-rknpu-run \
  CC="$PWD/research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc --sysroot=$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  CFLAGS="-Os -Wall -Wextra -std=c11 -fno-use-linker-plugin"
```

Copy the small runtime, compiled model, and input to the board. Run:

```sh
./open-rknpu-run --inspect model.bin
./open-rknpu-run model.bin input.u8 output.i8
```

The runner validates the container, requires the exact input length, adds input
row padding, submits the NPU job, synchronizes the output, and removes native
channel padding. The model carries its dimensions; no shape arguments or RKNN
library are needed. Compile only trusted models: checksums and header validation
detect corruption/incompatibility, not hostile register programs.

Applications can use `runtime/open_rknpu.h`: `ornpu_open`, `ornpu_get_info`,
`ornpu_run`, and `ornpu_close`. Opt-in v4 native Conv files also expose
`ornpu_get_constant` and `ornpu_set_constant` for complete packed parameter
updates. Calls on one model instance must be serialized.
`ornpu_info.height/width` describe the input; `output_height/output_width`
describe the output. Rebuild C applications with the current header and library
after this addition to the development API.
The current profile reserves 20 KiB of DMA memory (4 KiB task plus 16 KiB arena).
Buffers persist across runs and are released on close. This is not a measurement
of total process or driver memory use.

Sequence compilation also covers bounded Conv-to-depthwise-to-pointwise graphs,
Conv/ReLU branch operands feeding Mul, per-batch immutable scalar/channel Mul
constants through N16, and explicit final output conversion for Conv-ReLU-Conv.
QLinearConv and Q/DQ Conv imports retain their supplied INT8 weight codes.
Container v3 remains immutable. Compile a bounded native Conv sequence with
`--mutable-weights` to emit the v4 `conv.parameters` descriptor. Updates must
replace the complete region and retain the compiled geometry/global output conversion.
Use `--mutable-constants` on a constant Mul to expose its packed `mul.factor`
region; replacement codes retain the factor scale recorded at compilation.

## Distribution

Build the wheel with `python -m pip wheel . --no-deps --no-build-isolation`. The
wheel contains the Python compiler (`open-rknpu` console script) and the libc-only
runtime C sources under `share/open-rknpu/runtime/`, and excludes the development
`research/` fixtures, tests and vendor binaries. In a clean environment with NumPy
and ONNX, `open-rknpu compile` produced byte-identical executables for a legacy
Conv chain, a v5 two-head fan-out graph and a calibrated sequence graph.

## Repository layout

| Path | What lives there |
| --- | --- |
| `src/open_rknpu/` | the compiler: 38 modules, from ONNX normalization (`normalize.py`, `network.py`) through per-profile emitters and the stage composer (`compose.py`, `join_dag.py`, `graph.py`, `walk.py`) to the container format (`sequence.py`, `model.py`) |
| `runtime/` | the libc-only C runtime (`open_rknpu.c/.h`) that talks to `/dev/rknpu`, plus `sequence_format.md` |
| `tests/` | 311 host tests: per-profile regressions, container/C-loader parity, the ledger checks and the retained board records |
| `examples/` | the MNIST and Fashion-MNIST hybrid examples and `mel-kws/`, a fully NPU-resident trained audio classifier (FSDD spoken digits, 98.00% INT8 on the board) |
| `research/` | the evidence: one directory per verified suite (models, expected bytes, `board_results_*.json`, `README.md`), the vendor-oracle captures, the analysis scripts, the suite generators (`research/build_*.py`) and the verification scripts (`verify_suites.py` + `container_baseline.json`, `campaign_sweep.py`, `check_docs_links.py`) |
| `dist/` | the built wheel and sdist (regenerated by `uv build`; the tree it was built from is the source of truth) |
| top-level vendor artifacts | `librknn*.so/.c`, `rknn_decompiled.c`, `librknnmrt_disasm.txt`, `rknn_analysis_summary.md`, `ghidra_project/`, `ghidra_scripts/` - the original vendor-library investigation, kept as reference, **not** part of the distribution |

### Inside `src/open_rknpu/`

The 38 modules group into six layers; `scheduler.py` picks the profile and every profile
ends in the same container writer.

| Layer | Modules |
| --- | --- |
| front end | `normalize.py` (ONNX graph cleanup, constant folding, layout), `network.py`, `quantized_import.py` (QLinearConv), `graph.py` (DAG analysis and the diamond/join-chain emitters) |
| scheduling | `scheduler.py` (profile selection and fallbacks), `walk.py` (the op-level walk used for chains-with-pools and fan-in joins) |
| profile emitters | `native.py`, `native_elementwise.py`, `strided.py`, `depthwise.py`, `pooling.py`, `reduction.py`, `chain.py`, `chain_n.py`, `tiled_chain.py`, `elementwise.py`, `elementwise_chain.py`, `elementwise_multi.py`, `join_dag.py`, `pool_join.py`, `depthwise_join.py`, `pooled_branches.py`, `transposed.py`, `lut.py`, `layout.py` |
| composition | `compose.py` (stages, bindings, liveness arena), `liveness.py`, `sequence.py` (task linking and tail control), `model.py` (task/register model) |
| numerics | `quantization.py`, `calibration.py` (`minmax`/`percentile`/`kl`), `activation.py`, `padding.py`, `register_profile.py` |
| tooling | `cli.py` (`open-rknpu compile`/`run`/`plan`), `compiler.py`, `accuracy.py` |

The 2026-09-11 cleanup ([cleanup-plan](cleanup-plan.md)) removed the dead locals and
unused imports left by the composer port, moved the suite generators out of `tests/` and
the stray root artifacts into `research/legacy_onnx/`, and dropped the regenerable
`build/` tree. It changed **no emitted container**: all 2,244 suite models compile to the
same bytes before and after.

## Verification and remaining work

The connected RV1103 has verified independently compiled dense 1x1, dense 3x3,
and 3x3 + ReLU graphs, including bias and negative weights. The public C API
passed the full three-channel shape/kernel/activation grid: 64 models,
1,024 inferences, and 129,792 output bytes. It reused each model for 16 inputs
and rejected incorrect API buffer sizes. Each earlier held-out dense graph also
passed 8,568 logical output values across 68 inputs. A separate public
file-to-file test passed all 126 output bytes. These graphs were not compiled
with RKNN. See [research evidence](../../research/README.md) for exact artifacts,
numerical error relative to float models, and the development-only vendor oracle.
The wheel was installed and tested in a clean environment with no RKNN package;
it produced a byte-identical executable.

Wider 1x1 output support passed a further 30 independently compiled models covering
every output channel count 2–16, with and without ReLU, at representative shapes.
The public C API passed 480 inferences and **178,112/178,112** output bytes;
a separate relocated-arena harness passed 600 inferences. Wider-channel coverage
does not yet exhaust every combination of spatial dimensions.
The corresponding wider 3x3 suite also passes all 30 models, 480 inferences,
and 178,112 output bytes, including dense signed weights, bias, ReLU, and padding.

The two-layer public API passed 14 new models covering every hidden count 3–16:
**448 inferences and 86,016 exact output bytes**. Four additional independently
generated graphs passed 272 relocated-harness inferences. The updated runtime
also passes the original 1,024 single-layer inferences. Integer exactness is
measured against the quantized computation; useful trained-model accuracy and
dataset calibration remain unfinished.

Public pooling tests passed 256 inferences and 12,288 exact output bytes;
Python/C header checks validate the reduced output shape. The file runner was
also checked with 48-byte pooled and 480-byte wide-channel outputs.
Three-stage pooling passed another 256 public-API inferences with exact 1×1
outputs, plus 272 independent raw-harness inferences.

```sh
python -m unittest discover -s tests -v
python -m unittest discover -s research -p test_emit_conv.py -v
```

A trained model now runs end to end: `examples/mel-kws/` trains a 4,090-parameter
mel-CNN on the Free Spoken Digit Dataset and runs the *whole* model on the NPU - 3x32x32
UINT8 features, 8x8x10 logits, one job per inference - at **98.00%** INT8 test accuracy
with 191,968 / 192,000 board-exact output bytes (float: 98.33%). It is what motivated two
walk capabilities: a native16 image input for images larger than the legacy 8x8 emitter,
and calibrated Conv bands (analytic bands collapse a trained network to chance).

The compiler-side contract is reproducible from the tree: `research/verify_suites.py`
recompiles all 2,244 published suite models and compares them with the checked-in
`research/container_baseline.json` (2,244 identical, 46 profiles pinned as
`ERR:ValueError`), `research/campaign_sweep.py` re-diffs the 169 campaign containers
against their artifacts (157 same / 12 documented drifts / 0 errors), and
`research/check_docs_links.py` validates every relative link and heading anchor.

The [completion plan](completion-plan.md#residual-disposition-2026-09-11) records the
disposition and evidence for every residual item. The broader
[project goal](project-goals.md) remains active, now
sequenced by the [completion plan](completion-plan.md): general DAG graphs and a
named tensor/lifetime ABI, ConvTranspose K5 and general
dilation, remaining Mul quantization modes, calibration on every public
profile, a useful trained classifier/detector with measured held-out accuracy,
a distinct RV1106 SoC, and publication. Dense native Conv input channels
C1..128 are public and board-verified (the vendor runs a C128 Conv as one CNA
task, so no channel-split accumulation is needed). The host suite has 311 tests. The current fixed
register profile still contains fields whose individual semantics have not been
decoded. Batched task submission is measured on the board: a container links its tasks
and the runtime submits one ioctl per linked run, where each tail carries the successor
program's fetch amount (`PC_DATA_AMOUNT`, e.g. `0x40` for a 126-word Conv, `0x14` for a
37-word pool, `0x28` for a 78-word elementwise task). Because that value is a fetch size
rather than an engine code, every transition links and a whole DAG - mixed engines
included - is **one job** (`research/mixed_batched_probe/`, `research/grouped_probe/`:
4-9-task graphs exact, roughly half the serial minimum latency). Serial remains the
default because it wins below roughly eight tasks per run (`docs/plans/pipelining-plan.md`
S1/S2/S8/S10). Height-strip tiling covers 1x1 and 3x3 chains, and the chain family
applies its hidden Relu (`research/tiled_k3_probe/`, S9).

New compiler/runtime sources are MIT licensed. The third-party development
binaries, cross toolchain, and GPL kernel reference files retain their licenses
and are excluded from the Python package.

Native intermediate 3x3 convolution passes 14 independent two-layer models,
448 board inferences, and 86,016 exact bytes across hidden counts 3–16.

A full spatial path is now supported: Conv → ReLU → Conv → three matching
2×2 pooling stages, input `[1,3,8,8]` and output `[1,3,1,1]`. Either convolution
can use 1×1 or padded 3×3 kernels, with 3–16 hidden channels. All five tasks
run on the NPU; calibration is supported. The 28-model board suite passes
896 inferences and 2,688 exact output bytes.
# ONNX frontend normalization

`open-rknpu normalize model.onnx -o normalized.onnx` folds immutable constant
reshapes, folds eligible channel-broadcast Add into Conv bias, and resolves
static Conv auto-padding. It preserves branches and overridable initializers.
This preprocessing command does not expand the hardware profiles below.
The pretrained MNIST integration target and remaining hardware gaps are recorded
in [research/pretrained/mnist/README.md](../../research/pretrained/mnist/README.md);
the second trained model (Fashion-MNIST, 88.15% over 10,000 board images) is in
[examples/fashion/README.md](../../examples/fashion/README.md).

The C runtime also accepts an [explicit task-table format](../../runtime/sequence_format.md)
with variable payloads, arena offsets, and output dimensions. The independently
generated 14x14 native graph runs through the public API using it. Use `compile --sequence` for an initial Conv[/Relu] followed by 2x2 stride-2
MaxPool/AveragePool stages. This path normalizes ONNX, allocates commands,
constants and activations, and submits tasks serially on the NPU. Input
quantization overrides are supported; calibration/output overrides are pending.
Version 5 of the format adds a named tensor table: `compile --sequence` lowers a
bounded two-head fan-out graph (`Conv → Relu → {Conv, Conv}`) to one executable
with two external outputs, and the C API binds them with `ornpu_run_io`. Verified
on RV1103: 14 models, 448 inferences, 172,032 exact bytes
([evidence](../../research/two_head_suite/README.md)). A second shape adds fan-*in*:
`Conv → Relu → {Conv, Conv} → {Add, Mul, Sub, Max}` compiles to four tasks whose
arena is placed by `open_rknpu.liveness` from the task read/write sets — the stem
stays live until both heads have read it and the external output is kept clear of
every internal tensor. Verified on RV1103: 12 models, 384 inferences, 73,728 exact
bytes ([evidence](../../research/diamond_suite/README.md)). Unequal runtime inputs and a
scheduler that assembles arbitrary emitter outputs by tensor name remain in
progress.
Further convolution layers and dense output lowering remain in progress.


## Affine input quantization

For a single Conv[/Relu], `--input-scale S --input-zero-point Z` defines
`real_input = (byte - Z) * S`, with S positive and Z in 0–255. Encode input
with `clip(rint(real_input / S) + Z, 0, 255)` and pack it in NHWC order.
The compiler accounts for input quantization in biases, output scaling, and
spatial padding. Explicit input quantization currently cannot be combined with
`--calibration` or multi-layer profiles.

Nondefault input quantization uses format version 2 (same 96-byte header);
version 1 files remain supported. `ornpu_info` now exposes `input_scale` and
`input_zero_point`; rebuild C applications with the matching header/library.
The pretrained MNIST first Conv/Relu is hardware-verified using this path.
The full MNIST graph runs as a verified NPU-prefix + CPU-suffix hybrid
(`examples/mnist/README.md`). Calibrating the native Conv2 output range makes the
fully-offloaded both-Conv variant accurate: 100/100 on 100 held-out images versus
13/100 uncalibrated.

### Native depthwise convolution

`compile --sequence` also supports `Conv[/Relu] -> DepthwiseConv` at fixed
`[1,3,8,8]`: a 1x1/3x3 stem and 3x3 depthwise kernel, stride1, same padding,
constant weights/biases. The emitter generates commands independently and uses
symmetric per-channel depthwise weight quantization. Other depthwise profiles
are rejected. Verified on RV1103: 12 new models, 192 runs, 36,864 exact bytes.
See [depthwise details and reproduction](../../research/depthwise_suite/README.md).

### Elementwise Add

`compile --sequence` supports two 1x1 Conv branches feeding Add, with fixed
`[1,3,8,8]` tensors and no broadcasting. The compiler uses a shared branch scale;
all three tasks run on the NPU. Rebuild the C runtime for the new 78-word
`24/768` elementwise descriptor. Verified: 12 fresh graphs, 384 inferences,
73,728 exact bytes. [Details](../../research/add_suite/README.md).

### Elementwise Mul

`compile --sequence` also supports two 1x1 Conv branches feeding Mul at fixed
`[1,3,8,8]`, without broadcasting. Both branches use scale s and zero point 0;
output uses scale 128*s*s and zero point 0. Verified on RV1103: 12 fresh models,
384 inferences, 73,728 exact output bytes. Quantization remains conservative.
[Scope and reproduction](../../research/mul_suite/README.md).

Sub and Max now use the same two-branch 8x8/C3 sequence profile. Each passed
384 board inferences (73,728 exact bytes). See the [ordered remaining-mode
roadmap](primitive-roadmap.md) for current work and unimplemented modes.

`compile --sequence` supports a single stride-2 Conv at 8x8/C3 -> 4x4/C3,
kernels 1/3/5 with symmetric padding. [Evidence](../../research/stride2_suite/README.md).

Depthwise 3x3 also supports stride 2 at 8x8/C3 -> 4x4/C3 after the Conv[/Relu]
stem. [192-run evidence](../../research/depthwise_stride2_suite/README.md).

Additional bounded profiles now include depthwise 1x1/5x5 at 8x8/C3 (stride1)
and depthwise 3x3 at 8x8/C4 (stride1). The normalizer can lower constant RGB
group3 and 3x3 dilation2 Conv to dense kernels, and fold scalar/channel constant
Mul after an unbranched Conv. `compile --sequence` accepts terminal Reshape that
preserves batch, channels and spatial pixel count. See [coverage, failed hypotheses
and remaining work](../../research/MODE_EXPANSION_STATUS.md). These additions do not
provide general native group/dilation modes, arbitrary broadcasts or graph scheduling.

Depthwise 3x3/stride1 now supports C5..16 after a 1x1 Conv stem at 8x8;
192 new board runs passed exactly. See the [complete Conv/Mul checklist](coverage-matrix.md)

### Per-channel Mul quantization

`compile --sequence --per-channel-mul` compiles `Mul(input, [C,1,1])` with an
immutable per-channel constant onto the verified 1x1 depthwise profile instead of
the elementwise Mul task. The elementwise task carries one output multiplier/shift,
so its operand stream stores one scale for the whole constant (`[.02,.35,1.9]`
becomes `[1,23,127]`); the depthwise profile keeps the native per-output-channel
weight scale, so each channel of the constant is quantized on its own grid and the
float-domain error drops 2.2x-4.7x. Bias is zero, so the arithmetic is unchanged.
Verified on RV1103: 12 models, 192 inferences, 23,232 exact output bytes
(`research/per_channel_mul_suite/`). The output grid stays one per-tensor scale: a
per-channel *output* conversion is not available in a verified profile: with the
table written at the address `0x5020` names and in the decoded Conv block format,
`BS_OW_CFG.OW_SRC=1` still hangs the elementwise task, and with `OD_BYPASS=1` the
same task ignores the table (byte-identical output). Retained with all six variants
in `research/mul_per_channel_ow_suite/`.

### Runtime per-channel scale Mul

`compile --sequence` also accepts `Mul(image[1,3,H,W], scale[1,3,1,1])` where the
per-channel operand is a second *external input* with a different shape. The
runtime packs the three INT8 codes into the 16-byte operand row that the verified
elementwise per-channel mode already reads, so a caller can supply a new
per-channel factor at run time without recompiling. Verified on RV1103: 16 models,
256 inferences, 32,832 exact output bytes across four geometries
([details](../../research/runtime_scale_suite/README.md)).

### Join-chain fan-out

A shared 1x1 Conv stem can feed **three to five dense heads**, folded left to
right by `n-1` elementwise joins with mixed `Add`/`Sub`/`Max`/`Mul` kinds before
an optional Conv tail. A Mul join folds two free operand scales; Add/Sub/Max need
a shared scale, so the next head is re-quantized onto the running result. The task
order and arena come from `open_rknpu.liveness`, which reuses a dead head buffer
for a later join. Verified on RV1103: 12 models, 384 inferences, 73,728 exact
bytes with non-degenerate outputs ([details](../../research/join_chain_suite/README.md)).

### Depthwise branch join

One shared 1x1 Conv stem can also feed a **group-3 depthwise branch** beside a
dense branch, with Add/Sub/Max/Mul folding the two 8x8/C3 grids. The depthwise
task is the verified standalone program relocated into the shared arena, so
kernels 1/3/5 and the asymmetric per-channel weight zero points all carry over.
Verified on RV1103: 12 models, 384 inferences, 73,728 exact bytes with
non-degenerate outputs ([details](../../research/depthwise_join_suite/README.md)).

### Pooled branch join

Both branches of a shared stem can also end in a **2x2 stride-2 pool**, with the
join running on the two 4x4/C3 pooled grids. The pool task uses the same register
builder as the sequence lowering, so MaxPool and AveragePool both carry over.
Verified on RV1103: 12 models, 384 inferences, 18,432 exact bytes
([details](../../research/pool_join_suite/README.md)).

### Mixed head families

A shared stem can feed dense and depthwise heads **in the same fan-out**, folded by
chained Add/Sub/Max/Mul joins: each depthwise head relocates the verified depthwise
program into the shared container, so two emitter families compose by tensor name.
Verified on RV1103: 12 models, 384 inferences, 73,728 exact bytes
([details](../../research/mixed_head_suite/README.md)).

### Dilated ConvTranspose

A depthwise `ConvTranspose` with `kernel_shape=[3,3]` and `dilations=[2,2]` is
emitted directly as a sparse K5 task: the dilated taps are the same operator as a
five-tap kernel with zeros at the odd positions, so the verified depthwise K5
profile carries it. Verified on RV1103: 8 models, 128 inferences, 82,432 exact bytes
([details](../../research/transpose_k5_dilation_suite/README.md)).

### Runtime scale on a DAG result

A chained fan-out can end in `Mul(result, scale)` where the per-channel scale is a
second external input, giving a runtime gain on the computed feature map. The scale
step reuses the verified per-channel elementwise program, so no extra conversion
task is needed. Verified on RV1103: 12 models, 384 inferences, 73,728 exact bytes
([details](../../research/join_scale_suite/README.md)).

### Runtime residual on a DAG result

A fan-out result can be combined with an **external feature map** (`Add/Sub/Max`
with a `[1,3,8,8]` runtime input quantized on the join scale), giving a runtime
residual connection into a computed DAG. Verified on RV1103: 12 models, 384
inferences, 73,728 exact bytes ([details](../../research/join_residual_suite/README.md)).

### General join expression

Joins may combine **any two previously produced tensors**, so a head or join result
can feed several consumers; one INT8 scale is propagated per tensor and the
allocator keeps a reused grid live across its consumers. Verified on RV1103: 12
models, 384 inferences, 73,728 exact bytes
([details](../../research/join_dag_suite/README.md)).

### Multi-layer branches

A branch of the general join DAG may be a chain of one to three Conv layers with its
own channel counts and bands, so residual-style blocks compile without being reduced
to single-Conv heads. Verified on RV1103: 12 models, 384 inferences, 73,728 exact
bytes ([details](../../research/branch_join_suite/README.md)).

### Depthwise-separable blocks

A depthwise layer may sit inside a multi-layer branch: it is rewritten to an
equivalent block-diagonal dense kernel so the per-layer dense path can emit it, which
covers `Conv -> Depthwise -> Conv` blocks inside a DAG. Verified on RV1103: 12 models,
384 inferences, 73,728 exact bytes
([details](../../research/depthwise_chain_suite/README.md)).

### Pooled DAG result

A join expression may end in a 2x2 stride-2 MaxPool/AveragePool, so
`branches -> joins -> pool` compiles as one container; the pool reuses the shared
register builder and keeps the join's band. Verified on RV1103: 12 models, 384
inferences, 18,432 exact bytes ([details](../../research/pooled_dag_suite/README.md)).

### Pooled multi-layer branches

A pooled join's branches may be Conv chains of one to three layers (with depthwise
layers expanded exactly to block-diagonal dense kernels), so `Conv -> Conv -> pool`
branches fold at 4x4. Verified on RV1103: 13 models, 416 inferences, 19,968 exact bytes
([details](../../research/pooled_branches_suite/README.md)).

See the ledger's [tested bounds, failed hypotheses and remaining
modes](../../research/COVERAGE_EXPANSION_RESULTS.md).
