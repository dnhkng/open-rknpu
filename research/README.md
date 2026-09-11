# RV1103 open NPU research

This directory contains a development oracle, command captures, and experiment
evidence. Production sources now live in `../src/open_rknpu/` and `../runtime/`;
the old Python entry points here are compatibility wrappers. **It is not yet a
general model framework.** The oracle uses Rockchip's compiler and mini runtime.
The open compiler imports ONNX and NumPy only; the open runtime uses libc and
`/dev/rknpu`. See the [project README](../README.md) for current usage.

## Independent single-op milestone

`emit_conv.py` now compiles static ONNX 1x1 or 3x3 convolution with three input
channels and 2–16 outputs, H/W between 5 and 8,
dense positive/negative float32 weights,
and optional float32 bias. Stride/dilation are 1; 3x3 uses symmetric padding 1.
UINT8 input has scale 1. Weight quantization is affine per output channel; output
quantization uses an analytic interval for inputs in [0,255], or an explicitly
supplied output scale/zero point. This is not yet a general calibration pipeline.
The original fixed-coefficient profile below is historical evidence.

The emitter constructs register words from a fixed, experimentally recovered
profile plus derived shape/address fields. It constructs weights and channel
tables from the accepted model. It does not read any captured binary, RKNN model,
or Rockchip library. Several fixed register fields remain semantically opaque;
these are documented as profile settings, not a complete ISA specification.

The generated layout omits unused vendor conversion programs and moves weights
from 0xac0 to 0x440, with the channel table at 0x480. It passes:

* 8x8 baseline: **768/768** logical output bytes across four input patterns.
* Held-out 6x5 graph, input-channel permutation `[1,2,0]`: **360/360** bytes.
  This graph was never compiled by RKNN. Both runs use a relocated DMA base.

Evidence: `generated_identity.log`, `generated_heldout.log`, and `generated/`.
The first single-op milestone is established only for this limited profile.
Other channel counts, strides, input quantization, and useful multi-op models
remain unimplemented. See the continuation results below for dense 1x1 and 3x3.

```sh
python3 research/emit_conv.py research/generated/heldout.onnx \
  -o research/generated/heldout.bin
python3 research/verify_identity.py research/generated_heldout.log \
  research/generated/heldout.json
```

Use a host Python environment with `onnx` and `numpy`. The vendor-oracle
environment is `/home/dnhkng/Documents/Projects/rknn-toolkit2/.venv/bin/python`;
the open emitter does not import the RKNN package installed there. Public
compiler/runtime verification uses the repository's open environment
(the open host environment, where the vendor RKNN module is not installed), as the suite commands
below specify.

On the board, the development harness command for this model is:

```sh
./raw_replay heldout.bin --relocate 6 5
```

The harness name reflects its initial purpose; it now accepts emitted buffers
as well. It still supplies its own four test inputs rather than exposing a
general application runtime API. Shape arguments must match the compiled file;
the production runtime now supplies a self-describing, validated executable
format and application API; this harness remains for low-level experiments.

## Verified on the connected board

On 2026-09-08, Luckfox RV1103, Linux 5.10.110, ARM32 uClibc:

* A deterministic ONNX 1x1 convolution maps three channels with multipliers
  `[0.25, 0.5, 0.75]`, no bias, input shape `[1,3,8,8]`.
* RKNN toolkit 2.3.0 compiles the development fixture; mini runtime 2.3.2 runs it.
* UINT8 NHWC input with width stride 16 works. Logical input is 192 bytes;
  the allocated tensor size requested by the API is 384 bytes.
* Query 9 returns native output `[1,1,8,8,16]`, INT8, 1024 bytes, zero point
  -128 and scale 0.75. Only the three logical channels are compared.
* Zero, constant-128, spatial/channel ramp, and one-channel impulse each match
  all 192 reference integers exactly: **768/768 values**.
* A preload recorder captures the allocation and submission ABI, mapping each
  allocation's returned DMA-buffer fd. No kernel probes or debugger are needed.
* The open C harness runs the same four cases with no Rockchip userspace library.
  It also passes with its command/data base displaced by 4096 bytes, validating
  base-relative addressing for this fixture.
* `rkipc` remained running as PID 283 throughout these tests. Process presence
  alone does not verify the camera stream's image quality.

## Files and evidence

| File | Purpose |
| --- | --- |
| `build_identity.py` | Rebuild deterministic ONNX, calibration inputs, vendor RKNN oracle |
| `board_probe.c` | Small vendor-reference runner using the public C API |
| `capture_ioctl.c` | Recorder for this runner's allocations and four submissions |
| `raw_replay.c` | Open memory allocation, cache sync, task construction, submission, cleanup |
| `emit_conv.py` | Narrow ONNX compiler with derived shape/address/weight encoding |
| `rv1103_register_profile.py` | Observed fixed register settings for this profile |
| `test_emit_conv.py` | Host compiler boundary and buffer-encoding checks |
| `verify_identity.py` | Independent integer reference for all four inputs |
| `identity_build.log` | Compiler layout and allocation information |
| `identity_board_c.log` | Vendor oracle outputs |
| `identity_capture.log` | Allocation correlation and instrumented oracle outputs |
| `capture_identity/` | 36 binary snapshots, 164256 bytes total |
| `raw_replay.log` | Open replay outputs |
| `raw_replay_relocated.log` | Open replay with command/data base displaced |
| `vendor/` | GPL driver reference sources and retrieval provenance; not linked into the harness |

`board_probe.py` is an earlier ctypes probe. On this board Python startup waited
on `spinand_mtd_read` and the test timed out before producing results. Prefer the
C runner. Host timeouts can leave an ADB-launched process alive; check its PID
before retrying. No probe Python process remained after the experiment.

## Reproduce the host verification

The compiler-side contract from the 2026-09-11 cleanup ([cleanup-plan](../docs/plans/cleanup-plan.md))
is three scripts, all run from the workspace root:

There is also a trained-model example: `examples/mel-kws/` (FSDD spoken digits) trains,
calibrates and compiles a mel-CNN whose whole graph runs on the NPU, with the suite
evidence in `research/mel_kws_suite/`.

```sh
PYTHONPATH=src python3 research/verify_suites.py    # every research/*suite*/model*.onnx vs container_baseline.json
PYTHONPATH=src python3 research/campaign_sweep.py   # recompile the campaign suites and diff the .bin artifacts
python3 research/check_docs_links.py                # relative links and heading anchors in every markdown file
```

`container_baseline.json` records, per published suite model, either the sha256 of the
emitted executable or `ERR:<ExceptionType>` for the profiles that deliberately reject their
model. It was captured before the cleanup and is the "no emitted container changed"
contract; `verify_suites.py --update` rewrites it and must only be used when a container
change is intended and the affected suites have fresh board evidence.

Older identity evidence:

```sh
python3 research/verify_identity.py research/identity_board_c.log
python3 research/verify_identity.py research/identity_capture.log
python3 research/verify_identity.py research/raw_replay.log
python3 research/verify_identity.py research/raw_replay_relocated.log
```

`fetch_toolchain.py` fetches the C subset of Luckfox's cross toolchain into
`research/toolchain/`, using `toolchain_tree.json`. This host is x86-64; the old
investigation's aarch64-host constraints do not apply here. Toolchain files are
development dependencies, not part of the intended framework distribution.

Build the standalone replay (from the workspace root):

```sh
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -Os -Wall -Wextra -fno-use-linker-plugin \
  research/raw_replay.c -o research/raw_replay
```

Build the oracle using the public `rknn_api.h` include directory, linking the
local `librknnmrt_232.so`. Its SONAME is `librknnmrt.so`; on the board the research
directory contains a symlink to that version, isolated from `/oem/usr/lib`.

Board artifacts are under `/userdata/open-npu-research`, not `/tmp`. Read
`../docs/board-access.md` before accessing the board. Do not run this recorder
against the camera service or large models: its deliberately small fixed table
and capture limits are for this test process only.

## Command encoding and next work

The first task is 40 bytes and reports op index 1, enable mask 29, interrupt mask
768, clear mask 131071, and register count 126. The submission is 104 bytes,
flags 5, one task. Each observed command is a little-endian 64-bit word:

```
bits  0..15: register offset
bits 16..47: register value
bits 48..63: command tag (semantics still under investigation)
```

For this capture, register `0x1070` contains input offset `0x2000` and `0x4020`
contains output offset `0x3000`, relative to the submitted DMA base. Weight and
per-channel tables occupy offsets `0xac0` and `0xb00`. The replay preserves these
offsets in one DMA arena. This does not establish all address-register meanings
or correctness for other shapes.

Additional oracle captures `capture_permuted`, `capture_h7w8`, and
`capture_h8w7` isolate weight ordering and shape fields. Permuting channels changes
only six weight bytes. Each output-channel weight row occupies four bytes, with
three logical INT8 weights and padding. The 7x8 and 8x7 oracle probes each pass
672/672 output values.

Next: broaden channel counts and graph coverage. `build_identity.py` accepts
`--output-channels 1..16` for controlled vendor probes, with three input channels.
These probes do not expand the open compiler's supported profile until command
differences have been decoded and independently emitted results pass hardware.
Fixtures `out1`, `out2`, `out4`, `out5`, `out8`, and `out16` build successfully
and have board captures; corresponding `out*_build.log` and `out*_capture.log`
files record layouts and synchronized outputs. USB access after the restart
required a new ACL for bus 001 device 007 and an ADB server restart.
The public compiler now supports 2–16 output channels for both 1x1 and 3x3.

For these 1x1 probes, let A be output channels rounded up to four. Captures for
2/4/5/8/16 channels give `0x1030 = A*4`, `0x1038 = 0x01010000 | A`, and
`0x403c = ((A-1)<<16) | 15`. Channel tables repeat in 32-byte groups of four:
16 bytes of biases, 8 bytes of negated weight zero points, and 8 bytes of channel
multipliers. `analyze_output_channels.py` checks this arithmetic and packing
against **8,960/8,960** synchronized output bytes. The one-output oracle uses
a different DPU specialization, which remains to be decoded. These observations
have not yet been generalized to wider 3x3. Independent wider 1x1 generation is
now verified as described below.

### Independent wider 1x1 implementation

The compiler derives the three output-tile register fields, packs arbitrary
signed weights and bias into four-channel groups, and emits distinct input and
output channel counts in the executable header. The runtime validates those
counts, checks input/output lengths separately, and returns every logical output
channel from the native 16-byte pixel. Existing three-channel files remain valid.

`generated_wide*_r*.log` cover 30 independently compiled signed, biased graphs,
output counts 2–16, ReLU on/off, and representative rectangular spatial shapes.
All 600 inferences match the integer reference, including relocated DMA arenas.
`research/build_wide_suite.py` uses another deterministic seed to build 30 public
API models; `wide_suite.log` proves **480 inferences, 178,112 output bytes, zero
mismatches**, with persistent model handles and invalid-size rejection.
No wider graph in either suite was compiled with RKNN. Nine host tests include
Python/C header-validation parity for widened channels and unsupported options.
Broader input channels and intermediate-layer tensor layouts are the next gaps.

### Two-task native intermediate investigation

`build_chain.py` builds a development oracle with Conv(3→4), Relu, Conv(4→3)
at 8x8. `chain4_build.log` confirms both convolutions remain NPU operators.
`capture_chain4` and `chain4_capture.log` capture the submission and synchronized
outputs. The submission has two 40-byte tasks, each with 126 register words;
command programs start at offsets 0 and 0x440. Their op indices are 1 and 2.
The first program's link words write next offset 0x440 to register 0x10 and
value 0x40 to register 0x14; the second has the observed final value 0x28.

The first convolution writes native signed INT8 to arena offset 0x1200. The
second reads it directly: no UINT8 conversion, unpacking, or CPU operator lies
between them. Weight rows for the second layer have 16 entries, with four
logical weights followed by weight-zero-point padding. Native-input CNA fields
differ substantially from the first-layer UINT8 path; second-layer output count
is literal three, rather than the first-layer output count rounded up to four.

`analyze_chain.py` applies both captured weight/channel tables and each layer's
requantization, matching **768/768** logical bytes. The second accumulator uses
the intermediate INT8 values directly; its bias already compensates the input
zero point. `raw_replay.c` now accepts development environment setting
`OPEN_NPU_TASKS=2` for this two-program layout. `chain4_replay.log` proves the
libc-only harness reproduces all **4,096 native bytes** with a relocated DMA
base. This result uses captured commands; independent multi-layer compilation
remains outstanding and is not claimed by this replay.

Hidden-count probes 3/4/8/16 now all pass the same two-layer arithmetic reference:
**3,072/3,072** logical bytes. Among the second layer's 126 registers, only
`0x1024 = ((input_channels-1)<<16) | 16` and the three output-quantization fields
`0x4080/0x4084/0x4088` vary across these captures. Native weight rows always have
16 entries. This isolates a single native-input tile covering up to 16 channels.

For the 8x8, 1x1, three-output native-input profile, the key fixed CNA fields are:
`100c=0, 1010=104, 101c=0, 1030=30, 1034=10, 1038=1010003,
103c=40000, 1044=80004, 104c=9, 1050=10001, 1054=10001,
1058=0, 105c=0, 107c=20, 1080=40, 1084=80008, 108c=0,
1188=8, 118c=70007, 301c=0, 403c=2000f` (all hexadecimal).
Address and quantization fields must be independently generated. Shape
dependence outside 8x8 and native-output counts other than three remain unproven.

Independent emission now verifies the second-layer bias formula
`round(float_bias / (input_scale * weight_scale)) - input_zero_point * sum(centered_weights)`;
the evidence and remaining public integration work are below.

### Independently generated two-layer milestone

`build_heldout_chains.py` generates four ONNX Conv-Relu-Conv graphs with entirely
new signed weights and biases, using seed 110312. They have hidden counts
3/4/8/16, input/output `[1,3,8,8]`, and 1x1 kernels. These graphs are never
compiled with RKNN. `emit_chain.py` reads their ONNX tensors and independently
emits both command programs, quantized weights, biases, channel multipliers,
links, and memory offsets. It imports the open compiler/ONNX/NumPy; it reads no
RKNN models, captured binary buffers, or vendor libraries.

The raw layout is: programs at 0/0x440, first weights/table at 0x880/0x8c0,
second weights/table at 0x940/0x980, intermediate at 0x1000, input at 0x2000,
and output at 0x3000. The intermediate stays native signed INT8 throughout.
Second-layer accumulator scale is `input_scale * weight_scale`; its global
requantization factor is `input_scale * max(weight_scale) / output_scale`.

`generated_chain_heldout{3,4,8,16}.log` and matching metadata under `generated/`
record **272 inferences and 52,224 exact logical output bytes**, tested with
relocated DMA bases and the open two-task harness. Each graph has four base
inputs and 64 deterministic random inputs. `verify_chain.py` composes the two
integer references and separately compares with the original float graph:

| Hidden channels | Exact bytes | Float maximum error | Float mean error | Output scale |
| --- | ---: | ---: | ---: | ---: |
| 3 | 13,056 | 1.62531 | 0.446559 | 1.62996 |
| 4 | 13,056 | 3.13425 | 0.904509 | 3.51220 |
| 8 | 13,056 | 7.87215 | 2.57905 | 9.75691 |
| 16 | 13,056 | 8.85771 | 2.67847 | 10.2796 |

The analytic range bounds grow conservatively through layers. Exact integer
agreement does not establish useful float-model accuracy; dataset calibration
remains a required next capability. ONNX regeneration followed by compilation
reproduced all four raw payloads byte-for-byte in the open host environment, where importing
the RKNN module is impossible because it is not installed.

Reproduce a host compilation and its saved hardware verification:

```sh
python research/build_heldout_chains.py
python research/emit_chain.py \
  research/generated/chain_heldout4.onnx -o research/generated/chain_heldout4.bin
python research/verify_chain.py \
  research/generated_chain_heldout4.log research/generated/chain_heldout4.json
```

The emitter has now moved into `src/open_rknpu/chain.py`, with explicit graph
and quantization validation. `research/emit_chain.py` is a compatibility wrapper.
The public compiler dispatches Conv-Relu-Conv graphs to this implementation;
the public container uses profile 2 for its fixed two-task layout. Profile 1
retains single-layer behavior. Python and C both restrict profile 2 to 8x8,
three external input/output channels, 1x1 kernels, and an intermediate Relu.
The runtime builds tasks at offsets 0/0x440 and submits both in one ioctl.
The 96-byte header and 20 KiB DMA allocation budget remain unchanged.

`research/build_chain_suite.py` independently generates every hidden count 3–16
with seed 110313. `chain_suite.log` proves **14 public models, 448 inferences,
86,016 exact output bytes**, model reuse, and invalid buffer-size rejection.
`application_suite_profile2_runtime.log` proves the updated runtime also passes
the original 64 models/1,024 inferences/129,792 bytes. Thirteen host tests cover
payload preservation, unsupported second-layer operations/shapes/quantization,
and Python/C model-header validation parity. The wheel includes the chain
compiler and no research inputs or vendor libraries.

Next: broaden native convolution kernels/spatial shapes and graph depth, add
pooling/add, implement calibration, and verify a useful trained end-to-end model.
A distinct RV1106 SoC and publication also remain outstanding (the attached
board's NPU is the shared rv1106-rknpu IP; see hardware_identity.md).

### Independent 2x2 pooling milestone

`build_identity.py --pool max|average` creates development Conv→Pool oracles.
The pools are separate NPU tasks: enable mask 96, interrupt mask 3072,
37 register words, command offset 0x440. The preceding Conv link control is
0x14 (rather than 0x40 for a following Conv). Pool commands target blocks 0x6000
and 0x7000, tags 0x4001 and 0x8001. Read address is register 0x701c; write address
is 0x6070. Both pools preserve the convolution's INT8 scale and zero point.

`emit_pool.py` independently emits the preceding Conv[/Relu] and pool program,
with newly packed constants at 0x600/0x640 and intermediate tensor at 0x1000.
It reads ONNX weights only, with no capture or RKNN model input. Currently it
accepts input `[1,3,8,8]`, 1x1 Conv[/Relu], 2x2 stride-2 pooling, and output
`[1,3,4,4]`. `raw_replay.c` has a development `OPEN_NPU_POOL=1` setting to submit
the second task with the pool descriptor and read 4x4 output.

`build_pool_suite.py` produces four new graphs (MaxPool/AveragePool, each with
and without Relu), with dense signed weights and bias, seed 110314. These were
never compiled with RKNN. `generated_pool_{max,average}_r{0,1}.log` record
**272 inferences and 13,056 exact pooled INT8 bytes**. Each graph passed four
base inputs and 64 deterministic random inputs using a relocated DMA arena.
`verify_pool.py` compares synchronized output to the independently quantized
Conv result followed by a host pooling reference. MaxPool chooses the signed
maximum. AveragePool divides the four signed values' sum by four and rounds
halfway values to even; both positive and negative cases are included.

Pooling now lives in `src/open_rknpu/pooling.py`, with explicit graph checks;
`emit_pool.py` is a compatibility wrapper. Public container profiles 3/4 identify
max/average pooling and infer a 4x4 output from the validated 8x8 input. The C
runtime constructs the 37-register task, checks output lengths, and unpacks only
the reduced output. `ornpu_info` now has output_height/output_width at the end;
C applications must rebuild against the updated development header/library.

`research/build_pool_api_suite.py` generates 64 inputs per graph (seed 110315).
`pool_api_suite.log` proves **256 public-API inferences and 12,288 exact bytes**,
including persistent models and invalid-length rejection. Board file-runner
checks also matched 48-byte pooled and 480-byte wide-channel outputs; the latter
exposed and fixed an undersized output buffer in `runtime/main.c` left over from
the original three-channel profile. Runner output capacity is now 1024 bytes,
with explicit size checks before reading inputs or submitting hardware work.

Next is feature reduction to 1x1, deeper graphs, and a trained classifier with
calibration. `pool_average8` is a
development probe for an 8x8 kernel/stride average pool over the full feature map.
Its capture completed: the vendor compiler lowers this operation to two extra
convolution tasks (7x7 producing 2x2, then 2x2 producing 1x1), rather than one
pool task. The native output is `[1,2,1,1,16]`, not the single-tile profile used
above. See `pool_average8_build.log`, `pool_average8_capture.log`, and
`capture_pool_average8/`. These commands are not independently emitted yet;
the 2x2 pool results do not establish arbitrary/global pooling support.

### Reduction to 1x1 using explicit pooling stages

Additional captures `pool_average_h4`, `pool_max_h2`, and `pool_average_h3`
isolate smaller pooling geometry. For an even square input width N, pool
dimension fields use N-1 and N/2-1, input row stride `N*16`, input surface stride
`N*N*16`, and output stride `(N/2)*(N/2)*16`. The 2x2 max-pool capture confirms
the pooling engine handles 1x1 output. The 3x3 average-pool capture consumes its
top-left 2x2 region using the same pool engine; the 2x2 global-average oracle
instead chooses a convolution. That compiler choice does not prevent direct
pool execution on 2x2 INT8 input.

Independent Conv[/Relu]→Pool→Pool→Pool graphs now run on the pooling engine for
all three stages (8→4→2→1), using matching MaxPool or AveragePool operators.
`generated_reduce_{max,average}_r{0,1}.log` prove **272 inferences and 816 exact
logical output bytes**. `verify_pool.py` rounds average pooling separately after
each stage; this is intentionally the explicit graph's quantized execution,
not a claim of a single unrounded/global mean. `build_pool_suite.py` generates
these four graphs reproducibly alongside the one-pool graphs.

`src/open_rknpu/reduction.py` emits four tasks, with command offsets
0/0x440/0x5c0/0x740, weights/table at 0x900/0x940, and intermediate buffers
0x1000/0x1400/0x1800. Input/output remain 0x2000/0x3000 in a 16 KiB arena.
The public format profiles 5/6 identify max/average three-stage reduction.
Python/C loaders derive 1x1 output; the runtime submits four tasks and returns
three packed output bytes. `reduction_api_suite.log` proves **256 public API
inferences, 768 exact bytes**, persistent model reuse, and invalid-size rejection.
Fifteen host tests include header parity for the four-task profiles.

This enables spatial reduction, but the supported graphs still need broader
convolution shapes/channels, calibration, and a trained useful model. General
GlobalAveragePool and arbitrary graph scheduling remain unimplemented.

### Dataset calibration

`open-rknpu compile --calibration DIRECTORY` accepts uint8/float32 NCHW `.npy`
batches `[N,3,H,W]` with finite values in [0,255]. `calibration.py` evaluates each
sample using ONNX's host reference and records
min/max (including zero) for every Conv output after any fused Relu/Clip, without
requiring the legacy compiler to accept the graph. Samples are
streamed from memory-mapped arrays. Each range becomes an affine INT8 scale and
zero point, used by the corresponding layer's independent emitter. The CLI
writes a `.calibration.json` sidecar. No vendor compiler/runtime is involved.
Calibration now works under `--sequence` for profiles that define an output-range
contract; `sequence_calibration_suite/README.md` records **6 models, 96 board
inferences and 77,824 exact bytes** plus an analytic-versus-calibrated accuracy
report. `open_rknpu/accuracy.py` adds per-layer round-trip error and
classification-accuracy helpers that keep integer exactness separate from
float-model accuracy.

`calibration_experiment.py` uses seed 110316 to generate 32 calibration samples
and 64 separate evaluation samples for six graph profiles. Each is compiled
with analytic and measured ranges. `calibration_suite.log` records **12 models,
768 board inferences, 53,760 exact integer output bytes**. Metrics in
`calibration_suite/manifest.json` compare dequantized predictions against the
ONNX float reference on evaluation inputs:

| Graph | Analytic MAE | Calibrated MAE |
| --- | ---: | ---: |
| Dense 1x1 | 0.44099 | 0.43126 |
| Two-layer, 16 hidden channels | 2.67775 | 1.00817 |
| Conv-Relu-MaxPool | 0.05580 | 0.06128 |
| Conv-AveragePool | 0.41722 | 0.40094 |
| Three MaxPool stages | 0.10381 | 0.14659 |
| Three AveragePool stages | 0.41867 | 0.38801 |

The max-pool models expose missed calibration extrema: maximum absolute error
increased from 0.594 to 12.578 for one-stage max pooling and from 0.559 to 5.901
for three stages. Min/max calibration is data-dependent, not a guaranteed
accuracy improvement. Representative calibration and separate model-level
accuracy evaluation remain necessary. This implements basic calibration, not
percentile/KL optimization, calibration-aware training, or a trained model demo.

### Wider spatial convolution

`k3out4`, `k3out8`, and `k3out16` vendor probes place a single off-center tap
in padded 3x3 filters. They confirm the existing independent tap-major packing
extends across all output channels: each tap stores output channels rounded up
to four, each with three input weights and one weight-zero-point padding byte.
Weight sizes are `9 * aligned_output_channels * 4`. Output-channel register
fields are the same formulas already established for 1x1.

The public compiler/runtime now accept 2–16 outputs for padded 3x3 Conv[/Relu].
`research/build_wide_suite.py --kernel 3` generates 30 new signed dense graphs with
bias and ReLU on/off, representative rectangular H/W 5..8, and 16 inputs each.
`wide_k3_suite.log` records **480 inferences, 178,112 exact output bytes**, with
invalid-length checks. No graph in this suite was compiled with RKNN.

For spatial convolution on native intermediates, `chain4k3` captures
Conv(3→4,1x1)→Relu→Conv(4→3,3x3,pad1). Compared with native 1x1, fields change
to `1010=108, 1030=1b0, 1034=90, 1038=3030003, 1068=101, 1188=48` (hex),
plus addresses/quantization. Native 3x3 packing is also tap-major: three output
rows of 16 INT8 weights per tap, including padding with each weight zero point.
The 432-byte packed weight candidate matches every captured weight byte.
`analyze_chain.py` composes both layers with signed INT8 padding -128 and matches
**768/768 logical bytes**. This is captured-profile arithmetic evidence;
independent native 3x3 multi-layer emission is the next step.

## Dense convolution, quantization, and public runtime checkpoint

The original three-channel checkpoint supports dense signed 1x1 and padded 3x3
convolution, bias, and fused Relu. H/W remain 5..8,
and input values represent unsigned bytes with scale 1 and zero point 0.

Per output channel, affine INT8 weight quantization includes zero in the weight
range. The channel scale is `(max - min) / 255`; its zero point is rounded and
clipped to INT8. Weights use float32 arithmetic before rounding. The stored bias
is `round(float_bias / weight_scale) + 128 * sum(qweight - weight_zero_point)`.
This compensates for the hardware treating input bytes as centered at 128.
Padded 3x3 input positions represent raw zero, hence centered value -128.

Requantization has two separate rounding stages. With accumulator `a`, channel
multiplier `c`, global multiplier `m`, and shift `s`:

```
p = a * c
t = (p + 8191 + ((p >> 14) & 1)) >> 14
y = clip_int8(((t * m + (1 << (s-1))) >> s) + output_zero_point)
```

For `s == 0`, omit the second rounding term. Relu clamps the accumulator at zero
before these stages. The first stage rounds halfway values to even. Earlier
tests did not distinguish this from rounding halves upward; output-channel
probes exposed the difference. Channel multipliers approximate weight-scale ratios with
14 fractional bits; the global multiplier approximates maximum weight scale
divided by output scale. See `../src/open_rknpu/quantization.py` for constraints
and the executable integer reference. Combining the rounding stages is wrong:
the signed 1x1 random-input experiment produced 111 differing bytes with combined
rounding, versus zero with separate rounding.

Independent three-channel `ties_positive` and `ties_negative` graphs with
diagonal weights ±[0.25, 0.5, 1] confirm the first-stage halfway rule for both
signs: **7,680/7,680** bytes across 40 board inferences. Neither graph was
compiled with RKNN. `tests/test_quantization.py` retains a regression against
their recorded ramp outputs and verifies that the old rounding rule fails.
All six prior dense/signed/biased/kernel experiment logs still match exactly.

3x3 weight storage is tap-major: each tap occupies 16 bytes, with four output
rows of four input weights. Three logical input weights are followed by the
channel weight zero point; the fourth output row is zero. The weight block is
aligned to 64 bytes. The following channel table stores INT32 biases, negated
weight zero points, and UINT16 channel multipliers.

Independent graphs never compiled by RKNN passed these checks:

| Graph | Inputs | Matching integer output bytes |
| --- | ---: | ---: |
| Dense signed 1x1, 7x6, bias | 68 | 8,568 / 8,568 |
| Dense signed padded 3x3, 7x6, bias | 68 | 8,568 / 8,568 |
| Same 3x3 with Relu | 68 | 8,568 / 8,568 |

Evidence is in `generated_dense_heldout.log`, `generated_k3dense_heldout.log`,
`generated_k3relu_heldout.log`, and their metadata under `generated/`.
Exactness here means the specified quantized integer computation; float-model
error is measured separately by `verify_generated.py`.

The production container has a 96-byte versioned header, validated profile and
shape fields, FNV-1a checksum, and an 8192-byte command/constant payload. This is
a trusted-compiler artifact format, not a sandbox for hostile hardware commands.
`../runtime/` provides a libc-only runner and static C library. The API accepts
packed NHWC UINT8 inputs, returns packed NHWC INT8 outputs, and reuses DMA
allocations across calls. Synchronization after submission is essential: recorder
snapshots taken before FROM_DEVICE synchronization can contain stale data. Use
the oracle's synchronized `OUTPUT` lines as correctness evidence.

`../research/build_application_suite.py` generated every combination of H/W 5..8,
kernel 1/3, and Relu off/on. `../tests/board_api.c` tested 16 inputs per model,
reused model handles, and rejected invalid input/output lengths. The board log
`application_suite.log` records **64 models, 1,024 inferences, and 129,792 output
bytes with zero mismatches**. The camera process remained present and reported
CMA allocation totals returned to their pre-test values.

A clean wheel installation without the RKNN module reproduced identical model
bytes and passed host validation tests. Packaging exists locally; publication,
multi-layer models, broader channels/shapes, calibration, pooling/add, and
distinct-RV1106-SoC validation remain outstanding.

### Independent native 3x3 emission

The public chain compiler now accepts a padded 3x3 second convolution. It
constructs tap-major native weights, compensates input quantization in the bias,
and pads signed intermediate tensors with their zero point. The second bias
table moves after the aligned weight block; profile 2 retains two tasks.
`research/build_chain_suite.py --kernel 3` and `chain_k3_suite.log` prove 14 unseen
models, 448 inferences, and 86,016 exact bytes. Eighteen host tests pass.
The first convolution remains 1x1; deeper spatial networks remain outstanding.

Both chain convolutions now independently support 1x1 or padded 3x3. Constant
placement derives from weight sizes, preserving the old 1x1 payload layout.
`chain_spatial_{1,3}_suite.log` prove 28 new graphs, 896 inferences, and
172,032 exact output bytes for a 3x3 first layer and either second kernel,
covering hidden counts 3–16. The 18 host tests still pass.

### Combined spatial network and reduction

`network.py` combines two independent convolution programs with three pool
programs. Profiles 7/8 select max/average reduction. Programs occupy
0/0x440/0x880/0xa00/0xb80, constants start at 0xd00 and are bounded below
0x1400, and intermediates use 0x1400/0x1800/0x1c00/0x1d00. The 20 KiB DMA
budget is unchanged. `network_suite.log` proves 28 independently compiled
networks, 896 inferences, and 2,688 exact output bytes across hidden counts
3–16 and both pooling kinds. Inputs include 0/255/128 and random seed 110317.
Weights come from the independently generated spatial-chain fixtures.
This is an end-to-end spatial inference path; trained-model accuracy remains
unproven and the overall goal remains incomplete.


## 5x5 first convolution (2026-09-08)

The `k5tap` oracle differs from `k3tap` only in kernel-dependent sizes,
spatial padding, and relocated bias constants: registers 1030=0x190,
1034=0x64, 1038=0x05050004, 1068=0x202, 1188=0x32.
This confirms the existing tap-major weight layout extends to 5x5.
The open compiler now emits 5x5 pad-2 single Conv[/Relu], three UINT8
input channels, 2–16 outputs, existing H/W 5–8 range. Multi-layer profiles
still accept only 1x1/3x3; Python and C enforce that boundary.

Independent sparse corner-tap execution: 36 inputs, 6912 output values,
zero integer mismatches (`generated_k5tap.log`). Independently generated
dense weights and biases: 30 models, 480 inferences, 178112 output bytes,
all exact through the public C API (`wide_k5_suite.log`). Both Relu choices
and every output channel count 2–16 are covered, at representative shapes.
Generator: `PYTHONPATH=src python research/build_wide_suite.py --kernel 5`.
All 22 host tests pass including C/Python container validation for 5x5.
This removes the first-convolution kernel-size gap for MNIST, but its
one-channel 28x28 input and native second convolution remain unsupported.


## Grayscale first convolution (2026-09-08)

Oracle `grayk5` establishes the grayscale input conversion fields relative to
three-channel `k5tap`: 100c=0x20008000, 101c=0, 1024=0x10, 104c=0xe0,
1050=0x14000, 1054=0x10001, 105c=0, 1078=0xc00f300f. Input conversion
sizes 107c/1080 become 2*H, 1084=(2*H)<<16|1, corroborated at H=7
by `grayh7`. The native weight tile is still four input lanes; unused lanes
hold the weight zero point. Runtime input row stride is 16 bytes for gray.

Public single-convolution profiles now support one or three input channels.
All three kernels 1/3/5, 2–16 output channels, with/without fused Relu,
at representative H/W 5–8 passed: 90 models, 1440 inferences, 534336 exact
output bytes. Logs: gray_k{1,3,5}_suite.log. The generator accepts
`--input-channels 1`. C/Python loader parity and grayscale input byte count
are covered; all 23 host tests pass. Multi-task profiles still require RGB.

Oracle runner now sets LD_LIBRARY_PATH explicitly to the isolated research
runtime. Its capture pull uses source `/.` so retries do not nest directories.
The first grayk5 attempt selected the system runtime and failed decoding;
the successful captured run used the isolated development runtime.


## Larger grayscale tensors (2026-09-08)

`gray28k5` captures a 28x28, one-input/eight-output 5x5 convolution.
The first task remains 126 commands. For grayscale, S=align_up(W,16):
103c=S<<16, 1044=S<<16|S/8, 107c=1080=H*S/8,
1084=(H*S/8)<<16|1, and both halves of 118c are H*S/8-1.
The existing kernel, output-shape, and weight layout formulas continue to hold.
The open runtime retains input at 8192 and output at 12288, with arena size
12288+align_up(H*W*16,4096). At 28x28/32x32 this is 28672 bytes,
plus the 4096-byte task allocation. UINT8 grayscale row stride is S bytes.

Public single-convolution grayscale H/W now ranges 5–32; RGB and linked
profiles retain their earlier bounds. Board verification covers 24 models,
192 inferences, 619008 exact output bytes, with signed dense weights/biases,
all three kernel sizes and both activation choices across shapes 8x9,9x8,
15x16,16x15,16x17,17x16,28x28,32x32. Generator:
`PYTHONPATH=src python research/build_large_gray_suite.py`.
Tests were transferred one at a time to fit board storage; `gray_large_suite.log`
records each original model index. The initial sparse 28x28 probe separately
passed eight inputs. All 24 host tests pass, including C/Python rejection of
incorrect large-input strides and undersized arenas.


## Trained MNIST first layer and affine input (2026-09-08)

Open compilation of the upstream normalized first Conv/Relu now runs on the
board with original trained weights/bias. Generator `build_mnist_first_layer.py`
uses the fixture range for input scale 0.2710929811 and UINT8 zero point 135.
Three additional configurations exercise zero points 0,128,255 and other scales.
`mnist_first_suite.log`: 4 configurations, 32 inferences, 200704 exact output
bytes versus independent integer reference. Inputs include fixture, real zero,
byte extrema, and deterministic random data. Fixture first-layer mean absolute
error versus float ONNX is 0.1651212; max 1.2003005, output scale 1.24404645.
This is not full-model or dataset accuracy evidence.

Input real value is (u8-Z)*S. Accumulator bias becomes
round(bias/(weight_scale*S))+(128-Z)*sum(centered_weights); global multiplier
includes S. Register 1184 holds (Z-128)&0xffff for zero-valued spatial padding.
Version 2 of the container stores input scale bits at 88 and unsigned Z at 92;
version 1 remains byte-compatible with defaults. Both loaders validate these
fields and the C API exposes them. CLI input overrides are limited to a single
Conv[/Relu], with calibration combination explicitly rejected for now.
All 25 host tests pass, including invalid v2 quantization and v1 reserved fields.


## MNIST-shaped native second convolution (2026-09-08)

Oracle `native14k5` is grayscale 14x14 -> Conv1x1/Relu 8 channels ->
native Conv5x5 pad2, 16 output channels. The generalized `analyze_chain.py`
checks all 12544 synchronized vendor output bytes exactly. Native weights
remain tap-major, then output channel, then 16 input lanes. Bias/channel
parameter groups remain four outputs per 32 bytes.

`PYTHONPATH=src python research/emit_native14.py` builds an
independent random-weight graph using no capture data. The first task comes
from the open grayscale compiler, the second from semantic register fields.
`OPEN_NPU_NATIVE14=1 OPEN_NPU_TASKS=2 OPEN_NPU_RANDOM_CASES=32` selects the
bounded research harness layout: payload 16KiB, input0x4000, hidden0x5000,
output0x6000, arena32KiB, plus relocation4096 and task4096. Log
`generated_native14.log`: 36 cases, 112896 bytes, zero integer mismatches.
An initial attempt inherited grayscale register1078=0xc00f300f; native
input requires 0x400f400f. Correcting that field resolved the mismatch.

For this native shape: 1010=0x9c (buffer configuration, not yet generalized),
103c=7<<16,1044=14<<16|7,107c=56,1080=196,1084=14<<16|14,
1188=8*K*K,118c=13<<16|13. Native_quantize now derives output count from
weights rather than hardcoding three. Public linked profiles remain restricted;
this experiment will feed general task scheduling and memory allocation.
All 25 host tests pass; camera process283 remained running throughout.


## Task-table runtime integration (2026-09-08)

The native14 independent graph now runs through the public runtime as
`ORNPUSEQ` version3, with explicit tasks, payload/arena sizes, IO offsets,
output dimensions, and input/output quantization. See runtime/sequence_format.md.
`sequence_suite.log`: 36 cases,112896 exact bytes. Legacy profiles now map to
internal task descriptors and share submission/IO handling with sequences.
`network_suite_sequence_runtime.log`:28 models,896 inferences,2688 exact bytes.
All 26 host tests pass, including C/Python parity on malformed sequence
headers, task bounds, overlapping IO, quantization, truncation, and checksum.
The runtime and packages were rebuilt; CLI inspect recognizes the format.
Automatic graph scheduling/emission into this format is the next compiler step.


## ONNX sequence lowering and trained pooling prefix (2026-09-08)

`compile --sequence` now invokes scheduler.py: normalize ONNX; lower initial
Conv[/Relu]; validate and lower 2x2 stride2 MaxPool/AveragePool stages; allocate
programs/constants, external input and all native activations; emit ORNPUSEQ.
No captured command data or vendor compilation participates. Affine input
quantization propagates from the initial convolution through pooling.

`build_mnist_pool.py` compiles the original trained Conv/Relu/MaxPool prefix.
`mnist_pool_serial.log`:8 inputs,12544 exact bytes. Broader rectangular and
repeated-pooling coverage: `scheduled_pool_serial.log`,12 models,96 runs,
88320 exact bytes. Generator research/build_scheduled_pool_suite.py covers
8x9,16x17,28x28, both pool types, one/two pooling stages, nonzero input ZP.

Linked submission reproducibly timed out for a repeated-pooling 16x17 case
(task counter2); isolated execution sometimes passed. Cause remains unresolved.
The scheduler therefore emits terminal programs and requests serial NPU
submissions via sequence header byte84=1. Each task completes before the next;
all tensor processing remains on NPU. This eliminated the observed timeouts
in the full suite. Earlier linked logs remain as failure evidence. A repeated
ADB directory push also created a nested copy; corrected transfers use source
`/.` to update the intended destination. All final serial results used updated
board files. The C/Python validators accept byte84 only as 0/1.
