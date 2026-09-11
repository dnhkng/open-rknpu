# RV1103 primitives

Eleven deterministic, host-only examples that compile one verified RV1103
primitive at a time, check the result against the profile's Python integer
reference, and publish a board suite. No board, network, vendor toolchain or
PyTorch is needed; the board commands are printed as text.

```sh
PYTHONPATH=src python examples/primitives/01_native_conv.py
# or all of them: make primitives
```

Every script exits 0, prints one report line per compiled model and a one-line
summary, and writes `examples/primitives/build/<script>/` (git-ignored) with
`modelNNN.bin`, `inputNNN.u8`, `expectedNNN.i8` and `manifest.json`.

## What "verified" means here

The correctness criterion is **byte equality with the profile's Python integer
reference for the same quantization parameters** - the same discipline the
`research/build_*_suite.py` generators and the board runs use:

```
int8 reference: exact (4/4 cases, 8192 bytes) via native.native_input_reference
float quantization error: 0.58 LSB max (secondary, not correctness)
```

The first line is asserted by `common.assert_int8_reference`; a mismatch raises
and the script exits non-zero, so a wrong `expected*.i8` is never published. The
second line is only a quantization-quality metric against ONNX's float evaluator -
it is never called "exact". Scripts also assert that the published v5 container's
decoded shapes and input/output band match the emitter metadata the expected bytes
were computed from, and, where a board-verified suite exists, they replay that
suite's recorded `expected*.i8` files with the same pipeline (see the cross-check
lines).

`manifest.json` records `index`, `cases`, `output_bytes`, `input_shape`,
`output_shape`, `profile`, the container/task sizes and the quantization, plus any
other small JSON-safe emitter metadata.

## The examples

| Script | Primitive | Verified bounds | Demonstrates | Run |
| --- | --- | --- | --- | --- |
| `01_native_conv.py` | dense `Conv[/Relu]` (native16-input) | batch 1..16, in/out C1..128, H/W 1..128, odd K1..31, stride 1..4, explicit/asymmetric pads 0..K-1, dilation 1..17, fused Relu/Clip[0,6] | 6 shapes: RGB K3, C17 asymmetric stride2, C32 K1, K5 dilation2, C1->C128 stride4, C128->C3 stride3; `native_input_reference`; metadata + container bytes | `PYTHONPATH=src python examples/primitives/01_native_conv.py` |
| `02_conv_pool.py` | `Conv[/Relu]` + 2x2/2 pools | first Conv is the legacy RGB image layer; `MaxPool`/`AveragePool`, K2, stride2, zero pad; terminal pool = pooling profile, 2-3 pools = walk | one terminal pool (`conv-pool-terminal`) and two/three pools (`chain-walk`), both reference-exact | `PYTHONPATH=src python examples/primitives/02_conv_pool.py` |
| `03_conv_chain.py` | N-layer `Conv/Relu` chain + pool chain | `native-chain`: `[Conv,Relu]*(N-1)+[Conv]`, `[1,3,8,8]`, hidden C3..16, K1/K3; `chain-walk`: a pool that is not the last node | N=3, N=4, and an interior-pool walk; replays `native_chain_suite` + `walk_chain_suite` | `PYTHONPATH=src python examples/primitives/03_conv_chain.py` |
| `04_depthwise.py` | RGB multiplier-1 depthwise (+ pointwise) | C1..16, K1/3/5, stride 1/2, pads K//2, H/W 5..8; optional dense 1x1 pointwise | K3 stride1, C8 K3 stride2, K5 stride1 + pointwise head; replays 84 `depthwise_combined_suite` models | `PYTHONPATH=src python examples/primitives/04_depthwise.py` |
| `05_elementwise.py` | two 1x1 Conv branches joined | RGB 5..8, branch output C2..16, no broadcasting; Add/Sub/Max share one scale, Mul keeps both | Add, Mul, Sub, Max variants and their join references | `PYTHONPATH=src python examples/primitives/05_elementwise.py` |
| `06_constant_mul.py` | `Mul` by an immutable constant | one external RGB input 5..8 (batch 1..16) and a broadcastable float32 constant | scalar, per-channel, spatial `[1,1,8,8]`, full `[1,3,8,8]`, and the `Conv`+`Mul(scalar/per-channel)` fold | `PYTHONPATH=src python examples/primitives/06_constant_mul.py` |
| `07_lut_activation.py` | bounded `Conv`->`Sigmoid`/`Tanh` LUT | diagonal C3 1x1 stem, 8x8, input scale1/zp128, one scalar gain per channel, power-of-two weight scale, analytic range inside the sign-split table domain | Sigmoid/Tanh identity and a signed stem; `lut_reference`; replays 12 `lut_domain_suite` models | `PYTHONPATH=src python examples/primitives/07_lut_activation.py` |
| `08_transpose.py` | dense `ConvTranspose` | C1..16 -> C1..16, K3/K5, per-axis stride 1..2, asymmetric pads below K, legal `output_padding`, `output_shape`/SAME modes, K3 dilation2 -> sparse K5 | K3 stride2, K3 stride2 asymmetric, K3 dilation2 rewrite, K5 stride1; replays both dense transpose suites | `PYTHONPATH=src python examples/primitives/08_transpose.py` |
| `09_join_dag.py` | diamond fan-out/fan-in + join chain | 1x1 stem (optional Relu), dense heads C3..16, 3-channel head outputs, zero-centred joins, optional Conv tail; chains need >=3 heads | diamond Add, diamond Mul + Conv tail, three-head Add chain; replays `walk_join_suite`, `diamond_suite`, `join_chain_suite` | `PYTHONPATH=src python examples/primitives/09_join_dag.py` |
| `10_calibration.py` | measured activation bands | `Conv[/Relu/Clip]`, the `Conv-ReLU-Conv` chain per layer, and elementwise/Mul profiles accept measured ranges | same graph compiled analytic, `minmax`, `percentile` and `kl`, with the band table and per-variant quality | `PYTHONPATH=src python examples/primitives/10_calibration.py` |
| `11_mnist_digits.py` | real model first stage | terminal-pool profile on a 28x28x1 grayscale image (legacy emitter accepts C1 up to 32x32) | `Conv1->Relu->MaxPool1` from `research/pretrained/mnist/normalized.onnx`; replays `mnist_pool_suite` | `PYTHONPATH=src python examples/primitives/11_mnist_digits.py` |

`common.py` holds the shared graph builders (`conv_node`, `relu_node`,
`pool_node`, `join_node`, `model_graph`), `compile_and_report`,
`assert_int8_reference` / `report_checks`, the v5 `publish_suite` +
`print_board_recipe` pair and `cross_check_suite`.

## Running a published suite on the board

`publish_suite` always writes the **v5 named-tensor framing** that
`tests/board_io.c` loads. Profiles that emit the older v3 container are re-framed
losslessly (`as_named_tensor_container`: identical payload, task table, arena,
shapes and quantization, plus the two external tensor descriptors), and the
manifest records both `container_bytes` and `suite_container_bytes`. Every script
prints the exact cross-compile and adb commands; the shape is:

```sh
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime tests/board_io.c \
  runtime/open_rknpu.c -o examples/primitives/build/01_native_conv/board_io
adb shell mkdir -p /userdata/open-npu-research/01_native_conv
adb push examples/primitives/build/01_native_conv/{board_io,*.bin,*.u8,*.i8} \
  /userdata/open-npu-research/01_native_conv/
adb shell 'cd /userdata/open-npu-research/01_native_conv && ./board_io . 6'
# or, automated: PYTHONPATH=src python research/run_v5_suite.py 01_native_conv
```

`board_io` derives the per-model case count from each input file, compares every
external output byte with `expectedNNN.i8` and must print `model N: C inputs
passed`. `research/run_v5_suite.py` wraps the same steps and writes
`board_results_0.json`.

## Combining primitives into a model

A model is a graph of verified primitives. Keep every op inside a documented
profile and the compiler can take the whole graph; step outside it and the CPU has
to finish the job.

* `examples/mnist/` continues a compiled graph **on the CPU**. Its `build.py`
  compiles the `Conv1->Relu->MaxPool1` prefix exactly like script 11, then runs
  `Conv2/Relu/Pool2` and the 256x10 `MatMul` head in NumPy because `MatMul` is not
  a primitive. The board report is 98.67% (both-Conv calibrated) against 98.90%
  float on the full 10,000-image test set.
* `examples/mel-kws/` runs a **whole model on the NPU**: its 3x32x32 mel-CNN is
  only `Conv/Relu/MaxPool` plus a 1x1 head, so `open_rknpu.walk` emits every layer
  and the board reaches 98.00% INT8 with 191,968/192,000 exact bytes.
* `11_mnist_digits.py` is the bridge: it builds the first stage from the pinned
  pretrained weights, compiles it with `open_rknpu.scheduler.compile_sequence`,
  verifies the reference and explains both placements in its docstring.

The practical sequence is: start from `01`-`09` to confirm each op you need,
calibrate the trained bands with `10`, then compose the graph and let the
scheduler pick the profile (or the op-level walk) for you.

## Capabilities deliberately not exercised

Where a first choice was outside a profile, the example was adapted instead of
hidden:

* `08_transpose.py` uses the **dense** `ConvTranspose` only. The multiplier-1
  depthwise `ConvTranspose` shares the depthwise tap lanes and is board-verified
  in `research/transpose_k5_suite/` and `research/transpose_k5_dilation_suite/`,
  but its explicit integer reference does not reproduce every depthwise geometry
  used here, so it is not claimed in this script. `open_rknpu.transposed` exposes
  no reference function, so the dense reference lives in the script (the exact
  loop from the suite generators).
* `06_constant_mul.py` uses the standalone constant-`Mul` profile. A `Mul` with a
  *spatial* constant directly after a `Conv` is not folded by `normalize_model`
  and the elementwise profile requires both operands to be Conv branches, so that
  combination is rejected by the compiler and is not shown.
* `07_lut_activation.py` stays inside the diagonal-stem profile. Mixed input
  weights (`research/lut_domain_failed/`), table interpolation and other
  shapes/channels are retained blockers, not examples.
* `02_conv_pool.py` states plainly that the pooling *profile* only receives a
  terminal pool; two or three pools route to the already-verified chain walk.

Every profile used here has an integer reference, so no script prints
`int8 reference: not available`. `common.report_reference_unavailable` exists so a
future primitive cannot quietly fall back to the float metric and call that
"exact".

## Style and checks

Files start with the MIT SPDX docstring, explain the op and its verified bounds
(from `research/COVERAGE_EXPANSION_RESULTS.md` and `PRIMITIVE_ROADMAP.md`), keep
lines under 120 characters, and are clean under
`ruff check examples/primitives` (config in `pyproject.toml`: `E9` + `F`).

```sh
ruff check examples/primitives
for script in examples/primitives/[01]*.py; do
  PYTHONPATH=src python "$script" || exit 1
done
```
