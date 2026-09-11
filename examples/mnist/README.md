# Trained MNIST: RV1103 hybrid example

Complete inference runs on the board. Conv1/Relu1/MaxPool1 use the open NPU
runtime; Conv2/Relu2/MaxPool2/Reshape/MatMul/Add use model-specific C. The CPU
suffix fuses convolution, ReLU and pooling to avoid a full convolution buffer.
This example accepts only the pinned normalized MNIST graph, not arbitrary ONNX.

The original model and weights are Apache-2.0 licensed; see
[source provenance](../../research/pretrained/mnist/source.json) and
[model license](../../research/pretrained/mnist/LICENSE). The handwritten
compiler/example/runtime code is MIT licensed.

## Reproduce from repository root

The host environment needs NumPy and ONNX; no RKNN packages are used.
The normalized model and upstream fixture are in `research/pretrained/mnist`.
Build generates the NPU prefix, a C weight header and eight reference cases:

```bash
PYTHONPATH=src python examples/mnist/build.py
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime -Iexamples/mnist/build \
  examples/mnist/main.c runtime/open_rknpu.c -o examples/mnist/build/mnist-run
```

Use the installed ADB binary (full path is recorded in the investigation notes)
in place of `adb` below if it is not on PATH. Check available flash space before
copying. The executable, prefix and input file total 45,908 bytes; flash allocation
and outputs need additional space. Keep the camera service running.

```bash
adb shell 'pidof rkipc; free -m; df -h /userdata'
adb shell mkdir -p /userdata/open-npu-research/mnist-hybrid
adb push examples/mnist/build/mnist-run examples/mnist/build/prefix.bin \
  examples/mnist/build/inputs.u8 /userdata/open-npu-research/mnist-hybrid/
adb shell 'cd /userdata/open-npu-research/mnist-hybrid && chmod +x mnist-run && ./mnist-run prefix.bin inputs.u8 actual.f32 actual-prefix.i8'
adb pull /userdata/open-npu-research/mnist-hybrid/actual.f32 \
  /userdata/open-npu-research/mnist-hybrid/actual-prefix.i8 examples/mnist/build/
python examples/mnist/verify.py
```

## Measured 2026-09-08

Eight cases: the supplied trained-model fixture, three constant inputs, and four
deterministic random inputs. All 12,544 NPU output bytes match the integer
reference exactly. All 80 final logits match the independent ONNX suffix
evaluation within **0.000092 absolute error**. The supplied fixture predicts **3**,
as does the original float model. Its maximum logit difference from that original
float model is approximately **1.925**, including prefix/input quantization.
Input range was chosen from this single fixture; dataset accuracy is unmeasured.

One eight-case run averaged **10.793 ms NPU prefix + 12.828 ms CPU suffix =
23.621 ms**. Timing includes the runtime's prefix packing/submission/readback,
but excludes model loading and file I/O. This is a small baseline, not a sustained
throughput benchmark; the camera ran concurrently (PID 283 before and after).

Peak process RSS was **856 KiB**. This does not fully account for kernel/DMA
memory: the NPU arena requests 24 KiB and task storage 4 KiB. The CPU suffix
uses 7,296 bytes of float scratch plus caller buffers and stack overhead.
The executable is 37,140 bytes and its only ELF shared-library dependency is
`libc.so.0`. Board RAM remained about 6 MiB available after the run.

Next: offload additional proven native Conv shapes and measure the benefit before
generalizing this example into a graph executor.

## Second convolution offload (2026-09-08)

`build.py --native` extends the prefix with an independently generated native
14x14, 8-input/16-output-channel, padded 5x5 Conv. ReLU2, 3x3 pooling and dense
classification remain on the CPU. This is an opt-in model-specific experiment;
the original float CPU suffix remains the default because its numerical fidelity
is currently better.

```bash
PYTHONPATH=src python examples/mnist/build.py --native
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime -Iexamples/mnist/build-native \
  examples/mnist/main.c runtime/open_rknpu.c -o examples/mnist/build-native/mnist-run
```

Repeat the deployment commands above, replacing host `build/` with `build-native/`
and board `mnist-hybrid` with `mnist-native`. Verify using:

```bash
python examples/mnist/verify.py --native
```

All 25,088 native Conv output bytes match the independent integer reference over
eight cases. All 80 final logits match the ONNX suffix reference within 0.000031.
Fixture prediction remains 3. However, maximum fixture logit error versus the
original float model rises from 1.925 to **17.806**: the conservative uncalibrated
Conv2 output scale is 35.303. This confirms command execution, not acceptable
model accuracy. Representative calibration and held-out accuracy evaluation are
needed before preferring this variant for deployment.

Five interleaved eight-case batches per variant, using the same executable with
the camera running, averaged:

| Variant | NPU prefix | CPU suffix | Total |
|---|---:|---:|---:|
| CPU Conv2 baseline | 10.403 ms | 10.186 ms | 20.589 ms |
| NPU Conv2 | 4.422 ms | 0.067 ms | 4.490 ms |

Observed mean speedup: **4.59x**. Timing varied substantially: batch means ranged
14.828–24.994 ms for baseline and 0.336–12.574 ms for native. These are shared-board
observations, not an isolated measurement of Conv2 hardware speed. Raw samples
are in `build-native/timings.json`; correctness reports are in both build dirs.
Peak measured process RSS was 856 KiB for both; native arena requests 40 KiB plus
4 KiB task storage. The combined executable is 41,236 bytes. Camera PID 283 stayed
running and available board memory remained about 6 MiB.

An initial attempt timed out because 28x28 spatial settings remained in the
second Conv's output/bias blocks. Setting all affected dimensions and strides to
the verified native14 profile resolved the observed timeout. No driver change or
camera restart was required. Existing 28 host tests pass.

## Rough held-out sanity check (2026-09-08)

Tested 100 MNIST test-set images sampled without replacement using seed 110309,
with no calibration, parameter changes or tuning. Used upstream preprocessing
(pixel / 255, NCHW). Dataset test files are cached host-side under
`research/pretrained/mnist/test-data`; source URLs and SHA256 hashes are recorded
there. This small sample is not a full-dataset accuracy estimate.

| Variant | Correct / 100 | Agreement with float |
|---|---:|---:|
| Original float model (host) | 100 | 100 |
| Float model with existing quantized/dequantized input (host) | 100 | 100 |
| First Conv on NPU, second Conv on CPU (board) | 100 | 100 |
| Both Conv layers on NPU, analytic Conv2 range (board) | 13 | 13 |
| Both Conv layers on NPU, calibrated Conv2 range (board) | 100 | 100 |

The uncalibrated both-Conv variant predicts **5 for every image**. Its independent
host integer reference plus ONNX suffix reproduces the board logits within the
verification tolerance, localizing the failure to the conservative analytic
quantization rather than an unexplained board discrepancy. The earlier synthetic
supplied fixture was not evidence of useful digit accuracy.

## Calibrated Conv2 (2026-09-10)

`build.py --native --calibrate` measures the trained Conv2 output range on 256
MNIST test images that are disjoint from the 100 held-out images, using the same
open compiler and no vendor tools. The analytic range is extremely conservative
(output scale **35.303**); the measured range is `[-21.86, 12.34]`, giving output
scale **0.13411** and zero point **35** — about 263x finer.

Board results for the same 100 held-out images (integer reference reproduced on
the host and matched to the board within `2e-5` logit):

| Variant | Correct / 100 | Mean NPU ms | Mean CPU ms | Mean total ms |
|---|---:|---:|---:|---:|
| Hybrid (Conv2 on CPU) | 100 | 9.740 | 9.477 | 19.218 |
| Both Conv on NPU, analytic | 13 | 3.323 | 0.192 | 3.515 |
| Both Conv on NPU, calibrated | **100** | 2.383 | 0.070 | **2.454** |

Calibrating only the Conv2 output range takes the fully-offloaded network from
13/100 to **100/100** on this sample, matching the float model, at ~2.45 ms per
image with the camera service running (`rkipc` PID 283). This is a 100-image
sample, not a full-dataset accuracy estimate. The gain is the intended
calibration use: measured ranges replace a data-independent analytic bound, and
accuracy must still be reported separately from integer exactness
(`examples/mnist/accuracy.py`, `sanity-results/accuracy.json`).

Board averages over the 2026-09-08 run: hybrid 18.221 ms, both-Conv NPU 1.802 ms.
Camera PID 283 remained running.

## Full 10,000-image test set (2026-09-10)

`examples/mnist/full_dataset.py` streams the complete MNIST test set to the board in
1,000-image chunks (flash-limited), runs the prefix, and scores the pulled logits.

| Variant | Correct / 10,000 | Accuracy | Agreement with float |
|---|---:|---:|---:|
| Float model (host) | 9,890 | 98.90% | — |
| Both Conv on NPU, calibrated | **9,867** | **98.67%** | 9,922 |
| Both Conv on NPU, analytic | 892 | 8.92% | 903 |

The calibrated both-Conv NPU network is within **0.23 percentage points** of the
float model over the full test set, while the analytic variant collapses to one
class (predicts 5 for every image). Reports:
`sanity-results/full_dataset_calibrated_prefix_report.json` and
`..._native_prefix_report.json`. `tests/test_mnist_reports.py` guards the
relationship.

Reproduce preparation and reporting (the IDX files are downloaded and hash-checked by
`python examples/fetch_idx.py --dataset mnist`):

```bash
PYTHONPATH=src python examples/mnist/sanity.py
```

Push `sanity-results/inputs.u8` to the board. Run the existing `mnist-run` with
each unchanged prefix model and this input file; use `/dev/null` for the prefix
output argument to avoid large temporary output. Pull final float outputs as
`sanity-results/hybrid.f32` and `sanity-results/native.f32`, then:

```bash
PYTHONPATH=src python examples/mnist/sanity.py --report
```

Exact selected indices, labels, predictions, model hashes and counts are in
`sanity-results/report.json`; timings are in `sanity-results/board.log`.
