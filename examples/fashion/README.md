# Trained Fashion-MNIST: RV1103 hybrid and fully-offloaded variants

Second trained model for the P8 milestone. Same pinned graph as the MNIST
example, so `main.c` and the build/evaluation harnesses are shared; only the
weights, the training script and the dataset differ.

```
Input3 28x28x1
  -> Conv(1->8,  5x5, pad2) -> Relu -> MaxPool(2x2/2)      # 14x14x8
  -> Conv(8->16, 5x5, pad2) -> Relu -> MaxPool(3x3/3)      # 4x4x16
  -> Reshape[1,256] -> MatMul(256->10) -> Add
```

6,994 parameters. The model is trained locally by `train.py` (MIT, PyTorch CPU,
16 epochs, ~2.5 minutes); the dataset is Fashion-MNIST (MIT, Zalando SE) and is
not vendored — `train.py` expects the four official IDX gzip files in
`research/pretrained/fashion-mnist/data/`. Provenance and checksums are recorded
in `research/pretrained/fashion-mnist/source.json` and `model.sha256`.

## Measured 2026-09-10

Float model: **88.18%** on the full 10,000-image Fashion-MNIST test set.

| Variant | NPU placement | CPU placement | Board accuracy | Agreement with float |
| --- | --- | --- | --- | --- |
| Hybrid | Conv1, Relu1, MaxPool1 | Conv2 … Add | **88.15%** (8,815/10,000) | 9,979/10,000 |
| Both-Conv, analytic range | + Conv2, Relu2 | MaxPool2 … Add | **88.14%** (8,814/10,000) | 9,862/10,000 |
| Both-Conv, calibrated range | + Conv2, Relu2 | MaxPool2 … Add | **88.15%** (8,815/10,000) | 9,957/10,000 |

Conv2's analytic output range was 38.86 wide; the calibrated range measured over
256 held-out test images is `[-9.943, 10.812]`, i.e. output scale 0.08139 with zero
point -6 (analytic: 0.37294 / -15). Calibration barely moves the final accuracy
here because the float model is the limit, but it more than doubles the number of
images whose integer logits match the float model exactly (9,862 -> 9,957).

Eight-case reference check (`verify.py --calibrated`): all **25,088** NPU output
bytes match the integer reference and the final logits match the independent ONNX
suffix evaluation within **6.7e-6**; the fixture prediction is 9 (an ankle boot).

### Latency and memory (camera service running)

1,000 images streamed in one process, measured on the board:

| Variant | NPU ms/image | CPU ms/image | Wall time | Peak RSS |
| --- | --- | --- | --- | --- |
| Both-Conv (calibrated) | 1.448 | 0.089 | **1.64 s** (1.54 ms/image) | 536 KiB |
| Hybrid | 8.468 | 9.544 | 18.44 s (18.0 ms/image) | 536 KiB |

Warm steady state is faster than the per-image average: repeated 8-case runs
settle at ~0.28 ms/image for the both-Conv prefix, while the first run after model
load costs ~14 ms/image (NPU power/clock bring-up). The hybrid variant is
CPU-bound: its Conv2/ReLU/pool suffix needs ~7.7 ms/image of scalar C, an order of
magnitude more than the NPU spends on Conv1.

Arena sizes: hybrid prefix 2,496 bytes, both-Conv prefix 12,432 bytes.

## Reproduce

```sh
# 1. dataset (MIT): download the four pinned IDX files and verify their sha256
python examples/fetch_idx.py --dataset fashion
# 2. train and export the ONNX graph (writes normalized.onnx + model.sha256)
python3 examples/fashion/train.py --epochs 16
sha256sum research/pretrained/fashion-mnist/normalized.onnx | tee research/pretrained/fashion-mnist/model.sha256
# 3. compile the three variants
PYTHONPATH=src python examples/fashion/build.py
PYTHONPATH=src python examples/fashion/build.py --native
PYTHONPATH=src python examples/fashion/build.py --native --calibrate
# 4. cross-compile, push and check the eight reference cases
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime -Iexamples/fashion/build-native-calibrated \
  examples/fashion/main.c runtime/open_rknpu.c -o examples/fashion/fashion-run
# push fashion-run, the chosen prefix.bin and inputs.u8, then
./fashion-run prefix.bin inputs.u8 actual.f32 actual-prefix.i8
PYTHONPATH=src python examples/fashion/verify.py --calibrated
# 5. full 10,000-image accuracy through the board, chunked to fit flash
PYTHONPATH=src python examples/fashion/full_dataset.py   # board: drives adb over the 10,000-image set
```

`full_dataset.py` writes
`sanity-results/full_dataset_<prefix>_report.json`; the three reports from this
run are checked in. `ADB` and the serial can be overridden through the environment
(`ADB`) and the constants at the top of the script.

## Scope

The example accepts only this pinned graph, like the MNIST one. It establishes a
second trained model with measured held-out accuracy, an explicit NPU/CPU
placement, latency and memory figures, and a clear failure mode: an unsupported
graph or prefix geometry is rejected before any submit (`ornpu_inspect`).
