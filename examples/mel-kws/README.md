# Mel-CNN spoken digits: RV1103 NPU audio example

A real, trained keyword-spotting model running **entirely on the NPU**: 10-class spoken
digit recognition on the [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)
(FSDD, CC BY-SA 4.0, 8 kHz, 3,000 recordings, 6 speakers). This is the first real trained
audio model on this stack and the largest image the walk emits: `3x32x32` in, `8x8x10`
logits out, one job per inference.

```
input 3x32x32 UINT8 (log-mel, delta, delta-delta)
  -> Conv(3->16,  3x3, pad1) -> BatchNorm -> Relu
  -> Conv(16->16, 3x3, pad1) -> BatchNorm -> Relu -> MaxPool(2x2/2)   # 16x16x16
  -> Conv(16->16, 3x3, pad1) -> BatchNorm -> Relu
  -> Conv(16->16, 3x3, pad1) -> BatchNorm -> Relu -> MaxPool(2x2/2)   # 8x8x16
  -> Conv(16->10, 1x1)                                                # 8x8x10 logits
```

BatchNorm is a training aid only: it is folded into the preceding convolution before
export, so the ONNX graph contains only Conv/Relu/MaxPool and the *whole* model is emitted
by `open_rknpu.walk` (the op-level chain walk, including its new native16 image input and
calibrated bands). Classification averages the 8x8 logit map per class on the host -
global average pooling is linear, so it is exact, and `ReduceMean` is not an NPU
primitive.

## Measured 2026-09-11 (Luckfox Pico Mini B, `rkipc` running)

| Metric | Value |
| --- | --- |
| float ONNX test accuracy (official 300-utterance split) | **98.33%** (295/300) |
| INT8 NPU accuracy on the board | **98.00%** (294/300) |
| INT8 vs float agreement | 98.33% |
| board-exact output bytes | **191,968 / 192,000** (99.983%) |
| container | 16,992 bytes, 7 tasks, serial submission, input scale 1 / zero point 0 |
| calibration | percentile 99.9 over the 2,700 training utterances |
| latency | **3.47 ms** first sweep, **2.54 ms** mean of 5 sweeps (1,500 inferences) |
| model size | 4,090 parameters, ONNX 32,168 bytes |

The 32 differing bytes are four utterances that each differ in **one** of the 64 output
cells (|delta| <= 4 LSB at the 0.23 output scale) with no classification change. Stage
probes localise the residual: the first two convolutions and the first pool are byte-exact
on those utterances, the difference appears in the third or fourth convolution, and the
affected cells are interior - a reference-vs-hardware rounding residual in the calibrated
convolutions, not a padding/tiling/addressing error. `research/mel_kws_suite/README.md`
records the suite evidence (16 utterances, byte-exact) and the full measurement.

## Pipeline

```sh
PYTHONPATH=src python examples/mel-kws/fetch_data.py   # FSDD -> research/pretrained/fsdd
python3 examples/mel-kws/train.py --epochs 60                         # PyTorch, ~2 minutes
PYTHONPATH=src python examples/mel-kws/build.py        # calibrate + compile + reference
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime examples/mel-kws/main.c \
  runtime/open_rknpu.c -o examples/mel-kws/build/mel-kws-run
```

Then stage the runner and the four build artifacts on the board and run it; it streams all
300 test utterances through one container, compares every byte with the reference and
prints the accuracy and latency:

```sh
adb shell mkdir -p /userdata/open-npu-research/mel-kws
adb push examples/mel-kws/build/{mel-kws-run,prefix.bin,inputs.u8,labels.u8,expected.i8} \
  /userdata/open-npu-research/mel-kws/
adb shell 'cd /userdata/open-npu-research/mel-kws && ./mel-kws-run \
  prefix.bin inputs.u8 labels.u8 expected.i8 5 actual.i8'
```

## Front end (`features.py`)

`wave` (stdlib) + NumPy only, deterministic and dependency-free: 8 kHz mono PCM, a
256-sample Hann window every 128 samples over 4,224 samples (32 frames, 528 ms;
longer recordings are centre-cropped, shorter ones zero-padded), a 32-band triangular mel
filter bank (20-4000 Hz), `log(mel + 1e-6)`, per-utterance standardization clipped to
+-3 sigma, and first/second differences standardized the same way. The three channels are
mapped to `[0,1]` and then rounded to bytes - the model is trained on those exact bytes,
so the input has no quantization error of its own.

## Notes and limits

* FSDD is CC BY-SA 4.0 and is **not** redistributed here; `fetch_data.py` downloads the
  pinned archive (sha256 in `research/pretrained/fsdd/source.json`).
* PyTorch is a training dependency only; `build.py` and the board path need NumPy and ONNX.
* The walk's analytic bands are useless for this network (the trained activations are
  nowhere near the analytic bound), so `build.py` measures every Conv's band with
  `open_rknpu.calibration.measure`; that calibration path was added for this example.
* Input channels are exactly 3 and H/W is 32 because that is the walk's image contract
  (one native16 input task, <= 6144 atoms). Silero VAD and other 1-D/rectangular/LSTM
  models remain outside the primitive set - see `docs/plans/primitive-roadmap.md`.
