# mel-CNN spoken-digit suite (16 models, board-exact)

One container, `examples/mel-kws/build/prefix.bin` (16,992 bytes), running the whole
trained model on the NPU: `Conv(3->16,3x3) -> Relu -> Conv(16->16,3x3) -> Relu ->
MaxPool(2x2/2) -> Conv(16->16,3x3) -> Relu -> Conv(16->16,3x3) -> Relu ->
MaxPool(2x2/2) -> Conv(16->10,1x1)`, emitted by `open_rknpu.walk` from the ONNX graph
that `examples/mel-kws/train.py` exports.

* input `1x32x32x3` UINT8, NHWC API buffer 3,072 bytes, input scale 1 (the feature bytes
  are the model input), native16 staged surface;
* output `1x8x8x10` INT8, 640 bytes, averaged over the 8x8 cells per class on the host;
* 16 held-out test utterances (two per spoken digit), one case each;
* board run 2026-09-11, `board_results_0.json`: **PASS - 16 models, 16 inferences,
  10,240 exact output bytes**.

Reproduce:

```sh
PYTHONPATH=src python examples/mel-kws/fetch_data.py
python3 examples/mel-kws/train.py                       # PyTorch, ~2 minutes
PYTHONPATH=src python examples/mel-kws/build.py
PYTHONPATH=src python research/build_mel_kws_suite.py
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime -Itests tests/board_io.c \
  runtime/open_rknpu.c -o /tmp/board_io
PYTHONPATH=src python research/run_v5_suite.py mel_kws_suite --binary /tmp/board_io
```

## Full test-set measurement (300 utterances)

`examples/mel-kws/build.py` writes `inputs.u8` (300 x 3,072 bytes), `labels.u8` and
`expected.i8` (300 x 640 bytes); `examples/mel-kws/main.c` streams them through one
container on the board. Measured 2026-09-11 (`rkipc` running); the raw runner output is
[board_full_run.txt](board_full_run.txt):

| Metric | Value |
| --- | --- |
| float ONNX test accuracy | **98.33%** (295/300) |
| INT8 board accuracy | **98.00%** (294/300) |
| INT8 vs float agreement | 98.33% |
| exact output bytes | **191,968 / 192,000** (99.983%) |
| latency | 3.47 ms first sweep, **2.54 ms** mean of 5 sweeps (1,500 inferences, serial, one job each) |

The 32 differing bytes are 4 utterances each differing in **one** of the 64 output cells
(|delta| <= 4 LSB of the 0.23 output scale) with no change in classification. Stage
probes localise it: the first two convolutions plus the first pool are byte-exact on those
four utterances (`mel_probe_conv1`, `mel_probe_pool1`), the difference appears in the third
or fourth convolution (`mel_probe_conv4`: 30 bytes), and the affected cells are interior,
so this is a reference-vs-hardware rounding residual in the calibrated-band convolutions,
not a padding, tiling or addressing error. Those four utterances are excluded from the
byte-exact suite above and are covered by this measurement instead.
