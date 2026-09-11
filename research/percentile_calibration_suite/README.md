# Percentile and KL activation calibration (P7 residual)

`open_rknpu.calibration.measure` selects activation ranges with one of three methods:

* `minmax` — the observed minimum and maximum (the original behaviour);
* `percentile` — the observed minimum with the upper `100-percentile`% tail clipped
  (default 99.9%), so a handful of large activations cannot stretch the scale;
* `kl` — a TensorRT-style saturation search: build a 2048-bin histogram per measured
  tensor, and for every candidate upper bound quantize the range into 128 levels
  (saturation folded into the top level) and keep the bound whose quantized distribution
  has the smallest KL divergence from the histogram. `tests/test_calibration.py` pins the
  criterion on synthetic histograms (non-negative divergence, a spike is clipped, the
  minimum sits at the bulk boundary).

`percentile` and `kl` need a second measurement pass: the first pass takes the observed
range, the second fills the histogram.

## Graph and data

One `Conv 1x1 -> Relu -> Conv 1x1` graph at 8x8/C3 with one long-tailed hidden channel
(`w1[0] *= 20`, seed 110410). Calibration data is 32 samples of mostly small pixels with
0.2% saturated ones; evaluation is 16 bulk samples drawn the same way plus three extremes
(all-zero, all-255, all-128). Every variant is compiled through
`compile_sequence(path, calibration_ranges=...)`; the expected bytes are the integer
reference of that variant's own quantization.

## Accuracy: what a measured range trades

`accuracy_report.json` (MAE and max error against the ONNX float reference; the report
covers the graph output tensor):

| variant | output scale | bulk MAE | bulk max err | extreme max err |
| --- | --- | --- | --- | --- |
| analytic (no calibration) | 39.69 | 10.16 | 28.97 | **23.45** |
| `minmax` | 4.07 | 1.89 | 7.63 | 850.4 |
| `percentile` 99.9 | 3.85 | **1.01** | 5.35 | 1569.5 |
| `kl` | 0.84 | **0.31** | **1.17** | 1674.9 |

* Any measured range beats the conservative analytic one on the bulk by ~5x.
* Clipping the 0.1% tail (`percentile`) halves the bulk error against `minmax`
  (1.01 against 1.89) at the cost of the saturated extremes (1569 against 850).
* `kl` favours the bulk most (0.31) and clips the most (1675).
* The extremes are far outside the calibration distribution (all-255 input never appears
  in it), which is exactly the case a measured range is not designed to cover: the
  conservative analytic range is best there.

So the method choice is a deployment decision - which inputs the range must protect -
and the numbers above are the evidence, not a claim that one method wins.

## Board evidence

`tests/board_api.c` via `research/run_profile_suite.py`, 19 cases per model:
**4 models, 76 inferences, 14,592 exact output bytes** (`board_results_0.json`,
`board_summary.txt`). Exactness is independent of the range choice, which is the point:
calibration changes the quantization, not the integer semantics of the container.

Reproduction:

```sh
PYTHONPATH=src python3 research/build_percentile_calibration_suite.py
PYTHONPATH=src python3 research/run_profile_suite.py percentile_calibration_suite
PYTHONPATH=src python3 -m unittest tests.test_calibration -v
```
