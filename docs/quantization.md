# Quantization

The NPU computes in INT8 with INT32 accumulators. A *band* is the affine map between real
values and integer codes: `real = (code - zero_point) * scale`. Every tensor in a compiled
container carries one band, and getting those bands right is the difference between 10% and
98% accuracy on a trained model.

## Conventions

| Item | Convention |
| --- | --- |
| API input | UINT8, packed NHWC, `real = (byte - input_zero_point) * input_scale` |
| API output | INT8, packed NHWC, `real = (code - output_zero_point) * output_scale` |
| Internal grids | INT8, one band per tensor, zero point stored signed in register `0x1184` for the tensor a Conv reads |
| Image staging | a native16 input surface stores `byte - 128`, so the first Conv's input band is `(input_scale, input_zero_point − 128)` |
| Weights | INT8 per output channel (`weight_scales`, `weight_zero_points`) plus an INT32 bias |
| Requantization | per channel: `channel_multiplier = round(scale / max_scale × 16384)`, then a multiplier/shift pair; the reference model of the hardware path is in `quantization.reference` |
| Rounding | the hardware path rounds half away from zero with a `+8191 + ((product >> 14) & 1)` correction; the Python reference reproduces it and is byte-exact on the board suites |

Analytic bands (derived from weight magnitudes and the input range) are what the compiler
uses by default. They are safe for the random-weight regression suites and *wrong* for
trained networks: in `examples/mel-kws/`, the first Conv's analytic band came out ~20× too
wide and the later grids collapsed onto their zero point, giving 10% (chance) accuracy.
Always calibrate a trained model.

## Calibration

`open_rknpu.calibration.measure(model_path, directory, method=..., percentile=..., bins=...)`
runs ONNX's host reference over a directory of `.npy` batches (`uint8` or `float32`, shape
`[N,C,H,W]`, values in `[0,255]`, matching the model's input shape) and returns a report:

```python
{
  "method": "percentile",
  "ranges": {
     "<conv-or-activation-tensor>": {"min": …, "max": …, "scale": …, "zero_point": …},
     …
  },
}
```

Three selectors share one measurement pass:

* `minmax` — the observed minimum and maximum (exact for the samples seen);
* `percentile` — the lower bound plus the `percentile`-th quantile of the upper tail, so a
  few outliers cannot stretch the scale over the whole INT8 range;
* `kl` — a TensorRT-style KL-divergence saturation search over a fine histogram, trading a
  little clipping for better resolution.

Pass the report straight into the compiler:

```python
from open_rknpu.calibration import measure
from open_rknpu.scheduler import compile_sequence

report = measure("model.onnx", "calibration_batches", method="percentile", percentile=99.9)
binary, meta = compile_sequence("model.onnx", 1.0, 0, calibration_ranges=report["ranges"])
```

Rules the scheduler enforces:

* calibration and an explicit `output_range` cannot be combined — put the output band in
  the report instead (`ranges["output"]`), which is what `examples/mel-kws/build.py` does;
* profiles that have no calibration path reject it rather than ignoring it; the
  per-profile answer — which profiles accept `calibration_ranges`, which accept
  `output_range`, and which tensor names each one needs — is tabulated in
  [calibration-cookbook.md](calibration-cookbook.md#which-profiles-accept-calibration);
* a Conv that feeds a pool is still re-quantized onto a zero-point-0 grid, because the DPU
  pool task assumes it.

## A trick that removes input error: train on the bytes

The input band is yours to choose. `examples/mel-kws/` trains the network on the exact
bytes the runtime will feed it (float32 values in `[0,255]`), so the model's input is the
byte image and the compiler uses `input_scale = 1.0`, `input_zero_point = 0`. The input
then contributes **no** quantization error, and the calibration ranges are measured in the
same units the hardware sees.

## Measuring accuracy, not vibes

* `open_rknpu.accuracy` has the numeric helpers: `dequantize`, `quantize_roundtrip`,
  `error_metrics`, `classification_accuracy`, `layerwise_quantization_error`.
* Every profile has an integer reference next to its emitter; a compiled container is
  checked against it byte-for-byte. The board proof is an exact-byte comparison, not a
  float tolerance.
* Report quality with three numbers, as the examples do: float model accuracy, INT8
  accuracy, and agreement between them. `examples/mel-kws/` reports 98.33% float, 98.00%
  INT8 and 98.33% agreement.

## Failure modes seen in practice

| Symptom | Cause |
| --- | --- |
| INT8 accuracy at chance while the float model is fine | analytic bands too wide (no calibration), or a band that collapses a later grid onto its zero point |
| good accuracy, a handful of differing bytes | reference-vs-hardware rounding ties in one Conv, not a structural error — localize with per-stage probes (see `research/mel_kws_suite/README.md`) |
| a profile rejects calibration | that profile has no measured band path; the table in [calibration-cookbook.md](calibration-cookbook.md#which-profiles-accept-calibration) says which profiles have one |
| output band saturated | the output override is narrower than the real logit range; widen it or calibrate |
