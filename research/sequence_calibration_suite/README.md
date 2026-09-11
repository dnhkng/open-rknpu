# Sequence calibration — analytic versus measured ranges

Each supported graph is compiled twice through `open_rknpu.scheduler` (the
`compile --sequence` path): once with analytic output ranges and once with ranges
measured from a separate calibration set. Expected bytes come from the documented
integer references, so the board check establishes exact INT8 execution of both
variants, not accuracy.

**Verified on RV1103: 6 models, 96 inferences, 77,824 exact output bytes**
(`board_api_test sequence_calibration_suite 6`).

| # | Graph | Calibrated | Output scale | Float MAE | Float max error |
|---|---|---|---|---:|---:|
| 0 | RGB Conv 3x3 | no | 11.2327 | 2.9575 | 8.7566 |
| 1 | RGB Conv 3x3 | yes | 5.5023 | 1.7419 | 17.4086 |
| 2 | native C3->C32 K3 | no | 15.8994 | 3.9159 | 12.6223 |
| 3 | native C3->C32 K3 | yes | 9.2303 | 3.0491 | 573.7002 |
| 4 | Conv-Relu-Conv (C5) | no | 2.9638 | 0.7962 | 2.1702 |
| 5 | Conv-Relu-Conv (C5) | yes | 0.8407 | 0.3244 | 1.2367 |

Calibration lowered mean absolute error on all three graphs, but it **raised**
maximum error on two: min/max ranges are data-dependent and evaluation values
outside the measured range clip. This is the intended honest signal; calibration
is not a guaranteed accuracy improvement.

`accuracy_report.json` records per-layer round-trip quantization error for both
range sets, produced by `open_rknpu.accuracy.layerwise_quantization_error`.

## Reproduction

```sh
PYTHONPATH=src python research/build_sequence_calibration_suite.py
# cross-compile tests/board_api.c with runtime/open_rknpu.c, push the suite,
# then run: board_api_test sequence_calibration_suite 6
```

## Scope

Calibration is available for profiles that define an output-range contract.
Profiles that reject output overrides (LUT, LeakyReLU/PReLU, terminal Reshape,
strided, two-head) also reject calibration with a clear error. Percentile/KL
calibration, calibration-aware training and a trained-model accuracy claim remain
open in [completion-plan](../../docs/plans/completion-plan.md) phases P7–P8.
