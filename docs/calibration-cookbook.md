# Calibration cookbook

Getting accuracy out of a trained model on this stack is almost entirely about the
**bands** — the affine map `real = (code − zero_point) × scale` that every tensor in a
compiled container carries ([quantization.md](quantization.md)). The compiler's default
bands are *analytic* (derived from weight magnitudes and the input range). They are safe
for the random-weight regression suites and wrong for a trained network. This page is the
practical path from "my INT8 model is at chance" to a measured band that works, grounded
in the suites and examples that are in this repository.

Every number below is tagged with the file it comes from; nothing here is illustrative.

## The 30-second version

1. If your float model is fine and the container is at chance, the bands are the problem —
   calibrate. The mel-CNN example is the canonical case
   ([examples/mel-kws/README.md](../examples/mel-kws/README.md)).
2. Build a directory of `[N,C,H,W]` `.npy` batches and run
   `open_rknpu.calibration.measure` ([calibration.py](../src/open_rknpu/calibration.py)).
3. Pass `report["ranges"]` as `calibration_ranges` to `compile_sequence` — and put the
   **output** band in `ranges["output"]`, never in a separate `output_range`
   ([quantization.md](quantization.md)).
4. Measure float accuracy, INT8 accuracy and their agreement, and report all three
   ([examples/mel-kws/README.md](../examples/mel-kws/README.md)).
5. If accuracy is right but a few bytes differ, that is a reference-vs-hardware rounding
   residual, not a calibration range bug — localize it instead of re-tuning ranges
   ([research/mel_kws_suite/README.md](../research/mel_kws_suite/README.md)).

## When analytic bands are enough — and when they are not

Analytic bands are what the compiler uses by default. They are safe for the
random-weight regression suites, because those weights and activations were generated to
sit inside the analytic bound. They are **wrong for trained networks**: the trained
activations are nowhere near the analytic bound.

The mel-CNN case is the worked example. In `examples/mel-kws/`, the first Conv's analytic
band came out **~20× too wide** and the later grids **collapsed onto their zero point**,
giving **10% (chance) accuracy**; the walk's analytic bands are described as "useless for
this network" ([docs/quantization.md](quantization.md),
[examples/mel-kws/README.md](../examples/mel-kws/README.md)). Calibrating every Conv fixed
it: the board run reached **98.00%** INT8 against **98.33%** float accuracy
([research/mel_kws_suite/README.md](../research/mel_kws_suite/README.md)).

The same collapse is recorded for a trained MNIST model: the Conv2 output scale was
**35.303 analytic vs 0.13411 calibrated** over 256 held-out images, and the fully-offloaded
both-Conv NPU variant went from **13/100 to 100/100** on a fixed 100-image held-out set,
matching the float model; on the full 10,000-image test set the calibrated variant scored
**98.67%** against float **98.90%** and analytic **8.92%**
([research/COVERAGE_EXPANSION_RESULTS.md](../research/COVERAGE_EXPANSION_RESULTS.md),
[plans/completion-plan.md](plans/completion-plan.md) P8).

Calibration is not always dramatic. On the Fashion-MNIST example the difference is inside
the noise: float **88.18%**, both-Conv analytic **88.14%**, both-Conv calibrated **88.15%**
([research/COVERAGE_EXPANSION_RESULTS.md](../research/COVERAGE_EXPANSION_RESULTS.md),
[examples/fashion/README.md](../examples/fashion/README.md)). Calibrate a trained model
anyway: the cost is a measurement pass, and the downside is chance accuracy.

## Building the `.npy` batches `measure` expects

`open_rknpu.calibration.measure(model_path, directory, method="minmax", percentile=99.99,
bins=2048)` runs ONNX's host reference over a directory of `.npy` files
([calibration.py](../src/open_rknpu/calibration.py)). Its contract is exact and it fails
closed:

* the model must have **exactly one graph input**, a **static rank-four NCHW** input with
  every dimension `> 0`, and **at least one `Conv`**;
* `directory` is globbed for `*.npy`; an empty directory is a `ValueError`;
* each file must be rank-4, `uint8` or `float32`, with `shape[1:]` equal to the model
  input's `shape[1:]` and `shape[0] >= 1`; every value must be finite and in `[0,255]`;
* the arrays are **NCHW even though the runtime consumes packed NHWC**; the suite notes
  this explicitly — "NCHW arrays for the ONNX host reference; the runtime fixtures below
  are NHWC" ([research/build_percentile_calibration_suite.py](../research/build_percentile_calibration_suite.py)).

Two details surprise people:

* **A batch file is a batch.** Every sample in `dim 0` is measured separately; the report's
  `"samples"` counts individual images, not files
  ([tests/test_calibration.py](../tests/test_calibration.py)).
* **Only Conv outputs are measured**, using the fused activation output when a `Relu` or
  `Clip` immediately follows the Conv (`measured_tensor_names`,
  [calibration.py](../src/open_rknpu/calibration.py)). If you want a band for the graph
  output, the final Conv's output *is* the tensor named in the report (that is why
  `ranges["output"]` works — see below).

A minimal, literal construction (the shape of the real one in
`examples/mel-kws/build.py`):

```python
import tempfile
from pathlib import Path
import numpy as np
from open_rknpu.calibration import measure

with tempfile.TemporaryDirectory() as folder:
    np.save(Path(folder) / "train.npy", train_images.astype(np.uint8))  # [N,3,32,32]
    report = measure("build/model.onnx", folder, method="percentile", percentile=99.9)

ranges = report["ranges"]
for name, entry in ranges.items():
    print(name, entry["min"], entry["max"], entry["scale"], entry["zero_point"])
```

`examples/mel-kws/build.py` measures on the **training** split and keeps the official test
split for scoring; the calibration and evaluation sets are disjoint
([examples/mel-kws/build.py](../examples/mel-kws/build.py)). `examples/primitives/10_calibration.py`
does the same with a synthetic `[32,3,8,8]` calibration set
([examples/primitives/10_calibration.py](../examples/primitives/10_calibration.py)).

**Use data that looks like deployment.** Measured ranges clip everything outside them, and
a calibration set that does not cover the real inputs can make things worse: on the
`sequence_calibration` suite calibration lowered output MAE on all three graphs but
**raised** maximum error on two — RGB float max `8.76 → 17.41`, native `12.62 → 573.70`
([research/sequence_calibration_suite/README.md](../research/sequence_calibration_suite/README.md),
[research/COVERAGE_EXPANSION_RESULTS.md](../research/COVERAGE_EXPANSION_RESULTS.md)).

## Choosing `minmax`, `percentile` or `kl`

All three share one measurement pass over the samples; the report records the method:

* `minmax` — the observed minimum and maximum, exact for the samples seen, **one pass**;
* `percentile` — the observed minimum plus the `percentile`-th quantile of the upper tail,
  so a few outliers cannot stretch the scale over the whole INT8 range. The lower bound
  stays the observed minimum (deliberately asymmetric), and the selection needs a **second
  pass** to fill a `bins`-wide histogram;
* `kl` — a TensorRT-style saturation search: a fine histogram per measured tensor, and for
  every candidate upper bound the range `[min, edges[i]]` is quantized into `levels = 128`
  levels (mass above the bound saturated into the top level), keeping the bound whose
  quantized distribution has the smallest KL divergence from the histogram. Also **two
  passes** ([calibration.py](../src/open_rknpu/calibration.py)).

The resulting band is the same arithmetic for all three: `scale = (max − min)/255`,
`zero_point = clip(rint(−128 − min/scale), −128, 127)`
([calibration.py](../src/open_rknpu/calibration.py)).

The trade-off is measured, not asserted. `research/percentile_calibration_suite/` compiles
one `Conv 1x1 → Relu → Conv 1x1` graph with a heavy-tailed channel four ways; the numbers
are from its `accuracy_report.json` and `README.md`
([research/percentile_calibration_suite/accuracy_report.json](../research/percentile_calibration_suite/accuracy_report.json)):

| Variant | Output scale | Bulk MAE | Bulk max err | Extreme max err |
| --- | ---: | ---: | ---: | ---: |
| analytic (no calibration) | 39.69 | 10.16 | 28.97 | **23.45** |
| `minmax` | 4.07 | 1.89 | 7.63 | 850.4 |
| `percentile` 99.9 | 3.85 | 1.01 | 5.35 | 1569.5 |
| `kl` | 0.84 | **0.31** | **1.17** | 1674.9 |

* Any measured range beats the conservative analytic one on the bulk by roughly 5×.
* Clipping the 0.1% tail (`percentile`) halves the bulk error against `minmax` (1.01 vs
  1.89) and pays for it on saturated extremes (1569 vs 850).
* `kl` favours the bulk most (0.31) and clips the most (1675).
* The extremes are far outside the calibration distribution, which is exactly the case a
  measured range is not designed to cover — the conservative analytic range is best there.

So the method is a deployment decision about **which inputs the range must protect**. The
board check is unaffected by the choice: exactness is independent of the range, which is
the point — `4 models, 76 inferences, 14,592 exact output bytes`
([research/percentile_calibration_suite/board_summary.txt](../research/percentile_calibration_suite/board_summary.txt)).

Rules the selector implementation enforces that are easy to trip:

* `percentile` must be in `(0,100]`; `100` reproduces min/max exactly
  ([tests/test_calibration.py](../tests/test_calibration.py)).
* `kl` returns the full observed range when `bins <= levels` (`128`), so leave `bins` well
  above `levels`; the default is 2048 ([calibration.py](../src/open_rknpu/calibration.py)).
* The KL criterion is a proper divergence and picks a threshold at the bulk boundary; for
  a bulk in `[0,10]` with a spike at `100` the test asserts the chosen bound is `< 11`
  ([tests/test_calibration.py](../tests/test_calibration.py)).

**The CLI exposes only `minmax`.** `open-rknpu compile --calibration DIR` calls
`measure(model, dir)` with the defaults; to choose `percentile` or `kl` use the Python API
(`measure(..., method="percentile", percentile=99.9)`) and pass `report["ranges"]` to
`compile_sequence` ([cli.py](../src/open_rknpu/cli.py),
[quantization.md](quantization.md)).

## Passing the report to the compiler

```python
from open_rknpu.calibration import measure
from open_rknpu.scheduler import compile_sequence

report = measure("model.onnx", "calibration_batches", method="percentile", percentile=99.9)
binary, meta = compile_sequence("model.onnx", 1.0, 0, calibration_ranges=report["ranges"])
```

Two rules decide most failures ([quantization.md](quantization.md)):

* **Calibration and an explicit `output_range` cannot be combined.** The scheduler raises
  `calibration and output quantization overrides cannot be combined`
  ([scheduler.py](../src/open_rknpu/scheduler.py)). Put the output band **inside the
  report** as `ranges["output"]` — that is exactly what `examples/mel-kws/build.py` does,
  and it then sweeps tighter variants of that one band to pick the most accurate
  ([examples/mel-kws/build.py](../examples/mel-kws/build.py)).
* **Profiles without a band contract reject calibration rather than ignoring it.** The
  pooled-branches, LUT, LeakyReLU/PReLU, terminal Reshape and two-head profiles all raise
  rather than silently dropping the ranges, and so do the two elementwise DAGs and the
  strided profile ([scheduler.py](../src/open_rknpu/scheduler.py),
  [research/sequence_calibration_suite/README.md](../research/sequence_calibration_suite/README.md)).
  The join chain, join DAG, depthwise join, pool join, walk joins, native chain and the
  diamond *do* carry measured bands — see "Which profiles accept calibration" above for
  the exact per-profile contract.
  `compile_sequence` raises `calibration ranges lack tensor <name>` if the report does
  not cover a tensor the dispatched profile needs
  ([scheduler.py](../src/open_rknpu/scheduler.py)).

## Which profiles accept calibration

Not every profile the scheduler can dispatch has a measured-band contract, and the
difference is not visible from the graph alone. The table below is the audit: one row per
profile in `open_rknpu.scheduler.DISPATCH_PROFILES`, in dispatch order. "Required measured
tensors" is exactly the set of report keys the profile asks for through the shared
`measured_range` helper ([calibration.py](../src/open_rknpu/calibration.py)); a key that
is missing raises `calibration ranges lack tensor <name>`
([scheduler.py](../src/open_rknpu/scheduler.py)). `output_range` is listed separately
because a profile may accept one override and not the other.

| Profile | `calibration_ranges` | `output_range` | Required measured tensors |
| --- | --- | --- | --- |
| `qlinearconv` | rejected — `QLinearConv carries its own quantization parameters` | rejected | none (the graph carries its own bands) |
| `qdq-conv` | rejected — `Q/DQ Conv carries its own quantization parameters` | rejected | none (the graph carries its own bands) |
| `join-chain` | accepted | accepted (a `Mul` join or a Conv tail) | the stem and every head/tail Conv or fused-activation output; the join output is not a measured tensor |
| `join-dag` | accepted | accepted (the last join, `Mul` only) | the stem and every branch-layer Conv output |
| `depthwise-join` | accepted | accepted (`Mul` join only) | the stem, the dense branch Conv and the depthwise branch Conv |
| `pool-join` | accepted | accepted (`Mul` join only) | the stem and both branch Conv outputs |
| `pooled-branches` | rejected — `calibration is unsupported for pooled branches` | accepted | none |
| `lut` | rejected — `output override unsupported for LUT profile` | rejected | none |
| `mul-relu` | accepted | accepted | the graph output, supplied by hand: a bare `Mul[/activation]` graph has no `Conv`, so `measure` refuses it |
| `mul-clip` | accepted only when the graph-output band is exactly scale `6/255`, zero point `-128` (the fixed `Clip[0,6]` band); otherwise `Mul Clip[0,6] currently uses output scale 6/255 and zero point -128` | same fixed band only | the graph output, supplied by hand: `measure` refuses a graph with no `Conv` |
| `mul-add` | accepted | required (the scalar Add is folded into the output band) | the graph output, supplied by hand: `measure` refuses a graph with no `Conv` |
| `leaky-relu` | rejected — `output override unsupported for LeakyRelu profile` | rejected | none |
| `prelu` | rejected — `output override unsupported for PRelu profile` | rejected | none |
| `transposed-conv` | accepted | accepted | the graph output, supplied by hand: `measure` names `Conv` outputs, not the `ConvTranspose` output |
| `depthwise-pointwise` | accepted | accepted | the graph output |
| `reshape` | rejected — `output override unsupported for Reshape profile` | rejected | none |
| `standalone-mul` | accepted | accepted | the graph output, supplied by hand: `measure` refuses a graph with no `Conv` |
| `constant-mul` | accepted | accepted | the graph output, supplied by hand: `measure` refuses a graph with no `Conv` |
| `per-channel-constant-mul` | accepted | accepted | the graph output, supplied by hand: `measure` refuses a graph with no `Conv` |
| `runtime-scale-mul` | accepted | accepted | the graph output, supplied by hand: `measure` refuses a graph with no `Conv` |
| `two-head` | rejected — `output override unsupported for the two-head profile` | rejected | none |
| `join-walk` | accepted | accepted (a `Mul` join or a Conv tail) | the stem and every head/tail Conv or fused-activation output |
| `diamond` | accepted | accepted (a `Mul` join or a Conv tail) | the stem and every head/tail Conv or fused-activation output |
| `native-chain` | accepted, except with height-strip tiling (`tiles=`) which rejects — `calibration is unsupported for the height-strip tiled chain` | accepted (the final layer) | every layer tensor: the fused-activation output for hidden layers, the Conv output for the last |
| `legacy-conv-chain` | accepted | accepted | the first layer's fused-activation output and the final Conv output |
| `multi-input-elementwise-dag` | rejected — `output override unsupported for the multi-input elementwise DAG` | rejected | none |
| `elementwise-dag` | rejected — `output override unsupported for the elementwise DAG` | rejected | none |
| `elementwise-join` | accepted | accepted | the graph output, supplied by hand: `measure` names the branch Conv outputs, not the join output (a `Mul` join honours the band; `Add`/`Sub`/`Max` derive the output from `2 ×` the shared operand scale) |
| `chain-walk` | accepted | accepted | every Conv or fused-activation output |
| `native-input` | accepted | accepted | the graph output |
| `strided` | rejected — `output override unsupported for this strided profile` | rejected | none |
| `depthwise` | accepted | accepted | the graph output |
| `pooling-sequence` | accepted | rejected — `output override unsupported for this scheduled profile` | the first Conv or fused-activation output |

Three consequences are worth stating plainly:

* **Requiring a tensor that `measure` cannot produce is a hard stop.** The rows marked
  "supplied by hand" need a report key `open_rknpu.calibration.measure` cannot emit: a
  bare `Mul`/`Mul`+activation graph has no `Conv` at all (the measurer refuses it), and for
  an `elementwise` or `ConvTranspose` terminal the graph output is not a measured `Conv`
  tensor. Those profiles still enforce the measured-band contract when a caller builds the
  dict itself, but a raw `measure` report cannot satisfy them — `compile_sequence` raises
  `calibration ranges lack tensor <name>`. Measure the graph you compile, or calibrate a
  `Conv`-terminal variant.
* **A join that captures a shared band re-centers its operands.** In the join family the
  measured band is the *natural* band of a branch output; an `Add`/`Sub`/`Max` join then
  moves each operand onto one shared zero-point-0 grid (`_adjusted_scale`), so the
  emitted band is the widest zero-point-0 grid covering the measured one, not the
  measured band verbatim. A Conv that feeds a pool moves the same way. Only stages the
  join leaves free (a `Mul` operand, a Conv tail, the final layer) carry the measured
  band exactly.
* **`Mul`-to-`Clip[0,6]` has no free output band.** The clamp task is fixed at
  scale `6/255`, zero point `-128`, so a report whose output entry disagrees is refused
  rather than silently ignored.

## The zero-point-0 rule for a Conv that feeds a pool

A pooled tensor keeps the band of the Conv that feeds it, but the verified DPU pool
programs **assume zero point 0**, so a Conv feeding a pool is re-quantized onto a
zero-point-0 grid ([walk.py](../src/open_rknpu/walk.py), [quantization.md](quantization.md)).
The compiler does this for you (and `pooled_branches` does the same); the practical
consequences are:

* the measured band you passed for that Conv is not the band the container uses — it is
  moved onto zero point 0 with the widest scale that still covers it
  (`_adjusted_scale`, [walk.py](../src/open_rknpu/walk.py));
* so do not be surprised if `meta` disagrees with the report's `zero_point` for a pooled
  Conv, and do not "fix" it by removing the pool.

## A trick that removes input error: train on the bytes

The input band is yours to choose, and it is the one band that does not have to be
calibrated. `examples/mel-kws/` trains the network on the **exact bytes the runtime feeds
it** — the feature pipeline ends in `to_uint8`, and `train.py` casts those bytes to
float32 and trains on them directly, so both training and the board input files have a
single source of truth ([examples/mel-kws/train.py](../examples/mel-kws/train.py),
[examples/mel-kws/features.py](../examples/mel-kws/features.py)).

The model then consumes the byte image, and `examples/mel-kws/build.py` compiles with
`INPUT_SCALE = 1.0`, `INPUT_ZERO_POINT = 0`. The input contributes **no quantization error
of its own**, and the calibration ranges are measured in the same units the hardware sees
([examples/mel-kws/build.py](../examples/mel-kws/build.py), [quantization.md](quantization.md)).
If instead you train on `[0,1]` features, the runtime maps `byte = round(value × 255)` and
the input band is `1/255` with zero point 0
([examples/mel-kws/features.py](../examples/mel-kws/features.py)).

## Measuring float vs INT8 accuracy, and their agreement

Report **three** numbers, as the examples do: float accuracy, INT8 accuracy, and agreement
between them. The helpers are `open_rknpu.accuracy`:

* `dequantize`, `quantize_roundtrip` — band arithmetic;
* `error_metrics(integer, float_reference, scale, zero_point)` — MAE / RMSE / max error of
  the dequantized INT8 output against the float reference;
* `classification_accuracy(logits, labels)` — top-1 on `[N,C]` logits;
* `layerwise_quantization_error(model_path, ranges, inputs)` — round-trip error per
  measured Conv/activation tensor for one range set, the cheapest signal for judging a
  calibration set without touching hardware
  ([accuracy.py](../src/open_rknpu/accuracy.py)).

The mel-CNN measurements, on the official 300-utterance test split
([research/mel_kws_suite/README.md](../research/mel_kws_suite/README.md)):

| Metric | Value |
| --- | --- |
| float ONNX test accuracy | 98.33% (295/300) |
| INT8 board accuracy | 98.00% (294/300) |
| INT8 vs float agreement | 98.33% |
| exact output bytes | 191,968 / 192,000 (99.983%) |

Keep the two claims apart: **byte-exactness is against the profile's integer reference for
the same quantization parameters, not against the float ONNX model**. A compiled container
can be byte-exact on the board and still differ from its float parent; a small float gap is
normal, a byte mismatch against the integer reference is a finding
([quantization.md](quantization.md), [verification.md](verification.md)).

## Decision table: symptom → likely calibration cause → fix

| Symptom | Likely calibration cause | Fix |
| --- | --- | --- |
| INT8 at chance while the float model is fine; output bytes all equal (often `output_zero_point`) | analytic bands, not measured ones — the mel-CNN's first Conv band was ~20× too wide and later grids collapsed onto their zero point (10% accuracy); MNIST analytic Conv2 scale 35.303 vs measured 0.13411 (8.92% vs 98.67%) | calibrate every Conv and pass `calibration_ranges`; print the bands ([quantization.md](quantization.md), [research/COVERAGE_EXPANSION_RESULTS.md](../research/COVERAGE_EXPANSION_RESULTS.md)) |
| Good accuracy but a handful of output bytes differ | reference-vs-hardware **rounding ties** in a calibrated Conv — the mel-CNN's 32 differing bytes (192,000 total, ≤4 LSB, no classification change, interior cells, difference starting in the 3rd/4th Conv) | localize with per-stage probes; this is not a range-choice bug ([research/mel_kws_suite/README.md](../research/mel_kws_suite/README.md)) |
| MAE improves but max error explodes | the calibration set does not cover the real input extremes, so `percentile`/`kl` clips the tail — sequence_calibration: MAE down on all three graphs, max error `8.76 → 17.41` and `12.62 → 573.70` | use `minmax` or a higher percentile, and widen the calibration set ([research/sequence_calibration_suite/README.md](../research/sequence_calibration_suite/README.md)) |
| `calibration ranges lack tensor <name>` | the report came from a different graph, or a Conv has no entry | measure the exact ONNX you compile ([scheduler.py](../src/open_rknpu/scheduler.py)) |
| `calibration and output quantization overrides cannot be combined` (legacy path: `calibration and chain output override cannot be combined`) | both a report and an explicit `output_range`/`--output-scale` were passed | put the output band in `ranges["output"]` and pass only `calibration_ranges` ([quantization.md](quantization.md)) |
| `calibration is unsupported for pooled branches` / `output override unsupported for LUT/LeakyRelu/PRelu/Reshape profile` / `output override unsupported for the ... DAG` | the profile has no measured-band contract | pick a profile that has one, or calibrate a graph the walk accepts; see "Which profiles accept calibration" for the full table ([troubleshooting.md](troubleshooting.md#calibration-and-output-override-conflict)) |
| All output codes equal the container's `output_zero_point` even after calibrating the Convs | the **output** band is narrower than the real logit range | widen it or calibrate the output tensor ([quantization.md](quantization.md#failure-modes-seen-in-practice)) |
| A hidden-band shift on a `chain`/`chain_n` graph | fixed 2026-09-12: every chain layer now programs the border register `0x1184` with the band it reads (before the fix, deep chains were off by up to 1.6e10 mean error against float) — if you see it, the container predates the fix; recompile ([roadmap.md](roadmap.md)) |

## Worked recipe: the percentile calibration suite

This uses only checked-in generators and runners, so nothing has to be invented. It
reproduces the method trade-off table above and proves the board exactness is independent
of the method choice.

```sh
cd /path/to/open-rknpu

# 1. Regenerate the suite: analytic + minmax + percentile + kl on one graph.
PYTHONPATH=src python3 research/build_percentile_calibration_suite.py

# 2. Check the selector logic (histogram/percentile/KL criterion).
PYTHONPATH=src python3 -m unittest tests.test_calibration -v

# 3. Board-check all four containers against their own integer references.
#    (Legacy v3 containers go through research/run_profile_suite.py.)
PYTHONPATH=src python3 research/run_profile_suite.py percentile_calibration_suite
```

What to check afterwards:

* `research/percentile_calibration_suite/calibration_report_{minmax,percentile,kl}.json`
  — one report per selector, with `method`, `samples`, `tensors` and per-tensor
  `min`/`max`/`scale`/`zero_point`, plus `percentile` or `bins` where relevant
  ([calibration.py](../src/open_rknpu/calibration.py));
* `research/percentile_calibration_suite/accuracy_report.json` — the bulk-vs-extreme
  trade-off, exactly the table above
  ([accuracy_report.json](../research/percentile_calibration_suite/accuracy_report.json));
* `research/percentile_calibration_suite/board_results_0.json` and `board_summary.txt` —
  `PASS: 4 models, 76 inferences, 14592 exact output bytes`; the same total for every
  variant is the point ([board_summary.txt](../research/percentile_calibration_suite/board_summary.txt)).

For a trained network, the end-to-end recipe is the mel-CNN example: train, calibrate
(`percentile 99.9` over the 2,700 training utterances), compile, then stream the test set
through one container on the board and compare every byte
([examples/mel-kws/README.md](../examples/mel-kws/README.md),
[research/mel_kws_suite/README.md](../research/mel_kws_suite/README.md)):

```sh
PYTHONPATH=src python examples/mel-kws/fetch_data.py
python3 examples/mel-kws/train.py --epochs 60
PYTHONPATH=src python examples/mel-kws/build.py          # measure + compile + integer reference
PYTHONPATH=src python research/build_mel_kws_suite.py   # needs build/prefix.bin from the step above
PYTHONPATH=src python research/run_v5_suite.py mel_kws_suite --binary /tmp/board_io
```

`examples/primitives/10_calibration.py` prints the band table (analytic vs `minmax` vs
`percentile` vs `kl`) for its own graph and checks every variant against the composed
integer reference, which is the fastest way to see what your calibration changes
([examples/primitives/10_calibration.py](../examples/primitives/10_calibration.py)).

## See also

* [quantization.md](quantization.md) — bands, the zero-point-0 pool rule and the band failure modes.
* [troubleshooting.md](troubleshooting.md) — "Accuracy collapsed" and "Calibration and output override conflict".
* [research/percentile_calibration_suite/README.md](../research/percentile_calibration_suite/README.md) — the method trade-off with board evidence.
* [research/mel_kws_suite/README.md](../research/mel_kws_suite/README.md) — the trained-model residual, localized.
* [examples/mel-kws/README.md](../examples/mel-kws/README.md) — the whole pipeline and its measured accuracy.
