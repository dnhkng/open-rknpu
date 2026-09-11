# LUT argument index: direct readout of the non-power-of-two residual

`compile --sequence` accepts a diagonal `Conv(1x1) -> Sigmoid|Tanh` LUT stem only
when `BASE_WEIGHT_SCALE/weight_scale` is a power of two. Other bands get the right
whole-bit negative-half gain but the board output differs from the integer
reference by one code on roughly 10% of values. This directory holds the probe
that measured **which table index the hardware actually reads**, so the residual
is no longer inferred from output bytes.

## Instrument

Every stem is emitted twice:

* with the real Sigmoid table, so a candidate index rule can be scored against the
  board byte for byte (`model000`, `model010`, ... , `expected*.i8`, `out*.i8`);
* with the table replaced by a **sign-aware linear ramp** whose output code is an
  affine readout of the index actually used. Bank 0 (the negative argument half,
  indices 0..512) is written as `clip((neg_pivot - i)*128)`, bank 1 (indices
  512..1024) as `clip((i - pos_pivot)*128)`, in Q15 units. Four
  `(neg_pivot, pos_pivot)` windows per stem `(256,512)`, `(384,640)`, `(448,768)`,
  `(512,896)` cover the reachable index range (`model001..004`, ...).

The input is a full 0..255 code sweep on every channel in four 8x8 images, so the
index is sampled at every input code for every channel (768 samples per model).

The Q15 value -> output code conversion was calibrated on the **control stem**
(`g = 1/32`, ratio 1), whose index `512 + 2q` is already verified:

```
code = clip(round(v * 255 / 32768) - 128)      # Sigmoid table, multiplier 255, shift 15
```

This reproduces all four control ramp windows exactly (768/768), so the ramp is a
trustworthy index readout. The four windows are intersected per sample; a unique
integer is an exact measurement.

## Result

`PYTHONPATH=src python research/analyze_lut_index.py`
(writes `analysis.json`, no board needed):

| stem | `BASE/scale` | `H` | index readable | index exact | byte mismatch | worst |
| --- | --- | --- | --- | --- | --- | --- |
| `ctrl_1_32` | 1.0 | 1 | 768/768 | **768/768** | 0 | 0 |
| `npot_0_02` | 1.5625 | 2 | 552/768 | 459/552 | 87/768 | 1 |
| `npot_0_03` | 1.0417 | 2 | 534/768 | 444/534 | 69/768 | 1 |
| `npot_1_48` | 1.5 | 2 | 573/768 | 489/573 | 90/768 | 1 |
| `npot_0_024` | 1.3021 | 2 | 504/768 | 414/504 | 84/768 | 1 |
| `npot_0_012` | 2.6042 | 4 | 570/768 | 468/570 | 93/768 | 1 |

What the readout establishes:

1. **The slope model is right.** The measured index is linear in the input code
   with the emitted slope `64·g` on the positive half and `64·H·g` on the negative
   half (`H = 2**ceil(log2(BASE/scale))`); a least-squares fit lands within
   0.05–0.15% of those slopes (the sparsest window is noisier). The measured
   negative/positive slope ratio is `2.00`, `2.00`, `2.00`, `2.00`, `4.02` for the
   five non-power-of-two stems, matching the `H = 2, 2, 2, 2, 4` that
   `negative_half_gain` predicts to within 0.5%.
2. **The residual is exactly one index step.** Every deviation from the emitted
   reference `512 + round(64·G·x)` is `-1`, `0` or `+1`, on ~17% of the readable
   codes, and it converts to the observed 1-code output differences. There is no
   gross error and no sign error.
3. **No affine fixed-point rule reproduces it.** For each stem and half the
   analysis searches every integer `(A, B, J)` with `J <= 14` for which
   `floor((A*q + B)/2**J)` equals the measured index on all readable samples. The
   control stem has many solutions (its rule is `2q`); every non-power-of-two stem
   has **none**. So the hardware index is not a rounded affine function of the
   input code: it comes from a quantized intermediate (an accumulator/declared-scale
   code stage) whose rounding this probe cannot separate from the measurement grid.

## Why the band stays rejected

`compile_lut` keeps raising

```
LUT stem weight scale must make BASE_WEIGHT_SCALE/scale a power of two; other bands
advance the negative table half by whole bits and leave ±1 rounding
```

The probe replaces the earlier byte-level evidence for that message with a direct
index measurement, but it does not produce a rule that makes the band exact, and
an approximate stem is not accepted. Closing it needs either a per-channel table
(channel-split tasks) or the hardware's internal argument quantization, which these
probes do not expose.

## Reproduction

```sh
PYTHONPATH=src python research/build_lut_index_probe.py
# stage the model/input files under /userdata, then per model:
#   board_run model###.bin input###.u8 out###.i8
PYTHONPATH=src python research/analyze_lut_index.py
PYTHONPATH=src python -m pytest tests/test_lut_index_probe.py
```

`out###.i8` files are the retained board outputs; `tests/test_lut_index_probe.py`
re-derives the measurement from them without a board.
