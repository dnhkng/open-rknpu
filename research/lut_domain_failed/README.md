# Retained failure: mixed-input LUT stems

This directory keeps the failing experiment behind the P6 blocker. The accepted
LUT family is a *diagonal* C3 1x1 stem with one scalar gain per channel; a stem
that mixes several input channels in one output channel is rejected by the
compiler because a single table cannot match the per-channel negative-half gain.

## Artifacts

* `mixed_stem_model.bin` — a compiled model from before the restriction, with
  weights `[[0.02,0.005,0],[0.005,0.02,0],[0,0.005,0.02]]` and bias
  `[0.8,-0.8,0]` (Sigmoid, 8x8/C3, declared stem scale 1/2048).
* `mixed_stem_ramp_input.u8` — an 8x8 ramp: every channel carries pixels
  `0,4,...,252`.
* `mixed_stem_ramp_output.i8` — the board output for that input
  (`board_run mixed_stem_model.bin mixed_stem_ramp_input.u8 out`).

## Observation

Inverting the table (the output is `sigmoid` sampled over `(index-512)/64`)
gives the hardware's table argument `u`. With the table built for a
`weight_scale/BASE_WEIGHT_SCALE = 0.64` correction, the measured `u` is
`1.29·x` (uniform across all three channels), so the hardware's negative-half
gain is `1.29/0.64 = 2.016`. The weight-scale rule predicts
`BASE_WEIGHT_SCALE/weight_scale = 1.5625`.

| channel | bias | `u/x` at negative `x` | implied hardware gain |
| --- | --- | --- | --- |
| 0 | +0.8 | 1.290, 1.290 | 2.016 |
| 1 | -0.8 | 1.291, 1.290 | 2.017 |
| 2 | 0.0 | 1.293, 1.283 | 2.020 |

The board harness reported the corresponding mismatch directly:

```
model 0 run 0 output 0: got -117 expected -107
model 0 failed: -22
```

Diagonal stems with the same single gain (including per-channel sign flips and
biases) pass exactly — see [`../lut_domain_suite/`](../lut_domain_suite/).

## Why it is a blocker, not a bug in the emitter

The negative-half gain of a mixed channel is a function of its effective weight
scale, and each output channel can carry a different mix. A single 1026-entry
table is therefore not exact for a general 1x1 Conv stem. Removing the blocker
needs either a per-channel table (channel-split tasks, plan P1) or a documented
per-channel gain formula; until then the compiler rejects the profile instead of
silently clipping.

The related declared-scale probe and ramp-table readout are retained in
`research/check_lut_domain.py` and `research/build_lut_ramp_probe.py`
(`research/lut_domain_probe/`).
