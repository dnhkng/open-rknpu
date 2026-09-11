# LUT negative-half gain: measured mechanism and the residual blocker

The LUT table's **negative half** advances one entry per `1/64` of
`x * (BASE_WEIGHT_SCALE / weight_scale)`, where `BASE_WEIGHT_SCALE = (1/32)/255` is
the scale of the original verified identity stem. This directory holds the probes
that measured the gain on the board and the evidence for the remaining blocker.

## Measured rule

The accelerator shifts by **whole bits**, so the hardware gain is

```
H = 2 ** ceil(log2(BASE_WEIGHT_SCALE / weight_scale))
```

verified with the sigmoid-inverse readout (`build_lut_mixed_gain_probe.py` plus a
diagonal sweep): `H` came out `0.5, 1, 2, 4` exactly where the rule predicts, and
the gain is **uniform across output channels** and independent of the bias and of
the declared output scale (`board_evidence.json`).

| Stem | `BASE/scale` | predicted `H` | measured `H` |
| --- | --- | --- | --- |
| diagonal 1/32 | 1.0 | 1 | 1.000 |
| diagonal 1/16 | 0.5 | 0.5 | 0.500 |
| diagonal 1/64 | 2.0 | 2 | 2.000 |
| diagonal 0.02 | 1.5625 | 2 | 1.995 |
| mixed `[0.02,0.005]` | 1.5625 | 2 | 2.001 |
| mixed `[0.02,0.01]` | 1.5625 | 2 | 1.997 |
| mixed triple 0.01 | 3.125 | 4 | 3.985 |
| mixed `[0.03,0.01]` | 1.0417 | 2 | 1.999 |
| mixed signs `[0.02,-0.005]` | 1.25 | 2 | 1.999 |

The emitter samples the negative half with the inverse gain, which is why the
accepted profiles compose to `fn(x)`.

## Residual blocker (why mixed / non-power-of-two stems stay rejected)

When `BASE/scale` is **not** a power of two, compensating the gain is not enough:
the board output differs from the reference by **±1** on roughly 10% of values —
`321/3072` for diagonal 0.02 (Sigmoid), `394/3072` (Tanh), `318/3072` for 0.024,
`334/3072` for 1/48, `302/3072` for 1/24, and `372/3072` for the mixed
`[0.02,0.005]` stem. The gain is right (no gross errors any more), but the
hardware's argument **grid** rounds differently:

* rounding the code first and then shifting (`(rint(64x) >> k)`) gives 490/3072
  mismatches;
* shifting the wide requantized code (`r >> (5-m)`) gives 2864/3072 mismatches
  under the current reconstruction of `r`;
* the accepted reference (round `64*H*x`) gives 372/3072.

So the exact rounding step between the accumulator and the table index is still
unmodelled. Until it is, `compile_lut` requires
`BASE_WEIGHT_SCALE/weight_scale` to be a power of two (the originally verified
band, plus its powers) and rejects everything else with that message. The accepted
suite is `../lut_domain_suite/` (12 models, 192 board inferences, 36,864 bytes).

## Reproduction

```sh
PYTHONPATH=src python research/build_lut_mixed_gain_probe.py
# push research/lut_mixed_gain_probe/ to the board and run each model with
# tests/board_run.c over its ramp input, then invert the sigmoid on the output
```

`board_evidence.json` records the per-stem prediction, measurement and mismatch
counts; the probe emitter path is opt-in (`allow_mixed_stems=True`,
`negative_gain_override=...`) and is never used by the public scheduler.
