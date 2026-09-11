# Generalized LUT stems (accepted) with a retained mixed-weight blocker

`compile --sequence` accepts `Conv(1x1) -> Sigmoid|Tanh` for a **diagonal** C3 1x1
stem with one scalar gain `g` for every channel (per-channel signs allowed) and
any bias, provided the analytic output range fits the sign-split table domain.
`g` must scale that domain by a power of two.

**Verified on RV1103: 12 models, 192 inferences, 36,864 exact output bytes**
(`board_results_0.json`, six Sigmoid and six Tanh models, 16 inputs each).

## The domain model

The 1026-entry table (two banks of 513) is sampled as `fn` over the argument
interval `(-8, 8)` in steps of `1/64`. The board showed that the hardware reaches
the table differently on each side of zero:

* the **positive** half advances one entry per `1/64` of the dequantized stem
  value `x`, so `index = 512 + round(64x)`;
* the **negative** half advances one entry per `1/64` of
  `x * (BASE_WEIGHT_SCALE / weight_scale)`, where
  `weight_scale = |g|/255` and `BASE_WEIGHT_SCALE = (1/32)/255` is the scale of
  the original identity probe.

The emitter therefore samples the negative half with the inverse gain
`negative_gain = weight_scale/BASE_WEIGHT_SCALE`, so both halves compose to
`fn(x)`. `lut_reference` reproduces the quantized bytes exactly, including the
1/64 index grid, and it reproduces the previously verified identity/32 expected
bytes byte-for-byte.

Measured gains (board, ramp input, channel-wise inverse sigmoid of the output):

| Stem gain `g` | negative-half hardware gain | table correction |
| --- | --- | --- |
| 1/32 | 1 | 1 |
| 1/16 | 0.5 | 2 |
| 1/64 | 2 | 0.5 |

Declared output scale also moves the positive half only: with `g = 1/32`,
declaring `1/1024` halved the positive argument and `1/4096` doubled it, while the
negative half stayed at `1`. The profile therefore keeps the declared scale fixed
at `1/2048` (`research/check_lut_domain.py`, `research/lut_domain_probe/`).

## Accepted bounds

* `w = diag(g, ±g, ±g)` exactly (no off-diagonal weights), `g ≠ 0`.
* `g` such that `32|g|` is a power of two: `|g| ∈ {1/32, 1/16, 1/64, 1/128, ...}`.
  Other gains would only be approximate, so they are rejected.
* Analytic range inside `[-8·32|g|, 8]` (the negative half is wider for small
  `|g|` and the positive half is fixed); larger ranges raise a clear error.
* Fixed RGB 8x8, input scale 1 / zero point 128, output grids 255 (Sigmoid) and
  127 (Tanh) with zero points -128 and 0.

## Refined blocker: the gain is a whole-bit shift

The board gain is `H = 2**ceil(log2(BASE_WEIGHT_SCALE/weight_scale))` — the
accelerator shifts by whole bits (`../lut_mixed_gain_probe/`). For bands where the
ratio is not a power of two, compensating the gain still leaves **±1 on ~10% of
values** because the hardware's argument grid rounds differently from every
candidate model tried, so those stems remain rejected:
`BASE_WEIGHT_SCALE/weight_scale` must be a power of two. The residual was later
read directly off the table index with a sign-aware output-code ramp
([`../lut_index_probe/`](../lut_index_probe/README.md)): the measured index is
linearly right (slope within 0.15% of `64*g` / `64*H*g`) but off by one entry on
~17% of codes, and no affine fixed-point rule in the input code reproduces it.

## Retained blocker: mixed input weights

A channel that mixes several input channels shares one gain band here (the probe
shows a uniform gain), but its argument grid still rounds differently, so the same
±1 residual applies. The failing case is retained in
[`../lut_domain_failed/`](../lut_domain_failed/): a diagonal-free stem
`[[0.02,0.005,0],[0.005,0.02,0],[0,0.005,0.02]]` with bias `[0.8,-0.8,0]` whose
channels share one `(max-min)` but whose measured negative-half gain is `2.016`
against the `1.5625` the weight-scale rule predicts. The board output then misses
by up to 10 codes on a 255-level grid (`run 0 output 0: got -117 expected -107`).

## Reproduction

```sh
PYTHONPATH=src python research/build_lut_domain_suite.py
PYTHONPATH=src python research/run_profile_suite.py lut_domain_suite
# cross-compile tests/board_api.c with runtime/open_rknpu.c first; the runner
# streams each model through /userdata/open-npu-research/board_api_test
```

`manifest.json` records each stem's weights, bias, analytic range and output
quantization. `tests/test_lut_domains.py` checks the table sampling, the family
bounds and the rejections.
