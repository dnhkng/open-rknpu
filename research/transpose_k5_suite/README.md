# Depthwise K5 ConvTranspose (accepted)

Bounded public K5 transpose profile, emitted from the layout recovered from the
vendor capture `../capture_transpose_k5`:

```
input 8x8x3 -> Conv[/Relu] stem (C1..16) -> ConvTranspose(group=C, K5, stride1/2,
pads < K, output_padding < stride) -> output
```

**Verified on RV1103: 9 models, 144 inferences, 109,872 exact output bytes**
(`board_results_0.json`; 16 inputs per model, including all-zero, all-255,
constant-128 and random data).

| Model | Geometry | Output |
| --- | --- | --- |
| 000 | stride 2, pads 2/2/2/2 (vendor geometry) | 15x15x3 |
| 001 | stride 1, pads 2 | 8x8x3 |
| 002 | stride 2, pads 1 | 17x17x3 |
| 003 | stride 2, pads 0 | 19x19x3 |
| 004 | stride 2, pads 2, output_padding 1 | 16x16x3 |
| 005 | stride [1,2], pads [4,0,1,3], output_padding [0,1] | 7x17x3 |
| 006 | stride 2, pads 2, C1 | 15x15x1 |
| 007 | stride 2, pads 2, C4 | 15x15x4 |
| 008 | stride 2, pads 2, C8 | 15x15x8 |

## Emission

* **Weight table**: 25 taps x 32 bytes. Each tap is 16 lanes of
  `(value int8, -weight_zero_point int8)`; the emitter stores the flipped kernel
  and asymmetric per-channel zero points (`native_quantize(..., symmetric=False)`).
* **Phase field `0x1068`**: `((k-1-pads_left) << 8) | (k-1-pads_top)`. This was
  measured on the board for the pads=1 geometry (candidates `0x101` -> 1303
  mismatches, `0x202` -> 1773, `0x303` -> **0**), and it matches every vendor
  capture: K3/stride1/pads1 `0x101`, K3/stride2/pads1 `0x101`, K3/dilation2
  (effective 5) `0x202`, K5/stride2/pads2 `0x202`, and the unequal-stride
  capture's `0x201` (x byte 2 with left pad 0, y byte 1 with top pad 1).
* **Register set**: the rest of the K5 task matches the vendor capture, including
  the explicitly written output conversion (`0x4080/0x4084/0x4088`) and the
  `0x1010 = 0x3ff` feature field.
* Expected bytes come from `transposed_reference`-style scatter arithmetic:
  centered stem output, centered weights, `bias + stem_zp * sum(centered weights)`
  and the verified two-stage requantization.

## Stale artifacts

`stale/` keeps the pre-investigation experiment: its binary populates only 17 of
25 taps with a reversed kernel and symmetric zero points, and its board output is
uncorrelated with its own expected bytes (0.006) and the float model (0.0017).
`stale/failure.json` records that diagnosis. `../analyze_k5_vendor_layout.py`
prints the vendor-vs-stale register diff.

## Reproduction

```sh
PYTHONPATH=src python research/build_transpose_k5_suite.py
PYTHONPATH=src python research/run_profile_suite.py transpose_k5_suite
# cross-compile tests/board_api.c with runtime/open_rknpu.c first
```

`tests/test_transpose_k5.py` checks the layout, the phase field, the reference
against the float model and the retained stale artifacts.

## Scope

Dilation is still lowered through the verified sparse-K3 rewrite; a direct
dilated-K5 emission (effective kernel 5) is now unblocked but not yet claimed.
