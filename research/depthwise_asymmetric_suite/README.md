# Asymmetric depthwise weight zero points

Resolves the retained blocker: the depthwise weight pair's second byte is the
**negated per-channel weight zero point**, stored as a raw byte. The earlier
hypotheses that used register `0x4054` failed because the field lives in the
weight pair, not in that register.

`compile --sequence --asymmetric-depthwise` now uses asymmetric per-channel
weight quantization and stores `(offset code, (-zp) & 0xff)` per input channel
per tap. For `zp = -128` the negated byte is `0x80`, which is why the symmetric
pair (centered weight, 0) and the asymmetric one coincide at that point.

**Verified on RV1103: 27 models, 432 inferences, 73,728 exact output bytes**
(`board_api_test depthwise_asymmetric_suite 27`), covering channels C1/C3/C4,
kernels K1/K3/K5 and positive-only, negative-only and mixed weights with weight
zero points `-128`, `127` and per-channel mixed values. Expected bytes come from
`depthwise_reference`, which subtracts the per-channel weight zero point.

## Reproduction

```sh
PYTHONPATH=src python research/build_depthwise_asymmetric_suite.py
# cross-compile tests/board_api.c with runtime/open_rknpu.c, push the suite,
# then run: board_api_test depthwise_asymmetric_suite 27
```

`tests/test_depthwise_asymmetric.py` checks the flag produces nonzero zero points
while the default symmetric path stays at zero. The symmetric default is
unchanged, so the earlier depthwise suites remain valid.
