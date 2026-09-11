# Dense K3 dilation2 ConvTranspose: sparse-K5 rewrite (P3 residual)

A ConvTranspose with a dilated K3 kernel spans exactly the five taps of a K5 kernel with
zeros between them, so `open_rknpu.transposed` zero-stuffs `(C_in, C_out, 3, 3)` weights
to `(C_in, C_out, 5, 5)` and emits the dense-K5 form. The rewrite always existed but
only accepted depthwise weights, and the dense emitter understood K3 alone; both are now
general (the dense emitter derives the kernel size from the weights, and the dispatch
lets dilated nodes reach the rewrite before the dense path).

Two host checks guard the rewrite, in the builder:

* the original dilated graph and its zero-stuffed sibling produce the same ONNX float
  output (`allclose`, tolerance 1e-4: the zero taps only change the summation order);
* the integer reference is computed twice - scattering the original taps at data offsets
  `(2ky, 2kx)` and scattering all 25 stuffed taps - and the two must agree byte for byte.

Configurations (Conv 1x1 stem, 8x8 C3 input):

| # | C_in -> C_out | strides | pads | output | output bytes |
| --- | --- | --- | --- | --- | --- |
| 0 | 4 -> 3 | 1x1 | 0 | 12x12 | 432 |
| 1 | 4 -> 3 | 1x1 | 1 | 10x10 | 300 |
| 2 | 8 -> 5 | 2x2 | 1 | 17x17 | 1,445 |
| 3 | 3 -> 3 | 1x1 | 2 | 8x8 | 192 |

## Board evidence

`tests/board_api.c` via `research/run_profile_suite.py`, 8 cases per model:
**4 models, 32 inferences, 18,952 exact output bytes** (`board_results_0.json`,
`board_summary.txt`). All four report the rewrite
(`K3 dilation2 expanded to sparse K5`) and the dense-K5 profile.

The existing suites are unchanged by the generalization: `transpose_dense_suite` (8
models), `transpose_dilation_suite` (K2 dilation2, 4) and `transpose_k5_suite` (9) all
recompile byte-identically, which `tests/test_transpose_dilation.py` pins.

Reproduction:

```sh
PYTHONPATH=src python3 research/build_transpose_dilation_dense_suite.py
PYTHONPATH=src python3 research/run_profile_suite.py transpose_dilation_dense_suite
```
