# K3-dilation2 ConvTranspose — direct sparse-K5 emission

A depthwise `ConvTranspose` with `kernel_shape=[3,3]` and `dilations=[2,2]` has a
support of five taps per axis, so it is the *same operator* as a dense K5
`ConvTranspose` whose odd kernel positions are zero. `open_rknpu.transposed` now
emits that directly through the verified depthwise K5 branch instead of leaving the
mode rejected, which is the last item under P3's "general dilation" heading.

**Verified on RV1103: 8 models, 128 inferences, 82,432 exact output bytes**
(`board_results_0.json`).

| Models | Channels | Strides | Resolved pads | Output padding | Output |
| --- | --- | --- | --- | --- | --- |
| 000 | 3 | 1,1 | 0,0,0,0 | 0,0 | 12×12 |
| 001 | 3 | 1,1 | 1,1,1,1 | 0,0 | 10×10 |
| 002 | 3 | 2,2 | 0,0,0,0 | 0,0 | 19×19 |
| 003 | 3 | 2,2 | 1,1,1,1 | 1,0 | 18×17 |
| 004 | 1 | 1,1 | 2,2,2,2 | 0,0 | 8×8 |
| 005 | 4 | 2,2 | 2,2,2,2 | 1,0 | 16×15 |
| 006 | 3 | 1,2 | 1,0,0,1 | 0,1 | 11×19 |
| 007 | 3 | 2,2 | SAME_UPPER → 2,2,2,2 | 1,1 | 16×16 |

Each model runs 16 input cases including all-zero, all-255 and 128.

## Emission

The rewrite expands the `(C,1,3,3)` kernel into `(C,1,5,5)` with
`expanded[:,:,::2,::2] = raw` and lowers `kernel_shape` to `[5,5]` with
`dilations=[1,1]`, keeping the pads, strides and `output_padding` unchanged. The
output geometry is identical because `(3-1)·2+1 = 5`. The resulting task is the
verified [depthwise K5 profile](../transpose_k5_suite/README.md), including its
asymmetric per-channel weight zero points and the vendor phase field
`0x1068 = ((k-1-pads_x)<<8) | (k-1-pads_y)` with `k=5`.

`tests/test_transpose_k5_dilation.py` proves the algebra independently: for three
stride/pad/output-padding combinations it evaluates the dilated K3 and the
zero-filled K5 scatter in NumPy and asserts they agree exactly. Expected board bytes
come from an explicit integer loop over the quantized expanded weights, centered by
the asymmetric weight zero points, matching the retained K2-dilation2 suite's
reference.

## Remaining dilation blockers

* **Dense** K3-dilation2 (`group=1`, weights `(C_in,C_out,3,3)`) is rejected with a
  specific message: it needs a dense K5 ConvTranspose field set, which no verified
  capture covers. Dense K2-dilation2 already lowers through the verified dense K3
  path.
* Per-axis dilation (`[2,1]` / `[1,2]`) is rejected: the effective kernel would be
  rectangular and the K5 tap table is square.

## Reproduction

```sh
PYTHONPATH=src python research/build_transpose_k5_dilation_suite.py
PYTHONPATH=src python research/run_profile_suite.py transpose_k5_dilation_suite
```

These are legacy v3 containers, so the board evidence comes from
`research/run_profile_suite.py` streaming one model at a time through
`board_api_test`, not from the v5 file runner. The containers here are legacy
`ORNPUSEQ` v1-v4 profiles, so `run_v5_suite.py` rejects them with "model 0 is not a
named-tensor executable"; always use the profile runner for this suite.
