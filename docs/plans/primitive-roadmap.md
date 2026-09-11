# RV1103 primitive roadmap

## Latest status — 2026-09-09

The active work is the expanded Conv/Mul inventory, following the completed MNIST
baseline. [Current results and remaining tasks](../../research/COVERAGE_EXPANSION_RESULTS.md)
record the newly verified native input, geometry, grouped/dilated rewrites,
depthwise, transposed Conv, Mul and activation profiles. Earlier limits and
failed hypotheses below are historical; all ten categories are not complete.
[The completion plan](completion-plan.md) sequences the remaining work; the
unchecked boxes below are the short form of it.


Work in this order. Completion means an explicitly bounded compiler profile,
independently generated commands, integer-reference board checks, and documented
limits. Capture-only evidence does not establish independent support. Extend each
profile incrementally; hardware limits remain unknown until tested. Quantization
accuracy tuning stays deferred. The ordered execution plan for everything still
open is [completion-plan](completion-plan.md); this list is the historical
milestone record.

- [x] Initial same-shape Add and Mul (8x8/C3, two Conv branches).
- [x] Sub — 12 models, 384 board runs, 73,728 exact bytes.
- [x] Max — 12 models, 384 board runs, 73,728 exact bytes.
- [x] Dense Conv stride 2 — initial 8x8/C3, kernels 1/3/5; later strides 3/4 and broader geometry verified.
- [x] Depthwise stride 2 — initial 8x8/C3 k3 pad1; later C1..16, K1/3/5 stride1/2 verified.
- [x] Depthwise wider shapes/channels, other kernels and channel multipliers — H/W5..8, C1..16, K1/3/5, stride1/2; multipliers2..4 and even/rectangular kernels lower through dense rewrites. Dedicated native multiplier and asymmetric-zero-point modes remain unverified.
- [x] Dense Conv wider shapes/channels, arbitrary padding, dilation, grouped Conv — input C1..128, output C1..128, H/W1..128, odd K1..31, stride1..4, dilation≤17, explicit/auto padding, grouped rewrites through group32.
- [x] Mul wider shapes/channels and independent external inputs — two external inputs, same-input, reversed operands, Conv-intermediate × external, C1..16, H/W1..32, batch N1..16.
- [x] Mul scalar/channel constants and broadcasting — scalar, per-channel, spatial, full, per-row/column and per-batch immutable constants. Complementary singleton axes with two runtime operands still need the DAG ABI (plan P1).
- [x] Mul independent input scales/zero points — independent scales and INT8 boundary zero points verified (correctness, not accuracy tuning).
- [x] Independently generated transposed Conv — depthwise/dense/grouped C1..16, K1/K2/K3, per-axis stride1/2. Direct K5 and general dilation remain rejected pending phase recovery (plan P3).
- [x] Fused activations beyond ReLU — Clip[0,6]/ReLU6 and scalar LeakyReLU / scalar-per-channel PReLU within explicit bounds.
- [x] LUT activations (Sigmoid/Tanh first) — diagonal C3 1x1 stems with one scalar gain per channel, per-channel signs, any bias and power-of-two domain gains (12 models, 192 board inferences, `research/lut_domain_suite/`); mixed-input stems stay blocked with a retained failure (`research/lut_domain_failed/`), and domain calibration/interpolation fields remain open (plan P6).
- [ ] Layout operations and graph integration — terminal spatial Reshape verified; two-head fan-out (two outputs), diamond fan-out/fan-in, a variable fan-out to 3..5 heads folded by mixed joins (12 models, 384 board inferences, `research/join_chain_suite/`) a dense+depthwise branch join (12 models, 384 board inferences, `research/depthwise_join_suite/`) a pooled-branch join (12 models, 384 board inferences, `research/pool_join_suite/`) and a mixed dense/depthwise fan-out (12 models, 384 board inferences, `research/mixed_head_suite/`) a runtime per-channel scale on a fan-out result with the scale as a second external input (12 models, 384 board inferences, `research/join_scale_suite/`) a runtime residual feature map combined with a fan-out result (12 models, 384 board inferences, `research/join_residual_suite/`) a general join expression where internal grids feed several joins (12 models, 384 board inferences, `research/join_dag_suite/`) multi-layer residual-style branch chains folded by those joins (12 models, 384 board inferences, `research/branch_join_suite/`) depthwise-separable blocks with a depthwise layer inside a chain (12 models, 384 board inferences, `research/depthwise_chain_suite/`) a join expression reduced by a terminal pool (12 models, 384 board inferences, `research/pooled_dag_suite/`) and pooled multi-layer branches whose chains pool before the joins (13 models, 416 board inferences, `research/pooled_branches_suite/`) are board-verified, and depthwise K3 dilation2 lowers to a direct sparse K5 (8 models, 128 board inferences, `research/transpose_k5_dilation_suite/`), plus a generic lifetime/reuse pass (`open_rknpu.liveness`). Channel-changing reshape/transpose/concat and a scheduler that assembles arbitrary
emitter outputs by tensor name remain; unequal runtime input shapes are solved by the
runtime per-channel scale Mul (`research/runtime_scale_suite/`), plan P1.

Baseline evidence: research/add_suite/README.md, research/mul_suite/README.md,
research/depthwise_suite/README.md. Related-hardware hypotheses and caveats:
research/hardware_refs/README.md. Update this checklist and investigation log at
every milestone or blocked hypothesis; keep unimplemented modes unchecked.

## Real-model envelope: why Silero VAD does not fit (2026-09-11)

The question "can we run a small pretrained model such as Silero VAD?" was checked
against the actual ONNX, not against the op list. `silero_vad.onnx` (opset 16, 2.3 MB,
snakers4/silero-vad) is a wrapper `If(sr==16000)` around two subgraphs; the 16 kHz
branch is:

* `Pad` → `Unsqueeze` → **STFT `Conv`** (`kernel_shape [256]`, `strides [128]`,
  weight `258x1x256`) → `Slice/ Pow / Add / Sqrt` magnitude;
* **four encoder `Conv`+`Relu`** layers, 1-D (`kernel_shape [3]`, strides 1/2/2/1),
  channels 129→128→64→64→128 with pad 1;
* `Shape/Gather/Equal/If/Squeeze/Cast` shape plumbing, **two `LSTM`(hidden 128)** decoder
  branches, then `Relu` → **`Conv` 1x1** (128→1) → `Sigmoid` → `ReduceMean`, plus the
  recurrent `stateN` outputs.

Measured against the compiler with `research/probe_conv_envelope.py` (2026-09-11), which
builds each shape as a one-conv ONNX graph and records the accept/reject decision:

| Requirement | Silero VAD | Compiler today |
| --- | --- | --- |
| 1-D convolution (3-D NCHW) | STFT + 4 encoder Convs | rejected: `static NCHW input required` |
| STFT kernel | `[256]`, stride 128 | odd kernels ≤31 |
| Rectangular kernel | `[1,3]` with one-sided pad | rejected: `native padding/stride/dilation unsupported` |
| First encoder channels | 129 | rejected: input C1..128 is the compiler cap (`input C1..128`), so the encoder's first layer is one channel out of range |
| Recurrence | 2× `LSTM` hidden 128 | no RNN primitive, no `MatMul`, no decomposable GEMM profile |
| Elementwise/activation | `Pow`, `Sqrt`, `ReduceMean`, terminal `Sigmoid` | `Pow`/`Sqrt`/`ReduceMean` unsupported; `Sigmoid` only through the bounded LUT profile (C3 1x1 stem, 8x8, scale 1/zero point 128) |
| Control flow / dynamic shapes | `If`, `Shape`, `Gather`, `Equal`, dynamic `Slice` | static graphs only; non-constant `Shape`/`Slice` are out of scope |

One rewrite *is* close: a 1-D conv is a 2-D conv with `H=1`, and `k3x3 C64 H1 W6 stride2`
is accepted (`native16-input`), so a front-end that maps `kernel_shape [k]` to `[1,k]`
would make the encoder's 2-D-shaped layers compilable. It would still not run Silero VAD:
the STFT kernel stays out of range, `C129` stays out of range, the rectangular `k1x3`
layers are rejected, and the LSTM has no primitive.

So every layer of the model hits at least one independent limit: this is not a small
step but a new primitive family (1-D/rectangular conv), a new datatype path (C>128
beyond the native profile) and a new execution model (recurrence with state). The
encoder alone is also not worth offloading: at 32 ms per frame it is 3 time steps with
≤129 channels, roughly 150k MACs, which the Cortex-A7 does faster than an ioctl round
trip, and the NPU would still need the CPU for the STFT and the LSTM.

The smallest *useful* audio milestone on this stack is the opposite direction: a
purpose-built mel/spectrogram CNN that only uses verified primitives (Conv/Relu/Pool,
1x1 head, no MatMul, no recurrence), which the existing calibration and board flows can
take end to end. That is now done: [examples/mel-kws](../../examples/mel-kws/README.md)
trains a 4,090-parameter mel-CNN on FSDD and runs it entirely on the NPU - **98.00%**
INT8 test accuracy, 191,968 / 192,000 board-exact bytes, 1.86 ms per utterance - after
the walk gained a native16 image input for `3x32x32` and calibrated Conv bands
(`research/mel_kws_suite/`). The 2026-09-11 board-function audit that accompanied this
check is in [research/action_probe/README.md](../../research/action_probe/README.md).

## Historical issue log (2026-09-08)

Topic 1: 8x8/C3 depthwise kernels 1 and 5 passed initial independent board tests
(16 inputs each). 6x6/C3 k3 mismatched at run2/output54 (-22 vs -21).
Broader shape support is not enabled. Evidence: research/depthwise_expansion_suite.
Channel expansion and multipliers still need layout/packing hypotheses; not claimed
as hardware limitations. Public 8x8/C3 stride1/2 behavior remains the baseline.
Both the channel-packing and 6x6 geometry issues were subsequently resolved; see
the 2026-09-09 ledger.

## Historical: Topics 1–7 first pass (2026-09-08)

See [verified additions, exact failures and pending work](../../research/MODE_EXPANSION_STATUS.md).
No topic is marked fully complete while its requested modes remain pending.
New public support: depthwise C3 k1/k5 and C4 k3 (bounded profiles), algebraic
RGB group3/dilation2 lowering, constant Mul folding, spatial-only terminal Reshape.
Superseded next step: depthwise shape/channel packing was resolved by the
C5–C16 experiment below.

Expanded checklist: [Conv/Mul coverage inventory](coverage-matrix.md).

## 2026-09-09: depthwise C5–C16 packing resolved

The C5 vendor capture matched our configuration except addresses/quantization.
Bias/scale groups are packed as four INT32 biases plus four UINT16 scales:
24 bytes per four channels, not the 32-byte dense-Conv layout. Correcting this
independently generated packing passed every channel count 5..16: 12 graphs,
192 board inferences, 129,024 exact bytes. Public compiler binaries match all
12 tested programs; the host suite had 46 tests at this milestone and now has 63.

New bounded public profile: external RGB 8x8 -> 1x1 Conv stem with C5..16 outputs
-> depthwise 3x3, pad1, stride1, multiplier1. Other kernels/strides/stem kernels
at these channel counts remain rejected or unverified. This resolves the earlier
C5 hypothesis failure and, with it, the 6x6 spatial-layout mismatch; it does not
resolve all depthwise modes.

Evidence: research/depthwise_c5_suite through depthwise_c16_suite, each containing
board_results_0.json; research/capture_depthwise_c5 and depthwise_c5_capture.log.
Generator: research/probe_depthwise_channels.py CHANNELS; board runner:
research/run_profile_suite.py depthwise_cCHANNELS_suite (open Python/PYTHONPATH=src).
The generator uses independently compiled stem weights and commands, not captures.
