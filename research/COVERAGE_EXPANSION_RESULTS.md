# Conv/Mul expansion results — 2026-09-09

This supersedes the earlier first-pass bounds and failed hypotheses. It does
**not** complete all ten categories in docs/plans/coverage-matrix.md. Public support means
independently generated commands executed by the open C runtime. Quantization
accuracy tuning remains deferred; exact integer agreement is a separate claim.

## Public compiler additions

| Area | Implemented and board-verified bounds | Remaining work |
|---|---|---|
| Dense Conv | Native16 input packing, input C1..128/output C1..128; static batch1..16; H/W 1..128; odd K1..31; optional ReLU; strides 1..4; explicit padding 0..K-1; VALID/SAME_UPPER/SAME_LOWER. Inputs above6144 pixel-planes are split into aligned serial height tasks with overlap halos. | Unaligned tile geometries remain. K33 and native dilation18+ failed and are rejected. Runtime dimensions are immutable, so dynamic ONNX shapes require recompilation. |
| Kernel rewrites | Even/rectangular constant kernels through5x5 embed into odd square kernels. Grouped kernels through group32/inputC32/outputC128 become zero-filled dense kernels. Effective dilation≤5 may expand into weights; larger dilation uses native registers through17. | Runtime weights and dedicated depthwise multiplier modes. Vendor group4 also chose dense expansion, so a separate native-group path is not required for ONNX semantics. |
| Depthwise | RGB stem → multiplier1 depthwise, H/W5..8, C1..16, K1/3/5, stride1/2, with optional bias. A dense pointwise successor is verified from depthwise C3/5/9/16 to output C1/7/16/32. Direct-input depthwise uses the dense rewrite through C32→C128. | Wider chained spatial profiles, dedicated native dilation/asymmetric weight zero points and arbitrary chained padding. |
| Arithmetic | Dense outputC1, optional bias, all-zero/bias-only and individually zero output channels; asymmetric dense and symmetric depthwise weights. Independent output conversion covers all public profiles. V4 permits complete packed native-Conv parameter replacement. Accumulators pass through about +/-2.076 billion. | Channel-tiled accumulation and unverified precisions. |
| Quantized import | Constant-parameter QLinearConv and input-DQ/weight-DQ/Conv/output-Q graphs preserve the supplied INT8 weight bytes: UINT8 input, INT8 output, scalar activation parameters, scalar/per-output weight parameters, optional bias; C1..32→C1..64 and K1/3/5 sampled. | Dynamic quantization tensors and longer quantized graphs remain. Q/DQ float bias is rounded once to INT32; QLinearConv INT32 bias remains exact. |
| Transposed Conv | Conv[/ReLU] stem → learned multiplier1 depthwise C1..16 with square K1/K2/K3/K5, small rectangular kernels, per-axis stride1/2 and padding/output-shape modes; K5 verified across stride1/2, pads 0/1/2, output_padding, rectangular per-axis geometry and C1/C4/C8 (9 models, 144 inferences). Dense C1..16→C1..16 supports K3, per-axis stride1/2, asymmetric pads and legal output padding; group2/3/4/8, reduction and depthwise multiplier2 lower to zero-filled dense weights. K1/rectangular and K2-dilation2 use sparse K3 rewrites. | General dilation (direct effective-K5), broader grouped geometry and arbitrary predecessor geometry. |
| Mul sources | Two external matching inputs, the same external input twice, reversed operands, two learned 1x1 Conv intermediates, one Conv intermediate plus external RGB, and Conv->Relu branch operands. Same-input N1..16 and two-input N1..8 emit serial task groups. Terminal Relu fuses into Mul; terminal spatial Reshape propagates layout. | Multi-task pool/depthwise branches, other successors, fan-out and general DAG scheduling. |
| Mul broadcast | Standalone RGB H/W5..8 × NumPy-compatible constant, plus scalar or per-channel [N,C,1,1] constants for N2..16/C1..16/H/W1..32. Batch constants stay compressed. V4 permits packed factor updates at fixed scale. Spatial constants are materialized into a native plane; the ERDMA reads its secondary operand strictly linearly (`mul_broadcast_notch_suite/`), so there is no compact spatial mode. | Two expanding runtime operands. |
| Mul quantization | Independent branch scales and INT8 boundary zero points; full centered product; independently selectable representable output scale/zero point; offset-before-ties-even conversion and clipping. | Per-channel quantization and other product/output formats. External UINT8 zero points are handled by Conv conversions. |
| Fused activation | Conv→LeakyReLU and Conv→PReLU encode scalar/per-channel slopes0..1 as Q14 with a conservative INT32 positive-path overflow guard. PReLU is verified at C3/C5/C8/C16. Conv→Clip[0,6]/ReLU6 uses matched CNA/DPU upper thresholds and supports the native geometry/channel path with independent output conversion. | Arbitrary fusion orders, profiles that exceed the guard, and broader Leaky/PReLU spatial/stride combinations. |
| Residual Add | RGB pointwise Conv(input)+input lowers to learned and identity Conv branches followed by the verified equal-scale Add task. | General channels, spatial Conv, projection residuals, fan-out and longer residual blocks. |
| LUT activation | Public Conv→Sigmoid/Tanh sequence for a diagonal C3 1x1 stem with one scalar gain per channel (per-channel signs and any bias), gains whose table-domain scale is a power of two, and the analytic range inside the sign-split domain. Python/C accept the1106-word setup and use submission flags1. All256 codes and repeated mixed inputs pass; the generalized family adds12 models/192 inferences. | Mixed input weights (per-channel negative-half gain; retained failure in `lut_domain_failed/`), table interpolation/domain calibration, other shapes/channels and mixed predecessor/successor graphs. |

## Register and numerical findings

- Depthwise bias/scale blocks are24 bytes per four channels, not dense Conv's32.
- Two-stage elementwise DAGs `Mul({Add,Mul,Sub,Max}(a,b), a)` are board-verified
  (12 models, 192 inferences; `elementwise_dag_suite/`), with two external inputs
  and fan-out on the first branch output. `Mul(Mul(a,b),a)` is the first verified
  multi-Mul graph; an earlier 8-model `elementwise_chain_suite` covers `Add`.
- Multi-input elementwise DAGs `Mul(...Mul(Op(a,b),c_k)...,c)` are emitted as
  version-5 executables with **3, 4 and 5 named external inputs**; 18 models and
  144 board inferences passed (`elementwise_multi_suite/`). Programs, inputs and
  buffers are laid out dynamically, so larger graphs cannot collide.
- The DAG emitter is generalised to N stages (`Mul(...Mul(Op(a,b),a)...,a)`);
  2/3/4-stage chains with `Add` or `Mul` first stages passed 12 models and 192
  inferences with up to six NPU tasks (`elementwise_deep_suite/`).
- A leading `Pad` on the graph input is folded into the input shape with the mode
  and amounts recorded (`input_padding`); the caller pads with
  `open_rknpu.padding.pad_input`. Constant/reflect/edge/wrap all passed 12 models
  and 192 board inferences (`padding_suite/`). A native copy/pad producer for
  non-constant borders is not implemented.
- The depthwise weight pair's second byte is the negated per-channel weight zero
  point (as in the dense Conv channel table). Storing `(offset code, (-zp)&0xff)`
  with asymmetric quantization passed 27 models and 432 board inferences across
  C1/C3/C4, K1/K3/K5 and positive/negative/mixed weights. The earlier failure came
  from probing register 0x4054 instead of the weight pair; the symmetric default is
  unchanged.
- Toolkit2 2.3.0 rejects `build(do_quantization=False)` for target RV1103.
  This supports keeping the public ABI INT8-only; it is not proof that every
  floating-point circuit is absent from the silicon.
- Native geometry uses ceil(W/2) at0x103c/0x1044 and rounded pixel surfaces.
  The validated feature-grain formula at0x107c is W*min(4,2*ceil(H/4)).
  Extending the earlier unbounded formula to9x9 caused wrong outputs; the native_h9
  vendor capture suggested the capped value, which independent programs verified.
- Native16 entry count is ceil(W*input-planes/2), which matters for odd widths
  with C17..32. Register0x1048 selects one to three2048-atom input surfaces;
  larger inputs require multiple tasks. Height tiles retain full input/output
  plane strides and use overlapping input rows for convolution halos.
- Odd square kernels K9/11/13/15/17/21/25/31 passed. K33 produced an exact
  mismatch on the first board case, consistent with a five-bit dimension field;
  the public compiler therefore caps this native mode at31.
- Input channels C33..64 are now public and board-verified, and **C65..128 were
  added on 2026-09-11**. The native16 weight layout stores input lanes as 16-wide
  planes grouped in 32-lane parts: within each 16-output block, all taps of the
  first two planes, then all taps of the next two, with a trailing part holding
  the last 16 or 32 lanes. Vendor marker captures with one nonzero weight per
  (tap), (input channel) and (output channel) established the order, including a
  C65/C17 second-block capture; `tests/test_native_c48.py` reproduces each marker
  set exactly. The earlier failure came from a wrong assumption about the plane
  grouping, not from an unrecoverable swizzle. A vendor C128 Conv is a single CNA
  task, so no channel-split accumulation or INT32 partial-sum surface is involved
  (`native_c65_suite/`: 20 models, 320 inferences, 127,744 exact bytes).
- Output channel blocks C65/80/127/128 passed with C1/C17/C32 inputs.
- Static batches2/3/4/16 run as serial tasks sharing constants. Batch and height
  tiling compose (N2,128x128,K3 used six tasks). Sequence metadata and the C API
  expose batch size; old binaries decode as batch1.
- Native atrous fields encode dilation minus one. Unequal2x3/3x2, dilation16/17,
  stride combinations and a five-task K7 tiled case passed. Dilation18/20/22/23/
  24/25/31/32 failed independently generated references, establishing17 as the
  current public maximum despite the related RK3588 field being five bits.
- Grouped zero-fill lowering passed group2/3/4/16/32, per-group output expansion
  through multiplier8, outputC128 and combined unequal dilation. A direct vendor
  group4 capture used dense weight bytes (`16*16*3*3`) and no group control.
- Auto-padding lowering passed VALID and SAME_UPPER/SAME_LOWER, including odd
  total padding and even-kernel rewrites. A separate explicit-pad ONNX oracle
  avoids the installed ReferenceEvaluator's dilated-auto-pad indexing bug.
- RV1103 padding0x1068 is left<<8|top. The swapped-byte hypothesis failed
  asymmetric cases and is retained in the evidence directory.
- Exact Conv final half ties round to even. A32-input discriminating suite separated
  this from half-up and ties-away, fixing a rare downstream Mul mismatch.
- A zero-weight bias-only endpoint established that the output zero point enters
  before the final ties-even shift: 127.5-1 rounds to even126. This formula also
  preserves the older saturated case; the pre-offset-saturation hypothesis failed.
- ConvTranspose zero insertion requires bias compensation for all four taps.
  Learned2x2 weights are flipped on both axes inside the central2x2 of a4x4 table.
- K3 transposed kernels use the same two-axis flip and correctly accumulate
  overlapping stride2 contributions. Independent pads, output_padding, explicit
  output_shape and SAME/VALID were verified in 15 additional models.
- The transpose stride field is `((stride-1)*0x900)|9`: four independently
  generated K2/K3 stride1 profiles passed 64 inferences, and all retained
  stride2 padding/output-shape/overlap/no-bias profiles passed again.
- A fresh `[1,2]` vendor capture isolated `0x1014=0x109`, confirming separate
  Y bits11..13 and X bits8..10. Six public `[1,2]`/`[2,1]` profiles passed
  K2/K3, explicit/SAME padding and legal per-axis output padding.
- Sparse K3 lowering passed K1, K1x3, K2x3, K3x2 and K2 dilation2 while
  preserving output geometry. Direct K5 now passes: the vendor 25-tap lane layout
  with asymmetric per-channel weight zero points plus the phase field
  `0x1068 = ((k-1-pads_left)<<8)|(k-1-pads_top)` gives 9 models / 144 exact board
  inferences (`transpose_k5_suite/`); the earlier probe is retained as stale under
  `transpose_k5_suite/stale/`.
- Multiplier1 depthwise transpose uses the same 32-byte tap lanes and 24-byte
  per-four-channel bias/scale blocks as depthwise Conv. Every C1..16 passed,
  alternating K2/K3 and equal/unequal stride profiles.
- Dense transpose uses 16 input lanes per output, asymmetric per-output weight
  zero points and ordinary dense bias correction. Eight C1..16 input/output
  combinations passed. Grouped weights expand to block-diagonal dense weights;
  six group2/3/4/8, reduction and multiplier2 profiles passed.
- Dense stride1/unequal captures showed that register0x501c remains14 for the
  fixed 8x8 input domain rather than following output height. Eight dense
  channel-tail models now alternate equal/unequal stride, asymmetric padding
  and output padding; all grouped rewrite models were reverified afterward.
- Four independent transpose output conversions passed depthwise and dense
  modes with output zero points -128, 127, -37 and 91.
- Four chained depthwise output conversions passed C4/C5/C8/C16 with output
  zero points -128, 127, -31 and 73.
- Four Conv-intermediate × external-input Mul graphs passed 32 runs with
  independent boundary zero points and output conversion.
- A matched Mul/Mul→Relu capture isolated lower-clamp registers0x407c and0x40e8.
  Four public fused-Relu graphs then passed 32 exact runs.
- Conv→Clip[0,6] requires the same accumulator-domain upper threshold in both
  CNA0x4028 and DPU0x40e4. Four profiles spanning C3/C17/C32, stride1/2,
  K1/K3 and independently wider output ranges passed exactly.
- PReLU uses a UINT16 Q14 slope table at0x502c. Register0x5028 stays literal8
  even above four channels; a fresh C5 capture resolved that field before
  C3/C5/C8/C16 public profiles passed.
- Four RGB Conv(input)+input residual graphs passed H/W5..8 and external zero
  points0/17/128/255 through an explicit identity residual conversion.
- QLinearConv and Q/DQ import now retain source INT8 weight codes and zero points.
  The existing seven-model suites passed all28 board runs after regeneration.
- Omitted chained-depthwise bias synthesizes a zero table; C1/C4/C9/C16,
  K1/K3/K5 and stride1/2 passed32 runs. A following dense pointwise task passed
  C3/C5/C9/C16 to output C1/C7/C16/C32 in another32 runs.
- Legacy Conv-ReLU-Conv final output overrides passed zero points -128/127/-43/79.
- Immutable per-batch scalar/channel Mul constants passed N2/N3/N8/N16. They
  occupy one16-byte native vector per batch rather than a materialized spatial tensor.
- Conv-ReLU branch operands before Mul passed one-sided and two-sided activation
  combinations through C16. Pool/depthwise predecessors remain a scheduler task.
- Four C32/K31 profiles exercised signed accumulators through approximately
  +/-2.076 billion and passed16 exact board runs; out-of-range intervals remain
  rejected before command generation.
- Opt-in sequence v4 adds a named `conv.parameters` region. The ARM C API
  replaced a C3/K1 model's complete packed parameters with a channel-permuted
  donor and produced126/126 expected bytes; wrong sizes/indices were rejected.
- The same v4 API replaced a packed three-channel `mul.factor` vector at runtime
  and produced105/105 expected bytes. Its scale/output conversion stay compiled.
- Four combined Conv->Relu branch -> Mul -> Relu graphs passed32 runs, verifying
  predecessor clamps, centered multiplication, output conversion and final clamp order.
- A direct Mul upper-clamp hypothesis failed. A matched vendor capture instead
  exposed a fourth78-register INT8 clamp task; the independently rebuilt task
  passed Mul->Clip[0,6] at C3/C5/C9/C16 in32 runs.
- Four scalar Add successors exactly representable in the output scale folded
  into Mul's internal output zero point and passed32 runs, including negative offsets.
- Mul output conversion uses DPU output zero point, multiplier and shift registers.
  Five independently generated scale/zero-point profiles passed 80 board inputs.
- A constructed Mul product/2 half with odd output zero point proves the offset
  participates before ties-even rounding. The same parity-sensitive order explains
  the previously unresolved native Conv bias-only endpoint.
- The auxiliary Mul operand uses signed EW offset0x4074. The primary operand uses
  BS constant-source Add (`0x4040=0x120050`, operand at0x4044). Mesa's memory-source
  form timed out in the pure elementwise path and is retained only as failed evidence.
- LeakyReLU retains a raw positive BS product. Its negative path first rounds by
  2^14, then multiplies Q14 alpha; final conversion adds14 shift bits. Large positive
  BS products can saturate INT32, so the compiler rejects the conservative overflow bound.
- LUT setup writes two513-entry Q15 tables and uses1106 register words, shared op
  indices, linked submission and flags1. The public sequence loader validates
  this exact long descriptor. Tanh's reference includes Q15 table rounding before final conversion;
  direct floating-point Tanh rounding disagreed at a discriminating value.
- Format v5 adds a 64-byte named tensor descriptor with per-tensor role, layout
  (packed UINT8, native16, packed INT8), shape, arena offset and size. The runtime
  enumerates it with `ornpu_get_tensor` and binds per-tensor buffers with
  `ornpu_run_io`. Arena storage for packed layouts keeps the 16-aligned CNA row
  stride, so a packed descriptor is larger than its flat API buffer.
- Sequence compilation now accepts measured calibration ranges for profiles with an
  output-range contract: single Conv[/Relu/Clip] through the native and scheduled
  paths, the Conv-ReLU-Conv chain (per-layer), and the elementwise/Mul profiles.
  `calibration.measure` no longer requires the legacy compiler to accept the graph.
  On the `sequence_calibration` suite, calibration lowered output MAE for all three
  graphs but raised maximum error on two (RGB float max 8.76 -> 17.41; native
  12.62 -> 573.70). Min/max calibration is data-dependent; values outside the
  measured ranges clip, so a calibration set can improve or worsen a model.
- Chain intermediate buffers are reused by lifetime (`--reuse-intermediates`):
  layer L writes the buffer layer L-1 last read, halving live intermediates for
  4+ layers (4-layer arena 16 KiB -> 12 KiB) with byte-identical board output
  (5 models, 80 inferences; `chain_reuse_suite/`).
- The N-layer chain can expose every layer output as a named v5 external output
  (`--expose-intermediates`); 4 models, 32 board inferences and 40,448 exact bytes
  passed across 3- and 4-output chains (`chain_multi_suite/`).
- Independently generated N-layer native chains (`[Conv,Relu]*(N-1)+[Conv]`, N=3/4)
  run from one shared arena with explicit intermediate lifetimes: 5 models, 80
  board inferences, 15,360 exact bytes across hidden C3..16 and mixed 1x1/3x3
  kernels (`native_chain_suite/`, `open_rknpu/chain_n.py`).
- Calibrating the trained MNIST Conv2 output range (analytic scale 35.303 to
  measured 0.13411 over 256 held-out images) raised the fully-offloaded both-Conv
  NPU variant from 13/100 to 100/100 on the fixed 100-image held-out set, matching
  the float model, at ~2.45 ms/image with the camera service running. Board logits
  match the independent integer reference within 2e-5. The full 10,000-image test
  set was then streamed to the board: calibrated both-Conv NPU **98.67%** vs float
  **98.90%** and analytic **8.92%** (9,922/10,000 agreement with float), so the
  fully-offloaded network is within 0.23 points of the float model
  (`examples/mnist/README.md`).
- The same pipeline now has a second trained model: `examples/fashion/` trains the
  pinned graph on Fashion-MNIST (6,994 parameters) and streams the full 10,000-image
  test set to the board for all three placements. Float **88.18%**; hybrid
  **88.15%**, both-Conv analytic **88.14%** and both-Conv calibrated **88.15%**, with
  9,979 / 9,862 / 9,957 of 10,000 agreeing with the float model. Eight-case reference
  bytes are exact (25,088 bytes) and logits match the ONNX suffix within 6.7e-6.
  Both-Conv calibrated runs a 1,000-image stream at **1.54 ms/image** end-to-end
  (1.448 ms NPU + 0.089 ms CPU, 536 KiB peak RSS); the hybrid variant is CPU-bound at
  18.0 ms/image. Reports are in `examples/fashion/sanity-results/`
  (`examples/fashion/README.md`).

## Channel-plane expansion

Single Conv now accumulates input C17..64 and produces output C17..64 in one
NPU task. Input/output use separate native16 channel planes; the runtime packs
and unpacks logical NHWC values. Weight storage orders output blocks of16,
kernel taps, output lane, then aligned input channels. The initial unblocked
weight hypothesis failed K3 and was corrected before enabling public support.
Capture_native_in17_k3 additionally confirmed input weight ordering by correlation
with the known float weights. Board tests cover six tail/block counts per
input/output direction and mixed stride/ReLU/large geometry cases.

Additional captures: capture_native_out17, capture_native_out17_k3,
capture_native_in17_k3. Rebuild the C runtime for these executables.

## Reproduction and evidence

Host generators under research/build_*.py compile with the open Python environment;
vendor oracle builders are separate and explicitly named *_oracle.py. Test commands:

```sh
PYTHONPATH=src python research/build_native_combined.py
PYTHONPATH=src python research/run_profile_suite.py native_combined_suite
PYTHONPATH=src python research/run_v5_suite.py runtime_scale_suite
PYTHONPATH=src python -m unittest discover -s tests -q
```

`run_profile_suite.py` streams v3/v4 models one at a time through `board_api_test`;
`run_v5_suite.py` stages a whole v5 directory, runs the file runner once and writes
both evidence files. Each suite retains ONNX, compiled commands, test input bytes,
integer expected outputs, a manifest, and board_results_*.json. The runner stops at the first
failure and now returns a nonzero status; older runs required checking JSON. Old failed
experiments are retained and are not counted as passing models.

Relevant captures: capture_depthwise_h5, capture_depthwise_h6,
capture_depthwise_h5w6, capture_native_h9. They guide hypotheses; public emitters
do not read their contents. Mesa/RK3588 references remain related-hardware evidence,
not proof of RV1103 compatibility.

### Pre-existing container drifts (12 of 169 campaign models)

`research/campaign_sweep.py` recompiles the campaign suites and diffs the result against
their published `.bin` artifacts. It reports **157 same / 12 diff / 0 err**; the 12 are
*artifact* drifts, not compiler regressions - the published container was captured before
the later serializer fixes, so a recompile cannot reproduce its bytes. They are pinned in
the script's `EXPECTED_DRIFT` set, and a change to that set (a new drift or one that
disappears) fails the sweep:

| Suite | Models | Why the artifact differs |
| --- | --- | --- |
| `depthwise_join_suite` | `model009`, `model010` | published before the join tail-control serialization fix |
| `mixed_head_suite` | `model002`, `model005`, `model007`, `model010` | published before the head/tail control and Relu-register fixes |
| `join_scale_suite` | `model002`, `model005`, `model008` | published with the older shared-scale band emission |
| `join_residual_suite` | `model002`, `model005`, `model008` | published with the older shared-scale band emission |

The compiler-side contract is the checked-in `research/container_baseline.json` map,
verified by `research/verify_suites.py` (2,244 suite models; 46 profiles that reject their
model are pinned as `ERR:ValueError`).

## Remaining evidenced blockers

[completion-plan](../docs/plans/completion-plan.md) sequences these into phases P1–P6;
each item below is either a bounded implementation pass or a retained blocker to
close. The blockers are implementation/hardware prerequisites, not silent fallbacks.

1. **Channel planes and accumulation.** External input C1..128 is now solved and
   board-verified as a single CNA task per Conv (the vendor's C128 Conv submits one
   104-byte task over an eight-plane native surface). Remaining: intermediate
   channel planes above 16 in scheduled graphs still need DAG allocation, and input
   above C128 exceeds both the tensor-descriptor bound and the vendor lane field we
   have measured.
2. **Transpose phase arithmetic — solved.** Direct depthwise K5 is board-verified
   (9 models, 144 inferences, 109,872 exact bytes; `transpose_k5_suite/`) with the
   vendor-derived 25-tap lane layout, asymmetric per-channel weight zero points and
   the phase field `0x1068 = ((k-1-pads_left)<<8) | (k-1-pads_top)`, which was
   measured on the board (pads=1: `0x303` gives zero mismatches where `0x101`/`0x202`
   give 1303/1773). The old probe turned out to pack only 17 of 25 taps with a
   reversed kernel; it is retained under `transpose_k5_suite/stale/`. **Depthwise K3
   dilation2 is now emitted directly as a sparse K5** (8 models, 128 inferences,
   82,432 exact bytes; `transpose_k5_dilation_suite/`), which closes the direct
   effective-K5 dilation item for the depthwise family. Remaining: *dense*
   K3-dilation2 needs a dense K5 ConvTranspose field set (rejected with a specific
   message), and per-axis dilation stays rejected because the tap table is square.
3. **General graph scheduling.** The v5 named-tensor table and `ornpu_run_io` now
   support fan-out and multiple external outputs for the bounded two-head profile
   (14 models, 448 inferences, 172,032 exact bytes), fan-in for the diamond
   profile `stem -> {head_a, head_b} -> {Add,Mul,Sub,Max}` (12 models, 384
   inferences, 73,728 exact bytes), emitter composition by tensor name for a
   diamond plus `[Conv, Relu]* Conv` tail (12 models, 192 inferences, 36,864 exact
   bytes), unequal external input shapes for a runtime per-channel scale Mul
   (16 models, 256 inferences, 32,832 exact bytes), a variable fan-out to 3..5
   heads folded by mixed Add/Sub/Max/Mul joins (12 models, 384 inferences, 73,728
   exact bytes), a dense branch and a depthwise branch from one stem joined by
   Add/Sub/Max/Mul (12 models, 384 inferences, 73,728 exact bytes), two Conv+pool
   branches joined after 2x2 stride-2 MaxPool/AveragePool (12 models, 384 inferences,
   18,432 exact bytes), dense and depthwise heads mixed inside one chained fan-out
   (12 models, 384 inferences, 73,728 exact bytes), a depthwise K3-dilation2
   ConvTranspose emitted directly as a sparse K5 (8 models, 128 inferences, 82,432
   exact bytes), a runtime per-channel scale applied to a chained fan-out with the
   scale as a second external input (12 models, 384 inferences, 73,728 exact bytes),
   an external `[1,3,8,8]` residual feature map combined with a fan-out result by
   Add/Sub/Max (12 models, 384 inferences, 73,728 exact bytes), a general join
   expression where internal grids feed more than one join (12 models, 384 inferences,
   73,728 exact bytes), residual-style branches of one to three Conv layers folded by
   those joins (12 models, 384 inferences, 73,728 exact bytes), depthwise-separable
   blocks with a depthwise layer inside a branch chain (12 models, 384 inferences,
   73,728 exact bytes), a join expression reduced by a terminal 2x2 pool (12 models, 384
   inferences, 18,432 exact bytes), pooled multi-layer branches whose chains pool before
   the joins (13 models, 416 inferences, 19,968 exact bytes), and a generic lifetime
   pass (`open_rknpu.liveness`) that orders tasks,
   computes live intervals and reuses arena bytes only where they are disjoint (the
   same pass reproduces the chain ping-pong and refuses in-place reuse). The join
   chain extends that to a variable number of consumers with chained mixed joins, so
   "multiple Mul" is covered, a depthwise branch is joined with a dense branch from
   the same stem by relocating the verified depthwise task program into this
   container's arena, and two pooled branches are joined after 2x2 stride-2 pooling
   using the shared pool register builder. The join chain now also mixes dense and
   depthwise heads in one fan-out, composing two emitter families by tensor name.
   Runtime inputs now reach inside a fan-out: the join chain accepts a final
   `Mul(result, scale)` whose operand is a declared `[1,3,1,1]` graph input, using the
   verified per-channel elementwise program. Two hardware findings are retained for
   that composition: a scaled primary whose slot was reused by an earlier Conv task
   returns stale data (this profile therefore places internals sequentially), and the
   folded scale product is not exactly 1/128 in float32 so the reference must use the
   hardware requantization formula. The join form is now general: any two previously
   produced tensors may feed a join, so a head or join result can have several
   consumers, with one INT8 scale propagated per tensor (Mul folds two free scales,
   Add/Sub/Max require one shared band and re-quantize an uncommitted head operand),
   and a branch may be a chain of one to three Conv layers with per-layer channel
   counts and input bands.
   Still open: a *fully general* pass that dispatches every normalized op to its
   emitter instead of matching whole graph patterns, and auto-derived task bindings
   for every existing emitter. V3
   describes one output and one/two matching inputs; v4 adds constants without
   changing that tensor contract.
4. **Unisolated hardware modes.** Asymmetric depthwise weight zero points and
   per-channel Mul quantization are now solved (weight pair second byte; 1x1
   depthwise lowering with native per-channel weight scales, 12 models / 192
   inferences). Spatial Mul broadcasts are refuted with the fields decoded: the RK3588
   TRM names `0x506c` `ew_surf_notch` and `0x5010` bits 28:16 `ew_line_notch_addr`,
   and setting the notch turns every retained compact-operand hang into a completed
   run whose bytes are exactly the *linear* read of the compact table (one 16-byte
   atom per output pixel, unchanged by stride 16/576, notch 0x40/0x50 or `surf_mode`;
   `data_mode=2` still hangs). `mul_broadcast_notch_suite/` retains the six variants
   and the linear-read proof, so the verified modes stay per-channel and per-pixel
   plane and constants stay materialized. A per-channel *output* conversion was also
   probed through `BS_OW_CFG.OW_SRC=1`: re-run with the block at the address `0x5020`
   names and in the decoded Conv block format it still hangs, while `OW_SRC=1` plus
   `OD_BYPASS=1` completes and ignores the block entirely (output byte-identical to
   the baseline for zeroed channel multipliers), so the EW output stage has no
   BS-table read (`mul_per_channel_ow_suite/`). Toolkit2 rejects unquantized RV1103 builds,
   and no verified FP16/external-INT8/wider-output layout exists.
5. **Separate padding producers.** CNA border padding injects only the activation
   zero point. Reflection, replication and circular modes use the folded host
   preprocessing path (`input_padding` + `open_rknpu.padding.pad_input`, board
   verified); a dedicated NPU copy/pad task is not implemented.
6. **General LUT domains — partially solved.** The sign-split domain is now
   measured: the positive table half advances per 1/64 of the dequantized stem value
   and the negative half per 1/64 of `x·(BASE_WEIGHT_SCALE/weight_scale)`, so a
   diagonal stem with one scalar gain per channel is exact after the table applies
   the inverse gain (12 models, 192 inferences, 36,864 bytes; `lut_domain_suite/`).
   A board sweep then showed the gain is a **whole-bit shift**,
   `H = 2**ceil(log2(BASE_WEIGHT_SCALE/weight_scale))`, uniform across channels and
   independent of bias and declared scale (0.5/1/2/4 measured where predicted).
   Mixed and non-power-of-two bands are still rejected: compensating the gain
   leaves ±1 on ~10% of values because the accumulator-to-index rounding step is
   unmodelled (`lut_mixed_gain_probe/`; 321-394 of 3072 mismatches).
   The residual was then measured *at the table index* with a sign-aware
   output-code ramp (`lut_index_probe/`): the measured index is linearly right
   (slope within 0.15% of `64*g` / `64*H*g`, whole-bit `H` confirmed) but off by
   exactly one entry on ~17% of readable codes, and no affine fixed-point rule
   `floor((A*q+B)/2**J)`, `J <= 14`, reproduces it for any non-power-of-two band
   (`analysis.json`, re-checked by `tests/test_lut_index_probe.py`).
   Interpolation/domain-calibration fields beyond the two banks remain undecoded.


These are implementation/hardware prerequisites rather than silent fallbacks.
The attached board's NPU is the shared `rockchip,rv1106-rknpu` IP, so the passing
ledger is NPU-level RV1106 evidence; no distinct-RV1106-SoC or trained-model
accuracy claim follows from it.

## Board suite totals

Unique passing model indices within each suite; restarted test segments are
deduplicated. These counts exclude exploratory failures and the replay-only LUT,
legacy-C1, halfway-rounding and first transposed probes.

| Suite | Models | Inferences | Exact output bytes |
|---|---:|---:|---:|
| [depthwise_geometry](depthwise_geometry_suite/) | 32 | 512 | 41,856 |
| [depthwise_combined](depthwise_combined_suite/) | 84 | 1,344 | 343,120 |
| [depthwise_small_channels](depthwise_small_channels_suite/) | 12 | 192 | 8,688 |
| [conv_geometry](conv_geometry_suite/) | 48 | 768 | 86,768 |
| [conv_boundary](conv_boundary_suite/) | 48 | 768 | 28,176 |
| [conv_asymmetric](conv_asymmetric_suite/) | 32 | 512 | 62,144 |
| [rectangular_conv](rectangular_conv_suite/) | 12 | 192 | 14,400 |
| [mul_geometry](mul_geometry_suite/) | 32 | 1,024 | 319,488 |
| [add_geometry](add_geometry_suite/) | 32 | 1,024 | 319,488 |
| [sub_geometry](sub_geometry_suite/) | 32 | 1,024 | 319,488 |
| [max_geometry](max_geometry_suite/) | 32 | 1,024 | 319,488 |
| [two_input_mul](two_input_mul_suite/) | 12 | 384 | 131,904 |
| [standalone_mul](standalone_mul_suite/) | 12 | 384 | 44,928 |
| [mul_broadcast](mul_broadcast_suite/) | 12 | 384 | 44,928 |
| [transpose_learned](transpose_learned_suite/) | 6 | 96 | 73,728 |
| [leaky_verified](leaky_verified_suite/) | 23 | 736 | 219,264 |
| [native_input](native_input_suite/) | 18 | 288 | 73,760 |
| [native_combined](native_combined_suite/) | 42 | 672 | 101,904 |
| [group_dilation](group_dilation_suite/) | 52 | 832 | 159,168 |
| [group_dense](group_dense_suite/) | 48 | 768 | 139,200 |
| [conv_stride34](conv_stride34_suite/) | 48 | 768 | 31,184 |
| [native_stride34](native_stride34_suite/) | 42 | 672 | 36,640 |
| [native_large](native_large_suite/) | 44 | 704 | 800,800 |
| [native_large_combined](native_large_combined_suite/) | 44 | 704 | 348,944 |
| [native_mul](native_mul_suite/) | 48 | 384 | 388,920 |
| [native_elementwise](native_elementwise_suite/) | 64 | 512 | 1,085,760 |
| [native_output_blocks](native_output_blocks_suite/) | 24 | 384 | 583,136 |
| [native_input_blocks](native_input_blocks_suite/) | 24 | 384 | 328,544 |
| [native_blocks_combined](native_blocks_combined_suite/) | 24 | 72 | 75,354 |
| [native_large_kernels_combined](native_large_kernels_combined_suite/) | 12 | 48 | 41,640 |
| [native_kernel_limits](native_kernel_limits_suite/) | 8 | 32 | 14,000 |
| [native_spatial_limits](native_spatial_limits_suite/) | 7 | 14 | 50,210 |
| [native_spatial_boundary](native_spatial_boundary_suite/) | 28 | 56 | 78,500 |
| [native_spatial_tiling](native_spatial_tiling_suite/) | 6 | 12 | 185,792 |
| [native_output_limits](native_output_limits_suite/) | 4 | 16 | 42,024 |
| [native_batch](native_batch_suite/) | 5 | 10 | 37,464 |
| [native_dilation_verified](native_dilation_verified_suite/) | 7 | 21 | 65,670 |
| [group_expansion](group_expansion_suite/) | 6 | 18 | 146,076 |
| [auto_padding](auto_padding_suite/) | 7 | 21 | 921 |
| [transpose_padding](transpose_padding_suite/) | 6 | 96 | 67,728 |
| [transpose_output_shape](transpose_output_shape_suite/) | 5 | 80 | 61,344 |
| [transpose_overlap](transpose_overlap_suite/) | 4 | 64 | 43,536 |
| [mul_output_quantization](mul_output_quantization_suite/) | 5 | 80 | 9,264 |
| [mul_sources](mul_sources_suite/) | 5 | 80 | 182,224 |
| [depthwise_rewrite_expansion](depthwise_rewrite_expansion_suite/) | 6 | 18 | 122,235 |
| [weight_edge](weight_edge_suite/) | 4 | 16 | 17,020 |
| [mul_broadcast_output](mul_broadcast_output_suite/) | 4 | 48 | 6,228 |
| [transpose_no_bias](transpose_no_bias_suite/) | 2 | 32 | 21,552 |
| [lut_public](lut_public_suite/) | 2 | 32 | 6,144 |
| [qlinearconv_import](qlinearconv_import_suite/) | 4 | 16 | 32,236 |
| [mul_output_rounding](mul_output_rounding_suite/) | 1 | 8 | 600 |
| [mul_boundary_zero_points](mul_boundary_zero_points_suite/) | 4 | 48 | 7,200 |
| [qdq_conv_import](qdq_conv_import_suite/) | 3 | 12 | 31,792 |
| [mul_batch](mul_batch_suite/) | 3 | 9 | 18,090 |
| [mul_reshape](mul_reshape_suite/) | 3 | 18 | 6,588 |
| [mul_two_input_batch](mul_two_input_batch_suite/) | 3 | 9 | 12,342 |
| [output_channels_complete](output_channels_complete_suite/) | 15 | 30 | 21,130 |
| [border_padding](border_padding_suite/) | 3 | 12 | 1,704 |
| [transpose_stride1](transpose_stride1_suite/) | 4 | 64 | 12,720 |
| [transpose_unequal](transpose_unequal_suite/) | 6 | 96 | 38,064 |
| [transpose_dilation](transpose_dilation_suite/) | 4 | 64 | 29,232 |
| [transpose_rectangular](transpose_rectangular_suite/) | 5 | 80 | 33,408 |
| [transpose_channels](transpose_channels_suite/) | 16 | 64 | 80,256 |
| [transpose_dense](transpose_dense_suite/) | 8 | 32 | 27,920 |
| [transpose_grouped](transpose_grouped_suite/) | 6 | 24 | 44,100 |
| [transpose_output_quantization](transpose_output_quantization_suite/) | 4 | 16 | 9,472 |
| [depthwise_output_quantization](depthwise_output_quantization_suite/) | 4 | 16 | 8,448 |
| [mul_external_intermediate](mul_external_intermediate_suite/) | 4 | 32 | 4,248 |
| [mul_relu](mul_relu_suite/) | 4 | 32 | 4,248 |
| [conv_clip](conv_clip_suite/) | 4 | 16 | 3,040 |
| [prelu_public](prelu_public_suite/) | 4 | 32 | 11,968 |
| [residual_add](residual_add_suite/) | 4 | 32 | 4,632 |
| [depthwise_no_bias](depthwise_no_bias_suite/) | 4 | 32 | 4,064 |
| [chain_output_quantization](chain_output_quantization_suite/) | 4 | 32 | 6,144 |
| [mul_batch_broadcast](mul_batch_broadcast_suite/) | 4 | 32 | 17,808 |
| [depthwise_pointwise](depthwise_pointwise_suite/) | 4 | 32 | 7,280 |
| [mul_after_relu](mul_after_relu_suite/) | 4 | 32 | 10,784 |
| [accumulator_boundary](accumulator_boundary_suite/) | 4 | 16 | 16 |
| [mutable_weights](mutable_weights_suite/) | 1 | 1 | 126 |
| [mutable_mul_factor](mutable_mul_factor_suite/) | 1 | 1 | 105 |
| [mul_relu_order](mul_relu_order_suite/) | 4 | 32 | 10,784 |
| [mul_clip](mul_clip_suite/) | 4 | 32 | 10,784 |
| [mul_add](mul_add_suite/) | 4 | 32 | 10,784 |
| [two_head](two_head_suite/) | 14 | 448 | 172,032 |
| [sequence_calibration](sequence_calibration_suite/) | 6 | 96 | 77,824 |
| [native_c48](native_c48_suite/) | 12 | 192 | 71,328 |
| [native_c64](native_c64_suite/) | 12 | 192 | 71,328 |
| [native_chain](native_chain_suite/) | 5 | 80 | 15,360 |
| [chain_multi](chain_multi_suite/) | 4 | 32 | 40,448 |
| [chain_reuse](chain_reuse_suite/) | 5 | 80 | 15,360 |
| [depthwise_asymmetric](depthwise_asymmetric_suite/) | 27 | 432 | 73,728 |
| [padding](padding_suite/) | 12 | 192 | 66,048 |
| [elementwise_chain](elementwise_chain_suite/) | 8 | 128 | 24,576 |
| [elementwise_dag](elementwise_dag_suite/) | 12 | 192 | 36,864 |
| [elementwise_deep](elementwise_deep_suite/) | 12 | 192 | 36,864 |
| [elementwise_multi](elementwise_multi_suite/) | 18 | 144 | 27,648 |
| [per_channel_mul](per_channel_mul_suite/) | 12 | 192 | 23,232 |
| [diamond](diamond_suite/) | 12 | 384 | 73,728 |
| [lut_domain](lut_domain_suite/) | 12 | 192 | 36,864 |
| [transpose_k5](transpose_k5_suite/) | 9 | 144 | 109,872 |
| [diamond_tail](diamond_tail_suite/) | 12 | 192 | 36,864 |
| [runtime_scale](runtime_scale_suite/) | 16 | 256 | 32,832 |
| [join_chain](join_chain_suite/) | 12 | 384 | 73,728 |
| [depthwise_join](depthwise_join_suite/) | 12 | 384 | 73,728 |
| [pool_join](pool_join_suite/) | 12 | 384 | 18,432 |
| [mixed_head](mixed_head_suite/) | 12 | 384 | 73,728 |
| [transpose_k5_dilation](transpose_k5_dilation_suite/) | 8 | 128 | 82,432 |
| [join_scale](join_scale_suite/) | 12 | 384 | 73,728 |
| [join_residual](join_residual_suite/) | 12 | 384 | 73,728 |
| [join_dag](join_dag_suite/) | 12 | 384 | 73,728 |
| [branch_join](branch_join_suite/) | 12 | 384 | 73,728 |
| [depthwise_chain](depthwise_chain_suite/) | 12 | 384 | 73,728 |
| [pooled_dag](pooled_dag_suite/) | 12 | 384 | 18,432 |
| [pooled_branches](pooled_branches_suite/) | 13 | 416 | 19,968 |
| [deep_chain](deep_chain_suite/) | 3 | 48 | 9,216 |
| [percentile_calibration](percentile_calibration_suite/) | 4 | 76 | 14,592 |
| [transpose_dilation_dense](transpose_dilation_dense_suite/) | 4 | 32 | 18,952 |
| [native_c65](native_c65_suite/) | 20 | 320 | 127,744 |
| [walk_chain](walk_chain_suite/) | 12 | 192 | 6,368 |
| [walk_join](walk_join_suite/) | 6 | 96 | 4,608 |
| [mel_kws](mel_kws_suite/) | 16 | 16 | 10,240 |
| **Total** | **1,696** | **28,266** | **10,216,467** |

## Final validation for this pass

760 host tests pass, including Python/C loader parity, v4 constant descriptors, the v5
named-tensor table and two-head fan-out profile, sequence calibration, channel-plane bounds,
missing native bias, graph normalization equivalence and reproduction of board
programs, large kernels and tiled spatial outputs. The ARM runtime builds with
-Wall -Wextra -Werror. The board API harness now allocates tensors from inspected
sizes rather than imposing its old32-KiB/64-KiB test-only limits. Its file runner
uses dynamically inspected sizes; the C32→C64 smoke model produced3072 exact
output bytes (runtime_block_smoke/). Existing packed
Conv boundary and larger two-input Mul cases passed again with the final runtime.
The camera service remained running. These checks establish bounded integer
execution, not all ten checklist categories or trained-model accuracy.
