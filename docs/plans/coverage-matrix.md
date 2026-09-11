# RV1103 convolution and Mul coverage inventory

Reviewed 2026-09-09. This expands the earlier seven-topic roadmap.

Unchecked entries indicate incomplete coverage, not necessarily total absence.
Record each result as **native verified**, **rewrite verified**, **failed hypothesis**,
**implementation prerequisite**, or **untested**, with exact shapes and evidence.
Passing separate options does not prove their combination. Hardware limits remain
unknown until established; RK3588/Mesa definitions provide hypotheses only.
Quantization representation/correctness is in scope; accuracy tuning remains deferred.

Current baseline and failures: [mode status](../../research/MODE_EXPANSION_STATUS.md).
Current results and next tasks: [September 9 expansion results](../../research/COVERAGE_EXPANSION_RESULTS.md).
Ordered remaining work: [completion plan](completion-plan.md).
The historical work log below is retained; the latest results supersede its old bounds.
This inventory's checkboxes are current as of the 2026-09-09 ledger; several
historical "not complete" phrasings elsewhere are superseded by that ledger.

## 1. Dense convolution geometry

- [x] Square kernels 1/3/5 broadened; native single-Conv H/W 1..128, input C1..128/output C1..128
- [x] Odd kernels through31 native verified; K33 failed independently generated commands and is rejected
- [x] Even kernels 2x2/4x4: zero-filled square-kernel rewrite
- [x] Rectangular kernels through 5x5: zero-filled square-kernel rewrite
- [x] Stride1 coverage across the public dense, depthwise, grouped-rewrite and bounded transpose profiles
- [x] Stride2 beyond 8x8/C3, including native input and odd geometry
- [x] Unequal strides (1,2)/(2,1), bounded Conv profiles
- [x] Strides 3/4, packed and native inputs
- [x] Native dilation registers, encoding dilation-1 in0x1014
- [x] Dilation2 beyond the existing 3x3-to-5x5 rewrite
- [x] Unequal/larger dilation through17; dilation18+ failed and is rejected
- [x] Combined stride and dilation, including tiled K7
- [x] Small spatial dimensions, including 1xN/Nx1, within the native profile
- [x] H/W through128 with automatic aligned height tiling above6144 input pixel-planes
- [x] Rectangular feature maps within documented profile bounds
- [x] Odd dimensions with stride2 within documented bounds
- [x] Static batch1..16 via shared constants and serial per-batch native tasks
- [x] Dynamic-shape contract: executable dimensions are immutable; recompile for each shape

## 2. Padding and output geometry

- [x] VALID spatial convolution with reduced output
- [x] Independent top/bottom/left/right padding, each 0..K-1
- [x] Different horizontal and vertical padding within those bounds
- [x] SAME_UPPER versus SAME_LOWER with odd total padding, including even kernels
- [x] Execute normalized auto-padding; SAME/VALID lowered to explicit pads and board verified
- [x] Explicit padding combined with stride and native/rewrite dilation; even-kernel combination remains below
- [x] Padding values with tested UINT8 activation zero points 0/128/255
- [x] Multiple-edge border cases, including 1x1/1x2/2x1 inputs and K3/K5/K7 windows larger than the source
- [x] Single-pixel outputs in native and strided Conv suites
- [x] Padding and overlapping convolution halos at height-tile boundaries, K3/5/7
- [x] Reflection/replication/circular padding: folded host preprocessing (`input_padding` + `open_rknpu.padding.pad_input`), board-verified 12 models; a native copy/pad producer is not implemented

## 3. Channels, groups and depthwise

- [x] External input channels 1..64 via native16 planes (four 16-lane planes); board-verified suites native_c48 and native_c64
- [ ] General intermediate input channels: C1..16 verified in depthwise-to-pointwise; C17..64 needs the channel-plane DAG allocation (external-input layout is now solved)
- [x] Single dense output channel, including bias-only cases
- [x] Every dense output channel count C2..16 with varied K1/3/5, input C1/3/17/32 and asymmetric padding
- [x] Dense output channels above16: native planes through C128
- [x] Channel tails around 3/4/5,7/8/9,15/16/17 plus every output C2..16
- [x] General grouped Conv: constant dense rewrite through group32, inputC32/outputC128, including unequal dilation
- [x] Native-group investigation: vendor group4 capture itself expands to dense; no separate public native mode is needed for semantics
- [x] Depthwise C3: H/W 5..8, K1/3/5, stride1/2
- [x] Depthwise C4: H/W 5..8, K1/3/5, stride1/2
- [x] Depthwise C1..16, multiplier1; C5 packing failure resolved
- [x] Depthwise H/W 5..8; 6x6 geometry failure resolved
- [x] Depthwise stride2 for C1..16, K1/3/5
- [x] Depthwise K1/K5 with stride2, C1..16
- [x] Direct depthwise dilation through3 sampled, lowered by dense rewrite plus expanded/native dilation
- [x] Direct depthwise rectangular/even K2x3 and K2x4 via square-kernel rewrite
- [x] Direct depthwise channel multipliers2/3/4 through C32→C128 via dense rewrite; dedicated native mode unneeded for semantics
- [x] Bounded Conv -> depthwise -> pointwise composition: depthwise C3/5/9/16, stride1/2, K1/3/5, pointwise output C1/7/16/32

## 4. Weights, bias and convolution arithmetic

- [x] Constant versus runtime-supplied weights: opt-in v4 native Conv exposes one named packed parameter region through the C API; v3 remains immutable
- [x] Weight updates without full recompilation for the bounded native Conv profile via `ornpu_set_constant`; replacement covers weights, bias correction and per-channel conversion together and must retain geometry/output conversion
- [x] Bias absent/present in public dense, chained depthwise, depthwise-to-pointwise and bounded ConvTranspose profiles; omitted bias synthesizes zero tables
- [x] All-zero weight tensors and bias-only Conv in native dense Conv
- [x] Individual zero-weight output channels mixed with active channels
- [x] Symmetric depthwise and asymmetric per-output-channel dense weight quantization
- [x] Per-output-channel weight scales; equal scales cover the per-tensor special case for compiler-generated quantization
- [x] Nonzero native depthwise weight zero points: the weight pair's second byte is `-zp`; asymmetric quantization board-verified on 27 models (`compile --sequence --asymmetric-depthwise`)
- [x] UINT8 external activation storage and INT8 intermediate/output storage
- [x] Caller/calibration-selected chained activation scales and zero points within multiplier/shift range; calibration and explicit final chain overrides board verified
- [x] Output quantization overrides in every public arithmetic profile, including legacy Conv-ReLU-Conv and depthwise-to-pointwise chains
- [x] Accumulator boundaries: ties-even, output-offset ordering, compile-time overflow guards and exact board values through -2,076,148,792..2,076,148,792
- [x] Independently generated N-layer dense Conv chains, `[Conv,Relu]*(N-1)+[Conv]`, N=3/4,
  hidden C3..16, mixed 1x1/3x3, one shared arena with explicit intermediate lifetimes
  (native_chain_suite: 5 models, 80 board inferences)
- [ ] Arena byte reuse for tensors whose lifetime has ended; exposing chain intermediates as outputs
- [x] Single-task accumulation across two to four native input-channel planes, C17..64
- [x] Bias and requantization after single-task C17..64 accumulation; above C64 needs multi-task splitting
- [x] Bit-preserving constant quantized import: bounded QLinearConv and Q/DQ Conv preserve source INT8 weights
- [ ] Dynamic quantization tensors/longer quantized graphs: a runtime per-channel factor is now a typed unequal-shape external input (`research/runtime_scale_suite/`, 16 models / 256 board inferences); longer quantized graphs still need general scheduling
- [x] FP16/other types investigated: Toolkit2 2.3.0 rejects `do_quantization=False` for RV1103; related Rocket fields alone do not establish an RV1103 float profile

## 5. Transposed convolution

- [x] Dense ConvTranspose: chained K3, per-axis stride1/2, independent pads/output_padding, input/output C1..16
- [x] Grouped ConvTranspose: constant zero-filled dense rewrite within C1..16, sampled group2/3/4/8
- [x] Depthwise ConvTranspose: C1..16 fixed-input stem, K2/K3, per-axis stride1/2, learned weights
- [x] Arbitrary learned weights for the bounded depthwise C3 K2/K3 profile
- [x] Bias/no bias in the bounded depthwise C3 profile; absent bias synthesizes a zero hardware table
- [ ] Square/rectangular/odd/even kernels: K1/K2/K3 and K1x3/K2x3/K3x2 verified; direct K5 rejected after a retained 98/675 off-by-one phase failure
- [x] Stride1/2/unequal strides within per-axis stride1/2, including depthwise K3 dilation2 emitted directly as a sparse K5 (8 models, 128 board inferences, `research/transpose_k5_dilation_suite/`)
- [ ] Dilation: K2 dilation2 verified by sparse K3 rewrite; larger effective kernels enter the rejected K5 phase regime
- [x] Independent side padding 0..K-1 in the bounded K2/K3 profile
- [x] output_padding 0/1 on each axis
- [x] Explicit output_shape resolved to legal pads when pads are omitted
- [x] VALID/SAME_UPPER/SAME_LOWER automatic padding and output-size interactions
- [x] Channel expansion/reduction within the bounded dense/grouped C1..16 profile, including depthwise multiplier2
- [x] Weight ordering and two-axis kernel flip for K2/K3
- [x] Overlapping contribution accumulation for K3/stride2
- [x] Quantized zero insertion, borders, bias correction and final rounding in the bounded profile

## 6. Mul operand sources and graph placement

- [x] Two independent external inputs, packed by tensor into one API buffer, including a runtime per-channel scale consumed by a fan-out result (`research/join_scale_suite/`, 12 models / 384 board inferences)
- [x] External plus intermediate input: bounded Conv(input0) × matching RGB input1 through identity conversion, plus a runtime per-channel scale and a runtime `[1,3,8,8]` residual feature map combined with a fan-out result (`research/join_scale_suite/`, `research/join_residual_suite/`, 24 models / 768 board inferences)
- [x] Two learned 1x1 Conv intermediate tensors in the bounded branch profile
- [x] Logical standalone Mul; implementation inserts two identity Conv conversions
- [x] Mul after depthwise and pooling: a shared stem feeds a dense branch and a group-3 depthwise branch, joined by Add/Sub/Max/Mul (12 models, 384 board inferences, `research/depthwise_join_suite/`); two Conv+pool branches (2x2 stride-2 MaxPool/AveragePool) joined after the pool also pass (12 models, 384 board inferences, `research/pool_join_suite/`)
- [x] A*A with the same input buffer, C1/C5/C16 and H/W through32 tested
- [x] Reversed operand order for two external inputs, C3/C16 tested
- [x] Shared operands with multiple consumers: the v5 named-tensor table carries the shared intermediate edge and the diamond profile consumes both consumers in one join (12 models, 384 board inferences, `research/diamond_suite/`); a variable fan-out to 3..5 heads consumes the shared stem once per head (12 models, 384 inferences, `research/join_chain_suite/`)
- [ ] Mul feeding subsequent NPU operations: terminal Relu fusion and spatial Reshape verified; general successors need a serialized named producer edge
- [x] Mul consumed by terminal spatial Reshape with propagated output geometry and unchanged NPU storage
- [x] Multiple Mul nodes in one graph: head grids fold through chained joins with mixed Add/Sub/Max/Mul kinds, ordered and placed by `open_rknpu.liveness` (12 models, 384 board inferences, `research/join_chain_suite/`); the heads themselves may mix the dense and depthwise emitters (12 models, 384 board inferences, `research/mixed_head_suite/`); joins may combine any two previously produced tensors, so a grid can feed several consumers (12 models, 384 board inferences, `research/join_dag_suite/`); a branch may be a one-to-three layer Conv chain with per-layer channel counts (12 models, 384 board inferences, `research/branch_join_suite/`); and a depthwise layer may sit inside such a chain via an exact block-diagonal dense expansion (12 models, 384 board inferences, `research/depthwise_chain_suite/`)
- [x] Standalone immutable broadcast constants, RGB H/W 5..8
- [x] Runtime-changing immutable scalar/channel factors: opt-in v4 `mul.factor` packed region updates at fixed compiled factor scale; full runtime tensor operands use the existing matching-input path
- [x] Bounded standalone shapes/channels/batches: C1..16, H/W1..32, same-input N1..16 and two-input N1..8 sampled

## 7. Mul broadcasting

- [x] Scalar/all-singleton immutable factors, RGB H/W 5..8
- [x] Immutable per-channel [1,C,1,1], RGB H/W 5..8
- [x] Immutable spatial mask [1,1,H,W], RGB H/W 5..8
- [x] Immutable full tensor [1,C,H,W], N=1, RGB H/W 5..8
- [x] Immutable per-batch [N,1,1,1], N2..16, C1..16, H/W1..32
- [x] Immutable per-row [1,1,H,1], RGB H/W 5..8
- [x] Immutable per-column [1,1,1,W], RGB H/W 5..8
- [x] Immutable per-batch/per-channel [N,C,1,1], N2..16, C1..16, H/W1..32
- [x] Complementary singleton axes with two runtime operands: `Mul(image[1,3,H,W], scale[1,3,1,1])` binds both operands as externals, the runtime packing three INT8 codes into the 16-byte operand row (16 models / 256 board inferences, `research/runtime_scale_suite/`)
- [x] Lower-rank immutable operand alignment: scalar, [W] and [H,W] tested by NumPy trailing-axis rules
- [x] Broadcast either operand for immutable constants
- [x] Singleton axes within otherwise matching immutable shapes
- [x] Reject incompatible broadcast shapes at compilation
- [ ] Execute broadcasts without materializing expanded tensors: scalar/channel native mode verified. ERDMA fields are named (`0x5034` DATA_MODE/SURF_MODE, `0x5040` EW_SURF_STRIDE) but compact per-row variants hung the NPU (`research/mul_broadcast_mode_suite/`); no working spatial-source mode

## 8. Mul quantization and numerical behavior

- [x] Different Mul operand scales sA/sB; both boundary zero points remain zero
- [x] Independent nonzero operand zero points zA/zB: BS constant-source ALU centers primary, 0x4074 centers auxiliary
- [x] Independent output scale/zero point for standalone matching-shape Mul; five conversions including nonzero output zero points
- [x] Full centered product sA*sB*(qA-zA)*(qB-zB), including boundary zero points -128/127
- [x] Per-channel Mul quantization: `--per-channel-mul` lowers an immutable per-channel constant onto the verified 1x1 depthwise profile, whose native per-output-channel weight scale quantizes each channel on its own grid (12 models / 192 board inferences, `research/per_channel_mul_suite/`). The EW task itself still carries one output multiplier/shift; a per-channel EW conversion through `BS_OW_CFG.OW_SRC=1` + a `0x5020` block was re-probed with the block at the named address in the decoded Conv format and still hangs, while `OW_SRC=1` + `OD_BYPASS=1` ignores the block (retained, `research/mul_per_channel_ow_suite/`)
- [ ] Signed/unsigned combinations: UINT8 external to INT8 operands/output is verified; INT8 external descriptors are absent from v3
- [ ] Wider products/output formats: the public ABI and verified DPU store path are INT8-only
- [x] Multiplier/shift representability is validated; unsupported scales are rejected
- [x] Small/large output scales and output saturation in the standalone conversion suite
- [x] Signed halfway rounding uses ties-even after adding output zero point; exact odd-parity Mul and Conv probes distinguish ordering
- [x] Extreme input codes occur in the exhaustive deterministic standalone vectors, including the signed product bounds
- [x] Zero/negative/greater-than-one immutable factors in broadcast tests
- [x] Operand-swap conversion behavior for matching external tensors and immutable broadcasts
- [x] Distinguish float ONNX Mul quantized-real lowering from QLinearConv import in metadata/contracts

## 9. Fusions and runtime requirements

- [x] Conv+ReLU across the public dense native stride/channel/tiling profiles
- [x] Conv+Clip/ReLU6: constant scalar Clip[0,6], native geometry/channel/tiling path and independent output conversion
- [x] Conv+LeakyReLU/PReLU: bounded pointwise Conv profiles, scalar Leaky alpha and scalar/per-channel PReLU slopes0..1
- [x] Conv→Sigmoid/Tanh public LUT profile: diagonal C3 1x1 stem with one scalar gain per channel (per-channel signs, any bias, power-of-two domain gains), all256 input codes, Q15 table rounding, sign-split table domain (`research/lut_domain_suite/`); mixed-input stems rejected with a retained failure (`research/lut_domain_failed/`)
- [x] Conv+residual Add: bounded RGB pointwise Conv(input)+input through an identity residual conversion
- [x] Conv+immutable scalar/per-channel scaling folded into weights and bias
- [x] Bounded Mul sequences: terminal Relu fusion, separate Clip[0,6] clamp task, and exactly representable scalar Add folded into output zero point
- [x] Fusion order and intermediate rounding for Conv->Relu branch operands -> Mul -> Relu, including one/two activated branches and nonzero output zero points
- [x] Multiple external inputs/outputs: v5 serializes the tensor table and board-verified profiles cover 2-5 named inputs and two outputs, including *unequal* runtime input shapes (`Mul` image plus `[1,1,1,C]` operand)
- [x] General buffer allocation/lifetimes: `open_rknpu.liveness` orders tasks, computes live intervals and places arena tensors by first fit, reusing bytes only where intervals are disjoint. The diamond places three simultaneously live internals and the chain ping-pong case is reproduced in `tests/test_liveness.py`
- [x] Packed/native16 layout conversion at public external boundaries and between currently scheduled operator paths
- [ ] Channel/spatial tiling: height tiling verified; input channels are solved to C128 by wider native planes (`native_c65_suite/`), while intermediate channel counts above 16 inside scheduled graphs still need the P1 DAG allocation (no partial-sum format is involved)
- [ ] Multi-task accumulation/synchronization: serial producer/consumer tasks pass, but partial Conv sums need accumulator storage and final conversion
- [ ] Aliasing and in-place execution: v3 deliberately requires disjoint input/output arena ranges; no safe hardware alias case is established
- [x] Hardware observations, failed hypotheses, rewrite bounds and compiler-imposed limits are labeled separately in this inventory and results log
- [x] Long LUT command submission: 1,106 register words with submission flags1 in Python and C loaders

## 10. Coverage combinations

- [x] Kernel x stride x dilation x padding sampled through native dilation/tiled and auto-padding suites
- [x] Input channels x output channels x groups sampled through group2..32, inputC32/outputC128
- [x] Odd/even geometry x channel tails sampled through square rewrite and depthwise/output-tail suites
- [x] Activation zero point x weight quantization x bias sampled at UINT8 zp0/17/128/255 and weight-edge cases
- [ ] Operator mode x predecessor/successor: bounded Conv/ReLU/Mul/Relu/Clip/Add and depthwise/pointwise orders pass; full matrix depends on the DAG planner and new operator profiles
- [x] Immutable Mul broadcast x operand order x output quantization in the bounded RGB profile
- [x] Single-task x tiled multi-task execution, including batch plus spatial tiling and shared constants

## Interpretation notes

- Mul uses independently scaled symmetric INT8 operands; output scale is 128*sA*sB.
  Standalone external Mul inserts identity Conv conversions; constant broadcasts
  use a true per-channel operand mode or a materialized native tensor.
- NCHW [C] aligns with the trailing axis; it is not automatically channel scaling.
- Group/dilation zero-filled dense rewrites are distinct from native hardware modes.
- The initial ConvTranspose failure was resolved by zero-insertion bias correction
  and kernel flipping; this proves only the bounded profile in the results ledger.
- Conv1D/Conv3D, MatMul/Gemm, deformable convolution and sparse acceleration are
  adjacent investigations, not implied by this 2D Conv/elementwise Mul inventory.

## Specification references

- [ONNX Conv](https://onnx.ai/onnx/operators/onnx__Conv.html)
- [ONNX ConvTranspose](https://onnx.ai/onnx/operators/onnx__ConvTranspose.html)
- [ONNX Mul](https://onnx.ai/onnx/operators/onnx__Mul.html)
- [Mesa/RK3588 crosswalk and known incompatibilities](../../research/hardware_refs/README.md)

## Work log

- 2026-09-09: Saved expanded inventory; resumed depthwise channel-packing investigation.
- 2026-09-09: Native odd K9..31 and H/W1..128 established. K33 failed; large surfaces use aligned serial height tiles with kernel halos.
- 2026-09-09: SAME/VALID auto-padding and bounded K2/K3 depthwise ConvTranspose padding, output shape and overlap suites passed.
- 2026-09-09: Standalone Mul gained independent output scale/zero-point conversion; five models and 80 board inferences passed.
- 2026-09-09: Same-input and reversed-source Mul passed C1/C5/C16 and H/W1..32: five models, 80 inferences.
- 2026-09-09: Direct depthwise rewrite expansion passed dilation, K2x3/K2x4, multipliers2/3/4 and C32→C128.
- 2026-09-09: Native weight edges passed absent bias, bias-only, zero channels and mixed per-channel ranges. A saturating bias-only halfway case remains unresolved and is retained as a failed hypothesis.
- 2026-09-09: Immutable scalar/channel/spatial/full broadcast Mul gained independent output conversion; four models and 48 inferences passed.
- 2026-09-09: Optional ConvTranspose bias verified for K2/K3 by synthesizing the required zero hardware bias table.
- 2026-09-09: Sigmoid/Tanh LUT setup promoted to a bounded public sequence profile; two models, 32 inferences and all256 codes passed.
- 2026-09-10: ERDMA secondary-operand fields named; compact per-row Mul variants hung the NPU and are retained as a failed hypothesis (spatial broadcast stays materialized).
- 2026-09-11: the hang is decoded - the missing `0x506c` `ew_surf_notch` (TRM) - and with the notch set the compact operand completes as a *linear* read, byte-identical to a host-side prediction, so no spatial broadcast exists (`research/mul_broadcast_notch_suite/`).
- 2026-09-10: Leading Pad folded to host preprocessing for constant/reflect/edge/wrap (padding_suite, 192 board inferences).
- 2026-09-10: Asymmetric depthwise weight zero points solved via the weight pair's second byte (depthwise_asymmetric_suite, 432 board inferences).
- 2026-09-10: Native input channels C33..64 solved and board-verified (native_c48_suite, native_c64_suite).
- 2026-09-10: N-layer native Conv chains board-verified over a shared arena (native_chain_suite).
- 2026-09-09: Bounded QLinearConv import passed C1..32→C1..64, K1/3/5, optional bias and per-output source scales; contract explicitly requantizes weights.
- 2026-09-09: Exact Mul halfway probe proved output zero point is added before ties-even shift, matching the independently isolated Conv ordering.
- 2026-09-09: Independent Mul boundary zero points verified through BS constant-source ALU plus EW offset: four models and 48 inferences.
- 2026-09-09: Bounded Q/DQ Conv import passed scalar/per-output weights, optional bias and C3/C17/C32 K1/3/5.
- 2026-09-09: Combined nonzero Mul operand boundaries with independent output conversion; cross-mode suite passed unchanged count.
- 2026-09-09: Batched same-input Mul passed N2/N4/N16 with C1/C5/C16 and nonzero boundary zero points.
- 2026-09-09: Terminal spatial Reshape after Mul passed two-input/same-input C3/C5/C16 profiles.
- 2026-09-09: Sequence/C ABI now declares one/two logical inputs and exposes tensor shape/offset/size; 384 two-input runs reverified through descriptor-aware harness.
- 2026-09-09: Tensor-contiguous two-input batch packing passed N2/N4/N8, C3/C8/C16 with nonzero Mul boundary zero points.
- 2026-09-09: Dense output C2..16 completed across K1/3/5, asymmetric pads and input C1/3/17/32: 15 models, 30 inferences.
- 2026-09-09: Four-edge padding passed kernels larger than 1x1/1x2/2x1 source images.
- 2026-09-09: Depthwise C3 ConvTranspose gained equal stride1 for K2/K3; four padding profiles passed 64 exact board inferences, and all retained stride2 profiles were reverified.
- 2026-09-09: Independent X/Y transpose stride fields were confirmed by capture and six public unequal-stride K2/K3 profiles passed 96 exact board inferences.
- 2026-09-09: ConvTranspose K1 and rectangular K1x3/K2x3/K3x2 sparse-K3 rewrites passed 80 exact runs; K2 dilation2 rewrite passed 64. A native K5 command remained off by one on 98/675 values after symmetric-boundary isolation and stays rejected.
- 2026-09-09: Depthwise ConvTranspose channel packing generalized and every C1..16 passed K2/K3 with equal or unequal strides: 16 models and 64 exact board inferences.
- 2026-09-09: Dense ConvTranspose passed eight C1..16 input/output tail combinations; grouped zero-fill lowering passed group2/3/4/8, reduction and depthwise multiplier2 in six more models.
- 2026-09-09: Dense transpose geometry broadened across stride1/2, unequal strides, asymmetric pads and legal output padding; a stride1 capture identified fixed input-domain register0x501c=14.
- 2026-09-09: Independent ConvTranspose output scale/zero-point overrides passed depthwise and dense modes, including zero points -128/127/-37/91.
- 2026-09-09: Chained depthwise Conv output overrides passed C4/C5/C8/C16 and output zero points -128/127/-31/73.
- 2026-09-09: Conv intermediate × second external RGB tensor lowered through one identity conversion; four shapes/zero-point/output-range combinations passed 32 exact runs.
- 2026-09-09: Terminal Relu fused into Mul by enabling both hardware lower clamps; four external-plus-intermediate graphs passed 32 exact runs.
- 2026-09-09: Conv→Clip[0,6]/ReLU6 fused with matching CNA/DPU upper thresholds at0x4028/0x40e4; four native geometry/channel/output-range profiles passed.
- 2026-09-09: Conv→PReLU gained Q14 constant slope tables; C3/C5/C8/C16 and slope endpoints0/1 passed 32 exact runs. Register0x5028 remains literal8 while hardware reads all channel entries.
- 2026-09-09: RGB Conv(input)+input residual graphs lower to learned and identity branches; four shapes/input quantizations passed 32 exact runs.

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
C5 hypothesis failure and the 6x6 spatial-layout mismatch; it does not resolve all
depthwise modes.

Evidence: research/depthwise_c5_suite through depthwise_c16_suite, each containing
board_results_0.json; research/capture_depthwise_c5 and depthwise_c5_capture.log.
Generator: research/probe_depthwise_channels.py CHANNELS; board runner:
research/run_profile_suite.py depthwise_cCHANNELS_suite (open Python/PYTHONPATH=src).
The generator uses independently compiled stem weights and commands, not captures.
