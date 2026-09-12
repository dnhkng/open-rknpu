# Support matrix

Which ONNX constructs this compiler accepts, with the exact bounds, the profile that
accepts them, a runnable example, and the board evidence behind the bound. The
vocabulary (profile, band, native16, runner, …) is defined in
[glossary.md](glossary.md); when a graph is rejected, the error text and the fix are
in [troubleshooting.md](troubleshooting.md).

## How to read a row

* **Bounds** are the numbers the emitter enforces. They come from the per-profile
  validation code in `src/open_rknpu/*.py`; the ledger
  ([research/COVERAGE_EXPANSION_RESULTS.md](../research/COVERAGE_EXPANSION_RESULTS.md))
  records how each was measured.
* **Profile** is the `meta["profile"]` value of the first profile that accepts the
  graph; where a profile has no distinct `meta["profile"]` string, the emitter module
  from [architecture.md](architecture.md#the-schedulers-profile-order) is named instead.
  The dispatch order is in that same section.
* **Evidence** is prefixed `board:` or `host:`.
  * `board:` names a directory holding recorded results — usually a
    `research/*_suite/` with a `board_results_*.json`, and for the MNIST classifier the
    example's own recorded board run. The container ran on the attached RV1103 board and
    every output byte matched the profile's Python integer reference.
  * `host:` means the compiler accepts it and the example checks the integer
    reference on the host, but no board run is recorded for that exact row.
* `board:` evidence is byte-exact, never a float tolerance; `host:` is not called
  board-verified ([verification.md](verification.md), [primitives.md](primitives.md)).
* Example scripts are host-only: they compile, print the profile metadata and check
  the reference. A script that replays a published suite is noted in its row.
* Anything not listed is rejected today; see [Rejected constructs](#rejected-constructs).

---

## 1. Dense convolution (`Conv`)

The workhorse. A UINT8 NHWC image is staged on a native16 surface; the CNA multiplies
against packed INT8 weights and writes an INT8 native16 grid
([primitives.md](primitives.md), [quantization.md](quantization.md)).

| ONNX op / attribute | Accepted bounds (exact) | Profile | Example | Evidence |
| --- | --- | --- | --- | --- |
| `Conv`, explicit `pads` | batch 1..16 · input C1..128 · output C1..128 · H/W 1..128 · odd K1..31 · stride 1..4 per axis · dilation 1..17 per axis · pads 0..K−1 per side with `pt,pb < effective K_h` and `pl,pr < effective K_w` · constant float32 weights/bias · output shape `[N,OC,oh,ow]` must match the graph | `native16-input` | `../examples/primitives/01_native_conv.py` | `board: native_input_suite, conv_geometry_suite, conv_boundary_suite, conv_asymmetric_suite, native_output_limits_suite` |
| `Conv`, `auto_pad` = `VALID`/`SAME_UPPER`/`SAME_LOWER` | folded to explicit `pads` before lowering (constant weights + static spatial shape required); same geometry bounds as above | `native16-input` (rewrite) | `../examples/primitives/01_native_conv.py` | `board: auto_padding_suite, native_large_suite` |
| `Conv`, `strides` 3/4 | stride 3 or 4 on either axis, both packed and native16 input paths | `native16-input`, `strided` | `../examples/primitives/01_native_conv.py` | `board: conv_stride34_suite, native_stride34_suite` |
| `Conv`, `dilations` 2..17 | per-axis dilation 1..17 native; an effective kernel ≤5 is expanded into weights instead; unequal 2×3/3×2 verified; dilation 18+ rejected | `native16-input` | `../examples/primitives/01_native_conv.py` | `board: native_dilation_verified_suite, group_dilation_suite, native_large_suite` |
| `Conv`, even/rectangular kernel | each side 1..5, not already an odd square (1/3/5); requires `dilations=[1,1]` and `auto_pad=NOTSET`; embedded into a K3 (max≤3) or K5 kernel with extra bottom/right padding | `native16-input` (rewrite) | `../examples/primitives/01_native_conv.py` | `board: rectangular_conv_suite, auto_padding_suite` |
| `Conv`, `group` 2..32 | group 2..32; total input channels `group × weights.shape[1]` ≤32; output channels ≤128 and divisible by group; rewritten to a zero-filled dense kernel | grouped rewrite → `native16-input` | `../examples/primitives/01_native_conv.py` | `board: group_dense_suite, group_expansion_suite, group_dilation_suite` |
| `Conv` + fused `Relu` | one directly connected standard `Relu`; activation registers set by the emitter | `native16-input`, `strided`, legacy image layer | `../examples/primitives/01_native_conv.py` | `board: native_input_suite, conv_geometry_suite` |
| `Conv` + fused `Clip[0,6]`/`ReLU6` | constant scalar `min=0`, `max=6`; compiled output band scale 6/255, zero point −128; matched CNA/DPU thresholds | `native16-input` | `../examples/primitives/01_native_conv.py` | `board: conv_clip_suite` |
| `Conv`, batch 2..16 | static batch 2..16 as serial per-batch native tasks sharing one weight/bias block; batch and height tiling compose | `native16-input` | `../examples/primitives/01_native_conv.py` | `board: native_batch_suite` |
| `Conv`, input above 6144 pixel-atoms | `H·W·ceil(C/16) > 6144` is split into aligned serial height tiles with overlap halos; tiling refused if it cannot align | `native16-input` | `../examples/primitives/01_native_conv.py` | `board: native_large_suite, native_spatial_tiling_suite` |
| `Conv`, accumulator range | INT32 accumulation guarded to ±2,076,148,792; ties round half to even; output zero point is added before the final ties-even shift | all dense profiles | — | `board: accumulator_boundary_suite, weight_edge_suite, output_channels_complete_suite` |
| `Conv` + `Mul` (scalar / `[C,1,1]`) | constant factor folded into weights and bias when the Conv has one consumer; factor shape `()`, `(1,)`, `(C,1,1)` or `(1,C,1,1)` and float32 | fold in `normalize.py` → dense profile | `../examples/primitives/06_constant_mul.py` | `host:` fold; the resulting Conv is covered by the dense rows above |

## 2. Depthwise and grouped convolution

| ONNX op / attribute | Accepted bounds (exact) | Profile | Example | Evidence |
| --- | --- | --- | --- | --- |
| `Conv` with `group = C` after a stem | graph input RGB `[1,3,H,W]`, H/W 5..8 → Conv stem (K1 or K3, output C1..16, stride 1, symmetric pad) → group-`C` depthwise; C1..16, K1/3/5, stride 1/2, symmetric `pads=K//2`, `dilations=[1,1]`, optional bias, weights `[C,1,K,K]`; C5..16 requires a 1×1 stem | `depthwise_profile` (see `depthwise.py`) | `../examples/primitives/04_depthwise.py` | `board: depthwise_combined_suite, depthwise_geometry_suite, depthwise_small_channels_suite` |
| depthwise + dense 1×1 pointwise successor | depthwise C3/5/9/16, stride 1/2, K1/3/5 → dense 1×1 Conv, input C1..16, output C1..128, no pads/stride/dilation, group 1 | depthwise + `native16-input` | `../examples/primitives/04_depthwise.py` | `board: depthwise_pointwise_suite` |
| depthwise, asymmetric weight zero points | opt-in `--sequence --asymmetric-depthwise`; per-channel weight zero point stored in the weight pair's second byte; C1/C3/C4, K1/3/5, positive/negative/mixed weights | `depthwise_profile` | `../examples/primitives/04_depthwise.py` | `board: depthwise_asymmetric_suite` |
| depthwise, no bias | omitted bias synthesizes a zero hardware table; C1/C4/C9/C16, K1/3/5, stride 1/2 | `depthwise_profile` | `../examples/primitives/04_depthwise.py` | `board: depthwise_no_bias_suite` |
| depthwise, output override | `output_range`/calibration on the depthwise output, including nonzero output zero points | `depthwise_profile` | `../examples/primitives/10_calibration.py` | `board: depthwise_output_quantization_suite` |
| image-input depthwise (`graph input → group Conv`) | lowered by the front-end dense rewrite: group 2..32, effective input C ≤32, output C ≤128 divisible by group; covers multiplier 1 and channel multipliers 2..4 (C32→C128) | rewrite → `native16-input` | `../examples/primitives/04_depthwise.py` | `board: depthwise_rewrite_expansion_suite` |
| image-input depthwise, even/rectangular kernel | K2x3/K2x4 through the square-kernel rewrite (each side ≤5) | rewrite → dense rows | `../examples/primitives/04_depthwise.py` | `board: depthwise_rewrite_expansion_suite` |
| image-input depthwise, dilation 2/3 | effective kernel ≤5 expanded into weights by the dense rewrite; larger effective kernels need the native dilation field | rewrite → `native16-input` | `../examples/primitives/04_depthwise.py` | `board: depthwise_rewrite_expansion_suite` |

## 3. Transposed convolution (`ConvTranspose`)

| ONNX op / attribute | Accepted bounds (exact) | Profile | Example | Evidence |
| --- | --- | --- | --- | --- |
| depthwise `ConvTranspose`, K2/K3 | graph input RGB 8×8 → `Conv[/Relu]` stem C1..16 at 8×8 → depthwise group-C ConvTranspose; C1..16, square K2/K3, per-axis stride 1/2, `pads` each `0..K−1`, `output_padding` each `< axis stride`, `auto_pad` NOTSET/VALID/SAME_UPPER/SAME_LOWER, `output_shape` (2 dims, no explicit pads) | `transposed_profile` | `../examples/primitives/08_transpose.py` | `board: transpose_stride1_suite, transpose_unequal_suite, transpose_channels_suite, transpose_padding_suite, transpose_output_shape_suite, transpose_no_bias_suite, transpose_learned_suite` |
| depthwise `ConvTranspose`, K5 | direct 25-tap K5 with asymmetric per-channel weight zero points and the vendor-derived phase field `0x1068 = ((K−1−pads_left)<<8) \| (K−1−pads_top)`; stride 1/2, pads 0/1/2, `output_padding` | `transposed_profile` | `../examples/primitives/08_transpose.py` | `board: transpose_k5_suite` |
| depthwise `ConvTranspose`, K3 + `dilations=[2,2]` | emitted directly as a sparse K5 (zero-stuffed odd taps) | sparse rewrite | `../examples/primitives/08_transpose.py` | `board: transpose_k5_dilation_suite` |
| dense `ConvTranspose`, K3 | stem output C1..16 at 8×8 → dense ConvTranspose C1..16→C1..16, K3, per-axis stride 1/2, asymmetric pads `0..K−1`, legal `output_padding`, SAME/VALID/`output_shape` | `transposed_profile` (`8x8-dense-…`) | `../examples/primitives/08_transpose.py` | `board: transpose_dense_suite` |
| dense `ConvTranspose`, K3 dilation2 | zero-stuffed to K5 and emitted through the dense K5 form; C4→C3 pads0/pads1, C8→C5 stride2, C3→C3 pads2 | dense sparse-K5 rewrite | `../examples/primitives/08_transpose.py` | `board: transpose_dilation_dense_suite` |
| dense `ConvTranspose`, K5 with arbitrary taps | the dense emitter accepts square K3/K5, but only the zero-stuffed K3-dilation2 geometry above has board evidence; treat a general dense K5 field set as unverified | `transposed_profile` | `../examples/primitives/08_transpose.py` | `host:` (no board suite for arbitrary dense K5 taps) |
| grouped `ConvTranspose` | group 2/3/4/8, input/output C1..16; expanded to zero-filled block-diagonal dense weights; reduction and depthwise multiplier 2 also lower this way | dense rewrite | `../examples/primitives/08_transpose.py` | `board: transpose_grouped_suite` |
| rectangular `ConvTranspose` | K1x3/K2x3/K3x2 (each side 1..3, not square) with per-axis pads below the kernel; zero-stuffed into K3 | sparse-K3 rewrite | `../examples/primitives/08_transpose.py` | `board: transpose_rectangular_suite` |
| `ConvTranspose`, K2 dilation2 | zero-stuffed into a sparse K3 | sparse-K3 rewrite | `../examples/primitives/08_transpose.py` | `board: transpose_dilation_suite` |
| `ConvTranspose`, output override | `output_range` or calibration; verified for output zero points −128, 127, −37, 91 (dense/depthwise) | `transposed_profile` | — | `board: transpose_output_quantization_suite` |

## 4. Pooling (`MaxPool` / `AveragePool`)

| ONNX op / attribute | Accepted bounds (exact) | Profile | Example | Evidence |
| --- | --- | --- | --- | --- |
| `MaxPool`/`AveragePool`, terminal | `kernel_shape=[2,2]`, `strides=[2,2]`, `pads=[0,0,0,0]`, `ceil_mode=0`, `count_include_pad=0`, `auto_pad=NOTSET`, `dilations=[1,1]`, `storage_order=0`; preceding Conv from the legacy profile (I=1 or 3, O1..16, K1/3/5, stride 1, symmetric pad, H/W 5..8 for I=3 or 5..32 for I=1); integer average path rounds half to even | legacy pooling profile 3/4, or the scheduler's pool branch | `../examples/primitives/02_conv_pool.py`, `../examples/primitives/11_mnist_digits.py` | `board: examples/mnist/README.md` (Conv1→Relu→MaxPool1 on the NPU, 98.67%); the 8×8→4×4 research replay `mnist_pool_suite` is `host:` only |
| `MaxPool`/`AveragePool`, N of them / interior | one or more 2×2 stride-2 pools in any position of a linear Conv/Relu chain; input H/W ≥2 before each pool; ≤64 NPU tasks per container; a Conv feeding a pool is re-quantized onto a zero-point-0 grid | `chain-walk` | `../examples/primitives/02_conv_pool.py`, `../examples/primitives/03_conv_chain.py` | `board: walk_chain_suite, mel_kws_suite` |
| three-stage reduction to 1×1 | the legacy `network` profile (`Conv,Relu,Conv` + three matching 2×2 stride-2 pools → float32 `[1,3,1,1]`) and the `reduction` profile (a Conv followed by three matching pools); a terminal `GlobalAveragePool`, or `ReduceMean(axes=[2,3], keepdims=1)`, on a static `[N,C,8,8]` tensor is rewritten by the front end into those three 2×2 stride-2 stages, so the mean carries the explicit graph's **per-stage INT8 rounding**, not an unrounded float mean | legacy profile 5/6/7/8, `chain-walk`, or the scheduler's pooling-sequence branch | `../examples/primitives/02_conv_pool.py` | `board: global_pool_suite` (12 v5 models / 192 inferences / 1,248 exact bytes), `board: global_pool_reduce_suite` (12 v3 models / 192 / 1,408); `../research/global_pool_suite/`, `../research/global_pool_reduce_suite/` |
| pools inside branch joins | two Conv+2×2-pool branches joined after the pool (Max or Average), or pooled multi-layer branches, or a terminal pool on a join DAG | `pool-join`, `pooled-branches`, `join-dag` | `../examples/primitives/09_join_dag.py` | `board: pool_join_suite, pooled_branches_suite, pooled_dag_suite` |

## 5. Elementwise joins (`Add` / `Sub` / `Mul` / `Max`)

| ONNX op / attribute | Accepted bounds (exact) | Profile | Example | Evidence |
| --- | --- | --- | --- | --- |
| two `Conv` branches → `Add`/`Sub`/`Max` | same-input RGB branches, input H/W 5..8 (RGB) or native16 batch 1..16/C1..16; branch output C2..16 (RGB profile) or C1..16 (native profile); branches 1×1 Conv; `Add`/`Sub`/`Max` require one shared band and re-quantize | `elementwise_profile` (`add`/`sub`/`max`, `equal-scale`) or `…-native16-input` | `../examples/primitives/05_elementwise.py` | `board: add_geometry_suite, sub_geometry_suite, max_geometry_suite, native_elementwise_suite` |
| two `Conv` branches → `Mul` | as above, but each branch keeps its own scale and the join folds both; full centered product `sA·sB·(qA−zA)·(qB−zB)`; operand zero points are INT8 | `elementwise_profile` (`mul`, `independent-scales`) | `../examples/primitives/05_elementwise.py` | `board: mul_geometry_suite, mul_sources_suite, native_mul_suite` |
| `Add(Conv(input), input)` residual | RGB pointwise `Conv(input)` plus the matching RGB graph input, lowered through an identity-Conv conversion | elementwise residual rewrite | `../examples/primitives/05_elementwise.py` | `board: residual_add_suite` |
| `Mul(Conv(input0), input1)` | `Conv(input0)` × a matching external RGB input1, lowered through an identity-Conv conversion | elementwise external rewrite | `../examples/primitives/05_elementwise.py` | `board: mul_external_intermediate_suite` |
| standalone `Mul`, one or two external inputs | C1..16, H/W 1..32, batch N1..16 (same input twice) or N1..8 (two inputs); the same buffer may feed both operands; reversed operand order verified; output scale/zero point independently selectable | `standalone_mul` | `../examples/primitives/05_elementwise.py`, `../examples/primitives/06_constant_mul.py` | `board: standalone_mul_suite, two_input_mul_suite, mul_two_input_batch_suite, mul_batch_suite` |
| `Mul` operand zero points | `--sequence --mul-a-zero-point Z --mul-b-zero-point Z`, each INT8; accepted only when the graph contains a `Mul` node, otherwise `Mul operand zero points require a Mul profile` | `elementwise.py` / `mul_*` | — | `board: mul_boundary_zero_points_suite` |
| `Mul` output quantization | independent output scale/zero point; the offset participates before ties-even rounding | `mul_output_conversion` in `elementwise.py` | `../examples/primitives/10_calibration.py` | `board: mul_output_quantization_suite, mul_output_rounding_suite` |
| `Mul` + terminal `Relu` | one directly connected standard `Relu`, fused into the elementwise task via both hardware lower clamps | elementwise terminal fusion | `../examples/primitives/05_elementwise.py` | `board: mul_relu_suite, mul_relu_order_suite, mul_after_relu_suite` |
| `Mul` + terminal `Clip[0,6]` | needs constant scalar `[0,6]`; appends the captured-and-rebuilt INT8 clamp task; output band fixed at scale 6/255, zero point −128 | elementwise terminal fusion | `../examples/primitives/05_elementwise.py` | `board: mul_clip_suite` |
| `Mul` + terminal scalar `Add` | `Add` by a finite float32 scalar **exactly representable** at the requested output scale; folded into Mul's internal zero point | elementwise terminal fusion | `../examples/primitives/05_elementwise.py` | `board: mul_add_suite` |
| runtime per-channel scale `Mul(image, scale[1,3,1,1])` | two external inputs of unequal shape: image `[1,3,H,W]` H/W 5..8 and per-channel `[1,3,1,1]`; the scale is a second named v5 input | `runtime-scale-mul` | `../examples/primitives/05_elementwise.py` | `board: runtime_scale_suite, join_scale_suite` |
| multi-input / multi-stage DAG | `Conv,Conv,{Add\|Mul\|Sub\|Max},(Conv,Mul)*` with 3, 4 or 5 named external inputs; or `Mul({Add,Mul,Sub,Max}(a,b),a)` up to four stages | `elementwise_multi.py`, `elementwise_chain.py` | `../examples/primitives/05_elementwise.py` | `board: elementwise_multi_suite, elementwise_dag_suite, elementwise_chain_suite, elementwise_deep_suite` |

## 6. Constant `Mul` and broadcasting

| ONNX op / attribute | Accepted bounds (exact) | Profile | Example | Evidence |
| --- | --- | --- | --- | --- |
| `Mul` by immutable scalar | one external input, batch 1 as RGB `[1,3,H,W]` with H/W 5..8; batch 2..16 uses the batched constant path (C1..16, H/W 1..32); broadcastable finite float32 constant | standalone Mul + constant lowering | `../examples/primitives/06_constant_mul.py` | `board: mul_broadcast_suite, mul_broadcast_output_suite` |
| `Mul` by immutable `[N,C,1,1]` / scalar | N2..16, C1..16, H/W 1..32; a scalar or per-batch-channel vector stored once per batch (16-byte vector), not a materialized spatial tensor | batched constant lowering | `../examples/primitives/06_constant_mul.py` | `board: mul_batch_broadcast_suite` |
| `Mul` by immutable per-channel `[1,C,1,1]` | single batch, RGB H/W 5..8; one 16-byte operand row; the constant scale is one per tensor unless `--per-channel-mul` is used | constant lowering (`per-channel` data mode) | `../examples/primitives/06_constant_mul.py` | `board: mul_broadcast_suite` |
| `Mul` by immutable spatial `[1,1,H,W]`, full `[1,C,H,W]`, per-row `[1,1,H,1]`, per-column `[1,1,1,W]` | single batch, RGB H/W 5..8 (the batched path requires a spatially invariant constant); a spatial constant is **materialized into a native plane**; the ERDMA reads its secondary operand linearly, so there is no compact spatial broadcast | constant lowering (`per-pixel` data mode) | `../examples/primitives/06_constant_mul.py` | `board: mul_broadcast_suite, mul_broadcast_output_suite` |
| `Mul` by per-channel constant, per-channel quantized | `--sequence --per-channel-mul`: one external input, one immutable spatially invariant constant with 3 distinct channel values, RGB H/W 5..8; lowered onto the verified 1×1 depthwise profile so each channel gets its own weight scale | `per-channel-constant-mul` | `../examples/primitives/06_constant_mul.py` | `board: per_channel_mul_suite` |
| mutable `Mul` factor (v4) | opt-in `--mutable-constants`: exactly one constant `Mul`; a named `mul.factor` packed region the runtime may overwrite at a fixed compiled factor scale | standalone Mul + v4 descriptor | — | `board: mutable_mul_factor_suite` |
| immutability | the constant must be an ONNX initializer, not an overridable graph input; a constant masquerading as an input is rejected | all constant profiles | `../examples/primitives/06_constant_mul.py` | `host:` |

## 7. Activations

| ONNX op / attribute | Accepted bounds (exact) | Profile | Example | Evidence |
| --- | --- | --- | --- | --- |
| `Relu` (fused) | one directly connected standard `Relu` after one Conv; also every hidden `[Conv,Relu]` layer of a chain | dense, chain, DAG profiles | `../examples/primitives/01_native_conv.py` | `board: native_input_suite, native_chain_suite, join_chain_suite` |
| `Clip[0,6]` / `ReLU6` (fused) | constant scalar `min=0`, `max=6`; must be `Conv` followed by `Clip` | `native16-input` | `../examples/primitives/01_native_conv.py` | `board: conv_clip_suite` |
| `LeakyRelu` | exactly one `Conv` followed by `LeakyRelu`; `alpha` finite and `0 ≤ alpha ≤ 1`, encoded Q14; conservative INT32 positive-path overflow guard | `activation.py` (`compile_leaky`) | — | `board: leaky_verified_suite` |
| `PRelu` | exactly one `Conv` followed by `PRelu`; constant float32 slopes in `[0,1]`, scalar or `[C,1,1]`, encoded Q14 in a UINT16 table; same overflow guard | `activation.py` (`compile_prelu`) | — | `board: prelu_public_suite` |
| `Sigmoid` / `Tanh` (LUT) | exactly `Conv`→`Sigmoid`/`Tanh`; input and output `[1,3,8,8]`; Conv is a diagonal C3 1×1 stem with one scalar magnitude per channel, input scale 1 and zero point 128, one scalar gain per channel (per-channel signs allowed), any bias, and `BASE_WEIGHT_SCALE/weight_scale` a power of two; analytic stem range inside the sign-split table domain; output band Sigmoid `(1/255, −128)` / Tanh `(1/127, 0)`; 1106-word setup in both loaders | `lut_profile` (`sigmoid-8x8-c3` / `tanh-8x8-c3`) | `../examples/primitives/07_lut_activation.py` | `board: lut_public_suite, lut_domain_suite` |
| `Sigmoid`/`Tanh` after anything else | only the bounded LUT profile above; a `Sigmoid`/`Tanh` elsewhere is not a primitive | — | — | see [Rejected constructs](#rejected-constructs) |
## 8. Layout, padding and quantized import

| ONNX op / attribute | Accepted bounds (exact) | Profile | Example | Evidence |
| --- | --- | --- | --- | --- |
| terminal spatial `Reshape` | immutable shape; rank 4 → rank 4; batch 1; channels preserved; `H·W` pixel count preserved; float32 in/out; only the emitted output geometry changes, the NPU storage is unchanged | prefix profile + shape patch (`layout.py`) | `../examples/primitives/05_elementwise.py` | `board: mul_reshape_suite` |
| channel-changing `Reshape`/`Transpose` | not supported | — | — | `host:` rejected |
| leading `Pad` on the graph input | constant pads tensor of 8 values (rank-4 NCHW); mode `constant`/`reflect`/`edge`/`wrap`; non-negative amounts; no explicit `axes`; folded into the input shape and recorded in `meta["input_padding"]`; the caller pads the packed UINT8 input with `open_rknpu.padding.pad_input` | front-end fold + any profile | `../examples/primitives/01_native_conv.py` | `board: padding_suite` |
| CNA border padding (constant) | inside a Conv: explicit `pads` or auto-pad, injecting the activation zero point | dense/chain profiles | `../examples/primitives/01_native_conv.py` | `board: border_padding_suite` |
| `Pad` in any other position | non-leading `Pad`, dynamic pads, negative amounts, explicit `axes`, or a rank other than 4 NCHW are rejected | — | — | `host:` rejected |
| one constant-parameter `QLinearConv` | UINT8 input, INT8 output, scalar activation parameters, scalar or per-output-channel weight scale/zero point, optional INT32 bias; source INT8 weight bytes preserved bit-for-bit; geometry is the native bounds (batch 1..16, H/W 1..128, C1..128, odd K1..31, stride 1..4, dilation 1..17) plus preserved-quantization range checks; board-verified on C1..32→C1..64 and K1/3/5 sample points | `quantized_import.py` | — | `board: qlinearconv_import_suite` |
| `DequantizeLinear → DequantizeLinear → Conv → QuantizeLinear` | constant-weight DQ (`axis=0` for per-channel), one activation input, one output; float32 bias rounded once to INT32; output Q scalar | `quantized_import.py` | — | `board: qdq_conv_import_suite` |
| other quantized graphs | dynamic quantization tensors, longer quantized graphs, non-axis-0 per-channel DQ and non-scalar output Q are rejected | — | — | see [Rejected constructs](#rejected-constructs) |

## 9. Graph shapes (composition profiles)

Accepted graphs that are not a single ONNX op. Same evidence convention.

| Graph shape | Accepted bounds (exact) | Profile | Example | Evidence |
| --- | --- | --- | --- | --- |
| `[Conv,Relu]*(N−1)+[Conv]` chain | fixed `[1,3,8,8]`, N≥3 odd, hidden C3..16, K1 or padded K3, stride 1, group 1, one shared arena with explicit intermediate lifetimes | `native-chain` | `../examples/primitives/03_conv_chain.py` | `board: native_chain_suite` |
| chain intermediates exposed | `--expose-intermediates`: every layer output as a named v5 external output | `native-chain` (v5) | — | `board: chain_multi_suite` |
| chain arena reuse | `--reuse-intermediates`: layer L writes the buffer layer L−1 last read; byte-identical output | `native-chain` (lifetime pass) | — | `board: chain_reuse_suite` |
| height-strip tiled chain | `--tiles N` with N dividing 8 (1/2/4/8); K1 and padded K3; hidden C3..16; refuses output override, exposed intermediates or arena reuse | `tiled-native-chain` | — | `board: tiled_chain_probe, tiled_k3_probe` |
| legacy `Conv,Relu,Conv` 8×8/C3 chain | one standard `Relu` between two 1×1 or padded 3×3 Convs; per-layer calibration and final output overrides | legacy profile 2 | `../examples/primitives/03_conv_chain.py` | `board: chain_output_quantization_suite` |
| linear Conv/Relu chain with a pool in any position | walk handles a bounded linear graph with 2×2 stride-2 pools and a native16 image input (C≤16, ≤6144 atoms); calibrated bands accepted | `chain-walk` | `../examples/primitives/02_conv_pool.py` | `board: walk_chain_suite, mel_kws_suite` |
| two-head fan-out | `Conv,Relu,Conv,Conv`, two named outputs | `two-head-fanout` | `../examples/primitives/09_join_dag.py` | `board: two_head_suite` |
| diamond (stem → two heads → one join → optional Conv tail) | stem `[Conv]` or `[Conv,Relu]`, two dense 1×1/3×3 heads, one `Add`/`Mul`/`Sub`/`Max` join, optional `[Conv,Relu]*Conv` tail | `diamond-join`, `diamond-tail`, `walk-join` | `../examples/primitives/09_join_dag.py` | `board: diamond_suite, diamond_tail_suite, walk_join_suite` |
| chained fan-out (3..5 heads) | shared 1×1 stem (optional Relu), dense or group-3 depthwise heads, N−1 mixed joins, optional Conv tail; optional runtime scale or residual tail | `join-chain`, `join-chain-tail`, `join-chain-scale`, `join-chain-residual` | `../examples/primitives/09_join_dag.py` | `board: join_chain_suite, mixed_head_suite, join_scale_suite, join_residual_suite` |
| general join DAG | two or three joins over any two previously produced tensors; branches of one to three dense Conv layers or block-diagonal depthwise expansions; optional terminal 2×2 pool | `join-dag` | `../examples/primitives/09_join_dag.py` | `board: join_dag_suite, branch_join_suite, depthwise_chain_suite, pooled_dag_suite` |
| dense + depthwise branch join | one 1×1 stem → dense branch and group-3 depthwise branch → `Add`/`Mul`/`Sub`/`Max` | `depthwise-join` | `../examples/primitives/09_join_dag.py` | `board: depthwise_join_suite` |
| pooled branch join | two Conv+pool branches joined after the 2×2 stride-2 pool; and pooled multi-layer branches | `pool-join`, `pooled-branches` | `../examples/primitives/09_join_dag.py` | `board: pool_join_suite, pooled_branches_suite` |
| sibling-Conv `Concat(axis=1)` | two or more plain Conv branches reading one shared graph input with identical kernel/strides/pads/dilations and `group=1`, no other consumer of a branch output, and a terminal Concat; the branch weights and biases are stacked into one wide Conv | front-end rewrite → any single-Conv profile | — | `board: wide_concat_suite` (12 models / 192 inferences / 270,336 exact bytes); host byte-equality with the hand-written wide Conv |
| calibration | `calibration_ranges=report["ranges"]` for a single `Conv[/Relu/Clip]`, the legacy Conv-ReLU-Conv chain per layer, and the elementwise/Mul profiles; `minmax`, `percentile` and `kl` | profile-dependent | `../examples/primitives/10_calibration.py` | `board: sequence_calibration_suite, percentile_calibration_suite` |
| batched submission | `--submission batched` relinks any emitted container into one job per engine run; serial stays the default and is faster below roughly eight same-engine tasks | any | — | `board: grouped_probe, mixed_batched_probe, batched_all_probe` |
| mutable Conv parameters (v4) | `--mutable-weights`: one native `Conv[/Relu]`; a named `conv.parameters` region replaceable at run time | `native16-input` (v4) | — | `board: mutable_weights_suite` |

---

## Rejected constructs

Every row here is enforced by the compiler with an explicit error (no silent
fallback). "Roadmap item" points at where the decision is recorded: a phase in
[plans/completion-plan.md](plans/completion-plan.md), a row in
[roadmap.md](roadmap.md), or a category in
[plans/coverage-matrix.md](plans/coverage-matrix.md).

| ONNX construct / graph | Reason (representative error text) | Roadmap item | Evidence / retained artifact |
| --- | --- | --- | --- |
| 1-D convolution (`kernel_shape [k]`, rank-3 input) | **supported**: the front end promotes a rank-3 `[N,C,L]` graph to `[N,C,1,L]` in place (pads `[a,b]` -> `[0,a,0,b]`), then the native Conv/pool emitters handle it; a graph with any other node type is rejected by name (`1-D rank promotion cannot rewrite …`) | `../research/conv1d_suite/` (12 models, ic 1/3, oc 1..16, k 2/3/5, pads 0..2, stride 1/2, with/without Relu, L 16..128) | board: 12 models / 192 inferences / 38,464 exact bytes |
| Conv input channels > 128 | `native Conv requires static batch1..16, H/W1..128, input C1..128, constant weights/bias` | [plans/completion-plan.md](plans/completion-plan.md) P2 (C1..128 accepted; >C128 is the measured lane/tensor bound) | `../research/native_c65_suite/`, `../research/hardware_identity.md` |
| Kernels > 31 | `native Conv supports odd K1..31, explicit padding, stride 1..4, input C1..128/output C1..128` | [roadmap.md](roadmap.md) (kernels > 31; the K33 mismatch) | `../research/native_kernel_limits_suite/` |
| Dilation > 17 | `native padding/stride/dilation unsupported` | [plans/coverage-matrix.md](plans/coverage-matrix.md) §1 (dilation 18+ failed independently generated references) | `../research/native_dilation_verified_suite/` |
| Even/rectangular kernel with one side > 5, or one-sided padding | `native Conv supports odd K1..31, …` / `native padding/stride/dilation unsupported` | [roadmap.md](roadmap.md) (rectangular kernels with one-sided padding) | `../research/rectangular_conv_suite/` |
| Grouped Conv with `group > 32`, total input C > 32, or output > 128 | the dense rewrite declines and the native profile then rejects `group != 1` | [plans/coverage-matrix.md](plans/coverage-matrix.md) §3 | `../research/group_dense_suite/` |
| Depthwise multiplier > 1 after a stem | the depthwise profile requires weights `[C,1,K,K]`; multiplier forms only lower for a direct image input | [plans/coverage-matrix.md](plans/coverage-matrix.md) §3 | `../research/depthwise_rewrite_expansion_suite/` |
| Per-axis ConvTranspose dilation, or dilation beyond K2/K3-dilation2 | `ConvTranspose dilation is supported only for square K2 or depthwise K3 with dilation2` / `K3 dilation2 ConvTranspose requires constant float32 …` | [plans/completion-plan.md](plans/completion-plan.md) P3 (the K5 tap table is square) | `../research/transpose_dilation_suite/` |
| `MatMul`/`Gemm` (dense layer) | **supported** when the weight is a constant rank-2 tensor and the activation is `[N,C,1,1]` or flattened from it: lowered to the verified 1×1 Conv path; `Softmax`, `Slice`, `Pow`, `Sqrt` and `Shape`-driven control flow still have no primitive (`sequence lowering requires one input, one output, and an initial Conv`) | `../research/matmul_suite/` (12 models: direct, Flatten, Reshape, Gemm ± bias, transB=1) | board: 12 models / 192 inferences / 1,152 exact bytes; host byte-equality with the hand-written 1×1 Conv |
| `LSTM`/`GRU` and other recurrence | no hardware primitive and no GEMM to decompose into | [roadmap.md](roadmap.md) "Deliberately out of scope" | [plans/primitive-roadmap.md](plans/primitive-roadmap.md) |
| Dynamic shapes / variable batch | every dimension must be static: `native Conv requires static batch1..16, …` | [roadmap.md](roadmap.md) | — |
| Non-leading / dynamic / rank-3 `Pad`, or `Pad` with `axes` | `leading Pad requires constant pads`, `leading Pad must pad a rank-four NCHW input`, `leading Pad with explicit axes is not supported`, or the node is simply not folded | [plans/completion-plan.md](plans/completion-plan.md) P5 (host preprocessing by design) | `../research/padding_suite/` |
| A native NPU copy/pad producer for non-constant borders | the CNA border path injects only the activation zero point | [plans/completion-plan.md](plans/completion-plan.md) P5 | `../research/border_padding_suite/` |
| Calibration combined with an explicit `output_range` | `calibration and output quantization overrides cannot be combined` (chain: `calibration and chain output override cannot be combined`) | [quantization.md](quantization.md) | — |
| Calibration on profiles without a band contract | `calibration is unsupported for the join chain` / `… join DAG` / `… pooled branches`; `output override unsupported for LUT profile` / `… LeakyRelu profile` / `… PRelu profile` / `… Reshape profile` | [plans/completion-plan.md](plans/completion-plan.md) P7 | `../examples/primitives/10_calibration.py` |
| LUT `Sigmoid`/`Tanh` with mixed input weights | `LUT stem must be a diagonal 1x1 Conv with one scalar magnitude per channel` | [plans/completion-plan.md](plans/completion-plan.md) P6 | `../research/lut_domain_failed/` |
| `Sigmoid`/`Tanh` outside the bounded LUT tail | `LUT profile requires Conv followed by Sigmoid or Tanh`; no other profile lowers an activation tail | [plans/completion-plan.md](plans/completion-plan.md) P6 | `../examples/primitives/07_lut_activation.py` |
| LUT non-power-of-two / mixed bands | `LUT stem weight scale must make BASE_WEIGHT_SCALE/scale a power of two; …` / `LUT stem channels must share one negative-half gain band` | [plans/completion-plan.md](plans/completion-plan.md) P6 (index residual measured; no affine rule) | `../research/lut_mixed_gain_probe/`, `../research/lut_index_probe/` |
| Native spatial broadcast of a second operand | the ERDMA reads its secondary operand strictly linearly, so a spatial constant must be materialized | [roadmap.md](roadmap.md) "Closed as measured negatives" | `../research/mul_broadcast_notch_suite/` |
| Per-channel **output** conversion on the DPU/ERDMA path | measured negative: `OW_SRC=1` hangs with a well-formed block; `OW_SRC=1` + `OD_BYPASS=1` ignores it | [roadmap.md](roadmap.md) "Closed as measured negatives" | `../research/mul_per_channel_ow_suite/` |
| In-place / aliased execution | v3+ requires disjoint input/output arena ranges; no safe hardware alias case is established | [plans/completion-plan.md](plans/completion-plan.md) §6 (non-goals) | — |
| FP16 / other datatypes, wider outputs | INT8-only ABI; Toolkit2 2.3.0 rejects `do_quantization=False` for RV1103 | [plans/coverage-matrix.md](plans/coverage-matrix.md) §4 | [research/COVERAGE_EXPANSION_RESULTS.md](../research/COVERAGE_EXPANSION_RESULTS.md) |
| Dynamic quantization tensors, longer quantized graphs | only the bounded `QLinearConv` and DQ→Conv→Q forms are parsed | [plans/coverage-matrix.md](plans/coverage-matrix.md) §4 | `../research/qlinearconv_import_suite/`, `../research/qdq_conv_import_suite/` |
| Two expanding runtime operands | the runtime per-channel scale is the one verified unequal-shape form | [plans/coverage-matrix.md](plans/coverage-matrix.md) §7 | `../research/runtime_scale_suite/` |

## How to read a rejection

1. The message names the profile and the bound that failed. The CLI prints it as
   `open-rknpu: <message>` and exits non-zero; the Python API raises `ValueError`.
2. Find the construct in the tables above: the profile column says which emitter
   decided, and the bounds column says which number is out of range.
3. `meta["profile"]` is only set after a successful compile; on rejection, run
   `open-rknpu normalize model.onnx -o norm.onnx` to see whether the front-end
   rewrites (auto-pad, group/dilation, even kernels, leading `Pad`, Conv+Mul) already
   changed the graph.
4. Read [troubleshooting.md](troubleshooting.md) symptom first if the graph compiles
   but the result is wrong; read the roadmap item if the graph does not compile and
   the missing capability is deliberate.

## Related documents

* [primitives.md](primitives.md) — the same primitives with register and emitter notes.
* [roadmap.md](roadmap.md) — what is out of scope and why.
* [quantization.md](quantization.md) — bands, calibration and accuracy failure modes.
* [architecture.md](architecture.md) — the dispatch order and the composer.
* [examples/primitives/README.md](../examples/primitives/README.md) — what each example
  script demonstrates.
