# Primitives

Every NPU operation this project emits, with its verified bounds, its emitter, its example
script and the reference you can compare against. "Verified" means the container was run on
the attached board and matched a Python integer reference byte-for-byte; the suite that
proves each bound is named in [research/COVERAGE_EXPANSION_RESULTS.md](../research/COVERAGE_EXPANSION_RESULTS.md).

Each primitive has a runnable, host-only example under
[`examples/primitives/`](../examples/primitives/README.md). The scripts compile the graph,
print the profile metadata and check the host reference, so they double as documentation
and as a smoke test.

## 1. Dense convolution (image input)

`Conv` from a UINT8 NHWC image. The workhorse: the CNA reads the image surface, multiplies
against packed INT8 weights and writes an INT8 native16 grid.

* emitter: `native.compile_native_input`, `strided.compile_strided`
* bounds: batch 1–16 · input C1–128 · output C1–128 · H/W 1–128 · odd K1–31 · stride 1–4 ·
  dilation ≤17 · explicit or `SAME`/`VALID` auto padding · optional fused `Relu` or
  `Clip[0,6]` · inputs above 6144 pixel-atoms are split into serial height tiles
* example: `examples/primitives/01_native_conv.py`
* reference: `open_rknpu.native.native_input_reference(inputs, quantization, zero_point, pads, strides, dilations)`
* notes: the image is staged as a native16 surface (16 lanes per pixel); the CNA read
  address, feature grains and scan flags all describe that surface, which is why the
  geometry-aware builder is reused for every later Conv.

## 2. Conv + pooling

`Conv[/Relu]` followed by 2×2 stride-2 pools (Max or Average), including three-stage
reduction to 1×1.

* emitter: `pooling.py`, `reduction.py`
* bounds: single Conv (same bounds as §1) · 2×2 stride-2 pools to the output · RGB 8×8
  geometry for the classic profile · integer pools round half to even on the average path
* example: `examples/primitives/02_conv_pool.py`
* notes: the DPU pool task reads a zero-point-0 grid, so a Conv that feeds a pool is
  re-quantized onto a zero-point-0 band (`_adjusted_scale` widens the scale to keep the
  band inside the grid).

## 3. Chains

Two flavours:

* **native chain** — an odd-length `Conv,Relu,Conv,Relu,…,Conv` list at 8×8 where every
  Conv sees the previous grid; `--tiles N` splits it into N height strips with halos.
  Emitter `chain_n.py` / `tiled_chain.py`, profile `native-chain` / `tiled-native-chain`.
* **walk chain** — any linear graph of Conv/Relu with 2×2 pools *in any position*,
  including the large-image native16 first stage and calibrated bands.
  Emitter `walk.compile_chain_walk`, profile `chain-walk`.
  Example: `examples/primitives/03_conv_chain.py`.

## 4. Depthwise and grouped convolution

* **depthwise** — RGB stem → multiplier-1 depthwise, H/W 5–8, C1–16, K1/3/5, stride 1/2,
  optional bias, optionally followed by a dense pointwise Conv (`depthwise.py`).
* multiplier 2–4, even/rectangular kernels and asymmetric weight zero points lower through
  dense/expanded rewrites; grouped Conv through group 32 becomes a zero-filled dense
  kernel (`graph.py` normalizes it).
* example: `examples/primitives/04_depthwise.py`.

## 5. Elementwise joins, fan-out and fan-in

* two-branch `Add`/`Mul`/`Sub`/`Max` at 8×8/C3 from two Conv branches
  (`elementwise.py`), with `Mul` needing a full centered product and an independent output
  band;
* terminal `Mul`+`Relu`/`Clip`/`Add` fusions;
* fan-out to two heads (`graph.compile_two_head`), to 3–5 heads folded by mixed joins
  (`graph.compile_join_chain`), a diamond (stem + two heads + one join + optional Conv
  tail, `graph.compile_diamond` / `walk.compile_join_walk`), dense+depthwise branch joins,
  pooled-branch joins, multi-layer pooled branches, general join DAGs where internal grids
  feed several joins (`join_dag.py`), and runtime scale/residual operands
  (`elementwise.py` runtime-scale profiles).
* example: `examples/primitives/05_elementwise.py`, `examples/primitives/09_join_dag.py`;
  board evidence: `research/diamond_suite/`, `join_chain_suite/`, `join_dag_suite/`,
  `pool_join_suite/`, `depthwise_join_suite/`, `pooled_branches_suite/`, `join_scale_suite/`,
  `join_residual_suite/`, `branch_join_suite/`, `mixed_head_suite/`.

## 6. Immutable constants and broadcasting (`Mul`)

A Conv grid (or an external input) multiplied by an immutable constant: scalar,
per-channel `[N,C,1,1]`, spatial, full, per-row/column and per-batch, plus `--per-channel-mul`
which quantizes the constant per channel through a 1×1 depthwise Conv.

* emitter: `elementwise.py`, `native_elementwise.py`
* example: `examples/primitives/06_constant_mul.py`
* note: the ERDMA secondary operand is read **linearly**; there is no compact spatial
  broadcast mode, so a spatial constant is materialized into a plane (measured negative:
  `research/mul_broadcast_notch_suite/`).

## 7. Activations

* fused `Relu` and `Clip[0,6]`/`ReLU6` inside the Conv task;
* scalar / scalar-per-channel `LeakyRelu` and `PRelu` with a conservative INT32
  positive-path overflow guard (`activation.py`);
* **Sigmoid/Tanh LUT** — a C3 1×1 diagonal stem with one scalar gain per channel
  (per-channel signs allowed, any bias, power-of-two domain gains), sign-split
  negative/positive banks (`lut.py`). Mixed-input stems and non-power-of-two bands stay
  rejected with retained failure evidence (`research/lut_domain_failed/`,
  `research/lut_mixed_gain_probe/`).
* example: `examples/primitives/07_lut_activation.py`.

## 8. Transposed convolution

`ConvTranspose` as a depthwise or dense layer: K1/K2/K3, per-axis stride 1/2, padding and
`output_shape` modes, rectangular kernels via sparse rewrites, K5 through the vendor-derived
phase field (`transposed.py`). Grouped, reduction and multiplier-2 forms lower to zero-filled
dense weights.

* example: `examples/primitives/08_transpose.py`
* board evidence: `research/transpose_k5_suite/`, `transpose_k5_dilation_suite/`,
  `transpose_dilation_dense_suite/`.

## 9. Layout

Terminal spatial `Reshape` preserves pixel order and channel count (`layout.py`); a channel
-changing reshape/transpose is not supported.

## 10. Quantized-model import

Constant-parameter `QLinearConv` and `DequantizeLinear → Conv → QuantizeLinear` graphs keep
the supplied INT8 weight bytes, UINT8 input and INT8 output with scalar or per-output
activation parameters (`quantized_import.py`). Float Q/DQ bias is rounded once to INT32.

## 11. Runtime-level primitives

| Primitive | Where | What it does |
| --- | --- | --- |
| Batched submission | `sequence.relink_for_batched`, `--submission batched` | links the tasks of one engine run into a single job; the tail control word is the successor's fetch amount |
| Non-blocking pipelining | `ornpu_submit_flags(model, ORNPU_JOB_NONBLOCK, NULL)` | queue the next inference while the current one runs (1.5–2.25× on paired rounds) |
| Fence-free completion | barrier submission (`research/barrier_probe/`) | the kernel has no fence support, so a tiny blocking job after a non-blocking one marks completion at lag 0 |
| Mutable parameters | `--mutable-weights`, `--mutable-constants` | v4 packed descriptors the runtime can overwrite between inferences |
| Arena reuse | `--reuse-intermediates`, `open_rknpu.liveness` | reuses a dead tensor's arena space; the diamond suite asserts `allocated_bytes` |
| Exposed intermediates | `--expose-intermediates` | emits v5 chain intermediates as external outputs for debugging |

## Rejected today (with the reason)

| Graph | Why |
| --- | --- |
| 1-D convolution (`kernel_shape [k]`, rank-3 input) | the front end requires static NCHW rank 4 |
| rectangular kernels with one-sided padding | the native profile accepts odd square kernels (even/rectangular only through 5×5 rewrites) |
| input channels > 128 | `native Conv requires ... input C1..128` |
| kernels > 31 (e.g. the STFT Conv of Silero VAD) | odd K1..31 is the verified range |
| LSTM/GRU, `MatMul`/`Gemm`, `Softmax`, `ReduceMean`, `Concat`, `Slice`, `Shape`-driven control flow | no primitive: the project accepts a bounded static CNN class only |
| dynamic shapes | every dimension must be static at compile time; recompile for a new shape |
| calibration + output override together | the scheduler rejects the combination; put the output band in the calibration report instead |

The measured envelope against a real pretrained model (Silero VAD) is tabulated in
[primitive-roadmap](plans/primitive-roadmap.md); the smallest useful audio model that
*does* fit is implemented in [`examples/mel-kws/`](../examples/mel-kws/README.md).
