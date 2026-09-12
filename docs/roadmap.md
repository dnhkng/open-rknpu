# Roadmap, limits and measured residuals

This page is the honest boundary of the project: what works today, what was closed with
hardware evidence, what was closed as a *measured negative*, and what is deliberately out of
scope. The chronological record behind every item is in
[investigation-log.md](investigation-log.md); the detailed plans are in [`plans/`](plans/).

## Closed with board evidence

| Item | Result |
| --- | --- |
| Universal batched submission | every emitter's tasks link through the successor's fetch amount; a mixed-engine DAG is one job (`research/grouped_probe/`, `mixed_batched_probe/`) |
| Fence-free completion | the kernel has no fences, so a queued non-blocking job followed by a small blocking barrier job gives lag-0 completion (`research/barrier_probe/`) |
| Per-family cost model | duplicated-task fitting cross-checked on held-out containers (`research/family_cost_crosscheck/`) |
| Calibration | `minmax` / `percentile` / `kl` measured and used per Conv, including the walk (`research/percentile_calibration_suite/`, `examples/mel-kws/`) |
| Dense K3-dilation2 ConvTranspose | `research/transpose_dilation_dense_suite/` |
| Input channels to C128 | a vendor C128 Conv is a single CNA task, so no channel-split accumulation is needed (`research/native_c65_suite/`) |
| Emitter port to the composer | every DAG emitter composes byte-identically (`research/composer_port_reference.json`, `tests/test_composer_emitters.py`) |
| Trained audio model, whole graph on the NPU | `examples/mel-kws/`: 98.00% INT8, 191,968/192,000 exact bytes |
| Large-image chains | native16 image input for `3x32x32` (single task, ≤6144 atoms) with calibrated bands |

## Closed as measured negatives

| Item | Why it is closed |
| --- | --- |
| Per-channel output conversion | the DPU/ERDMA path has no per-channel output scale; measured, not assumed (`docs/plans/coverage-matrix.md`) |
| Native spatial broadcast of a second operand | the ERDMA reads its secondary operand strictly linearly, so a spatial constant must be materialized (`research/mul_broadcast_notch_suite/`) |
| Non-power-of-two LUT bands | compensating the whole-bit gain leaves ±1 on ~10% of codes; no affine fixed-point rule reproduces the measured index (`research/lut_mixed_gain_probe/`, `lut_index_probe/`) |
| Clock scaling / bandwidth policy | `SET_FREQ` is an empty body, `GET_BW_*` return `-EINVAL`, no SRAM pool, no regulator (`research/action_probe/`) |
| `GET_VOLT` | oopses the caller because the device tree has no rknpu regulator — never issue it |

## Residuals that need hardware or a human decision

1. **A distinct RV1106 SoC.** The attached board's NPU is the shared RV1106 NPU IP, so
   NPU-level results already apply to that IP. A different RV1106 die (clock tree, memory
   map, regulator, driver image) would need that hardware; the check is a re-run of the
   ledger, not a code change. Decision: acquire a second board or keep the claim scoped.
2. **A detector.** The pieces are measured (float reference, INT8 execution, calibration,
   two trained classifiers end to end), but a detector additionally needs a labelled
   dataset, a training budget and a board memory/latency budget. Decision: which dataset
   and budget.
3. **Publication.** The wheel and sdist build offline and the runtime sources ship inside
   them. What remains is an index release (name/credentials) and an announcement.

## Known residual inside a passing result

The mel-CNN board run reproduces 191,968 of 192,000 output bytes. The 32 differing bytes are
four utterances, each differing in one of the 64 output cells by ≤4 LSB, with **no**
classification change. Stage probes show the first two Convs and the first pool byte-exact
on those utterances and the difference starting in the third or fourth Conv, with interior
cells affected — a reference-vs-hardware rounding-tie residual, not a geometry error
(`research/mel_kws_suite/README.md`). Those four utterances are excluded from the byte-exact
suite.

## Known quirk: `chain`/`chain_n` hidden borders use code -128

`chain.py` and `chain_n.py` leave the internal border register `0x1184` at its `-128` reset
instead of programming the producer's zero point. A Conv border then reads
`(-128 - zero_point) * scale` rather than the real zero ONNX pads with. The integer
reference models the same behaviour, so containers stay byte-exact and board-verified, but
when a hidden band's zero point differs from `-128` the model loses real accuracy: measured
on a seeded 5-layer chain with hidden zero points `[-128, -17, -2, 0, 0]`, the container is
38 LSB from the ONNX-float result (and within 1 LSB of the `-128`-border pipeline). 178 of
the 242 retained chain models have a non-`-128` hidden zero point.
Every fan-out emitter (`graph.py`, `join_dag.py`, `pool_join.py`, `depthwise_join.py`,
`pooled_branches.py`, `depthwise.py`, `transposed.py`) *does* program `0x1184`, so the chain
family is the outlier. `examples/primitives/03_conv_chain.py` keeps hidden weights and
biases non-negative so the hidden band stays at `-128`. Changing the convention is a
register-profile change that would invalidate the chain family's board evidence
(`research/native_chain_suite/`), so it is left as a documented quirk with a test
(`tests/test_emit_semantics.py`) that pins both the convention and the gap.

## Known API warts

* `open_rknpu.model.decode` does not return the `register_count` field that `model.encode`
  requires, so a literal `encode(decode(blob))` is not callable; tests and callers pass the
  126 the decoder already enforces.
* `open_rknpu.native.native_input_reference` models the CNA accumulator but not the
  `Clip[0,6]` upper clamp (registers `0x4028`/`0x40e4`), so the Clip profile's *band* and
  clamp code are asserted instead of its integer output.
* `open_rknpu.compose.compose` validated neither duplicate stage names nor a stage binding
  the same register twice until the 2026-09-11 test expansion; both are now rejected with
  explicit errors because either one silently drops a task or a binding.

## Deliberately out of scope today

| Capability | Why | Closest workaround |
| --- | --- | --- |
| 1-D convolution (`kernel_shape [k]`) | **supported since 2026-09-12**: a rank-3 `[N,C,L]` graph is promoted to `[N,C,1,L]` in place (`research/conv1d_suite/`) | none needed; the container reports `H = 1` |
| Rectangular kernels with one-sided padding | the native profile accepts odd square kernels | even/rectangular kernels through 5×5 are rewritten to odd square |
| Kernels > 31 | outside the verified register encoding | split large kernels (e.g. an STFT basis) into shorter taps |
| `MatMul`/`Gemm` with a constant rank-2 weight | **supported since 2026-09-12** by lowering to the verified 1×1 Conv path (`research/matmul_suite/`) | none needed for the `[N,C,1,1]` / Flatten form |
| `Softmax`, `Concat`, `Slice`, `Pad` (non-constant), `ReduceMean`, `Shape`-driven control flow | no primitive; the project accepts a bounded static CNN class | host-side post-processing for the rest |
| LSTM/GRU and other recurrence | no hardware primitive and no GEMM to decompose into | keep the recurrent part on the CPU (`examples/mel-kws/` shows host-side pooling/argmax; a hybrid LSTM model is the same pattern) |
| Dynamic shapes / variable batch | containers are immutable | recompile per shape; batch ≤16 is supported inside one container |
| Multi-stage detection heads (upsample/concat/anchors) | build on unsupported ops | none yet |
| dma-buf zero-copy from the ISP | not implemented | stage through the runtime's buffers (measured overhead is in the example READMEs) |

A real pretrained model (Silero VAD) was checked against this envelope and blocked by four
independent gaps at once — 1-D convs, a 256-tap STFT kernel, input C129, and an LSTM — so the
project's audio milestone was instead a purpose-built mel-CNN that only uses verified
primitives (`examples/mel-kws/`). The full table is in
[plans/primitive-roadmap.md](plans/primitive-roadmap.md).

## How to contribute a primitive

See the eight-step recipe at the end of [architecture.md](architecture.md#adding-a-primitive)
and the evidence requirements in [verification.md](verification.md#adding-evidence-for-a-new-primitive).
The short version: reproduce it without vendor code, bound it, write the emitter and the
integer reference, add the scheduler branch and the suite, and land the board evidence in
the ledger.
