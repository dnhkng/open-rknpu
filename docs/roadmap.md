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
| Input channels past C128 | a CNA task reads far more than eight 16-lane planes: the limit is the 511-part weight table, so C1..16352 in / C1..8192 out are lowered and the emitted containers are board-exact (`research/wide_channel_suite/`, `research/probe_wide_channel_wall.py`) |
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

## Fixed: the `chain`/`chain_n` hidden border zero point

`chain.py` and `chain_n.py` used to leave the internal border register `0x1184` at its
`-128` reset instead of programming the producer's zero point, so a hidden Conv border read
`(-128 - zero_point) * scale` rather than the real zero ONNX pads with. The integer
reference modelled the same behaviour, which is why the containers stayed byte-exact and
board-verified while the *model* lost accuracy whenever a hidden band's zero point differed
from `-128`.

Fixed on 2026-09-12: every chain layer now declares the band it reads (`chain.py`,
`chain_n.py`, and the opt-in height-strip `tiled_chain.py`), and the references pass the
predecessor's zero point into the border. The change was intentional and re-baselined with
fresh board evidence; the measured effect against the float ONNX model is:

| Model | Mean error before | Mean error after |
| --- | ---: | ---: |
| `native_chain_suite/model001` | 1,598.06 | 40.86 |
| `native_chain_suite/model003` | 46,197.56 | 1,122.73 |
| `chain_reuse_suite/model001` | 1,378.48 | 85.03 |
| `chain_reuse_suite/model003` | 41,444.34 | 1,247.81 |
| `deep_chain_suite/model000` | 195,508.84 | 5,584.92 |
| `deep_chain_suite/model001` | 15,728,511,708.80 | 2,197.65 |
| `chain_calibration_suite` (chain5, analytic) | 53,161.90 | 9,404.41 |

Chains whose hidden bands are all zero point `-128` are byte-identical before and after (the
register value does not change). The 17 affected containers are re-pinned in
`research/container_baseline.json` and re-run on the board
(`native_chain_suite` 5/80/15,360, `chain_reuse_suite` 5/80/15,360, `deep_chain_suite`
3/48/9,216, `chain_multi_suite` 4/32/40,448, `chain_calibration_suite` 6/96/18,432 exact
bytes), and the two height-strip probes were regenerated and re-run.

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
| Rectangular kernels with one-sided padding | **supported since 2026-09-12** where the explicit-pad path accepts them: one-sided top/bottom/left/right, asymmetric pairs and rectangular K1xK3 (`research/rect_pad_suite/`) | none needed inside those bounds |
| Kernels > 31 | outside the verified register encoding | split large kernels (e.g. an STFT basis) into shorter taps |
| `MatMul`/`Gemm` with a constant rank-2 weight | **supported since 2026-09-12** by lowering to the verified 1×1 Conv path (`research/matmul_suite/`) | none needed for the `[N,C,1,1]` / Flatten form |
| `Concat` of sibling Conv branches; terminal `ReduceMean(axes=[2,3])` on 8x8 | **supported since 2026-09-12** in the bounded cases (stacked weights / three chained pools) | `Softmax`, `Slice`, `Resize`, non-constant `Pad` and `Shape`-driven control flow stay out of scope; host-side post-processing for the rest |
| LSTM/GRU and other recurrence | no hardware primitive and no GEMM to decompose into | keep the recurrent part on the CPU (`examples/mel-kws/` shows host-side pooling/argmax; a hybrid LSTM model is the same pattern) |
| Dynamic shapes / variable batch | containers are immutable | recompile per shape; batch ≤16 is supported inside one container |
| Multi-stage detection heads (upsample/concat/anchors) | build on unsupported ops | none yet |
| dma-buf zero-copy from the ISP | not implemented | stage through the runtime's buffers (measured overhead is in the example READMEs) |

A real pretrained model (Silero VAD) was checked against this envelope and blocked by four
independent gaps at once — 1-D convs, a 256-tap STFT kernel, input C129, and an LSTM. Two of
those are closed (a rank-3 1-D Conv lowers through the promotion path, and the channel wall
is now C16352), so the model's remaining blockers are the 256-tap/stride-128 STFT basis, the
two `LSTM` layers and the `If`/`Shape`/`Gather`/dynamic-`Slice` control flow; a multi-layer
1-D chain is still not dispatched (the walk rejects `[1,129,...,1,L]` chains with
`unsupported chain walk graph`). The project's audio milestone was instead a purpose-built
mel-CNN that only uses verified primitives (`examples/mel-kws/`). The full table is in
[plans/primitive-roadmap.md](plans/primitive-roadmap.md).

## How to contribute a primitive

See the eight-step recipe at the end of [architecture.md](architecture.md#adding-a-primitive)
and the evidence requirements in [verification.md](verification.md#adding-evidence-for-a-new-primitive).
The short version: reproduce it without vendor code, bound it, write the emitter and the
integer reference, add the scheduler branch and the suite, and land the board evidence in
the ledger.
