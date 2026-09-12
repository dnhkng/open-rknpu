# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `docs/`: generated register reference and error index (with drift-guard tests), op
  support matrix, glossary, troubleshooting guide, performance guide, C API guide, an
  annotated container walkthrough, the calibration cookbook, the board runbook, container
  migration and compatibility notes, the mutation-testing report, and the evidence-storage
  decision.
- `examples/cookbook/`: minimal C program, quantized-model import, mutable parameters,
  batched/pipelined submission, wheel-installed smoke test, rejection walkthrough, and an
  ONNX compatibility report. `examples/notebooks/` adds a runnable (and tested) walkthrough.
- Repository scaffolding: `CODE_OF_CONDUCT.md`, `SECURITY.md`, `AUTHORS`, `CITATION.cff`,
  `THIRD_PARTY.md`, `CHANGELOG.md`, `docs/provenance.md`, issue/PR templates, dependabot,
  pre-commit, `.editorconfig`, `.gitattributes`, mkdocs config, release, docs and mutation
  workflows.
- CI jobs that compile the C runtime and every board harness, install the built wheel in a
  clean venv and smoke test it, validate the distribution with `twine check`, run the loader
  and the board runner under AddressSanitizer/UndefinedBehaviorSanitizer, and check the
  evidence, cost-model, reproducible-build and coverage-table gates.
- Tests: container fuzzing and encoder properties, generated-graph properties, reference-doc
  drift guards, CLI flag contract, Clip-reference board replay, host runs of the classifier
  examples, the C loader over 48 crafted containers, the board runner's CLI (`--inspect`
  compared against the Python decoder for all 2,340 published containers), evidence
  integrity for every suite, the documented-command runner, rejection/boundary modules for
  the front end, emitters, joins and scheduler, and the release-gate logic
  (1,155 tests total, 99.20% compiler line coverage).
- Tools: `research/perf_regression.py` with a checked-in cost-model baseline,
  `research/check_reproducible_build.py` (normalise and audit the sdist),
  `research/run_mutation_tests.py` with `research/mutation_scope.json`,
  `research/coverage_doc_table.py` (generates the uncovered-line table in the docs), and
  `research/build_suite_readmes.py` (generates the 135 marker-based suite pages).

### Added (second pass)
- **Bounded `Concat` and global pooling** (F6): `Concat(axis=1)` of sibling Conv branches on
  the same input is lowered to one wide Conv by stacking weights, and
  `GlobalAveragePool`/`ReduceMean(axes=[2,3])` on an 8x8 map becomes three chained 2x2 pools
  through the verified reduction emitter. `Softmax`, `Slice` and `Resize` stay rejected with
  their exact messages. Board: `wide_concat_suite` 12/192/270,336 exact bytes,
  `global_pool_suite` 12/192/1,248, `global_pool_reduce_suite` 12/192/1,408.
- **Elementwise stages inside the op-level walk** (F5): `Conv[/Relu] -> Add|Sub|Max|Mul(constant)
  -> (2x2 pool ->)* Conv[/Relu]` now lowers through the walk, reusing the standalone
  elementwise register program and its reference. Board: `walk_elementwise_suite`
  12/192/27,648 exact bytes.
- **One-sided and asymmetric padding** (F7) is covered as a supported form rather than a gap:
  the emitter's explicit-pad path already handled it (bounded `pads < K`), and
  `rect_pad_suite` now proves it (12/192/30,144 exact bytes). K > 31 keeps the measured
  `odd K1..31` bound.
- **A per-inference timing API** (F9): `ornpu_run_timed`/`ornpu_run_io_timed` report
  `pack_ns`/`submit_ns`/`readback_ns`/`total_ns`; `tests/board_timed.c` records the numbers.
- **dma-buf zero-copy input** (F8): `ornpu_open_shared` allocates the arena from the Rockchip
  CMA heap and returns the fd, `ornpu_input_view` describes the producer's layout, and
  `ornpu_run_prefilled` runs without copying the input. Board: the packed
  `add_geometry_suite` through this path, 32 models / 64 inferences / 19,968 exact bytes.

### Changed (second pass)
- The container baseline covers 2,328 models (0 changed; 84 new suite models, one of which is
  a documented default rejection), the ledger is 129 rows / 1,786 models / 29,706 inferences
  / 10,605,299 exact output bytes, and the host suite is 1,155 tests at 99.20% line coverage.
- `ornpu_info` gained `arena_bytes`; `tests/test_container_bindings.py` now accepts a read
  address inside the container's own payload (an emitter-materialised constant) while still
  rejecting a write into the payload.
- The suite-page generator emits a factual "What this suite pins" section instead of a
  template stub.

### Changed
- `compile --target/--quantize` are validated and named in the summary instead of being
  accepted and ignored; the coverage floor is 99% (measured 99.85%) and CI covers Python
  3.10–3.13.
- `model.decode` now returns `register_count`, so `encode(decode(blob))` round-trips.
- `native_input_reference` models the fused `Clip[0,6]` clamp (`upper_code`, published as
  `meta["clip_upper_code"]`), so users can reproduce Clip containers.
- The C loader rejects a repeated or missing external tensor binding instead of silently
  dropping a caller buffer, and rejects a v5 container whose declared header size is not 96 -
  the Python decoder always did, so the two now agree.
- Removed the four GPL-2.0 kernel sources from `research/vendor/` (provenance retained), plus
  two pieces of provably dead code (`elementwise.py`'s redundant broadcast guard,
  `elementwise_chain.py`'s always-false payload padding and v5's unreachable constant
  packing).
- The source distribution ships the attribution files and prunes the evidence tree; the
  wheel carries `THIRD_PARTY.md` alongside `LICENSE`; calibration reports record
  repository-relative paths instead of a maintainer-specific absolute path.

### Fixed
- Documentation corrected where it disagreed with the code: checksum coverage and the v5
  layout description, `ornpu_inspect` reading the whole file, the v5 `batch − 1` header
  byte (it is `batch − 1` in v3/v4 too, not reserved), the legacy profile range (1–8, one to
  five tasks), and the `examples/mel-kws/features.py` input-scale note.

## [0.1.0] - 2026-09-11

The first public release: an open compiler and libc-only runtime for the Rockchip
RV1103 / RV1106 NPU, verified against the attached board rather than against a vendor
SDK. There is no vendor SDK, no RKNN library and no captured binary in the tree.

### Added

* **Compiler** (`src/open_rknpu/`) — an ONNX front end and normalizer, a scheduler with
  profile dispatch, one module per profile/emitter, a stage composer with declared
  bindings and a lifetime/arena allocator, and the container writer. It emits the
  register commands, weight/quantization blocks and container format itself, and
  rejects out-of-envelope graphs with an explicit `ValueError` instead of falling back
  to a vendor runtime.
* **Runtime** (`runtime/`) — a libc-only C runtime and file runner that talk to
  `/dev/rknpu` and nothing else, with the container specification in
  `runtime/sequence_format.md` (legacy `ORNPUBIN` v1/v2 and task-table `ORNPUSEQ`
  v3/v4/v5).
* **Verified primitive set** — dense Conv, even/rectangular and grouped kernel
  rewrites, depthwise Conv (including RGB stem and wider channels via dense rewrites),
  Max/AveragePool and multi-stage reduction, elementwise Add/Mul/Sub/Max with
  scalar/per-channel/spatial constants and general join DAGs, fused Relu, Clip[0,6] /
  ReLU6, LeakyReLU/PReLU and bounded Sigmoid/Tanh LUT activations, dense and depthwise
  transposed Conv, and quantized import (`QLinearConv` and QDQ graphs). Every profile
  ships an independent Python integer reference next to its emitter.
* **Batched and pipelined submission** — a whole DAG is submitted as one job through a
  linked single-engine task list (up to 11x on deep chains), with engine-run splitting
  for DPU→CNA graphs, double-buffered intermediate surfaces, non-blocking submission
  pipelining (measured 1.5–2.25x) and fence-free completion.
* **Calibration and quantization** — UINT8 input / INT8 output bands with zero points
  and requantization, and `minmax`, `percentile` and `kl` calibration methods with
  measured error paths and round-half-to-even tie behaviour pinned by tests.
* **Mel-CNN example** (`examples/mel-kws/`) — a trained 4,090-parameter mel-CNN for
  spoken digits runs entirely on the NPU: **98.00%** INT8 test accuracy,
  **191,968 / 192,000** board-exact output bytes, 2.54 ms mean latency.
* **Low-level examples** — `examples/primitives/` has one runnable script per
  primitive plus an MNIST-style model, and `examples/mnist/` and `examples/fashion/`
  hold the hybrid classifiers.
* **Evidence baseline and ledger** — `research/container_baseline.json` pins the
  container bytes (and the deliberate `ERR:<Type>` rejections) for a
  **2,244-model** baseline, checked by `research/verify_suites.py`; a **121-row board
  ledger** covers **1,696 models / 28,266 inferences / 10,216,467 exact output
  bytes**; `research/campaign_sweep.py` and `research/check_docs_links.py` are the
  other reproducible contracts.
* **Test suite** — **873 host tests at 98% compiler line coverage**, gated in CI with
  a `fail_under = 97` floor. The 2026-09-11 expansion added 449 tests across 16
  modules, including independent float64 semantics checks, per-profile bounds tests
  with first-out-of-bounds neighbours, composer/arena invariants, container
  round-trips and a decode sweep over 2,340 published containers.
* **Documentation** — `docs/` covering getting started, architecture, the primitive
  catalog, quantization, the container format, the board workflow, verification, the
  Python API, the roadmap and limits, the publish checklist, and the 68-entry
  investigation log that keeps the failed hypotheses alongside the findings.

### Changed

* The scheduler's op-level walk dispatches the join class, lowers fan-in joins and
  pools inside a Conv chain, and stages out-of-range image inputs as native16 surfaces
  so larger (for example 3×32×32) inputs have a path.
* `compile_chain_walk` accepts the `calibration.measure` report and uses each Conv's
  measured band, and the scheduler passes `calibration_ranges` through instead of
  rejecting it, which is what makes a trained model compile with usable accuracy.
* The tail control word names the successor's fetch amount, so every DAG submits as one
  job; the per-family cost table is cross-checked against held-out containers.

### Fixed

* The chain family applies its hidden Relu (S9).
* `K>1` strip tiling uses the real pad value and the real activation (S7).
* Input channels C1–128 are supported — the earlier C64 cap was the compiler's, not the
  hardware's (P2).
* `compose.compose` now raises on duplicate stage names and on a stage binding the same
  address register twice; previously both were silently dropped, losing a task or a
  binding. No emitted container changed, because no existing emitter does either.
* The ERDMA secondary operand is read linearly — there is no spatial broadcast (P4).
* The elementwise output stage performs no BS-table read (P4 sub-item).
* Known and pinned rather than fixed: `chain.py`/`chain_n.py` leave the internal border
  register `0x1184` at `-128` instead of the producer zero point (38 LSB from the float
  model on a seeded 5-layer chain; 178 of 242 retained chain models affected). The board
  evidence is byte-exact against that convention, so changing it would invalidate
  `research/native_chain_suite/`; `docs/roadmap.md` records it.

[Unreleased]: https://github.com/dnhkng/open-rknpu/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dnhkng/open-rknpu/releases/tag/v0.1.0
