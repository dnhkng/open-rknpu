# Pre-publication checklist

What stands between the current repository and a public release. Written as a gap analysis:
items already done are marked ✅ so the list stays honest about how much is finished.
Effort: **S** < 1 h · **M** ~ half a day · **L** 1–3 days · **XL** > 3 days.

"Publish" means three separate things, with different gates:

| Target | Gate |
| --- | --- |
| **Public GitHub repository** | the P0 list below (legal, no silently-wrong interfaces, CI that builds everything shipped) |
| **PyPI release** | P0 plus the packaging/release-engineering list (metadata, changelog, trusted publishing, install smoke test) |
| **Announcement** (blog/HN/Reddit) | P1 plus a docs site, a compatibility report and at least one non-toy demo |

Current baseline: 1,155 tests, 99.20 % compiler line coverage (floor 99 %), 2,328-model
container baseline, 129-row board ledger, 27 documentation pages (~78 k words) plus the
planning records, 5 example sets, ~16,400 tracked files / 38.5 MiB pack.

## Status after the 2026-09-12 pass

Everything marked ✅ below is in the tree and verified by the test suite; the rest is the
open list. Counted by category: **P0 legal 6/6 · P0 interfaces 4/4 · P0 CI 4/4 · P0 release
2/3 · P1 tests 14/14 · P1 documentation 14/14 · P1 examples 11/13 · P1 automation 7/8 ·
P1 features 7/14**. That leaves **11 open rows**: F4–F10 (the board-gated feature work),
E5 and E12 (camera and audio examples), A7 (signed tags) and R3 (claiming the PyPI name,
which needs the maintainer's account). The remaining feature rows each need a probe and
fresh board evidence; the plan and the measured probes so far are in
[plans/batch-ab-plan.md](plans/batch-ab-plan.md).

What the 2026-09-12 batch changed:

* Batch A closed in full: fetch-script tests (T14) with the toolchain fetcher now verifying
  every download against its pinned git blob id; the branch-protection contract (A5) and a
  `pull_request` trigger for the docs build; the mutable-parameter API (F11,
  `open_rknpu.mutable` + a cookbook example); the API-stability contract (F12); the missing
  integer references for transposed Conv, pooling, reduction and the legacy network
  profiles (F13, replayed byte-for-byte against 91 transposed models plus four suites); and
  the depthwise-separable, benchmark and multi-model examples (E7/E9/E13) with real board
  runs recorded in their READMEs;
* Batch B started: `MatMul`/`Gemm` lower to the verified 1×1 Conv path (F1,
  `matmul_suite` board-verified), 1-D convolution is promoted to the `[N,C,1,L]` form (F2,
  `conv1d_suite` board-verified), and the calibration contract is centralised and threaded
  through the chain/join/diamond/walk profiles (F3, `chain_calibration_suite` board-verified
  with float MAE 6.96/672.8/53161.9 → 0.72/33.8/251.6 on 3/4/5-layer chains);
* probes that decide the remaining rows: the NPU driver **imports dma-bufs** (`CREATE`
  flag `0x80`, probed with `research/probe_dmabuf.c`), so F8 is feasible and needs the
  runtime binding; `Concat`/`Slice`/`Resize`/`Softmax`/`ReduceMean`/`GlobalAveragePool`
  after a Conv are all rejected today, with the exact messages in the plan (F6);
* the host suite is 1,155 tests at 99.20% line coverage with every remaining line carrying
  a reason in the generated table (`research/coverage_doc_table.py`, checked in CI), the
  container baseline covers 2,328 models (0 changed, 84 new suite models; one of them is a documented default rejection), and the ledger is
  124 rows / 1,726 models / 28,746 inferences / 10,274,515 exact output bytes.

## P0 — blockers before the repository goes public

### Legal and licensing

| # | Item | Why it blocks | Effort |
| --- | --- | --- | --- |
| ✅ L1 | **Remove the four GPL-2.0 kernel sources** in `research/vendor/README.md` (`rknpu_drv.c`, `rknpu_job.c`, `rknpu_mem.c`, `rknpu_ioctl.h`) — or add `THIRD_PARTY.md` with the GPL-2.0 text and correct the README | MIT `LICENSE` + `README.md` currently claims "GPL kernel sources are **not** redistributed here", and the repo contradicts it. Mixed licensing is legal but must be stated, and the current statement is false | S |
| ✅ L2 | **`THIRD_PARTY.md` / `NOTICE`** listing every non-MIT component that *is* shipped or fetched: NumPy (BSD-3), ONNX (Apache-2.0), Luckfox cross toolchain (fetched, not shipped), pretrained MNIST model (Apache-2.0), Fashion-MNIST (MIT), FSDD (CC BY-SA 4.0, fetched), `research/hardware_refs/rocket_registers.h` (GPL-2.0-only OR MIT — note the MIT choice) | Without it a user cannot tell what the MIT licence covers | M |
| ✅ L3 | **Clean-room / provenance statement** (`docs/provenance.md`): no vendor code is included; register values were recovered by experiment, from public documentation and from GPL sources read as documentation; the vendor toolkit was used only as a development oracle | This is the project's central claim; it needs to be explicit and citable, not inferred from the log | M |
| ✅ L4 | **Trademark / non-affiliation note** (Rockchip, RV1103, RV1106, RKNN, Luckfox) | Prevents the impression of an official Rockchip project | S |
| ✅ L5 | **Audit remaining `research/` provenance**: `.rknn` fixtures and instrument dumps are vendor-toolkit outputs — record how they were produced and that they are generated from our own ONNX models (`research/vendor/provenance.json` exists for the driver copies; extend it) | "No vendor artifacts" claim must be precise about what *is* kept as evidence | S |
| ✅ L6 | Confirm the toolchain download path (`research/fetch_toolchain.py`, Luckfox SDK) is permitted and documented, or make it optional and point users at their own toolchain | Users will otherwise redistribute it themselves | S |

### Interfaces that are silently wrong

| # | Item | Why it blocks | Effort |
| --- | --- | --- | --- |
| ✅ I1 | `open-rknpu compile --target` and `--quantize` are **accepted and ignored** (`cli.py:15-16`) | The worst kind of interface bug: a user passes `--target rv1106` and gets a container that is not that | S |
| ✅ I2 | `model.decode` does not return `register_count`, which `model.encode` requires, so `encode(decode(x))` is not callable | Published API that cannot round-trip | S |
| ✅ I3 | `native_input_reference` does not model the `Clip[0,6]` upper clamp that the emitter programs | Users comparing against the reference see a mismatch and cannot tell whether the hardware or the reference is wrong | S |
| ✅ I4 | Remove or wire the dead code and unreachable guards found by the test expansion (`chain.py:122`; the defensive guards in `depthwise_join.py:238`, `pooled_branches.py:206`) | Dead code in a published compiler reads as unfinished | S |

### CI must build everything that ships

| # | Item | Why it blocks | Effort |
| --- | --- | --- | --- |
| ✅ C1 | **Cross-compile/host-compile the C runtime in CI** — `make -C runtime` and a host gcc build of `tests/board_*.c` | CI currently never compiles any C; a syntax error in `runtime/open_rknpu.c` would reach users | S |
| ✅ C2 | **Wheel/sdist install smoke test in a clean venv** (install the built wheel, import, run `open-rknpu compile/inspect` on a fixture) | The distribution is built but never installed in CI | S |
| ✅ C3 | `python -m build` + `twine check` on the artifacts | Metadata errors are only found at upload time otherwise | S |
| ✅ C4 | Fix the version claim mismatch: `requires-python >= 3.10` but `ruff target-version = "py39"`, and CI tests only 3.10/3.12 | Either support 3.9 or drop the claim; also add 3.13 | S |

### Release hygiene

| # | Item | Why it blocks | Effort |
| --- | --- | --- | --- |
| ✅ R1 | Real project URLs, authors/maintainers in `pyproject.toml` (`github.com/dnhkng/open-rknpu`, maintainer `dnhkng`) | A published package with placeholder metadata cannot be corrected after upload | S |
| ✅ R2 | Decide the version: drop `.dev0` → `0.1.0` for the first release, and add `CHANGELOG.md` | "dev0" on PyPI is a pre-release that some tools will not install by default | S |
| R3 | Confirm/claim the PyPI name (`open-rknpu`); the GitHub repository URL is fixed to `github.com/dnhkng/open-rknpu`. Checked 2026-09-12: `https://pypi.org/pypi/open-rknpu/json` returns 404 (the name is unclaimed); claiming it needs the maintainer's PyPI account and is the last manual step before `PYPI_PUBLISH=true` | Last-minute name collisions are avoidable | S |

---

## P1 — strongly recommended before announcing

### Features

| # | Item | Value | Effort |
| --- | --- | --- | --- |
| ✅ F1 | **`MatMul`/`Gemm`** (via 1×1 Conv lowering) — `normalize.py` lowers MatMul/Gemm with a constant rank-2 weight (and the Flatten/Reshape-to-[N,C] form) to the verified 1x1 Conv path; `tests/test_dense_lowering.py`; `research/matmul_suite` board-verified `PASS: 12 models, 192 inferences, 1152 exact output bytes` | Unlocks most ONNX-zoo classifiers, including the MNIST/Fashion *full* models instead of hybrid CPU suffixes | L |
| ✅ F2 | **1-D convolution** (`kernel_shape [k]` → `[1,k]`) — `normalize.py` promotes a rank-3 [N,C,L] graph to [N,C,1,L] in place (pads [a,b] -> [0,a,0,b]); `tests/test_dense_lowering.py`; `research/conv1d_suite` board-verified `PASS: 12 models, 192 inferences, 38464 exact output bytes` | First step towards audio models; the front end currently rejects rank-3 input | M |
| ✅ F3 | **Calibration parity**: document which profiles accept `calibration_ranges` and add it to the chain/join profiles that lack it — `docs/calibration-cookbook.md` now carries one row per `scheduler.DISPATCH_PROFILES` (33 profiles) with the required measured tensors and the exact rejection messages; the band contract moved into `calibration.measured_range` and is threaded through chain_n, join-chain, diamond, join_dag, pool_join, depthwise_join and join-walk; `tests/test_calibration_parity.py` pins the table against the dispatch set; `research/chain_calibration_suite` is board-verified (`PASS: 6 models, 96 inferences, 18432 exact output bytes`) and shows float MAE 6.96->0.72, 672.8->33.8, 53161.9->251.6 for 3/4/5-layer chains | Real models need measured bands; the walk has it, several profiles do not | M |
| F4 | **`chain`/`chain_n` border zero point** (`0x1184` left at −128) | 178 of 242 retained chain models lose accuracy versus float; needs new board evidence for the whole family | L |
| ✅ F5 | **Walk coverage**: elementwise ops inside a chain and multi-join DAGs — the walk now lowers `Add|Sub|Max|Mul(constant)` stages inside a chain, reusing the verified elementwise register emitter; `tests/test_walk_elementwise.py` and `research/walk_elementwise_suite` board-verified (`PASS: 12 v5 models, 192 inferences, 27648 exact output bytes`) | Removes the "profile-matched" caveats from the docs | L |
| ✅ F6 | **`Concat`/`Slice`/`Resize`/`Softmax`/`ReduceMean`** at least for the bounded cases the hardware supports — `GlobalAveragePool`/`ReduceMean(axes=[2,3])` on 8x8 lower to three chained 2x2 pools (the reduction emitter) and `Concat(axis=1)` of sibling Conv branches on the same input is lowered to one wide Conv by stacking weights; `Softmax`, `Slice` and `Resize` stay rejected with their exact messages (`tests/test_bounded_ops.py`). Board: `global_pool_suite` PASS 12/192/1,248 (v5 chain-walk), `global_pool_reduce_suite` PASS 12/192/1,408 (v3 pooling-sequence), `wide_concat_suite` PASS 12/192/270,336 | Detection/segmentation heads, classification tails | L each |
| ✅ F7 | **Rectangular kernels with one-sided padding; K > 31** — one-sided/asymmetric padding was found already expressible (the emitter's explicit-pad path, bounded `pads < K`); `research/rect_pad_suite` covers one-sided top/bottom/left/right, asymmetric pairs, K5 and rectangular K1xK3, board-verified (`PASS: 12 models, 192 inferences, 30144 exact output bytes`). K > 31 keeps the measured `native Conv supports odd K1..31` bound (a 33-tap kernel cannot be split without overlap-add); `tests/test_bounded_ops.py` pins both | Unblocks STFT-style and large-kernel models | M each |
| ✅ F8 | **dma-buf / zero-copy input** (V4L2/ISP → NPU) — `ornpu_open_shared` (arena from the Rockchip CMA heap, imported with `CREATE` flag `0x80`), `ornpu_input_view` (offset, row stride, geometry) and `ornpu_run_prefilled` (no input copy; fills only the stride padding). Board evidence with `tests/board_shared.c`: the whole packed `add_geometry_suite` — 32 models / 64 inferences / 19,968 exact bytes, 0 mismatches; `-ENOTSUP` for native16 and two-input legacy layouts (`docs/c-api.md`, `docs/investigation-log.md`, `tests/test_runtime_shared.py`) | The board is a camera SoC; the real application is a camera pipeline | L |
| ✅ F9 | **Per-job timing / profiling API** — the runtime now exposes `ornpu_run_timed`/`ornpu_run_io_timed` (`struct ornpu_timing { pack_ns, submit_ns, readback_ns, total_ns }`), the untimed calls are `NULL` wrappers, and `tests/board_timed.c` reports the breakdown with exactness. Board-recorded minima on `walk_chain_suite/model000` (16/64 runs, 768/3,072 exact bytes, 0 mismatches): min `submit_ns` 80.5 µs / 62.1 µs, min `total_ns` 151.1 µs / 112.3 µs (`docs/performance.md`, `docs/c-api.md`; `tests/test_runtime_timing.py` pins the ABI) | The board has no userspace cycle counter; users need a supported way to measure | M |
| F10 | **C>128 channel-split accumulation** | Would need the undocumented partial-sum mechanism; keep as a research item | XL |
| ✅ F11 | **Mutable-parameter (v4) workflow**: a supported way to update weights/constants between inferences, with a helper API — `open_rknpu.mutable` (`compile_mutable`, `constant_regions`, `constant_payload`, `graft_region`, `program_bytes`, `replace_constant`) with `tests/test_mutable_api.py` and `examples/cookbook/08_mutable_api.py` | v4 descriptors exist but are only reachable through raw containers | M |
| ✅ F12 | **Stable public API surface**: mark which modules are supported (`open_rknpu.scheduler`, `calibration`, `sequence`, `compose`?) and version the container format promise — `docs/api-stability.md` (supported vs internal, exact signatures, the `meta`-key and container-format promises, the 0.x deprecation policy) plus `tests/test_public_api.py`, which pins the names and signatures | Users need to know what will not break | M |
| ✅ F13 | **Depthwise `ConvTranspose` integer reference** and the other reference gaps the test expansion documented (`pooling`, `transposed` off-centre taps) — `transposed.transposed_reference`, `pooling.pool_reference`, `reduction.reduction_reference`, `network.network_reference`, all replayed byte-for-byte against the retained board evidence (91 transposed models / 1,016 cases, pool_api 256, scheduled_pool 96, reduction_api 256, network_suite 896); `tests/test_reference_coverage.py` pins the inventory. The two remaining container-only paths (`two_head`, `spatial_reshape`) and the ONNX-less `transpose_stem_debug_suite` are documented as out of scope | Users cannot self-verify those profiles today | M |
| ✅ F14 | **Documented blocked features** (clock scaling, fences, IOMMU, SRAM) as a first-class "known limitations" page rather than log entries | Sets expectations and stops repeat questions | S |

### Tests

| # | Item | Value | Effort |
| --- | --- | --- | --- |
| ✅ T1 | **Examples under test**: run `examples/*/build.py`, `sanity.py`, `verify.py`, `accuracy.py` (host paths only, tiny fixtures) | 27 example entry points are untested; they are the first thing a user runs | M |
| ✅ T2 | **Installed-CLI test** via subprocess (`open-rknpu compile …`) rather than only in-process `cli.main()` | Proves the console-script entry point works | S |
| ✅ T3 | **C API/ABI tests on the host**: struct `_Static_assert`s, encode/decode parity with the C loader, `tests/board_core.c` compiled and run on x86 | The loader is currently only compiled for ARM and exercised on hardware | M |
| ✅ T4 | **Malformed-container fuzzing** of the C loader (structure-aware, seeded corpora) plus `hypothesis` properties for the encoder/decoder and the parsers | Container parsing is the highest-risk attack surface | M |
| ✅ T5 | **ASAN/UBSAN build of the runtime in CI** | Memory safety in the shipped C | S |
| ✅ T6 | **Close the remaining 220 uncovered lines** or justify them (weakest: `sequence.py`/`walk.py`/`liveness.py` 92 %, `elementwise.py` 93 %, `compose.py` 94 %) | 96 % is good; the last points are in the container writer and allocator | M |
| ✅ T7 | **Mutation testing** (`mutmut`) on the emitters and the container writer | 873 tests make this affordable; measures test *strength*, not coverage | M |
| ✅ T8 | **Cross-Python matrix**: 3.10–3.13, and a `numpy` min/max version job | The compiler is pure Python; users will hit version skew | S |
| ✅ T9 | **Reproducible-build check**: build the wheel twice, compare; assert sdist contents | Distribution integrity | S |
| ✅ T10 | **Docs-command tests**: execute the reproduce commands from the suite READMEs and the example READMEs in a temp tree | Documentation drift is currently only caught for links | M |
| ✅ T11 | **Performance regression harness** (host compile time; board numbers recorded, not gated) | Compile-time regressions are silent today | M |
| ✅ T12 | **Evidence-integrity tests for all suites**: checksum/manifest cross-checks beyond the sampled replay | The ledger test covers counts; deeper integrity is sampled | S |
| ✅ T13 | **`tests/board_*.c` host compile + the `runtime/main.c` CLI** | Same as T3, for the user-facing runner | S |
| ✅ T14 | **Fetch-script tests** (URL pins, sha256 mismatch handling) without network — `tests/test_fetch_scripts.py` covers the three fetchers offline (pins, accept, exact mismatch/connection messages, no-socket assertion); `research/fetch_toolchain.py` now verifies every downloaded file against its pinned git blob id | Keeps datasets reproducible | S |

### Documentation

| # | Item | Value | Effort |
| --- | --- | --- | --- |
| ✅ D1 | **Register/ISA reference**: the 126 task registers with meaning, direction and the value each profile writes (derived from `register_profile.py` + the log) | The single most requested artefact for anyone extending the compiler | L |
| ✅ D2 | **ONNX → profile support matrix** as one generated table (op, attributes, bounds, profile, example, evidence) | Answers "will my model compile?" at a glance | M |
| ✅ D3 | **Error-message index / troubleshooting FAQ**: every rejection string with its cause and fix | Users hit these constantly | M |
| ✅ D4 | **Glossary** (CNA, DPU, ERDMA, native16, band, zero point, tail control, engine run, arena, …) | Onboarding | S |
| ✅ D5 | **Performance guide**: methodology, per-family costs, latency tables, how to measure on your own board | Users need to size their models | M |
| ✅ D6 | **C API guide** ("using the runtime from C"): lifecycle, packing rules, error codes, the board runner | The header is commented but there is no narrative | M |
| ✅ D7 | **Annotated container walkthrough** (hexdump with offsets, field by field) | Validates the format doc and helps third-party writers | S |
| ✅ D8 | **Calibration cookbook**: method choice, percentiles, accuracy measurement, the failure modes seen in practice | Calibration is where real accuracy is won or lost | S |
| ✅ D9 | **Board bring-up + runbook**: wiring/power/storage limits, `rkipc`, recovery from a wedged NPU, space budgeting | Only partly covered in `docs/board-access.md` | S |
| ✅ D10 | **Clean-room/legal statement** (L3) and **release/versioning policy** (semver, container compatibility, deprecation) | Publication prerequisites | S each |
| ✅ D11 | **`CHANGELOG.md`**, **`SECURITY.md`**, **`CODE_OF_CONDUCT.md`**, **`CITATION.cff`**, **`AUTHORS`** | Standard OSS expectations; cheap to add | S each |
| ✅ D12 | **Docs site** (mkdocs-material) + generated API reference (docstrings are good) + badges — config and deploy workflow are in-tree; enabling GitHub Pages is a repository-settings step | Discoverability; GitHub-only docs do not get indexed well | M |
| ✅ D13 | **Migration/compat notes for container v1–v5** (`docs/container-migration.md`) | Third-party writers need them | S |
| ✅ D14 | **FAQ: "why is my model rejected?"** built from D3 | Support load | S |

### Examples

| # | Item | Value | Effort |
| --- | --- | --- | --- |
| ✅ E1 | **Minimal C example** ("hello NPU": open, run, print) for host and board | The C API has no small standalone example | S |
| ✅ E2 | **Quantized import** (`QLinearConv`, DQ→Conv→Q) end to end | The profile exists and is board-verified, but no example uses it | M |
| ✅ E3 | **Mutable parameters (v4)**: update weights/constants between inferences | Same: verified, undocumented in practice | M |
| ✅ E4 | **Batched + pipelined throughput** demo with a measured comparison | The 1.5–2.25× pipeline result deserves a runnable example | M |
| E5 | **Camera/V4L2 → NPU** pipeline | The board's actual purpose; currently no example | L |
| ✅ E6 | **Wheel-installed usage** (install from the wheel, compile, inspect) | Proves the distribution works outside the checkout | S |
| ✅ E7 | **Depthwise-separable classifier** (depthwise + pointwise + pool) — `examples/depthwise_separable/` (chain-walk, 8x8x3 -> 4x4x4, 4 tasks) with `tests/test_example_depthwise_separable.py`; board run `cases=8 inferences=8 exact_bytes=512 mismatches=0` | The most common mobile block, currently only implied | M |
| ✅ E8 | **ONNX-zoo compatibility report**: run a set of small zoo models, record accept/reject and why | The best possible answer to "will it run my model?" | M |
| ✅ E9 | **Benchmark harness example** (N models, latency/throughput table) — `examples/benchmark/` (`bench.py --board --runner bench --stat min`) with `tests/test_example_benchmark.py`; four previously empty board rows now measured by the harness (chain_suite 0.026 ms, sequence_suite 0.078 ms, two_head_suite 0.058 ms, walk_chain_suite 0.044 ms, min of 5) | Reusable by users | M |
| ✅ E10 | **Troubleshooting example**: feed an unsupported graph, show how to read the rejection and what to do | Turns error messages into a teaching moment | S |
| ✅ E11 | **Notebook/Colab walkthrough** for the host path | Lowers the barrier for the library audience | M |
| E12 | **Audio VAD** (the Silero study lists exactly what is missing: 1-D conv, C129, LSTM, dynamic shapes) | High-visibility application; gated on F2/F6 | L |
| ✅ E13 | **Multi-model scheduling example** (two containers alternating, shared arena lessons) — `examples/multi_model/` with `tests/test_example_multi_model.py`; board run `PASS: 2 models, 800 inferences, 1843200 exact bytes, one open per model` (A 2.080 ms, B 4.987 ms, total 3.534 ms per inference) | Shows the runtime's per-model lifecycle | S |

### CI, automation, community

| # | Item | Value | Effort |
| --- | --- | --- | --- |
| ✅ A1 | **Release workflow**: tag → build → `twine check` → PyPI trusted publishing → GitHub release with artifacts | Repeatable releases | M |
| ✅ A2 | **Docs-site deploy workflow** | Pairs with D12 | S |
| ✅ A3 | **Issue/PR templates**, `SUPPORT.md`, discussions | Directs the first wave of questions | S |
| ✅ A4 | **Dependabot** for GitHub Actions, `pre-commit` (ruff), `.editorconfig`, `.gitattributes` | Repository hygiene | S |
| ✅ A5 | **Branch protection + required checks documentation** — `docs/branch-protection.md`: the required contexts, the ruleset JSON, the failure playbook and the local reproduction of every gate; `docs.yml` gained a `pull_request` trigger so the strict site build can be a required check | Keeps the baseline/coverage gates honest | S |
| ✅ A6 | **Evidence storage decision**: 16 k files / 38 MB pack. Document it, or move the per-suite artifacts to release assets and keep the manifests/READMEs in-tree | Clone weight and GitHub limits | M |
| A7 | **Signed tags / build provenance** (later: SLSA, Sigstore) | Supply-chain expectations | M |
| ✅ A8 | **Code of conduct, security policy, support policy, contributor list, citation** (same as D11) | Community baseline | S |

---

## P2 — after the first release

* INT16/FP16 or other precisions; per-channel output conversion and native spatial
  broadcast (both closed as *measured negatives* — only revisit with new evidence);
  non-power-of-two LUT bands (same).
* Dynamic shapes and batch > 16 (recompile-per-shape is the current answer).
* RNN/GRU/LSTM (likely blocked by the hardware, not by the compiler).
* Multi-stage detection heads (upsample/concat/anchors).
* Channel-split accumulation for C > 128.
* A vendor-zoo detector trained end to end (needs a labelled dataset and a training budget).
* A second RV1106-class board to generalise the SoC-level claim (currently scoped to the
  attached RV1103).
* GPU-less CI performance benchmarks; long-run soak tests on the board.
* Localised documentation (at least a second language) if the audience warrants it.

---

## Definition of done for the first public release (the short gate)

1. L1–L3 done: no mis-stated licensing, third-party notices, provenance statement.
2. I1–I4 done: no silently-ignored flags, `encode(decode(x))` round-trips, reference gaps
   documented or closed, dead code removed.
3. C1–C3 done: CI compiles the C, installs the wheel in a clean venv, and validates the
   distribution metadata. C4 done: version claims consistent.
4. R1–R2 done (real metadata, `0.1.0`); R3: claim the PyPI name (`open-rknpu`, verified
   unclaimed on 2026-09-12) from the maintainer's account.
5. `make lint test coverage primitives docs-check baseline campaign evidence perf
   reproducible host-c` green on a fresh clone, plus a manual wheel-install smoke test.
6. `CHANGELOG.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff` added (D11).
7. Tag `v0.1.0`, publish, then verify: `pip install open-rknpu` in a clean venv compiles and
   inspects a model.
