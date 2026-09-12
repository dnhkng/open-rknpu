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

Current baseline: 873 tests, 96 % compiler line coverage (floor 95 %), 2,244-model container
baseline, 121-row board ledger, 27 documentation pages (~52 k words), 4 example sets,
16,313 tracked files / 38 MB pack.

## Status after the 2026-09-11 pass

Everything marked ✅ below is in the tree and verified by the test suite; the rest is the
open list. Counted by category: **P0 legal 6/6 · P0 interfaces 4/4 · P0 CI 4/4 · P0 release
2/3 · P1 tests 7/14 · P1 documentation 12/14 · P1 examples 6/13 · P1 automation 7/8 ·
P1 features 1/14**. That leaves **37 open rows**: 13 features, 9 tests, 4 documentation,
6 examples, 3 automation and the 2 metadata placeholders that only the maintainer can fill.
The feature rows are the deliberately untouched part — each needs a probe and fresh board
evidence, not a checklist pass.

What this pass changed: the GPL-2.0 sources are gone with provenance retained;
`THIRD_PARTY.md`, `docs/provenance.md` and the trademark note are in; `--target/--quantize`
are validated and reported; `model.decode` round-trips; the Clip reference models the
hardware clamp; dead code is gone; CI now compiles the C, installs the wheel in a clean venv
and runs `twine check`, over Python 3.10–3.13; the register reference and error index are
generated with drift guards; support matrix, glossary, troubleshooting, performance, C API
and container walkthrough docs are written; container/emitter fuzzing, example and CLI-flag
tests landed; and the cookbook examples (C, quantized import, mutable parameters,
batched/pipelined, wheel-installed, troubleshooting, zoo compatibility) run green.

---

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
| R3 | Confirm/claim the PyPI name (`open-rknpu`); the GitHub repository URL is fixed to `github.com/dnhkng/open-rknpu` | Last-minute name collisions are avoidable | S |

---

## P1 — strongly recommended before announcing

### Features

| # | Item | Value | Effort |
| --- | --- | --- | --- |
| F1 | **`MatMul`/`Gemm`** (via 1×1 Conv lowering) | Unlocks most ONNX-zoo classifiers, including the MNIST/Fashion *full* models instead of hybrid CPU suffixes | L |
| F2 | **1-D convolution** (`kernel_shape [k]` → `[1,k]`) | First step towards audio models; the front end currently rejects rank-3 input | M |
| F3 | **Calibration parity**: document which profiles accept `calibration_ranges` and add it to the chain/join profiles that lack it | Real models need measured bands; the walk has it, several profiles do not | M |
| F4 | **`chain`/`chain_n` border zero point** (`0x1184` left at −128) | 178 of 242 retained chain models lose accuracy versus float; needs new board evidence for the whole family | L |
| F5 | **Walk coverage**: elementwise ops inside a chain and multi-join DAGs | Removes the "profile-matched" caveats from the docs | L |
| F6 | **`Concat`/`Slice`/`Resize`/`Softmax`/`ReduceMean`** at least for the bounded cases the hardware supports | Detection/segmentation heads, classification tails | L each |
| F7 | **Rectangular kernels with one-sided padding; K > 31** | Unblocks STFT-style and large-kernel models | M each |
| F8 | **dma-buf / zero-copy input** (V4L2/ISP → NPU) | The board is a camera SoC; the real application is a camera pipeline | L |
| F9 | **Per-job timing / profiling API** | The board has no userspace cycle counter; users need a supported way to measure | M |
| F10 | **C>128 channel-split accumulation** | Would need the undocumented partial-sum mechanism; keep as a research item | XL |
| F11 | **Mutable-parameter (v4) workflow**: a supported way to update weights/constants between inferences, with a helper API | v4 descriptors exist but are only reachable through raw containers | M |
| F12 | **Stable public API surface**: mark which modules are supported (`open_rknpu.scheduler`, `calibration`, `sequence`, `compose`?) and version the container format promise | Users need to know what will not break | M |
| F13 | **Depthwise `ConvTranspose` integer reference** and the other reference gaps the test expansion documented (`pooling`, `transposed` off-centre taps) | Users cannot self-verify those profiles today | M |
| ✅ F14 | **Documented blocked features** (clock scaling, fences, IOMMU, SRAM) as a first-class "known limitations" page rather than log entries | Sets expectations and stops repeat questions | S |

### Tests

| # | Item | Value | Effort |
| --- | --- | --- | --- |
| ✅ T1 | **Examples under test**: run `examples/*/build.py`, `sanity.py`, `verify.py`, `accuracy.py` (host paths only, tiny fixtures) | 27 example entry points are untested; they are the first thing a user runs | M |
| ✅ T2 | **Installed-CLI test** via subprocess (`open-rknpu compile …`) rather than only in-process `cli.main()` | Proves the console-script entry point works | S |
| T3 | **C API/ABI tests on the host**: struct `_Static_assert`s, encode/decode parity with the C loader, `tests/board_core.c` compiled and run on x86 | The loader is currently only compiled for ARM and exercised on hardware | M |
| ✅ T4 | **Malformed-container fuzzing** of the C loader (structure-aware, seeded corpora) plus `hypothesis` properties for the encoder/decoder and the parsers | Container parsing is the highest-risk attack surface | M |
| T5 | **ASAN/UBSAN build of the runtime in CI** | Memory safety in the shipped C | S |
| ✅ T6 | **Close the remaining 220 uncovered lines** or justify them (weakest: `sequence.py`/`walk.py`/`liveness.py` 92 %, `elementwise.py` 93 %, `compose.py` 94 %) | 96 % is good; the last points are in the container writer and allocator | M |
| T7 | **Mutation testing** (`mutmut`) on the emitters and the container writer | 873 tests make this affordable; measures test *strength*, not coverage | M |
| ✅ T8 | **Cross-Python matrix**: 3.10–3.13, and a `numpy` min/max version job | The compiler is pure Python; users will hit version skew | S |
| T9 | **Reproducible-build check**: build the wheel twice, compare; assert sdist contents | Distribution integrity | S |
| T10 | **Docs-command tests**: execute the reproduce commands from the suite READMEs and the example READMEs in a temp tree | Documentation drift is currently only caught for links | M |
| T11 | **Performance regression harness** (host compile time; board numbers recorded, not gated) | Compile-time regressions are silent today | M |
| T12 | **Evidence-integrity tests for all suites**: checksum/manifest cross-checks beyond the sampled replay | The ledger test covers counts; deeper integrity is sampled | S |
| T13 | **`tests/board_*.c` host compile + the `runtime/main.c` CLI** | Same as T3, for the user-facing runner | S |
| T14 | **Fetch-script tests** (URL pins, sha256 mismatch handling) without network | Keeps datasets reproducible | S |

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
| D8 | **Calibration cookbook**: method choice, percentiles, accuracy measurement, the failure modes seen in practice | Calibration is where real accuracy is won or lost | S |
| D9 | **Board bring-up + runbook**: wiring/power/storage limits, `rkipc`, recovery from a wedged NPU, space budgeting | Only partly covered in `docs/board-access.md` | S |
| ✅ D10 | **Clean-room/legal statement** (L3) and **release/versioning policy** (semver, container compatibility, deprecation) | Publication prerequisites | S each |
| ✅ D11 | **`CHANGELOG.md`**, **`SECURITY.md`**, **`CODE_OF_CONDUCT.md`**, **`CITATION.cff`**, **`AUTHORS`** | Standard OSS expectations; cheap to add | S each |
| D12 | **Docs site** (mkdocs-material) + generated API reference (docstrings are good) + badges | Discoverability; GitHub-only docs do not get indexed well | M |
| D13 | **Migration/compat notes for container v1–v5** | Third-party writers need them | S |
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
| E7 | **Depthwise-separable classifier** (depthwise + pointwise + pool) | The most common mobile block, currently only implied | M |
| ✅ E8 | **ONNX-zoo compatibility report**: run a set of small zoo models, record accept/reject and why | The best possible answer to "will it run my model?" | M |
| E9 | **Benchmark harness example** (N models, latency/throughput table) | Reusable by users | M |
| ✅ E10 | **Troubleshooting example**: feed an unsupported graph, show how to read the rejection and what to do | Turns error messages into a teaching moment | S |
| E11 | **Notebook/Colab walkthrough** for the host path | Lowers the barrier for the library audience | M |
| E12 | **Audio VAD** (the Silero study lists exactly what is missing: 1-D conv, C129, LSTM, dynamic shapes) | High-visibility application; gated on F2/F6 | L |
| E13 | **Multi-model scheduling example** (two containers alternating, shared arena lessons) | Shows the runtime's per-model lifecycle | S |

### CI, automation, community

| # | Item | Value | Effort |
| --- | --- | --- | --- |
| ✅ A1 | **Release workflow**: tag → build → `twine check` → PyPI trusted publishing → GitHub release with artifacts | Repeatable releases | M |
| ✅ A2 | **Docs-site deploy workflow** | Pairs with D12 | S |
| ✅ A3 | **Issue/PR templates**, `SUPPORT.md`, discussions | Directs the first wave of questions | S |
| ✅ A4 | **Dependabot** for GitHub Actions, `pre-commit` (ruff), `.editorconfig`, `.gitattributes` | Repository hygiene | S |
| A5 | **Branch protection + required checks documentation** | Keeps the baseline/coverage gates honest | S |
| A6 | **Evidence storage decision**: 16 k files / 38 MB pack. Document it, or move the per-suite artifacts to release assets and keep the manifests/READMEs in-tree | Clone weight and GitHub limits | M |
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
4. R1–R3 done: real metadata, `0.1.0`, name reserved.
5. `make lint test coverage primitives docs-check baseline campaign` green on a fresh clone,
   plus a manual wheel-install smoke test.
6. `CHANGELOG.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff` added (D11).
7. Tag `v0.1.0`, publish, then verify: `pip install open-rknpu` in a clean venv compiles and
   inspects a model.
