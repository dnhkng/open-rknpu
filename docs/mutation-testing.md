# Mutation testing report

Status: host-only measurement, no board involved. Driver: `research/run_mutation_tests.py`.
Scope: `research/mutation_scope.json`. Date of measurement: 2026-09-12.

Coverage says which lines run. It does not say whether any test would fail if a line
did something different. This report measures that for seven high-value compiler
modules with a bounded, deterministic, single-edit mutator, and records which mutants
survive and why.

## TL;DR

| Module | Tests in subset | Selected | Killed | Survived | Timeout | Error | Subset score |
|---|---|---:|---:|---:|---:|---:|---:|
| `quantization.py` | 3 | 10 | 7 | 3 | 0 | 0 | **70.0%** |
| `sequence.py` | 2 | 10 | 5 | 5 | 0 | 0 | **50.0%** |
| `compose.py` | 1 | 10 | 4 | 6 | 0 | 0 | **40.0%** |
| `walk.py` | 2 | 10 | 5 | 5 | 0 | 0 | **50.0%** |
| `native.py` | 2 | 10 | 4 | 6 | 0 | 0 | **40.0%** |
| `padding.py` | 1 | 10 | 4 | 6 | 0 | 0 | **40.0%** |
| `normalize.py` | 2 | 10 | 8 | 2 | 0 | 0 | **80.0%** |
| **Total** | | **70** | **37** | **33** | **0** | **0** | **52.9%** |

Mutation score = `(killed + timeout) / (killed + timeout + survived)`; a timeout counts as
killed-by-timeout. Errors (a broken pytest run rather than a test failure) are excluded and
there were none.

33 survivors is the useful result, and the per-survivor assessment below splits them into:

* **1 genuinely equivalent mutant** — no observable behaviour change; the score is penalised
  for it and it should be suppressed.
* **12 subset gaps** — the configured per-module test subset misses them, but the rest of the
  tracked suite kills them (verified by re-running each against the other test files that
  import the module). These say the `mutation_scope.json` mapping is too narrow, not that the
  suite is blind.
* **20 suite-wide gaps** — neither the configured subset nor the rest of the tracked suite
  kills them. These are real behavioural blind spots (mostly boundary and negative-path cases).

Crediting the 12 subset gaps, the score against the whole tracked suite is **49/70 = 70.0%**.

## What was mutated

`research/mutation_scope.json` maps each module to the test subset that must detect wrong
behaviour in it (the driver reads this file; `seed` fixes the selection):

| Module | Test subset |
|---|---|
| `quantization.py` | `tests/test_quantization_edge_cases.py`, `tests/test_native_clip_reference.py`, `tests/test_emit_semantics.py` |
| `sequence.py` | `tests/test_sequence_roundtrip.py`, `tests/test_container_fuzz.py` |
| `compose.py` | `tests/test_composer_arena.py` |
| `walk.py` | `tests/test_walk.py`, `tests/test_emitter_fuzz.py` |
| `native.py` | `tests/test_native_clip_reference.py`, `tests/test_emit_semantics.py` |
| `padding.py` | `tests/test_padding.py` |
| `normalize.py` | `tests/test_normalize.py`, `tests/test_normalize_matrix.py` |

`quantization.py` is probed against both its dedicated edge-case suite and the native
clip/emit suites, because `native.py` consumes its multiplier/shift/rounding arithmetic, so
the two mappings were unioned into one entry.

The mutator applies exactly one single-edit mutation per mutant, found by AST walk:

| Kind | Edit |
|---|---|
| operator | `+` ↔ `-` |
| comparison | `<` ↔ `<=` |
| boolean op | `and` ↔ `or` (every adjacent pair of a boolean expression) |
| constant | integer `n` → `n + 1`; `True` ↔ `False` |
| return negation | `return X` → `return not (X)` |

There are many more candidates than the 10-per-module cap (e.g. `walk.py` has 624), so the
driver selects deterministically with a fixed seed and stratified round-robin across the five
kinds. That prevents the hundreds of integer literals from crowding out operator, comparison,
boolean and return mutants.

### Why not `mutmut`

`mutmut` 3.7.0 installs cleanly into the task interpreter, but it was rejected for this job:

* its per-mutant test selection is one global `pytest_add_cli_args_test_selection` list, not a
  per-module subset, so the module → test-subset mapping in `mutation_scope.json` cannot be
  expressed;
* it requires configuration in `pyproject.toml`/`setup.cfg` (out of bounds for this task) and
  materialises a `mutants/` directory inside the repository;
* its generated mutant set/order is not seed-controlled, so two runs are not guaranteed to
  select the same mutants.

The driver implements the required operators directly and keeps the whole run inside a
temporary directory.

## Reproducing

Run everything from the repository root. `python` below is an interpreter with the project
dependencies installed (NumPy, ONNX, coverage, pytest) plus `PYTHONPATH=src`.

```bash
cd /path/to/open-rknpu

# Full run (about 63 s, 70 mutants, fixed seed -> identical selection every time)
PYTHONPATH=src python research/run_mutation_tests.py --json /tmp/mutation.json

# Bounded smoke run (about 36 s, 4 mutants per module, under two minutes)
PYTHONPATH=src python research/run_mutation_tests.py --quick

# One module, more mutants, explicit seed
PYTHONPATH=src python research/run_mutation_tests.py --modules compose --max-mutants 20 --seed 20240517

# Show the selected mutants without running pytest
PYTHONPATH=src python research/run_mutation_tests.py --dry-run

# CI gate: exit non-zero below 50%
PYTHONPATH=src python research/run_mutation_tests.py --json /tmp/mutation.json --fail-under 50
```

Flags: `--modules` (comma/space list of module names, with or without `.py`, or `all`),
`--max-mutants`, `--timeout`, `--quick`,
plus `--budget-seconds`, `--seed`, `--dry-run`, `--json`, `--fail-under`, `--scope`.

## Bounded, deterministic, and safe

* **Bounded.** Each mutant runs under `pytest -x` with a per-mutant `--timeout` (default 30 s)
  and the run has a wall-clock `--budget-seconds` (default 240 s). `--quick` caps mutants at 4
  per module, timeout at 20 s and budget at 90 s; the measured full-scope `--quick` run took
  **36.4 s**. Mutants still unrun when the budget expires are reported as `skipped`, never
  silently dropped. A hanging mutant is killed and counted as `timeout` (killed-by-timeout).
* **Deterministic.** Candidates are collected by AST walk in source order, shuffled per kind
  with `random.Random(seed)`, drawn round-robin, then sorted by `(line, col)`. Two identical
  `--dry-run` invocations produce byte-identical selections (verified: only the temporary
  workdir path differs).
* **Never touches the working tree.** All mutants are written into a `copytree` of
  `src/open_rknpu` under a `tempfile.mkdtemp()` directory, with `PYTHONPATH` pointed at the
  copy. The original package is never opened for writing. The driver takes a SHA-256 manifest
  of every `src/open_rknpu/*.py` before and after the run and aborts if they differ; the run
  printed `source tree unchanged after run: True`. The copy is removed in a `finally` block.
  pytest runs with `-p no:cacheprovider` and `PYTHONDONTWRITEBYTECODE=1` so no cache or
  bytecode is written into the checkout.

The harness itself was spot-checked: a padding mutant that survived was re-applied by hand in
an isolated copy, its mutated module was confirmed to be the one imported, and the test still
passed — so the survivors are real, not a `sys.path` accident.

## Runtime

* Full run: **62.7 s** wall (70 mutants; 37 killed, 33 survived), budget 240 s.
* `--quick` full scope: **36.4 s** wall, budget 90 s.
* Per module: `sequence.py` 28.5 s (its subset includes the 5 s container fuzz), `walk.py`
  6.4 s, `native.py` 4.7 s, `quantization.py` 4.2 s, `compose.py` 3.7 s, `normalize.py` 3.6 s,
  `padding.py` 3.1 s.

## Working-tree cleanliness

The driver adds no files to the checkout: every mutant is applied in a `copytree` copy of
`src/open_rknpu` under `mkdtemp()`, and it prints a SHA-256 manifest of the real tree before
and after the run (`source tree unchanged after run: True`). A run therefore leaves
`git status --short` untouched - only the three files that make up the harness are new
(`research/run_mutation_tests.py`, `research/mutation_scope.json`, this page).

## Surviving mutants

Location is `file:line:col` in `src/open_rknpu`. "Rest of suite" is the result of re-running
the same mutant against every other tracked test file that imports the module (excluding the
configured subset); `killed` there means the configured mapping is too narrow.

### `quantization.py` — 7/10 killed (70.0%)

| Location | Mutation | Rest of suite | Assessment |
|---|---|---|---|
| `quantization.py:58:20` | `qbias < -(1<<31)` → `<=` | survived | Test gap (boundary). The guard allows an INT32 minimum of exactly `-2**31`; the mutant rejects it. No test builds a bias that lands exactly on that value. |
| `quantization.py:62:34` | `or` → `and` in the accumulator-overflow guard | survived | Test gap. One-sided accumulator overflow no longer raises. No test drives the accumulator out of INT32 range; the guard is unexercised. |
| `quantization.py:72:57` | `or` → `and` in the output-quantization validation | survived | Test gap (negative path). A non-finite/invalid `output_scale` can now slip through validation. No test passes an explicit invalid output scale. |

### `sequence.py` — 5/10 killed (50.0%)

| Location | Mutation | Rest of suite | Assessment |
|---|---|---|---|
| `sequence.py:108:54` | default `input_zero_point=0` → `1` | survived | Test gap (default value). Callers always pass `input_zero_point` explicitly, so the default is never exercised. |
| `sequence.py:121:20` | `or` → `and` in constant-descriptor validation | survived | Test gap (negative path). An empty constant name becomes acceptable. No test encodes an empty name. |
| `sequence.py:160:53` | `offset+(amount+4)*8` → `offset-(...)` in the descriptor bounds check | survived | Test gap. The bounds check becomes trivially true; no test encodes an over-large task descriptor. |
| `sequence.py:232:46` | `or` → `and` in the stride/count shape check | **killed** | Subset gap. `decode_sequence`'s invalid-stride branch is covered by other suite files, not by the two files in the subset. Widen the mapping. |
| `sequence.py:253:76` | `amount<=256` → `<256` | survived | Test gap (boundary). The documented 256-word DPU fetch maximum is never encoded/decoded at exactly 256. |

### `compose.py` — 4/10 killed (40.0%)

| Location | Mutation | Rest of suite | Assessment |
|---|---|---|---|
| `compose.py:62:31` | register `0x6070` → `0x6071` in the `POOL` family table | **killed** | Subset gap. `test_composer_arena.py` never calls `derive_bindings`; the family register tables are checked by `test_compose.py`/`test_container_fuzz.py`. |
| `compose.py:170:28` | `len(inputs) <= 8` → `< 8` | survived | Test gap (boundary). The 8-input maximum is never exercised. |
| `compose.py:365:25` | `or` → `and` in the `batched_layout` last-task check | **killed** | Subset gap. Covered by `test_submission.py`/`test_tiled_chain.py`, which are not in the subset. |
| `compose.py:377:20` | `return X` → `return not (X)` on the `batched_layout` error message | **killed** | Subset gap. `test_submission.py` asserts the message text; the subset does not call the function at all. |
| `compose.py:404:55` | `base + index * 8` → `base - index * 8` in `derive_bindings` | **killed** | Subset gap. The read-back offsets are asserted by `test_compose.py`/`test_container_fuzz.py`. |
| `compose.py:409:11` | `return out` → `return not (out)` in `derive_bindings` | **killed** | Subset gap. Same as above: the subset never imports `derive_bindings`. |

### `walk.py` — 5/10 killed (50.0%)

All five are suite-wide gaps: the configured subset and the rest of the tracked suite both
miss them.

| Location | Mutation | Rest of suite | Assessment |
|---|---|---|---|
| `walk.py:127:41` | `shape[2] < 2` → `<= 2` | survived | Test gap (boundary). A two-pixel-high walk input is accepted/rejected differently; the minimum input dimension is untested. |
| `walk.py:145:39` | `weight.shape[0] < 1` → `<= 1` | survived | Test gap (boundary). Single-output-channel (`O == 1`) convolution is not distinguished. |
| `walk.py:389:58` | `input_shape[2]` → `input_shape[3]` | survived | Test gap. Only square inputs are walked, so height and width are interchangeable and the mutant is indistinguishable. Add a non-square case. |
| `walk.py:472:49` | `(kernel + 1) // 2 + 1` → `- 1` | survived | Test gap. The feature-grain clamp is insensitive to this term for the shapes tested. |
| `walk.py:550:65` | `or` → `and` in the entry validation | survived | Test gap. Same class as line 127 for the second graph-entry check. |

### `native.py` — 4/10 killed (40.0%)

| Location | Mutation | Rest of suite | Assessment |
|---|---|---|---|
| `native.py:74:63` | default `input_zero_point=0` → `1` | **killed** | Subset gap. Other native tests exercise the default; the subset calls `compile_native_input` with an explicit zero point. |
| `native.py:91:8` | `or` → `and` in the native-Conv structural validation | survived | Test gap. One structural precondition is bypassed for a combination no test builds. |
| `native.py:100:57` | `or` → `and` in the pad/stride/dilation validation | **killed** | Subset gap. Covered by other native/profile tests. |
| `native.py:130:50` | `last=(oy1-1)*sy-pt+ekh-1` → `-ekh-1` | **killed** | Subset gap. Tiling/row-range coverage is asserted elsewhere. |
| `native.py:181:15` | `return X` → `return not (X)` (weight-plane byte offset) | **killed** | Subset gap. Covered where native weight packing is decoded byte-for-byte. |
| `native.py:192:49` | `lane * 2` → `lane * 3` (zero-point write offset) | **killed** | Subset gap. Same: the subset does not inspect the packed native constant block. |

### `padding.py` — 4/10 killed (40.0%)

No other tracked test file imports `open_rknpu.padding`, so every survivor here is suite-wide.

| Location | Mutation | Rest of suite | Assessment |
|---|---|---|---|
| `padding.py:17:60` | default `constant_values=0` → `1` | survived | Test gap (default). Tests pass `constant_values=7` explicitly or use non-constant modes. |
| `padding.py:30:30` | `pads[0]` → `pads[1]` in the NHWC spec | survived | Test gap. The rank-4 branch is only tested with symmetric pads `(1,1,1,1)`, where every index swap is indistinguishable. Add an asymmetric NHWC case (`test_padding.py:41-43`). |
| `padding.py:30:39` | `pads[2]` → `pads[3]` | survived | Test gap. Same root cause. |
| `padding.py:30:50` | `pads[1]` → `pads[2]` | survived | Test gap. Same root cause. |
| `padding.py:33:13` | `p < 0` → `p <= 0` | survived | Test gap. Zero padding is now rejected as if negative; no test passes a zero pad. |
| `padding.py:36:86` | `astype(..., copy=False)` → `copy=True` | survived | **Genuinely equivalent.** The returned array is element-wise and dtype-identical either way; only an internal copy differs, which is unobservable. Suppress this mutant. |

### `normalize.py` — 8/10 killed (80.0%)

| Location | Mutation | Rest of suite | Assessment |
|---|---|---|---|
| `normalize.py:51:29` | `(size-1)*d + 1` → `- 1` in the effective-kernel computation | **killed** | Subset gap. The dilated dense-rewrite path is exercised by `test_geometry_expansion.py`/`test_mode_expansion.py`, not by the two normalize files in the subset. |
| `normalize.py:155:30` | `1 <= kh` → `1 < kh` | survived | Test gap (boundary). The rectangular-kernel embedding branch with `kh == 1`, `kw != 1` (e.g. `1x3`) is never exercised. |

## Is a CI gate worth it, and at what threshold?

**Yes, but as a scheduled ratchet, not a per-PR blocker — and only after the mapping is
widened.** The current 52.9% is a useful signal precisely because the suite is not weak
overall; it is weak on boundaries, negative paths and validation guards, and those are cheap
to test. Two caveats argue against making it block PRs today:

1. 12 of 33 survivors are artefacts of a deliberately narrow test subset, not suite blindness.
   A gate would be measuring the mapping as much as the tests.
2. One survivor is a genuinely equivalent mutant. Mutation scores always carry a noise floor
   of equivalent mutants; a threshold set too close to the current value will produce flaky
   red builds for no behavioural reason.

Recommended path, mirroring the existing coverage ratchet in `pyproject.toml`
(`fail_under = 97`):

1. **Now:** run nightly + `workflow_dispatch` (not on PRs), publish the JSON artifact, and gate
   at `--fail-under 50`. That is just below the measured 52.9%, so it catches regressions
   without failing on known-equivalent noise.
2. **Then:** widen `mutation_scope.json` to add the covering files identified above
   (`compose.py` → `test_compose.py`, `test_submission.py`, `test_tiled_chain.py`;
   `native.py` → `test_geometry_expansion.py`, `test_profile_bounds.py`; `normalize.py` →
   `test_geometry_expansion.py`, `test_mode_expansion.py`; `sequence.py` →
   `test_submission.py`, `test_container_bindings.py`). With no new tests this lifts the
   measured score to about 70%; raise the gate to `--fail-under 65`.
3. **Target:** add the 20 suite-wide gap cases (mostly one-line boundary tests, listed by
   location above) and raise the gate to `--fail-under 80`, still nightly. Keep the per-PR job
   to `--quick` as an informational (non-blocking) annotation.

Once a threshold is stable, a per-module floor is more useful than a global one: `padding.py`
and `walk.py` should not be able to hide behind `normalize.py`'s 80%.

## Makefile and CI targets

`Makefile` targets (next to `coverage`; `research/run_mutation_tests.py` is also in
`MAINTAINED`, so `make lint` ruff-checks it):

```make
mutation:  ## mutation-test the highest-value compiler modules (bounded, seed-fixed)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) research/run_mutation_tests.py --json build/mutation.json

mutation-quick:  ## fast mutation smoke run (under two minutes)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) research/run_mutation_tests.py --quick
```

`.github/workflows/mutation.yml` runs the same bounded pass on a weekly schedule and on
`workflow_dispatch` (`--fail-under 50`), and uploads `build/mutation.json` as an artifact.
It is deliberately not a per-PR gate; run `make mutation-quick` locally for a fast smoke
check.

## Caveats

* This is first-order, single-edit mutation. It does not test interacting edits, and the
  operator set is deliberately small, so the absolute score is not comparable to a full
  `mutmut`/`cosmic-ray` run. It is comparable across runs of this driver.
* The 10-mutants-per-module cap is a bound, not a survey: increasing `--max-mutants` will find
  more survivors (the candidates number in the hundreds for `walk.py` and `native.py`).
* `--quick` uses a different (smaller) mutant sample than the full run, so its score is not
  directly comparable; use it for smoke checks only.
* The rest-of-suite re-check used each module's other importing test files as they existed at
  measurement time and excluded the untracked files created by concurrent work.
