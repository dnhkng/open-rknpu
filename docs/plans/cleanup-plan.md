# Cleanup plan

Goal: remove dead and stale weight from the codebase and make the documentation current,
**without changing a single emitted container**. The evidence chain is the contract: the
311 host tests, the 121-row board ledger, the campaign sweep baseline (157 same / 12 diff
/ 0 err) and a fresh-compile hash of all 2,244 suite models captured before the first edit
(`/tmp/cleanup_baseline.json`) must be identical afterwards.

## C1 — Tooling and hygiene

* add a `[tool.ruff]` section to `pyproject.toml` (line length, `E9`+`F` lint) and
  `.pytest_cache/` to `.gitignore`;
* drive `ruff check src tests --select F,E9` from **64 findings to zero**:
  6 `F811` redefinitions in `elementwise.py`, 27 unused imports, the rest unused locals;
* compact multi-statement lines in the touched hunks only (no reformat-everything pass,
  which would bury the diff).

## C2 — Dead code and drift

* delete `elementwise_multi.three_input_reference` (referenced nowhere);
* remove the unused locals that signal drift (`sequence.payload_start`,
  `join_dag.pool_basis`, `walk.stem`/`walk.in_channels`, `transposed.centered`, ...) after
  reading each: an unused local that used to feed a register is a bug, one that is a
  leftover binding is noise.

## C3 — Repository layout

* delete the empty stray `auto_padding_suite/` at the top level (the real suite is
  `research/auto_padding_suite/`, which the ledger links);
* move the three unreferenced `check*.onnx` leftovers to `research/legacy_onnx/` with a
  note rather than deleting them;
* delete the regenerable `build/` and `.pytest_cache/` trees;
* move the eight suite generators out of `tests/` into `research/` (the project
  convention) and update every reference.

## C4 — Documentation

* `README.md`: a repository-layout table (what lives where, including the vendor/Ghidra
  artifacts that stay at the top level) and a refreshed module inventory;
* `docs/plans/completion-plan.md` / `docs/plans/pipelining-plan.md` / `research/COVERAGE_EXPANSION_RESULTS.md`:
  counts, residual disposition and the cleanup record stay in sync;
* `docs/investigation-log.md`: a table of contents for the 1,500-line log;
* record this cleanup (its scope, the verification, and what was deliberately *not*
  touched) in the investigation log.

## C5 — Verification

The contract is reproducible from the tree instead of an ad-hoc script: the three commands
below are the checked-in half of the verification (the board ledger stays the hardware half).

```sh
PYTHONPATH=src python3 research/verify_suites.py    # 2,244 suite models vs research/container_baseline.json
PYTHONPATH=src python3 research/campaign_sweep.py   # 157 same / 12 pinned drift / 0 err
python3 research/check_docs_links.py                # relative links + heading anchors
```

1. all 2,244 `research/*suite*/model*.onnx` recompile to the checked-in baseline
   (`research/container_baseline.json`, captured before the first edit): **2,244 same /
   0 changed / 0 added / 0 removed**, including the 46 profiles that pin their rejection as
   `ERR:ValueError`;
2. `python -m unittest discover -s tests` and `python -m pytest tests -q`: **311 passing**
   (the new `tests/test_suite_evidence.py::test_container_baseline_covers_every_suite_model`
   keeps the map complete when a suite is added);
3. campaign sweep: **157 same / 12 diff / 0 err**, the 12 being the artifact drifts
   documented in `research/COVERAGE_EXPANSION_RESULTS.md` and pinned in the sweep;
4. links: **85 markdown files, 0 broken links, 0 unresolved anchors** (the 10 dangling links
   in the vendored `research/pretrained/mnist/upstream_README.md` are out of scope and
   skipped);
5. wheel rebuilt from the cleaned tree: **38 modules, byte-identical to `src/`**, and a
   clean-venv install of the wheel recompiles all **2,244 / 2,244** baseline entries;
6. board untouched (no board run needed: no container changed).

## Deliberately not touched

* `research/` evidence directories, decoded captures, oracle scripts and the vendor/Ghidra
  artifacts at the top level - they are the project's evidence, not code;
* the 12 pre-existing container drifts recorded by the campaign sweep;
* the pre-existing lint findings in the `research/` tooling: the lint gate is
  `ruff check src tests` (plus the three new verification scripts), because the
  generators/oracles are evidence tooling and a refactor there cannot be validated by
  re-running every board suite;
* column-heavy emitters that are correct but dense: reformatting them would bury the
  changes and risk transcription errors for no functional gain.
