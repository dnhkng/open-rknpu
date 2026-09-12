# Verification

The project's rule is simple: **a claim is a number produced by a script that anyone can
re-run, and hardware claims are byte-exact comparisons, not tolerance checks.** Three
checked-in contracts make that enforceable.

## 1. The container baseline

`research/container_baseline.json` maps every published suite model
(`research/*suite*/model*.onnx`, 2,244 of them) to either

* the sha256 of the container the compiler produces for it, or
* `ERR:<ExceptionType>` for the 46 models whose profiles deliberately reject them.

```sh
PYTHONPATH=src python research/verify_suites.py
# models=2244 baseline=2244 same=2244 changed=0 added=0 removed=0
# rejections: Counter({'ERR:ValueError': 46})
```

The map was captured before a large internal cleanup and is the "no emitted container may
change" contract: a refactor that alters any accepted container, or that silently starts
accepting a rejected model, fails here. Update it (`--update`) **only** when a container
change is intended and the affected suites have fresh board evidence.

Pinned rejections matter as much as hashes: a profile that starts compiling a graph it used
to refuse is a behaviour change too.

## 2. The host test suite

```sh
PYTHONPATH=src python -m unittest discover -s tests      # 873 tests
make coverage                                            # 98% line coverage, floor 97% in pyproject.toml
PYTHONPATH=src python -m pytest tests -q
```

What the tests pin, in broad strokes:

* **parsers and dispatch** — every profile's accepted shapes, the scheduler's order, and the
  rejection messages (`test_native.py`, `test_chain*.py`, `test_walk.py`, …);
* **container/C-loader parity** — the Python encoder and the C loader agree on every field
  (`test_model.py`, `test_container_bindings.py`);
* **reference equivalence** — for each profile, the composed container's arithmetic and the
  Python integer reference agree on generated cases;
* **retained board evidence** — the tests re-read the published `board_results_*.json` and
  assert the counts and the summary line (`test_*_suite` style tests);
* **ledger integrity** — every ledger row's link resolves, the totals equal the sum of the
  rows, and each row's models/inferences/bytes are reproduced by the union of its passing
  board results (`test_ledger.py`);
* **documentation links** — relative links and heading anchors across every markdown file
  (`research/check_docs_links.py`, also runnable stand-alone).

PyTorch is never needed for the tests; NumPy and ONNX are.

## Coverage

The host suite is measured with `coverage` and gated at the floor configured in
`pyproject.toml` (`[tool.coverage.report] fail_under`); `make coverage` runs it and
`make coverage` is part of CI. The expansion added after the initial commit took the
compiler from 90% to 92%+ line coverage, with the generated `graph.py` branches and the
per-emitter rejection paths being the deliberately last areas to close. Coverage is a floor,
not a goal: a line covered by a test that only asserts "it returned" is worth less than one
covered by a boundary or semantic equality check, which is why the suite is organised around
bounds and independent references rather than around coverage alone.

### The last ten lines

Coverage is 98%; the remaining lines are listed here because a floor that nobody can explain
is useless. All of them are defensive guards or dead branches reachable only by
construction-breaking inputs — `tests/test_emitter_fuzz.py` names each one:

| Line | Why it cannot be reached |
| --- | --- |
| `compose.py:250` | external-overlaps-internal guard; inputs end before the arena, internals start at it |
| `elementwise.py:386` | guarded two lines earlier by a successful `np.broadcast_to` |
| `join_dag.py:342,347,350` | join-operand commit paths that the branch selection excludes |
| `join_dag.py:471` | external-overlap guard, impossible by placement |
| `join_dag.py:529` | `recompiled` is always set by `_prepare` |
| `liveness.py:165` | the aligned end of the placed tensors is always a valid candidate |
| `walk.py:271,282` | `parse_chain` already rejects a non-Conv first node and non-RGB input |

## 3. The board ledger

`research/COVERAGE_EXPANSION_RESULTS.md` is the evidence index: 121 rows, each pointing at a
suite directory and counting models, inferences and **exact output bytes**. Ledger totals:
**1,696 models / 28,266 inferences / 10,216,467 exact output bytes**.

A suite directory contains:

| File | Meaning |
| --- | --- |
| `modelNNN.onnx` | the graph that was compiled (the generator's input) |
| `modelNNN.bin` | the container that ran on the board |
| `inputNNN.u8`, `expectedNNN.i8` | the cases and the integer reference outputs |
| `manifest.json` | per-model metadata: shapes, `cases`, `output_bytes`, profile, node list |
| `board_results_*.json` | per-model `passed`, `inferences`, `bytes`, and the runner's line |
| `board_summary.txt` | the runner's `PASS: …` summary |
| `README.md` | what the suite demonstrates and the exact reproduce recipe |

`research/*/build_*.py` regenerates a suite (models, containers and reference outputs); the
board results are never regenerated, they are recorded facts.

## The campaign sweep

A second, stricter oracle for the largest suites: recompile them and diff the *published*
container bytes, keeping the known-bad artifacts visible instead of re-baselining them.

```sh
PYTHONPATH=src python research/campaign_sweep.py
# campaign sweep: same=157 diff=12 err=0
```

The 12 diffs are pre-existing artifact drifts: the published container predates a later
serializer/relayout fix, so a fresh compile cannot reproduce its bytes. They are pinned in
the script's `EXPECTED_DRIFT` set and documented in the ledger's "Pre-existing container
drifts" table — a new drift, or one that disappears, fails the sweep.

## Reproducing the doc build checks

```sh
python research/check_docs_links.py     # every markdown file: 0 broken links, 0 unresolved anchors
ruff check src tests examples research/verify_suites.py
```

## Adding evidence for a new primitive

1. Write `research/build_<name>.py` — a deterministic generator that behaves like the
   others: fixed RNG seed, `modelNNN.onnx` + `.bin` + `inputNNN.u8` + `expectedNNN.i8` +
   `manifest.json`.
2. Run it, then run the suite on the board with `research/run_v5_suite.py` (or
   `run_profile_suite.py` for v3/v4).
3. Add a row to the ledger table **and** update the `**Total**` row.
4. Extend `research/container_baseline.json` with the new models only if the new emitter is
   intended to change behaviour; otherwise the baseline test will flag it.
5. Add host tests: parser acceptance, reference equivalence, rejection cases, and a retained
   board-run assertion that reads `board_results_*.json`.
6. Write the suite `README.md` with the reproduce recipe and the measured numbers.

A suite whose evidence is incomplete (a `board_results` without its `board_summary`, a
generator that deletes its own evidence) is caught by `tests/test_suite_evidence.py`, a rule
added after exactly that mistake happened.
