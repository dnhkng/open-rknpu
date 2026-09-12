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
PYTHONPATH=src python -m unittest discover -s tests      # 1,005 tests
make coverage                                            # 99.85% line coverage, floor 99% in pyproject.toml
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
`pyproject.toml` (`[tool.coverage.report] fail_under`, 99%); `make coverage` runs it and CI
runs both. The measured value is **99.85%** - 6,187 statements with 9 uncovered lines, all
of them proven unreachable in the current tree. That took the compiler from 90% to here
through two passes: the test expansion that closed the profile bounds, the emitter
rejections and the parser errors, and a later pass that closed the remaining boundary guards
and deleted two pieces of provably dead code (`elementwise.py`'s redundant broadcast check
and `elementwise_chain.py`'s always-false payload padding).

Coverage is a floor, not a goal: a line covered by a test that only asserts "it returned" is
worth less than one covered by a boundary or semantic equality check, which is why the suite
is organised around bounds and independent references. `make mutation-quick` measures the
complementary property - whether the tests would notice a wrong operator - and its report is
in [mutation-testing.md](mutation-testing.md).

### The uncovered lines

<!-- coverage-table:start -->
Coverage is 99.85% (6,187 statements, 9 uncovered lines in 4 of 38 modules). Every remaining line is listed here because a floor nobody can explain is useless; the reason column is the module-level justification and the quoted source is there to check it against.

| Line | Source | Why it is not executed |
| --- | --- | --- |
| `compose.py:250` | `raise ValueError("external tensor %s overlaps internal %s"` | externals are placed outside the internal span by construction |
| `graph.py:473` | `raise ValueError('diamond join requires adjacent head buffers')` | diamond heads overlap until the join, so the allocator places them adjacent by construction |
| `graph.py:598` | `raise ValueError('join chain heads must be Conv nodes reading the stem output')` | the matcher only records Conv heads reading the stem output, and the emitter rechecks the same condition |
| `graph.py:846` | `raise ValueError('join chain external tensor %s overlaps %s'%(external,name))` | externals are placed outside the internal span by policy, so the overlap test cannot be true |
| `join_dag.py:347` | `raise ValueError('join DAG cannot re-quantize a join result onto another band')` | the free operand is by definition not committed, and every uncommitted produced tensor is a by_name key |
| `join_dag.py:350` | `raise ValueError('join DAG operand was already committed to another band')` | same branch as 347: the free operand is never in the committed set |
| `join_dag.py:471` | `raise ValueError('join DAG external tensor %s overlaps %s' % (external, name))` | offsets are laid by the cursor loop after input0 and the output after the last end, so an overlap cannot occur |
| `join_dag.py:529` | `payload, _ = compile_depthwise(entry['standalone'],` | _prepare recompiles every branch final and a depthwise entry can only be a single-layer branch final, so recompiled is never None |
| `liveness.py:165` | `raise ValueError("no arena placement for tensor %s" % name)` | first-fit always finds a slot; an exhaustive and randomized search found no counterexample |
<!-- coverage-table:end -->

The table is generated from the coverage data by `research/coverage_doc_table.py` and checked
in CI (`--check`), so the claim and the code cannot drift apart. Each reason is a
module-level statement about *why* the line is not executed, quoted next to its source so the
claim can be checked by reading the two lines above it.

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

## Every gate, and what it proves

Run from the repository root; all of them are host-only and CI runs the applicable ones.

| Command | What it proves |
| --- | --- |
| `make test` | the 1,005-test host suite passes on 3.10-3.13 |
| `make coverage` | the suite executes 99.85% of the compiler's lines, floor 99% |
| `make baseline` | all 2,244 published suite models still compile to identical bytes (or stay rejected) |
| `make campaign` | the largest suites recompile to their published containers; the 12 known drifts stay known |
| `make evidence` | every retained suite agrees with its manifest, board results, references and README |
| `make perf` | the cost model (tasks, engine blocks, registers, arena, payload) did not regress |
| `make primitives` | every low-level op example still runs and matches its reference |
| `make docs-check` | 0 broken relative links and 0 unresolved anchors across every markdown file |
| `make host-c` | the runtime, every board harness and the host loader compile with `-Werror` |
| `make reproducible` | the sdist's contents are audited and normalised; two builds are byte-identical |
| `make mutation-quick` | the configured test subsets kill the injected faults (report: [mutation-testing.md](mutation-testing.md)) |

## Reproducing the doc build checks

```sh
python research/check_docs_links.py     # every markdown file: 0 broken links, 0 unresolved anchors
python research/coverage_doc_table.py --check   # the uncovered-line table below matches the coverage data
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
