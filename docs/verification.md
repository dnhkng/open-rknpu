# Verification

The project's rule is simple: **a claim is a number produced by a script that anyone can
re-run, and hardware claims are byte-exact comparisons, not tolerance checks.** Three
checked-in contracts make that enforceable.

## 1. The container baseline

`research/container_baseline.json` maps every published suite model
(`research/*suite*/model*.onnx`, 2,328 of them) to either

* the sha256 of the container the compiler produces for it, or
* `ERR:<ExceptionType>` for the 47 models whose profiles deliberately reject them (46 from the original
  capture plus the one `walk_elementwise_suite` model that is a documented default rejection).

```sh
PYTHONPATH=src python research/verify_suites.py
# models=2268 baseline=2268 same=2268 changed=0 added=0 removed=0
# rejections: Counter({'ERR:ValueError': 47})
```

The map was captured before a large internal cleanup and is the "no emitted container may
change" contract: a refactor that alters any accepted container, or that silently starts
accepting a rejected model, fails here. Update it (`--update`) **only** when a container
change is intended and the affected suites have fresh board evidence.

Pinned rejections matter as much as hashes: a profile that starts compiling a graph it used
to refuse is a behaviour change too.

## 2. The host test suite

```sh
PYTHONPATH=src python -m unittest discover -s tests      # 1,155 tests
make coverage                                            # 99.20% line coverage, floor 99% in pyproject.toml
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
runs both. The measured value is **99.20%** - 6,851 statements with 55 uncovered lines in 10
modules; each one is listed with its reason in the generated table below. That took the compiler from 90% to here
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
Coverage is 99.20% (6,851 statements, 55 uncovered lines in 10 of 39 modules). Every remaining line is listed here because a floor nobody can explain is useless; the reason column is the module-level justification and the quoted source is there to check it against.

| Line | Source | Why it is not executed |
| --- | --- | --- |
| `chain_n.py:92` | `raise ValueError('calibration and output quantization overrides cannot be combined')` | duplicate of the scheduler's central calibration-plus-output-override check; the public entry point rejects the combination first |
| `compose.py:250` | `raise ValueError("external tensor %s overlaps internal %s"` | externals are placed outside the internal span by construction |
| `depthwise_join.py:147` | `raise ValueError('calibration and output quantization overrides cannot be combined')` | duplicate of the scheduler's central calibration-plus-output-override check; the public entry point rejects the combination first |
| `graph.py:365` | `raise ValueError('calibration and output quantization overrides cannot be combined')` | duplicate of the scheduler's central calibration-plus-output-override check; the public entry point rejects the combination first |
| `graph.py:383` | `tail_tensor=(tail_nodes[2*index+1].output[0] if 2*index+1<len(tail_nodes)` | the tail's measured band on a diamond with a Conv tail and calibration ranges; the join-chain variant is pinned by test_reference_edges, this one still needs a diamond-tail fixture |
| `graph.py:385` | `selected=measured_range(calibration_ranges,tail_tensor)` | the tail's measured band on a diamond with a Conv tail and calibration ranges; the join-chain variant is pinned by test_reference_edges, this one still needs a diamond-tail fixture |
| `graph.py:486` | `raise ValueError('diamond join requires adjacent head buffers')` | diamond heads overlap until the join, so the allocator places them adjacent by construction |
| `graph.py:611` | `raise ValueError('join chain heads must be Conv nodes reading the stem output')` | the matcher only records Conv heads reading the stem output, and the emitter rechecks the same condition |
| `graph.py:720` | `raise ValueError('calibration and output quantization overrides cannot be combined')` | duplicate of the scheduler's central calibration-plus-output-override check; the public entry point rejects the combination first |
| `graph.py:866` | `raise ValueError('join chain external tensor %s overlaps %s'%(external,name))` | externals are placed outside the internal span by policy, so the overlap test cannot be true |
| `join_dag.py:355` | `raise ValueError('join DAG cannot re-quantize a join result onto another band')` | the free operand is by definition not committed, and every uncommitted produced tensor is a by_name key |
| `join_dag.py:358` | `raise ValueError('join DAG operand was already committed to another band')` | same branch as 355: the free operand is never in the committed set |
| `join_dag.py:380` | `raise ValueError('calibration and output quantization overrides cannot be combined')` | duplicate of the scheduler's central calibration-plus-output-override check; the public entry point rejects the combination first |
| `join_dag.py:481` | `raise ValueError('join DAG external tensor %s overlaps %s' % (external, name))` | offsets are laid by the cursor loop after input0 and the output after the last end, so an overlap cannot occur |
| `join_dag.py:539` | `payload, _ = compile_depthwise(entry['standalone'],` | _prepare recompiles every branch final and a depthwise entry can only be a single-layer branch final, so recompiled is never None |
| `liveness.py:165` | `raise ValueError("no arena placement for tensor %s" % name)` | first-fit always finds a slot; an exhaustive and randomized search found no counterexample |
| `normalize.py:101` | `continue` | a pads attribute the rank promotion does not rewrite (not a two- or one-element list) leaves the node untouched for the scheduler to reject |
| `normalize.py:148` | `return None, None, None` | the Flatten/Reshape matcher declines an unproven or non-[N,C] shape and leaves the graph to the normal rejection |
| `normalize.py:156` | `return None, None, None` | as 148: the flatten axis is not 1, or it carries an attribute the matcher does not model |
| `normalize.py:170` | `return None, None, None` | as 148: the Reshape target is not a constant rank-1 tensor |
| `normalize.py:173` | `return None, None, None` | as 148: the Reshape target is not exactly two dimensions |
| `normalize.py:189` | `return None` | as 148: the resolved Reshape target is not the producer's [N, C] |
| `normalize.py:332` | `raise _global_pool_error(` | a GlobalAveragePool/ReduceMean outside the bounded 8x8 single-output envelope; the message names the bound |
| `normalize.py:335` | `raise _global_pool_error(node, "does not produce the single graph output")` | as 332: the pooling node is not the single graph output |
| `normalize.py:344` | `raise _global_pool_error(` | as 332: the node does not read one tensor with the expected rank |
| `normalize.py:349` | `raise _global_pool_error(node, "carries attributes %s" % sorted(extra))` | as 332: the pooling node carries attributes the rewrite does not model |
| `normalize.py:357` | `raise _global_pool_error(node, "sets noop_with_empty_axes")` | as 332: ReduceMean with noop_with_empty_axes is outside the rewrite |
| `normalize.py:386` | `return None` | the global-pool planner declines an unsupported form and leaves the graph to the normal rejection |
| `normalize.py:388` | `return None` | as 386: the tensor shape cannot be proven from value_info |
| `normalize.py:390` | `return None` | as 386: the shape is not the required 8x8 spatial form |
| `normalize.py:446` | `raise _concat_error(branch, "reads '%s', not the shared graph input '%s'"` | a Concat branch that does not read the shared input; the message names the branch |
| `normalize.py:453` | `raise _concat_error(branch, "has no constant float32 rank-four weights")` | as 446: a branch without constant float32 rank-four weights |
| `normalize.py:456` | `raise _concat_error(branch, "carries attributes that cannot be stacked")` | as 446: a branch carrying attributes that cannot be stacked |
| `normalize.py:466` | `raise _concat_error(branch, "has a bias that is not float32 [output_channels]")` | as 446: a branch bias that is not float32 [output_channels] |
| `normalize.py:475` | `weight_name += "_"` | the stacked weight initializer name collides with an existing one and is suffixed; the suite's graphs never hit the collision |
| `normalize.py:483` | `bias_name += "_"` | as 475: the stacked bias name collides and is suffixed |
| `pool_join.py:139` | `raise ValueError('calibration and output quantization overrides cannot be combined')` | duplicate of the scheduler's central calibration-plus-output-override check; the public entry point rejects the combination first |
| `transposed.py:47` | `raise ValueError("transposed reference requires a square kernel weight layout")` | the square-kernel guard; every retained transposed suite packs a square layout (the rectangular case is rewritten by the emitter before it is packed) |
| `transposed.py:65` | `height,width=activation.shape[:2]` | derived output geometry; the sampled suites always pass an explicit output_shape, and the emitter records one |
| `transposed.py:66` | `output_shape=((height-1)*strides[0]+k-pads[0]-pads[2]+output_padding[0],` | derived output geometry; the sampled suites always pass an explicit output_shape, and the emitter records one |
| `transposed.py:69` | `raise ValueError("transposed reference output_shape must match the emitted channels")` | the output_shape-channel guard; the emitted containers always agree with their weights |
| `transposed.py:87` | `result=product+int(q.output_zero_point)` | the shift-0 requantization branch; every public transposed profile emits shift > 0 |
| `walk.py:138` | `except ValueError:` | the malformed-stage guard: a stage the parser accepts but the band step rejects; the suite pins the parser-level messages instead |
| `walk.py:139` | `raise ValueError(_elementwise_error(` | as 138: the raise itself |
| `walk.py:223` | `return "supports one elementwise stage per chain"` | the error-message helper for a second elementwise stage (the suite pins the message through the parser) |
| `walk.py:234` | `return _elementwise_error(index, op,` | as 223: the helper's formatted branch |
| `walk.py:488` | `elif len(ops) > 1 and ops[1]["kind"] == "ew":` | the band bookkeeping for an elementwise stage directly after the first chain Conv; the suite's first Conv always has an interior pool or a later Conv between the samples |
| `walk.py:492` | `pending_elementwise = _elementwise_band(ops[1], _adjusted_scale(` | as 488: the pending elementwise band |
| `walk.py:494` | `first_quantization = native_quantize(` | as 488: re-quantizing the first Conv onto the stage's operand scale |
| `walk.py:587` | `natural = native_quantize(op["weights"], op["bias"], scale, zero_point, None)` | the natural-band branch for a stage after a later Conv; the sampled models all take the pending branch |
| `walk.py:588` | `pending_elementwise = _elementwise_band(ops[index + 1], _adjusted_scale(` | as 587: the pending band lookup |
| `walk.py:590` | `selected = dict(scale=pending_elementwise["scale"], zero_point=0)` | as 587: selecting the band the next Conv reads |
| `walk.py:753` | `raise ValueError("walk elementwise stage band disagrees with the emitter")` | the band-disagreement guard between the stage and the emitter |
| `walk.py:1015` | `raise ValueError("calibration and output quantization overrides cannot be combined")` | duplicate of the scheduler's central calibration-plus-output-override check |
| `walk.py:1085` | `selected = measured(tail["node"])` | the join-walk tail's measured band with calibration ranges; the join-chain variant is pinned by test_reference_edges |
<!-- coverage-table:end -->

The table is generated from the coverage data by `research/coverage_doc_table.py` and checked
in CI (`--check`), so the claim and the code cannot drift apart. Each reason is a
module-level statement about *why* the line is not executed, quoted next to its source so the
claim can be checked by reading the two lines above it.

## 3. The board ledger

`research/COVERAGE_EXPANSION_RESULTS.md` is the evidence index: 121 rows, each pointing at a
suite directory and counting models, inferences and **exact output bytes**. Ledger totals:
**1,786 models / 29,706 inferences / 10,605,299 exact output bytes**.

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
| `make test` | the 1,155-test host suite passes on 3.10-3.13 |
| `make coverage` | the suite executes 99.20% of the compiler's lines, floor 99% |
| `make baseline` | all 2,328 published suite models still compile to identical bytes (or stay rejected) |
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
