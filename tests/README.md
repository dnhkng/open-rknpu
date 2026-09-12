# Tests

The host suite validates the compiler without hardware; the board runs are validated by the
retained evidence suites under `research/`.

```sh
make test                       # unittest discover -s tests
make test-one M=tests.test_walk # a single module
make coverage                   # + coverage floor (config in pyproject.toml)
make baseline                   # recompile 2,328 suite models against the baseline
make evidence                   # audit every published suite against its manifest
make perf                       # cost-model regression (tasks, engine blocks, arena)
make reproducible               # build the sdist twice, diff every byte, audit contents
make host-c                     # compile the runtime, board harnesses and host loader
make test-one M=tests.test_runtime_cli
```

## What each group pins

| Group | Modules | What it establishes |
| --- | --- | --- |
| Front end | `test_normalize`, `test_normalize_matrix`, `test_network`, `test_front_end_rejections`, `test_dense_lowering` | constant folding, bias/padding folding, kernel rewrites, grouped lowering, Q/DQ handling, loud rejection of unsupported graphs |
| Parsers and bounds | `test_native`, `test_profile_bounds`, `test_scheduler`, `test_padding`, `test_strided`, `test_chain*`, `test_depthwise*`, `test_walk`, `test_join_*`, `test_join_emitters_deep`, `test_join_variants_deep`, `test_mixed_heads`, `test_pool_join`, `test_pooled_*`, `test_graph`, `test_join_dag`, `test_elementwise*`, `test_activation_profiles`, `test_lut_*`, `test_transpose*`, `test_tiled_chain`, `test_scheduler_boundaries`, `test_graph_join_rejections` | every profile's accepted shapes *and* the first out-of-bounds neighbour, with the specific error message |
| Semantics | `test_emit_semantics`, `test_emitter_fuzz`, `test_chain`, `test_depthwise*`, `test_mul_*`, `test_elementwise*`, `test_reduction_and_dags`, `test_emitter_rejections`, `test_bounded_ops`, `test_walk_elementwise` | the compiled container's integer output against an independent implementation of the graph arithmetic, plus each emitter's own Python reference |
| Numerics | `test_quantization_edge_cases`, `test_calibration_methods`, `test_calibration`, `test_native_clip_reference`, `test_family_cost` | band math, zero points, requantization rounding, the three calibration methods and their error paths |
| Containers and runtime | `test_model`, `test_legacy_container`, `test_legacy_compiler_paths`, `test_sequence_roundtrip`, `test_container_fuzz`, `test_container_bindings`, `test_submission`, `test_async`, `test_compose`, `test_composer_arena`, `test_liveness`, `test_quantized_import_deep` | encode/decode round-trips, checksum and truncation handling, C-loader parity, task linking, arena allocation and reuse |
| CLI and examples | `test_cli`, `test_cli_matrix`, `test_cli_flags`, `test_examples_host`, `test_docs_commands`, `test_example_multi_model`, `test_example_depthwise_separable`, `test_example_benchmark` | the compile/inspect/normalize flag matrix, output files, the documented argument errors, and every command the documentation tells a reader to run (executed in an isolated copy of the tree, or skipped with a recorded reason) |
| Evidence | `test_ledger`, `test_suite_evidence`, `test_suite_replay`, `test_reference_docs`, `test_evidence_integrity`, `test_reference_coverage`, `test_mnist_reports`, `test_fashion_reports`, `test_lut_index_probe` | the board ledger rows match the recorded `board_results_*.json`, the baseline covers every suite model, sampled suites recompile to their published containers, and every retained suite keeps a README, manifest, containers, references and inputs that agree with each other |
| Release | `test_reproducible_build`, `test_perf_regression` | the sdist/wheel audit and normalisation logic, and the pinned cost model (task counts, engine blocks, registers, arena, payload) that a refactor must not inflate |
| Fetch scripts | `test_fetch_scripts` | the three dataset/toolchain fetchers offline: URL and sha256 pins, accept paths, exact mismatch and connection messages, no socket opened |
| Board harnesses | `board_*.c` (compiled on the host and on the target) | the C API behaviours the host tests mirror; `research/run_*_suite.py` drives them on hardware |
| Containers and API | `test_mutable_api`, `test_public_api` — the v4 mutable-parameter workflow and the frozen public surface |
| C runtime | `test_host_loader`, `test_runtime_cli`, `host_loader.c` | the loader's rejection surface over crafted and real containers (`ornpu_inspect`/`ornpu_open`), and the runner's CLI: usage and exit codes, `--inspect` over every published container against the Python decoder, buffer limits and input-length checks |

## Conventions

* A test that touches hardware behaviour must compare **bytes**, not tolerances. Float
  tolerances are only used in the semantic cross-checks (documented per test, ≤ 1 LSB).
* Every module starts with an MIT SPDX docstring stating what it pins and why, and every
  test names the bound or invariant it protects in its method name.
* Tests must be deterministic and fast: no network, no board, fixed seeds, and each module
  under ~10 seconds (the sampled suite replay is the one deliberate exception).
* When a profile's bound changes, the test for that bound changes in the same commit, and
  `research/container_baseline.json` is only regenerated when a container change is
  intended.
