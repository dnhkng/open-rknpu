# Tests

The host suite validates the compiler without hardware; the board runs are validated by the
retained evidence suites under `research/`.

```sh
make test                       # unittest discover -s tests
make test-one M=tests.test_walk # a single module
make coverage                   # + coverage floor (config in pyproject.toml)
make baseline                   # recompile 2,244 suite models against the baseline
```

## What each group pins

| Group | Modules | What it establishes |
| --- | --- | --- |
| Front end | `test_normalize`, `test_normalize_matrix`, `test_network` | constant folding, bias/padding folding, kernel rewrites, grouped lowering, Q/DQ handling, loud rejection of unsupported graphs |
| Parsers and bounds | `test_native`, `test_profile_bounds`, `test_scheduler`, `test_padding`, `test_strided`, `test_chain*`, `test_depthwise*`, `test_walk`, `test_join_*`, `test_join_emitters_deep`, `test_join_variants_deep`, `test_mixed_heads`, `test_pool_join`, `test_pooled_*`, `test_graph`, `test_join_dag`, `test_elementwise*`, `test_activation_profiles`, `test_lut_*`, `test_transpose*`, `test_tiled_chain` | every profile's accepted shapes *and* the first out-of-bounds neighbour, with the specific error message |
| Semantics | `test_emit_semantics`, `test_chain`, `test_depthwise*`, `test_mul_*`, `test_elementwise*`, `test_reduction_and_dags` | the compiled container's integer output against an independent implementation of the graph arithmetic, plus each emitter's own Python reference |
| Numerics | `test_quantization_edge_cases`, `test_calibration_methods`, `test_calibration`, `test_family_cost` | band math, zero points, requantization rounding, the three calibration methods and their error paths |
| Containers and runtime | `test_model`, `test_legacy_container`, `test_legacy_compiler_paths`, `test_sequence_roundtrip`, `test_container_bindings`, `test_submission`, `test_async`, `test_compose`, `test_composer_arena`, `test_liveness`, `test_quantized_import_deep` | encode/decode round-trips, checksum and truncation handling, C-loader parity, task linking, arena allocation and reuse |
| CLI | `test_cli`, `test_cli_matrix` | the compile/inspect/normalize flag matrix, output files and the documented argument errors |
| Evidence | `test_ledger`, `test_suite_evidence`, `test_suite_replay`, `test_mnist_reports`, `test_fashion_reports`, `test_lut_index_probe` | the board ledger rows match the recorded `board_results_*.json`, the baseline covers every suite model, and sampled suites recompile to their published containers |
| Board harnesses | `board_*.c` (compiled on the host and on the target) | the C API behaviours the host tests mirror; `research/run_*_suite.py` drives them on hardware |

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
