# Latest expansion status — 2026-09-09

See [current results and remaining tasks](COVERAGE_EXPANSION_RESULTS.md).
The depthwise geometry, transposed-kernel, LeakyReLU and LUT hypotheses below
have since been resolved within explicit bounds. This file retains the historical
first-pass results; use the current ledger for public support.

Update 2026-09-09: the C5 packing failure below is resolved for C5..16 at
8x8/k3/stride1 after a 1x1 stem. 192 new exact board runs. See
[expanded inventory](../docs/plans/coverage-matrix.md). The following is the historical first-pass report.

# Topics 1–7: implementation and issue log — 2026-09-08

This is a first pass through the seven topics, not completion of all modes.
Quantization accuracy tuning remains deferred. Runtime code was unchanged.
All 45 host tests pass; the camera remains running.

| Topic | Added / verified | Blocked or still pending |
|---|---|---|
| 1. Depthwise | Public 8x8/C3 k1 and k5 stride1, plus 8x8/C4 k3 stride1; 16 board inputs each, 10,240 exact bytes total. Public binaries match independently tested hypotheses. | 6x6/C3 k3 failed run2/output54 (-22 vs -21). C5 failed first fifth-channel output (3 vs -7). Shapes above/below 8x8, C5+, and channel multipliers require layout/packing work; no hardware limit inferred. |
| 2. Dense Conv | Constant group3 kernels with RGB input lower to dense zero-filled weights; 3x3 dilation2 lowers to 5x5. Three representative graphs (group, dilation, both), 48 board runs, 9,216 exact bytes. ONNX float equivalence checked. | Native group/dilation register modes remain unimplemented. Wider geometry and arbitrary padding remain pending; current emitter assumes symmetric padding and restricted input layout. These are implementation prerequisites, not demonstrated hardware blockers. |
| 3. Mul | Scalar/per-channel immutable constants after an unbranched Conv fold into weights/bias. Two graphs, 32 board runs, 6,144 exact bytes; float equivalence checked. | Independent external inputs require a multi-input API/container; current format accepts one input. Wider elementwise geometry, dynamic broadcasts and independent boundary quantization remain pending. |
| 4. Transposed Conv | Independent depthwise/transposed nearest-repeat hypothesis generated and executed. | Failed at run0/output0 (-128 vs -49). Kernel/padding/quantization interpretation is unresolved. Not exposed publicly. |
| 5. Fused activations | Independent LeakyReLU alpha0.5 hypotheses generated and executed. | Missing shift initially saturated; adding shift14 reduced error but still failed run0/output0 (27 vs 28), including a floor-rounding reference. Conversion ordering/rounding unresolved. No new activation exposed publicly. |
| 6. LUT activations | Inspected Sigmoid setup: 1,106 writes, including 1,027 LUT data writes, with a central value of 16,384. | Public runtime caps commands at 256 words; setup uses submission semantics that earlier flag changes broke. Need explicit long setup-task validation and submission support, then LUT-domain/interpolation tests. No new independent LUT hardware test this pass. |
| 7. Layout / graph integration | Terminal spatial Reshape preserving N,C and H*W, after an already-supported sequence. Three shapes, 96 board runs, 18,432 exact bytes. No data movement or extra NPU task is required. | Channel-changing reshape, transpose, concat and general graph scheduling remain pending. This is not a general layout engine. |

The confirmed additions total 224 board inferences and 44,032 exact bytes. These
counts describe the new tests, not exhaustive coverage of every accepted graph.
Algebraic lowering changes quantization choices and can differ slightly in float
rounding; tests establish float tolerance and exact lowered integer arithmetic,
not equivalence to a vendor's quantized graph or useful trained-model accuracy.

## Evidence and reproduction

- Depthwise kernels: `depthwise_expansion_suite/board_results_5.json`; C4:
  `depthwise_c4_suite/board_results_0.json`. C5 failure retained in
  `depthwise_channels_suite/board_results_0.json`; shape failure in
  `depthwise_expansion_suite/board_results.json`.
- `lowered_conv_suite/board_results_0.json`, `constant_mul_suite/board_results_0.json`,
  `spatial_reshape_suite/board_results_0.json`: exact board results.
- `transpose_independent_suite/board_results_0.json` and
  `leaky_independent_suite/*results*.json`: failed hypotheses retained.
- Generators: `build_lowered_conv_suite.py`, `build_constant_mul_suite.py`,
  `build_spatial_reshape_suite.py`, `probe_depthwise_channels.py 4`.
- Board runner: `PYTHONPATH=src python research/run_profile_suite.py SUITE`.
  It streams one case through the board's reusable mul_suite directory; that name
  does not describe the current contents. `--start N` selects a later starting case.
- The depthwise expansion generator included unverified shapes during research.
  The public compiler now rejects those shapes; saved failed artifacts are retained
  for analysis. Do not broaden public acceptance to reproduce failed hypotheses.

## Next work

Return to topic 1: compare C4/C5 and 8x8/6x6 vendor captures, isolate native layout
and bias/scale packing fields, then retest generated commands. Preserve passed
profiles while resolving these failures. Continue through the pending entries in
order; the seven topics are still open where indicated above.
