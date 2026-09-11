# Independent Max

Public `compile --sequence`: two 1x1 Conv branches feeding Max, fixed [1,3,8,8],
constant float32 weights/biases, no broadcasting. Shared symmetric branch scale s,
output scale 2*s and zero point 0. Integer reference: `rint(max(A,B)/2)` (ties-even).

12 models, 384 board inferences, 73,728 exact output bytes passed. See
../max_suite.log. These are independently generated commands; RKNN is not
used in compilation. Quantization accuracy tuning remains deferred.

Reproduce with `research/build_add_suite.py --op Max` under the open Python environment,
then `research/run_elementwise_suite.py max`. The board runner streams files
through the reusable mul_suite slot; that directory is scratch and may contain a
later experiment. All test artifacts are retained on the host.

This profile does not establish other shapes, broadcasts, scales or external inputs.
Remaining work: ../../docs/plans/primitive-roadmap.md.
