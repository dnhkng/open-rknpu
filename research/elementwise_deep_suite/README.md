# Deep elementwise DAG — N-stage Mul chains

Generalises the two-stage DAG to N stages:

```
a -> Conv(1x1) -+-> Op1(a,b) -> S1 -> Mul(S1,a) -> S2 -> ... -> Mul(S_{N-1},a) -> output
b -> Conv(1x1) -+
a -> Conv(1x1) ----------------------------------^  (fan-out on the first branch)
```

`Op1` is `Add` or `Mul`; every later stage is `Mul(stage, a)`. Two external
inputs, one output, no broadcasting.

**Verified on RV1103: 12 models, 192 inferences, 36,864 exact output bytes**
(`board_api_test elementwise_deep_suite 12`), covering 2, 3 and 4 stages with both
first stages; the 4-stage `Mul(Mul(Mul(Mul(a,b),a),a),a)` graph has six NPU tasks.
Expected bytes come from the composed integer references, applying each stage's
own `mul_output_conversion`. No vendor capture, RKNN model or library participates.

## Reproduction

```sh
PYTHONPATH=src python research/build_elementwise_deep_suite.py
# cross-compile tests/board_api.c with runtime/open_rknpu.c, push the suite,
# then run: board_api_test elementwise_deep_suite 12
```

`tests/test_elementwise_chain.py` checks task counts, stage metadata, the chain
label and the reference composition for every depth.

## Scope

The chain is linear with one fan-out operand and accumulates a conservative
output scale (`128^(N-1) * s1 * sA^N`), so deep chains become coarsely quantized
even though the integer arithmetic is exact. General DAG scheduling, arena reuse,
unequal runtime input shapes and three-input graphs remain open in
[completion-plan](../../docs/plans/completion-plan.md) phase P1.
