# Two-stage elementwise DAG — `Mul(Add(a,b), a)`

A genuine multi-operator graph with **fan-out on an intermediate tensor** and two
external inputs:

```
a -> Conv(1x1) -+-> Add -> C -+
                |             +-> Mul -> output
b -> Conv(1x1) -+             |
a -> Conv(1x1) ---------------+
```

The Add subgraph is compiled with the verified two-branch elementwise emitter; a
second, independently generated 78-word Mul task is appended and reads the Add
result (`C`) and the first branch output (`A`), so `A` is consumed twice. The two
programs share one arena with the IO region relocated past the appended program.

**Verified on RV1103: 8 models, 128 inferences, 24,576 exact output bytes**
(`board_api_test elementwise_chain_suite 8`). Expected bytes come from the
composed integer references (`open_rknpu.elementwise_chain.chain_reference`):
branch references, the equal-scale Add, then the Mul requantization. No vendor
capture, RKNN model or library participates.

## Reproduction

```sh
PYTHONPATH=src python research/build_elementwise_chain_suite.py
# cross-compile tests/board_api.c with runtime/open_rknpu.c, push the suite,
# then run: board_api_test elementwise_chain_suite 8
```

`tests/test_elementwise_chain.py` checks the four-task container, the two-input
stacking along H, the reference composition and the rejections (`out` must
consume the first branch; output overrides are unsupported).

## Scope

This is one bounded DAG pattern, not a general scheduler: the branches are 1x1
Conv, shapes are fixed `[1,3,8,8]`, Add scales must be equal, and there is no
broadcasting or arena reuse. General DAG scheduling, unequal runtime input shapes
and multi-stage Mul chains remain open in
[completion-plan](../../docs/plans/completion-plan.md) phase P1.
