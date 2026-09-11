# Elementwise DAG — `Mul({Add,Mul,Sub,Max}(a,b), a)`

Four first-stage operators feed a terminal Mul, all with **fan-out on the first
branch output** and two external inputs:

```
a -> Conv(1x1) -+-> {Add|Mul|Sub|Max} -> C -+
                |                            +-> Mul -> output
b -> Conv(1x1) -+                            |
a -> Conv(1x1) ------------------------------+
```

**Verified on RV1103: 12 models, 192 inferences, 36,864 exact output bytes**
(`board_api_test elementwise_dag_suite 12`), three models per first stage. The
`Mul(Mul(a,b),a)` models are the first verified **multi-Mul** graph; the others
combine two different elementwise operators.

Expected bytes come from the composed integer references
(`open_rknpu.elementwise_chain.chain_reference`): branch references, the
first-stage reference (equal-scale Add/Sub/Max or the Mul requantization), then
the terminal Mul requantization. No vendor capture, RKNN model or library
participates.

## Reproduction

```sh
PYTHONPATH=src python research/build_elementwise_dag_suite.py
# cross-compile tests/board_api.c with runtime/open_rknpu.c, push the suite,
# then run: board_api_test elementwise_dag_suite 12
```

`tests/test_elementwise_chain.py` covers the four-task container, the two-input
stacking along H, the reference composition for every first stage, and the
rejections.

## Scope

Still one bounded pattern: 1x1 Conv branches, fixed `[1,3,8,8]`, no broadcasting,
no arena reuse, and the terminal operator must be Mul consuming the first branch.
General DAG scheduling, unequal runtime input shapes and three-input graphs remain
open in [completion-plan](../../docs/plans/completion-plan.md) phase P1.
