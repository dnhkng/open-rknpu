# Walk join suite: mixed pool kinds between heads and a join

The pooling profiles use **one** pool kind for every branch and end at the join; the
diamond profile has no pool. Graphs with a different pool per branch - and optionally a
Conv tail after the join - had no path at all. The op-level walk lowers them: one stage
per op, with the join band rules `compile_diamond` uses.

**Verified on RV1103: 6 models, 96 inferences, 4,608 exact output bytes**
(`run_v5_suite.py walk_join_suite`). Expected bytes come from
`open_rknpu.walk.join_walk_reference`, which composes the established image reference,
the native grid reference, the 2x2 block max/mean and `join_reference`.

| model | join | head a | head b | tail |
| --- | --- | --- | --- | --- |
| 0 | Add | K1 + MaxPool | K1 + AveragePool | - |
| 1 | Mul | K1+Relu + AveragePool | K3+Relu + MaxPool | - |
| 2 | Max | K3+Relu + MaxPool | K3 + AveragePool | K3 |
| 3 | Sub | K1 + AveragePool | K1+Relu + MaxPool | K1+Relu, K3 |
| 4 | Add | K3 + MaxPool | K3+Relu + AveragePool | K1+Relu, K1 |
| 5 | Mul | K1 + MaxPool (stem without Relu) | K3+Relu + AveragePool | K1 |

`parse_pooled_branches` now declines a graph whose pools differ in kind, so it falls
through to the walk instead of raising "pooled branches require matching pool kinds".

## Reproduction

```sh
PYTHONPATH=src python3 research/build_walk_join_suite.py
PYTHONPATH=src python3 research/run_v5_suite.py walk_join_suite
PYTHONPATH=src python3 -m pytest tests/test_walk.py -q
```
