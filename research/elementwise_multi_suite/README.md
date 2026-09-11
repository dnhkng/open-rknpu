# Multi-input elementwise DAG (format v5)

`Mul(...Mul(Op(a,b), c1)..., c_last)` with **3, 4 or 5 named external inputs**:

```
a -> ConvA -+
            +-> Op1 -> C -+                  +-> ...
b -> ConvB -+              +-> Mul -> S1 -> Mul -> ... -> output
c1, c2, ... -> ConvC_k ----+                 (each extra input once)
```

`Op1` is `Add`, `Mul` or `Max`; every extra input is converted by an identity Conv
and multiplied with the running result. The version-5 tensor table names every
input and all internal tensors, and `ornpu_run_io` binds one buffer per input.

**Verified on RV1103: 18 models, 144 inferences, 27,648 exact output bytes**
(`board_io . 18`), two models per (first stage, input count) combination, up to
nine NPU tasks for five inputs. Expected bytes come from the composed integer
references (`open_rknpu.elementwise_multi.multi_input_reference`). No vendor
capture, RKNN model or library participates.

## Reproduction

```sh
PYTHONPATH=src python research/build_elementwise_multi_suite.py
# cross-compile tests/board_io.c with runtime/open_rknpu.c, push the suite,
# then run: cd elementwise_multi_suite && ./board_io . 18
```

`tests/test_elementwise_multi.py` checks the v5 container for 3/4/5 inputs, the
task counts, the chain labels and that the arena grows with the input count.

## Scope

Fixed `[1,3,8,8]` tensors, 1x1 Conv leaves, no broadcasting and no arena reuse.
Unequal runtime input shapes and general (non-chain) DAG scheduling remain open in
[completion-plan](../../docs/plans/completion-plan.md) phase P1. This suite supersedes the
earlier single-case three-input emitter.
