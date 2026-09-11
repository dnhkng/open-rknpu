# Multi-output N-layer chain (format v5)

`compile --sequence --expose-intermediates` emits a version-5 executable where
**every layer output is a named external output**, read through `ornpu_run_io`.
This exercises the named-tensor ABI with more than one output and with internal
activations promoted to the API boundary.

The hidden-layer `Relu` of every chain model is applied by the layer that precedes
it (activation registers `0x4060 = 0x12`, `0x406c = 0x40e0 = 0`) and by the composed
reference's accumulator clamp - the S9 fix of 2026-09-11, which re-ran this suite.

**Verified on RV1103: 4 models, 32 inferences, 40,448 exact output bytes**
(`board_io . 4`), including 3-output and 4-output chains:

| # | Layers | Outputs | Hidden | Kernels |
| --- | ---: | --- | --- | --- |
| 0 | 3 | 3 | 5, 5, 3 | 1, 3, 1 |
| 1 | 3 | 3 | 8, 3, 3 | 3, 1, 3 |
| 2 | 3 | 3 | 16, 12, 3 | 3, 3, 1 |
| 3 | 4 | 4 | 4, 8, 6, 3 | 3, 3, 1, 3 |

Expected bytes are the per-layer integer references
(`open_rknpu.chain_n.chain_n_reference_layers`) concatenated in output-tensor
order. `tests/board_io.c` also rejects a missing output and a surplus input
before any submit.

## Reproduction

```sh
PYTHONPATH=src python research/build_chain_multi_suite.py
# cross-compile tests/board_io.c with runtime/open_rknpu.c, push the suite,
# then run: ./board_io . 4
```

## Scope

The chain is linear and every intermediate is a distinct arena buffer. Arena
byte reuse for tensors whose lifetime has ended, unequal runtime input shapes
and a topologically general scheduler remain open in
[completion-plan](../../docs/plans/completion-plan.md) phase P1.
