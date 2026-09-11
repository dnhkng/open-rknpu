# N-layer native Conv chain

Independently generated chains of `[Conv, Relu] * (N-1) + [Conv]` at fixed
`[1,3,8,8]`, N = 3 and 4, hidden channels 3..16, kernels 1x1 or padded 3x3. The
first layer uses the established legacy program; every later layer is a native16
program reading the previous signed intermediate. One arena holds the input,
all intermediate buffers and the output, so intermediate lifetimes are explicit.

The hidden-layer `Relu` of every chain model is applied by the layer that precedes
it (activation registers `0x4060 = 0x12`, `0x406c = 0x40e0 = 0`) and by the composed
reference's accumulator clamp - the S9 fix of 2026-09-11, which re-ran this suite.

**Verified on RV1103: 5 models, 80 inferences, 15,360 exact output bytes**
(`board_api_test native_chain_suite 5`). Configurations:

| # | Layers | Hidden | Kernels |
| --- | ---: | --- | --- |
| 0 | 3 | 5, 5, 3 | 1, 3, 1 |
| 1 | 3 | 8, 3, 3 | 3, 1, 3 |
| 2 | 3 | 16, 12, 3 | 3, 3, 1 |
| 3 | 4 | 4, 8, 6, 3 | 3, 3, 1, 3 |
| 4 | 4 | 6, 6, 6, 3 | 1, 1, 1, 1 |

Expected bytes come from the composed integer references (legacy first layer,
native16 later layers) — `open_rknpu.chain_n.chain_n_reference`. No vendor
capture, RKNN model or library participates.

## Reproduction

```sh
PYTHONPATH=src python research/build_native_chain_suite.py
# cross-compile tests/board_api.c with runtime/open_rknpu.c, push the suite,
# then run: board_api_test native_chain_suite 5
```

## Scope

This generalises the fixed two-layer `chain.py` to N layers over a shared arena.
It is not yet a topological scheduler: the chain is linear, and the arena does
not yet reuse bytes for tensors whose lifetime has ended. Intermediate tensors
are internal; exposing them as v5 external outputs is future work in
[completion-plan](../../docs/plans/completion-plan.md) phase P1.
