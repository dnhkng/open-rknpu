# N-layer chain with reused intermediate buffers

`compile --sequence --reuse-intermediates` applies a lifetime rule to the linear
chain: layer L writes the buffer that layer L-1 last read, so only two
intermediate buffers are live at once. Outputs are byte-identical to the
distinct-buffer chain; the arena is smaller.

The hidden-layer `Relu` of every chain model is applied by the layer that precedes
it (activation registers `0x4060 = 0x12`, `0x406c = 0x40e0 = 0`) and by the composed
reference's accumulator clamp - the S9 fix of 2026-09-11, which re-ran this suite.

**Verified on RV1103: 5 models, 80 inferences, 15,360 exact output bytes**
(`board_api_test chain_reuse_suite 5`), the same models as `native_chain_suite`.
Four-layer chains drop from a 16 KiB to a 12 KiB arena; three-layer chains have
only two intermediates and are unchanged.

```sh
PYTHONPATH=src python research/build_chain_reuse_suite.py
# cross-compile tests/board_api.c with runtime/open_rknpu.c, push the suite,
# then run: board_api_test chain_reuse_suite 5
```

This is the first lifetime allocator in the compiler: it is specific to a linear
chain and the two ping-pong slots. A general liveness pass over a DAG, unequal
runtime input shapes and multi-Mul composition remain open in
[completion-plan](../../docs/plans/completion-plan.md) phase P1.
