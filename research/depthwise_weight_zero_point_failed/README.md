# Superseded — resolved 2026-09-10

The weight zero point is the depthwise weight pair's second byte (stored as
`-zp`), not register `0x4054`. See `../depthwise_asymmetric_suite/README.md`
for the board-verified fix (27 models, 432 inferences). This directory retains
the failed register-based hypothesis as evidence.

# Failed shared depthwise weight-zero-point hypothesis

This retained C1/K1 independently generated command tests a one-sided positive
weight tensor whose asymmetric INT8 zero point is -128. The public symmetric
emitter was temporarily modified to place `0x80-zp` in the low16 bits of
RV1103 register `0x4054`, following Mesa Rocket's related-hardware
`DPU_BS_OW_OP` definition, and to store the uncentered INT8 weights.

`board_results_0.json` records failure on output byte0 of the first input:
hardware produced -34 and the exact centered reference expected -69. The
alternative `-zp` value produced the same result. The public compiler was
reverted to symmetric depthwise weights. These artifacts are excluded from the
passing ledger.
