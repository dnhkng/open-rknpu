# Native input C49–64 (four 16-lane planes)

Extends the native input layout to four 16-lane planes, completing public input
channels **C1–64** for dense native Conv.

**Verified on RV1103: 12 models, 192 inferences, 71,328 exact output bytes**
(`board_api_test native_c64_suite 12`). Models cover input C49/56/64, output
C1/3/8/16, K1 and K3, 6x5 and 8x8, with independently generated commands and the
documented native integer reference as expected output.

The plane-pair layout is the same as `native_c48_suite`: the first two 16-lane
planes are stored per tap, then the remaining planes (`part_b = lanes - 32`).
The four-plane marker capture `capture_native_c64_channels` places input channel
c at byte `c` for c<32 and `256+c` for c>=32, matching the formula reproduced by
`tests/test_native_c48.py`.

## Reproduction

```sh
PYTHONPATH=src python research/build_native_c64_suite.py
# cross-compile tests/board_api.c with runtime/open_rknpu.c, push the suite,
# then run: board_api_test native_c64_suite 12
```

## Scope

This is the dense native Conv input path. Intermediate channel counts above 16
in scheduled multi-operator graphs still need the channel-plane DAG allocation
from [completion-plan](../../docs/plans/completion-plan.md) phase P1. Channels above 64 do
**not** need multi-task accumulation: a vendor C128 Conv is a single CNA task over
eight 16-lane planes, board-verified in
[`native_c65_suite/`](../native_c65_suite/) (input C1..128 public).
