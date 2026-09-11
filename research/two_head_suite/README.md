# Two-head fan-out — format v5 named tensors

First DAG profile on RV1103: one shared stem consumed by two heads.

```
input -> Conv(1x1) -> Relu -> t -> Conv_head_a -> outputA
                              -> Conv_head_b -> outputB
```

`t` is produced once and read by both heads (fan-out); the executable exposes
two external outputs. Hidden channels 3..16, dense 1x1/3x3 heads with three
output channels each, RGB 8x8 external tensors.

**Verified on RV1103: 14 models, 448 inferences, 172,032 exact output bytes.**
Each model ran 32 inputs, including all-zero, all-255 and constant-128 cases plus
deterministic random data. Commands are independently generated
(`src/open_rknpu/graph.py`); no RKNN model, library or capture is read. The
expected bytes come from the documented integer references: the legacy stem
reference followed by the native-input reference per head
(`research/build_two_head_suite.py`).

## Reproduction

```sh
PYTHONPATH=src python research/build_two_head_suite.py
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -fno-use-linker-plugin \
  -Iruntime tests/board_io.c runtime/open_rknpu.c -o /tmp/board_io
# push board_io and the suite to /userdata/open-npu-research/two_head_suite
./board_io . 14
```

`tests/board_io.c` also rejects a missing output and a surplus input before any
submit. Container bounds and the v5 tensor table are in `runtime/sequence_format.md`.

## Scope

This establishes the version-5 named-tensor ABI and fan-out for this bounded
profile, not a general scheduler. Unequal runtime input shapes, multiple
consumer chains, pool/depthwise branches and arena lifetime reuse remain open in
[completion-plan](../../docs/plans/completion-plan.md) phase P1.
