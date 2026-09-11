# Diamond DAG — fan-out plus fan-in with a liveness-planned arena

Second DAG shape on RV1103: one shared stem feeds two heads, and both head
outputs feed one elementwise task.

```
input -> Conv(1x1) -> Relu -> t -> Conv_head_a -+
                                -> Conv_head_b -+-> {Add,Mul,Sub,Max} -> output
```

`t` is produced once and read twice (fan-out), then both head outputs are
*consumed together* (fan-in). Hidden channels 3..16, dense 1x1/3x3 heads with
three output channels each, RGB 8x8 external tensors.

**Verified on RV1103: 12 models, 384 inferences, 73,728 exact output bytes.**

```
model 0: 32 inputs passed (1 outputs)
...
model 11: 32 inputs passed (1 outputs)
PASS: 12 v5 models, 384 inferences, 73728 exact output bytes; invalid IO rejected
```

Commands are independently generated (`src/open_rknpu/graph.py:compile_diamond`);
no RKNN model, library or capture is read. Expected bytes come from the composed
integer references: the legacy stem reference, the native-input reference per
head, then the join reference (`open_rknpu.graph.diamond_reference`).

## Arena plan from tensor lifetimes

The arena is no longer hand-computed. Each task declares its tensor reads and
writes, and `open_rknpu/liveness.py` produces the order, the live intervals and
the offsets:

| Tensor | Role | Lifetime (task positions) | Offset |
| --- | --- | --- | --- |
| `input0` | input | -1..0 | 8192 |
| `stem` | internal | 0..2 | 8576 |
| `head_a` | internal | 1..3 | 9600 |
| `head_b` | internal | 2..3 | 10624 |
| `output` | output | 3..3 | 11648 |

The stem stays live until the second head has read it, each head output until the
join, and the external output is placed after every internal tensor because the
version-5 loader rejects internal/external overlap. Three internal buffers are
live at once here (stem + one head while the other still runs), so the diamond
needs `lifetime_bytes == allocated_bytes == 3072`; the same allocator reproduces
the two-buffer ping-pong of long chains (`tests/test_liveness.py`).

## Quantization

* Stem: the legacy single-layer emitter's analytic range.
* Heads: each head's natural range is computed first, then **both** heads are
  re-quantized to the shared symmetric grid the elementwise join requires
  (`scale = max(head scales)`, `zero_point = 0`); `Add/Sub/Max` then use
  `output_scale = 2*scale` with zero point 0, while `Mul` keeps the per-head
  scales and folds them into its own conversion.
* Join: the register block is the verified native elementwise profile, with
  `0x5018` = `head_a` (accumulator side) and `0x5038` = `head_b` (ERDMA side,
  plane stride `0x5040` = 1024).

## Reproduction

```sh
PYTHONPATH=src python research/build_diamond_suite.py
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -fno-use-linker-plugin \
  -Iruntime tests/board_io.c runtime/open_rknpu.c -o /tmp/board_io
# push board_io and the suite to /userdata/open-npu-research/diamond_suite
./board_io . 12
```

## Models

| Index | Join | Hidden | Head kernels | Output scale | Lifetime bytes |
| --- | --- | --- | --- | --- | --- |
| 000 | Add | 8 | 1,3 | 94.5179 | 3072 |
| 001 | Add | 3 | 3,1 | 22.1730 | 3072 |
| 002 | Add | 16 | 3,3 | 225.8227 | 3072 |
| 003 | Mul | 8 | 1,3 | 41427.6875 | 3072 |
| 004 | Mul | 3 | 3,1 | 5230.2686 | 3072 |
| 005 | Mul | 16 | 3,3 | 2563793.0 | 3072 |
| 006 | Sub | 8 | 1,3 | 66.0266 | 3072 |
| 007 | Sub | 3 | 3,1 | 24.0626 | 3072 |
| 008 | Sub | 16 | 3,3 | 234.4146 | 3072 |
| 009 | Max | 8 | 1,3 | 99.0311 | 3072 |
| 010 | Max | 3 | 3,1 | 12.6543 | 3072 |
| 011 | Max | 16 | 3,3 | 241.0818 | 3072 |

Each model ran 32 inputs, including all-zero, all-255 and constant-128 cases plus
deterministic random data. `manifest.json` carries the schedule, lifetimes,
offsets and quantization of every model.

## Scope

This retires the "multi-consumer chains" blocker for this bounded shape. It is
not yet a general topological scheduler: the op mix is still restricted to
Conv/Relu plus one of four elementwise joins, and the pattern is matched as a
whole rather than assembling arbitrary emitter outputs by tensor name. Unequal
runtime input shapes also remain open.
