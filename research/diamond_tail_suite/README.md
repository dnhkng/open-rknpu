# Diamond + Conv tail — name-bound emitter composition

Third DAG shape on RV1103 and the first that composes two *different* emitters by
tensor name instead of matching one whole-graph pattern:

```
input -> Conv -> Relu -> t -> Conv_head_a -+
                           -> Conv_head_b -+-> {Add,Mul,Sub,Max} -> Conv[->Relu]* -> output
```

The join task's output is an **internal named tensor** that the first tail Conv
reads, so the elementwise join emitter and the native Conv emitter meet through the
version-5 tensor table. `open_rknpu/liveness` schedules the new edges: the join is
live until the first tail layer, each tail intermediate until the next, and the
external output is placed after every internal tensor.

**Verified on RV1103: 12 models, 192 inferences, 36,864 exact output bytes**
(`board_results_0.json`, `board_summary.txt`). Every model ran 16 inputs including
all-zero, all-255, constant-128 and random data.

| Models | Join | Tail | Tasks |
| --- | --- | --- | --- |
| 000–002 | Add | 1x1, 3x3, 1x1+3x3 | 5–6 |
| 003–005 | Mul | 1x1, 3x3, 1x1+3x3 | 5–6 |
| 006–008 | Sub | 1x1, 3x3, 1x1+3x3 | 5–6 |
| 009–011 | Max | 1x1, 3x3, 1x1+3x3 | 5–6 |

## Emission notes

* The tail is a `[Conv, Relu]* Conv` chain of 3-output 1x1/3x3 dense convolutions;
  each layer is quantized with the previous tensor's grid (`native_quantize`).
* The first tail program starts after the join program (`0xcc0 + 82 words`), which
  a first attempt overlapped — that produced a driver `-EINVAL` on submit.
* Each tail layer declares the **input zero point it reads** in `0x1184`
  (`zp & 0xffff`): the join grid is 0 for Add/Sub/Max, so the native default
  (`-128`) that the heads inherit from the stem was wrong for the first tail layer.
* Internals may share arena bytes when their lifetimes are disjoint (the allocator
  does this for the join and the stem); externals never overlap internals.

## Reproduction

```sh
PYTHONPATH=src python research/build_diamond_tail_suite.py
research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \
  --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \
  -O2 -std=gnu99 -Wall -Wextra -Werror -fno-use-linker-plugin \
  -Iruntime tests/board_io.c runtime/open_rknpu.c -o /tmp/board_io
# push board_io and the suite to /userdata/open-npu-research/diamond_tail
./board_io . 12
```

`tests/test_diamond_tail.py` checks the container, the lifetimes, the tail's input
grid register, the reference against the board-verified bytes and the rejections.

## Scope

Deep tails (three layers) run with coarse analytic per-tensor grids, so they are
pinned by board bytes rather than by a float comparison. The profile still needs
the stem/head shapes fixed at 8x8/C3 and one join; a scheduler that assembles
arbitrary emitter outputs by name is still the remaining P1 item.
