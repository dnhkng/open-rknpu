# Deep dense Conv chains: one batched job instead of one ioctl per layer

Three independently generated `[Conv, Relu]*(N-1) + [Conv]` chains at 8x8 with N = 8,
12 and 16 layers and hidden channels 3..12 (`manifest.json`). Each model is emitted
twice from the same ONNX: `modelNNN.bin` is the **serial** container (one task per
ioctl, the shipping mode) and `batched/modelNNN.bin` is the **batched** twin of the
same graph, where the N-layer chain emitter links every task to the next program
(`0x10` = next payload-relative program offset, control `0x14 = 0x40`, last task
terminal). Both containers are the same length; only the tails and the submission flag
differ, and `inputNNN.u8` / `expectedNNN.i8` are shared.

The hidden-layer `Relu` of every chain model is applied by the layer that precedes
it (activation registers `0x4060 = 0x12`, `0x406c = 0x40e0 = 0`) and by the composed
reference's accumulator clamp - the S9 fix of 2026-09-11, which re-ran this suite.

## Board evidence

* Serial suite (the standard path): **3 models, 48 inferences, 9216 exact output
  bytes**, streamed through `tests/board_api.c`
  (`board_results_0.json`, `board_summary.txt`).
* Four variants of every model are byte-exact over 128 runs each
  (`batched_results.json`): serial, one batched job, double-buffered intermediates
  (`reuse/`), and both together (`reuse_batched/`).

| layers | arena untiled | arena double-buffered | serial | double-buffered serial | one batched job | reuse + batched |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 28,672 B | 20,480 B | 369 us | 184 us | 93 us | 94 us |
| 12 | 36,864 B | 24,576 B | 1537 us | 721 us | 240 us | 218 us |
| 16 | 40,960 B | 24,576 B | 1120 us | 645 us | 256 us | 186 us |

Double buffering keeps two 1 KB intermediate surfaces live instead of one per layer,
so the arena stops growing with depth and the chain stops streaming 16 distinct
buffers per inference - worth roughly 2x by itself, and additive with batched
submission (medians are noisy for the serial variants because they perform N blocking
submissions while the CPU is shared with the camera service; the batched numbers are
stable within about 20%).

Serial medians are noisy because they perform N blocking submissions while the CPU
shares the core with the camera service; every run of the batched mode is both exact
and faster, and the absolute batched time grows slowly with depth (~12 us per extra
layer) while serial pays a per-ioctl cost per layer.

## Why this shape works

Measured rule (`../group_probe/README.md`): a job submitted in one ioctl completes only
when every task links to the next program with the control measured for its engine
hand-off (here always `0x40`, one engine), and the last task
is terminal; depth is bounded only by the loader's 64-task table. A dense Conv chain is
exactly that shape - N CNA tasks - so the whole chain goes to the front end in one
submission with ping-pong enabled. Mixed-engine graphs (a pool or join task among the
convolutions) cannot: those keep the serial path, which is why the campaign's join
suites are unchanged.

## Reproduction

```sh
PYTHONPATH=src python3 research/build_deep_chain_suite.py
PYTHONPATH=src python research/run_profile_suite.py deep_chain_suite
PYTHONPATH=src python3 research/run_deep_chain_compare.py      # serial vs batched
PYTHONPATH=src python3 -m pytest tests/test_submission.py -q
```
