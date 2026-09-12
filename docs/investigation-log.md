# RV1103 NPU / RKNN investigation log

Each entry below is prepended newest-first; the older narrative from the first
`regcmd` investigation is kept at the end.

<details>
<summary>Table of contents (74 entries)</summary>

* [2026-09-12: the channel-plane wall is the weight table, not C128 (F10 closed)](#2026-09-12-the-channel-plane-wall-is-the-weight-table-not-c128-f10-closed)
* [2026-09-12: the chain border zero point is programmed (F4)](#2026-09-12-the-chain-border-zero-point-is-programmed-f4)
* [2026-09-12: the camera path is held by rkipc (E5 probe)](#2026-09-12-the-camera-path-is-held-by-rkipc-e5-probe)
* [2026-09-12: dma-buf zero-copy input is implemented and board-verified (F8)](#2026-09-12-dma-buf-zero-copy-input-is-implemented-and-board-verified-f8)
* [2026-09-12: the NPU driver imports dma-bufs (CREATE flag 0x80) - F8 is feasible](#2026-09-12-the-npu-driver-imports-dma-bufs-create-flag-0x80---f8-is-feasible)
* [2026-09-11: publication-readiness pass - legal, interfaces, CI, docs, examples](#2026-09-11-publication-readiness-pass---legal-interfaces-ci-docs-examples)
* [2026-09-11: test expansion - 311 to 873 tests, 90% to 98% coverage](#2026-09-11-test-expansion---311-to-873-tests-90-to-98-coverage)
* [2026-09-11: a trained audio model runs on the NPU (mel-CNN spoken digits)](#2026-09-11-a-trained-audio-model-runs-on-the-npu-mel-cnn-spoken-digits)
* [2026-09-11: the driver ACTION surface measured; Silero VAD is outside the envelope](#2026-09-11-the-driver-action-surface-measured-silero-vad-is-outside-the-envelope)
* [2026-09-11: codebase cleanup - dead code, layout, docs (no container changed)](#2026-09-11-codebase-cleanup---dead-code-layout-docs-no-container-changed)
* [2026-09-11: the walk dispatches the join class, and mixed pool kinds get a path](#2026-09-11-the-walk-dispatches-the-join-class-and-mixed-pool-kinds-get-a-path)
* [2026-09-11: residual disposition closed (final pass)](#2026-09-11-residual-disposition-closed-final-pass)
* [2026-09-11: the walk lowers fan-in joins, and the diamond tail Relu was dropped](#2026-09-11-the-walk-lowers-fan-in-joins-and-the-diamond-tail-relu-was-dropped)
* [2026-09-11: the op-level walk lowers pools inside a Conv chain (P1)](#2026-09-11-the-op-level-walk-lowers-pools-inside-a-conv-chain-p1)
* [2026-09-11: first step of the op-level walk (P1) - Conv/Relu exact, pooling open](#2026-09-11-first-step-of-the-op-level-walk-p1---convrelu-exact-pooling-open)
* [2026-09-11: the ERDMA secondary operand is read linearly - no spatial broadcast (P4 closed)](#2026-09-11-the-erdma-secondary-operand-is-read-linearly---no-spatial-broadcast-p4-closed)
* [2026-09-11: every DAG emitter composes; the scheduler walk is the last P1 item](#2026-09-11-every-dag-emitter-composes-the-scheduler-walk-is-the-last-p1-item)
* [2026-09-11: the diamond and join-DAG emitters assemble through the composer (P1 port)](#2026-09-11-the-diamond-and-join-dag-emitters-assemble-through-the-composer-p1-port)
* [2026-09-11: the elementwise output stage has no BS-table read (P4 sub-item closed)](#2026-09-11-the-elementwise-output-stage-has-no-bs-table-read-p4-sub-item-closed)
* [2026-09-11: the per-family cost table cross-checked on held-out containers (S5 residual closed)](#2026-09-11-the-per-family-cost-table-cross-checked-on-held-out-containers-s5-residual-closed)
* [2026-09-11: input channels C1..128 - the C64 cap was ours, not the hardware (P2 closed)](#2026-09-11-input-channels-c1128---the-c64-cap-was-ours-not-the-hardware-p2-closed)
* [2026-09-11: residuals closed - universal batched submission, percentile/KL calibration, fence-free completion](#2026-09-11-residuals-closed---universal-batched-submission-percentilekl-calibration-fence-free-completion)
* [2026-09-11: the tail control is the successor's fetch amount - so every DAG is one job (S8 corrected, S10)](#2026-09-11-the-tail-control-is-the-successors-fetch-amount---so-every-dag-is-one-job-s8-corrected-s10)
* [2026-09-11: the chain family applies its hidden Relu now (S9 fixed)](#2026-09-11-the-chain-family-applies-its-hidden-relu-now-s9-fixed)
* [2026-09-11: engine-run splitting gives DPU->CNA graphs the fewest possible jobs (S10)](#2026-09-11-engine-run-splitting-gives-dpu-cna-graphs-the-fewest-possible-jobs-s10)
* [2026-09-11: the tail control word names the engine hand-off; mixed DAGs fit one job (S8) [SUPERSEDED IN PART]](#2026-09-11-the-tail-control-word-names-the-engine-hand-off-mixed-dags-fit-one-job-s8-superseded-in-part)
* [2026-09-11: K>1 strip tiling needs the pad value and the real activation (S7), and the chain family drops hidden Relu (S9)](#2026-09-11-k1-strip-tiling-needs-the-pad-value-and-the-real-activation-s7-and-the-chain-family-drops-hidden-relu-s9)
* [2026-09-10: height-strip tiled chains are exact (S3)](#2026-09-10-height-strip-tiled-chains-are-exact-s3)
* [2026-09-10: per-family task cost from duplicated-task fitting (S5)](#2026-09-10-per-family-task-cost-from-duplicated-task-fitting-s5)
* [2026-09-10: per-family task cost is below the 8x8 noise floor (S5 residual)](#2026-09-10-per-family-task-cost-is-below-the-8x8-noise-floor-s5-residual)
* [2026-09-10: non-blocking submission pipelines 1.5-2.25x; fences need kernel config (S4)](#2026-09-10-non-blocking-submission-pipelines-15-225x-fences-need-kernel-config-s4)
* [2026-09-10: double-buffered intermediate surfaces (S3)](#2026-09-10-double-buffered-intermediate-surfaces-s3)
* [2026-09-10: batched job = linked single-engine list; up to 11x on deep chains (S2)](#2026-09-10-batched-job--linked-single-engine-list-up-to-11x-on-deep-chains-s2)
* [2026-09-10: batched submission runs only a two-task single-engine job (S1) [SUPERSEDED]](#2026-09-10-batched-submission-runs-only-a-two-task-single-engine-job-s1-superseded)
* [2026-09-10: batched submission runs only a two-task single-engine job (S1)](#2026-09-10-batched-submission-runs-only-a-two-task-single-engine-job-s1)
* [2026-09-10: stage composer and declared bindings (P1)](#2026-09-10-stage-composer-and-declared-bindings-p1)
* [2026-09-10: LUT table index read directly; the residual is not affine (P6)](#2026-09-10-lut-table-index-read-directly-the-residual-is-not-affine-p6)
* [2026-09-10: LUT negative-half gain is a whole-bit shift (P6)](#2026-09-10-lut-negative-half-gain-is-a-whole-bit-shift-p6)
* [2026-09-10: join output consumed by a Conv tail (P1 emitter composition)](#2026-09-10-join-output-consumed-by-a-conv-tail-p1-emitter-composition)
* [2026-09-10: K5 ConvTranspose solved from the vendor-derived phase field (P3)](#2026-09-10-k5-convtranspose-solved-from-the-vendor-derived-phase-field-p3)
* [2026-09-10: K5 ConvTranspose blocker re-derived from the vendor capture (P3)](#2026-09-10-k5-convtranspose-blocker-re-derived-from-the-vendor-capture-p3)
* [2026-09-10: LUT domain measured by sign; mixed-input stems blocked (P6)](#2026-09-10-lut-domain-measured-by-sign-mixed-input-stems-blocked-p6)
* [2026-09-10: diamond fan-in DAG and the generic lifetime pass (P1)](#2026-09-10-diamond-fan-in-dag-and-the-generic-lifetime-pass-p1)
* [2026-09-10: per-channel Mul quantization (depthwise route), OW_SRC refuted](#2026-09-10-per-channel-mul-quantization-depthwise-route-owsrc-refuted)
* [Latest status — 2026-09-09](#latest-status--2026-09-09)
* [2026-09-10: task bindings are derived from every container and verified (P1)](#2026-09-10-task-bindings-are-derived-from-every-container-and-verified-p1)
* [2026-09-10: pooled multi-layer branches (P1/P5)](#2026-09-10-pooled-multi-layer-branches-p1p5)
* [2026-09-10: terminal pooling on a join-DAG result (P1/P5)](#2026-09-10-terminal-pooling-on-a-join-dag-result-p1p5)
* [2026-09-10: depthwise layers inside branch chains (P1)](#2026-09-10-depthwise-layers-inside-branch-chains-p1)
* [2026-09-10: packaging re-verified with the campaign's modules (P10)](#2026-09-10-packaging-re-verified-with-the-campaigns-modules-p10)
* [2026-09-10: the ledger is now machine-verified (P0)](#2026-09-10-the-ledger-is-now-machine-verified-p0)
* [2026-09-10: multi-layer branches and the arena-reuse hazard (P1)](#2026-09-10-multi-layer-branches-and-the-arena-reuse-hazard-p1)
* [2026-09-10: general join expression over reused internal grids (P1)](#2026-09-10-general-join-expression-over-reused-internal-grids-p1)
* [2026-09-10: runtime residual feature map on a fan-out result (P1)](#2026-09-10-runtime-residual-feature-map-on-a-fan-out-result-p1)
* [2026-09-10: runtime per-channel scale inside a fan-out (P1/P4)](#2026-09-10-runtime-per-channel-scale-inside-a-fan-out-p1p4)
* [2026-09-10: depthwise K3-dilation2 ConvTranspose as a direct sparse K5 (P3)](#2026-09-10-depthwise-k3-dilation2-convtranspose-as-a-direct-sparse-k5-p3)
* [2026-09-10: mixed dense/depthwise heads in one fan-out (P1)](#2026-09-10-mixed-densedepthwise-heads-in-one-fan-out-p1)
* [2026-09-10: pooled branches into the join (P1/P5)](#2026-09-10-pooled-branches-into-the-join-p1p5)
* [2026-09-10: dense and depthwise branches from one stem (P1/P4)](#2026-09-10-dense-and-depthwise-branches-from-one-stem-p1p4)
* [2026-09-10: join-chain fan-out (variable consumers, mixed joins)](#2026-09-10-join-chain-fan-out-variable-consumers-mixed-joins)
* [2026-09-10: runtime per-channel scale Mul (unequal external input shapes)](#2026-09-10-runtime-per-channel-scale-mul-unequal-external-input-shapes)
* [2026-09-09: depthwise C5–C16 packing resolved](#2026-09-09-depthwise-c5c16-packing-resolved)
* [2026-09-08: first pass through topics 1–7](#2026-09-08-first-pass-through-topics-17)
* [2026-09-08: depthwise stride2 public profile](#2026-09-08-depthwise-stride2-public-profile)
* [2026-09-08: stride-2 dense Conv initial public profile](#2026-09-08-stride-2-dense-conv-initial-public-profile)
* [2026-09-08: Sub and Max; ordered remaining-mode roadmap](#2026-09-08-sub-and-max-ordered-remaining-mode-roadmap)
* [2026-09-08: independent Mul milestone](#2026-09-08-independent-mul-milestone)
* [2026-09-08: Mesa/TRM-guided Add field validation](#2026-09-08-mesatrm-guided-add-field-validation)
* [2026-09-08: independent Add compiler/runtime milestone](#2026-09-08-independent-add-compilerruntime-milestone)
* [2026-09-08: independent depthwise compiler milestone](#2026-09-08-independent-depthwise-compiler-milestone)
* [2026-09-08: primitive hardware-path survey](#2026-09-08-primitive-hardware-path-survey)
* [2026-09-08: rough real-digit sanity check, no tuning](#2026-09-08-rough-real-digit-sanity-check-no-tuning)
* [2026-09-08: trained Conv2 successfully offloaded](#2026-09-08-trained-conv2-successfully-offloaded)
* [2026-09-08: complete hybrid MNIST milestone](#2026-09-08-complete-hybrid-mnist-milestone)
* [Summary: `regcmd` investigation (older findings, kept as-is)](#rv1103-npu--rknn-regcmd-investigation--summary)

</details>

## 2026-09-12: the channel-plane wall is the weight table, not C128 (F10 closed)

"Channel-split accumulation for C > 128" was carried as an XL research item on the theory
that more than 128 input channels need the undocumented INT32 partial-sum path. It does not:
one CNA task reads many more than eight 16-lane planes, and the only wall is structural.
This pass measured both walls, lifted the front-end cap to them, and pinned a suite on each
side.

**Input.** The weight table is built from 32-lane parts (`native.weight_byte`: all taps of
lanes 0..31, then of lanes 32..63, ..., inside each 16-output block), and the driver
completes at most **511 parts**. `ceildiv(lanes, 32) <= 511` therefore caps the input at
**16352 channels**. The boundary is exact and was measured twice - once with a scratch tree,
once with the shipped emitter and the shipped runtime built with the bounds lifted
(`research/probe_wide_channel_wall.py`):

| Model | planes | parts | Board result |
| --- | ---: | ---: | --- |
| `c16352-in-k1` (2x2, K1) | 1022 | 511 | PASS, byte-exact |
| `c16368-in-k1` (2x2, K1) | 1023 | 512 | never completes: `RKNPU: job timeout ... soft reset`, `-EINVAL` to the caller |
| `c14560-in-k3` (1x1, K3) | 910 | 455 | PASS (K3, `0x1188 = 65520`) |
| `c1808-in-k3` (8x8, K3) | 113 | 57 | PASS |
| `c1360-in-k3` (16x16, K3, height-tiled) | 85 | 43 | PASS |

The `0x1188 = 8*k*k*tiles` register is *not* the wall: C14560/K3 programs 65520 there, and
`c1808-in-k3` programs 8136 (above the 6144-atom height-tiling budget) - both exact. The
6144-atom budget in `native.tile_geometry` governs input atoms per task, so wide channels
shrink the per-task image and trigger height tiling instead (`c1360-in-k3`, 16x16, 7 tasks).

**Output.** The surface block index `(oc-1)//16` is nine bits, so **8192 output channels**
(512 blocks) is the largest exact geometry. C16384 (1024 blocks) runs and then lies: the
first 8192 channels come back correct and every byte after that is wrong (`output 8192: got
0 expected -21`), which is exactly the block-boundary signature. C8192 is byte-exact.

**What changed.** `sequence.MAX_NATIVE_CHANNELS = 16352`, `MAX_OUTPUT_CHANNELS = 8192` and
`MAX_WEIGHT_PARTS = 511` replace the old flat `128` in `native.compile_native_input` and in
both container decoders (Python `sequence.py`, C `runtime/open_rknpu.c`; the C header exposes
`ORNPU_MAX_NATIVE_CHANNELS`/`ORNPU_MAX_OUTPUT_CHANNELS`, overridable only for the probe).
The v5 tensor table uses the input bound for every role because an internal tensor can be
either side of a chain. No existing container changed: `verify_suites.py` reported
`changed=0` and 13 added models, which were re-baselined after their board run.

**Evidence.** `research/wide_channel_suite/` (built by `research/build_wide_channel_suite.py`)
holds 13 models across the new range - C129, C160, C192, C256, C1024, C4096, C8192, C16352,
C1360/K3 and C144/K3 height-tiled, C14560/K3, plus C8192 output - with expected bytes from
`native_input_reference`. Board: **13 models, 45 inferences, 168,994 exact output bytes**,
0 mismatches (`board_results_0.json`). The refused sides are re-measurable with
`research/probe_wide_channel_wall.py`, which prints the matching `-D` cross-compile line for
the lifted runtime.

**Silero VAD (E12) after F10.** The 2026-09-11 envelope listed input C129 as one of four
independent blockers. It is not one any more: C129 lowers, and so does the rank-3 1-D form
that the STFT/encoder Convs need. Re-measured 2026-09-12, a well-formed three-layer 1-D
encoder chain (`129->128->64->128`, K3, pad 1, rank-3 weights) still does not compile - the
legacy chain profiles require the fixed `[1,3,8,8]` float image (`native chain external
tensors must be float32 [1,3,8,8]`) and the walk rejects the promoted `H=1` chain
(`unsupported chain walk graph`, then `unsupported two-layer graph or quantization
parameters`) - and the model's other blockers are untouched: the 256-tap/stride-128 STFT
basis, the two `LSTM` layers and the `If`/`Shape`/`Gather`/dynamic-`Slice` control flow. E12
therefore closes as a documented negative with two blockers removed rather than four
remaining; the audio milestone stays `examples/mel-kws/`.

## 2026-09-12: the chain border zero point is programmed (F4)

The oldest documented accuracy quirk is gone. `chain.py` and `chain_n.py` built every layer
from the `NATIVE` register template and never wrote `0x1184`, so the CNA used the register's
`0xff80` reset - "declared input zero point 0, pad borders with 0" - for *every* layer,
including the hidden ones whose band is not zero point 0. `docs/registers.md` describes the
register exactly: "input zero point declared to the CNA; it also sets the value the engine
pads borders with". A hidden Conv therefore padded with `(-128 - zero_point) * scale`
instead of zero, and the integer reference modelled the same mistake, which is why the
containers were byte-exact and board-verified while the models lost accuracy.

The fix is one register per layer, in three emitters:

* `chain_n.py`: `0x1184 = quantizations[index-1].output_zero_point` for every layer after
  the first (the first reads the packed image, whose zero point is the container's);
* `chain.py` (legacy two-layer profile 2): the second task takes the first layer's output
  zero point;
* `tiled_chain.py` (the opt-in height-strip form): the same band, passed back through
  `native_fields`' API-domain parameter (`zero_point + 128`, since the builder subtracts
  128) - a first attempt passed the centered value straight through and every tiled variant
  mismatched on the board, which is how the domain convention was pinned down.

The references (`chain_n_reference_layers`, and the new `chain.reference`) pass each layer's
predecessor zero point into `native_reference`, so containers and expectations stay
together. Measured against the float ONNX model over the retained suites (16 cases/model):

| Model | Before | After |
| --- | ---: | ---: |
| `native_chain_suite/model001` | 1598.06 | 40.86 |
| `native_chain_suite/model003` | 46197.56 | 1122.73 |
| `chain_reuse_suite/model001` | 1378.48 | 85.03 |
| `chain_reuse_suite/model003` | 41444.34 | 1247.81 |
| `deep_chain_suite/model000` | 195508.84 | 5584.92 |
| `deep_chain_suite/model001` | 15728511708.80 | 2197.65 |
| `chain_calibration_suite/chain5` (analytic) | 53161.90 | 9404.41 |

Chains whose hidden bands are already zero point -128 are byte-identical (the register value
does not change), which is why only 17 of the family's containers moved. Those 17 were
re-baselined intentionally (`verify_suites.py --update`) and re-run on the board:
`native_chain_suite` 5 models/80 inferences/15,360 exact bytes, `chain_reuse_suite` 5/80/
15,360, `deep_chain_suite` 3/48/9,216, `chain_multi_suite` 4/32/40,448,
`chain_calibration_suite` 6/96/18,432 - all PASS with 0 mismatches. Both height-strip probes
(`tiled_chain_probe` 16 cases/variant, `tiled_k3_probe` 4 cases/variant) were regenerated and
re-run, also all exact. `tests/test_tiled_chain.py` now asserts the per-layer border value
instead of pinning the `0xff80` reset.

## 2026-09-12: the camera path is held by rkipc (E5 probe)

`rkisp_mainpath` (`/dev/video11`) is the ISP's 2304x1296 NV12 output and `rkipc` (pid 281)
holds it - `/proc/281/fd` shows `/dev/video11` open twice, plus `/dev/video0`, `1`, `12`,
`13`, `17`, `18`, `19`, `20`. A second streaming client is refused by the driver:

```text
$ v4l2-ctl -d /dev/video11 --stream-mmap --stream-count=1 --stream-to=/tmp/isp.raw
VIDIOC_REQBUFS returned -1 (Device or resource busy)
```

So a camera -> NPU example cannot open the ISP main path while `rkipc` runs, and stopping
`rkipc` is not an option for this board (it is the camera appliance; the project's rule is
that it stays alive). The free nodes are the ones nothing streams (`stream_cif_mipi_id2/3`,
`rkcif_scale_ch0..3`, `rkisp_lumapath`); a capture attempt on `rkcif_scale_ch2` blocked
waiting for a frame that never arrives, so a probe of a free node needs a timeout.

What E5 therefore needs: a V4L2 capture example that takes a device and format from the
caller, converts YUYV/NV12 to the model's packed RGB input, and runs the NPU through the
dma-buf path from F8 - verified on the board with a recorded frame (the plumbing) while the
README states the measured `rkipc` constraint for the live sensor. `ornpu_open_shared` makes
the zero-copy hand-off possible: a capture buffer written at `ornpu_input_view`'s offset is
the memory the engine reads.

## 2026-09-12: dma-buf zero-copy input is implemented and board-verified (F8)

The morning probe established that the driver imports dma-bufs (`CREATE` flag `0x80`); the
runtime now uses it. `ornpu_open_shared` allocates the model's arena from the Rockchip CMA
heap (`/dev/rk_dma_heap/rk-dma-heap-cma`), imports that fd into the NPU and hands the fd
back, so a producer maps the very memory the engine reads. `ornpu_input_view` reports the
arena offset plus the row stride and geometry, and `ornpu_run_prefilled` clears the output,
completes the 16-lane stride padding with the input zero point, syncs, submits and unpacks -
it never touches the producer's bytes. `ornpu_info` gained `arena_bytes` for the mapping.

Two details are worth recording because they are where a naive version fails:

* **A contiguous `memcpy` of the API bytes is wrong.** The packed layout stores rows at a
  16-byte-aligned stride, so the producer has to write rows (or already have that stride, as
  the ISP does). The first version of the harness copied the flat buffer and 8 of the first
  12 `add_geometry_suite` models failed; writing rows at `view.row_stride` fixed all of them.
* **The runtime still owns the padding.** `ornpu_run` fills the whole input region with the
  input zero point before copying rows; the prefilled path must fill only the stride padding,
  or it would overwrite the producer's data.

Board evidence, the whole packed `add_geometry_suite` through this path with
`tests/board_shared.c` (two runs per model):

```text
arena fd=6 offset=4096 row_stride=16 1x5x5x3
SUMMARY runs=2 arena_fd=6 input_offset=4096 input_bytes=75 exact_bytes=100 mismatches=0 result=PASS
ZEROCOPY add_geometry_suite: models=32 fail=0 inferences=64 exact_bytes=19968
```

`ornpu_input_view` returns `-ENOTSUP` for `native16` inputs (an internal packing a caller
cannot reproduce) and for the two-input legacy layout; those keep the copying path. The API,
the producer loop and the evidence are documented in `docs/c-api.md` and `docs/board.md`;
`tests/test_runtime_shared.py` pins the source contract.

## 2026-09-12: the NPU driver imports dma-bufs (CREATE flag 0x80) - F8 is feasible

The limitation table has said "dma-buf zero-copy from the ISP: not implemented" since the
first board session, and `docs/plans/pipelining-plan.md` S4 listed the driver's dma-buf path
among the interfaces the runtime deliberately does not use - but nobody had asked the driver
whether it *can*. It can.

The probe is 40 lines of C (`research/probe_dmabuf.c`, cross-compiled with the pinned toolchain
and run on the board (the source is checked in for reproducibility)): allocate 4096 bytes from the Rockchip CMA heap
(`/dev/rk_dma_heap/rk-dma-heap-cma`, the standard `DMA_HEAP_IOCTL_ALLOC`), then call the NPU
driver's `CREATE` ioctl (`_IOWR('r',2,struct allocation)`) with the dma-buf fd in `handle`
and every `flags` value from 0x00 to 0xFF:

```text
heap: allocated 4096 bytes, fd=4
flags=0x80 rc=0 handle=4 object=0xb062a980 dma=0x3e4d000 sram=0x0
flags=0x82 rc=0 handle=4 object=0xb062a9a0 dma=0x3e4d000 sram=0x0
...
summary: 128 of 256 flag values accepted a dma-buf fd
```

**128 of 256 flag values succeed, and they are exactly the ones with bit 7 (`0x80`) set**;
with `0x80` clear every value returns `EINVAL`. The driver writes back a device address
(`dma=0x3e4d000` here) and an object handle, and `DESTROY` releases it. So `flags |= 0x80`
is the "this handle is an existing dma-buf fd, import it" selector, and the accelerator can
address a buffer it never allocated - the mechanism a V4L2/ISP capture buffer or an RGA
output needs for a genuinely zero-copy input path.

What this does and does not establish:

* **Established**: the import ioctl accepts a CMA-heap dma-buf and maps it into the NPU's
  address space; the address is stable across repeated imports of the same fd (0x3e4d000).
  The heap is `/dev/rk_dma_heap/rk-dma-heap-cma`; there is no `/dev/dma_heap` and no DRM
  device on this board.
* **Not established**: that a submission reading *from* the imported address produces the
  same bytes as one reading the runtime's own buffer. That needs a runtime API that binds an
  imported region as an input tensor and a board replay against a recorded suite; it is the
  remaining half of F8 and it is now a known-reachable piece of work rather than an unknown
  driver capability.
* **Also relevant to E5**: `/dev/video0..20` exist (`stream_cif_mipi_id*`, `rkisp_mainpath`,
  `rkcif_scale_ch*`, `rkisp_lumapath`), so a capture buffer could feed the NPU through the
  same import path, with `rkipc` holding the ISP.

The runtime keeps using its own allocations until that binding exists; `docs/board.md` and
`docs/board-runbook.md` now say "the driver supports dma-buf import (CREATE flag 0x80); the
runtime does not expose a zero-copy input yet", which is the precise state.

## 2026-09-11: publication-readiness pass - legal, interfaces, CI, docs, examples

The pre-publication audit in `docs/publish-checklist.md` was worked through. The short
version, with the evidence that each item is closed:

**Legal (the blockers).** The four GPL-2.0 Rockchip kernel sources that were sitting in
`research/vendor/` are gone - the README claimed they were not redistributed while they
were - and their provenance (upstream repository, commit, per-file sha256) is retained in
`research/vendor/README.md` plus `provenance.json`. `THIRD_PARTY.md` now lists every non-MIT
component that is shipped or fetched, `docs/provenance.md` states exactly where the register
knowledge came from (board experiments, public kernel sources read as documentation, the
vendor runtime as a black-box oracle), and the README carries the trademark/non-affiliation
note.

**Interfaces that were silently wrong.** `--target`/`--quantize` are validated against the
supported sets and named in the compiled summary; `model.decode` returns the
`register_count` that `encode` needs, so `encode(decode(blob))` round-trips;
`native_input_reference` takes `upper_code` and `compile_native_input` publishes
`meta["clip_upper_code"]`, so the Clip[0,6] profile's *board-recorded* expected bytes can be
replayed (now a test); the dead rank check in `chain.py` is removed and the two defensive
guards are annotated; and the C loader now rejects a repeated or missing external tensor
binding instead of silently dropping a caller buffer. The six documentation claims that
disagreed with the code (checksum coverage, the v5 anatomy, the `batch - 1` header byte,
`ornpu_inspect` reading the whole file, the missing-constants rule, the mmap wording) were
corrected rather than left as footnotes.

**CI builds what ships.** New jobs compile the runtime and every `tests/board_*.c` harness
with the host gcc (CI previously built no C at all), install the built wheel in a clean venv
and smoke-test it from `site-packages`, and validate the distribution with `twine check`.
The matrix is 3.10/3.12/3.13 and the ruff target now matches `requires-python`.

**Documentation and examples.** Generated, drift-guarded `docs/registers.md` (126 register
rows, 46 honestly marked undecoded) and `docs/errors.md` (548 messages across 35 modules,
generated from an AST walk of every `raise`); `docs/support-matrix.md` (82 accepted rows
with bounds, profile, example and board/host evidence, plus 24 rejected constructs);
`docs/glossary.md`; `docs/troubleshooting.md`; `docs/performance.md`; `docs/c-api.md`;
`docs/container-example.md` (a real container walked field by field). `examples/cookbook/`
adds the missing task-oriented examples: a minimal C program, quantized-model import,
mutable parameters, batched/pipelined submission, a wheel-installed smoke test, a rejection
walkthrough and an ONNX compatibility report.

**Tests.** 873 tests (from 760) and 98% compiler line coverage (from 96%), with the ten
remaining lines documented as unreachable defensive branches. New: container fuzzing (~1,460
mutants over real v3/v4/v5/legacy containers, proving the decoder raises only `ValueError`
and that every accepted container is structurally self-consistent), generated-graph
properties across the emitters, reference-doc drift guards, the CLI flag contract, the
Clip-reference board replay, and host runs of the classifier examples. The container
baseline is unchanged (2,244/2,244) and the campaign sweep stays 157 same / 12 pinned drift
/ 0 err.

**One API wart found and closed:** `encode_sequence_v5(constants=...)` used to emit v4-style
constant descriptors that its own decoder then rejected; it now raises a clear error saying
that runtime-replaceable parameters require a v4 container.

## 2026-09-11: test expansion - 311 to 873 tests, 90% to 98% coverage

The published repository's suite was validated by expansion rather than by inspection:
seven workstreams added 449 tests in 16 modules, measured with `coverage` and gated in CI.

**What the new tests establish**

* **Bounds, not just failures** - every profile family now has, for each documented limit,
  the first out-of-bounds neighbour *and* a valid neighbour that compiles
  (`test_profile_bounds.py`, 52 tests). The front end got the same treatment plus semantic
  checks of folding and rewrites against ONNX's evaluator (`test_normalize_matrix.py`, 28).
* **Independent semantics** - `test_emit_semantics.py` (36 cases) implements the graph
  arithmetic in float64 in the test itself (dequantize the container bands, run the op with
  the container's own dequantized weights, requantize) and compares the *container's*
  integer output against it: 0 LSB for delta/identity constructions, <=1 LSB otherwise.
  No emitter reference is used to compute the expected side.
* **Numerics** - band selection, the `channel_multipliers == round(scale/max_scale*16384)`
  invariant, zero-point boundaries, and, importantly, the tie behaviour: constructed
  exact-`.5` accumulators pin round-half-to-even in both the channel and output steps
  (`test_quantization_edge_cases.py`, 19). Calibration gained real measurements and every
  error path for `minmax`/`percentile`/`kl` (`test_calibration_methods.py`, 18).
* **Composer and arena** - alignment, lifetime disjointness, ping-pong reuse, and five real
  compiled graphs checked for `allocated_bytes <= arena`, live-offset disjointness and
  external/internal separation (`test_composer_arena.py`, 18).
* **Containers and loader parity** - emitter meta vs decoded container for six emitters,
  byte-identical re-encode for v3/v4/v5, every v5 validator rejection, legacy ORNPUBIN
  round-trips, and a decode sweep over all 2,340 published suite containers
  (`test_sequence_roundtrip.py`, 13).
* **CLI matrix** - every compile flag, the calibration report, `inspect`, `normalize`, and
  the documented argument errors (`test_cli_matrix.py`, 14).
* **Published evidence replay** - a deterministic 58-model sample recompiles to the retained
  container bytes (55 identical, 3 pinned drifts) and recorded cases replay through the
  builders' integer references byte-exactly (`test_suite_replay.py`, 3).
* **The remaining emitters** - activation profiles (Leaky/PReLU/Clip[0,6]), the reduction
  and multi-input DAG profiles, the legacy single-Conv container, the join emitters
  (two-head/diamond/join-chain), the pool/depthwise joins, pooled branches and
  `QLinearConv`/QDQ import, all with independent float64 references and boundary tests
  (`test_activation_profiles`, `test_reduction_and_dags`, `test_legacy_container`,
  `test_join_emitters_deep`, `test_legacy_compiler_paths`, `test_join_variants_deep`,
  `test_quantized_import_deep`).

**Result**: 873 tests, `OK`, 98% line coverage (12 modules at 100%), engine floor now
`fail_under = 97`. `research/container_baseline.json` is unchanged - the expansion added no
container change - and the campaign sweep stays 157 same / 12 pinned drift / 0 err.

**One real bug fixed**: `compose.compose` silently accepted duplicate stage names (two
stages collapsed onto one program slot, losing a task) and a stage binding the same address
register twice (one binding silently dropped); both now raise explicit errors. No emitted
container changes, because no existing emitter does either.

**Findings pinned rather than "fixed"**: `chain.py`/`chain_n.py` leave the internal border
register `0x1184` at `-128` instead of the producer zero point (measured 38 LSB from the
float model on a seeded 5-layer chain with hidden zero points `[-128,-17,-2,0,0]`; 178 of
242 retained chain models are affected). The board evidence is byte-exact against that
convention, so changing it would invalidate `research/native_chain_suite/`; the tests pin
both the convention and the size of the gap, and `docs/roadmap.md` records it. Two API warts
are recorded there as well (`model.decode` omits the `register_count` that `encode` needs;
`native_input_reference` does not model the `Clip[0,6]` upper clamp).

## 2026-09-11: a trained audio model runs on the NPU (mel-CNN spoken digits)

The Silero question's follow-up: instead of a rejected pretrained zoo model, train a small
audio model that only uses verified primitives and run the *whole* thing on the NPU.
FSDD (3,000 utterances, 8 kHz, CC BY-SA 4.0) -> 3x32x32 log-mel/delta/delta-delta bytes ->
a 4,090-parameter CNN -> 8x8x10 logits averaged per class on the host.
`examples/mel-kws/` has the pipeline; `research/mel_kws_suite/` the board evidence.

Two gaps in the walk had to be closed to make a *trained* model compile at all:

* **native16 image input.** The legacy single-Conv emitter only accepts 5..8 pixels a
  side, so a 3x32x32 image had no path. The walk now stages an out-of-range image as a
  native16 surface and emits the first Conv with `native.py`'s verified register and
  weight builders (single task, 6144-atom limit). The same branch is what makes C3 images
  larger than 8x8 possible at all; the 2,244-model baseline is unchanged because the
  branch is only reachable for graphs the old path rejected.
* **calibrated bands.** The walk was analytic-only, and analytic bounds are worthless for
  a trained network: the first Conv's analytic band came out 20x too wide and the later
  grids collapsed onto the zero point (INT8 accuracy 10%). `compile_chain_walk` now takes
  the `calibration.measure` report (`ranges`), uses each Conv's measured band, and still
  re-quantizes a pool-fed Conv onto a zero-point-0 grid; the scheduler passes
  `calibration_ranges` through instead of rejecting it. Output quantization stays a
  measured band too (the model is trained on the exact input bytes, so input scale is 1).

Board measurement (300 held-out utterances, `rkipc` running): float ONNX **98.33%**, INT8
NPU **98.00%**, **191,968 / 192,000** exact output bytes, 3.47 ms first-sweep / 2.54 ms mean latency,
16,992-byte container with 7 tasks. The 32 differing bytes are four utterances with one
affected output cell each (|delta| <= 4 LSB, no classification change); stage probes show
the first two Convs and the first pool byte-exact and the difference starting in the third
or fourth Conv, i.e. a reference-vs-hardware rounding residual rather than a geometry
error. The ledger row is the 16-utterance byte-exact suite (16 inferences, 10,240 bytes).

## 2026-09-11: the driver ACTION surface measured; Silero VAD is outside the envelope

Two questions - "are we missing any board functions?" and "can we run a small pretrained
model such as Silero VAD?" - were answered on the hardware and against the compiler.

**ACTION surface** (`research/action_probe.c`, `research/action_probe/README.md`).
`runtime/open_rknpu.c` uses only `SUBMIT`/`MEM_CREATE`/`MEM_DESTROY`/`MEM_SYNC`; the
driver also exposes an `ACTION` ioctl, so the probe issues every GET action except
`GET_VOLT` (see below):

* informative: `GET_HW_VERSION` `0x54524548`, `GET_DRV_VERSION` `802` (v0.8.2),
  `GET_FREQ` **420 MHz**, and the three free-running bandwidth counters
  (`DT_WR`/`DT_RD`/`WT_RD`, with `GET_TOTAL_RW_AMOUNT` their sum) plus `GET_IOMMU_EN = 0`;
* inert on this board: `GET_BW_PRIORITY`/`EXPECT`/`TW` return `-EINVAL` (no devfreq/OPP
  policy), `GET_TOTAL_SRAM_SIZE`/`GET_FREE_SRAM_SIZE` are `0` (no SRAM pool), and the
  vendored driver source shows `SET_FREQ` has an empty body, so clock scaling is a no-op;
* **`GET_VOLT` (action 4) oopses the caller**: the device tree has no rknpu regulator
  (`dev_pm_opp_set_regulators: no regulator (rknpu) found: -19`), so
  `regulator_get_voltage(rknpu_dev->vdd)` dereferences an ERR_PTR. The trace ends in
  `regulator_get_voltage` from `rknpu_ioctl`; the board survives (`rkipc` alive,
  submissions keep working). The probe therefore omits it and the runtime never issues it.

So the missing board functions are not ioctls we forgot: they are capabilities this
kernel/device-tree combination does not provide (fences, IOMMU, SRAM residency,
bandwidth policy, clock/voltage control) plus driver fields nothing here needs
(`JOB_PINGPONG`, `priority`, the `task_counter` loop, `user_data`) and the interfaces the
runtime deliberately does not use (`MEM_MAP`, the DRM path, the `SRAM/NBUF/DMA32/SECURE`
allocation flags, dma-buf zero-copy from V4L2/ISP). `docs/plans/pipelining-plan.md` S4 now carries
the pointer: no clock-scaling experiment can close that residual on this board.

**Silero VAD** (`silero_vad.onnx`, opset 16, snakers4/silero-vad) does not fit, and not
because of one missing op: the STFT is a 1-D `Conv` with kernel 256, the four encoder
layers are 1-D rectangular convs with input channels 129, the decoder is two
`LSTM`(128) layers, and the graph is full of `If`/`Shape`/`Gather`/dynamic `Slice`. The
compiler rejects 1-D input (`static NCHW input required`) and rectangular kernels
(`native padding/stride/dilation unsupported`), caps the input at C128 (C129 is
rejected, so the first encoder layer is out of range), has no recurrence or `MatMul`, and
only accepts `Sigmoid` through the bounded LUT profile. The
encoder alone is ~150k MACs per 32 ms frame, so offloading it would not pay either. The
measured table is in `docs/plans/primitive-roadmap.md` ("Real-model envelope"), which also records
the smallest useful audio milestone that only uses verified primitives.

## 2026-09-11: codebase cleanup - dead code, layout, docs (no container changed)

The composer port left weight behind, so the tree was cleaned in three passes
([cleanup-plan](plans/cleanup-plan.md) records the scope):

* **hygiene** - `ruff check src tests --select F,E9` went from **64 findings to zero**
  (6 `F811` redefinitions and 27 unused imports in `elementwise.py`, unused locals such as
  `sequence.payload_start`, `join_dag.pool_basis`, `transposed.centered`, `walk.stem`);
  `pyproject.toml` now pins the lint (`E9`+`F`, line length 120) and `.gitignore` covers
  `.pytest_cache/` and `*.pyc`;
* **dead code** - `elementwise_multi.three_input_reference` and its unused
  `compile_three_input_dag` alias are gone (`multi_input_reference` is what the tests and
  the suite generator call); stale `__pycache__` trees from deleted modules were dropped;
* **research tools** - `research/build_join_chain_suite.py` called `diamond_reference`
  without importing it (and imported an unused sibling), so re-running the generator died
  with `NameError`; the import is fixed and the generator now rebuilds its 12-model suite
  (verified into a scratch directory, leaving the published suite untouched);
* **layout** - the eight suite generators moved from `tests/` to `research/` (the project
  convention, references updated), the three unreferenced `check*.onnx` leftovers moved to
  `research/legacy_onnx/` with a note, and the empty top-level `auto_padding_suite/` plus
  the regenerable `build/` tree were deleted;
* **docs** - `README.md` gained a repository-layout table and module inventory and lost its
  stale "input channels above 64 are rejected" claim (C1..C128 is verified, see the P2
  entry below), and this log gained a title-first structure and table of contents.

**Verification, now reproducible from the tree.** The ad-hoc comparison was turned into
three checked-in scripts: `research/verify_suites.py` (with the pre-cleanup map
`research/container_baseline.json`), `research/campaign_sweep.py` and
`research/check_docs_links.py`. Measured over the final tree:

* **2,244 / 2,244** published suite models recompile to the baseline - 0 changed, 0 added,
  0 removed - and the **46** models the profiles rejected beforehand still reject with the
  same `ValueError`, pinned as `ERR:ValueError` in the map
  (`tests/test_suite_evidence.py::test_container_baseline_covers_every_suite_model` keeps
  the map covering every `research/*suite*/model*.onnx`);
* **873 host tests** pass under both `unittest` and `pytest`;
* campaign sweep **157 same / 12 diff / 0 err**, the 12 now enumerated and pinned in
  `research/COVERAGE_EXPANSION_RESULTS.md` ("Pre-existing container drifts");
* **85 markdown files, 0 broken links, 0 unresolved anchors** (the vendored
  `research/pretrained/mnist/upstream_README.md` links are out of scope), which is also how
  the two duplicated log headings left by an earlier bad prepend were merged;
* the rebuilt wheel has the same **38 modules, byte-identical to `src/`**, and a clean-venv
  install of that wheel recompiles **2,244 / 2,244** baseline entries - the strongest form
  of the wheel-parity claim;
* the then-121-row ledger is unchanged.

**Deliberately not touched**: the `research/` evidence directories, decoded vendor
captures and oracle scripts; the top-level vendor/Ghidra artifacts; the 12 pre-existing
container drifts; the 45 pre-existing lint findings in the `research/` tooling (lint scope
stays `src tests` plus the three verification scripts); and the column-heavy emitters that
are correct but dense (reformatting them would bury the diff for no functional gain).

## 2026-09-11: the walk dispatches the join class, and mixed pool kinds get a path

The op-level walk is now the scheduler's dispatch for the diamond class: the attempt sits
inside the diamond branch (so the other join profiles keep their graphs) and falls back
to `compile_diamond` for anything the walk does not cover, such as Mul joins with operand
zero points. The containers are byte-identical either way - the equivalence test compiles
all 24 diamond/diamond-tail models both ways - so no board evidence moved, and the
profile-name assertions in the diamond, diamond-tail, join-chain, pool-join and
depthwise-join tests now read `walk-join`.

That left one real gap: a graph whose two heads are pooled with **different kinds**
(`MaxPool` on one branch, `AveragePool` on the other) and optionally a Conv tail after the
join. `parse_pooled_branches` matched such graphs and then `compile_pooled_branches`
raised "pooled branches require matching pool kinds", so they died in the profile
instead of reaching any path. The parser now declines mixed kinds (letting the graph fall
through) and the walk lowers it: **6 models, 96 inferences, 4,608 exact output bytes**
(`research/walk_join_suite/`, a ledger row) covering Add/Mul/Max/Sub joins, mixed pool
kinds, K1/K3 heads, 1-2 Conv tails and a stem without Relu.

## 2026-09-11: residual disposition closed (final pass)

Every item of the completion objective now has a disposition and evidence, tabulated in
`docs/plans/completion-plan.md` ("Residual disposition"). Closed with board evidence: S10 batched
submission, S4 fence-free completion (barrier job), S5 per-family cross-check, P7
percentile/KL calibration, P3 dense K3-dilation2 ConvTranspose, P2 input channels to
C128, and P1's emitter port (every emitter composes byte-identically). Closed as measured
negatives: P4 per-channel output conversion and native spatial broadcast, P6
non-power-of-two LUT bands. The op-level walk dispatches chains with pools and the join
class (including mixed pool kinds), with elementwise ops and multi-join DAGs left
profile-matched. FENCE, a second RV1106 board, a vendor-zoo detector and publication are
reported as hardware/user decisions.

Baseline at close: **1,680 models / 28,250 inferences / 10,206,227 exact output bytes**,
**873 host tests**, 120 ledger rows, campaign sweep 157 same / 12 diff / 0 err (the 12 are
documented pre-existing drifts), wheel (38 modules) byte-identical to the tree.

## 2026-09-11: the walk lowers fan-in joins, and the diamond tail Relu was dropped

`open_rknpu.walk` gained a join path: `parse_join_walk` accepts the diamond class (one
`Conv[,Relu]` stem, two heads, one Add/Mul/Sub/Max join, an optional `[Conv,Relu]* Conv`
tail, plus an optional 2x2 pool after each head) and `compile_join_walk` lowers it with
the same band rules the diamond emitter uses (natural head bands, a folded product for a
Mul join, one shared zero-point-0 scale for Add/Sub/Max, then the tail chain).

Comparing it against the profile exposed a **latent fidelity bug**: `compile_diamond`
never set `q.relu` for its tail layers, so a tail `[Conv, Relu]` dropped the Relu - in
the container *and* in `diamond_reference`, which is why the retained `diamond_tail_suite`
board runs still passed. The tail is now quantized like the chain family's hidden layers
(`q.relu = index < len(tails) - 1`, registers `0x4060/0x406c/0x40e0` in the tail stage),
`diamond_tail_suite` was regenerated and re-run on the board, and it passes with the
Relu applied: **12 models, 192 inferences, 36,864 exact bytes**.

With that, the walk and the profile agree exactly: **all 24 diamond and diamond-tail
models are byte-identical** whether compiled through `compile_diamond` or through
`compile_join_walk` (`tests/test_walk.py`), so the op-level join dispatch is equivalent to
the profile's hand-declared stages.

**The same latent bug in `compile_join_chain` is now fixed too.** Its tail had the
identical gap (`q_tails` never set `q.relu`, and its tail stage fields never wrote
`0x4060/0x406c/0x40e0`), so a join-chain tail Relu was dropped from the container and
from the references. Fixed the same way; no retained container changed, because a scan of
every suite shows **no stored join-chain model has a tail Relu** (the only tail Relus in
the repository are the four diamond-tail models fixed above), so there was no board
evidence to regenerate. `tests/test_join_chain.py` now pins the fixed path at register
level: a three-head chain with a `[Conv, Relu]` tail reports `tail_quantization[0].relu
= True`, the hidden tail task carries `0x4060=0x12/0x406c=0/0x40e0=0`, and the final one
carries the inactive values.

## 2026-09-11: the op-level walk lowers pools inside a Conv chain (P1)

`open_rknpu.walk` now compiles a linear Conv/Relu chain with a **pool in any position**
(`Conv -> pool -> Conv`, repeated pools, a pool before the output) by dispatching one op
at a time to a stage builder and letting `open_rknpu.compose` assemble the container.
The scheduler routes a graph to it only when a pool is followed by another node, so no
existing profile can be hijacked (`tests/test_walk.py` checks the boundary and the
campaign sweep stays at its 157/12 baseline).

**Board-verified: 12 models, 192 inferences, 6,368 exact output bytes**
(`research/walk_chain_suite/`, a ledger row). The suite covers MaxPool and AveragePool,
one to three pools, 3/8/16 hidden channels, Conv1x1/Conv3x3, 8x8/8x6/6x8/6x6 grids and
an output that is a pooled grid - the non-8x8 grids are there because the
geometry-dependent registers were the last bug in this path.

Three construction facts came out of the debugging, in the order they were found:

1. **Tensor shapes are `(batch, height, width, channels)`.** Declaring the input tensor
   as `(1,3,H,W)` made the runtime pack a 3-row, 8-channel image, which produced
   *byte-identical wrong outputs* across three unrelated fixes until the shape was
   corrected. The register/weight comparison against the established single-Conv
   container is what proved the payload was right and the packing was wrong.
2. **A Conv that feeds a pool is re-quantized onto a zero-point-0 band** (scale widened
   by `max(128+zp,127-zp)/127`), because the verified DPU pool programs assume a
   zero-point-0 grid - the same policy `pooled_branches` uses. With that, a pool as the
   graph output became exact.
3. **A Conv that reads a pooled grid needs geometry-aware fields.** The DAG emitters'
   `_native_fields` keeps 8x8-derived defaults for the registers that describe the
   surface a task reads and writes (`0x107c`, `0x1080`, `0x118c`, `0x3014`, `0x4030`,
   `0x4034`, `0x405c`, `0x500c`, `0x5010`), which is correct only for the 8x8 grids
   those profiles use. Calling `native_fields` with the *actual* input/output geometry
   closed the last gap; the isolation that found it was a controlled chain whose second
   Conv was 1x1 (exact) versus 3x3 (wrong) after a pool.

## 2026-09-11: first step of the op-level walk (P1) - Conv/Relu exact, pooling open

`open_rknpu.walk` parses a linear graph, walks its nodes, builds one `compose` stage per
node with the verified per-op builders (`compile_model` for the image-input Conv,
`_layer_fields`/`_pack_layer_head` for later Convs, `pool_registers` for a pool) and
tracks each tensor's band as it goes. It is the first path where a pool may sit in the
middle of a Conv chain.

Board diagnostics (2026-09-11, `board_run` on `[1,3,8,8]` graphs):

* single Conv and Conv+Relu: the walk container's registers, weights and bias are
  byte-identical to the established single-Conv profile, and the board output matches
  `chain_walk_reference` exactly;
* Conv+Relu -> Conv: exact, and `chain_walk_reference` reproduces `chain_n_reference`;
* Conv+Relu -> MaxPool/AveragePool -> Conv: **wrong** (164-172 of 192 bytes differ, up
  to 15 codes). The pool task's own registers match the board-verified
  `pooled_branches` pool exactly apart from the two addresses, so the open question is
  the band a pool carries into the next Conv: the verified pooling profiles only pool
  a zero-point-0 band, and re-quantizing the pool-fed Conv onto a zero-point-0 grid the
  way `pooled_branches` does (`_adjusted_scale`) narrowed but did not close the gap.

One construction bug is worth recording because it produced byte-identical wrong
outputs across three different fixes: the v5 tensor table stores shapes as
`(batch, height, width, channels)`, and declaring the input as `(1,3,H,W)` made the
runtime pack a 3-row, 8-channel image.

The module is deliberately **not** reachable from the scheduler (no profile dispatches
to it) because the pooling path is unverified; a mid-chain pool therefore still raises
"unsupported". `research/walk_chain_suite/` retains the failing board run (6 models, 96
inferences, 0 exact) and `tests/test_walk.py` pins the parser, the Conv/Relu
equivalence and the unreachability, so the next attempt starts from a recorded state.

## 2026-09-11: the ERDMA secondary operand is read linearly - no spatial broadcast (P4 closed)

The 2026-09-10 compact-operand sweep hung on every variant and left "spatial broadcast"
as a field-map blocker. The RK3588 TRM for the same NPU IP (`research/hardware_refs/`)
decodes what was missing:

* `0x5034` `erdma_cfg`: `data_mode` 31:30 (0 per channel, 1 per pixel, **2 per channel
  by pixel**, 3 reserved), `surf_mode` 29 (1 or 2 surface series), `data_size` 3:2
  (1 = 8-bit, the value every profile stores), `erdma_disable` 0;
* `0x5040` `ew_surf_stride` is a stride **in 16-byte atoms** (per-channel mode requires
  1, which the verified profile stores as `0x10`; the per-pixel plane stores 36 atoms
  as `0x240`);
* `0x5010` bits 28:16 `ew_line_notch_addr` and `0x506c` `ew_surf_notch` - "how many
  pixels from the end of this process feature map to the end of the shape feature map".

`0x506c` was already in the elementwise register list with default 0, so a compact
operand could be re-probed without changing the program. Six variants of the verified
per-row model (`mul_broadcast_notch_suite/`, 32 cases each):

| variant | `0x5034` | `0x5040` | `0x506c` | result |
| --- | --- | --- | --- | --- |
| baseline (materialized) | `0x40000004` | 0x240 | 0 | exact |
| compact, stride 36 atoms, notch `W-1` | `0x40000004` | 0x240 | 0x40 | runs, linear read |
| compact, stride 1 atom, notch `W-1` | `0x40000004` | 0x10 | 0x40 | runs, linear read |
| compact, stride 36 atoms, notch `W` | `0x40000004` | 0x240 | 0x50 | runs, linear read |
| compact, `data_mode=2` | `0x80000004` | 0x240 | 0x40 | hang (timeout, soft reset) |
| compact, `surf_mode=1` | `0x60000004` | 0x240 | 0x40 | runs, linear read |

Two results. **The notch removes the hang**: with `ew_surf_notch` set every compact
variant completes, so the 2026-09-10 timeouts were a missing notch rather than an
out-of-range address. **There is no broadcast**: all four completed variants produce
byte for byte the *linear read* of the compact table - one 16-byte atom per output
pixel in output order, unchanged by stride, notch or `surf_mode` - and
`tests/test_mul_broadcast_notch.py` recomputes that prediction from the broadcast
suite's own reference before asserting it. Spatial Mul constants stay materialized.
That closes P4 - both of its hardware-mode negatives now rest on decoded fields and
measured board runs rather than retained hangs.

## 2026-09-11: every DAG emitter composes; the scheduler walk is the last P1 item

The round-3 port left `compile_join_chain` hand-assembled because its runtime-scale
profile takes a **second external input**, which the composer rejected. Two changes
close that:

* `compose()` now places **one to eight external inputs** (contiguous external indices,
  the format's own bound) and accepts `late_inputs=`: an operand the profile reads last
  (the join chain's per-channel `scale` row) is placed after the arena and before the
  outputs, while the image input stays at the payload end. The single-input profiles are
  untouched.
* `compile_join_chain` declares one stage per task - the relocated depthwise program
  keeps its 64-byte weight slot while copying only its own block, the runtime-scale tail
  reuses the verified standalone Mul program with the join result and the external row
  as operands, and the runtime residual reuses the join task - and calls the composer
  with `reuse=False` for the runtime-scale profile.

Verification is byte-level and pre-port: a reference of **110 container hashes** captured
from the hand-assembled emitter (`research/composer_port_reference.json`) is recomputed
by `tests/test_composer_emitters.py`, and **all 110 are identical** - the port changed no
byte. Where a retained board artefact equals a fresh compile the test also pins the raw
bytes. Ten models in these suites already differed from their retained `.bin` before the
port (documented drift); their hashes cover them.

`open_rknpu.compose` is therefore the assembler for *every* emitter that builds a DAG
container (`pool_join`, `pooled_branches`, `diamond`/`diamond_tail`,
`join_dag`/`pooled_dag`, `join_chain`/`join_scale`/`join_residual`/`mixed_head`/
`depthwise_chain`, `two_head`). What remains of P1 is the scheduler: it still selects a
whole profile by pattern rather than walking normalized ops one at a time.

## 2026-09-11: the diamond and join-DAG emitters assemble through the composer (P1 port)

`open_rknpu.compose` was already the assembler for `pool_join` and `pooled_branches`;
the diamond and join-DAG emitters still picked program slots, constant blocks, arena
offsets and the tensor table by hand. Both now declare one `Stage` per task and let the
composer do the bookkeeping, and the port is pinned at the strongest acceptance the
plan asks for - **byte-identical containers**, so the retained board evidence stays
valid without a re-run:

* `compile_diamond` (and its `[Conv, Relu]* Conv` tail profile): 24 models byte-identical
  (`diamond_suite`, `diamond_tail_suite`);
* `compile_join_dag` (multi-layer dense/depthwise branches reused across a join tree,
  optional terminal pool): 24 models byte-identical (`join_dag_suite`, `pooled_dag_suite`);
* the composer's declared binding view checks out against every composed container
  (`check_declared_bindings` on the derived register view), and a batched compose equals
  the S10 submission post-pass (`relink_for_batched`) on the same model.

The port reuses the per-op builders unchanged (`native_fields`/`_pack_native_head`,
`_layer_fields`/`_pack_layer_head`, the relocated depthwise program, `_join_fields`,
`pool_registers`), so no emitted register word moved. `compile_join_chain` is still
hand-assembled - the remaining half of the P1 residual, together with replacing the
scheduler's whole-profile dispatch with an op-level walk. `tests/test_composer_emitters.py`
holds the equivalences and a guard that the hand-assembled join-chain suites still
reproduce their retained bytes.

## 2026-09-11: the elementwise output stage has no BS-table read (P4 sub-item closed)

The retained `BS_OW_CFG.OW_SRC=1` failure was rebuilt with both construction errors
fixed: the table now sits at the **payload** address `0x5020` names (the first probe
wrote it at the *file* offset, 0x80 bytes away) and uses the decoded Conv block layout
(32 bytes per four output channels: INT32 bias, INT16 `-weight_zero_point`, UINT16 Q14
multiplier at +24). Six variants of the same 8x6x3 scalar-constant Mul were run over the
retained 32-input batch (`mul_per_channel_ow_suite/`, `ow_results.json`):

| variant | EW `0x4050` | table | result |
| --- | --- | --- | --- |
| `model000` | `0x30000002` (verified) | absent | exact, 32/32 |
| `model001` | `0x30000001` | file-offset 4xUINT16 (superseded) | hang |
| `model002` | `0x30000001` | decoded block, unit multipliers | hang |
| `model003` | `0x30000001` | decoded block, ch1-2 zero | hang |
| `model004` | `0x30000003` (`OW_SRC=1`, `OD_BYPASS=1`) | decoded block, ch1-2 zero | exact, **identical to baseline** |
| `model005` | `0x30000002` | decoded block present | exact, identical to baseline |

`OW_SRC=1` hangs whether or not the block is well formed, so the failure is a property
of the field. With `OD_BYPASS=1` the task completes and never reads the block - two zero
channel multipliers leave the output byte-identical, so channels 1 and 2 are not scaled.
Clearing the bypass is not itself the problem: the DAG suites' two-surface elementwise
tasks run at `0x30000000`. The elementwise output stage therefore has no BS-table read,
and the accepted per-channel Mul route stays the 1x1 depthwise lowering with native
per-channel weight scales. `tests/test_mul_per_channel_ow.py` pins the variants
(regenerating them deterministically) and the retained board records.

## 2026-09-11: the per-family cost table cross-checked on held-out containers (S5 residual closed)

The duplication probe (`family_tasks_probe/`) measured *independent, overlapping* copies
of one task descriptor; the residual was that the median fits are load-dominated and the
table had never been checked against real dependency chains. `family_cost_crosscheck/`
selects **23 already board-verified containers from eight suites by their (conv, pool,
elementwise) task counts alone** - `(1,0,1)`, `(1,1,0)`, `(1,2,0)`, `(2,0,2..4)`,
`(3,0,1)`, `(3,0,3)`, `(4,0,2..3)`, `(5,1,2)`, `(5,3,2)`, `(6,1,2..3)`, `(8,0,8)`,
`(8,1,3)`, `(10,3,2)`, `(16,0,16)` and the pure-CNA chains 3/4/8/12/16 - and benches each
with `board_bench`. Two independent 64-run board passes are retained; **all 23
containers are exact in both**.

* **The CNA row cross-checks:** least squares over the held-out set alone gives conv
  **13.9 +- 1.0** and **13.4 +- 0.9 us/task** against the table's 12.2; the minimum over
  the same 23 containers repeats within **0.90-1.40x (median 1.01x)** between the runs,
  so that is agreement at the floor statistic's own spread.
* **The elementwise row is profile-specific, not family-wide:** real two-surface EW tasks
  cost **11.8-14.5 us/task** against the table's 20.3. The enable/register key (24, 78
  words) is identical, but the template is the *runtime-scale* Mul (`0x5018=0`,
  `0x5034=1`) while the DAG suites run two-surface EW (`0x5018=0x3000`,
  `0x5034=0x40000004`, `0x5038`/`0x5040` naming a second surface).
* **The pool row is not identifiable** from real containers (+-4 us): pool only appears
  with one to three copies next to varying conv counts.
* Slope-only predictions land at a median **-4.4% / -6.8%** error (mean +7.5% / +9.9%)
  over the 23 containers, with short pure-CNA chains up to **2.3x** the prediction: real
  chains pay a per-container fixed cost that idempotent, overlapping duplicates hide.
* Noise is now measured, not assumed: at 64 runs the **median is 1.24-6.23x the minimum**
  (median 1.68x), which is why the table uses minima.

`tests/test_family_cost.py` pins both runs, the exactness, the repeatability window, the
held-out-versus-table comparison, and recomputes each recorded prediction from
`family_tasks_probe/family_tasks.json`.

## 2026-09-11: input channels C1..128 - the C64 cap was ours, not the hardware (P2 closed)

The recorded P2 blocker ("input above C64 needs channel-split accumulation and an
undecoded INT32 partial-sum surface") was tested against the vendor oracle instead of
being assumed. Two new marker fixtures (`build_native_c128_oracle.py`) were built with
the vendor toolkit and captured on the board with `run_oracle.py`:

* C65 (`capture_native_c65_channels`) and C128 (`capture_native_c128_channels`) both
  submit **one** 104-byte CNA task (`SUBMIT n rc=0 size=104`) and allocate a 5/8-plane
  `NC1HWC2` input (`ATTR 8 dims=1,5,6,5,16` / `1,8,6,5,16`), and both produce output.
  So the CNA reads more than four 16-lane planes directly - no partial-sum surface, no
  second task, no requantization pass;
* the only vendor register fields that move with the plane count are the lane-derived
  ones we already emit (`0x1024=(C-1)<<16|lanes`, `0x1030=0x1034=k*k*lanes`,
  `0x1088=lanes`, `0x1188=k*k*lanes/2`, bias after the table). `0x1010` also changes,
  but our own C64 emitter already differs from the vendor there (0x48 vs 0x70) and is
  board-verified, so it is a scheduling hint, not a correctness gate;
* the weight table is the 32-lane-part generalization of the C33..64 rule: all taps of
  lanes 0..31, then all taps of lanes 32..63, with a trailing part holding the last 16
  or 32 lanes, inside each 16-output block. A third fixture with C65 input and **17**
  output channels (`capture_native_c65_oci`, 1,105 marker cells) confirms the
  second-block term `block*16*k*k*lanes`.

`research/build_native_c65_suite.py` regenerates 20 independent models (input
C65/80/96/128, output C1/3/8/16/17, K1/K3, 6x5 and 8x8) with expected bytes from the
documented native integer reference; the board passed **20 models, 320 inferences,
127,744 exact output bytes** on the first run (`native_c65_suite/`, also the reason the
Python and C loader bounds move from 64 to 128). `research/analyze_native_c48_layout.py`
validates all seven marker captures and `tests/test_native_c48.py` (11 tests) reproduces
every marker set. The rebuilt wheel (`dist/open_rknpu-0.1.0.dev0*`, 37 modules)
compiles the new C65/80/128 profiles byte-identically to the working tree, so the
packaging claim still holds.

## 2026-09-11: residuals closed - universal batched submission, percentile/KL calibration, fence-free completion

Three documented residuals are done, each with board evidence.

**Every container is one job (S10 residual).** The emitters that build containers by hand
or one task at a time never learned `submission='batched'`. Instead of threading `serial`
through each, `sequence.relink_for_batched(data)` rewrites only the tails (next-program
link plus the successor's fetch amount) and header flag bit 0, and `compile_sequence`
applies it to whatever an emitter produced; a legacy `ORNPUBIN` container has no task
table, so the mode is a no-op there (one program is one ioctl anyway).
`research/batched_all_probe/` runs seven of those containers in both modes: 3 to 48
tasks, all exact, one ioctl each (`add_suite` 3 tasks 64.8 -> 29.5 us, `mul_batch_suite`
48 tasks 889.0 -> 445.4 us, `mul_batch_broadcast_suite` 32 alternating-engine tasks
409.8 -> 128.6 us). A sweep of every `research/*/model000.onnx` now compiles 156 of 161
with the mode, and the five that fail do so identically without it (profile bounds, not
submission).

**Percentile and KL calibration (P7 residual).** `calibration.measure(method=...)` now
takes `minmax`, `percentile` (upper-tail cut) or `kl` (TensorRT-style saturation: build a
2048-bin histogram, quantize every candidate range to 128 levels with saturation folded
into the top level, keep the minimum-KL threshold). The first KL formulation evaluated
bin-count quantization, which is degenerate at `i == levels` (divergence exactly 0, so it
always clipped the least); the value-width formulation used here does not.
`research/percentile_calibration_suite/` compiles the same graph four ways and measures
the trade-off against the ONNX float reference: bulk MAE 10.16 analytic -> 1.89 min/max
-> 1.01 percentile -> 0.31 KL, with extreme-input max error 23.5 -> 850 -> 1569 -> 1675.
Measured ranges beat the analytic one on the bulk; tail clipping buys that with the
saturated extremes, which is the deployment choice the numbers document. All four
variants are board-exact (4 models, 76 inferences, 14,592 bytes) and the suite is a
ledger row.

**Fence-free completion (S4 residual).** `JOB_FENCE_OUT` still returns -EINVAL (no
`CONFIG_ROCKCHIP_RKNPU_FENCE`), but jobs run in order per core, so a small *barrier*
submission after a queued one completes only after it: queue with `ORNPU_JOB_NONBLOCK`,
run the barrier blocking, `ornpu_sync_outputs`, read. `tests/board_barrier.c` measures
lag-0 completion on two containers: a 7-task serial mixed-engine graph is *faster* with
the barrier (183.5 -> 134.5 us, one wait instead of seven blocking ioctls) and a 16-task
one-job chain pays the barrier's own 24 us (107.3 -> 134.5 us), where the lag-1
two-instance drain pipeline remains the better throughput choice. What the barrier cannot
give - a pollable fd for multi-process or dma-buf sharing - still needs the kernel
config, which stays a deployment decision rather than a code change.

## 2026-09-11: the tail control is the successor's fetch amount - so every DAG is one job (S8 corrected, S10)

The "engine hand-off" table recorded below was wrong, and the board said so twice. The
tail's register `0x14` is not a routing code: it is the value the driver itself would
write into `PC_DATA_AMOUNT` for the task the tail links to -
`(regcfg_amount + 4 + 2 - 1)/2 - 1` (`rknpu_job.c`, extra amount 4, RV1106 scale 2).
That is why the observed values happened to be `0x40` before a 126-word Conv, `0x14`
before a 37-word pool and `0x28` before a 78-word elementwise task: they are 64, 20 and
40 - the fetch amounts of those programs, not engine codes.

The earlier sweeps that "proved" Conv->elementwise, elementwise->elementwise,
elementwise->Conv and Pool->Conv impossible were confounded three ways: they set *both*
transitions of the probed run to the same candidate, they never tried `0x28` (which is
the amount for a 78-word program), and the DPU->CNA sweep ran on top of a prefix that
already carried `0x14` at a Conv->elementwise step. Single-transition experiments with
the successor's amount pass every one of them:

* `mixed_head_suite/model000` (4 Conv + 2 elementwise) with links 0..3 and `0x28` at the
  Conv->elementwise step: **exact**; the same container with `0x14` there: timeout
  (retained counter-example);
* the same container linked 0..4 with `0x28` at both elementwise steps: **exact, one
  job**;
* `mnist_pool_suite/model000` with the vendor's Conv->Pool `0x14`: **exact, one job**.

`open_rknpu.compose.amount_control` derives the value and `batched_layout` validates it.
Because it is a fetch size, it must be recomputed when the successor's register count
changes - which is why the old fixed table produced working containers for chains and
failures for graphs mixing program sizes.

Consequence: `submission='batched'` is now one job for every graph. The four DAG
emitters threaded in this pass (`graph.compile_diamond` for diamond/diamond-tail,
`depthwise_join.compile_depthwise_join`, `join_dag.compile_join_dag` for join-DAG and
pooled-DAG) join the composer and join-chain emitters, and `research/grouped_probe/`
verifies eight containers of 4-9 tasks in one job, exact, at roughly half the serial
minimum (for example pooled-DAG 8 tasks: 8 ioctls / 170.6 us serial against 1 ioctl /
51.0 us). Serial emission stayed byte-identical for every emitter, and the runtime's
per-run derivation (S10) is kept: it is what reads the runs out of the tails and it keeps
a container with no links safe.

## 2026-09-11: the chain family applies its hidden Relu now (S9 fixed)

The S9 finding (below) is fixed and re-verified. `chain_n` marks each native layer
`q.relu = index < count - 1` and writes the activation registers for the layers that
carry one, so a hidden `Relu` is no longer dropped; `chain.native_reference` clamps the
accumulator at zero (`if q.relu: acc = maximum(acc, 0)`) before the output conversion,
which is where the hardware clamps, and `tiled_chain` follows each layer's metadata so
the tiled containers apply it too.

The change is surgical: patching the four affected suites moved exactly three registers
per hidden layer (`0x4060` `0x13`->`0x12`, `0x406c`/`0x40e0` `0x80000000`->`0`), the
serial emission of every other suite is byte-identical, and `chain_output_quantization`
correctly keeps its last layer activation off. Rebuilt and re-run on the board:
`native_chain` 5 models/80 inferences/15,360 bytes, `chain_reuse` 5/80/15,360,
`deep_chain` 3/48/9,216, `chain_multi` 4/32/40,448 - the same counts as their ledger
rows, with the Relu applied - plus the deep chain's serial/batched/reuse twins and both
tiled probes, all exact. The `tiled_chain_probe/model_tiles2_pre_s7.bin` counter-example
is retained: it is the pre-fix container, which relied on the 1x1 scan path ignoring the
clamp registers.

## 2026-09-11: engine-run splitting gives DPU->CNA graphs the fewest possible jobs (S10)

A non-serial container now declares its runs in the tails, and the runtime submits **one
ioctl per maximal linked run**: a task whose link (`0x10`) is zero ends a run, a linked
task continues it. `compose` and the join-chain emitter merge adjacent tasks while the
measured hand-off control exists (`0x40` inside a run, `0x14` CNA->DPU) and end the run
where it does not, and the runtime derives the same list from the same words at load
time (`compute_runs`, exposed as `ornpu_info.engine_runs`). Container bytes are
otherwise unchanged: on the composer path the grouped form of a single-run graph is
byte-identical to the one-job form.

The board run then corrected S8's control table: the hand-off depends on the **task
family** (the enable value), not the engine class. `0x40` continues Conv->Conv,
Pool->Pool and Pool->elementwise; `0x14` is Conv->Pool. Conv->elementwise has no encoding
at all (17 candidates swept), and neither does elementwise->elementwise (6) or
elementwise->Conv / Pool->Conv (12). Treating pool (`96`) and elementwise (`24`) as one
"DPU" class had made the first grouped `mixed_head` container write `0x14` at a
Conv->elementwise step, which timed out on every run.

Consequences measured on the board (`research/grouped_probe/`, 4 cases x 16 runs, clean
boot per model; `ioctls` is what the runtime reports):

* `mixed_head_suite/model001` (Conv x4, elem, elem, Conv; 7 tasks): 4 ioctls vs 7 serial;
* `mixed_head_suite/model006` and `join_chain_suite/model005` (9 tasks): 5 ioctls vs 9;
* `mixed_head_suite/model000` (6 tasks): 3 ioctls vs 6;
* `pool_join_suite/model000` (6 tasks): 1 ioctl (its transitions are all measured), with
  the container byte-identical to the S8-verified one;
* a container whose tails are all terminal is one run per task, so the flag-flipped
  probe twins that used to hang the front end now complete correctly. The hardware rule
  they demonstrated (one job with terminal tails runs only its first task) is unchanged
  and still retained in `research/group_probe/job_results.json`.

The earlier grouped-submission sketch `research/run_grouped_compare.py` referenced an
`ornpu_set_grouped` runtime hook that was never built; it is replaced by
`research/run_grouped_probe.py`.

## 2026-09-11: the tail control word names the engine hand-off; mixed DAGs fit one job (S8) [SUPERSEDED IN PART]

The tail's second word (register `0x14`) is not a constant "continue": it names the
transition out of the task. The vendor captures had already shown it - `0x40` in front of
another Conv task (`capture_chain4`), `0x14` in front of a pool task
(`capture_pool_max`, `capture_identity`) - and the S1/S2 probe missed it because it wrote
`0x40` at every transition and concluded "one engine per job".

`research/run_job_field_probe.py` (18 attempts, one clean boot each, verified serial
containers changed only in the submission flag and the two link words per task):

* `mnist_pool_suite` (Conv CNA -> Pool DPU), linked with control `0x14`: **PASS**, exact;
* `pool_join_suite` (3 CNA + 3 DPU), linked `0x40,0x40,0x14,0x40,0x40`: **PASS**, exact;
* the same list with `0x14` at the DPU->DPU transitions: timeout - so `0x40` is the
  in-run continue and `0x14` is the CNA->DPU hand-off;
* `mixed_head_suite` (4 CNA + 2 DPU + CNA): every one of 12 candidate controls at the
  DPU->CNA transition timed out. The front end switches into the DPU group but not back.
  **[WITHDRAWN 2026-09-11:** that run's prefix carried `0x14` at a Conv->elementwise
  step, so it never reached the transition under test; see the amount-rule entry above.]

`open_rknpu.compose` now writes the measured control per transition and refuses a graph
whose order needs a DPU->CNA hand-off; `research/mixed_batched_probe/` verifies four
composer-emitted mixed containers (6, 6, 7 and 9 tasks) as one job, exact against the
suites' own expected bytes, at about half the serial minimum (33/35/36/45 us against
71/86/87/114 us).

The other two submission fields are inert here: `rv1106_rknpu_config` uses the
single-entry `rknpu_irqs`, so `rknpu_job_alloc` forces `core_mask = CORE0` and
`subcore_task[]` is only read when `num_irqs > 1`. `tests/board_core.c` submits the same
container with `core=0x7`, five nonzero windows and `core=0x1` and gets byte-identical
output every time (32 runs per configuration, `job_field_probe/core_fields.txt`). The
runtime exposes the probe hook as `ornpu_set_submit_core()`.

## 2026-09-11: K>1 strip tiling needs the pad value and the real activation (S7), and the chain family drops hidden Relu (S9)

`compile_tiled_chain` now covers the whole chain family. A 3x3 strip reads one halo row
from each neighbour, so the K>1 path switched from isolated strip surfaces to one shared
double-buffered 8x8 native16 surface per layer: every strip writes its own rows and reads
its `tpt`/halo window straight out of it, the same geometry the native emitter's
6144-atom height tiles use. The first board run mismatched on every tiled case, and
container surgery found two independent causes:

1. **Pad value.** Hidden native surfaces hold the 128-shifted INT8 value, so the engine
   input zero point is 0 in that domain: register `0x1184 = 0xff80`, the default the
   untiled chain's native layers use. The tiled emitter first passed the layer's
   `output_zero_point` (`0xff00`), which only changes the padded rows of a 3x3 strip -
   and every case was wrong.
2. **Activation.** The tiled emitter applied the graph's hidden Relu; the untiled
   `chain_n` emitter does not. Patching only the activation registers of an otherwise
   exact 3x3 tiled container made it mismatch, so those registers are live for the
   multi-row (K=3) scan. The same A/B on a 1x1 tiled container is exact both ways: on
   this IP the single-row scan path ignores the clamp registers, so a K=1 hidden Relu is
   silently a no-op either way.

The tiled path now takes each layer's activation from its quantization metadata, so it
is a drop-in tiling of the untiled chain, and S9's fix will propagate automatically.
Board evidence: `research/tiled_k3_probe/` (16-layer 3x3 and 12-layer mixed chains,
tiles 1/2/4, serial and one job, 4 cases each, all exact) and the S3 suite re-run over
all 16 inputs with batched twins (`research/tiled_chain_probe/`); `tiles=1` is the
single-strip control and the loader's 64-task table is enforced.

**S9 finding.** `chain_n` emits its native hidden layers with the activation registers
off (`0x4060 = 0x13`, `0x406c = 0x40e0 = 0x80000000`) and `chain_n_reference_layers`
uses a reference that ignores the `relu` flag, so a `Relu` after a native layer is
dropped from both the container and its expected bytes. The legacy first layer does
apply its Relu. That the encoding itself is right is visible in `capture_native_h64k5`
(a native16 program with `0x4060 = 0x12`). Fixing it means setting the hidden layers'
`q.relu`, writing the activation registers in the untiled native path, making the
reference apply `max(acc, 0)`, and re-running the affected chain suites; it changes
accepted container bytes, so it is recorded as S9 rather than folded into S7.

## 2026-09-10: height-strip tiled chains are exact (S3)

`open_rknpu.tiled_chain.compile_tiled_chain(model, tiles=T)` emits an 8x8 1x1 Conv chain
as T height strips: for every (layer, strip) pair a native16 program with strip geometry
through the extracted `native.native_fields(...)`, reading the strip's rows of the
previous surface and writing its rows of the next, with one double-buffered pair of
strip surfaces per strip. Quantization/weights/bias come from the untiled emitter, so
both containers share one integer reference and one expected file. Exposed as
`compile_sequence(..., tiles=N)` and `open-rknpu compile --sequence --tiles N`.

Board evidence (`research/tiled_chain_probe/`, 16 cases each, identical input/expected
bytes as the untiled 16-layer container):

| tiles | tasks | strip rows | min | median | mismatches |
| --- | --- | --- | --- | --- | --- |
| 1 | 16 | 8 | 218.5 us | 287.6 us | 0 |
| 2 | 32 | 4 | 432.8 us | 552.4 us | 0 |
| 4 | 64 | 2 | 831.0 us | 1210.4 us | 0 |

All three are byte-exact, so the strip geometry, per-strip double buffering, 16-lane
weight/bias packing and the quantization chain are correct. The measured trade at 8x8 is
memory-vs-time with no overlap to exploit (a job may not mix engines): latency grows
roughly linearly with the task count and the arena grows as well, because a task program
is 1088 B and there is nothing to overlap with. Tiling stays an option, not a default.
`K > 1` strips need halo rows in the strip input; the emitter rejects non-1x1 kernels.

## 2026-09-10: per-family task cost from duplicated-task fitting (S5)

Differencing two containers (`research/family_cost_probe/`) was inside the noise, so the
same families were re-measured by repeating a *verified* task descriptor N times inside
one container (`research/family_tasks_probe/`): every copy reads and writes the same
addresses, so the extra runs are idempotent and the expected bytes stay valid, and
`cost(N) = C + N * task` can be fitted over N = 1..32. All 18 variants are exact over 64
runs each. Min-based least-squares fits (the board floor; medians carry up to 9x load
noise and give negative intercepts):

| family | intercept | per task | R2 |
| --- | --- | --- | --- |
| conv (CNA) | 6.7 us | 12.2 us | 1.00 |
| pool (DPU) | -4.5 us | 15.4 us | 0.98 |
| elementwise (DPU) | -22.2 us | 20.3 us | 0.97 |

The ordering matches program size (126 / 37 / 78 register words) plus the DPU store
path, and the near-zero intercepts say the fixed per-inference cost is a few
microseconds once tasks are in flight. Also in this round the verified native16
per-task field block was extracted as `native.native_fields(...)` with explicit
input-row and surface geometry (`tests/test_geometry_expansion.py` and the full suite
stay byte-identical), which is the foundation for the strip-tiling emitter.

## 2026-09-10: per-family task cost is below the 8x8 noise floor (S5 residual)

The per-family cost table was attempted with matched containers
(`research/family_cost_probe/`): one 1x1 Conv at 8x8 (one CNA task), the same Conv
weights plus a 2x2 MaxPool (CNA + DPU pool), and the verified Conv + per-channel Mul
container (CNA + DPU elementwise). All three are exact on the board over 128 runs.
Differencing against the single-task container gives *negative* marginals (pool
-15 us, elementwise -7 us on the minima), because the one-task container measures no
faster than the two-task ones (median 403 us vs 89/66 us; min 45 us vs 30/39 us): the
per-inference fixed cost (25-45 us) and +-15 us of CPU-load jitter exceed a
few-microsecond task marginal, and serial medians on this board swing up to 9x. The
reliable per-task figures therefore stay the batched-list ones (3-7 us per CNA task in
one job, ~14 us serial). A real per-family table needs a single-family **multi-task**
profile so the cost can be fitted as a slope over N; pool and elementwise task lists
need an emitter addition.

## 2026-09-10: non-blocking submission pipelines 1.5-2.25x; fences need kernel config (S4)

The `SUBMIT` ioctl's job flags were previously unused except `PC|PINGPONG`. A probe
harness (`tests/board_async.c`) with new experimental primitives in the runtime
(`ornpu_submit_flags`, `ornpu_wait_fence`, `ornpu_sync_outputs`, `ornpu_set_input`)
measured, over 48 interleaved paired rounds (one round = two inferences, medians, so
board-load drift cancels):

* `JOB_NONBLOCK` is accepted (`rc=0`); the queued job's output matched the reference
  and the expected bytes once drained;
* a queue-then-drain pipeline - queue inference A non-blocking, run B blocking (B's
  completion implies A's because the driver runs one job at a time per core in order),
  then read A - is **2.25x** faster than synchronous submission for a batched 3-conv
  chain (195 us vs 87 us per pair) and **1.49x** for the same chain submitted as three
  serial jobs (220 us vs 148 us);
* `JOB_FENCE_OUT` and `JOB_FENCE_IN` return **-EINVAL**: the running kernel was built
  without `CONFIG_ROCKCHIP_RKNPU_FENCE` (`rknpu_job.c` returns -EINVAL for both fence
  branches under `#else`). So there is no pollable completion fd; a pipeline must drain
  with a following job and outputs lag one inference, and a single-instance stream that
  consumes each output as it completes needs that kernel config or a double-buffered
  output arena. Evidence: `research/async_probe/`.

The synchronous path is byte-identical for every committed container (the whole
campaign re-passed after the refactor).

## 2026-09-10: double-buffered intermediate surfaces (S3)

The 8x8 N-layer chain can keep only **two** intermediate surfaces live (layer L writes
buffer (L-1)%2) instead of one per layer. New twins in `research/deep_chain_suite/`
(`reuse/`, `reuse_batched/`) verify all four combinations (serial, batched, reuse,
reuse+batched) byte-exact on the board over 128 runs each:

| layers | arena untiled | arena reuse | serial | reuse serial | batched | reuse+batched |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 28,672 B | 20,480 B | 369 us | 184 us | 93 us | 94 us |
| 12 | 36,864 B | 24,576 B | 1537 us | 721 us | 240 us | 218 us |
| 16 | 40,960 B | 24,576 B | 1120 us | 645 us | 256 us | 186 us |

The arena stops growing with depth (24,576 B for 12 and 16 layers) and the chain is
about twice as fast with two live surfaces - with 16 buffers the 1 KB native16 planes
stream through DDR on every layer. Median serial timings scatter because N blocking
submissions interleave with the CPU; the batched medians are stable within ~20%.

Height-strip tiling was **not** implemented: a job may not mix engines (S1/S2) and the
driver runs one job at a time per core, so a strip pipeline has no engine to overlap
with; the chain emitter also hard-codes 8x8 geometry fields (the first layer comes from
the legacy profile requiring H,W >= 5, later layers use the fixed native fields). It
stays interesting only for memory on inputs larger than the arena, where compile-time
height tiling already covers a single Conv (`native.py:tile_geometry`); lifting that
per-layer builder into the chain emitter plus halo handling for K > 1 is the
prerequisite, recorded in docs/plans/pipelining-plan.md S3.

## 2026-09-10: batched job = linked single-engine list; up to 11x on deep chains (S2)

The first batched-submission pass (entry below) concluded that a job is limited to two
tasks and that next-command links do not help. Both were **wrong**: the link register
`0x10` is *payload-relative* (the runtime programs `PC_DMA_BASE_ADDR` with the payload
base), and that probe wrote `payload + offset`, so every linked case jumped past its
program. The corrected probe (`research/group_probe/`, a clean boot per case, with
`tests/board_bench.c`) measured:

* 2 CNA tasks with terminal tails, batched: timeout (-110);
* 3 CNA tasks with the correct link, batched: **PASS**, 32/32 exact;
* 4 tasks over a two-program loop with links: **PASS**;
* 8, 32 and 64 tasks over a loop with links: **PASS** at every depth (64 is the
  loader's `MAX_TASKS`; `PC_TASK_CONTROL` carries the count in 16 bits);
* a six-task v5 pool join (CNA+DPU) with links: timeout, with per-task masks and with
  the job-union mask `0xF00`.

So a job submitted in one ioctl runs its whole list when every task links to the next
program (`0x10` = next payload-relative program offset), the last task is terminal, and
every transition carries the control measured for its engine hand-off. **[SUPERSEDED in
part, 2026-09-11 (S8):]** this entry concluded "all tasks use one engine, mixed lists
cannot be batched"; the probe wrote `0x40` at every transition, which times out at a
CNA->DPU one. The control is `0x40` inside a run, `0x14` for CNA->DPU and `0x28`
terminal, so a CNA-run-then-DPU-run graph does fit one job. **[Further corrected
2026-09-11: the control is the successor program's fetch amount, not an engine hand-off
at all, so every transition links and every DAG is one job - see the amount-rule entry at
the top of this log and `research/grouped_probe/`.]**

Value, measured on the same graphs in both modes over 128 runs
(`research/deep_chain_suite/`, real 8/12/16-layer Conv chains, plus loop jobs):
3 tasks 67 us serial vs 116 us batched (batched loses); 8 layers 1325 vs 194 us
(6.8x); 12 layers 1932 vs 176 us (11.0x); 16 layers 345 vs 163 us (2.1x); 32 tasks
3172 vs 229 us (13.8x); 64 tasks 1528 vs 449 us (3.4x). The model is roughly
serial ~ 22 us/task plus a per-ioctl cost against batched ~ 100 us + 8 us/task, so the
crossover is near eight tasks per same-engine run. Serial stays the default; the
composer and the N-layer chain emitter emit linked single-engine batched containers on
request and refuse a mixed list (`open_rknpu.compose.batched_layout`).

## 2026-09-10: batched submission runs only a two-task single-engine job (S1) [SUPERSEDED]

## 2026-09-10: batched submission runs only a two-task single-engine job (S1)

The NPU front end takes a whole task list per ioctl (`PC_TASK_CONTROL` = `(0x6 |
task_pp_en) << pc_task_number_bits | task_number`, ping-pong from `JOB_PINGPONG`),
and our runtime already submits that way whenever the container header asks for it
(`serial=0`). Almost every verified container is serial, so the question was what a
batched job actually requires. `tests/board_bench.c` (new) repeats one inference and
reports median/min/max plus `unstable`/`mismatches`, and `research/submission_probe/`
builds the probe twins by flipping header bit 0 of byte 84, which leaves programs,
tensor table and arena byte-identical - so any difference is caused by submission
alone. `research/run_submission_rule.py` runs each case after a clean boot, because
a timed-out job leaves the NPU wedged until reboot.

Measured on the attached RV1103 (each case after a reboot):

* two conv tasks, one engine (`chain_output_quantization_suite`, the only batched
  container the project ever emitted): **PASS**, exact and stable, 16 runs;
* the same container with its programs padded onto the vendor 0x440 grid: still
  PASS - layout is not the constraint;
* two-task mixed engine (DPU+CNA: `mnist_pool_suite`, `mul_broadcast_mode_suite`;
  CNA+DPU: v5 `runtime_scale_suite`): **timeout (-110)**;
* three conv tasks with explicit next-command links (`0x10 = next program`,
  control `0x40`, exactly the `graph.py` chain layout): **timeout (-110)**;
* six-task v5 pool join, with links and/or the job-union interrupt mask 0xF00:
  **timeout (-110)**;
* the driver log for a mixed case shows `task counter: 1` with `require mask: 0x300`:
  the front end finishes the first task and never signals the rest, and
  `rknpu_fuzz_status` collapses the CNA/DPU status groups so a single job mask
  cannot cover both engines.

Conclusions, now enforced in `open_rknpu.scheduler.batched_supported`:
batched submission is accepted only for a two-task single-engine list (the 67
verified CNA+CNA containers keep working); everything else stays serial, and
`compile_sequence(..., submission='batched')` refuses with that reason instead of
emitting a container that hangs the board. Two follow-ups are tracked in
`docs/plans/pipelining-plan.md`: engine-group splitting (one same-engine job per group with the
existing synchronous wait between groups), and decoding what `PC_DMA_BASE_ADDR` /
`subcore_task[]` expect for 3+ task lists (the vendor user space fills per-engine
subcore lists that our runtime leaves zeroed). Timing baselines for 8x8 profiles:
two-task conv ~30-60 us, three-task chain ~78 us, six-task v5 pool join ~125 us
medians; individual runs scatter up to a few milliseconds because the CPU and the
camera service share the core.

## 2026-09-10: stage composer and declared bindings (P1)

The v5 emitters assembled their containers by hand: literal program offsets, a
hand-picked constant cursor, a hand-written tensor table and a hand-listed task
order. `open_rknpu.compose` replaces that with one declared pass.

A `Stage` names the tensors it `reads`/`writes`, the register words it sets
(`fields(addresses, constants)`), the constant blocks it owns and the address
registers it binds. `compose` then: orders the tasks topologically with
`open_rknpu.liveness`; places the arena (or gives every internal a fresh slot when
`reuse=False`, which some profiles need after the stale-data sightings); assigns
program slots `(words + 4) * 8` apart in declared order and constant blocks after
them; substitutes the planned addresses into each stage's register words; generates
the tensor table and the task list; and returns `meta['declared_bindings']`.
`compose.derive_bindings` reads the *same* container back from its address
registers, and `check_declared_bindings` proves the two agree (offset-exact per
register, identical register sets). The task-family registry
(`FAMILY_BY_SIGNATURE`, with read/write registers per family) is shared by the
composer and `tests/test_container_bindings.py`, replacing that test's hand-written
family table.

Two profiles are ported onto the pass and **recompile byte-identically**:
`pool_join` with arena reuse (12 models,
`tests/test_pool_join.py::test_recompiles_byte_identically`) and `pooled_branches`
on the `reuse=False` fresh-slot policy (13 models,
`tests/test_pooled_branches.py::test_recompiles_byte_identically`) — the strongest
available proof that the pass reproduces hand-built layouts. Both were re-run on the
board afterwards (18,432 and 19,968 exact bytes). `tests/test_compose.py` covers the pass itself: an out-of-order declaration
still emits a topological task order while program slots follow the declaration,
address substitution lands in the emitted words, a broken declaration is rejected,
an unknown tensor or an unwritten read is rejected, and the registry covers every v5
signature in use.

Still open: `join_dag`, `pooled_branches`, `join_chain` and `diamond` keep their own
assembly (fixed 0x440 strides, an adjacency constraint on the join's head buffers,
per-profile constant policies), so a single op-level walk that dispatches every
normalized op still waits on porting each emitter to a stage producer. The composer
is the pass those ports will use.

## 2026-09-10: LUT table index read directly; the residual is not affine (P6)

The non-power-of-two LUT blocker was previously inferred from output bytes. It is
now measured directly. `research/lut_index_probe/` emits each stem twice, once with
the real Sigmoid table and once with the table replaced by a **sign-aware
output-code ramp** - bank 0 written as `clip((neg_pivot-i)*128)` and bank 1 as
`clip((i-pos_pivot)*128)` in Q15 - with four `(neg_pivot, pos_pivot)` windows per
stem. The board's Q15 -> code conversion is
`clip(round(v*255/32768) - 128)` (multiplier 255, shift 15, zero point -128):
calibrated on the control stem, whose index `512 + 2q` is already verified, it
reproduces all four of its ramp windows exactly, so the ramp is a trustworthy index
readout. Intersecting the four windows per sample gives the hardware index exactly
for 504-573 of the 768 samples per stem.

Three results:

1. **The slope model holds.** A least-squares fit of the measured index against the
   input code lands within 0.15% of the emitted slopes (`64*g` on the positive half,
   `64*H*g` on the negative half), and the measured negative/positive ratio matches
   the predicted whole-bit `H = 2, 2, 2, 2, 4` within 0.5%.
2. **The residual is exactly one index step.** Every deviation from the emitted
   reference `512 + round(64*G*x)` is `-1`, `0` or `+1` (83% exact on the readable
   samples), which is what produces the 1-code output differences.
3. **No affine fixed-point rule reproduces it.** For each stem and half the
   analysis searches every integer `(A, B, J)`, `J <= 14`, with
   `floor((A*q+B)/2**J)` equal to the measured index on all readable samples. The
   control has many solutions (its rule is `2q`); every non-power-of-two stem has
   none. The hardware argument therefore comes from a quantized intermediate
   (an accumulator/declared-scale code stage) whose rounding this probe cannot
   separate from the measurement grid.

So the band stays rejected, but the retained evidence is now a direct index
measurement (`research/lut_index_probe/analysis.json`) rather than byte-level
inference; `tests/test_lut_index_probe.py` re-derives the measurement from the
retained board outputs and pins the emitter's rejection message.

## 2026-09-10: LUT negative-half gain is a whole-bit shift (P6)

The mixed-stem blocker was re-measured with a sigmoid-inverse readout over eight
stems (`research/lut_mixed_gain_probe/`). The hardware gain is
`H = 2**ceil(log2(BASE_WEIGHT_SCALE/weight_scale))`: a *whole-bit* shift, uniform
across output channels and independent of the bias and of the declared output
scale. Measured `H` matched the rule for every stem (0.5, 1, 2 and 4 where
predicted, including `0.02 -> 2` and `0.03 -> 2` where the ideal ratio is 1.5625
and 1.0417).

That explains why only power-of-two bands were ever exact. When the ratio is not a
power of two, compensating the gain still leaves **±1 on ~10% of values**
(321-394 of 3072 for diagonal 0.02/0.024/1/48/1/24, 372 for a mixed stem): the gain
is right, but the hardware's argument grid rounds differently from both the
rounded and the truncated code-shift models, so the accumulator-to-index rounding
step stays unmodelled. `compile_lut` now requires the ratio to be a power of two
and rejects the rest with that message; the accepted 12-model suite is unchanged
and byte-identical.

## 2026-09-10: join output consumed by a Conv tail (P1 emitter composition)

The diamond could fan out and fan in, but its join wrote the external output
directly. The new profile appends a `[Conv, Relu]* Conv` tail whose first layer
reads the join output as an internal named tensor, so the elementwise emitter and
the native Conv emitter are bound by name through the v5 table. Two failures were
found and fixed on the way:

1. **Overlapping programs.** The join task is 78+4 registers (656 bytes from
   `0xcc0`), and the first tail program was placed at `0xd40` inside it. The tail
   program overwrote the join fields (`0x5020` etc.), the driver rejected the
   submit with `-EINVAL`, and the board harness reported `model 0 failed: -22`.
   The tail now starts at `_align(0xcc0 + 82*8, 64) = 0xf80`.
2. **Wrong declared input zero point.** `0x1184` is the zero point the CNA border
   path injects. The heads inherit the native default (`-128`), which matches the
   stem grid, but the join grid is `0` for Add/Sub/Max, so a 3x3 tail layer padded
   with the wrong code and diverged on 44% of values. Each tail layer now writes
   `input_zero & 0xffff`.

Board evidence: `research/diamond_tail_suite/` passes **12 models, 192 inferences,
36,864 exact bytes** across all four joins and three tail shapes. The 12 existing
diamond models still compile byte-identically, and `tests/test_diamond_tail.py`
pins the container, lifetimes, the `0x1184` grid and the board-verified bytes.

## 2026-09-10: K5 ConvTranspose solved from the vendor-derived phase field (P3)

Following the re-derivation, direct depthwise K5 is now implemented and
board-verified.

1. **Emission.** The 25-tap table uses 32-byte lanes of
   `(value int8, -weight_zero_point int8)` with asymmetric per-channel weight
   quantization (`native_quantize(..., symmetric=False)`); the base depthwise
   compile gets a 3x3 placeholder kernel because it only supplies the verified
   stem/arena skeleton. The task reuses the vendor register set, including
   `0x1010 = 0x3ff` and the explicit output conversion.
2. **Phase field.** `0x1068 = ((k-1-pads_left)<<8) | (k-1-pads_top)`. Measured on
   the board for the pads=1 geometry: `0x101` -> 1303 mismatches, `0x202` -> 1773,
   `0x303` -> **0**. The formula also matches every vendor capture (K3 stride1/2
   pads1 `0x101`, K3 dilation2 effective-5 `0x202`, K5 stride2 pads2 `0x202`, and
   the unequal-stride capture's `0x201`).
3. **Evidence.** `research/transpose_k5_suite/` passes **9 models, 144 inferences,
   109,872 exact bytes** across stride 1/2, pads 0/1/2, output_padding, a
   rectangular per-axis geometry and C1/C4/C8. All 66 existing K1/K2/K3 and
   sparse-rewrite transpose models still compile byte-identically.

The stale pre-investigation probe (17 of 25 taps, reversed kernel, output
uncorrelated with its own reference) is retained under
`research/transpose_k5_suite/stale/`. Still open: a direct dilated
(effective-kernel-5) emission; dilation currently lowers through the verified
sparse-K3 rewrite.

## 2026-09-10: K5 ConvTranspose blocker re-derived from the vendor capture (P3)

The recorded blocker ("98 of 675 values off by one after an exact symmetric stem")
could not be reproduced, and the investigation found out why.

1. **The retained probe is stale.** `research/transpose_k5_suite/model000.bin` packs
   **17 of 25** weight taps (reversed kernel, symmetric zero points) while its
   `expected000.i8` was generated from a 25-tap reference. Re-running it on the
   board (16 clean inferences, `runs=16 rc=0`) gives an output that is
   **uncorrelated** with its own expected bytes (correlation 0.006) and with the
   float ONNX model (0.0017), so the historical count cannot come from these files.
2. **The register set itself was never the problem.** `research/analyze_k5_vendor_layout.py`
   extracts the vendor's 126-word K5 task from `capture_transpose_k5` and diffs it
   against the retained binary: every field matches except the three addresses
   (`0x1070`, `0x1110`, `0x4020`) and `0x1010`. The vendor sets the output
   conversion explicitly (`0x4080=0xffffffed`, `0x4084=0x7d36`, `0x4088=0x18`,
   i.e. zero point -19 and scale 0.00793062337, matching the capture ATTR line).
3. **The vendor weight layout is 25 taps x 32 bytes**, each tap 16 lanes of
   `(value int8, -weight_zero_point int8)`; the captured fixture uses per-lane zero
   points `(2, 1, -3)`, i.e. asymmetric per-channel weight quantization rather than
   the symmetric weights the retained experiment baked in.

Conclusion: K5 stays rejected, but with a concrete recipe - pack all 25 taps in the
vendor lane layout with per-channel weight zero points and reuse the vendor register
set. (That historical rejection is now superseded: see the entry above;
`tests/test_transpose_k5.py` covers the accepted profile.)

## 2026-09-10: LUT domain measured by sign; mixed-input stems blocked (P6)

The old blocker was "the identity/64 probe produced -123 where -98 was expected",
i.e. the table's argument mapping was unknown. Three probes settled most of it:

1. **Declared-scale probe** (`research/check_lut_domain.py`, models with the same
   table but declared stem scales 1/1024, 1/2048, 1/4096 and a zero point of 32):
   the *positive* half's argument scales with the declared output scale
   (0.5x, 1x, 2x) while the *negative* half does not, and the zero point does not
   move the table at all. The profile therefore keeps the declared scale fixed at
   1/2048.
2. **Ramp-table readout** (`research/build_lut_ramp_probe.py`) confirmed the table
   index advances one entry per 1/64 of the argument, and that the two banks hold
   indices 0..512 and 512..1024.
3. **Stem-gain probe** (`research/lut_domain_suite/`): with the table sampled as
   `fn((index-512)/64)` the *negative* half's argument is
   `x·(BASE_WEIGHT_SCALE/weight_scale)` — measured gain 2 for a 1/64 stem and 0.5
   for a 1/16 stem. Sampling the negative half with the inverse gain composes both
   halves back to `fn(x)`.

`compile_lut` now accepts a diagonal C3 1x1 stem with one scalar gain for every
channel (per-channel signs allowed), any bias, and a gain whose domain scale is a
power of two, with the analytic range inside the sign-split window. Board evidence:
**12 models, 192 inferences, 36,864 exact bytes** across Sigmoid and Tanh, all 256
codes and mixed inputs. `lut_reference` reproduces the bytes exactly, including the
1/64 index grid, and still reproduces the original identity/32 expected files.

A stem that *mixes* input channels is still rejected: its negative-half gain
follows the effective per-channel weight scale, so one table cannot be exact for
all channels. The retained failure is a stem `[[0.02,0.005,0],[0.005,0.02,0],
[0,0.005,0.02]]`, bias `[0.8,-0.8,0]`, whose channels even share one `(max-min)`:
the measured negative-half gain is 2.016 against 1.5625 predicted, and the board
output misses by 10 codes (`got -117 expected -107`). Artifacts in
`research/lut_domain_failed/`; removing the blocker needs a per-channel table
(channel-split tasks) or a documented per-channel gain formula.

## 2026-09-10: diamond fan-in DAG and the generic lifetime pass (P1)

The two-head profile only *fanned out*: one stem, two independent outputs. The
next P1 step was a consumer that reads two internal tensors at once, which needs a
real schedule and arena plan rather than hand-picked offsets.

* `src/open_rknpu/liveness.py` now does producer-before-consumer ordering (Kahn,
  stable to declared order), live intervals over task positions, and first-fit
  arena placement. A task that reads and writes the same buffer conflicts with
  itself, so in-place execution is never silently assumed, and external tensors
  are excluded from the reuse pool because the v5 loader rejects
  internal/external overlap.
* `compile_diamond` (in `src/open_rknpu/graph.py`) emits `stem -> {head_a,
  head_b} -> {Add,Mul,Sub,Max}`: the legacy stem program, two native16 head
  programs, and the verified native elementwise join block with `0x5018` =
  `head_a` and `0x5038` = `head_b` (plane stride `0x5040`). The arena comes from
  `liveness.plan`, which independently lands `stem`=8576, `head_a`=9600,
  `head_b`=10624, `output`=11648 for an 8x8 RGB model — the allocator's own
  answer, not a copied table.
* Head quantization: both heads are re-quantized to the shared symmetric grid the
  elementwise join requires (`scale = max(head natural scales)`, zero point 0);
  `Add/Sub/Max` then use `output_scale = 2*scale`, while `Mul` keeps both head
  scales in its own conversion.

Board evidence: `research/diamond_suite/` passed **12/12 models, 384/384 exact
inferences, 73,728 bytes** (`board_io . 12`) across the four joins and hidden
3/8/16 with 1x1/3x3 heads. `tests/test_diamond.py` pins the container, lifetimes,
offsets and the reference-versus-float agreement; `tests/test_liveness.py` proves
the same allocator reproduces the chain two-buffer ping-pong and refuses in-place
reuse. Still open in P1: a scheduler that assembles arbitrary emitter outputs by
tensor name (the diamond is matched as a whole pattern) and unequal runtime input
shapes.

## 2026-09-10: per-channel Mul quantization (depthwise route), OW_SRC refuted

`Mul(input, [C,1,1])` by an immutable per-channel constant was the last P4 gap.
Two routes existed; both were executed rather than assumed.

1. **New hardware mode (refuted).** Mesa names `0x4050` `BS_OW_CFG` (`OW_SRC@0`,
   `OD_BYPASS@1`) and `0x5020` `RDMA_BS_BASE_ADDR`, and the vendor path never sets
   `OW_SRC`. `research/mul_per_channel_ow_suite/model001.bin` is the unchanged
   verified constant-Mul binary with `0x4050=0x30000001`, `0x5020=0x1800` and a
   32-byte multiplier table: the first run returned `runs=0 rc=1` and dmesg showed
   `failed to wait job` / `job timeout` / `soft reset` / `job abort ret: -22`. The
   baseline `model000.bin` still passes 32/32, and `rkipc` (PID 283) stayed alive.
   The experiment is retained as the blocker for a per-channel EW output
   conversion; blind register sweeps are not repeated.
2. **Channel-split assembly (accepted, no task rewrites).** The verified 1x1
   depthwise profile already applies a *native per-output-channel weight scale*,
   so lowering the same product onto it per-channel-quantizes the constant: the
   operand bytes stay `[127,127,127]` instead of one shared `[1,23,127]` scale for
   `[.02,.35,1.9]`. Implemented as `--per-channel-mul`
   (`elementwise.compile_per_channel_constant_mul`), bias zero so the arithmetic is
   unchanged, and the output grid stays a single per-tensor scale.

`research/build_per_channel_mul_suite.py` generated 12 independent models (six
magnitude patterns jittered per model, six geometries), all 12 passed on the board
with 192 exact inferences (23,232 bytes); the worst shared-scale float error was
4.55 versus 1.54 per-channel, and per-channel won on every model by 2.2x-4.7x.
`tests/test_per_channel_mul.py` pins the profile tag, the per-channel weight scales,
the default-path regression and the rejected profiles.

## Latest status — 2026-09-09

The active work is the expanded Conv/Mul inventory, following the completed MNIST
baseline. [Current results and remaining tasks](../research/COVERAGE_EXPANSION_RESULTS.md)
record the newly verified native input, geometry, grouped/dilated rewrites,
depthwise, transposed Conv, Mul and activation profiles. Earlier limits and
failed hypotheses below are historical; all ten categories are not complete.

Latest continuation added native spatial tiling, batch1..16, outputC128, dilation
through17, direct grouped/depthwise rewrites, SAME padding, broader K2/K3
ConvTranspose, independent Mul output conversion, same-input/reversed Mul, and
static weight edge cases, and a public bounded Sigmoid/Tanh LUT path with long
command submission, plus bounded QLinearConv reconstruction/requantization.
The board ledger now contains 1,680 passing models and 28,250 inferences (10,206,227 exact output bytes); the
trained-model examples add two full 10,000-image evaluations (MNIST 98.67% and Fashion-MNIST 88.15% both-Conv NPU).
Current blockers are specific: general graph tensor/lifetime ABI, spatial Mul broadcast
without materialization (compact ERDMA variants hang the NPU), and the
generalization of LUT domains beyond the identity/32 probe. Dense native Conv input
channels C1..128 are solved (one CNA task for C65..128; see the 2026-09-11 entry). See the linked results file for exact evidence;
[completion-plan](plans/completion-plan.md) sequences them.


## 2026-09-10: task bindings are derived from every container and verified (P1)

The emitters declare each task's tensor read/write set by hand; `tests/test_container_bindings.py`
now *derives* the bindings from every container in the repository and checks them, so the
two views cannot silently diverge. The check walks all **2,412** compiled models in about
1.4 s and needs no board:

* v5 tensor tables: roles, contiguous external indices, sizes equal to
  `tensor_native_bytes`, offsets inside the arena, and no internal overlapping an
  external;
* v5 tasks: the address registers are known per task family (Conv `0x1070`/`0x4020`,
  elementwise `0x5018`/`0x5038`/`0x4020`, pool `0x701c`/`0x6070`, LUT setup none), so
  every read offset must be a declared external input or a tensor written by an
  **earlier** task (a topological producer/consumer proof), every write must target a
  declared internal or the output, and no task may write an external input;
* v3/v4 and ORNPUBIN containers: header, task table, constant descriptors, payload and
  arena bounds all agree.

The verifier immediately found a real defect: a pooled DAG whose **last join** wrote to
an internal slot that was never declared in the v5 tensor table (the tensor list only
included joins before the last one). The board had tolerated it because the memory was
still covered by other descriptors, but `ornpu_get_tensor` could not expose it. The
emitter now declares the last join whenever a pool follows, `pooled_dag_suite` was
regenerated and re-verified on the board (12 models, 384 inferences, 18,432 bytes), and
the whole corpus passes the derived-binding check.


## 2026-09-10: pooled multi-layer branches (P1/P5)

Every branch of a pooled join can now be a **Conv chain** rather than a single layer:
`open_rknpu.pooled_branches` accepts two or three branches of one to three Conv layers
(a depthwise layer inside a chain is expanded to an exact block-diagonal dense kernel),
each followed by a 2x2 stride-2 MaxPool or AveragePool, and one or two elementwise joins
fold the pooled 4x4/C3 grids. The join scale propagation runs over the branch finals
because pooling preserves the band, the pool tasks come from the shared
`open_rknpu.pooling.pool_registers` builder, and the profile keeps a fresh arena slot
per internal. It is dispatched *after* `pool_join`, so the earlier two-single-Conv
`pool_join_suite` containers are unchanged byte-for-byte (a first attempt routed them
into the new emitter and was caught by their byte-identity test).

Board result: **13 models, 416 inferences, 19,968 exact output bytes**
(`research/pooled_branches_suite/`) across two and three branches, one and two joins,
MaxPool and AveragePool, chains up to three layers, depthwise layers inside chains and
16-channel intermediates. At least 62% of every model's expected outputs are nonzero.


## 2026-09-10: terminal pooling on a join-DAG result (P1/P5)

The join DAG now accepts a final 2x2 stride-2 MaxPool or AveragePool, so a
`branches -> joins -> pool` graph compiles as one container: the joined 8x8/C3 result
is reduced to the 4x4/C3 output. The pool task is built by the shared
`open_rknpu.pooling.pool_registers` builder with the join's grid as its input, the
pooled grid keeps the join's band (the header keeps the join's output quantization and
only the geometry changes), and the profile keeps the fresh-slot placement rule.

Board result: **12 models, 384 inferences, 18,432 exact output bytes**
(`research/pooled_dag_suite/`) across MaxPool and AveragePool, three or four branches,
two or three joins, multi-layer branches and depthwise layers inside a chain. At least
62% of every model's expected outputs are nonzero. The earlier no-pool `join_dag_suite`
and `branch_join_suite` recompile byte-identically, so the added tail is opt-in.


## 2026-09-10: depthwise layers inside branch chains (P1)

A depthwise layer may now appear *inside* a multi-layer branch, which covers
depthwise-separable blocks (`Conv -> Depthwise -> Conv`) in a DAG. The dedicated
depthwise emitter models a branch that reads the stem directly, so a chained depthwise
layer cannot use it; instead the `(C,1,k,k)` kernel is rewritten to an equivalent
`(C,C,k,k)` block-diagonal dense kernel and fed to the per-layer dense path. The
rewrite is exact in the quantized domain: every added tap has weight zero, so it
contributes nothing whatever the input grid's zero point, and the existing dense
quantization handles the per-output-channel scales and biases. Single-layer depthwise
branches still use the dedicated emitter, so the earlier `join_dag_suite` containers
are byte-identical.

Board result: **12 models, 384 inferences, 73,728 exact output bytes**
(`research/depthwise_chain_suite/`) across one to three depthwise layers per model,
8- and 16-channel intermediates, four-branch expressions, depthwise kernels 1 and 3 and
a depthwise layer that is itself a branch. At least 78% of every model's expected
outputs are nonzero.


## 2026-09-10: packaging re-verified with the campaign's modules (P10)

The distribution was rebuilt after the campaign and re-checked end to end, because the
previous wheel predated the new emitters. `setuptools.build_meta` produced the wheel
and sdist without the `build` front end (which is not installed offline), and both now
contain all 37 compiler modules — `pool_join.py`, `depthwise_join.py`, `join_dag.py`,
`pooled_branches.py` and `compose.py` included — plus the runtime sources under
`share/open-rknpu/runtime/`.

In a fresh virtual environment the wheel installs with `--no-index --no-deps` and
`open-rknpu compile --sequence` reproduces byte-identical executables for four
profiles: the diamond fan-in (`diamond_suite`), the v5 two-head fan-out
(`two_head_suite`), the general join DAG (`join_dag_suite`) and the multi-layer branch
profile (`branch_join_suite`). `open-rknpu inspect` decodes a v5 container,
`normalize` rewrites a graph, and `--version` reports `0.1.0.dev0`. The sdist installs
and compiles byte-identically too, using `--no-build-isolation` because the declared
`setuptools>=77.0.3` build requirement cannot be downloaded offline here.

Publication remains a user decision; the technical distribution steps are complete.

**Rebuilt again (2026-09-10)** after the stage composer was added: the wheel and sdist
now carry `compose.py`, and compiling `pool_join_suite/model003` from the extracted
wheel is byte-identical to the committed container.


## 2026-09-10: the ledger is now machine-verified (P0)

The board ledger was audited row by row against the artifacts, and the audit found one
stale row plus a methodology gap:

* Board evidence for a suite is sometimes split across several
  `board_results_<start>.json` files, because a runner streams models and stops at the
  first failure (debug sessions therefore leave a partial file per attempt). The
  ledger counts the *whole* suite, so the check now takes the **union of per-model
  passing entries** across every evidence file. With that method 97 of 111 rows
  reproduce their model, inference and exact byte counts, and all 112 ledger links
  resolve.
* `native_large_suite` had grown to 44 models (44/704/800,800 verified) while the
  ledger still recorded the earlier 42/672/738,960. The row is corrected; the grand
  totals move to **1,594 models, 26,302 inferences, 9,912,619 exact bytes**.
* `native_spatial_boundary_suite` looked wrong under the old method (its complete run
  covers 16 models) but every one of its 28 models passes in some recorded run, so
  the union method vindicates the row rather than the suite.

`tests/test_ledger.py` now enforces: rows sum to the totals, every link resolves,
board evidence reproduces every row that has it (>=97 rows), manifests reproduce
their rows where the schema records cases and output bytes (>=11 rows), and the
eleven campaign suites are pinned by name and total.


## 2026-09-10: multi-layer branches and the arena-reuse hazard (P1)

The join-expression DAG now accepts **branch chains**: a branch may be one to three
dense Conv layers off the stem, so residual-style blocks compile without collapsing
to single-Conv heads. Two things had to generalize: the native Conv fields and the
weight/bias packing are emitted for each layer's own output-channel count (an
intermediate may be C1..16; only a branch final must be C3 for a join to consume it),
and each layer's input band is the previous layer's band, so `0x1184` and the weight
quantization follow the chain while intermediates keep their natural quantization.

Two bugs surfaced on the way, both fixed and covered by tests:

* `_native_fields`/`_pack_native_head` are three-output helpers, so an 8-channel
  intermediate emitted three-channel registers and the next layer read garbage
  (`got 2 expected -2` on the board). Generalizing the field set and the 32-byte bias
  groups fixed it; nine of twelve models then passed.
* The remaining failure (model 009) was **arena reuse**: a slot written by two
  different tasks of a mixed dense/depthwise family returned stale data on the board
  (`got 6 expected 12`, broadly wrong across the grid). Placing every internal in a
  fresh slot fixed all twelve. This is the second independent sighting of the same
  hazard, after the runtime-scale profile, so the join DAG now always uses fresh
  slots and the round-31 `join_dag_suite` containers were regenerated and re-verified
  under that rule. Liveness still computes the per-tensor intervals; only the
  placement is conservative.

Board result: **12 models, 384 inferences, 73,728 exact output bytes**
(`research/branch_join_suite/`) across two- and three-layer branches, 8- and
16-channel intermediates, depthwise branches, four-branch expressions and joins that
read an intermediate layer.


## 2026-09-10: general join expression over reused internal grids (P1)

The fan-out family is now a general expression DAG rather than a left fold:
`open_rknpu.join_dag` accepts two or three joins whose operands are *any* previously
produced tensors, so a head or an earlier join result can feed several consumers.
The left-fold chain keeps its own emitter (the chain predicate was tightened to the
left-fold shape so non-chain expressions fall through), and the new profile is
dispatched only for expressions the chain declines.

Scale propagation is the heart of it: one INT8 scale per tensor, assigned in
topological node order. A Mul join folds two free operand scales and commits both;
Add/Sub/Max require a shared band, re-quantizing an uncommitted head operand onto the
other operand's scale, and reject a tensor that a later join wants on a different
band. `open_rknpu.liveness` is what makes the reuse safe to *schedule*: a reused grid
stays live across all of its consumers, which the host test asserts by checking that
`h0`'s interval still extends to the second join.

Board result: **12 models, 384 inferences, 73,728 exact output bytes**
(`research/join_dag_suite/`) across two- and three-join expressions, dense/depthwise
head mixes and kernel sizes 1/3. Together with the chain, residual and scale tails,
the P1 graph form now covers trees, chains, general join expressions and runtime
inputs at both elementwise roles.


## 2026-09-10: runtime residual feature map on a fan-out result (P1)

The runtime tail of the join chain now accepts `Add/Sub/Max(result, residual)` where
the second operand is a declared `[1,3,8,8]` input, so an externally supplied
feature map can be combined with a computed DAG at run time. The residual is
interpreted as zero-centered INT8 on the join result's scale (the caller passes
`value + 128`, matching the loader's native16 packing), so the verified join task
applies unchanged: `out = rint((a + b)/2)` on output scale `2*s`, with the external
grid as the ERDMA secondary at the per-pixel plane stride. The fresh-slot rule found
for the per-channel scale also applies here (a join primary reading a slot that an
earlier Conv task wrote returned stale data on the board), and it is asserted by
`tests/test_join_residual.py`.

Board result: **12 models, 384 inferences, 73,728 exact output bytes**
(`research/join_residual_suite/`) across Add/Sub/Max tails, 3..5 dense/depthwise
heads, kernel sizes 1/3/5, asymmetric depthwise weights and a no-Relu stem. The
runtime tail now covers both elementwise roles an external tensor can play in a DAG:
a per-channel gain and a full spatial operand.


## 2026-09-10: runtime per-channel scale inside a fan-out (P1/P4)

The join chain now accepts a final `Mul(result, scale)` whose second operand is a
declared `[1,3,1,1]` graph input, so a runtime per-channel gain can be applied to a
folded DAG result. The scale task reuses the verified per-channel elementwise
program (primary `0x5018` = the join grid, secondary `0x5038` = the scale row with
`0x5034 = 4` and `0x5040 = 16`); no conversion task is needed because the join grid
is already native16. Board result: **12 models, 384 inferences, 73,728 exact output
bytes** (`research/join_scale_suite/`).

Two hardware findings surfaced and are retained:

* **A scaled primary needs a fresh slot.** The first container let the last join
  reuse a buffer that an earlier Conv head had written (legal for the diamond, which
  reuses the stem slot). Every run then returned zeros: `tests/board_dump.c` (new
  debug harness) showed an all-zero output for every code pattern, while the same
  model compiled without the scale tail matched the board exactly. Placing each
  internal of this profile in its own slot fixed it, so `compile_join_chain` now
  places internals sequentially whenever a runtime scale follows. Registers were
  compared one by one between the passing and failing containers; only the primary's
  offset and the constant addresses differed.
* **`rint(a*b/128)` is not the hardware model for this composition.** The folded
  scale product is not exactly `1/128` in float32, so hardware requantization differs
  from `rint` by one on ~1% of boundary values (71 of 6144 bytes in the probe, all
  +-1). The suite reference therefore uses `elementwise.mul_requant_reference` with
  the container's own multiplier and shift. The standalone runtime-scale suite is
  unaffected because its single branch scale yields an exact factor.

The full host suite passes (180 tests) and the repository-wide recompile sweep is
unchanged (187 of 422 models byte-identical with default arguments, 15 suites
requiring their build flags).


## 2026-09-10: depthwise K3-dilation2 ConvTranspose as a direct sparse K5 (P3)

The last item under P3's "general dilation" heading is closed for the verified
family. A depthwise `ConvTranspose` with `kernel_shape=[3,3]` and `dilations=[2,2]`
spans five taps per axis, so it is exactly a K5 kernel with zeros at the odd
positions. `open_rknpu.transposed` now expands `(C,1,3,3)` to `(C,1,5,5)` with the
dilated taps in place, lowers `kernel_shape` to `[5,5]` and `dilations` to
`[1,1]`, and emits the already-verified depthwise K5 task (asymmetric per-channel
weight zero points, vendor phase field `0x1068 = ((k-1-pads)<<8)|(k-1-pads)`). The
output geometry is unchanged because `(3-1)*2+1 = 5`.

Two reference details mattered and were fixed by mirroring the verified K5 suite:
the K5 tap table is asymmetric, so the integer reference must center the weights by
the per-channel weight zero points and fold the stem zero point into the bias base,
and its final rounding differs from the symmetric K2/K3 path. Reproducing the K2
path's reference initially produced a board mismatch (`got 17 expected 8`).

Board result: **8 models, 128 inferences, 82,432 exact output bytes**
(`research/transpose_k5_dilation_suite/`) across stride 1/2, pads 0/1/2,
output_padding, a rectangular per-axis stride, C1/C3/C4 and SAME_UPPER. A host test
additionally evaluates the dilated K3 and the zero-filled K5 scatters in NumPy and
asserts they agree exactly. Dense K3-dilation2 remains rejected with a specific
message (it needs a dense K5 ConvTranspose field set), and per-axis dilation stays
rejected because the K5 tap table is square.


## 2026-09-10: mixed dense/depthwise heads in one fan-out (P1)

`compile_join_chain` now generalizes the head slot: any of the 3..8 heads may be a
group-3 depthwise Conv instead of a dense Conv. Dense heads keep the shared native
field builder; a depthwise head compiles the verified standalone depthwise profile
and relocates its four address registers and `0x1184` into the shared container,
copying its weight/bias blocks verbatim (including the asymmetric per-channel weight
zero points). This is the plan's "dispatch each op to its existing emitter and bind
producer/consumer edges by name" step in miniature: two emitter families now compose
inside one fan-out, with per-head quantization metadata tracked per family and the
task order still coming from `open_rknpu.liveness`.

The generalization was checked against the verified path first: the all-dense
`join_chain_suite` recompiles byte-identically (12/12), and the full host suite
passes. Depthwise heads require a three-channel stem because the depthwise task's
group is its channel count, so the mixed models use hidden 3 while an all-dense
control keeps hidden 8.

Board result: **12 models, 384 inferences, 73,728 exact output bytes**
(`research/mixed_head_suite/`) across 3..5 heads, dense/depthwise mixes, all four
join kinds, kernels 1/3/5, a Conv tail, a no-Relu stem and asymmetric depthwise
weights. At least 72% of every model's expected outputs are nonzero and the host
test asserts it.


## 2026-09-10: pooled branches into the join (P1/P5)

The last explicit half of the plan's "multi-task pool/depthwise branches into
Mul" blocker is closed: two Conv branches from one shared stem each end in a **2x2
stride-2 MaxPool or AveragePool**, and one elementwise join folds the two 4x4/C3
pooled grids. The 37-word pool task is built by `open_rknpu.pooling.pool_registers`
- the same builder the sequence lowering now uses, extracted so the pool program has
a single definition - with only the input/output addresses and the geometry fields
substituted. The dense branch Convs are emitted with the shared native field
builder and declare the stem zero point in `0x1184`.

Pooling preserves the grid scale, so the join arithmetic is unchanged from the
other diamond variants: Mul folds two free branch scales, Add/Sub/Max re-quantize
both branches onto one shared scale. The join task runs at 4x4 (256-byte plane
stride) and `open_rknpu.liveness` orders the six tasks and places all five internal
tensors.

Board result: **12 models, 384 inferences, 18,432 exact output bytes**
(`research/pool_join_suite/`) across MaxPool and AveragePool, all four joins, 1x1/3x3
heads, hidden 3/8/16 and a stem without Relu. At least 96% of every model's expected
outputs are nonzero and the host test asserts it. Both halves of the blocker are now
retired; the remaining P1 work is a general emitter-output scheduler and
auto-derived task bindings.


## 2026-09-10: dense and depthwise branches from one stem (P1/P4)

The dense diamond's two branches are now heterogeneous: a shared 1x1 Conv stem
feeds a dense 1x1/3x3 Conv and a **group-3 depthwise 1x1/3x3/5x5 Conv**, and one
elementwise join folds the two 8x8/C3 grids. Rather than re-deriving the depthwise
task, `open_rknpu.depthwise_join` compiles the standalone depthwise profile with
`compile_depthwise`, then relocates its four address registers (`0x1070` input,
`0x4020` output, `0x1110` weights, `0x5020` bias) and its input zero point
`0x1184` into this container's arena, copying the weight and bias blocks verbatim.
Keeping the verified program intact is what carries the asymmetric per-channel
weight zero points into the composed graph without new field work.

Quantization follows the join chain: Mul folds two free operand scales
(`out = rint(a*b/128)`), Add/Sub/Max re-quantize both branches onto one shared
scale. The stem width is bound to three channels so both branches stay C3, and a
no-Relu stem exercises the `0x1184` declaration again.

Board result: **12 models, 384 inferences, 73,728 exact output bytes**
(`research/depthwise_join_suite/`) across all four joins, depthwise kernels 1/3/5,
dense kernels 1/3, a stem without Relu and two asymmetric-weight models. At least
92% of every model's expected outputs are nonzero and the host test asserts it.
This retires the plan's "multi-task pool/depthwise branches into Mul" blocker for
the depthwise half; a pooling branch still carries absolute offsets in its program
and needs the same relocation treatment.


## 2026-09-10: join-chain fan-out (variable consumers, mixed joins)

The diamond's fixed two heads are now a variable fan-out: one shared 1x1 Conv
stem feeds **three to five dense heads**, and `n-1` independently generated
elementwise joins fold the head grids left to right with mixed
`Add`/`Sub`/`Max`/`Mul` kinds before an optional `[Conv, Relu]* Conv` tail.
`open_rknpu.graph.parse_join_chain` recognises the shape from the node graph (both
topological orders) and the scheduler dispatches it before the terminal
elementwise profiles, which otherwise hijack a graph ending in `Mul, Add`.

Two arithmetic facts drive the head quantization. A Mul join folds two free
operand scales into its own conversion (`out = rint(a*b/128)`, output scale
`128·sa·sb`); Add/Sub/Max require both operands on one shared scale
(`out = rint((a+b)/2)`), so the next head is re-quantized onto the running
result. Each head task also declares the stem grid zero point in `0x1184`; a stem
without Relu has a zero point other than -128, and the first board run failed
model 10 (`got -1 expected 0`) until that register was set - the same
CNA-border-injection bug previously fixed for the diamond tail. The equivalent
declaration in the N-layer native chain was tried and reverted: it changes bytes
without changing any board result, so the verified chain containers were left
untouched.

Board result: **12 models, 384 inferences, 73,728 exact output bytes**
(`research/join_chain_suite/`). The suite deliberately uses small positive weights
so the head grids use the int8 range; at least 95% of every model's expected
outputs are nonzero and the host test asserts it, so the comparison cannot pass
vacuously. Liveness reuses the dead stem buffer for the first join.


## 2026-09-10: runtime per-channel scale Mul (unequal external input shapes)

`Mul(image[1,3,H,W], scale[1,3,1,1])` now compiles, where `scale` is a *named
external input* of an unequal shape rather than a v4 constant. Two independently
generated tasks: a 29/768 Conv task that copies the image into the elementwise
operand arena at `0x1000` (stride 16), then the verified 78/1106 elementwise
per-channel task reading the scale at `0x3000` with `RDMA_ERDMA_CFG=4`. Payload
4096 bytes, arena 20480, three tensors (packed image, native16 scale, internal
converted grid).

Board result: **16 models, 256 inferences, 32,832 exact output bytes**, covering
four geometries (8x8, 5x5, 6x7, 5x8) and four operand patterns
(`[127,127,127]`, `[64,0,-64]`, `[-128,127,32]`, `[1,2,3]`). The v5 loader's named-tensor rules were already
sufficient: the semantic program and the C loader agree byte-for-byte.

The scheduler dispatch change is deliberately narrow. A `Mul` whose second
operand has shape `(1,1,1,C)` and traces to an external input takes the new
runtime-scale path; a `Mul` tracing to an initializer still takes the verified
constant path, and an initializer that is *also* declared an input is rejected by
both the dispatcher and the emitter (regression test `test_runtime_scale.py`).
Recompiling the `mul_sources_suite` and standalone Mul corpora is byte-identical,
so no verified container changed.

This retires the plan's "unequal logical inputs" and "complementary singleton
axes with two runtime operands" blockers. Still open: general graph
tensor/lifetime ABI (bounded scalar task bindings, not graph-wide elaboration).


## 2026-09-09: depthwise C5–C16 packing resolved

The C5 vendor capture matched our configuration except addresses/quantization.
Bias/scale groups are packed as four INT32 biases plus four UINT16 scales:
24 bytes per four channels, not the 32-byte dense-Conv layout. Correcting this
independently generated packing passed every channel count 5..16: 12 graphs,
192 board inferences, 129,024 exact bytes. Public compiler binaries match all
12 tested programs; the host suite has since grown to 873 tests, all passing.

New bounded public profile: external RGB 8x8 -> 1x1 Conv stem with C5..16 outputs
-> depthwise 3x3, pad1, stride1, multiplier1. Other kernels/strides/stem kernels
at these channel counts remain rejected or unverified. This resolves the earlier
C5 hypothesis failure and the 6x6 spatial-layout mismatch; it does not resolve all
depthwise modes.

Evidence: research/depthwise_c5_suite through depthwise_c16_suite, each containing
board_results_0.json; research/capture_depthwise_c5 and depthwise_c5_capture.log.
Generator: research/probe_depthwise_channels.py CHANNELS; board runner:
research/run_profile_suite.py depthwise_cCHANNELS_suite (open Python/PYTHONPATH=src).
The generator uses independently compiled stem weights and commands, not captures.

## 2026-09-08: first pass through topics 1–7

New verified additions: depthwise k1/k5 at C3 and k3 at C4; constant grouped/dilated
Conv lowering; scalar/channel Mul folding; spatial-only terminal Reshape.
224 new board inferences / 44,032 exact bytes; 45 host tests pass.
Depthwise 6x6/C5, independent transposed Conv and LeakyReLU hypotheses failed;
LUT setup exceeds public task limits and needs submission support. General modes
remain pending. Full evidence, limitations and next work: research/MODE_EXPANSION_STATUS.md.

## 2026-09-08: depthwise stride2 public profile

12 independently generated graphs / 192 board runs / 9,216 exact bytes pass.
Public compiler binaries match those board-tested hypotheses exactly. Conv[/Relu]
stem -> depthwise3x3, group3/pad1/stride2, 8x8/C3 -> 4x4/C3. 41 host tests pass.
See research/depthwise_stride2_suite/README.md. Next: depthwise shape/channel
expansion, followed by the remaining modes tracked in docs/plans/primitive-roadmap.md.

## 2026-09-08: stride-2 dense Conv initial public profile

8x8/C3 -> 4x4/C3, kernels 1/3/5: 96 independent board runs, 4,608 exact bytes.
Public compile --sequence produces byte-identical tested programs. 40 host tests
pass. Initial timeout resolved by setting output width/pixel count at 0x1028/102c,
identified by comparing stride2 oracle captures. See research/stride2_suite/README.md.
Next: depthwise stride2. Full queue: docs/plans/primitive-roadmap.md.

## 2026-09-08: Sub and Max; ordered remaining-mode roadmap

Independent public Sub and Max each pass 12 graphs, 384 board runs and 73,728
exact bytes. Sub uses operand scale 0xc000 at 0x4078; Max uses ALU selector 0.
Both retain shared branch scale s, output 2*s and zero point 0, fixed 8x8/C3.
39 host tests pass. See research/sub_suite/README.md and research/max_suite/README.md.
User authorized working through all missing Mul/Conv modes in order. The durable
checklist is docs/plans/primitive-roadmap.md; stride-2 Conv is now under investigation.

## 2026-09-08: independent Mul milestone

The public `compile --sequence` path now supports two 1x1 Conv branches feeding
Mul at fixed `[1,3,8,8]`, without broadcasting. All three tasks execute on RV1103.
Both branches use symmetric INT8 activations with shared scale s; output scale
is 128*s*s and zero point 0. Integer arithmetic is nearest-even(A*B/128), clipped
to INT8. No calibration or float-model accuracy claim is made.

12 independently compiled models passed 384 board inferences and 73,728 exact
output bytes. All 37 host tests pass. Mesa/TRM EW_OP_TYPE and OD_BYPASS names guided
the profile; complete command behavior was checked on RV1103, without claiming
that each changed field has been isolated. See research/mul_suite/README.md.
Next primitive: Sub.

## 2026-09-08: Mesa/TRM-guided Add field validation

Inspected Mesa Rocket registers.xml/rkt_regcmd.c and RK3588 TRM Part1 chapter36
architecture and relevant registers. Verified on RV1103: EW_ALU_ALGO bits19:16
(Add=2, Max=0), output rounding bit30 (0 ties-even, 1 ties-away-zero in this Add
path). Two independent field probes: 64 cases, 12,288 exact bytes. An initial
signed half-up hypothesis was rejected; the corrected signed rounding hypothesis
passed. Pad layout and reserved-bit differences prevent wholesale RK3588 reuse.
Sources/hashes, caveats and evidence: research/hardware_refs/README.md.

---

## 2026-09-08: independent Add compiler/runtime milestone

Added src/open_rknpu/elementwise.py, public compile --sequence route for two
1x1 Conv branches -> Add at 8x8/C3. Independent weights/program generation;
no captured/vendor compilation inputs. Equal branch scale, zero point 0;
output scale twice branch scale. Add rounds halves to even (unlike Conv final
rounding). New exact descriptor 78 words/enable24/mask768 is accepted by Python
and C. 12 models, 384 inferences, 73,728 exact bytes pass; 35 host tests pass.
Rebuilt board runtime also passes all 192 depthwise cases. Camera remains up.
Details and limitations: research/add_suite/README.md. Next: Mul.

---

## 2026-09-08: independent depthwise compiler milestone

Implemented src/open_rknpu/depthwise.py through public compile --sequence.
Fixed 8x8/C3, 3x3 pad1 stride1 DW after Conv1x1/3x3 with optional stem ReLU.
No captured/vendor inputs used by the emitter. Symmetric per-channel DW weights.
12 fresh models, 192 runs, 36,864 exact bytes through existing C runtime; camera
running. All 32 host tests pass. Evidence: research/depthwise_suite/README.md.
Wider depthwise shapes/channels and affine weight pair encoding remain unverified.
Next: independently generated elementwise Add/Mul/Sub/Max path.

---

## 2026-09-08: primitive hardware-path survey

28 small graphs built with vendor compiler 2.3.0. 26 graphs pass libc/direct-ioctl
replay: 104 inferences, 135,552 exact bytes. Div is CPU fallback (replay misses it);
CPU Min is rejected by board runtime. Activations use fused register fields or a
large likely LUT setup; MatMul lowers to Conv, Resize to ConvTranspose, 8x8 global
average pooling to two Conv stages. Softmax expands to 76 tasks with FP16 conversion.
Full evidence, constraints and reproduction: research/primitive_survey/README.md.
This is captured-code replay, not independent generation of the new primitives.
Public compiler coverage unchanged. Camera stayed running; 28 host tests pass.

---

## 2026-09-08: rough real-digit sanity check, no tuning

100 fixed held-out MNIST test images: float 100/100, float with quantized inputs
100/100, board CPU-Conv2 hybrid 100/100, board both-Conv NPU 13/100. Native variant
predicts 5 for every image. Independent integer reference + ONNX suffix matches
board logits, reproducing the collapse. Keep hybrid baseline; native Conv2 remains
an offload demonstration until quantization is improved. No calibration/tuning done.
Update 2026-09-10: calibrating the Conv2 output range took the both-Conv NPU
variant to 100/100 on the same images at ~2.45 ms/image; see examples/mnist/README.md.
Evidence and reproducibility: examples/mnist/sanity-results/report.json and README.md.
Camera stayed running. Sample is small; not a full-dataset accuracy claim.

---

## 2026-09-08: trained Conv2 successfully offloaded

Opt-in examples/mnist/build.py --native emits the third serial NPU task for the
trained 14x14 8->16 5x5 Conv. All 25,088 output bytes exact across eight cases;
final CPU suffix logits within 0.000031 of ONNX reference. Five interleaved batches
show 20.589 ms baseline vs 4.490 ms native (4.59x), with considerable timing noise.
Fixture still predicts 3, but uncalibrated int8 Conv2 increases float-model maximum
logit error from 1.925 to 17.806. Retain baseline default; calibration/held-out
accuracy is the next prerequisite. Full details: examples/mnist/README.md.
Native arena 40 KiB, task storage 4 KiB, peak process RSS 856 KiB. Camera stays up.
Initial timeout fixed by resetting inherited 28x28 output/bias dimensions to 14x14.

---

## 2026-09-08: complete hybrid MNIST milestone

The current host is x86_64; historical aarch64/tooling notes below describe an
older environment. The current roadmap is docs/plans/project-goals.md.

Complete pretrained MNIST now runs on RV1103: first Conv/Relu/MaxPool on NPU,
remaining Conv/Relu/MaxPool/Reshape/MatMul/Add in a small generated-weight C
example. No vendor compiler or runtime is used in this build/run pipeline.
See examples/mnist/README.md for reproduction, placement and limitations.
Eight cases: 12,544 exact NPU bytes, 80 logits within 0.000092 of independent
ONNX suffix evaluation. Fixture predicts 3, matching float model; no dataset
accuracy claim. Mean 23.621 ms (10.793 NPU + 12.828 CPU), max RSS 856 KiB plus
kernel/DMA allocation caveats. Camera PID 283 remained running. All 28 existing
host tests pass. Binary ELF dependency is libc.so.0 only.

ADB used: /home/dnhkng/Unity/Hub/Editor/2022.3.14f1/Editor/Data/PlaybackEngines/AndroidPlayer/SDK/platform-tools/adb
Board artifacts: /userdata/open-npu-research/mnist-hybrid/
Host artifacts and reports: examples/mnist/build/

---

# RV1103 NPU / RKNN `regcmd` Investigation — Summary

> **Historical:** this summary and sections 1–7 record the original aarch64-host
> investigation. Its "unresolved input path" and "unread command" framing is
> superseded by the independently generated compiler/runtime above; aarch64-specific
> tooling notes apply only to that older environment.

> **Independent pooling milestone:** Conv[/Relu]→MaxPool/AveragePool, 2x2
> stride 2, now passes 272 inferences and 13,056 exact pooled bytes through
> independently generated raw commands. Average pooling rounds the signed
> four-value mean to even. Public pooling integration now passes another
> 256 inferences/12,288 bytes, using profiles 3/4 and explicit output dimensions.
> An 8x8 average-pool oracle lowers to convolution tasks and needs further work.

> **Independent two-layer milestone:** Four new ONNX Conv-Relu-Conv graphs now
> run through independently emitted commands and the libc-only two-task harness:
> 272 inferences, 52,224 exact integer bytes. No RKNN compilation or runtime is
> used for those graphs. Hidden channels 3/4/8/16, spatial shape 8x8, kernels 1x1.
> See [multi-layer evidence](../research/README.md#independently-generated-two-layer-milestone).
> Public CLI/container/runtime integration now passes another 448 inferences
> (86,016 bytes) across all hidden counts 3–16. Profile 2 builds both tasks in
> the public runtime. Useful trained-model accuracy and calibration remain
> unfinished.

> **Current checkpoint, 2026-09-08:** An independent ONNX compiler and libc-only
> C runtime now run dense signed 1x1/3x3 Conv, bias, and fused Relu. The public
> API passed 64 models, 1,024 inferences, and 129,792 integer output bytes exactly
> on RV1103. Wider 1x1 now supports 2–16 output channels; another 30 models passed
> 480 public-API inferences and 178,112 bytes exactly. Inputs remain three
> channels and H/W 5..8; 3x3 still has three output channels. Useful
> multi-layer models and a distinct-RV1106-SoC validation are not complete. See the
> [current evidence and quantization rules](../research/README.md#dense-convolution-quantization-and-public-runtime-checkpoint)
> and [usage](README.md). The sections below preserve earlier investigation
> history; their unresolved-input and unread-command claims are superseded.

> **2026-09-08 continuation:** The unresolved input path and command capture have
> now been demonstrated for a new deterministic 8x8, three-channel 1x1 convolution.
> A C oracle passes 768/768 integer outputs using UINT8 NHWC input with row stride
> 16 and native NC1HWC2 output. An open libc-only ioctl harness reproduces those
> outputs from captured commands, including with a relocated DMA base. This is
> followed by an independent narrow ONNX emitter: a held-out 6x5 convolution
> with a new channel permutation passes 360/360 integer outputs, with no RKNN
> compilation or runtime for that graph. General operators remain unimplemented.
> See [research evidence and next
> steps](../research/README.md). Historical claims below that constant output proves
> a universal NHWC output layout, or that register buffers require kernel probes,
> are superseded by these tests. The current host is x86-64.

**Board:** Luckfox Pico (Rockchip RV1103, single-core "mini" NPU, armv7 uClibc, 33MB RAM)
**Goal:** Get the NPU running real inference via our own code (not the vendor's `rkipc` camera app), and understand the low-level `regcmd` mechanism well enough to eventually build an open alternative to Rockchip's closed compiler/runtime.

---

## TL;DR

- **Achieved:** our own Python/ctypes code drives the physical NPU end-to-end (`rknn_init` → `rknn_create_mem` → `rknn_set_io_mem` → `rknn_run` → readback), using nothing but Rockchip's redistributable `librknnmrt.so` — no `rkipc`, no vendor demo app.
- **Verified correct:** output-tensor unpacking, via a zero-input diagnostic with an unambiguous structural signature.
- **Not yet solved:** input-tensor byte-exact packing. Every layout hypothesis tried failed a clean impulse-response test.
- **Key structural discovery:** the NC1HWC2 "native" buffer size is **not computed by the on-device runtime**. It's decided by the PC-side compiler (`rknn-toolkit2` / `librknnc.so`) at compile time and simply serialized into the `.rknn` file; the runtime reads it back verbatim. This reframes the open question from "what does the runtime do" to "what does the compiler decide."
- **Real regcmd bytes remain unread.** The kernel-level `task_obj_addr`/`regcfg_obj_addr` fields are opaque driver-internal DMA handles, not plain userspace pointers — GDB can't dereference them even from within the owning process. Reading them needs kernel-side instrumentation (kprobe) or intercepting the runtime's internal buffer *before* it's handed to `ioctl()` (e.g. via Frida/Ghidra on `librknnmrt.so`, not via GDB after the fact).

---

## 1. Confirmed working: our own NPU stack

### API surface used (from the real, public `rknn_api.h`)
```c
int rknn_init(rknn_context*, void* model, uint32_t size, uint32_t flag, rknn_init_extend*);
int rknn_query(rknn_context, rknn_query_cmd, void* info, uint32_t size);
rknn_tensor_mem* rknn_create_mem(rknn_context, uint32_t size);
int rknn_set_io_mem(rknn_context, rknn_tensor_mem*, rknn_tensor_attr*);
int rknn_run(rknn_context, rknn_run_extend*);
int rknn_mem_sync(rknn_context, rknn_tensor_mem*, rknn_mem_sync_mode);
```
Full header source: `rknpu2/runtime/Linux/librknn_api/include/rknn_api.h` in `airockchip/rknn-toolkit2` on GitHub.

### Two real bugs found and fixed (via evidence, not guessing)

1. **Runtime/model version mismatch.** Board shipped with `librknnmrt version: 1.4.1b9` (Oct 2022), but the only natively-installable PC toolkit (aarch64 wheels only exist from v2.3.0+) produces models tagged `2.3.2` (Apr 2025) — the old runtime refused them (`E RKNN: failed to decode config data!`). **Fix:** fetched the matching **2.3.2** `librknnmrt.so` from `rknpu2/runtime/Linux/librknn_api/armhf-uclibc/librknnmrt.so` in the same GitHub repo and used it directly (via `CDLL()`), completely bypassing the system-installed runtime — no board modification needed.
2. **"mini runtime" requires the zero-copy API.** The convenience path (`rknn_inputs_set`/`rknn_outputs_get`) fails with `context config invalid!` — confirmed via `strace` that **zero ioctl calls occur** for this failure, meaning it's a pure userspace check, not a kernel/driver rejection. **Fix:** switched to the explicit zero-copy path (`rknn_create_mem` + `rknn_set_io_mem`), which works.

### Native NC1HWC2 buffer sizing
The simple (`RKNN_QUERY_OUTPUT_ATTR`/`INPUT_ATTR`) query returns *logical* size. The "native" query (`RKNN_QUERY_NATIVE_NC1HWC2_INPUT_ATTR` = cmd 8) is **unreliable in this mini runtime** — it returns plain 4D dims when the true native shape is 5D. Ground truth for native shape came from the PC-side compiler's own verbose build log (`RKNNModelRegCmdbuildPass` output), e.g.:
```
Conv input INT8 NC1HWC2 (1,3,8,8) → native (1,2,8,8,3)   [384 bytes]
Conv input INT8 NC1HWC2 (1,4,8,8) → native (1,2,8,8,4)   [512 bytes]
```

---

## 2. Output correctness — proven, via a clean diagnostic

**Method:** feed an all-zero input. Since `0.0` quantizes to a uniform `zp` byte everywhere, the *correct* output must be exactly constant per channel (pure bias term) at every interior spatial position, with variation only at the border (where the conv's own zero-padding differs from the interior).

**Result:** exactly this pattern was observed — interior positions bit-identical, border positions differing only in the expected way. This is unambiguous: **output unpacking (plain NHWC, `C2` contiguous, no channel-splitting despite the "NC1HWC2" label) is bit-exact correct.**

A red herring along the way: an earlier attempt assumed the *primary* native output tensor `(1,1,8,8,16)` (1024 bytes) was what to read — wrong. The runtime only ever writes 256 bytes (the size of the *secondary*, already-flattened NHWC tensor); the rest of a larger buffer is left as literal zero bytes (uninitialized), not real data.

---

## 3. Input correctness — still open

**What's ruled out (all tested via real hardware, not guessed):**
- Two different channel `(C1, C2)` grouping conventions (`c1 = c // C2` vs `c1 = c % C1`) — both gave **identical** results for the 3-channel case, and neither matches a correct impulse response.
- Width-stride padding hypothesis (`w_stride=16` from `rknn_query` taken literally as real per-row padding).
- Missing `rknn_mem_sync(TO_DEVICE)` cache flush — added it, zero change in output (byte-identical), ruling out cache coherency as the cause.
- Buffer **size** being wrong — ruled out analytically (see §4): for our specific 8×8 test tensors, all plausible tile-alignment formulas collapse to the same simple product, so size was never the issue.

**Decisive negative result — impulse-response test:** wrote a single non-zero value at one (channel, row, col) coordinate, rest zero. A mathematically correct conv should show a small localized 3×3-ish response around that coordinate. **Under every layout hypothesis tried, the output was structurally identical to the pure-zero-input case** — i.e., the impulse produced no detectable effect anywhere. This means the bug isn't simply "channels in the wrong order" — the data likely isn't reaching the compute engine at all via this write path, or reaches it through a mechanism this manual zero-copy approach doesn't correctly engage (e.g. the "synthetic first-conv" input-loading path may expect something we haven't identified).

**Best remaining leads:**
- Try the classic `rknn_inputs_set` path again, now that we understand the "mini runtime" better — its earlier failure (`context config invalid!`) was never fully diagnosed and might be a separate, more tractable bug than the manual zero-copy path's silent failure.
- Actually read the regcmd bytes (see §5) to see literally what address the hardware DMA reads input from — the only fully conclusive way to resolve this.

---

## 4. Key structural discovery: size is compiler-decided, not runtime-computed

Traced (via disassembly of the **on-device runtime**, `librknnmrt.so`) the code path behind the `"memory size must be: %d, but is: %d"` error in `rknn_set_io_mem` (ARM32, entry `0x114b0`). The "expected size" value is read from a fixed struct offset (`ctx + 0x16c`) that is **never written anywhere in that function**, nor in `rknn_init`'s obvious call graph within the runtime — it's populated by a generic key-value parser reading a metadata blob **already embedded in the `.rknn` file**, not computed via `H×W×C` arithmetic at runtime at all.

Confirmed by checking the file's outer JSON footer: it only contains the *logical* shape (`"size": [1,4,8,8]`) — no stride/native fields — meaning the real native-size number lives in an **inner, separate metadata section** (near the `"task"`/`"regcmd"` field names found earlier in the file, likely FlatBuffers-encoded per the build log's `RKNNFlatcModelBuildPass`), written once by the compiler and never recomputed by the device.

### Following the trail to the compiler (`librknnc.so`)
This shifted the target from the ARM32 runtime (hard to tool against — see §6) to the **PC-side compiler core**, `librknnc.so` — critically, this is a **native aarch64** binary already present on this host (inside the `rknn-toolkit2` venv), so no cross-compilation or emulation needed at all.

- 28.5MB, stripped, but **RTTI/mangled C++ class names survive** (e.g. `rknn::ConvForTP_NC1HWC2_2_NCHW`), confirming a dedicated C++ class hierarchy handles layout conversion.
- Located the actual size-computation logic in `FUN_016bb6c0` (4928 bytes, heavily NEON-vectorized). Confirmed by direct code reading (not paraphrase) that the native shape array is ordered exactly `[N, C1, H, W, C2]` — matching the compiler's own build-log output.
- **The real formula** (traced line-by-line, both independently and cross-checked by a second Claude session running Ghidra natively):
  ```c
  tile_H = *(int*)(tensor_desc + 0x150);   // per-tensor alignment constant
  tile_W = *(int*)(tensor_desc + 0x14c);   // second, different alignment constant

  padded_W  = ceil(W / tile_H) * tile_H;
  padded_HW = ceil(padded_W * H / tile_W) * tile_W;
  native_size = padded_HW * C1 * C2;       // × element_size
  ```
- **However:** plugging in our two known-good data points (384 bytes for the 3-channel case, 512 bytes for the 4-channel case) shows `padded_HW = 64 = H×W` in both cases — i.e., **for our specific 8×8 test tensors, the tile-alignment ceiling operations are no-ops** (8 and 64 are already clean multiples of any plausible tile size). The formula is real and confirmed, but it doesn't explain our remaining bug — buffer *size* was correct from our very first attempt. The unresolved problem (§3) is about internal byte *ordering*, not size.
- The actual `tile_H`/`tile_W` write site (where these get initialized — likely fixed hardware/dtype constants) was not yet located; searching for it is the natural next step *if* the tile question becomes relevant again (e.g. for non-8-aligned tensor shapes).

---

## 5. Why we can't just read the regcmd bytes (yet)

Traced a real `SUBMIT` ioctl from our own known-good inference run (`strace -v -x`, ioctl request `_IOC(RW,'r',0x1,0x68)` = `IOCTL_RKNPU_SUBMIT`, `sizeof(struct rknpu_submit)=0x68=104` bytes — confirms our kernel-header-derived struct layout is exactly right). Observed real values: `flags=5` (`RKNPU_JOB_PC | RKNPU_JOB_PINGPONG`), `task_number=1` (single task, matches our 1-op model).

**Blocker:** `task_obj_addr` (`0xb198bee0` in one capture) is **not a dereferenceable userspace pointer**, even from within the very process that owns it — GDB reports `Cannot access memory at address`. This is a driver-internal DMA/GEM object handle, meaningful only kernel-side. Getting the actual `regcmd` bytes needs one of:
- A **kprobe** on the driver's `rknpu_submit_ioctl` (dynamic, no module build/insmod needed, lowest risk) — the clean way to read the struct exactly as the kernel sees it.
- **Frida**, hooking `librknnmrt.so`'s internal buffer-building calls *before* the ioctl — catches the bytes while still in an ordinary CPU-accessible buffer. This is the approach OpenNPU (the RK3588 open-compiler project) actually used ("captured bytes on the wire").
- Static analysis of `librknnmrt.so`/`librknnc.so` to find the buffer-construction code directly (in progress, see §4).

**Frida was attempted and abandoned** for this specific board: official builds are glibc-linked (won't run on uClibc without a from-source rebuild), and — more decisively — the board's `/tmp` is a 16.4MB tmpfs; pushing the ~30MB `frida-server` binary immediately overflowed it and hung the board's USB stack. A physical power cycle recovered it cleanly (confirmed: `/tmp` was empty after reboot, rootfs untouched — no data loss, just a transient overflow). **Any future large-file push to this board must go through the flash-backed `/oem` or `/userdata` partitions, never `/tmp`.**

---

## 6. Tooling notes / environment gotchas (for continuity)

Host was **aarch64** Linux during this investigation (historical: the current host
listed in `research/README.md` is x86-64); this caused most of the friction below.

- **`radare2`** (`apt install radare2`) — worked well for ARM32 disassembly + xref search of `librknnmrt.so`. Built-in decompiler (`pdc`) is unreliable on this binary (mixes x86/ARM register naming, garbled output) — don't trust it; read raw `pdf`/`pd` disassembly by hand instead.
- **Ghidra headless (own-built path):** downloaded portable JDK + Ghidra zip (no sudo needed). **Public Ghidra releases ship no `linux_arm_64` native decompiler binary** (only x86_64/win/mac) — a real gap in the aarch64-Linux ecosystem. Tried running the JVM itself under x86_64 QEMU emulation (via `tonistiigi/binfmt` + Docker): **the JVM's background-thread/signal-handler setup crashes under QEMU user-mode even with zero real work** (`pc=0x0` SIGSEGV, survived interpreter-mode/serial-GC/no-signal-handler flag combinations — genuine environmental wall, not a flag-tuning problem). **Resolution:** had a second Claude Code session run Ghidra natively on a real x86_64 Ubuntu machine instead — trivial there, no emulation needed. *(Lesson: for any JVM-based tool on an aarch64 host, don't fight QEMU — find/use real x86_64 hardware.)*
- **`librknnc.so` (the PC-side compiler core)** is **native aarch64** — no emulation needed at all, disassemble directly on this host.
- Cross-compiling **for the board** (`arm-rockchip830-linux-uclibcgnueabihf`, x86_64-hosted toolchain from Luckfox's SDK repo) *does* work under the same Docker+QEMU setup — confirmed with a real "hello world" that ran correctly on-device. Needed `--sysroot=<toolchain>/arm-rockchip830-linux-uclibcgnueabihf/sysroot` explicitly (binary has a bogus hardcoded `/opt/x-tool/...` sysroot path) and `-fno-use-linker-plugin` (the bundled `liblto_plugin.so` hits the same QEMU-mmap instability as some Python wheels — a recurring pattern on this host, unrelated to JIT specifically). *This toolchain is genuinely useful infrastructure for any future on-device C/C++ work.*
- **Docker + QEMU x86_64 emulation:** works fine for plain compiled binaries (gcc, objdump). **Fails unpredictably for:** certain prebuilt Python wheels (`numpy`'s manylinux wheel — "failed to map segment from shared object"; fixed *for that one case* by using Debian's apt-built numpy instead of pip's, but the same failure recurred on `dpkg` unpacking other `.deb`s under Debian bookworm specifically — switching base image to **Ubuntu 20.04** avoided it for plain `apt`, but not for `pip`-installed wheels, which still crash regardless of base image). **General rule: on this host, prefer natively-compiled/apt-installed x86_64 binaries over pip wheels when working under emulation; JIT runtimes (JVM) are effectively unusable under this QEMU setup regardless of workaround.**
- The old (v1.4.0, matching the board's original runtime) `rknn-toolkit2` only ships **x86_64** wheels — never got it running due to the above emulation instability. Abandoned in favor of using the **2.3.2 runtime directly** (§1) instead of downgrading the compiler.

---

## 7. Suggested next steps, roughly in order of effort/payoff

1. **Retry `rknn_inputs_set`** (classic path) now that the mini-runtime's quirks are better understood — may sidestep the whole manual-packing question if it works.
2. **kprobe on `rknpu_submit_ioctl`** — lowest-risk way to finally read real `regcmd` bytes kernel-side; we already have the exact struct layout from the GPL driver source (Luckfox's `luckfox-pico` SDK repo, `sysdrv/source/kernel/drivers/rknpu/`).
3. **Frida on `librknnmrt.so`**, but only after either (a) building a from-source uClibc-compatible `frida-server`, or (b) finding a lighter injection method — and always push large files to `/oem` or `/userdata`, never `/tmp`.
4. **Finish the RTTI class recovery** on `librknnc.so` (Ghidra's `RecoverClassesFromRTTIScript`, on the x86_64 machine) to get a clean decompile of `ConvForTP_NC1HWC2_2_NCHW`'s actual conversion method — separate from the size-calc function already found, and the most direct path to the real byte-ordering answer for §3.

---

## Appendix: useful addresses & paths

| Item | Value |
|---|---|
| Board chip | `rockchip,rv1103` (NPU compatible: `rockchip,rv1106-rknpu`, shared IP with RV1106) |
| Kernel NPU driver version | `v0.8.2` |
| Board's original runtime | `librknnmrt version: 1.4.1b9` (2022-10-19) |
| Runtime we use instead | `2.3.2 (429f97ae6b@2025-04-09T09:11:49)` |
| `rknn_set_io_mem` (runtime, ARM32) | `0x000114b0`, size 6108 bytes |
| `rknn_query` (runtime, ARM32) | `0x0000e03c` |
| Size-check error strings | `librknnmrt.so` `.rodata` `0x00032504`/`0x000325b8` |
| NC1HWC2 size-calc function (compiler, aarch64) | `FUN_016bb6c0` in `librknnc.so`, `0x016bb6c0`, 4928 bytes |
| `ConvForTP_NC1HWC2_2_NCHW` RTTI string | `librknnc.so` `.rodata` `0x017d2768` |
| `IOCTL_RKNPU_SUBMIT` | `_IOC(READ\|WRITE, 'r', 1, 0x68)` = `0xC0687201` |
| Kernel driver source | `LuckfoxTECH/luckfox-pico`, `sysdrv/source/kernel/drivers/rknpu/` |
| Matching runtime `.so` for any target version | `airockchip/rknn-toolkit2`, `rknpu2/runtime/Linux/librknn_api/<arch>/librknnmrt.so` |
