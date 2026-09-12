# Batch A + B execution plan

Implementation plan for the remaining 22 open rows of
[`docs/publish-checklist.md`](../publish-checklist.md) (58 closed at the time of writing).
"Batch A" is the eight host-only rows; "Batch B" is the twelve rows that need fresh board
evidence. This page is the plan; the checklist is the state.

Board: Luckfox Pico Mini B, RV1103, driver v0.8.2, USB serial `498063e3262e55b7`,
`/userdata` 4.1 MB free, `rkipc` running. `adb` is not on `PATH` in this workspace; use
`ADB=/home/dnhkng/Unity/Hub/Editor/2022.3.14f1/Editor/Data/PlaybackEngines/AndroidPlayer/SDK/platform-tools/adb`.
The cross toolchain is present at `research/toolchain/` (git-ignored) and `make board-io`
builds the v5 runner with it.

## Ground rules

1. **Board safety.** `rkipc` must stay alive (it is also the camera appliance). Never leave
   a staged tree under `/userdata` while an experiment runs; a wedged NPU is recovered with
   `adb shell reboot`. Keep `/userdata` under ~4 MB.
2. **Evidence, not claims.** A feature that changes or extends the compiler lands with:
   a deterministic generator (`research/build_<name>_suite.py`), a suite (`modelNNN.onnx`,
   `.bin`, `inputNNN.u8`, `expectedNNN.i8`, `manifest.json`, `board_results_*.json`,
   `board_summary.txt`, `README.md`), a ledger row, and host tests for the parser, the
   reference and the rejection cases.
3. **The baseline contract.** `research/container_baseline.json` may only change when the
   container change is intended, and then every affected suite is re-run on the board and
   the ledger/README evidence is refreshed in the same commit. Pure front-end additions
   (F1, F2) and new profiles must not perturb any existing container.
4. **Measured negatives are results.** F8, F10 and E12 may end as documented "not supported
   by this hardware/driver, here is the probe and the error" rows. That closes them
   honestly, but only with a board probe that proves it.
5. **No silent fallbacks.** A new bounded feature raises a specific `ValueError` outside its
   envelope; the docs and the error index are regenerated (`research/build_reference_docs.py`).

## Batch A — host only

| # | Row | Deliverable | Verification |
| --- | --- | --- | --- |
| A1 | T14 | `tests/test_fetch_scripts.py`: the fetch scripts' URL pins, sha256 verification, mismatch and offline error paths (monkeypatched `urlopen`, no network) | module green, no sockets; a corrupted archive must raise with the expected message |
| A2 | A5 | `docs/branch-protection.md`: the required checks, why each gate exists, the ruleset JSON to apply, and how to reproduce every gate locally | docs link check; `make docs-check` |
| A3 | E13 | `examples/multi_model/`: two containers alternating on one board session, shared-arena lessons, measured table; host path compiles and decodes both | host runs green under `tests/test_examples_host.py`-style coverage; board numbers recorded in the example README |
| A4 | E7 | `examples/depthwise_separable/`: depthwise + pointwise + pool classifier head, integer reference, board accuracy | reference bytes exact vs the composed container; board run adds a ledger row |
| A5 | E9 | `examples/benchmark/`: N-model latency/throughput harness with a recorded table; host mode compiles/inspects, board mode drives the runner | runs on the host; board numbers recorded with the exact command |
| A6 | F11 | `open_rknpu.mutable` helper API: build a v4 container, replace named constants between inferences, with validation and a documented workflow | host tests for replacement, size/kind validation and the emitted container; example in `examples/cookbook/` |
| A7 | F12 | `docs/api-stability.md` + `tests/test_public_api.py`: the supported surface (`scheduler`, `calibration`, `sequence`, `compose`, `model`, `mutable`), the container-format promise, and a test that fails when a public name disappears | test pins `__all__`/signatures; docs page in nav |
| A8 | F13 | integer references for depthwise `ConvTranspose`, `pooling` and the transposed off-centre taps, each compared byte-for-byte with the recorded board suite | new reference functions + tests; replay of the retained suites must stay exact |

## Batch B — board required

Every row starts with a **probe** on the board (or the driver, for F8) that either proves the
capability or produces the measured negative. Probes are written down in
`docs/investigation-log.md` before implementation.

| # | Row | Probe | Implementation | Evidence |
| --- | --- | --- | --- | --- |
| B1 | F1 `MatMul`/`Gemm` | none needed: lower to 1×1 Conv at the front end | `normalize.py` lowering for rank-2/4 MatMul/Gemm with constant weights, batched and per-channel cases | new `matmul_suite`; host references; ledger row |
| B2 | F2 1-D conv | none needed: map `[k]` to `[1,k]` | front-end rank-3 input acceptance and kernel mapping into the existing conv/transposed emitters | new `conv1d_suite`; ledger row |
| B3 | F7 rectangular/one-sided/K>31 | envelope probe on the board (which shapes the DPU accepts) | extend the rectangular kernel rewrite bounds to one-sided padding and large K where the probe allows | `rect_kernel_suite` (or a documented bound if the probe refuses) |
| B4 | F3 calibration parity | none | add `calibration_ranges` to the chain/join/walk profiles that lack it, with the same band validation | `chain_calibration_suite`; ledger row |
| B5 | F9 timing API | probe the submit/wait ioctl round trip and any driver counter | runtime API (`ornpu_run_timed`, per-task timings where available) + Python surface | host tests + a board timing suite; numbers in `docs/performance.md` |
| B6 | F5 walk elementwise | none | elementwise ops inside a walk chain and multi-join DAGs | extend the walk suites; new models with board evidence |
| B7 | F6 bounded `Concat`/`Slice`/`Resize`/`Softmax`/`ReduceMean` | one probe per op: which cases the hardware/driver can express | implement the bounded subset that the probe proves; reject the rest with a specific message | per-op suites with board evidence, or a measured negative per op |
| B8 | F4 chain border zero point | register probe for `0x1184` on a chain model | set the border zero point to the chain's band instead of leaving −128 | rebuild the chain family on the board; re-baseline with fresh evidence |
| B9 | F8 dma-buf / zero-copy | inspect the driver's `dma-buf` support and `/dev/rknpu` mmap surface | only if the probe proves it; otherwise document the negative | probe log + (if supported) a zero-copy example |
| B10 | F10 C>128 channels | **done**: no split accumulation is needed; the walls are structural (511 32-lane weight parts in, nine-bit output block index) | `native.py` + both decoders bound input C1..16352 / output C1..8192 | probe log + `research/wide_channel_suite/` (13 models / 45 inferences / 168,994 exact bytes) + `research/probe_wide_channel_wall.py` |
| B11 | E5 camera/V4L2 | **done**: every capture node is multi-planar (`device_caps=0x04201000`) and `rkisp_mainpath` is held by `rkipc` | shipped harness replays a recorded frame and refuses the live nodes precisely | `examples/camera/` (recorded replay 8 inferences / 8,192 exact bytes; `--emit-input` == `convert.py` for yuyv/nv12/rgb) + measured negative |
| B12 | E12 audio VAD | **done as a measured negative**: F2/F10 removed the 1-D and C129 blockers; the 1-D encoder chain still is not dispatched | documented per-blocker envelope | `docs/plans/primitive-roadmap.md` + `docs/investigation-log.md` (F10 entry) |

## Order and parallelism

1. **Batch A** first (all eight rows, host only) — it is independent of the board and can run
   in parallel with the Batch B probes.
2. **B1, B2, B4, B6** next: front-end/emitter work that reuses verified primitives, so the
   board runs are new *shapes* on proven register programs.
3. **B3, B7, B5** after their probes; each probe decides how much of the row is reachable.
4. **B8 last among the emitters**: it changes existing containers, so it needs the whole
   chain family re-run and an intentional re-baseline.
5. **B9, B10, B11, B12** are probe-first and may close as documented negatives; they must not
   block the rest.

## Definition of done per row

* checklist row marked ✅ with the evidence path next to it;
* tests added (host) and the affected suites retained with `board_results_*.json`;
* docs regenerated (`build_reference_docs.py`) and updated (support matrix, performance,
  roadmap as applicable);
* `make lint test coverage baseline campaign evidence perf reproducible docs-check` green;
* pushed to `main` with CI green.

## Measured state (2026-09-12, after the first two batches)

Closed: Batch A in full, plus F1, F2 and F3. What the probes decided for the rest:

| Row | Probe result (exact) | Next action |
| --- | --- | --- |
| F4 chain border zero point | not probed yet | register probe for `0x1184`, then the whole chain family re-run |
| F5 walk elementwise | `Conv -> Add(const) -> MaxPool -> Conv` is rejected at dispatch (`depthwise sequence requires Conv[/Relu] -> depthwise Conv`); the walk handles pools and joins but not an elementwise stage | extend the walk's stage parser to lower an elementwise stage, then re-run the walk suites |
| F6 bounded ops | `GlobalAveragePool`, `AveragePool 8x8/8`, `Concat`, `Softmax`, `Slice`, `Resize` and `ReduceMean` after a Conv are all rejected (`sequence lowering currently supports Conv[/Relu] followed by 2x2 pooling`, `unsupported pooling attributes, shape, or graph connections`); `MaxPool 2x2` is accepted | implement the two that the hardware can express - an 8x8->1x1 average/mean through the three-level reduction emitter (which already exists), and `Concat` of sibling Conv branches reading the same input by stacking weights into one Conv; document `Softmax`, `Slice` and `Resize` as measured negatives |
| F7 rectangular/one-sided | `k1x3` is accepted (walk, 8,304 B); K=31 accepted (native16, 62,768 B); K=33 rejected by the documented `odd K1..31` bound; a 3x3 Conv with one-sided pads `[0,1,0,1]` is rejected by the front-end output-shape check even with the correct output shape | accept asymmetric padding where the native emitter's explicit-pad path allows it; keep K>31 rejected with the measured message |
| F8 dma-buf | **done**: the runtime binding landed (`ornpu_open_shared` / `ornpu_input_view` / `ornpu_run_prefilled`) and the whole packed `add_geometry_suite` passes through it (32 models / 64 inferences / 19,968 exact bytes, 0 mismatches) | E5 can now capture into the shared arena |
| F9 timing API | no userspace cycle counter; the board_bench harness times `ornpu_run` around `clock_gettime(CLOCK_MONOTONIC)` | promote that to a supported runtime call (`ornpu_run_timed`) plus a Python wrapper and a board timing suite |
| F10 C>128 | **done**: C16352 in / C8192 out are byte-exact; C16368 (512 weight parts) times out and soft-resets the core, C16384 out writes the first 8192 channels and then wrong bytes | lift the flat 128 bound to the measured walls in the emitter and both decoders; pin a suite on each side |
| E5 camera | **probed**: `rkisp_mainpath` (2304x1296 NV12) is held by `rkipc` and a second streaming client gets `VIDIOC_REQBUFS ... Device or resource busy`; free CIF/scale nodes never produce a frame while nothing streams them | write the generic `--device/--format/--frame` capture example and verify the conversion + NPU plumbing with a recorded frame on the board (F8's shared arena is the hand-off); document the `rkipc` constraint |
| E12 audio VAD | Silero VAD needs 1-D conv (now supported, F2), C129 input, LSTM and dynamic shapes | a bounded 1-D audio front end is now possible; the VAD itself stays a documented negative unless F6 changes that |

Each remaining row keeps the definition of done above: host tests, regenerated reference
docs, a board run whose summary line is pasted into the suite README, a ledger row and a
checklist tick.
