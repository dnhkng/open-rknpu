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
| B10 | F10 C>128 split accumulation | try the documented register sequences/edge cases | likely blocked; document the measured negative precisely | probe log in `docs/investigation-log.md` + roadmap note |
| B11 | E5 camera/V4L2 | list `/dev/video*`, driver and formats while `rkipc` runs | capture a frame, feed the NPU, print the result | example + board run, or a measured negative if the ISP is unavailable |
| B12 | E12 audio VAD | depends on F2/F6 | a small VAD graph on the supported subset, or a documented "gated on F6" negative | example or negative |

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
| F8 dma-buf | `research/probe_dmabuf.c`: 128/256 `CREATE` flag values accept a CMA-heap fd, exactly those with bit `0x80` set; the driver returns a device address | runtime binding: allocate the arena from a CMA-heap dma-buf (or import the producer's), expose the fd, and add a run path that skips the input copy; board test compares against recorded expected bytes, then the V4L2 example (E5) can consume the same buffer |
| F9 timing API | no userspace cycle counter; the board_bench harness times `ornpu_run` around `clock_gettime(CLOCK_MONOTONIC)` | promote that to a supported runtime call (`ornpu_run_timed`) plus a Python wrapper and a board timing suite |
| F10 C>128 | the front end caps at C1..128; the partial-sum mechanism is undocumented | craft containers with C=129+ and probe the register fields; expect a documented negative |
| E5 camera | `/dev/video0..20` (`stream_cif_mipi_id*`, `rkisp_mainpath`, `rkcif_scale_ch*`, `rkisp_lumapath`) exist, `rkipc` holds the ISP | capture one V4L2 frame (rkisp_mainpath or a CIF channel), feed the NPU through the F8 path if it lands, else copy into a container input |
| E12 audio VAD | Silero VAD needs 1-D conv (now supported, F2), C129 input, LSTM and dynamic shapes | a bounded 1-D audio front end is now possible; the VAD itself stays a documented negative unless F6 changes that |

Each remaining row keeps the definition of done above: host tests, regenerated reference
docs, a board run whose summary line is pasted into the suite README, a ledger row and a
checklist tick.
