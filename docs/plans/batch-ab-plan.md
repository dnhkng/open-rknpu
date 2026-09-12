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
