# open-rknpu

**An open, from-scratch compiler and runtime for the Rockchip RV1103 / RV1106 NPU — no
vendor SDK, no RKNN library, no captured binaries.**

`open-rknpu` turns a static ONNX convolution graph into a task program the RV1103 NPU
executes directly: an MIT-licensed Python compiler in this repository emits the register
commands, weight/quantization blocks and container format itself, and a libc-only C runtime
talks to `/dev/rknpu` and nothing else. Every claim in the repository is backed by a board
run on the attached Luckfox Pico Mini B (`rockchip,rv1106-rknpu`, driver v0.8.2) and by
integer-exact comparisons against an independent Python reference.

```text
ONNX graph ──► normalize ──► scheduler (profile dispatch) ──► composer (tasks + arena)
         ──► container (v5 named-tensor ABI) ──► /dev/rknpu ──► INT8 outputs
```

* **A real trained model runs entirely on the NPU**: `examples/mel-kws/` trains a
  4,090-parameter mel-CNN on spoken digits and runs the whole graph on the board —
  **98.00%** INT8 test accuracy, 191,968 / 192,000 board-exact output bytes, 2.54 ms per
  utterance.
* **311 host tests**, a **2,244-model container baseline** (every published suite model
  still compiles to identical bytes), and a **121-row board ledger** counting
  **1,696 models / 28,266 inferences / 10,216,467 exact output bytes**.
* **Low-level op examples**: `examples/primitives/` has one runnable script per primitive —
  a single Conv, Conv+pool, chains, depthwise, elementwise joins, broadcast Mul, LUT
  activations, transposed Conv, join DAGs, calibration — plus an MNIST-style model that
  combines them.

## Quick start (host only)

```sh
python -m pip install -e .                # needs NumPy and ONNX
open-rknpu compile model.onnx -o model.bin --sequence
open-rknpu inspect model.bin              # container header, tasks, tensors as JSON
```

```python
from open_rknpu.scheduler import compile_sequence
binary, meta = compile_sequence("model.onnx", input_scale=1/255, input_zero_point=0)
print(meta["profile"], len(binary), "bytes")
```

```sh
python examples/primitives/01_native_conv.py      # runs a primitive end to end on the host
PYTHONPATH=src python research/verify_suites.py   # 2,244 models vs the checked-in baseline
PYTHONPATH=src python -m unittest discover -s tests
```

Running on hardware needs the cross toolchain and a board — see
[board](docs/board.md) and the per-example READMEs.

## What is supported

| Family | Verified on the board (bounds are conservative and evidence-backed) |
| --- | --- |
| Dense Conv | batch 1–16, input C1–128, output C1–128, H/W 1–128, odd K1–31, stride 1–4, dilation ≤17, explicit/auto padding, serial height tiling for large planes |
| Kernel rewrites | even/rectangular kernels through 5×5, grouped through group 32, effective dilation ≤5 expanded into weights |
| Depthwise | RGB stem, multiplier 1, H/W 5–8, C1–16, K1/3/5, stride 1/2; wider channels via dense rewrites |
| Pooling / reduction | 2×2 stride-2 Max/AveragePool, multi-stage reduction to 1×1, pools inside Conv chains |
| Elementwise | Add/Mul/Sub/Max between branches, scalar/per-channel/spatial constants, per-channel Mul, runtime scale/residual operands, fan-out to 3–5 heads, diamond and general join DAGs |
| Activations | fused Relu, Clip[0,6]/ReLU6, scalar/per-channel LeakyReLU and PReLU, bounded Sigmoid/Tanh LUT |
| Transposed Conv | depthwise and dense K1/K2/K3, per-axis stride 1/2, padding/output-shape modes, K5 via sparse rewrites |
| Quantized import | constant-parameter `QLinearConv` and input/weight DQ → Conv → Q graphs, INT8 weights preserved |
| Runtime | UINT8 input / INT8 output packed NHWC, format v5 named tensors, batched submission (a whole DAG is one job), non-blocking pipelining, fence-free completion |

The full bounds, the register/numerical findings and the board evidence index are in
[research/COVERAGE_EXPANSION_RESULTS.md](research/COVERAGE_EXPANSION_RESULTS.md); the
per-op walkthrough is [primitives](docs/primitives.md).

## Repository layout

| Path | Contents |
| --- | --- |
| `src/open_rknpu/` | the compiler: front end, scheduler, one module per profile/emitter, the stage composer and the container writer |
| `runtime/` | the libc-only board runtime (`open_rknpu.c/.h`, `main.c`, `Makefile`) and `sequence_format.md`, the container specification |
| `tests/` | 311 host tests plus the C board harnesses (`board_*.c`) |
| `examples/` | `primitives/` (one script per low-level op), `mnist/` and `fashion/` (hybrid CNN classifiers), `mel-kws/` (whole model on the NPU) |
| `research/` | the evidence: one directory per suite with its models, reference outputs, `manifest.json`, `board_results_*.json` and `README.md`, the generators (`build_*.py`), the oracle captures, and the verification scripts |
| `docs/` | architecture, primitive catalog, quantization, container format, board workflow, verification, roadmap, investigation log, planning records |

Reproducible contracts live at the repository root of `research/`:
`research/verify_suites.py` with `research/container_baseline.json`,
`research/campaign_sweep.py` and `research/check_docs_links.py`.

## Documentation

* [docs/README.md](docs/README.md) — documentation index.
* [Getting started](docs/getting-started.md) — install, compile your first graph, read the metadata.
* [Architecture](docs/architecture.md) — the pipeline, the scheduler's profile order, the composer, and how to add a primitive.
* [Primitives](docs/primitives.md) — every verified op, its bounds, its emitter and its example.
* [Quantization](docs/quantization.md) — UINT8/INT8 bands, zero points and calibration.
* [Container format](docs/container-format.md) — the on-device task/register/constant layout.
* [Board workflow](docs/board.md) — the hardware, adb staging, running, and timing.
* [Verification](docs/verification.md) — the baseline contract, the ledger and the board discipline.
* [Roadmap and limits](docs/roadmap.md) — what is deliberately out of scope and what is left.
* [Investigation log](docs/investigation-log.md) — the 60+ entry record of how the register
  profile and every primitive were recovered, with the failed hypotheses kept.

## Hardware and scope

The reference board is a Luckfox Pico Mini B whose NPU is the shared RV1106 NPU IP
(`rockchip,rv1106-rknpu`, driver v0.8.2, non-IOMMU). The compiler targets the documented
register profile of that IP; a *distinct RV1106 SoC* is not claimed. This is an
experimental research compiler: it accepts a bounded class of static graphs and rejects
everything else loudly rather than falling back to a vendor runtime. Recurrence
(LSTM/GRU), 1-D convolution, `MatMul`/`Gemm`, multi-stage detection heads and dynamic
shapes are out of scope today — see [roadmap](docs/roadmap.md).

## License

MIT for the compiler, runtime, tests, examples and documentation (see [LICENSE](LICENSE)).
Vendor binaries, the Rockchip cross toolchain, GPL kernel sources and the datasets are
**not** redistributed here; the documents that describe them stay in `research/` as
provenance records, and the examples fetch their datasets themselves.
