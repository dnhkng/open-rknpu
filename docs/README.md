# Documentation

`open-rknpu` is an open compiler and runtime for the Rockchip RV1103 / RV1106 NPU. These
documents are ordered from "run your first model" to "recover a new register field".

| Document | Read it for |
| --- | --- |
| [getting-started.md](getting-started.md) | installation, the CLI and Python entry points, a complete compile → inspect → run-on-board walkthrough |
| [architecture.md](architecture.md) | the compile pipeline, the scheduler's profile order, the composer (stages, bindings, liveness), the container writer, and the recipe for adding a new primitive |
| [primitives.md](primitives.md) | the low-level op catalog: every verified primitive, its bounds, its emitter module, its example script and the register/quantization notes that matter |
| [quantization.md](quantization.md) | UINT8 input and INT8 grid conventions, zero points, requantization, calibration (`minmax`/`percentile`/`kl`) and how accuracy is measured |
| [container-format.md](container-format.md) | the on-device format (headers v1–v5, task descriptors, register words, packed constants, the named-tensor table) |
| [board.md](board.md) | the reference hardware, the driver and `/dev/rknpu`, adb staging, running a container, timing, and what the board cannot do |
| [verification.md](verification.md) | the evidence discipline: exactness vs the Python reference, the container baseline, the board ledger, the campaign sweep, and how to add a suite |
| [api.md](api.md) | the Python API surface and the module inventory |
| [roadmap.md](roadmap.md) | what is supported today, what is deliberately out of scope, and the measured residuals |
| [investigation-log.md](investigation-log.md) | the chronological record of how the register profile and each primitive were recovered (newest first, failed hypotheses kept) |
| [plans/](plans/) | the internal planning records: completion plan, pipelining plan, cleanup plan, primitive roadmap, coverage matrix, project goals |

Examples with measured board results live in [`../examples/`](../examples/README.md):
`primitives/` (one script per op), `mnist/` and `fashion/` (hybrid classifiers),
`mel-kws/` (a trained audio model running entirely on the NPU).

## Conventions used throughout

* **"Verified"** means: compiled by this repository, executed on the attached board, and
  compared byte-for-byte against an independent Python integer reference built from the
  same quantization parameters. Host-only agreement is never called board-verified.
* **"Bounded"** means the profile rejects out-of-range shapes with an explicit
  `ValueError`; there is no silent fallback to a slower or vendor path.
* **"Exact"** means every output byte matches the reference. The ledger counts exact
  output bytes so a partial result cannot be mistaken for a full one.
* Register numbers are written in hex (`0x1070`); signed grid values are INT8 with a
  zero point, and the hardware's internal accumulator is INT32 unless stated otherwise.
