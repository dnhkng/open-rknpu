# Documentation

`open-rknpu` is an open compiler and runtime for the Rockchip RV1103 / RV1106 NPU. These
documents are ordered from "run your first model" to "recover a new register field".

## Start here

| Document | Read it for |
| --- | --- |
| [getting-started.md](getting-started.md) | installation, the CLI and Python entry points, a complete compile → inspect → run-on-board walkthrough |
| [support-matrix.md](support-matrix.md) | one table per op family: what is accepted, with which bounds, by which profile, with which example and evidence — plus what is rejected and why |
| [troubleshooting.md](troubleshooting.md) | symptom → cause → fix, built from the real rejection messages |
| [primitives.md](primitives.md) | the narrative primitive catalog: what each op does, its emitter, its reference and its bounds |
| [glossary.md](glossary.md) | CNA, DPU, ERDMA, native16, band, tail control, arena, profile … |

## The compiler

| Document | Read it for |
| --- | --- |
| [architecture.md](architecture.md) | the compile pipeline, the scheduler's profile order, the composer (stages, bindings, liveness), the container writer, and the recipe for adding a primitive |
| [quantization.md](quantization.md) | UINT8 input and INT8 grid conventions, zero points, requantization, calibration (`minmax`/`percentile`/`kl`) and how accuracy is measured |
| [registers.md](registers.md) | the generated task-register reference: every register a task writes, its default, and what is decoded versus still unknown |
| [errors.md](errors.md) | the generated error index: every user-facing message, where it comes from and what to do about it |
| [api.md](api.md) | the Python API surface and the module inventory |
| [verification.md](verification.md) | the evidence discipline: exactness versus the Python reference, the container baseline, the board ledger, the campaign sweep, and how to add a suite |
| [roadmap.md](roadmap.md) | what is supported today, what is deliberately out of scope, and the measured residuals |

## The runtime and the board

| Document | Read it for |
| --- | --- |
| [container-format.md](container-format.md) | the on-device format (headers v1–v5, task descriptors, register words, packed constants, the named-tensor table) |
| [container-example.md](container-example.md) | one real container walked field by field, with a hexdump, a decoded dump and annotated registers |
| [c-api.md](c-api.md) | using the libc-only runtime from C: lifecycle, structures, packing rules, async API, error handling, a minimal program |
| [board.md](board.md) | the reference hardware, the driver and `/dev/rknpu`, adb staging, running a container, timing, and what the board cannot do |
| [board-access.md](board-access.md) | getting a shell on the board and the original access and troubleshooting notes |
| [performance.md](performance.md) | how latency was measured, the recorded results, the per-family cost model, memory sizing, and how to benchmark your own model |

## Project records

| Document | Read it for |
| --- | --- |
| [investigation-log.md](investigation-log.md) | the chronological record of how the register profile and each primitive were recovered (newest first, failed hypotheses kept) |
| [provenance.md](provenance.md) | where the knowledge came from: board experiments, public kernel sources read as documentation, and the vendor runtime as a black-box oracle |
| [publish-checklist.md](publish-checklist.md) | the pre-publication gap analysis: blockers, missing features, tests, documentation and examples, with effort estimates |
| [plans/](plans/) | the internal planning records: completion plan, pipelining plan, cleanup plan, primitive roadmap, coverage matrix, project goals, milestone record |

Examples with measured board results live in [`../examples/`](../examples/README.md):
`primitives/` (one script per op), `cookbook/` (task-oriented recipes), `mnist/` and
`fashion/` (hybrid classifiers), `mel-kws/` (a trained audio model running entirely on the
NPU). [THIRD_PARTY.md](../THIRD_PARTY.md) lists every non-MIT component.

## Conventions used throughout

* **"Verified"** means: compiled by this repository, executed on the attached board, and
  compared byte-for-byte against an independent Python integer reference built from the
  same quantization parameters. Host-only agreement is never called board-verified.
* **"Bounded"** means the profile rejects out-of-range shapes with an explicit
  `ValueError`; there is no silent fallback to a slower or vendor path.
* **"Exact"** means every output byte matches the reference. The ledger counts exact
  output bytes so a partial result cannot be mistaken for a full one.
* Register numbers are written in hex (`0x1070`); signed grid values are INT8 with a zero
  point, and the hardware's internal accumulator is INT32 unless stated otherwise.
