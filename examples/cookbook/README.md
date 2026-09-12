<!-- SPDX-License-Identifier: MIT -->

# open-rknpu cookbook

Eight host-only, deterministic examples that answer the questions the
`examples/primitives/` folder does not: how a compiled container is *loaded and
run*, how already-quantized models are imported, how mutable parameters are
replaced (by hand and through the shipped `open_rknpu.mutable` API), why batched
submission wins on deep chains, how to check the installed package, what a
rejection looks like, and what of the tree's ONNX zoo compiles.

No board, no network, no PyTorch and no vendor toolchain is needed: the board
commands are printed as text, and every container is written under
`examples/cookbook/build/` (git-ignored via `examples/*/build/`).

```sh
PYTHONPATH=src python examples/cookbook/01_hello_npu_c.py     # ... 02 ... through 08
# or all of them:
for s in examples/cookbook/0*.py; do PYTHONPATH=src python "$s" || exit 1; done
```

Each script exits 0, prints a short readable report and a one-line summary, and
**asserts** its claims (container structure, byte-exact integer output, rejection
messages, ...). Nothing here is a print-only demo.

## The scripts

| Script | Demonstrates | Run | Asserts |
| --- | --- | --- | --- |
| `01_hello_npu_c.py` + `hello_npu.c` + `01_hello_npu_c.md` | the minimal C program: `ornpu_inspect`/`ornpu_open`/`ornpu_run`/`ornpu_close`, the Makefile snippet and the cross-compile + adb commands | `PYTHONPATH=src python examples/cookbook/01_hello_npu_c.py` | container decodes to 1 task with the expected packed shape, recompiles byte-identically, writes `model.bin` + `input.u8`; `hello_npu.c` compiles with host gcc |
| `02_quantized_import.py` | `QLinearConv` and `DQ->Conv->Q` import onto native16 with the supplied INT8 weights | `PYTHONPATH=src python examples/cookbook/02_quantized_import.py` | source INT8 weight bytes survive bit for bit (read back from the payload), unused lanes hold the weight zero point, integer output equals an independent accumulator/requantizer/ saturation implementation on 4 cases |
| `03_mutable_parameters.py` | v4 `mutable_weights=True` (`conv.parameters`) and `mutable_constants=True` (`mul.factor`), and `ornpu_set_constant` semantics | `PYTHONPATH=src python examples/cookbook/03_mutable_parameters.py` | descriptor name/kind/offset/size, task programs byte-identical between the two same-band containers, splice reproduces the second container exactly, the integer output changes by the expected amount |
| `04_batched_and_pipelined.py` | serial vs batched vs the deep-chain (double-buffered) container for one 8-layer chain | `PYTHONPATH=src python examples/cookbook/04_batched_and_pipelined.py` | tail control words (`0x28` serial, `0x40` linked), engine runs (8 vs 1), arena shrink (24576 -> 16384 B), all three outputs byte-identical; prints the measured board table + the `tests/board_async.c` command |
| `05_wheel_installed.py` | using the installed distribution, not `src/`, located via `importlib` | `PYTHONPATH=src python examples/cookbook/05_wheel_installed.py` | the source checkout is detected and refused politely (exit 0), and a child interpreter with `PYTHONPATH` cleared probes the installed package; otherwise it compiles a Conv, decodes the container and prints module path + version |
| `06_troubleshooting.py` | five deliberately unsupported graphs and their exact rejection text | `PYTHONPATH=src python examples/cookbook/06_troubleshooting.py` | each raises `ValueError` containing the expected substring, with the fix/roadmap pointer printed |
| `07_zoo_compatibility.py` | a bounded, network-free compatibility scan of the ONNX files in the tree | `PYTHONPATH=src python examples/cookbook/07_zoo_compatibility.py` | table is non-empty, at least one model compiles and at least one is rejected, scan is bounded and fast (< 60 s) |
| `08_mutable_api.py` | the supported mutable-parameter workflow: `compile_mutable`, `constant_regions`, `graft_region`, `program_bytes` | `PYTHONPATH=src python examples/cookbook/08_mutable_api.py` | a pinned band makes two compiles' task programs identical; `graft_region` reproduces the donor container byte-for-byte, refuses a donor from another band with the exact message, and the integer reference shows the output change |

`cookbook_common.py` is the shared helper: it loads
`examples/primitives/common.py` (graph builders, `compile_and_report`, `qfrom`,
`deterministic_cases`, ...) and adds container readers (`registers`,
`constant_region`, `tail_words`), the host-side `splice_constant` and
`checksum_masked`. Cookbook scripts do not duplicate the primitives helpers.

## When to use which

* **Write your first board program** -> `01_hello_npu_c.md`; it is the C API in one
  file and the two build commands.
* **Your model is already quantized** (QLinearConv or Q/DQ) -> `02`; the compiler
  preserves your INT8 weights instead of re-quantizing them. Check the band and the
  weight codes, then compile.
* **You need to swap weights or a Mul factor at runtime** (a personalization step,
  a calibration update) -> `03`; `ornpu_set_constant` replaces one *complete*
  packed region, so the replacement must share the container's band.
* **You are choosing a submission shape** for a deep chain -> `04`; serial is the
  default, batched wins from about four to eight same-engine tasks, and
  `reuse_intermediates=True` halves the arena. The board numbers and the exact
  reproduction command are in the report.
* **You installed the wheel and want to prove you are not testing `src/`** -> `05`.
* **Compilation fails** -> `06` first (the message and the roadmap pointer), then
  `docs/roadmap.md` and `docs/plans/primitive-roadmap.md`.
* **You want to know whether your ONNX file is in the envelope** -> `07`, then
  the full `tests/` suite for the precise profile bounds.

## Where to go next

* `examples/primitives/` - one verified operator at a time, each checked against
  the profile's Python integer reference and published as a board suite; `common.py`
  is the reference for graph builders and board recipes.
* `examples/mnist/` - a real trained model end to end (build, sanity, accuracy)
  with the calibrated-band workflow.
* `examples/mel-kws/` - the trained audio model that runs entirely on the NPU,
  including calibration and the host-side feature stage.
* `docs/` - start at `docs/getting-started.md`, then `docs/board.md` (build, adb,
  timing), `docs/api.md` (the C API), `docs/container-format.md` (the bytes),
  `docs/quantization.md` (bands and requantization), `docs/primitives.md` and
  `docs/roadmap.md` (the verified envelope and its edges).
* `tests/board_io.c` is the reference multi-model runner; `research/run_v5_suite.py`
  stages a suite and records board results.

## Notes and known gaps

* `docs/performance.md` is referenced by the task that produced this folder but is
  **not present in this tree**; `04_batched_and_pipelined.py` therefore cites
  `docs/board.md` ("Timing") and `docs/plans/pipelining-plan.md` (S2/S4), which are
  the measured record.
* `05_wheel_installed.py` refuses the source checkout by design. Clearing
  `PYTHONPATH` is what makes the installed distribution visible; if the installed
  wheel predates the current compiler the script reports the missing modules and
  exits 0 rather than pretending to smoke-test it.
* `07_zoo_compatibility.py` is bounded on purpose (16 candidates, 4 MiB per file,
  a 60 s budget). The complete regression corpus lives in `research/*_suite/` and
  is exercised by `tests/` and `make`.
