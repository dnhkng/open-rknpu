# Examples

Four sets of examples, ordered from "one primitive at a time" to "a trained model that runs
entirely on the NPU". All of them are host-only unless a step says otherwise; board steps
name their measured result.

| Example | What it shows | Measured result |
| --- | --- | --- |
| [`primitives/`](primitives/README.md) | one runnable script per low-level op: a single Conv, Conv+pool, chains, depthwise, elementwise joins, broadcast Mul, LUT activations, transposed Conv, join DAGs, calibration — and a script that combines them into an MNIST first stage | host reference matches every compiled container; the final script compiles a real pretrained prefix |
| [`mnist/`](mnist/README.md) | the **hybrid** pattern: the first Conv/Relu/Pool stage on the NPU, the rest of the classifier on the CPU, with a model-specific C suffix | 12,544 exact NPU bytes; logits within 9.2e-5 of an independent ONNX suffix evaluation |
| [`fashion/`](fashion/README.md) | the same graph trained on Fashion-MNIST, plus a calibrated-range variant | **88.15%** board accuracy (10,000 images), 9,957/10,000 integer logits identical to float |
| [`mel-kws/`](mel-kws/README.md) | the **fully offloaded** pattern: a trained audio CNN (Free Spoken Digit Dataset) whose whole graph runs on the NPU, with percentile calibration | **98.00%** INT8 accuracy, 191,968/192,000 exact bytes, 2.54 ms per utterance |

## Which pattern should I copy?

* **One op, or a small fixed pipeline** → `primitives/`. Each script is self-contained and
  prints the profile, the container size and the reference check.
* **An existing model with unsupported ops in the middle** → `mnist/`. Put the prefix on
  the NPU, keep the unsupported suffix on the CPU, and verify the seam exactly.
* **A model you can train yourself inside the supported op set** → `mel-kws/`. Train on the
  exact bytes the runtime feeds, calibrate every Conv, and the whole graph fits one
  container.

## Common recipe

Datasets are never redistributed: `examples/fetch_idx.py` downloads the MNIST and
Fashion-MNIST IDX files (sha256-pinned) and `examples/mel-kws/fetch_data.py` downloads FSDD;
the pinned pretrained ONNX models for the classifier examples ship in
`research/pretrained/`.

Every example follows the same five steps:

1. **Build or fetch the graph.** `primitives/` builds ONNX graphs in code; `mnist/` and
   `fashion/` use a pinned pretrained ONNX that ships in `research/pretrained/`;
   `mel-kws/` trains one (`fetch_data.py` downloads the dataset).
2. **Quantize.** Analytic bands for random-weight smoke tests, `calibration.measure`
   (`minmax`/`percentile`/`kl`) for anything trained.
3. **Compile.** `open_rknpu.scheduler.compile_sequence` returns `(container, meta)`;
   `meta["profile"]` tells you which emitter accepted the graph.
4. **Check on the host.** Compare against the profile's Python integer reference before
   touching hardware; a container that disagrees with its reference will not be fixed by a
   board run.
5. **Run on the board.** Cross-compile the runtime with a `board_*.c` runner, stage the
   container plus inputs and reference outputs, and compare every byte. `mel-kws/main.c`
   shows the general runner (sizes come from the inspected tensor table, so the same binary
   runs any container).

The exact commands for each example are in its README; the shared details (cross compile,
adb staging, `rkipc`, space limits) are in [board](../docs/board.md).
