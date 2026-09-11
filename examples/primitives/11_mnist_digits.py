"""SPDX-License-Identifier: MIT

Primitive 11/11: the MNIST first stage built from the pinned pretrained weights.

This is the "combine the primitives into a real model" example. It loads
`research/pretrained/mnist/normalized.onnx` (the pinned ONNX Model Zoo MNIST
network used by `examples/mnist/`), takes its first three nodes
`Conv(1->8, 5x5, pad2) -> Relu -> MaxPool(2x2/2)`, compiles that prefix with
`open_rknpu.scheduler.compile_sequence` and verifies the host integer reference.
The compiled container is the terminal-pool profile: the legacy 8x8 image emitter
also accepts a single-channel image up to 32x32 (`src/open_rknpu/compiler.py`), so
the 28x28 grayscale input and the 14x14x8 pooled output run as two NPU tasks.
The input band (scale 0.2710929811, zero point 135) is the band derived from the
pinned fixture in `research/mnist_first_suite/manifest.json`; changing it changes
quantization quality, not the reference semantics.

Where the rest of the graph runs:

* `examples/mnist/` continues this graph *on the CPU*. Its `build.py` compiles the
  same prefix for the NPU and runs Conv2/Relu/Pool2 plus the 256x10 MatMul head in
  NumPy, because `MatMul` and the second pool are not in the verified primitive set.
  The report shows the hybrid placement at 98.67% (both-Conv calibrated) against
  98.90% float on the full 10,000-image test set, so offloading the first stage is
  worth it even when the head stays on the host.
* `examples/mel-kws/` runs a *whole* model on the NPU: its 3x32x32 mel-CNN is only
  `Conv/Relu/MaxPool` with a 1x1 head, so `open_rknpu.walk` emits every layer, no
  CPU layer is needed, and the board reaches 98.00% INT8 with 191,968/192,000
  exact bytes. The lesson is the same as this file: a model is a graph of verified
  primitives - as long as every op is in the profile, the NPU can take all of it;
  the moment an op (here `MatMul`) is out of profile, the CPU has to finish the job.

The script replays `research/mnist_pool_suite/` (the recorded board suite for this
exact prefix) and asserts its expected bytes.

Board recipe (also printed at the end of a run):

    research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \\
      --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \\
      -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime tests/board_io.c \\
      runtime/open_rknpu.c -o examples/primitives/build/11_mnist_digits/board_io
    adb shell mkdir -p /userdata/open-npu-research/11_mnist_digits
    adb push examples/primitives/build/11_mnist_digits/{board_io,*.bin,*.u8,*.i8} \\
      /userdata/open-npu-research/11_mnist_digits/
    adb shell 'cd /userdata/open-npu-research/11_mnist_digits && ./board_io . 1'
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import onnx

from common import (REPO_ROOT, compile_and_report, cross_check_suite, deterministic_cases,
                    print_board_recipe, print_summary, publish_suite, qfrom, read_expected,
                    report_checks)
from open_rknpu.quantization import reference

FOLDER = Path(__file__).resolve().parent / "build" / "11_mnist_digits"
PRETRAINED = REPO_ROOT / "research" / "pretrained" / "mnist" / "normalized.onnx"
FIXTURE_MANIFEST = REPO_ROOT / "research" / "mnist_first_suite" / "manifest.json"
FIXTURE_INPUTS = REPO_ROOT / "research" / "mnist_first_suite" / "input000.u8"
SEED = 111101


def build_prefix():
    """`Conv -> Relu -> MaxPool` cut from the pinned graph, with its output shape."""
    source = onnx.load(PRETRAINED)
    graph = source.graph
    nodes = [copy.deepcopy(node) for node in graph.node[:3]]
    if [node.op_type for node in nodes] != ["Conv", "Relu", "MaxPool"]:
        raise ValueError("the pinned MNIST graph no longer starts with Conv/Relu/MaxPool")
    output = next(value for value in graph.value_info if value.name == nodes[-1].output[0])
    prefix = onnx.helper.make_model(
        onnx.helper.make_graph(nodes, "mnist_first_stage", list(graph.input), [output],
                               list(graph.initializer)),
        opset_imports=list(source.opset_import))
    prefix.ir_version = source.ir_version
    return onnx.shape_inference.infer_shapes(prefix)


def input_band():
    """The pinned fixture's input scale/zero point, or the upstream data range."""
    if FIXTURE_MANIFEST.exists():
        entry = json.loads(FIXTURE_MANIFEST.read_text())[0]
        return float(entry["input_scale"]), int(entry["input_zero_point"])
    return 0.00390625, 128


def mnist_reference(case, model, meta):
    """Conv/Relu reference followed by the verified 2x2 block maximum."""
    grid = reference(case, qfrom(meta["quantization"]))
    height, width, channels = grid.shape
    blocks = grid.astype(np.int64).reshape(height // 2, 2, width // 2, 2, channels)
    return blocks.max(axis=(1, 3)).astype(np.int8)


def main():
    model = build_prefix()
    scale, zero_point = input_band()
    binary, meta, report = compile_and_report(model, "mnist_first_stage.onnx",
                                              label="mnist-conv1-relu-pool1",
                                              input_scale=scale, input_zero_point=zero_point)
    report["profile"] = "conv-pool-terminal[%s] (28x28x1 -> 14x14x8)" % ",".join(meta["pool_stages"])

    if FIXTURE_INPUTS.exists():
        cases = np.fromfile(FIXTURE_INPUTS, dtype=np.uint8).reshape(-1, 28, 28, 1)
    else:
        cases = deterministic_cases(np.random.default_rng(SEED), 4, 28, 28, 1)
    reference_grid = np.stack([mnist_reference(case, model, meta) for case in cases])
    publish_suite(FOLDER, binary, cases, reference_grid, meta, index=0)
    published = read_expected(FOLDER, 0).reshape(reference_grid.shape)
    row = dict(report)
    row["check"] = report_checks("mnist-first-stage", reference_grid, published, model, cases, meta,
                                 "quantization.reference + 2x2 block max")
    rows = [row]

    # Board-verified evidence for this exact prefix.
    cross_check_suite("mnist_pool_suite", mnist_reference, compile_kwargs=(scale, zero_point))

    print_board_recipe(FOLDER)
    print_summary("11_mnist_digits", rows, FOLDER)


if __name__ == "__main__":
    main()
