"""SPDX-License-Identifier: MIT

Primitive 4/11: an RGB multiplier-1 depthwise layer, with an optional pointwise head.

Verified bounds (`research/COVERAGE_EXPANSION_RESULTS.md`, "Public compiler
additions" -> Depthwise, and the 2026-09-09 "depthwise C5-C16 packing resolved"
note in `PRIMITIVE_ROADMAP.md`): a `Conv[/Relu]` RGB stem followed by a
multiplier-1 depthwise Conv with C1..16 channels, K1/K3/K5, stride 1 or 2 and
`pads = K//2`; H/W 5..8. A dense 1x1 pointwise successor is verified from
depthwise C3/C5/C9/C16 to output C1/7/16/32. Depthwise weight pair bytes carry
the negated per-channel weight zero point, and the bias/scale blocks are 24 bytes
per four channels. Input H/W stays inside the legacy image profile because the
stem is compiled by the established single-Conv emitter.

The host integer reference is `open_rknpu.depthwise.depthwise_reference` over the
stem's `open_rknpu.quantization.reference` output; a stride-2 depthwise result is
that reference sampled at `[::stride, ::stride]`, exactly as
`research/build_depthwise_combined.py` generates it. The pointwise successor uses
`open_rknpu.chain.native_reference`. This script also replays the 84 board-verified
models of `research/depthwise_combined_suite/` and asserts the same bytes.

Board recipe (also printed at the end of a run):

    research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \\
      --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \\
      -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime tests/board_io.c \\
      runtime/open_rknpu.c -o examples/primitives/build/04_depthwise/board_io
    adb shell mkdir -p /userdata/open-npu-research/04_depthwise
    adb push examples/primitives/build/04_depthwise/{board_io,*.bin,*.u8,*.i8} \\
      /userdata/open-npu-research/04_depthwise/
    adb shell 'cd /userdata/open-npu-research/04_depthwise && ./board_io . 3'
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import onnx

from common import (compile_and_report, conv_node, cross_check_suite, deterministic_cases,
                    initializer, model_graph, print_board_recipe, print_summary, publish_suite,
                    qfrom, read_expected, relu_node, report_checks, tensor_info)
from open_rknpu.chain import native_reference
from open_rknpu.depthwise import depthwise_reference
from open_rknpu.quantization import reference

FOLDER = Path(__file__).resolve().parent / "build" / "04_depthwise"
SEED = 110401

# (label, stem channels, kernel, stride, pointwise output channels, stem Relu)
CONFIGS = [
    ("rgb-k3-s1", 3, 3, 1, 0, False),
    ("c8-k3-s2-relu", 8, 3, 2, 0, True),
    ("c3-k5-s1-pointwise", 3, 5, 1, 8, False),
]


def build(index, config):
    """Stem Conv (1x1) -> depthwise Conv -> optional dense pointwise Conv."""
    _, channels, kernel, stride, pointwise, stem_relu = config
    rng = np.random.default_rng(SEED + index)
    stem_weights = rng.uniform(-0.3, 0.3, (channels, 3, 1, 1)).astype(np.float32)
    stem_bias = rng.uniform(-0.5, 0.5, channels).astype(np.float32)
    weights = rng.uniform(-0.3, 0.3, (channels, 1, kernel, kernel)).astype(np.float32)
    bias = rng.uniform(-0.5, 0.5, channels).astype(np.float32)
    extent = (8 + stride - 1) // stride
    nodes = [conv_node("input", "stem", "stem_w", "stem_b", 1)]
    source = "stem"
    if stem_relu:
        nodes.append(relu_node("stem", "stem_relu"))
        source = "stem_relu"
    constants = [initializer("stem_w", stem_weights), initializer("stem_b", stem_bias),
                 initializer("dw_w", weights), initializer("dw_b", bias)]
    nodes.append(conv_node(source, "depthwise", "dw_w", "dw_b", kernel, strides=(stride, stride),
                           group=channels))
    output_channels = channels
    if pointwise:
        head_weights = rng.uniform(-0.3, 0.3, (pointwise, channels, 1, 1)).astype(np.float32)
        head_bias = rng.uniform(-1.0, 1.0, pointwise).astype(np.float32)
        constants += [initializer("pw_w", head_weights), initializer("pw_b", head_bias)]
        nodes.append(conv_node("depthwise", "output", "pw_w", "pw_b", 1))
        output_channels = pointwise
    else:
        nodes[-1].output[0] = "output"
    return model_graph(nodes, "depthwise", [tensor_info("input", [1, 3, 8, 8])],
                       [tensor_info("output", [1, output_channels, extent, extent])], constants)


def depthwise_node(model):
    """The group>1 Conv node (depthwise), whose strides shape the reference."""
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if int(attributes.get("group", 1)) > 1:
            return attributes
    raise ValueError("no depthwise Conv in graph")


def depthwise_reference_pipeline(case, model, meta):
    """Stem band, depthwise layer, then an optional dense pointwise successor."""
    stem = qfrom(meta["first"]["quantization"])
    layer = qfrom(meta["depthwise"])
    grid = depthwise_reference(reference(case, stem), layer, stem.output_zero_point)
    if meta.get("depthwise_pointwise"):
        return native_reference(grid, qfrom(meta["pointwise"]), layer.output_zero_point)
    stride = int(depthwise_node(model).get("strides", [1, 1])[0])
    return grid[::stride, ::stride]


def main():
    rows = []
    for index, config in enumerate(CONFIGS):
        label = config[0]
        model = build(index, config)
        binary, meta, report = compile_and_report(model, "%s.onnx" % label.replace("-", "_"),
                                                  label=label)
        rng = np.random.default_rng(SEED + 100 + index)
        cases = deterministic_cases(rng, 4, 8, 8, 3)
        reference_grid = np.stack([depthwise_reference_pipeline(case, model, meta) for case in cases])
        publish_suite(FOLDER, binary, cases, reference_grid, meta, index=index)
        published = read_expected(FOLDER, index).reshape(reference_grid.shape)
        row = dict(report)
        row["check"] = report_checks(label, reference_grid, published, model, cases, meta,
                                     "depthwise.depthwise_reference (+ chain.native_reference)")
        rows.append(row)

    # Board-verified evidence: 84 models / 129,024 exact bytes in the roadmap.
    cross_check_suite("depthwise_combined_suite", depthwise_reference_pipeline)

    print_board_recipe(FOLDER)
    print_summary("04_depthwise", rows, FOLDER)


if __name__ == "__main__":
    main()
