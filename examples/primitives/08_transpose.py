"""SPDX-License-Identifier: MIT

Primitive 8/11: a dense `ConvTranspose` layer after a 1x1 Conv stem.

Verified bounds (`research/COVERAGE_EXPANSION_RESULTS.md`, "Public compiler
additions" -> Transposed Conv): a `Conv[/Relu]` stem at 8x8 with C1..16 output
channels followed by a dense `ConvTranspose` with input/output C1..16, square
K3/K5, per-axis stride 1..2, asymmetric explicit padding below the kernel, legal
per-axis `output_padding`, `auto_pad` NOTSET/VALID/SAME_UPPER/SAME_LOWER and
`output_shape` modes. Grouped weights expand to zero-filled dense weights, and
`K3 dilation2` zero-stuffs into a sparse K5 (`transposed_rewrite`), which is a
direct emitter form here. Per-channel weight zero points are asymmetric, the tap
table flips both axes, and zero insertion needs bias compensation for all taps.

This script exercises the dense path only. The multiplier-1 *depthwise*
ConvTranspose shares the depthwise tap lanes and is board-verified in
`research/transpose_k5_suite/` and `research/transpose_k5_dilation_suite/`, but
its stage-2 reference is not reachable from the current graph in the same way
for every geometry, so it is not claimed here.

`open_rknpu.transposed` exposes no reference function, so the integer reference
below is the explicit scatter-and-requantize loop the suite generators use
(`research/build_transpose_dense.py`, `research/build_transpose_k5_dilation_suite.py`):
center the stem output, scatter each input position over
`iy*stride + ky - pad`, fold the stem zero point into the bias, then apply the
per-channel and global conversions. The script replays both board-verified dense
suites and asserts their recorded bytes.

Verification criterion: each published `expected*.i8` is asserted byte-equal to
that integer scatter reference; the float number is a secondary
quantization-quality metric and is never called exact.

Board recipe (also printed at the end of a run):

    research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \\
      --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \\
      -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime tests/board_io.c \\
      runtime/open_rknpu.c -o examples/primitives/build/08_transpose/board_io
    adb shell mkdir -p /userdata/open-npu-research/08_transpose
    adb push examples/primitives/build/08_transpose/{board_io,*.bin,*.u8,*.i8} \\
      /userdata/open-npu-research/08_transpose/
    adb shell 'cd /userdata/open-npu-research/08_transpose && ./board_io . 4'
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import onnx

from common import (compile_and_report, conv_node, cross_check_suite, deterministic_cases,
                    initializer, model_graph, print_board_recipe, print_summary, publish_suite,
                    qfrom, read_expected, report_checks, tensor_info)
from open_rknpu.quantization import reference

FOLDER = Path(__file__).resolve().parent / "build" / "08_transpose"
SEED = 110801

# (label, kernel, dilations, strides, pads, output_padding)
CONFIGS = [
    ("dense-k3-s2", 3, [1, 1], [2, 2], [1, 1, 1, 1], [0, 0]),
    ("dense-k3-s2-asym", 3, [1, 1], [2, 2], [0, 2, 1, 0], [1, 0]),
    ("dense-k3-dil2", 3, [2, 2], [1, 1], [1, 1, 1, 1], [0, 0]),
    ("dense-k5-s1", 5, [1, 1], [1, 1], [2, 2, 2, 2], [0, 0]),
]


def build(index, config):
    """RGB 8x8 -> 1x1 Conv stem (3 channels) -> dense ConvTranspose to C8."""
    _, kernel, dilations, strides, pads, output_padding = config
    rng = np.random.default_rng(SEED + index)
    stem_weights = rng.uniform(-0.3, 0.3, (3, 3, 1, 1)).astype(np.float32)
    stem_bias = rng.uniform(-0.5, 0.5, 3).astype(np.float32)
    weights = rng.uniform(-0.3, 0.3, (3, 8, kernel, kernel)).astype(np.float32)
    bias = rng.uniform(-0.5, 0.5, 8).astype(np.float32)
    effective = (kernel - 1) * dilations[0] + 1
    out_h = 7 * strides[0] - pads[0] - pads[2] + effective + output_padding[0]
    out_w = 7 * strides[1] - pads[1] - pads[3] + effective + output_padding[1]
    node = onnx.helper.make_node("ConvTranspose", ["stem", "w", "b"], ["output"], group=1,
                                 kernel_shape=[kernel, kernel], dilations=dilations,
                                 strides=strides, pads=pads, output_padding=output_padding)
    return model_graph([conv_node("input", "stem", "stem_w", "stem_b", 1), node], "transpose",
                       [tensor_info("input", [1, 3, 8, 8])],
                       [tensor_info("output", [1, 8, out_h, out_w])],
                       [initializer("stem_w", stem_weights), initializer("stem_b", stem_bias),
                        initializer("w", weights), initializer("b", bias)])


def transposed_geometry(model, meta):
    """Effective kernel, strides, resolved pads and output padding for the node."""
    node = next(node for node in model.graph.node if node.op_type == "ConvTranspose")
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    kernel = int(meta["transposed_quantization"]["kernel_size"])
    strides = [int(v) for v in attrs.get("strides", [1, 1])]
    output_padding = [int(v) for v in attrs.get("output_padding", [0, 0])]
    pads = [int(v) for v in attrs.get("pads", [0, 0, 0, 0])]
    auto = attrs.get("auto_pad", b"NOTSET")
    if isinstance(auto, bytes):
        auto = auto.decode()
    if auto in ("SAME_UPPER", "SAME_LOWER"):
        totals = [kernel + output_padding[i] - strides[i] for i in range(2)]
        begins = [(v + (auto == "SAME_LOWER")) // 2 for v in totals]
        pads = [begins[0], begins[1], totals[0] - begins[0], totals[1] - begins[1]]
    elif auto == "VALID":
        pads = [0, 0, 0, 0]
    return kernel, strides, pads, output_padding


def transposed_reference(case, model, meta):
    """Explicit integer scatter reference, shared with the dense suite generators."""
    kernel, strides, pads, output_padding = transposed_geometry(model, meta)
    stem = qfrom(meta["first"]["quantization"])
    layer = qfrom(meta["transposed_quantization"])
    channels = len(layer.weight_zero_points)
    weights = np.asarray(layer.weights).reshape(channels, layer.weights.size // (channels * kernel * kernel),
                                                kernel, kernel)
    weights = weights - np.asarray(layer.weight_zero_points, np.int64)[:, None, None, None]
    height = width = 8
    out_h = (height - 1) * strides[0] - pads[0] - pads[2] + kernel + output_padding[0]
    out_w = (width - 1) * strides[1] - pads[1] - pads[3] + kernel + output_padding[1]
    bias_units = np.asarray(layer.biases, np.int64) + stem.output_zero_point * weights.reshape(channels, -1).sum(axis=1)
    centered = reference(case, stem).astype(np.int64) - stem.output_zero_point
    accumulator = np.broadcast_to(bias_units, (out_h, out_w, channels)).astype(np.int64).copy()
    for iy in range(height):
        for ix in range(width):
            for ky in range(kernel):
                for kx in range(kernel):
                    oy = iy * strides[0] + ky - pads[0]
                    ox = ix * strides[1] + kx - pads[1]
                    if 0 <= oy < out_h and 0 <= ox < out_w:
                        accumulator[oy, ox] += centered[iy, ix] @ weights[:, :, ky, kx].T
    product = accumulator * layer.channel_multipliers
    scaled = (product + 8191 + ((product >> 14) & 1)) >> 14
    product = scaled * layer.multiplier
    if layer.shift:
        product = product + (layer.output_zero_point << layer.shift)
        result = (product + (1 << (layer.shift - 1)) - 1 + ((product >> layer.shift) & 1)) >> layer.shift
    else:
        result = product + layer.output_zero_point
    return np.clip(result, -128, 127).astype(np.int8)


def main():
    rows = []
    for index, config in enumerate(CONFIGS):
        label = config[0]
        model = build(index, config)
        binary, meta, report = compile_and_report(model, "%s.onnx" % label.replace("-", "_"),
                                                  label=label)
        if meta.get("transposed_rewrite"):
            report["profile"] = "%s [%s]" % (report["profile"], meta["transposed_rewrite"])
        rng = np.random.default_rng(SEED + 100 + index)
        cases = deterministic_cases(rng, 4, 8, 8, 3)
        reference_grid = np.stack([transposed_reference(case, model, meta) for case in cases])
        publish_suite(FOLDER, binary, cases, reference_grid, meta, index=index)
        published = read_expected(FOLDER, index).reshape(reference_grid.shape)
        row = dict(report)
        row["check"] = report_checks(label, reference_grid, published, model, cases, meta,
                                     "explicit transposed scatter (suite reference)")
        rows.append(row)

    # Board-verified evidence: 8 dense + 4 sparse-K5 dense models in research/.
    cross_check_suite("transpose_dense_suite", transposed_reference)
    cross_check_suite("transpose_dilation_dense_suite", transposed_reference)

    print_board_recipe(FOLDER)
    print_summary("08_transpose", rows, FOLDER)


if __name__ == "__main__":
    main()
