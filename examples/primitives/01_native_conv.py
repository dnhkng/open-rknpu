"""SPDX-License-Identifier: MIT

Primitive 1/11: one dense `Conv[/Relu]` through the native16-input profile.

Verified bounds (`research/COVERAGE_EXPANSION_RESULTS.md`, "Public compiler
additions" -> Dense Conv; `PRIMITIVE_ROADMAP.md`, "Dense Conv wider shapes"):
static batch 1..16, input and output channels C1..128, H/W 1..128, odd square
kernels K1..31, strides 1..4, explicit padding 0..K-1 on each side (asymmetric
allowed; `pads` is `[top,left,bottom,right]`), native dilations 1..17, and an
optional fused Relu or Clip[0,6]. Even/rectangular kernels and grouped/dilated
kernels outside this list are lowered to dense weights by `normalize_model`
before they reach this profile. Inputs above 6144 pixel-planes are split into
aligned serial height tasks with overlapping halos; K33 and dilation 18+ are
rejected.

The scheduler picks `native16-input` when the graph has a single input and
output and does not match the legacy 8x8 RGB profile - here that is forced by
H/W outside 5..8, input channels other than 1/3, more than 16 output channels,
or a kernel of 7 or more. The host integer reference is
`open_rknpu.native.native_input_reference`, which also reproduces the
asymmetric padding, stride and dilation fields. Every example is checked against
ONNX's float evaluator, and the suite is published for `tests/board_io.c`.

Verification criterion: each published `expected*.i8` is asserted byte-equal to
`native_input_reference`, which is the project's correctness criterion; the float
number printed next to it is a secondary quantization-quality metric and is never
called exact.

Board recipe (also printed at the end of a run) - cross-compile the v5 runner
and stage the suite:

    research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \\
      --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \\
      -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime tests/board_io.c \\
      runtime/open_rknpu.c -o examples/primitives/build/01_native_conv/board_io
    adb shell mkdir -p /userdata/open-npu-research/01_native_conv
    adb push examples/primitives/build/01_native_conv/{board_io,model000.bin,input000.u8,expected000.i8} \\
      /userdata/open-npu-research/01_native_conv/
    adb shell 'cd /userdata/open-npu-research/01_native_conv && ./board_io . 6'
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import (compile_and_report, conv_node, deterministic_cases, initializer, model_graph,
                    print_board_recipe, print_summary, publish_suite, qfrom, read_expected,
                    relu_node, report_checks, tensor_info)
from open_rknpu.native import native_input_reference

FOLDER = Path(__file__).resolve().parent / "build" / "01_native_conv"
SEED = 110101

# (label, input C, output C, H, W, kernel, pads, strides, dilations, Relu, input scale, zp)
CONFIGS = [
    ("rgb-k3-s1-relu", 3, 8, 16, 16, 3, [1, 1, 1, 1], [1, 1], [1, 1], True, 0.5, 128),
    ("c17-k3-s2-asym", 17, 32, 12, 12, 3, [0, 1, 2, 1], [2, 2], [1, 1], False, 1.0, 0),
    ("c32-k1-s1", 32, 64, 10, 10, 1, [0, 0, 0, 0], [1, 1], [1, 1], False, 0.25, 64),
    ("c3-k5-dil2-relu", 3, 16, 12, 12, 5, [4, 4, 4, 4], [1, 1], [2, 2], True, 1.0, 0),
    ("c1-to-c128-s4", 1, 128, 16, 16, 3, [1, 1, 1, 1], [4, 4], [1, 1], True, 1.0, 0),
    ("c128-to-c3-s3", 128, 3, 16, 16, 3, [1, 1, 1, 1], [3, 3], [1, 1], False, 0.75, 32),
]


def build(index, config):
    """One Conv[/Relu] graph whose output shape follows the explicit padding."""
    (_, in_channels, out_channels, height, width, kernel, pads, strides, dilations,
     relu, _, _) = config
    rng = np.random.default_rng(SEED + index)
    weights = rng.uniform(-0.25, 0.25, (out_channels, in_channels, kernel, kernel)).astype(np.float32)
    bias = rng.uniform(-0.5, 0.5, out_channels).astype(np.float32)
    effective_h = (kernel - 1) * dilations[0] + 1
    effective_w = (kernel - 1) * dilations[1] + 1
    out_h = (height + pads[0] + pads[2] - effective_h) // strides[0] + 1
    out_w = (width + pads[1] + pads[3] - effective_w) // strides[1] + 1
    nodes = [conv_node("input", "conv" if relu else "output", "w", "b", kernel,
                       pads=pads, strides=strides, dilations=dilations)]
    if relu:
        nodes.append(relu_node("conv", "output"))
    return model_graph(nodes, "native_conv", [tensor_info("input", [1, in_channels, height, width])],
                       [tensor_info("output", [1, out_channels, out_h, out_w])],
                       [initializer("w", weights), initializer("b", bias)])


def expected_int8(case, meta):
    """The profile's host integer reference, including padding/stride/dilation."""
    return native_input_reference(case, qfrom(meta["quantization"]), int(meta["input_zero_point"]),
                                  pads=meta["conv_pads"], strides=tuple(meta["conv_strides"]),
                                  dilations=tuple(meta["conv_dilations"]))


def main():
    rows = []
    for index, config in enumerate(CONFIGS):
        label, in_channels, _, height, width, _, _, _, _, _, input_scale, input_zero_point = config
        model = build(index, config)
        binary, meta, report = compile_and_report(model, "%s.onnx" % label.replace("-", "_"),
                                                  label=label, input_scale=input_scale,
                                                  input_zero_point=input_zero_point)
        rng = np.random.default_rng(SEED + 100 + index)
        cases = deterministic_cases(rng, 4, height, width, in_channels)
        reference = np.stack([expected_int8(case, meta) for case in cases])
        publish_suite(FOLDER, binary, cases, reference, meta, index=index)
        published = read_expected(FOLDER, index).reshape(reference.shape)
        row = dict(report)
        row["check"] = report_checks(label, reference, published, model, cases, meta,
                                     "native.native_input_reference")
        rows.append(row)
    print_board_recipe(FOLDER)
    print_summary("01_native_conv", rows, FOLDER)


if __name__ == "__main__":
    main()
