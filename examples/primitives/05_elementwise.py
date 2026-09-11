"""SPDX-License-Identifier: MIT

Primitive 5/11: two 1x1 Conv branches combined by Add, Mul, Sub or Max.

Verified bounds (`research/COVERAGE_EXPANSION_RESULTS.md`, "Public compiler
additions" -> Mul sources / Mul quantization / Residual Add; `PRIMITIVE_ROADMAP.md`
"Initial same-shape Add and Mul", "Sub - 12 models, 384 board runs", "Max"):
both branches are `Conv[/Relu]` on the same RGB input, H/W 5..8, branch output
C2..16 (C1 is a different native-elementwise profile), 1x1 kernels and matching
spatial shapes - broadcasting is rejected. `Add`, `Sub` and `Max` re-quantize both
branches onto one shared symmetric scale and set the output scale to twice it with
zero point 0; `Mul` keeps the two branch scales and folds their product into its own
conversion. Dense output C1 is also verified, but the branch shapes here stay in the
profile this example is about.

The host integer reference is the branch Conv reference
(`open_rknpu.quantization.reference`, or `open_rknpu.chain.native_reference` for a
branch that is a Conv after another Conv) followed by the matching join:
`open_rknpu.elementwise.add_reference`, `sub_reference`, `max_reference` or
`mul_reference`. Every published `expected*.i8` is asserted byte-equal to it; the
float number is a secondary quality metric.

Board recipe (also printed at the end of a run):

    research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \\
      --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \\
      -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime tests/board_io.c \\
      runtime/open_rknpu.c -o examples/primitives/build/05_elementwise/board_io
    adb shell mkdir -p /userdata/open-npu-research/05_elementwise
    adb push examples/primitives/build/05_elementwise/{board_io,*.bin,*.u8,*.i8} \\
      /userdata/open-npu-research/05_elementwise/
    adb shell 'cd /userdata/open-npu-research/05_elementwise && ./board_io . 4'
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import (compile_and_report, conv_node, deterministic_cases, initializer, join_node,
                    model_graph, print_board_recipe, print_summary, publish_suite, qfrom,
                    read_expected, report_checks, tensor_info)
from open_rknpu.elementwise import add_reference, max_reference, mul_reference, sub_reference
from open_rknpu.quantization import reference

FOLDER = Path(__file__).resolve().parent / "build" / "05_elementwise"
SEED = 110501

# (label, join op, hidden channels, branch kernels)
CONFIGS = [
    ("add-same-shape", "Add", 4, (1, 1)),
    ("mul-independent-scales", "Mul", 4, (1, 1)),
    ("sub-signed", "Sub", 4, (1, 1)),
    ("max-signed", "Max", 4, (1, 1)),
]

JOIN_REFERENCE = {"Add": add_reference, "Sub": sub_reference, "Max": max_reference,
                  "Mul": mul_reference}


def build(index, config):
    """Shared RGB input -> two 1x1 Conv branches -> one elementwise join."""
    _, kind, channels, _ = config
    rng = np.random.default_rng(SEED + index)
    constants, nodes = [], []
    for branch in ("a", "b"):
        weights = rng.uniform(-0.3, 0.3, (channels, 3, 1, 1)).astype(np.float32)
        bias = rng.uniform(-1.0, 1.0, channels).astype(np.float32)
        constants += [initializer("w_%s" % branch, weights), initializer("b_%s" % branch, bias)]
        nodes.append(conv_node("input", "branch_%s" % branch, "w_%s" % branch, "b_%s" % branch, 1))
    nodes.append(join_node(kind, "branch_a", "branch_b", "output"))
    return model_graph(nodes, "elementwise", [tensor_info("input", [1, 3, 8, 8])],
                       [tensor_info("output", [1, channels, 8, 8])], constants)


def elementwise_reference(case, model, meta, kind):
    """Both branch references, then the profile's join reference."""
    branches = [reference(case, qfrom(entry["quantization"])) for entry in meta["branches"]]
    return JOIN_REFERENCE[kind](*branches)


def main():
    rows = []
    for index, config in enumerate(CONFIGS):
        label, kind = config[0], config[1]
        model = build(index, config)
        binary, meta, report = compile_and_report(model, "%s.onnx" % label.replace("-", "_"),
                                                  label=label)
        rng = np.random.default_rng(SEED + 100 + index)
        cases = deterministic_cases(rng, 4, 8, 8, 3)
        reference_grid = np.stack([elementwise_reference(case, model, meta, kind) for case in cases])
        publish_suite(FOLDER, binary, cases, reference_grid, meta, index=index)
        published = read_expected(FOLDER, index).reshape(reference_grid.shape)
        row = dict(report)
        row["check"] = report_checks(label, reference_grid, published, model, cases, meta,
                                     "quantization.reference + %s_reference" % kind.lower())
        rows.append(row)
    print_board_recipe(FOLDER)
    print_summary("05_elementwise", rows, FOLDER)


if __name__ == "__main__":
    main()
