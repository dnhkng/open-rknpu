"""SPDX-License-Identifier: MIT

Primitive 6/11: a tensor multiplied by an immutable constant (broadcast).

Verified bounds (`research/COVERAGE_EXPANSION_RESULTS.md`, "Public compiler
additions" -> Mul broadcast): a standalone `Mul` with one external RGB input
(H/W 5..8, batch 1..16) and one immutable float32 constant that NumPy can
broadcast to the input shape - scalar, per-channel `[3,1,1]`, spatial `[1,1,H,W]`
or full `[1,3,H,W]`. `src/open_rknpu/elementwise.py:compile_constant_mul` uses the
verified elementwise Mul task: the constant is materialized as the ERDMA operand
(one 16-byte vector for per-channel, a native plane for per-pixel), the output
scale folds the branch scale and the constant scale, and the constant can be
exposed as the mutable v4 `mul.factor` region. The roadmap also notes that the
ERDMA reads its secondary operand strictly linearly, so there is no compact
spatial mode - a spatial constant is materialized.

Two profiles appear here:

* the constant-Mul profile (scalar / per-channel / spatial / full) whose reference
  is `open_rknpu.quantization.reference` for the input conversion followed by
  `open_rknpu.elementwise.mul_reference` against the packed constant codes at
  `meta["constant_scale"]`;
* a `Conv` followed by `Mul(scalar or per-channel constant)`. `normalize_model`
  folds that constant into the Conv weights and bias, so the container is the
  ordinary Conv profile and its reference is `open_rknpu.quantization.reference`.
  The fold applies only to scalar, `[1]`, `[C,1,1]` and `[1,C,1,1]` constants: a
  spatial broadcast after a Conv is *not* folded and the elementwise profile wants
  both operands to be Conv branches, so that combination is rejected rather than
  silently mishandled.

Board recipe (also printed at the end of a run):

    research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \\
      --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \\
      -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime tests/board_io.c \\
      runtime/open_rknpu.c -o examples/primitives/build/06_constant_mul/board_io
    adb shell mkdir -p /userdata/open-npu-research/06_constant_mul
    adb push examples/primitives/build/06_constant_mul/{board_io,*.bin,*.u8,*.i8} \\
      /userdata/open-npu-research/06_constant_mul/
    adb shell 'cd /userdata/open-npu-research/06_constant_mul && ./board_io . 5'
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import (compile_and_report, conv_node, deterministic_cases, initializer, join_node,
                    model_graph, print_board_recipe, print_summary, publish_suite, qfrom,
                    read_expected, report_checks, tensor_info)
from open_rknpu.elementwise import mul_reference
from open_rknpu.quantization import reference

FOLDER = Path(__file__).resolve().parent / "build" / "06_constant_mul"
SEED = 110601

# (label, constant, kind) - kind "constant" is the Mul profile, "fold" is Conv then Mul.
CONFIGS = [
    ("scalar", np.float32(0.7), "constant"),
    ("per-channel", np.array([0.2, -0.7, 1.3], np.float32).reshape(3, 1, 1), "constant"),
    ("spatial", None, "constant"),
    ("full", None, "constant"),
    ("conv-then-mul-fold", np.array([0.5, 1.5, -0.75], np.float32).reshape(3, 1, 1), "fold"),
]


def make_factor(index, config):
    """The immutable constant for config `index` (spatial/full need an RNG)."""
    _, factor, kind = config
    if factor is not None:
        return np.asarray(factor, np.float32)
    rng = np.random.default_rng(SEED + index)
    if config[0] == "spatial":
        return rng.uniform(0.1, 1.5, (1, 1, 8, 8)).astype(np.float32)
    return rng.uniform(0.1, 1.5, (1, 3, 8, 8)).astype(np.float32)


def build(index, config):
    """`Mul(input, const)` or `Conv(input) -> Mul(const)`."""
    _, _, kind = config
    factor = make_factor(index, config)
    constants = [initializer("factor", factor)]
    if kind == "fold":
        rng = np.random.default_rng(SEED + 50 + index)
        weights = rng.uniform(-0.3, 0.3, (3, 3, 1, 1)).astype(np.float32)
        bias = rng.uniform(-1.0, 1.0, 3).astype(np.float32)
        constants = [initializer("w", weights), initializer("b", bias)] + constants
        nodes = [conv_node("input", "conv", "w", "b", 1),
                 join_node("Mul", "conv", "factor", "output")]
    else:
        nodes = [join_node("Mul", "input", "factor", "output")]
    return model_graph(nodes, "constant_mul", [tensor_info("input", [1, 3, 8, 8])],
                       [tensor_info("output", [1, 3, 8, 8])], constants), factor, kind


def constant_reference(case, model, meta, factor):
    """Input conversion reference, packed constant codes, then `mul_reference`."""
    converted = reference(case, qfrom(meta["branches"][0]["quantization"]))
    expanded = np.broadcast_to(np.asarray(factor, np.float64), (1, 3, 8, 8))
    codes = np.clip(np.rint(expanded / meta["constant_scale"]), -128, 127).astype(np.int8)
    return mul_reference(converted, codes[0].transpose(1, 2, 0))


def main():
    rows = []
    for index, config in enumerate(CONFIGS):
        label, _, kind = config
        model, factor, kind = build(index, config)
        binary, meta, report = compile_and_report(model, "%s.onnx" % label.replace("-", "_"),
                                                  label=label)
        rng = np.random.default_rng(SEED + 100 + index)
        cases = deterministic_cases(rng, 4, 8, 8, 3)
        if kind == "fold":
            reference_name = "quantization.reference (constant folded into Conv weights)"
            reference_grid = np.stack([reference(case, qfrom(meta["quantization"])) for case in cases])
        else:
            reference_name = "quantization.reference + elementwise.mul_reference"
            reference_grid = np.stack([constant_reference(case, model, meta, factor) for case in cases])
        publish_suite(FOLDER, binary, cases, reference_grid, meta, index=index)
        published = read_expected(FOLDER, index).reshape(reference_grid.shape)
        row = dict(report)
        row["check"] = report_checks(label, reference_grid, published, model, cases, meta,
                                     reference_name)
        rows.append(row)
    print_board_recipe(FOLDER)
    print_summary("06_constant_mul", rows, FOLDER)


if __name__ == "__main__":
    main()
