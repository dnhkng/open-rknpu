"""SPDX-License-Identifier: MIT

Primitive 7/11: a bounded Conv -> Sigmoid/Tanh lookup-table activation.

Verified bounds (`research/COVERAGE_EXPANSION_RESULTS.md`, "Public compiler
additions" -> LUT activation; `PRIMITIVE_ROADMAP.md`, "LUT activations
(Sigmoid/Tanh first)"): the public profile is a diagonal C3 1x1 `Conv` stem at
8x8 followed by `Sigmoid` or `Tanh`, with one scalar gain magnitude per channel
(per-channel signs and any bias allowed), input scale 1 and input zero point 128.
The stem's analytic output interval must stay inside the sign-split table domain:
the positive half spans `(-8, 8)` in steps of 1/64, and the negative half advances
by the whole-bit gain `2**ceil(log2(BASE_WEIGHT_SCALE/weight_scale))`, so
`BASE_WEIGHT_SCALE/weight_scale` must be a power of two. Mixed input weights,
table interpolation/domain calibration and other shapes/channels stay blocked
(`research/lut_domain_failed/`, `research/lut_mixed_gain_probe/`).

The host integer reference is `open_rknpu.lut.lut_reference`, which reproduces the
quantized bytes including the 1/64 index grid and the Q15 table rounding. The
reference is exact by construction for this profile; the float comparison is a
secondary quality metric. The script also replays the 12 board-verified models of
`research/lut_domain_suite/`.

Board recipe (also printed at the end of a run):

    research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \\
      --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \\
      -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime tests/board_io.c \\
      runtime/open_rknpu.c -o examples/primitives/build/07_lut_activation/board_io
    adb shell mkdir -p /userdata/open-npu-research/07_lut_activation
    adb push examples/primitives/build/07_lut_activation/{board_io,*.bin,*.u8,*.i8} \\
      /userdata/open-npu-research/07_lut_activation/
    adb shell 'cd /userdata/open-npu-research/07_lut_activation && ./board_io . 3'
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import onnx

from common import (compile_and_report, cross_check_suite, deterministic_cases,
                    initializer, model_graph, print_board_recipe, print_summary, publish_suite,
                    read_expected, report_checks, tensor_info)
from open_rknpu.lut import lut_reference

FOLDER = Path(__file__).resolve().parent / "build" / "07_lut_activation"
SEED = 110701

# (label, activation, per-channel weights, bias)
CONFIGS = [
    ("sigmoid-identity", "Sigmoid", np.full(3, 1 / 32, np.float32), np.zeros(3, np.float32)),
    ("tanh-identity", "Tanh", np.full(3, 1 / 32, np.float32), np.zeros(3, np.float32)),
    ("sigmoid-signed", "Sigmoid", np.array([-1 / 32, 1 / 32, -1 / 32], np.float32),
     np.array([0.5, 0.5, -0.5], np.float32)),
]


def build(index, config):
    """Diagonal C3 1x1 Conv stem followed by the LUT activation."""
    _, kind, gains, bias = config
    weights = np.diag(np.asarray(gains, np.float32)).reshape(3, 3, 1, 1)
    # The LUT profile accepts only a bare `kernel_shape=[1,1]` attribute set, so the
    # Conv node is built directly instead of through the shared helper.
    nodes = [onnx.helper.make_node("Conv", ["input", "w", "b"], ["stem"], kernel_shape=[1, 1]),
             onnx.helper.make_node(kind, ["stem"], ["output"])]
    return model_graph(nodes, "lut_activation", [tensor_info("input", [1, 3, 8, 8])],
                       [tensor_info("output", [1, 3, 8, 8])],
                       [initializer("w", weights), initializer("b", bias)])


def lut_reference_pipeline(case, model, meta):
    """`lut_reference` from the stem the emitter recorded in its metadata."""
    weights = np.asarray(meta["stem_weights"], np.float32).reshape(3, 3)
    bias = np.asarray(meta["stem_bias"], np.float32)
    return lut_reference(case, weights, bias, meta["kind"])


def main():
    rows = []
    for index, config in enumerate(CONFIGS):
        label = config[0]
        model = build(index, config)
        binary, meta, report = compile_and_report(model, "%s.onnx" % label.replace("-", "_"),
                                                  label=label, input_scale=1, input_zero_point=128)
        rng = np.random.default_rng(SEED + 100 + index)
        cases = deterministic_cases(rng, 4, 8, 8, 3)
        cases.reshape(-1)[3:3 + 256] = np.arange(256, dtype=np.uint8)
        reference_grid = np.stack([lut_reference_pipeline(case, model, meta) for case in cases])
        publish_suite(FOLDER, binary, cases, reference_grid, meta, index=index)
        published = read_expected(FOLDER, index).reshape(reference_grid.shape)
        row = dict(report)
        row["profile"] = "%s domain=[%.2f, %.2f]" % (report["profile"], meta["stem_range"][0],
                                                     meta["stem_range"][1])
        row["check"] = report_checks(label, reference_grid, published, model, cases, meta,
                                     "lut.lut_reference", input_scale=1, input_zero_point=128)
        rows.append(row)

    # Board-verified evidence: 12 models / 192 inferences in research/lut_domain_suite.
    cross_check_suite("lut_domain_suite", lut_reference_pipeline, compile_kwargs=(1, 128))

    print_board_recipe(FOLDER)
    print_summary("07_lut_activation", rows, FOLDER)


if __name__ == "__main__":
    main()
