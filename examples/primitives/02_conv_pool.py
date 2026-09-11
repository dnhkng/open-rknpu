"""SPDX-License-Identifier: MIT

Primitive 2/11: `Conv[/Relu]` followed by two or three 2x2 stride-2 pools.

Verified bounds (`research/COVERAGE_EXPANSION_RESULTS.md`, "Public compiler
additions" -> Dense Conv and the pooling profile in
`src/open_rknpu/scheduler.py`): the first Conv is the legacy 8x8 image layer
(RGB C1..16, K1/3/5, symmetric pad, stride 1), and every pool is `MaxPool` or
`AveragePool` with `kernel_shape=[2,2]`, `strides=[2,2]`, zero padding and
`ceil_mode=0` - the only pool geometry the register profile verifies
(`src/open_rknpu/pooling.py`). The walk halves the row/column extent once per
pool.

Which emitter handles the graph is worth stating exactly, because it is not
obvious from the script name: the *terminal*-pool emitter
(`_compile_sequence`'s pooling branch) only fires when the last node is the pool,
so a single terminal pool compiles to the plain pooling container with
`meta["pool_stages"]`. As soon as there are two or more pools the first one is
interior, and the scheduler hands the graph to the op-level chain walk
(`open_rknpu.walk.compile_chain_walk`, profile `chain-walk`), which lowers each
Conv and each pool op by op and is itself board-verified
(`research/walk_chain_suite/`, 12 models). Both paths appear here and both are
checked against their documented host integer reference.

Verification criterion: each published `expected*.i8` is asserted byte-equal to
the profile's integer reference (the same pipeline the suite generators use); the
float number is a secondary quantization-quality metric and is never called exact.

Board recipe (also printed at the end of a run):

    research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \\
      --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \\
      -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime tests/board_io.c \\
      runtime/open_rknpu.c -o examples/primitives/build/02_conv_pool/board_io
    adb shell mkdir -p /userdata/open-npu-research/02_conv_pool
    adb push examples/primitives/build/02_conv_pool/{board_io,*.bin,*.u8,*.i8} \\
      /userdata/open-npu-research/02_conv_pool/
    adb shell 'cd /userdata/open-npu-research/02_conv_pool && ./board_io . 3'
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import (compile_and_report, conv_node, deterministic_cases, initializer, model_graph,
                    pool_node, print_board_recipe, print_summary, publish_suite, qfrom,
                    read_expected, relu_node, report_checks, tensor_info)
from open_rknpu.quantization import reference
from open_rknpu.walk import chain_walk_reference, load_quantizations, parse_chain

FOLDER = Path(__file__).resolve().parent / "build" / "02_conv_pool"
SEED = 110201

# (label, hidden channels, kernel, pools, input scale, input zero point)
CONFIGS = [
    ("one-terminal-maxpool", 8, 3, ["MaxPool"], 0.5, 128),
    ("two-maxpools", 8, 3, ["MaxPool", "MaxPool"], 1.0, 0),
    ("three-mixed-pools", 8, 3, ["MaxPool", "AveragePool", "MaxPool"], 1.0, 0),
]


def build(index, config):
    """RGB 8x8 -> Conv -> Relu -> pools, with the pooled output shape declared."""
    _, hidden, kernel, pools, _, _ = config
    rng = np.random.default_rng(SEED + index)
    weights = rng.uniform(-0.3, 0.3, (hidden, 3, kernel, kernel)).astype(np.float32)
    bias = rng.uniform(-1.0, 1.0, hidden).astype(np.float32)
    nodes = [conv_node("input", "conv", "w", "b", kernel),
             relu_node("conv", "relu")]
    source, extent = "relu", 8
    for position, kind in enumerate(pools):
        output = "pool%d" % position
        nodes.append(pool_node(kind, source, output))
        source, extent = output, extent // 2
    return model_graph(nodes, "conv_pool", [tensor_info("input", [1, 3, 8, 8])],
                       [tensor_info(source, [1, hidden, extent, extent])],
                       [initializer("w", weights), initializer("b", bias)])


def block_pool(grid, kind):
    """The verified pool semantics: 2x2 block max, or the rounded 2x2 block mean."""
    height, width, channels = grid.shape
    blocks = grid.astype(np.int64).reshape(height // 2, 2, width // 2, 2, channels)
    if kind == "MaxPool":
        pooled = blocks.max(axis=(1, 3))
    else:
        pooled = np.rint(blocks.sum(axis=(1, 3)) / 4)
    return np.clip(pooled, -128, 127).astype(np.int8)


def expected_terminal(case, meta):
    """Pooling profile: one Conv band followed by the recorded pool stages."""
    grid = reference(case, qfrom(meta["quantization"]))
    for kind in meta["pool_stages"]:
        grid = block_pool(grid, kind)
    return grid


def expected_walk(case, meta, model):
    """Chain walk: `chain_walk_reference` replays the parsed op list."""
    spec = parse_chain(model.graph)
    return chain_walk_reference(case, load_quantizations(meta), spec["ops"])


def main():
    rows = []
    for index, config in enumerate(CONFIGS):
        label, hidden, _, pools, input_scale, input_zero_point = config
        model = build(index, config)
        binary, meta, report = compile_and_report(model, "%s.onnx" % label.replace("-", "_"),
                                                  label=label, input_scale=input_scale,
                                                  input_zero_point=input_zero_point)
        if meta.get("pool_stages"):
            expected_for = expected_terminal
            reference_name = "quantization.reference + 2x2 pool stages"
            report["profile"] = "conv-pool-terminal[%s]" % ",".join(meta["pool_stages"])
        else:
            expected_for = lambda case, meta, model=model: expected_walk(case, meta, model)
            reference_name = "walk.chain_walk_reference"
        rng = np.random.default_rng(SEED + 100 + index)
        cases = deterministic_cases(rng, 4, 8, 8, 3)
        reference = np.stack([expected_for(case, meta) for case in cases])
        publish_suite(FOLDER, binary, cases, reference, meta, index=index)
        published = read_expected(FOLDER, index).reshape(reference.shape)
        row = dict(report)
        row["check"] = report_checks(label, reference, published, model, cases, meta, reference_name)
        rows.append(row)
    print_board_recipe(FOLDER)
    print_summary("02_conv_pool", rows, FOLDER)


if __name__ == "__main__":
    main()
