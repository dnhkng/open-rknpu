"""SPDX-License-Identifier: MIT

Primitive 3/11: an N-layer `Conv/Relu` chain and a chain with an interior pool.

Two verified emitters are shown:

* `native-chain` (`open_rknpu.chain_n`, `[Conv, Relu]*(N-1) + [Conv]`, fixed
  `[1,3,8,8]`, hidden C3..16, K1/K3 with symmetric padding, N >= 2). Bounds from
  `research/COVERAGE_EXPANSION_RESULTS.md` ("Independently generated N-layer
  native chains ... 5 models, 80 board inferences") and `PRIMITIVE_ROADMAP.md`:
  the first layer is the established legacy program, every later layer is a
  native16 program, and all intermediates share one arena with explicit
  lifetimes (`--reuse-intermediates` halves live buffers at 4+ layers). The
  expected bytes come from `open_rknpu.chain_n.chain_n_reference`.
* `chain-walk` (`open_rknpu.walk.compile_chain_walk`) for a chain whose pool is
  *not* the last node - the class no profile above matches. Bounds: RGB input,
  Conv with 1..16 output channels and K1/3/5, 2x2 stride-2 MaxPool/AveragePool,
  any H/W the walk can halve (here 8x8 -> 4x4). Expected bytes come from
  `open_rknpu.walk.chain_walk_reference` over `parse_chain`. The pool preserves
  the band: a 2x2 block maximum, or the rounded 2x2 block mean.

Verification criterion: each published `expected*.i8` is asserted byte-equal to
the integer reference above, and the script replays both board-verified suites
(`research/native_chain_suite/`, `research/walk_chain_suite/`) with the same
pipeline. The float number is a secondary quantization-quality metric and is
never called exact.

Board recipe (also printed at the end of a run):

    research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \\
      --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \\
      -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime tests/board_io.c \\
      runtime/open_rknpu.c -o examples/primitives/build/03_conv_chain/board_io
    adb shell mkdir -p /userdata/open-npu-research/03_conv_chain
    adb push examples/primitives/build/03_conv_chain/{board_io,*.bin,*.u8,*.i8} \\
      /userdata/open-npu-research/03_conv_chain/
    adb shell 'cd /userdata/open-npu-research/03_conv_chain && ./board_io . 3'
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import (compile_and_report, conv_node, cross_check_suite, deterministic_cases,
                    initializer, model_graph, pool_node, print_board_recipe, print_summary,
                    publish_suite, read_expected, relu_node, report_checks, tensor_info)
from open_rknpu.chain_n import chain_n_reference
from open_rknpu.walk import chain_walk_reference, load_quantizations, parse_chain

FOLDER = Path(__file__).resolve().parent / "build" / "03_conv_chain"
SEED = 110301


def chain_reference(case, model, meta):
    """The `native-chain` reference over the recorded per-layer quantizations."""
    return chain_n_reference(case, meta["quantizations"])


def walk_reference(case, model, meta):
    """The `chain-walk` reference over the parsed op list for this model."""
    return chain_walk_reference(case, load_quantizations(meta), parse_chain(model.graph)["ops"])


def chain_graph(hidden, kernels):
    """`[Conv, Relu]*(N-1) + [Conv]` with `hidden` channels on every layer.

    Hidden weights and biases are non-negative on purpose. The analytic band of a
    hidden layer then has a non-negative lower bound, so `native_quantize` picks
    zero point -128 - the value the chain emitter's default input-zero-point
    register carries - and the quantized chain tracks the float model to about one
    LSB. Mixed-sign hidden weights can still compile, but they leave the
    intermediate zero point away from -128 and the analytic bands then cost real
    accuracy (see `research/native_chain_suite/` models 1 and 3); that is a
    quantization-quality property, not a reference mismatch, and the integer
    reference stays exact either way.
    """
    rng = np.random.default_rng(SEED + len(kernels))
    nodes, constants, source = [], [], "input"
    last = len(kernels) - 1
    for index, kernel in enumerate(kernels):
        in_channels = 3 if index == 0 else hidden
        out_channels = 3 if index == last else hidden
        if index == last:
            weights = rng.uniform(-0.3, 0.3, (out_channels, in_channels, kernel, kernel)).astype(np.float32)
            bias = rng.uniform(-1.0, 1.0, out_channels).astype(np.float32)
        else:
            weights = rng.uniform(0.05, 0.3, (out_channels, in_channels, kernel, kernel)).astype(np.float32)
            bias = rng.uniform(0.0, 0.5, out_channels).astype(np.float32)
        constants += [initializer("w%d" % index, weights), initializer("b%d" % index, bias)]
        nodes.append(conv_node(source, "c%d" % index, "w%d" % index, "b%d" % index, kernel))
        source = "c%d" % index
        if index < last:
            nodes.append(relu_node(source, "r%d" % index))
            source = "r%d" % index
    return model_graph(nodes, "native_chain", [tensor_info("input", [1, 3, 8, 8])],
                       [tensor_info(source, [1, 3, 8, 8])], constants)


def walk_graph():
    """RGB 8x8 -> Conv/Relu -> interior MaxPool -> Conv (the untaken profile)."""
    rng = np.random.default_rng(SEED + 50)
    first = rng.uniform(-0.3, 0.3, (8, 3, 3, 3)).astype(np.float32)
    second = rng.uniform(-0.3, 0.3, (3, 8, 3, 3)).astype(np.float32)
    nodes = [conv_node("input", "c0", "w0", "b0", 3),
             relu_node("c0", "r0"),
             pool_node("MaxPool", "r0", "pool"),
             conv_node("pool", "output", "w1", "b1", 3)]
    return model_graph(nodes, "walk_chain", [tensor_info("input", [1, 3, 8, 8])],
                       [tensor_info("output", [1, 3, 4, 4])],
                       [initializer("w0", first), initializer("b0", rng.uniform(-1, 1, 8).astype(np.float32)),
                        initializer("w1", second), initializer("b1", rng.uniform(-1, 1, 3).astype(np.float32))])


def main():
    rows = []
    examples = [
        ("chain-n3-k3", chain_graph(8, [3, 3, 1])),
        ("chain-n4-k1", chain_graph(6, [1, 1, 3, 1])),
    ]
    for index, (label, model) in enumerate(examples):
        binary, meta, report = compile_and_report(model, "%s.onnx" % label, label=label)
        rng = np.random.default_rng(SEED + 100 + index)
        cases = deterministic_cases(rng, 4, 8, 8, 3)
        reference = np.stack([chain_reference(case, model, meta) for case in cases])
        publish_suite(FOLDER, binary, cases, reference, meta, index=index)
        published = read_expected(FOLDER, index).reshape(reference.shape)
        row = dict(report)
        row["check"] = report_checks(label, reference, published, model, cases, meta,
                                     "chain_n.chain_n_reference")
        rows.append(row)

    model = walk_graph()
    binary, meta, report = compile_and_report(model, "chain_walk_pool.onnx", label="walk-pool-conv")
    rng = np.random.default_rng(SEED + 200)
    cases = deterministic_cases(rng, 4, 8, 8, 3)
    reference = np.stack([walk_reference(case, model, meta) for case in cases])
    publish_suite(FOLDER, binary, cases, reference, meta, index=2)
    published = read_expected(FOLDER, 2).reshape(reference.shape)
    row = dict(report)
    row["check"] = report_checks("walk-pool-conv", reference, published, model, cases, meta,
                                 "walk.chain_walk_reference")
    rows.append(row)

    # Board-verified evidence: the same two pipelines reproduce the recorded
    # expected bytes of research/native_chain_suite and research/walk_chain_suite.
    cross_check_suite("native_chain_suite", chain_reference)
    cross_check_suite("walk_chain_suite", walk_reference)

    print_board_recipe(FOLDER)
    print_summary("03_conv_chain", rows, FOLDER)


if __name__ == "__main__":
    main()
