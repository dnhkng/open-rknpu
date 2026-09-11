"""SPDX-License-Identifier: MIT

Primitive 9/11: a diamond fan-out/fan-in and a three-head join chain.

Verified bounds (`research/COVERAGE_EXPANSION_RESULTS.md`, "Public compiler
additions" -> Mul sources, and the join entries under "Register and numerical
findings"; `PRIMITIVE_ROADMAP.md` "Layout operations and graph integration"): a
1x1 Conv stem (optionally with Relu) at 8x8 fans out to two or more dense heads
with 3 output channels and hidden C3..16, an elementwise join folds them, and an
optional `[Conv, Relu]* Conv` tail follows. Join operands are zero centred;
`Add`/`Sub`/`Max` re-quantize the heads onto one shared scale while `Mul` folds
the two scales. `research/walk_join_suite/` (6 models), `research/diamond_suite/`
(12 models) and `research/join_chain_suite/` (12 models, 3..5 heads) are
board-verified; `open_rknpu.liveness` places the arena from the task read/write
sets.

Two emitters appear: the op-level diamond/fan-in walk (profile `walk-join`),
whose reference is `open_rknpu.walk.join_walk_reference` over the plan from
`open_rknpu.walk.parse_join_walk`, and the left-fold chain (profile `join-chain`),
whose reference is `open_rknpu.graph.diamond_reference` with the recorded stem,
head, join and tail quantizations. A three-head fan-out is only recognised as a
chain at `head_count >= 3`; a two-head diamond stays with the walk. The script
also replays all three board-verified suites and asserts their recorded bytes.

Verification criterion: each published `expected*.i8` is asserted byte-equal to
the profile's integer reference; the float number is a secondary
quantization-quality metric and is never called exact.

Board recipe (also printed at the end of a run):

    research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \\
      --sysroot="$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot" \\
      -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime tests/board_io.c \\
      runtime/open_rknpu.c -o examples/primitives/build/09_join_dag/board_io
    adb shell mkdir -p /userdata/open-npu-research/09_join_dag
    adb push examples/primitives/build/09_join_dag/{board_io,*.bin,*.u8,*.i8} \\
      /userdata/open-npu-research/09_join_dag/
    adb shell 'cd /userdata/open-npu-research/09_join_dag && ./board_io . 3'
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import (compile_and_report, conv_node, cross_check_suite, deterministic_cases,
                    initializer, join_node, model_graph, print_board_recipe, print_summary,
                    publish_suite, qfrom, read_expected, relu_node, report_checks, tensor_info)
from open_rknpu.graph import diamond_reference
from open_rknpu.walk import join_walk_reference, parse_join_walk

FOLDER = Path(__file__).resolve().parent / "build" / "09_join_dag"
SEED = 110901

# (label, join kinds, head kernels, tail kernel or None, stem Relu)
CONFIGS = [
    ("diamond-add", ("Add",), (3, 1), None, True),
    ("diamond-mul-tail", ("Mul",), (1, 3), 3, True),
    ("three-head-add-chain", ("Add", "Add"), (1, 3, 1), None, True),
]


def build(index, config):
    """Stem -> heads -> chained joins -> optional Conv tail."""
    _, kinds, kernels, tail, stem_relu = config
    rng = np.random.default_rng(SEED + index)
    nodes = [conv_node("input", "stem0", "w1", "b1", 1)]
    constants = [initializer("w1", rng.uniform(-0.3, 0.3, (8, 3, 1, 1)).astype(np.float32)),
                 initializer("b1", rng.uniform(-1.0, 1.0, 8).astype(np.float32))]
    source = "stem0"
    if stem_relu:
        nodes.append(relu_node("stem0", "stem"))
        source = "stem"
    for head, kernel in enumerate(kernels):
        constants += [initializer("wh%d" % head,
                                  rng.uniform(-0.3, 0.3, (3, 8, kernel, kernel)).astype(np.float32)),
                      initializer("bh%d" % head, rng.uniform(-1.0, 1.0, 3).astype(np.float32))]
        nodes.append(conv_node(source, "h%d" % head, "wh%d" % head, "bh%d" % head, kernel))
    previous = "h0"
    for position, kind in enumerate(kinds):
        nodes.append(join_node(kind, previous, "h%d" % (position + 1), "j%d" % position))
        previous = "j%d" % position
    if tail is not None:
        constants += [initializer("wt", rng.uniform(-0.3, 0.3, (3, 3, tail, tail)).astype(np.float32)),
                      initializer("bt", rng.uniform(-1.0, 1.0, 3).astype(np.float32))]
        nodes.append(conv_node(previous, "output", "wt", "bt", tail))
    else:
        nodes[-1].output[0] = "output"
    return model_graph(nodes, "join_dag", [tensor_info("input", [1, 3, 8, 8])],
                       [tensor_info("output", [1, 3, 8, 8])], constants)


def walk_join_reference(case, model, meta):
    """Diamond walk reference: parse the plan, quantize each stage, fold the joins."""
    plan = parse_join_walk(model.graph)
    if plan is None:
        raise ValueError("graph is not a walk-join")
    quantizations = {name: (qfrom(params) if params else None)
                     for name, params in meta["quantizations"].items()}
    return join_walk_reference(case, quantizations, plan, meta["join_zero_point"])


def join_chain_reference(case, model, meta):
    """Chain reference: recorded stem/head/join/tail bands applied left to right.

    `join_kinds` is the chain emitter's key; the two-head diamond profile records
    a single `join`, so both spellings are accepted.
    """
    kinds = meta.get("join_kinds") or meta.get("join")
    tails = meta.get("tail_quantization") or ()
    return diamond_reference(case, meta["stem_quantization"], meta["head_quantization"],
                             kinds, tails, meta["join_zero_point"])


def main():
    rows = []
    for index, config in enumerate(CONFIGS):
        label, kinds = config[0], config[1]
        model = build(index, config)
        binary, meta, report = compile_and_report(model, "%s.onnx" % label.replace("-", "_"),
                                                  label=label)
        if len(kinds) == 1:
            reference_for, reference_name = walk_join_reference, "walk.join_walk_reference"
        else:
            reference_for, reference_name = join_chain_reference, "graph.diamond_reference"
        rng = np.random.default_rng(SEED + 100 + index)
        cases = deterministic_cases(rng, 4, 8, 8, 3)
        reference_grid = np.stack([reference_for(case, model, meta) for case in cases])
        publish_suite(FOLDER, binary, cases, reference_grid, meta, index=index)
        published = read_expected(FOLDER, index).reshape(reference_grid.shape)
        row = dict(report)
        row["check"] = report_checks(label, reference_grid, published, model, cases, meta,
                                     reference_name)
        rows.append(row)

    # Board-verified evidence: 6 diamond-walk, 12 diamond-profile, 12 join-chain models.
    cross_check_suite("walk_join_suite", walk_join_reference)
    cross_check_suite("diamond_suite", join_chain_reference)
    cross_check_suite("join_chain_suite", join_chain_reference)

    print_board_recipe(FOLDER)
    print_summary("09_join_dag", rows, FOLDER)


if __name__ == "__main__":
    main()
