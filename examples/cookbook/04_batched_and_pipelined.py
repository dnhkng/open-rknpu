"""SPDX-License-Identifier: MIT

Cookbook 4/7: serial vs batched vs the deep-chain (double-buffered) container.

The same 8-layer `Conv/Relu` chain is compiled three ways and the containers are
compared structurally:

* **serial** - one descriptor per task, every task tail terminal (`0x28`), so the
  runtime issues one ioctl per layer;
* **batched** - `submission='batched'` relinks every task tail to the successor's
  fetch amount (`0x40` for a 126-word program), so one ioctl runs the whole list
  (`docs/plans/pipelining-plan.md` S1/S2/S8);
* **deep-chain** - the same batched submission plus `reuse_intermediates=True`,
  the chain emitter's double-buffered arena (`pipelining-plan.md`, "Double
  buffering (done)"): layer L writes buffer `(L-1)%2`, so only two 1 KiB surfaces
  are ever live.

The script asserts the tail words, the engine-run counts, the arena shrink, and
that all three containers produce byte-identical integer output. It then prints
the measured board numbers as a table with citations and the exact
`tests/board_async.c` command that reproduces the pipelining measurement.

`docs/performance.md` is not present in this tree; every number below is quoted
from `docs/board.md` ("Timing") or `docs/plans/pipelining-plan.md`.

Run:
    PYTHONPATH=src python examples/cookbook/04_batched_and_pipelined.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from onnx import numpy_helper as nh

from cookbook_common import build_dir, conv_node, model_graph, relu_node, tail_words, tensor_info
from open_rknpu.chain_n import chain_n_reference
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import TERMINAL_CONTROL, amount_control, decode_sequence

FOLDER = build_dir("04_batched_and_pipelined")
SEED = 40404
LAYERS, HIDDEN, SIZE, CHANNELS = 8, 8, 8, 3
TERMINAL = TERMINAL_CONTROL
LINKED = amount_control(126)  # 0x40: the fetch amount of a 126-register successor


def build_chain():
    """`[Conv, Relu] * (LAYERS-1) + [Conv]` at [1,3,8,8] - the native-chain profile."""
    rng = np.random.default_rng(SEED)
    nodes, initializers = [], []
    source, out_channels = "input", CHANNELS
    for index in range(LAYERS):
        out_channels = HIDDEN if index < LAYERS - 1 else CHANNELS
        weights = rng.uniform(-0.3, 0.3, (out_channels, source_channels(index), 1, 1)).astype(np.float32)
        bias = rng.uniform(-0.2, 0.2, out_channels).astype(np.float32)
        nodes.append(conv_node(source, "conv%d" % index, "w%d" % index, "b%d" % index, 1,
                               pads=[0, 0, 0, 0], strides=(1, 1)))
        initializers += [nh.from_array(weights, "w%d" % index), nh.from_array(bias, "b%d" % index)]
        if index < LAYERS - 1:
            nodes.append(relu_node("conv%d" % index, "relu%d" % index))
            source = "relu%d" % index
        else:
            source = "conv%d" % index
    return model_graph(nodes, "deep_chain", [tensor_info("input", [1, CHANNELS, SIZE, SIZE])],
                       [tensor_info(source, [1, CHANNELS, SIZE, SIZE])], initializers)


def source_channels(index):
    """Input channels of layer `index` in `build_chain`."""
    return CHANNELS if index == 0 else HIDDEN


def compile_variant(model, label, **kwargs):
    """Compile one submission shape and report its structure."""
    binary, meta = compile_sequence(model, **kwargs)
    info = decode_sequence(binary)
    tails = tail_words(binary, info)
    engine_runs = 1 if meta.get("submission") == "batched" else info["task_count"]
    controls = [control for _, control in tails]
    assert info["task_count"] == LAYERS, (label, info["task_count"])
    if meta.get("submission") == "batched":
        assert engine_runs == 1, label
        assert controls[-1] == TERMINAL and all(c == LINKED for c in controls[:-1]), (label, controls)
        assert [link for link, _ in tails[:-1]] == [task["command_offset"] for task in info["tasks"][1:]]
    else:
        assert engine_runs == info["task_count"]
        assert all(link == 0 and control == TERMINAL for link, control in tails), (label, tails)
    return dict(label=label, binary=binary, meta=meta, info=info, engine_runs=engine_runs,
                controls=controls, arena=info["arena_bytes"], bytes=len(binary), tails=tails)


def print_measured():
    """The board-measured reference numbers, each with its source."""
    print("  measured board numbers (no board is used by this script):")
    rows = [
        ("3-conv chain, 3 tasks, one batched job", "67 us", "116 us", "0.6x",
         "pipelining-plan.md S2"),
        ("3-conv chain, one batched job, paired", "195 us", "87 us pipelined", "2.25x",
         "board.md Timing; pipelining-plan.md S4"),
        ("8-layer chain", "1325 us", "194 us", "6.8x", "board.md Timing; pipelining-plan.md S2"),
        ("12-layer chain", "1932 us", "176 us", "11.0x", "board.md Timing; pipelining-plan.md S2"),
        ("16-layer chain", "345 us", "163 us", "2.1x", "pipelining-plan.md S2"),
        ("7-task mel-CNN container", "2.54 ms mean", "-", "-", "board.md Timing"),
    ]
    print("    %-40s %-12s %-16s %-6s %s" % ("container", "serial/sync", "batched/pipelined", "gain", "source"))
    for name, serial, batched, gain, source in rows:
        print("    %-40s %-12s %-16s %-6s %s" % (name, serial, batched, gain, source))
    print("    note: docs/performance.md does not exist in this tree; citations above are the measured record")


def print_board_recipe():
    """The exact cross-compile + adb commands for the pipelining probe."""
    remote = "/userdata/open-npu-research/04_pipelining"
    print("  reproduce the pipelining measurement (`tests/board_async.c`, 48 paired rounds):")
    print("    research/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc \\")
    print("      --sysroot=\"$PWD/research/toolchain/arm-rockchip830-linux-uclibcgnueabihf/sysroot\" \\")
    print("      -O2 -std=gnu99 -Wall -Wextra -Werror -Iruntime tests/board_async.c \\")
    print("      runtime/open_rknpu.c -o %s/board_async" % FOLDER)
    print("    adb shell mkdir -p %s" % remote)
    print("    adb push %s/board_async %s/model_v5.bin %s/input_v5.u8 %s/" % (FOLDER, FOLDER, FOLDER, remote))
    print("    adb shell 'cd %s && ./board_async model_v5.bin input_v5.u8 48'" % remote)
    print("    # board_async needs the v5 named-tensor framing; model_v5.bin is the 8-layer chain")
    print("    # compiled with --expose-intermediates and submitted batched.")


def main():
    model = build_chain()
    variants = [
        compile_variant(model, "serial", submission="serial"),
        compile_variant(model, "batched", submission="batched"),
        compile_variant(model, "deep-reuse", submission="batched", reuse_intermediates=True),
    ]
    board = compile_variant(model, "v5-batched", submission="batched", expose_intermediates=True)
    (FOLDER / "model_v5.bin").write_bytes(board["binary"])
    rng = np.random.default_rng(SEED + 1)
    packed = rng.integers(0, 256, SIZE * SIZE * CHANNELS, dtype=np.uint8)
    (FOLDER / "input_v5.u8").write_bytes(packed.tobytes())

    # Structural comparison.
    assert variants[2]["arena"] < variants[0]["arena"], "intermediate reuse must shrink the arena"
    assert variants[1]["arena"] == variants[0]["arena"], "batched submission does not resize the arena"
    assert board["info"]["tensor_count"] > 0, "the board container must be v5 named-tensor framing"
    assert variants[2]["meta"]["reused_intermediates"] is True

    # Same arithmetic in every submission shape.
    case = np.random.default_rng(SEED + 2).integers(0, 256, (SIZE, SIZE, CHANNELS), dtype=np.uint8)
    reference = chain_n_reference(case, variants[0]["meta"]["quantizations"])
    for variant in variants:
        got = chain_n_reference(case, variant["meta"]["quantizations"])
        assert np.array_equal(got, reference), "%s changed the arithmetic" % variant["label"]

    print("  structural comparison (container bytes, tasks, engine runs, arena, tail controls):")
    print("    %-12s %8s %5s %6s %7s  %s" %
          ("variant", "bytes", "tasks", "runs", "arena", "tail control words"))
    for variant in variants:
        print("    %-12s %8d %5d %6d %7d  %s" %
              (variant["label"], variant["bytes"], variant["info"]["task_count"], variant["engine_runs"],
               variant["arena"], " ".join("0x%02x" % control for control in variant["controls"])))
    print("  v5 board container: %d bytes, %d tasks, %d named tensors, tail controls %s" %
          (board["bytes"], board["info"]["task_count"], board["info"]["tensor_count"],
           " ".join("0x%02x" % control for control in board["controls"])))
    assert variants[0]["controls"] == [TERMINAL] * LAYERS
    assert variants[1]["controls"][:-1] == [LINKED] * (LAYERS - 1)
    print_measured()
    print_board_recipe()
    print("04_batched_and_pipelined: serial %dB/%dtask x%d runs, batched %dB/%dtask x1 run, "
          "deep reuse arena %d->%d B, all outputs byte-identical -> %s" %
          (variants[0]["bytes"], LAYERS, variants[0]["engine_runs"], variants[1]["bytes"], LAYERS,
           variants[0]["arena"], variants[2]["arena"], FOLDER))


if __name__ == "__main__":
    main()
