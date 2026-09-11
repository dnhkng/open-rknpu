"""Depthwise-chain suite: depthwise-separable blocks inside DAG branches.

Independently generated commands (no RKNN, no captures). A branch may be a chain of one
to three Conv layers, and a **depthwise layer may sit inside that chain** (input
channels == group). The dedicated depthwise emitter models a branch that reads the stem
directly, so a chained depthwise layer is expanded to an equivalent block-diagonal dense
kernel and emitted through the per-layer dense path. Joins then fold branch finals and
earlier results exactly as in the other join-DAG profiles. Expected outputs come from
`open_rknpu.join_dag.join_dag_reference`; board runner: tests/board_io.c through
`research/run_v5_suite.py`.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.join_dag import join_dag_reference
from open_rknpu.scheduler import compile_sequence

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/depthwise_chain_suite")
parser.add_argument("--cases", type=int, default=32)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)
for pattern in ("model*.onnx", "model*.bin", "input*.u8", "expected*.i8"):
    for stale in root.glob(pattern):
        stale.unlink()
rng = np.random.default_rng(111700)

D, W = "dense", "depthwise"
CONFIGS = [
    (3, [[(D, 3, 8, 1), (W, 8, 8, 3), (D, 8, 3, 1)], [(D, 3, 3, 1)], [(D, 3, 3, 1)]],
     [("Add", "b0_2", "b1_0"), ("Mul", "j0", "b2_0")]),
    (3, [[(W, 3, 3, 3), (D, 3, 3, 1)], [(D, 3, 3, 1)], [(D, 3, 3, 1)]],
     [("Sub", "b0_1", "b1_0"), ("Add", "j0", "b2_0")]),
    (3, [[(D, 3, 16, 1), (W, 16, 16, 3), (D, 16, 3, 1)], [(D, 3, 3, 1)], [(D, 3, 3, 1)]],
     [("Add", "b0_2", "b1_0"), ("Mul", "j0", "b2_0")]),
    (3, [[(D, 3, 8, 3), (W, 8, 8, 1), (D, 8, 3, 3)], [(D, 3, 3, 1)], [(D, 3, 3, 1)]],
     [("Add", "b0_2", "b1_0"), ("Sub", "j0", "b2_0")]),
    (3, [[(D, 3, 8, 1), (W, 8, 8, 3), (D, 8, 3, 1)], [(D, 3, 8, 1), (W, 8, 8, 3), (D, 8, 3, 1)],
         [(D, 3, 3, 1)]],
     [("Add", "b0_2", "b1_2"), ("Mul", "j0", "b2_0")]),
    (3, [[(D, 3, 4, 1), (W, 4, 4, 3), (D, 4, 3, 1)], [(D, 3, 3, 1)], [(D, 3, 3, 1)],
         [(D, 3, 3, 1)]],
     [("Add", "b0_2", "b1_0"), ("Mul", "j0", "b2_0"), ("Mul", "j1", "b3_0")]),
    (3, [[(D, 3, 3, 1)], [(D, 3, 8, 1), (W, 8, 8, 3), (D, 8, 3, 1)], [(D, 3, 3, 1)]],
     [("Add", "b1_2", "b0_0"), ("Mul", "j0", "b2_0")]),
    (3, [[(D, 3, 8, 1), (W, 8, 8, 3), (D, 8, 3, 1)], [(W, 3, 3, 3)], [(D, 3, 3, 1)]],
     [("Add", "b0_2", "b1_0"), ("Sub", "j0", "b2_0")]),
    (3, [[(D, 3, 16, 3), (W, 16, 16, 3), (D, 16, 3, 3)], [(D, 3, 3, 1)], [(D, 3, 3, 1)]],
     [("Mul", "b0_2", "b1_0"), ("Add", "j0", "b2_0")]),
    (3, [[(D, 3, 8, 1), (W, 8, 8, 1), (D, 8, 3, 1)], [(D, 3, 8, 1), (W, 8, 8, 3), (D, 8, 3, 1)],
         [(D, 3, 8, 1), (W, 8, 8, 3), (D, 8, 3, 1)]],
     [("Add", "b0_2", "b1_2"), ("Sub", "j0", "b2_2")]),
    (3, [[(W, 3, 3, 3), (D, 3, 3, 3)], [(D, 3, 8, 1), (W, 8, 8, 3), (D, 8, 3, 1)],
         [(D, 3, 3, 1)], [(D, 3, 3, 1)]],
     [("Add", "b1_2", "b0_1"), ("Mul", "j0", "b2_0"), ("Mul", "j1", "b3_0")]),
    (3, [[(D, 3, 8, 3), (W, 8, 8, 3), (D, 8, 3, 1)], [(W, 3, 3, 3), (D, 3, 3, 1)],
         [(D, 3, 3, 1)]],
     [("Add", "b0_2", "b1_1"), ("Mul", "j0", "b2_0")]),
]

manifest = []
for index, (hidden, branches, expression) in enumerate(CONFIGS):
    constants = [nh.from_array(rng.uniform(.02, .06, (hidden, 3, 1, 1)).astype(np.float32), "w_stem"),
                 nh.from_array(rng.uniform(-1, 1, (hidden,)).astype(np.float32), "bs_stem")]
    nodes = [h.make_node("Conv", ["image", "w_stem", "bs_stem"], ["stem"], kernel_shape=[1, 1])]
    channels_of = {"stem": hidden}
    for branch_index, layers in enumerate(branches):
        previous = "stem"
        for layer_index, (kind, in_channels, out_channels, kernel) in enumerate(layers):
            shape = (out_channels, 1, kernel, kernel) if kind == W else (out_channels, in_channels, kernel, kernel)
            constants += [nh.from_array(rng.uniform(.02, .06, shape).astype(np.float32),
                                       f"w{branch_index}_{layer_index}"),
                          nh.from_array(rng.uniform(-1, 1, (out_channels,)).astype(np.float32),
                                        f"bs{branch_index}_{layer_index}")]
            attributes = dict(kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4)
            if kind == W:
                attributes["group"] = out_channels
            name = f"b{branch_index}_{layer_index}"
            nodes.append(h.make_node("Conv", [previous, f"w{branch_index}_{layer_index}",
                                              f"bs{branch_index}_{layer_index}"], [name], **attributes))
            previous = name
            channels_of[name] = out_channels
    last = None
    for position, (kind, first, second) in enumerate(expression):
        nodes.append(h.make_node(kind, [first, second], [f"j{position}"]))
        last = f"j{position}"
    graph = h.make_graph(nodes, "depthwise_chain",
        [h.make_tensor_value_info("image", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info(last, 1, [1, 3, 8, 8])], constants)
    for name, channels in channels_of.items():
        graph.value_info.append(h.make_tensor_value_info(name, 1, [1, channels, 8, 8]))
    for position in range(len(expression)):
        graph.value_info.append(h.make_tensor_value_info(f"j{position}", 1, [1, 3, 8, 8]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    path = root / f"model{index:03}.onnx"
    onnx.save(model, path)
    binary, meta = compile_sequence(path)
    assert meta["profile"] == "join-dag", (index, meta.get("profile"))
    path.with_suffix(".bin").write_bytes(binary)
    cases = rng.integers(0, 256, (args.cases, 8, 8, 3), dtype=np.uint8)
    cases[0] = 0
    cases[1] = 255
    cases[2] = 128
    cases.tofile(root / f"input{index:03}.u8")
    expected = np.stack([join_dag_reference(case, meta["stem_quantization"], meta["head_quantization"],
                                            meta["head_names"], meta["join_expression"],
                                            head_kinds=meta["head_kinds"],
                                            depthwise_quantizations=meta["depthwise_quantization"],
                                            branch_names=meta["branch_names"],
                                            branch_quantization=meta["branch_quantization"])
                         for case in cases])
    expected.tofile(root / f"expected{index:03}.i8")
    manifest.append({"index": index, "hidden_channels": hidden,
                     "branches": [[list(layer) for layer in branch] for branch in branches],
                     "expression": [[kind, first, second] for kind, first, second in expression],
                     "cases": args.cases, "outputs": 1, "output_bytes": 192,
                     "output_scale": meta["output_scale"], "output_zero_point": meta["output_zero_point"],
                     "branch_names": meta["branch_names"], "head_kinds": meta["head_kinds"],
                     "schedule": meta["schedule"], "tensor_offsets": meta["tensor_offsets"],
                     "expected_nonzero": int(np.count_nonzero(expected)),
                     "expected_values": int(np.size(expected))})
    print(index, "depthwise layers",
          sum(1 for branch in branches for layer in branch if layer[0] == W),
          "nonzero", int(np.count_nonzero(expected)), flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} depthwise-chain models in {root}")
