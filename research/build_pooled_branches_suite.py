"""Pooled-branches suite: multi-layer branch chains that pool before the joins.

Independently generated commands (no RKNN, no captures). Two or three branches each run
one to three dense Conv layers (a depthwise layer inside a chain is expanded to an exact
block-diagonal dense kernel) and then a 2x2 stride-2 MaxPool or AveragePool, so every
join operand is a 4x4/C3 grid. One or two elementwise joins fold the branches. Expected
outputs come from `open_rknpu.pooled_branches.pooled_branches_reference`; board runner:
tests/board_io.c through `research/run_v5_suite.py`.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.pooled_branches import pooled_branches_reference
from open_rknpu.scheduler import compile_sequence

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/pooled_branches_suite")
parser.add_argument("--cases", type=int, default=32)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)
for pattern in ("model*.onnx", "model*.bin", "input*.u8", "expected*.i8"):
    for stale in root.glob(pattern):
        stale.unlink()
rng = np.random.default_rng(111900)

D, W = "dense", "depthwise"
CONFIGS = [
    (3, [[(D, 3, 8, 1), (D, 8, 3, 3)], [(D, 3, 3, 1)]], "MaxPool", ["Add"]),
    (3, [[(D, 3, 3, 3)], [(D, 3, 16, 1), (D, 16, 3, 3)]], "AveragePool", ["Sub"]),
    (3, [[(D, 3, 8, 1), (W, 8, 8, 3), (D, 8, 3, 1)], [(D, 3, 3, 1)]], "MaxPool", ["Mul"]),
    (3, [[(D, 3, 8, 3), (W, 8, 8, 1), (D, 8, 3, 1)], [(D, 3, 8, 1), (D, 8, 3, 3)]],
     "AveragePool", ["Max"]),
    (3, [[(D, 3, 8, 1), (D, 8, 3, 3)], [(D, 3, 3, 1)], [(D, 3, 3, 1)]], "MaxPool", ["Add", "Add"]),
    (3, [[(D, 3, 3, 1)], [(D, 3, 8, 1), (W, 8, 8, 3), (D, 8, 3, 1)], [(D, 3, 3, 3)]],
     "AveragePool", ["Mul", "Add"]),
    (3, [[(D, 3, 16, 1), (D, 16, 3, 3)], [(D, 3, 8, 3), (D, 8, 3, 1)], [(D, 3, 3, 1)]],
     "MaxPool", ["Sub", "Max"]),
    (3, [[(D, 3, 3, 3), (D, 3, 3, 1)], [(D, 3, 3, 1)], [(D, 3, 8, 1), (W, 8, 8, 3), (D, 8, 3, 1)]],
     "AveragePool", ["Add", "Mul"]),
    (3, [[(D, 3, 3, 1), (D, 3, 16, 3), (D, 16, 3, 1)], [(D, 3, 3, 1)]], "MaxPool", ["Add"]),
    (3, [[(D, 3, 8, 1), (W, 8, 8, 1), (D, 8, 3, 1)],
         [(D, 3, 8, 1), (W, 8, 8, 1), (D, 8, 3, 1)],
         [(D, 3, 8, 1), (W, 8, 8, 1), (D, 8, 3, 1)]], "AveragePool", ["Add", "Sub"]),
    (3, [[(D, 3, 4, 1), (W, 4, 4, 3), (D, 4, 3, 1)], [(D, 3, 3, 3)]], "MaxPool", ["Mul"]),
    (3, [[(D, 3, 3, 1)], [(D, 3, 3, 1)], [(D, 3, 3, 3), (D, 3, 8, 1), (D, 8, 3, 1)]],
     "AveragePool", ["Mul", "Mul"]),
    (3, [[(D, 3, 3, 1), (D, 3, 8, 3), (D, 8, 3, 1)], [(D, 3, 3, 1)], [(D, 3, 3, 3)]],
     "MaxPool", ["Add", "Mul"]),
]

manifest = []
for index, (hidden, branches, pool, joins) in enumerate(CONFIGS):
    constants = [nh.from_array(rng.uniform(.02, .06, (hidden, 3, 1, 1)).astype(np.float32), "w_stem"),
                 nh.from_array(rng.uniform(-1, 1, (hidden,)).astype(np.float32), "bs_stem")]
    nodes = [h.make_node("Conv", ["image", "w_stem", "bs_stem"], ["stem"], kernel_shape=[1, 1])]
    channels_of = {"stem": hidden}
    pooled = []
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
        pool_name = f"p{branch_index}"
        nodes.append(h.make_node(pool, [previous], [pool_name], kernel_shape=[2, 2], strides=[2, 2]))
        pooled.append(pool_name)
    last = None
    for position, kind in enumerate(joins):
        first = pooled[0] if position == 0 else last
        nodes.append(h.make_node(kind, [first, pooled[position + 1]], [f"j{position}"]))
        last = f"j{position}"
    graph = h.make_graph(nodes, "pooled_branches",
        [h.make_tensor_value_info("image", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info(last, 1, [1, 3, 4, 4])], constants)
    for name, channels in channels_of.items():
        graph.value_info.append(h.make_tensor_value_info(name, 1, [1, channels, 8, 8]))
    for position in range(len(branches)):
        graph.value_info.append(h.make_tensor_value_info(f"p{position}", 1, [1, 3, 4, 4]))
    for position in range(len(joins)):
        graph.value_info.append(h.make_tensor_value_info(f"j{position}", 1, [1, 3, 4, 4]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    path = root / f"model{index:03}.onnx"
    onnx.save(model, path)
    binary, meta = compile_sequence(path)
    assert meta["profile"] == "pooled-branches", (index, meta.get("profile"))
    path.with_suffix(".bin").write_bytes(binary)
    cases = rng.integers(0, 256, (args.cases, 8, 8, 3), dtype=np.uint8)
    cases[0] = 0
    cases[1] = 255
    cases[2] = 128
    cases.tofile(root / f"input{index:03}.u8")
    expected = np.stack([pooled_branches_reference(case, meta["stem_quantization"], meta["branch_names"],
                                                   meta["branch_quantization"], meta["join_expression"],
                                                   meta["pool"], pooled_names=meta["pooled_names"])
                         for case in cases])
    expected.tofile(root / f"expected{index:03}.i8")
    manifest.append({"index": index, "hidden_channels": hidden, "pool": pool, "joins": list(joins),
                     "branches": [[list(layer) for layer in branch] for branch in branches],
                     "expression": [[step["kind"], step["inputs"][0], step["inputs"][1]]
                                    for step in meta["join_expression"]],
                     "cases": args.cases, "outputs": 1, "output_bytes": 48,
                     "output_scale": meta["output_scale"], "output_zero_point": meta["output_zero_point"],
                     "branch_names": meta["branch_names"], "pooled_names": meta["pooled_names"],
                     "schedule": meta["schedule"], "tensor_offsets": meta["tensor_offsets"],
                     "expected_nonzero": int(np.count_nonzero(expected)),
                     "expected_values": int(np.size(expected))})
    print(index, pool, "branches", [len(branch) for branch in branches], "joins", joins, "nonzero",
          int(np.count_nonzero(expected)), flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} pooled-branch models in {root}")
