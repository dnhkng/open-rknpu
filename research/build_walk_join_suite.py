"""Walk-join suite: a pool between each head and the join (op-level join walk).

Graphs whose join operands are pooled head outputs have no profile: the pooling
profiles pool both branches but end at the join, and the diamond profile has no pool.
The op-level walk lowers them; expected bytes come from `open_rknpu.walk`.
Board runner: `research/run_v5_suite.py` (v5 named-tensor containers).
"""
from pathlib import Path
import argparse
import json

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

from open_rknpu.quantization import Quantization
from open_rknpu.scheduler import compile_sequence
from open_rknpu.walk import join_walk_reference, parse_join_walk

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/walk_join_suite")
parser.add_argument("--cases", type=int, default=16)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(110770)

# (label, join, stem_relu, head_a(kernel, relu, pool), head_b(kernel, relu, pool), tails)
GRAPHS = [
    # Mixed pool kinds per branch (the pooling profiles use one kind for both) and a
    # Conv tail after the join are the classes no profile owns.
    ("add-mixed-pools", "Add", True, (1, None, "MaxPool"), (1, None, "AveragePool"), []),
    ("mul-mixed-pools", "Mul", True, (1, "Relu", "AveragePool"), (3, "Relu", "MaxPool"), []),
    ("max-mixed-pools-tail", "Max", True, (3, "Relu", "MaxPool"), (3, None, "AveragePool"),
     [(3, None)]),
    ("sub-mixed-pools-two-tails", "Sub", True, (1, None, "AveragePool"), (1, "Relu", "MaxPool"),
     [(1, "Relu"), (3, None)]),
    ("add-mixed-pools-two-tails", "Add", True, (3, None, "MaxPool"), (3, "Relu", "AveragePool"),
     [(1, "Relu"), (1, None)]),
    ("mul-mixed-pools-tail-norelu", "Mul", False, (1, None, "MaxPool"), (3, "Relu", "AveragePool"),
     [(1, None)]),
]


def build(label, join, stem_relu, head_a, head_b, tails, seed):
    rng = np.random.default_rng(seed)
    nodes, inits = [], []
    nodes.append(h.make_node("Conv", ["input", "w_stem", "b_stem"], ["stem"], kernel_shape=[1, 1]))
    inits += [nh.from_array(rng.uniform(-.08, .08, (8, 3, 1, 1)).astype(np.float32), "w_stem"),
              nh.from_array(rng.uniform(-1, 1, (8,)).astype(np.float32), "b_stem")]
    source = "stem"
    if stem_relu:
        nodes.append(h.make_node("Relu", ["stem"], ["stem_r"]))
        source = "stem_r"
    for tag, (kernel, activation, pool) in zip(("a", "b"), (head_a, head_b)):
        constants = [nh.from_array(rng.uniform(-.08, .08, (3, 8, kernel, kernel)).astype(np.float32), f"w_{tag}"),
                     nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), f"b_{tag}")]
        inits += constants
        name = f"head_{tag}"
        nodes.append(h.make_node("Conv", [source, f"w_{tag}", f"b_{tag}"], [name],
                                 kernel_shape=[kernel] * 2, pads=[kernel // 2] * 4))
        if activation:
            nodes.append(h.make_node("Relu", [name], [name + "_r"]))
            name = name + "_r"
        nodes.append(h.make_node(pool, [name], [f"pool_{tag}"], kernel_shape=[2, 2], strides=[2, 2]))
    nodes.append(h.make_node(join, ["pool_a", "pool_b"], ["joined"]))
    source = "joined"
    for index, (kernel, activation) in enumerate(tails):
        last = index == len(tails) - 1
        name = "output" if last else f"tail{index}"
        inits += [nh.from_array(rng.uniform(-.08, .08, (3, 3, kernel, kernel)).astype(np.float32), f"w_t{index}"),
                  nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), f"b_t{index}")]
        nodes.append(h.make_node("Conv", [source, f"w_t{index}", f"b_t{index}"], [name],
                                 kernel_shape=[kernel] * 2, pads=[kernel // 2] * 4))
        if activation:
            nodes.append(h.make_node("Relu", [name], [name + "_r"]))
            name = name + "_r"
            nodes[-1].output[0] = name
        source = name
    if not tails:
        nodes[-1].output[0] = "output"
        source = "output"
    shape_out = [1, 3, 4, 4]
    graph = h.make_graph(nodes, "walk_join", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, shape_out)], inits)
    graph.value_info.append(h.make_tensor_value_info("stem", 1, [1, 8, 8, 8]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def quantize(params):
    if isinstance(params, Quantization):
        return params
    values = dict(params)
    for key in ("weights", "weight_zero_points", "weight_scales", "biases", "channel_multipliers"):
        values[key] = np.array(values[key])
    return Quantization(**values)


manifest = []
for index, (label, join, stem_relu, head_a, head_b, tails) in enumerate(GRAPHS):
    model = build(label, join, stem_relu, head_a, head_b, tails, 8100 + index)
    path = root / f"model{index:03}.onnx"
    onnx.save(model, path)
    binary, meta = compile_sequence(path)
    if meta.get("profile") != "walk-join":
        raise SystemExit(f"model{index:03} ({label}) routed to {meta.get('profile')!r}")
    (root / f"model{index:03}.bin").write_bytes(binary)
    quantizations = {name: (quantize(params) if params else None)
                     for name, params in meta["quantizations"].items()}
    plan = parse_join_walk(model.graph)
    cases = rng.integers(0, 256, (args.cases, 8, 8, 3), dtype=np.uint8)
    cases[0] = 0
    cases[1] = 255
    cases[2] = 128
    cases.tofile(root / f"input{index:03}.u8")
    outputs = np.stack([join_walk_reference(case, quantizations, plan,
                                            join_zero_point=meta["join_zero_point"])
                        for case in cases])
    outputs.tofile(root / f"expected{index:03}.i8")
    manifest.append(dict(index=index, label=label, join=join, stem_relu=stem_relu,
                         head_a=list(head_a), head_b=list(head_b), tails=tails,
                         profile=meta["profile"], cases=args.cases,
                         output_bytes=int(outputs[0].size)))
    print(manifest[-1], flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} walk-join models in {root}")
