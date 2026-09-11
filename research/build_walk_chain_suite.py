"""Walk-chain suite: pooling anywhere in a Conv chain (op-level walk).

Independently generated graphs whose pool is *not* the last node, so no profile above
the walk matches them. Expected bytes come from the walk's own documented integer
reference (`open_rknpu.walk.chain_walk_reference`). Board runner:
`research/run_v5_suite.py` (v5 named-tensor containers).
"""
from pathlib import Path
import argparse
import json

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

from open_rknpu.scheduler import compile_sequence
from open_rknpu.walk import chain_walk_reference, load_quantizations, parse_chain

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/walk_chain_suite")
parser.add_argument("--cases", type=int, default=16)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(110660)

# (label, [(kind, out_channels, kernel, activation) or (pool kind,)], height, width)
# Every graph has a pool *followed by* another op: that is the class no profile above
# the walk matches (a terminal pool is the established pooling profile's).
GRAPHS = [
    ("conv-relu-pool-conv", [("conv", 8, 3, "Relu"), ("MaxPool",), ("conv", 3, 3, None)], 8, 8),
    ("conv-poolavg-conv-relu", [("conv", 8, 1, None), ("AveragePool",), ("conv", 3, 3, "Relu")], 8, 8),
    ("two-pools", [("conv", 8, 3, "Relu"), ("MaxPool",), ("conv", 8, 3, "Relu"),
                   ("MaxPool",), ("conv", 3, 1, None)], 8, 8),
    ("hidden16", [("conv", 16, 3, "Relu"), ("MaxPool",), ("conv", 3, 3, None)], 8, 8),
    ("post-pool-chain", [("conv", 8, 3, "Relu"), ("MaxPool",), ("conv", 8, 3, "Relu"),
                         ("conv", 3, 3, None)], 8, 8),
    ("mixed-pools", [("conv", 8, 3, "Relu"), ("AveragePool",), ("conv", 8, 1, "Relu"),
                     ("MaxPool",), ("conv", 3, 3, None)], 8, 8),
    # Narrow/wide grids: the Conv after a pool reads a non-8x8 surface, which is where
    # the geometry-dependent registers mattered.
    ("h8w6", [("conv", 8, 3, "Relu"), ("MaxPool",), ("conv", 3, 3, None)], 8, 6),
    ("h6w8", [("conv", 8, 3, "Relu"), ("MaxPool",), ("conv", 3, 3, None)], 6, 8),
    ("h6w6", [("conv", 8, 1, "Relu"), ("AveragePool",), ("conv", 3, 1, None)], 6, 6),
    # Three halvings, and a pooled graph output reached through a chain.
    ("three-pools", [("conv", 8, 3, "Relu"), ("MaxPool",), ("conv", 8, 3, "Relu"),
                     ("AveragePool",), ("conv", 8, 3, "Relu"), ("MaxPool",),
                     ("conv", 3, 1, None)], 8, 8),
    ("pool-last-chain", [("conv", 8, 3, "Relu"), ("MaxPool",), ("conv", 8, 3, "Relu"),
                         ("MaxPool",)], 8, 8),
    ("hidden3", [("conv", 3, 3, "Relu"), ("MaxPool",), ("conv", 3, 3, None)], 8, 8),
]


def build(ops, height, width, seed):
    rng = np.random.default_rng(seed)
    nodes, inits = [], []
    source, channels, index = "input", 3, 0
    height_out, width_out = height, width
    for spec in ops:
        if spec[0] == "conv":
            _, out_channels, kernel, activation = spec
            nodes.append(h.make_node("Conv", [source, f"w{index}", f"b{index}"], [f"conv{index}"],
                                     kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4))
            inits += [nh.from_array(rng.uniform(-.8, .8, (out_channels, channels, kernel, kernel)).astype(np.float32), f"w{index}"),
                      nh.from_array(rng.uniform(-1, 1, (out_channels,)).astype(np.float32), f"b{index}")]
            source, channels = f"conv{index}", out_channels
            if activation == "Relu":
                nodes.append(h.make_node("Relu", [source], [f"relu{index}"]))
                source = f"relu{index}"
        else:
            nodes.append(h.make_node(spec[0], [source], [f"pool{index}"],
                                     kernel_shape=[2, 2], strides=[2, 2]))
            source = f"pool{index}"
            height_out, width_out = height_out // 2, width_out // 2
        index += 1
    graph = h.make_graph(nodes, "walk_chain",
                         [h.make_tensor_value_info("input", 1, [1, 3, height, width])],
                         [h.make_tensor_value_info(source, 1, [1, channels, height_out, width_out])],
                         inits)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


manifest = []
for index, (label, ops, height, width) in enumerate(GRAPHS):
    model = build(ops, height, width, 5100 + index)
    path = root / f"model{index:03}.onnx"
    onnx.save(model, path)
    binary, meta = compile_sequence(path)
    if meta.get("profile") != "chain-walk":
        raise SystemExit(f"model{index:03} ({label}) routed to {meta.get('profile')!r}, not the walk")
    (root / f"model{index:03}.bin").write_bytes(binary)
    spec = parse_chain(model.graph)
    quantizations = load_quantizations(meta)
    cases = rng.integers(0, 256, (args.cases, height, width, 3), dtype=np.uint8)
    cases[0] = 0
    cases[1] = 255
    cases[2] = 128
    cases.tofile(root / f"input{index:03}.u8")
    outputs = np.stack([chain_walk_reference(case, quantizations, spec["ops"])
                        for case in cases])
    outputs.tofile(root / f"expected{index:03}.i8")
    manifest.append(dict(index=index, label=label,
                         ops=[op["kind"] for op in spec["ops"]],
                         activations=[op["activation"] for op in spec["ops"] if op["kind"] == "conv"],
                         input_shape=list(spec["input_shape"]),
                         output_shape=list(spec["output_shape"]),
                         profile=meta["profile"], cases=args.cases,
                         output_bytes=int(outputs[0].size)))
    print(manifest[-1], flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} chain-walk models in {root}")
