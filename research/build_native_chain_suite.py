"""N-layer native Conv chain suite (3 and 4 layers, shared arena).

Independently generated commands; expected bytes come from the composed integer
references (legacy first layer, native16 later layers). Board runner:
tests/board_api.c.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.chain_n import chain_n_reference

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/native_chain_suite")
parser.add_argument("--cases", type=int, default=16)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(110370)

CONFIGS = [
    [(5, 1), (5, 3), (3, 1)],
    [(8, 3), (3, 1), (3, 3)],
    [(16, 3), (12, 3), (3, 1)],
    [(4, 3), (8, 3), (6, 1), (3, 3)],
    [(6, 1), (6, 1), (6, 1), (3, 1)],
]

def chain_model(layers):
    nodes = []; tensors = []; previous = "input"
    for index, (channels, kernel) in enumerate(layers):
        inputs = 3 if index == 0 else layers[index - 1][0]
        w = rng.uniform(-.7, .8, (channels, inputs, kernel, kernel)).astype(np.float32)
        b = rng.uniform(-2, 2, (channels,)).astype(np.float32)
        tensors += [nh.from_array(w, f"w{index}"), nh.from_array(b, f"b{index}")]
        out = f"c{index}"
        nodes.append(h.make_node("Conv", [previous, f"w{index}", f"b{index}"], [out],
                                 kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4))
        previous = out
        if index < len(layers) - 1:
            nodes.append(h.make_node("Relu", [previous], [f"r{index}"]))
            previous = f"r{index}"
    graph = h.make_graph(nodes, "native_chain",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info(previous, 1, [1, layers[-1][0], 8, 8])], tensors)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model

manifest = []
for index, layers in enumerate(CONFIGS):
    path = root / f"model{index:03}.onnx"
    onnx.save(chain_model(layers), path)
    binary, meta = compile_sequence(path)
    (root / f"model{index:03}.bin").write_bytes(binary)
    cases = rng.integers(0, 256, (args.cases, 8, 8, 3), dtype=np.uint8)
    cases[0] = 0; cases[1] = 255; cases[2] = 128
    cases.tofile(root / f"input{index:03}.u8")
    outputs = np.stack([chain_n_reference(x, meta["quantizations"]) for x in cases])
    outputs.tofile(root / f"expected{index:03}.i8")
    manifest.append({"index": index, "layers": len(layers),
                     "hidden_channels": [c for c, _ in layers],
                     "kernels": [k for _, k in layers], "cases": args.cases,
                     "output_bytes": int(outputs[0].size)})
    print(manifest[-1], flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} native chain models in {root}")
