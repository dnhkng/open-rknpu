"""Diamond DAG suite: shared stem, two consuming heads, one elementwise join.

Independently generated commands (no RKNN, no captures). The arena is placed by
`open_rknpu.liveness` from the task read/write sets, and expected outputs come
from the composed integer references: the legacy stem reference followed by the
native-input reference for each head and the join reference. Board runner:
tests/board_io.c.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.graph import diamond_reference
from open_rknpu.scheduler import compile_sequence

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/diamond_suite")
parser.add_argument("--cases", type=int, default=32)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(110600)

CONFIGS = []
for kind in ("Add", "Mul", "Sub", "Max"):
    for hidden, kernel_a, kernel_b in ((8, 1, 3), (3, 3, 1), (16, 3, 3)):
        CONFIGS.append((kind, hidden, kernel_a, kernel_b))

manifest = []
for index, (kind, hidden, kernel_a, kernel_b) in enumerate(CONFIGS):
    nodes = [
        h.make_node("Conv", ["input", "w1", "b1"], ["stem"], kernel_shape=[1, 1]),
        h.make_node("Relu", ["stem"], ["relu1"]),
        h.make_node("Conv", ["relu1", "wa", "ba"], ["head_a"], kernel_shape=[kernel_a, kernel_a],
                    pads=[kernel_a // 2] * 4),
        h.make_node("Conv", ["relu1", "wb", "bb"], ["head_b"], kernel_shape=[kernel_b, kernel_b],
                    pads=[kernel_b // 2] * 4),
        h.make_node(kind, ["head_a", "head_b"], ["output"]),
    ]
    graph = h.make_graph(nodes, "diamond",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
        [nh.from_array(rng.uniform(-.7, .8, (hidden, 3, 1, 1)).astype(np.float32), "w1"),
         nh.from_array(rng.uniform(-4, 4, (hidden,)).astype(np.float32), "b1"),
         nh.from_array(rng.uniform(-.7, .8, (3, hidden, kernel_a, kernel_a)).astype(np.float32), "wa"),
         nh.from_array(rng.uniform(-4, 4, (3,)).astype(np.float32), "ba"),
         nh.from_array(rng.uniform(-.7, .8, (3, hidden, kernel_b, kernel_b)).astype(np.float32), "wb"),
         nh.from_array(rng.uniform(-4, 4, (3,)).astype(np.float32), "bb")])
    graph.value_info.append(h.make_tensor_value_info("relu1", 1, [1, hidden, 8, 8]))
    graph.value_info.append(h.make_tensor_value_info("head_a", 1, [1, 3, 8, 8]))
    graph.value_info.append(h.make_tensor_value_info("head_b", 1, [1, 3, 8, 8]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    path = root / f"model{index:03}.onnx"
    onnx.save(model, path)
    binary, meta = compile_sequence(path)
    path.with_suffix(".bin").write_bytes(binary)
    cases = rng.integers(0, 256, (args.cases, 8, 8, 3), dtype=np.uint8)
    cases[0] = 0
    cases[1] = 255
    cases[2] = 128
    cases.tofile(root / f"input{index:03}.u8")
    expected = np.stack([diamond_reference(case, meta["stem_quantization"], meta["head_quantization"], kind)
                         for case in cases])
    expected.tofile(root / f"expected{index:03}.i8")
    manifest.append({"index": index, "join": kind, "hidden_channels": hidden,
                     "head_kernels": [kernel_a, kernel_b], "cases": args.cases, "outputs": 1,
                     "output_bytes": 192, "output_scale": meta["output_scale"],
                     "output_zero_point": meta["output_zero_point"],
                     "head_scales": meta["head_scales"], "schedule": meta["schedule"],
                     "tensor_lifetimes": meta["tensor_lifetimes"],
                     "tensor_offsets": {name: int(value) for name, value in meta["tensor_offsets"].items()},
                     "lifetime_bytes": meta["lifetime_bytes"], "allocated_bytes": meta["allocated_bytes"]})
    print(manifest[-1]["index"], kind, hidden, kernel_a, kernel_b, meta["schedule"], flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} diamond models in {root}")
