"""Elementwise DAG suite: Mul({Add,Mul,Sub,Max}(a,b), a) with fan-out.

Independently generated commands; expected bytes come from the composed integer
references. Board runner: tests/board_api.c.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.elementwise_chain import chain_reference

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/elementwise_dag_suite")
parser.add_argument("--cases", type=int, default=16)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(110430)
manifest = []
index = 0
for first in ("Add", "Mul", "Sub", "Max"):
    for repeat in range(3):
        wa = rng.uniform(-.7, .8, (3, 3, 1, 1)).astype(np.float32); ba = rng.uniform(-2, 2, (3,)).astype(np.float32)
        wb = rng.uniform(-.7, .8, (3, 3, 1, 1)).astype(np.float32); bb = rng.uniform(-2, 2, (3,)).astype(np.float32)
        nodes = [h.make_node("Conv", ["a", "wa", "ba"], ["A"], kernel_shape=[1, 1]),
                 h.make_node("Conv", ["b", "wb", "bb"], ["B"], kernel_shape=[1, 1]),
                 h.make_node(first, ["A", "B"], ["C"]),
                 h.make_node("Mul", ["C", "A"], ["out"])]
        graph = h.make_graph(nodes, "elementwise_dag",
            [h.make_tensor_value_info("a", 1, [1, 3, 8, 8]), h.make_tensor_value_info("b", 1, [1, 3, 8, 8])],
            [h.make_tensor_value_info("out", 1, [1, 3, 8, 8])],
            [nh.from_array(wa, "wa"), nh.from_array(ba, "ba"), nh.from_array(wb, "wb"), nh.from_array(bb, "bb")])
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)]); model.ir_version = 8
        model = onnx.shape_inference.infer_shapes(model)
        path = root / f"model{index:03}.onnx"; onnx.save(model, path)
        binary, meta = compile_sequence(path)
        (root / f"model{index:03}.bin").write_bytes(binary)
        a = rng.integers(0, 256, (args.cases, 8, 8, 3), dtype=np.uint8)
        b = rng.integers(0, 256, (args.cases, 8, 8, 3), dtype=np.uint8)
        a[0] = 0; a[1] = 255; b[0] = 255; b[1] = 0
        np.concatenate([a, b], axis=1).tofile(root / f"input{index:03}.u8")
        outputs = np.stack([chain_reference(x, y, meta) for x, y in zip(a, b)])
        outputs.tofile(root / f"expected{index:03}.i8")
        manifest.append({"index": index, "first_stage": first, "cases": args.cases,
                         "output_scale": meta["output_scale"], "chain": meta["elementwise_chain"]})
        print(manifest[-1], flush=True)
        index += 1
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} elementwise DAG models in {root}")
