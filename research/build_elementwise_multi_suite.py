"""Multi-input elementwise DAG suite (format v5): Mul(...Mul(Op(a,b),c_k)...,c_last).

3, 4 and 5 external inputs. Independently generated commands; expected bytes from
the composed integer references. Board runner: tests/board_io.c.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.elementwise_multi import multi_input_reference

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/elementwise_multi_suite")
parser.add_argument("--cases", type=int, default=8)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(110460)
manifest = []
index = 0
for first in ("Add", "Mul", "Max"):
    for extra in (1, 2, 3):
        for repeat in range(2):
            def branch():
                return rng.uniform(-.7, .8, (3, 3, 1, 1)).astype(np.float32), rng.uniform(-2, 2, (3,)).astype(np.float32)
            wa, ba = branch(); wb, bb = branch()
            initializers = [nh.from_array(wa, "wa"), nh.from_array(ba, "ba"),
                            nh.from_array(wb, "wb"), nh.from_array(bb, "bb")]
            inputs = ["a", "b"] + [f"c{k}" for k in range(extra)]
            nodes = [h.make_node("Conv", ["a", "wa", "ba"], ["A"], kernel_shape=[1, 1]),
                     h.make_node("Conv", ["b", "wb", "bb"], ["B"], kernel_shape=[1, 1]),
                     h.make_node(first, ["A", "B"], ["C"])]
            previous = "C"
            for k in range(extra):
                w, b = branch()
                initializers += [nh.from_array(w, f"wc{k}"), nh.from_array(b, f"bc{k}")]
                out = f"S{k + 1}" if k < extra - 1 else "out"
                nodes.append(h.make_node("Conv", [f"c{k}", f"wc{k}", f"bc{k}"], [f"D{k}"], kernel_shape=[1, 1]))
                nodes.append(h.make_node("Mul", [previous, f"D{k}"], [out]))
                previous = out
            graph = h.make_graph(nodes, "elementwise_multi",
                [h.make_tensor_value_info(v, 1, [1, 3, 8, 8]) for v in inputs],
                [h.make_tensor_value_info("out", 1, [1, 3, 8, 8])], initializers)
            model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)]); model.ir_version = 8
            model = onnx.shape_inference.infer_shapes(model)
            path = root / f"model{index:03}.onnx"; onnx.save(model, path)
            binary, meta = compile_sequence(path)
            (root / f"model{index:03}.bin").write_bytes(binary)
            values = [rng.integers(0, 256, (args.cases, 8, 8, 3), dtype=np.uint8) for _ in range(2 + extra)]
            values[0][0] = 0; values[1][1] = 255
            np.stack(values, axis=1).reshape(args.cases, -1).tofile(root / f"input{index:03}.u8")
            outputs = np.stack([multi_input_reference(row[0], row[1], list(row[2:]), meta) for row in zip(*values)])
            outputs.tofile(root / f"expected{index:03}.i8")
            manifest.append({"index": index, "first_stage": first, "inputs": 2 + extra,
                             "chain": meta["elementwise_chain"], "output_scale": meta["output_scale"],
                             "cases": args.cases})
            print(manifest[-1], flush=True)
            index += 1
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} multi-input DAG models in {root}")
