"""Join-expression DAG suite: internal grids reused across several joins.

Independently generated commands (no RKNN, no captures). One shared 1x1 stem feeds
3..4 dense/depthwise heads, and two or three elementwise joins combine any two
*previously produced* tensors — so a head or an earlier join result can feed more
than one consumer. This is the general expression form of the fan-out profiles; the
left-fold chain keeps its own emitter and is checked separately. Expected outputs
come from `open_rknpu.join_dag.join_dag_reference`, which replays the recorded
expression; board runner: tests/board_io.c through `research/run_v5_suite.py`.
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
parser.add_argument("--directory", default="research/join_dag_suite")
parser.add_argument("--cases", type=int, default=32)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)
for pattern in ("model*.onnx", "model*.bin", "input*.u8", "expected*.i8"):
    for stale in root.glob(pattern):
        stale.unlink()
rng = np.random.default_rng(111500)

# (head kinds, kernels, join expression over produced tensor names)
CONFIGS = [
    (("dense", "dense", "dense"), (1, 3, 1), [("Add", "h0", "h1"), ("Mul", "j0", "h0")]),
    (("dense", "dense", "dense"), (3, 1, 3), [("Add", "h0", "h1"), ("Mul", "j0", "h1")]),
    (("dense", "dense", "dense"), (1, 1, 1), [("Mul", "h0", "h1"), ("Mul", "j0", "h0")]),
    (("dense", "dense", "dense"), (3, 3, 3), [("Sub", "h1", "h0"), ("Add", "j0", "h2")]),
    (("dense", "dense", "dense"), (1, 3, 1), [("Max", "h0", "h1"), ("Sub", "h2", "j0"),
                                               ("Mul", "j1", "h0")]),
    (("dense", "dense", "dense"), (3, 1, 3), [("Add", "h0", "h1"), ("Add", "j0", "h2"),
                                              ("Mul", "j1", "h1")]),
    (("dense", "dense", "dense"), (1, 3, 3), [("Mul", "h0", "h1"), ("Sub", "j0", "h2"),
                                              ("Mul", "j1", "h0")]),
    (("dense", "dense", "dense"), (3, 3, 1), [("Add", "h0", "h1"), ("Sub", "j0", "h2"),
                                              ("Mul", "j1", "h2")]),
    (("dense", "dense", "dense"), (1, 1, 3), [("Mul", "h0", "h1"), ("Mul", "h1", "h2"),
                                              ("Mul", "j0", "j1")]),
    (("dense", "dense", "dense"), (3, 1, 1), [("Add", "h0", "h1"), ("Mul", "j0", "h2"),
                                              ("Mul", "j1", "h0")]),
    (("depthwise", "dense", "dense"), (3, 1, 3), [("Add", "h0", "h1"), ("Mul", "j0", "h0")]),
    (("dense", "depthwise", "depthwise"), (1, 3, 1), [("Add", "h0", "h1"), ("Sub", "j0", "h2"),
                                                      ("Mul", "j1", "h0")]),
]

manifest = []
for index, (kinds, kernels, expression) in enumerate(CONFIGS):
    constants = [nh.from_array(rng.uniform(.02, .06, (3, 3, 1, 1)).astype(np.float32), "w_stem"),
                 nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), "b_stem")]
    nodes = [h.make_node("Conv", ["image", "w_stem", "b_stem"], ["stem"], kernel_shape=[1, 1])]
    for position, (kind, kernel) in enumerate(zip(kinds, kernels)):
        shape = (3, 1, kernel, kernel) if kind == "depthwise" else (3, 3, kernel, kernel)
        constants += [nh.from_array(rng.uniform(.02, .06, shape).astype(np.float32), f"wh{position}"),
                      nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), f"bh{position}")]
        attributes = dict(kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4)
        if kind == "depthwise":
            attributes["group"] = 3
        nodes.append(h.make_node("Conv", ["stem", f"wh{position}", f"bh{position}"], [f"h{position}"],
                                 **attributes))
    last = None
    for position, (kind, first, second) in enumerate(expression):
        nodes.append(h.make_node(kind, [first, second], [f"j{position}"]))
        last = f"j{position}"
    graph = h.make_graph(nodes, "join_dag",
        [h.make_tensor_value_info("image", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info(last, 1, [1, 3, 8, 8])], constants)
    graph.value_info.append(h.make_tensor_value_info("stem", 1, [1, 3, 8, 8]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    path = root / f"model{index:03}.onnx"
    onnx.save(model, path)
    binary, meta = compile_sequence(path)
    assert meta["profile"] == "join-dag", meta.get("profile")
    path.with_suffix(".bin").write_bytes(binary)
    cases = rng.integers(0, 256, (args.cases, 8, 8, 3), dtype=np.uint8)
    cases[0] = 0
    cases[1] = 255
    cases[2] = 128
    cases.tofile(root / f"input{index:03}.u8")
    expected = np.stack([join_dag_reference(case, meta["stem_quantization"], meta["head_quantization"],
                                            meta["head_names"], meta["join_expression"],
                                            head_kinds=meta["head_kinds"],
                                            depthwise_quantizations=meta["depthwise_quantization"])
                         for case in cases])
    expected.tofile(root / f"expected{index:03}.i8")
    manifest.append({"index": index, "head_kinds": list(kinds), "head_kernels": list(kernels),
                     "expression": [[kind, first, second] for kind, first, second in expression],
                     "cases": args.cases, "outputs": 1, "output_bytes": 192,
                     "output_scale": meta["output_scale"], "output_zero_point": meta["output_zero_point"],
                     "head_scales": meta["head_scales"], "join_scales": meta["join_scales"],
                     "schedule": meta["schedule"], "tensor_offsets": meta["tensor_offsets"],
                     "expected_nonzero": int(np.count_nonzero(expected)),
                     "expected_values": int(np.size(expected))})
    print(index, "+".join(kinds), kernels, [(k, a, b) for k, a, b in expression],
          "nonzero", int(np.count_nonzero(expected)), flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} join-DAG models in {root}")
