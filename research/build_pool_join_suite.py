"""Pooled-branch join suite: two Conv+pool branches from one stem.

Independently generated commands (no RKNN, no captures). Each branch is the
verified `Conv[/Relu] -> 2x2 stride-2 pool` profile reduced to a Conv plus a
37-word pool task; the two 4x4/C3 pooled grids feed one elementwise join. Pool
task fields come from `open_rknpu.pooling.pool_registers`, the same builder the
sequence lowering uses, and expected outputs come from
`open_rknpu.pool_join.pool_join_reference`. Board runner: tests/board_io.c through
`research/run_v5_suite.py`.

Weights are small and positive so the branch grids use the int8 range and the
pooled join is not degenerate; the host test asserts that.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.pool_join import pool_join_reference
from open_rknpu.scheduler import compile_sequence

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/pool_join_suite")
parser.add_argument("--cases", type=int, default=32)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)
for pattern in ("model*.onnx", "model*.bin", "input*.u8", "expected*.i8"):
    for stale in root.glob(pattern):
        stale.unlink()
rng = np.random.default_rng(111000)

# (join, pool, (head_a kernel, head_b kernel), hidden, stem Relu)
CONFIGS = [
    ("Add", "MaxPool", (1, 1), 8, True),
    ("Add", "AveragePool", (3, 1), 8, True),
    ("Mul", "MaxPool", (1, 3), 8, True),
    ("Mul", "AveragePool", (3, 3), 8, True),
    ("Sub", "MaxPool", (1, 1), 8, True),
    ("Sub", "AveragePool", (3, 1), 8, True),
    ("Max", "MaxPool", (1, 3), 8, True),
    ("Max", "AveragePool", (3, 3), 8, True),
    ("Add", "MaxPool", (3, 3), 8, False),
    ("Mul", "AveragePool", (1, 1), 16, True),
    ("Sub", "MaxPool", (3, 1), 16, True),
    ("Max", "AveragePool", (1, 3), 3, True),
]

manifest = []
for index, (kind, pool, kernels, hidden, stem_relu) in enumerate(CONFIGS):
    constants = [nh.from_array(rng.uniform(.02, .06, (hidden, 3, 1, 1)).astype(np.float32), "w1"),
                 nh.from_array(rng.uniform(-1, 1, (hidden,)).astype(np.float32), "b1")]
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["stem0"], kernel_shape=[1, 1])]
    source = "stem0"
    if stem_relu:
        nodes.append(h.make_node("Relu", ["stem0"], ["stem"]))
        source = "stem"
    for name, kernel in (("a", kernels[0]), ("b", kernels[1])):
        constants += [nh.from_array(rng.uniform(.02, .06, (3, hidden, kernel, kernel)).astype(np.float32), f"w{name}"),
                      nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), f"b{name}")]
        nodes.append(h.make_node("Conv", [source, f"w{name}", f"b{name}"], [f"head_{name}"],
                                 kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4))
        nodes.append(h.make_node(pool, [f"head_{name}"], [f"pool_{name}"],
                                 kernel_shape=[2, 2], strides=[2, 2]))
    nodes.append(h.make_node(kind, ["pool_a", "pool_b"], ["output"]))
    graph = h.make_graph(nodes, "pool_join",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, 3, 4, 4])], constants)
    graph.value_info.append(h.make_tensor_value_info(source, 1, [1, hidden, 8, 8]))
    graph.value_info.append(h.make_tensor_value_info("head_a", 1, [1, 3, 8, 8]))
    graph.value_info.append(h.make_tensor_value_info("head_b", 1, [1, 3, 8, 8]))
    graph.value_info.append(h.make_tensor_value_info("pool_a", 1, [1, 3, 4, 4]))
    graph.value_info.append(h.make_tensor_value_info("pool_b", 1, [1, 3, 4, 4]))
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
    expected = np.stack([pool_join_reference(case, meta["stem_quantization"],
                                             meta["head_quantization"], meta["join"], meta["pool"])
                         for case in cases])
    expected.tofile(root / f"expected{index:03}.i8")
    manifest.append({"index": index, "join": kind, "pool": pool,
                     "head_kernels": list(kernels), "hidden_channels": hidden,
                     "stem_relu": stem_relu, "cases": args.cases, "outputs": 1,
                     "output_bytes": 48, "output_scale": meta["output_scale"],
                     "output_zero_point": meta["output_zero_point"],
                     "join_scales": meta["join_scales"], "schedule": meta["schedule"],
                     "tensor_offsets": meta["tensor_offsets"],
                     "expected_nonzero": int(np.count_nonzero(expected)),
                     "expected_values": int(expected.size)})
    print(index, kind, pool, kernels, f"hidden{hidden}", "nonzero",
          int(np.count_nonzero(expected)), flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} pool-join models in {root}")
