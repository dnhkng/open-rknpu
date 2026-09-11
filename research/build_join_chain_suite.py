"""Join-chain runtime-scale suite: a fan-out chain followed by a per-channel scale.

Independently generated commands (no RKNN, no captures). One shared stem feeds
3..5 dense/depthwise heads folded by chained joins, and a final `Mul(result, scale)`
applies a runtime per-channel scale whose operand is the second external input
(three INT8 codes, passed as `code + 128`). Expected outputs come from
`open_rknpu.graph.join_chain_scale_reference`; board runner: tests/board_io.c
through `research/run_v5_suite.py`.


Independently generated commands (no RKNN, no captures). Every head reads the
stem output, then `n-1` elementwise joins fold them left to right with mixed
Add/Sub/Max/Mul kinds before an optional Conv tail. The arena and the task order
come from `open_rknpu.liveness` through the compiler, and the expected outputs
come from `open_rknpu.graph.diamond_reference`, which applies the same head
quantizations, join kinds and tail quantizations in order. Board runner:
tests/board_io.c through `research/run_v5_suite.py`.

Weights are small and positive so the head grids use the int8 range and the
joined outputs are non-degenerate; the host test asserts that they are.
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
parser.add_argument("--directory", default="research/join_scale_suite")
parser.add_argument("--cases", type=int, default=32)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)
for pattern in ("model*.onnx", "model*.bin", "input*.u8", "expected*.i8"):
    for stale in root.glob(pattern):
        stale.unlink()
rng = np.random.default_rng(111300)

# (join kinds, hidden, head kernels, tail kernel or None, stem Relu)
CONFIGS = [
    (("Add", "Add"), 8, (1, 3, 1), None, True),
    (("Sub", "Add"), 16, (3, 1, 3), None, True),
    (("Max", "Add"), 8, (1, 1, 3), None, True),
    (("Mul", "Add"), 16, (3, 3, 1), None, True),
    (("Mul", "Mul"), 8, (1, 3, 3), None, True),
    (("Add", "Add", "Add"), 8, (1, 3, 1, 3), 1, True),
    (("Sub", "Max", "Add"), 16, (3, 1, 3, 1), 3, True),
    (("Mul", "Add", "Add"), 8, (1, 3, 1, 3), 1, True),
    (("Add", "Sub", "Max", "Add"), 8, (1, 3, 1, 3, 1), None, True),
    (("Max", "Add", "Mul", "Add"), 16, (3, 1, 3, 1, 3), None, True),
    (("Add", "Add"), 16, (3, 3, 1), None, False),
    (("Mul", "Add", "Add"), 8, (1, 3, 1, 3), 3, True),
]

manifest = []
for index, (kinds, hidden, kernels, tail, stem_relu) in enumerate(CONFIGS):
    head_count = len(kinds) + 1
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["stem0"], kernel_shape=[1, 1])]
    source = "stem0"
    constants = [nh.from_array(rng.uniform(.02, .06, (hidden, 3, 1, 1)).astype(np.float32), "w1"),
                 nh.from_array(rng.uniform(-1, 1, (hidden,)).astype(np.float32), "b1")]
    if stem_relu:
        nodes.append(h.make_node("Relu", ["stem0"], ["stem"]))
        source = "stem"
    for head, kernel in enumerate(kernels[:head_count]):
        constants += [nh.from_array(rng.uniform(.02, .06, (3, hidden, kernel, kernel)).astype(np.float32), f"wh{head}"),
                      nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), f"bh{head}")]
        nodes.append(h.make_node("Conv", [source, f"wh{head}", f"bh{head}"], [f"h{head}"],
                                 kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4))
    previous = "h0"
    for position, kind in enumerate(kinds):
        nodes.append(h.make_node(kind, [previous, f"h{position + 1}"], [f"j{position}"]))
        previous = f"j{position}"
    if tail is not None:
        constants += [nh.from_array(rng.uniform(.02, .06, (3, 3, tail, tail)).astype(np.float32), "wt"),
                      nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), "bt")]
        nodes.append(h.make_node("Conv", [previous, "wt", "bt"], ["output"],
                                 kernel_shape=[tail, tail], pads=[tail // 2] * 4))
    else:
        nodes[-1].output[0] = "output"
    graph = h.make_graph(nodes, "join_chain",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], constants)
    graph.value_info.append(h.make_tensor_value_info(source, 1, [1, hidden, 8, 8]))
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
    expected = np.stack([diamond_reference(case, meta["stem_quantization"], meta["head_quantization"],
                                           meta["join_kinds"], meta["tail_quantization"],
                                           meta["join_zero_point"]) for case in cases])
    expected.tofile(root / f"expected{index:03}.i8")
    manifest.append({"index": index, "join_kinds": list(kinds), "hidden_channels": hidden,
                     "head_kernels": list(kernels[:head_count]), "tail_kernel": tail,
                     "stem_relu": stem_relu, "head_count": meta["head_count"],
                     "join_count": meta["join_count"], "cases": args.cases, "outputs": 1,
                     "output_bytes": 192, "output_scale": meta["output_scale"],
                     "output_zero_point": meta["output_zero_point"],
                     "schedule": meta["schedule"], "tensor_offsets": meta["tensor_offsets"],
                     "expected_nonzero": int(np.count_nonzero(expected)),
                     "expected_values": int(expected.size)})
    print(index, kinds, f"hidden{hidden}", f"tail{tail}", meta["profile"],
          "nonzero", int(np.count_nonzero(expected)), flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} join-chain models in {root}")
