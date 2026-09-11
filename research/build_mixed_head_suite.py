"""Mixed-head fan-out suite: 3..6 dense and depthwise heads folded by chained joins.

Independently generated commands (no RKNN, no captures). Every head reads one
shared 1x1 stem output; dense heads are emitted with the shared native Conv field
builder and depthwise heads relocate the verified standalone depthwise program into
the same container, so two emitter families compose by tensor name inside one
fan-out. Joins may be Add/Sub/Max/Mul at every position and the tail is an optional
`[Conv, Relu]* Conv` chain. Expected outputs come from
`open_rknpu.graph.diamond_reference` with `head_kinds`/`depthwise_quantizations`,
which selects the depthwise reference for the depthwise head grids. Board runner:
tests/board_io.c through `research/run_v5_suite.py`.

Weights are small and positive (mixed-sign for asymmetric depthwise models) so the
head grids use the int8 range and the folded outputs are non-degenerate; the host
test asserts that.
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
parser.add_argument("--directory", default="research/mixed_head_suite")
parser.add_argument("--cases", type=int, default=32)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)
for pattern in ("model*.onnx", "model*.bin", "input*.u8", "expected*.i8"):
    for stale in root.glob(pattern):
        stale.unlink()
rng = np.random.default_rng(111100)

# (head kinds, kernels, join kinds, tail kernel or None, hidden, stem Relu, asymmetric)
CONFIGS = [
    (("dense", "dense", "depthwise"), (1, 3, 1), ("Add", "Mul"), None, 3, True, False),
    (("dense", "depthwise", "dense"), (3, 1, 3), ("Mul", "Add"), 1, 3, True, False),
    (("depthwise", "dense", "depthwise"), (3, 1, 3), ("Add", "Mul"), None, 3, True, True),
    (("dense", "depthwise", "dense", "depthwise"), (1, 3, 1, 5), ("Add", "Sub", "Add"), None, 3, True, False),
    (("dense", "depthwise", "depthwise"), (1, 3, 5), ("Mul", "Mul"), None, 3, False, False),
    (("depthwise", "depthwise", "dense"), (3, 1, 1), ("Add", "Add"), None, 3, True, True),
    (("dense", "dense", "dense", "depthwise"), (3, 3, 1, 3), ("Add", "Mul", "Add"), 3, 3, True, False),
    (("depthwise", "dense", "dense", "dense"), (1, 3, 1, 3), ("Mul", "Add", "Mul"), None, 3, True, True),
    (("dense", "depthwise", "dense", "dense"), (3, 1, 3, 3), ("Sub", "Add", "Max"), None, 3, True, False),
    (("dense", "depthwise", "dense", "depthwise", "dense"), (1, 3, 1, 5, 1), ("Add", "Max", "Add", "Sub"), None, 3, True, False),
    (("depthwise", "dense", "depthwise", "dense", "depthwise"), (1, 1, 3, 3, 5), ("Add", "Add", "Mul", "Add"), None, 3, True, True),
    (("dense", "dense", "dense"), (1, 3, 1), ("Add", "Mul"), None, 8, True, False),
]

manifest = []
for index, (kinds, kernels, joins, tail, hidden, stem_relu, asymmetric) in enumerate(CONFIGS):
    constants = [nh.from_array(rng.uniform(.02, .06, (hidden, 3, 1, 1)).astype(np.float32), "w_stem"),
                 nh.from_array(rng.uniform(-1, 1, (hidden,)).astype(np.float32), "b_stem")]
    nodes = [h.make_node("Conv", ["input", "w_stem", "b_stem"], ["stem0"], kernel_shape=[1, 1])]
    source = "stem0"
    if stem_relu:
        nodes.append(h.make_node("Relu", ["stem0"], ["stem"]))
        source = "stem"
    heads = []
    for position, (kind, kernel) in enumerate(zip(kinds, kernels)):
        weight_range = (-.06, .06) if (asymmetric and kind == "depthwise") else (.02, .06)
        shape = (3, 1, kernel, kernel) if kind == "depthwise" else (3, hidden, kernel, kernel)
        constants += [nh.from_array(rng.uniform(*weight_range, shape).astype(np.float32), f"wh{position}"),
                      nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), f"bh{position}")]
        attributes = dict(kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4)
        if kind == "depthwise":
            attributes["group"] = 3
        nodes.append(h.make_node("Conv", [source, f"wh{position}", f"bh{position}"], [f"h{position}"],
                                 **attributes))
        heads.append(f"h{position}")
    previous = heads[0]
    for position, join in enumerate(joins):
        nodes.append(h.make_node(join, [previous, heads[position + 1]], [f"j{position}"]))
        previous = f"j{position}"
    if tail is not None:
        constants += [nh.from_array(rng.uniform(.02, .06, (3, 3, tail, tail)).astype(np.float32), "wt"),
                      nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), "bt")]
        nodes.append(h.make_node("Conv", [previous, "wt", "bt"], ["output"],
                                 kernel_shape=[tail, tail], pads=[tail // 2] * 4))
    else:
        nodes[-1].output[0] = "output"
    graph = h.make_graph(nodes, "mixed_heads",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], constants)
    graph.value_info.append(h.make_tensor_value_info(source, 1, [1, hidden, 8, 8]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    path = root / f"model{index:03}.onnx"
    onnx.save(model, path)
    binary, meta = compile_sequence(path, asymmetric_depthwise=asymmetric)
    path.with_suffix(".bin").write_bytes(binary)
    cases = rng.integers(0, 256, (args.cases, 8, 8, 3), dtype=np.uint8)
    cases[0] = 0
    cases[1] = 255
    cases[2] = 128
    cases.tofile(root / f"input{index:03}.u8")
    expected = np.stack([diamond_reference(case, meta["stem_quantization"], meta["head_quantization"],
                                           meta["join_kinds"], meta["tail_quantization"],
                                           meta["join_zero_point"], head_kinds=meta["head_kinds"],
                                           depthwise_quantizations=meta["depthwise_quantization"])
                         for case in cases])
    expected.tofile(root / f"expected{index:03}.i8")
    manifest.append({"index": index, "head_kinds": list(kinds), "head_kernels": list(kernels),
                     "join_kinds": list(joins), "tail_kernel": tail, "hidden_channels": hidden,
                     "stem_relu": stem_relu, "asymmetric_depthwise": asymmetric,
                     "cases": args.cases, "outputs": 1, "output_bytes": 192,
                     "output_scale": meta["output_scale"], "output_zero_point": meta["output_zero_point"],
                     "schedule": meta["schedule"], "tensor_offsets": meta["tensor_offsets"],
                     "expected_nonzero": int(np.count_nonzero(expected)),
                     "expected_values": int(expected.size)})
    print(index, "+".join(kinds), kernels, joins, f"tail{tail}", f"hidden{hidden}",
          "nonzero", int(np.count_nonzero(expected)), flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} mixed-head models in {root}")
