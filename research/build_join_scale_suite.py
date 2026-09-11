"""Join-chain runtime-scale suite: a fan-out chain followed by a per-channel scale.

Independently generated commands (no RKNN, no captures). One shared 1x1 stem feeds
3..5 dense/depthwise heads folded by chained joins, and a final `Mul(result, scale)`
applies a runtime per-channel scale whose operand is the **second external input**
(three INT8 codes, passed as `code + 128`). Expected outputs come from
`open_rknpu.graph.join_chain_scale_reference`; board runner: tests/board_io.c through
`research/run_v5_suite.py`.

Weights are small and positive (mixed-sign for asymmetric depthwise models) so the
head grids use the int8 range and the scaled join is non-degenerate; the host test
asserts that.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.graph import join_chain_scale_reference
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

# (head kinds, kernels, join kinds, stem Relu, asymmetric)
CONFIGS = [
    (("dense", "dense", "dense"), (1, 3, 1), ("Add", "Mul"), True, False),
    (("dense", "depthwise", "dense"), (3, 1, 3), ("Mul", "Add"), True, False),
    (("depthwise", "dense", "depthwise"), (3, 1, 3), ("Add", "Mul"), True, True),
    (("dense", "depthwise", "dense", "depthwise"), (1, 3, 1, 5), ("Add", "Sub", "Add"), True, False),
    (("dense", "dense", "dense", "dense"), (3, 3, 1, 3), ("Mul", "Mul", "Add"), True, False),
    (("depthwise", "depthwise", "dense", "dense"), (3, 1, 1, 3), ("Add", "Add", "Mul"), True, True),
    (("dense", "depthwise", "dense", "dense", "dense"), (1, 3, 1, 3, 1), ("Max", "Add", "Mul", "Sub"), True, False),
    (("dense", "dense", "dense"), (1, 1, 1), ("Mul", "Mul"), False, False),
    (("dense", "depthwise", "dense"), (1, 5, 3), ("Add", "Add"), True, True),
    (("dense", "dense", "dense", "dense", "dense"), (1, 3, 1, 3, 1), ("Add", "Sub", "Max", "Mul"), True, False),
    (("dense", "depthwise", "depthwise", "dense"), (3, 3, 5, 1), ("Mul", "Add", "Add"), True, False),
    (("dense", "dense", "dense"), (3, 3, 3), ("Sub", "Sub"), True, False),
]
CODES = [
    [127, 127, 127],
    [64, 0, -64],
    [-128, 127, 32],
    [96, -48, 24],
]

manifest = []
for index, (kinds, kernels, joins, stem_relu, asymmetric) in enumerate(CONFIGS):
    constants = [nh.from_array(rng.uniform(.02, .06, (3, 3, 1, 1)).astype(np.float32), "w_stem"),
                 nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), "b_stem")]
    nodes = [h.make_node("Conv", ["image", "w_stem", "b_stem"], ["stem0"], kernel_shape=[1, 1])]
    source = "stem0"
    if stem_relu:
        nodes.append(h.make_node("Relu", ["stem0"], ["stem"]))
        source = "stem"
    heads = []
    for position, (kind, kernel) in enumerate(zip(kinds, kernels)):
        weight_range = (-.06, .06) if (asymmetric and kind == "depthwise") else (.02, .06)
        shape = (3, 1, kernel, kernel) if kind == "depthwise" else (3, 3, kernel, kernel)
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
    nodes.append(h.make_node("Mul", [previous, "scale"], ["output"]))
    graph = h.make_graph(nodes, "join_scale",
        [h.make_tensor_value_info("image", 1, [1, 3, 8, 8]),
         h.make_tensor_value_info("scale", 1, [1, 3, 1, 1])],
        [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])], constants)
    graph.value_info.append(h.make_tensor_value_info(source, 1, [1, 3, 8, 8]))
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
    codes = np.asarray(CODES[index % len(CODES)], np.int32)
    packed = bytearray()
    expected = []
    for case in cases:
        packed.extend(case.tobytes())
        packed.extend((codes + 128).astype(np.uint8).tobytes())
        expected.append(join_chain_scale_reference(case, meta["stem_quantization"],
                                                   meta["head_quantization"], meta["join_kinds"], codes,
                                                   head_kinds=meta["head_kinds"],
                                                   depthwise_quantizations=meta["depthwise_quantization"],
                                                   runtime_scale=meta["runtime_scale"],
                                                   output_zero_point=meta["output_zero_point"]))
    (root / f"input{index:03}.u8").write_bytes(bytes(packed))
    np.stack(expected).tofile(root / f"expected{index:03}.i8")
    manifest.append({"index": index, "head_kinds": list(kinds), "head_kernels": list(kernels),
                     "join_kinds": list(joins), "stem_relu": stem_relu,
                     "asymmetric_depthwise": asymmetric, "codes": [int(v) for v in codes],
                     "cases": args.cases, "outputs": 1, "output_bytes": 192,
                     "output_scale": meta["output_scale"], "output_zero_point": meta["output_zero_point"],
                     "runtime_scale": meta["runtime_scale"], "schedule": meta["schedule"],
                     "tensor_offsets": meta["tensor_offsets"],
                     "expected_nonzero": int(np.count_nonzero(expected)),
                     "expected_values": int(np.size(expected))})
    print(index, "+".join(kinds), kernels, joins, CODES[index % len(CODES)], "nonzero",
          int(np.count_nonzero(expected)), flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} join-scale models in {root}")
