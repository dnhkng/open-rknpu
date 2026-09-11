"""Depthwise-branch join suite: one shared stem, a dense branch and a depthwise branch.

Independently generated commands (no RKNN, no captures). The dense branch is a
1x1/3x3 Conv, the depthwise branch is a group-3 1x1/3x3/5x5 Conv, and one
elementwise join folds the two 8x8/C3 grids. The depthwise task program comes from
the verified standalone depthwise emitter with its addresses relocated, and the
expected outputs come from `open_rknpu.depthwise_join.depthwise_join_reference`
(legacy stem reference, then the native and depthwise branch references, then the
join reference). Board runner: tests/board_io.c through
`research/run_v5_suite.py`.

Weights are small so both branch grids use the int8 range and the joined outputs
are non-degenerate; the host test asserts that.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.depthwise_join import depthwise_join_reference
from open_rknpu.scheduler import compile_sequence

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/depthwise_join_suite")
parser.add_argument("--cases", type=int, default=32)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)
for pattern in ("model*.onnx", "model*.bin", "input*.u8", "expected*.i8"):
    for stale in root.glob(pattern):
        stale.unlink()
rng = np.random.default_rng(110900)

# (join, dense kernel, depthwise kernel, stem Relu, asymmetric depthwise pair)
CONFIGS = [
    ("Add", 1, 3, True, False),
    ("Add", 3, 3, True, False),
    ("Mul", 1, 3, True, False),
    ("Mul", 3, 3, True, False),
    ("Sub", 1, 5, True, False),
    ("Sub", 3, 5, True, False),
    ("Max", 1, 1, True, False),
    ("Max", 3, 1, True, False),
    ("Add", 3, 5, False, False),
    ("Mul", 1, 5, True, True),
    ("Add", 3, 1, True, True),
    ("Max", 3, 3, True, False),
]

manifest = []
for index, (kind, dense_kernel, dw_kernel, stem_relu, asymmetric) in enumerate(CONFIGS):
    constants = [nh.from_array(rng.uniform(.02, .06, (3, 3, 1, 1)).astype(np.float32), "w1"),
                 nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), "b1")]
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["stem0"], kernel_shape=[1, 1])]
    source = "stem0"
    if stem_relu:
        nodes.append(h.make_node("Relu", ["stem0"], ["stem"]))
        source = "stem"
    constants += [nh.from_array(rng.uniform(.02, .06, (3, 3, dense_kernel, dense_kernel)).astype(np.float32), "wd"),
                  nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), "bd")]
    nodes.append(h.make_node("Conv", [source, "wd", "bd"], ["dense"],
                             kernel_shape=[dense_kernel] * 2, pads=[dense_kernel // 2] * 4))
    dw_range = (-.06, .06) if asymmetric else (.02, .06)
    constants += [nh.from_array(rng.uniform(*dw_range, (3, 1, dw_kernel, dw_kernel)).astype(np.float32), "ww"),
                  nh.from_array(rng.uniform(-1, 1, (3,)).astype(np.float32), "bw")]
    nodes.append(h.make_node("Conv", [source, "ww", "bw"], ["depthwise"],
                             kernel_shape=[dw_kernel] * 2, pads=[dw_kernel // 2] * 4, group=3))
    nodes.append(h.make_node(kind, ["dense", "depthwise"], ["output"]))
    graph = h.make_graph(nodes, "depthwise_join",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
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
    cases.tofile(root / f"input{index:03}.u8")
    expected = np.stack([depthwise_join_reference(case, meta["stem_quantization"],
                                                  meta["head_quantization"][0],
                                                  meta["depthwise_quantization"], meta["join"])
                         for case in cases])
    expected.tofile(root / f"expected{index:03}.i8")
    manifest.append({"index": index, "join": kind, "dense_kernel": dense_kernel,
                     "depthwise_kernel": dw_kernel, "stem_relu": stem_relu,
                     "asymmetric_depthwise": asymmetric, "cases": args.cases, "outputs": 1,
                     "output_bytes": 192, "output_scale": meta["output_scale"],
                     "output_zero_point": meta["output_zero_point"],
                     "join_scales": meta["join_scales"], "schedule": meta["schedule"],
                     "tensor_offsets": meta["tensor_offsets"],
                     "expected_nonzero": int(np.count_nonzero(expected)),
                     "expected_values": int(expected.size)})
    print(index, kind, f"dense{dense_kernel}", f"dw{dw_kernel}", f"asym{asymmetric}",
          "nonzero", int(np.count_nonzero(expected)), flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} depthwise-join models in {root}")
