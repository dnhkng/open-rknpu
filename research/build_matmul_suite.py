"""Dense `MatMul`/`Gemm` lowered to the verified 1x1 Conv path.

Independently generated dense graphs, each spelled directly or through `Flatten(axis=1)` /
a constant `Reshape`, lowered by `open_rknpu.normalize` into a 1x1 Conv and compiled as
`native16-input`. Expected bytes come from the native input profile's own documented integer
reference (`open_rknpu.native.native_input_reference`). Board runner: `tests/board_api.c`.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

from open_rknpu.native import native_input_reference
from open_rknpu.quantization import Quantization
from open_rknpu.scheduler import compile_sequence

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/matmul_suite")
parser.add_argument("--cases", type=int, default=16)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)

# (label, spelling, channels, outputs)
#   direct        MatMul(A[N, C, 1, 1], B[C, K])          output [N, K, 1, 1]
#   flatten       Flatten(axis=1) -> MatMul(flat, B)      output [N, K]
#   reshape       constant Reshape -> MatMul(flat, B)     output [N, K]
#   gemm          Gemm(A, B, bias) on the feature map     output [N, K, 1, 1]
#   gemm-flatten  Flatten -> Gemm(flat, B, bias)          output [N, K]
#   gemm-transB   Flatten -> Gemm(flat, B.T, bias, transB=1)
#   gemm-nobias   Flatten -> Gemm(flat, B)                output [N, K]
GRAPHS = [
    ("matmul-direct-c8-k4", "direct", 8, 4),
    ("matmul-direct-c16-k8", "direct", 16, 8),
    ("matmul-direct-c24-k1", "direct", 24, 1),
    ("matmul-flatten-c8-k4", "flatten", 8, 4),
    ("matmul-flatten-c32-k16", "flatten", 32, 16),
    ("matmul-reshape-c16-k8", "reshape", 16, 8),
    ("gemm-direct-c8-k4", "gemm", 8, 4),
    ("gemm-direct-c16-k3", "gemm", 16, 3),
    ("gemm-flatten-c8-k4", "gemm-flatten", 8, 4),
    ("gemm-flatten-c40-k8", "gemm-flatten", 40, 8),
    ("gemm-transB-c16-k8", "gemm-transB", 16, 8),
    ("gemm-nobias-c8-k4", "gemm-nobias", 8, 4),
]


def quant_from(params):
    params = dict(params)
    for key in ("weights", "weight_zero_points", "weight_scales", "biases", "channel_multipliers"):
        params[key] = np.array(params[key])
    return Quantization(**params)


def build(spelling, channels, outputs, seed):
    rng = np.random.default_rng(seed)
    operand = rng.uniform(-.8, .8, (channels, outputs)).astype(np.float32)
    bias = rng.uniform(-1, 1, outputs).astype(np.float32)
    initializers = [nh.from_array(operand, "operand"), nh.from_array(bias, "bias")]
    nodes = []
    if spelling in ("direct", "gemm"):
        dense_inputs, output_shape = ["input", "operand"], [1, outputs, 1, 1]
    else:
        dense_inputs, output_shape = ["flat", "operand"], [1, outputs]
        if spelling == "reshape":
            initializers.append(nh.from_array(np.array([0, -1], np.int64), "flat_shape"))
            nodes.append(h.make_node("Reshape", ["input", "flat_shape"], ["flat"]))
        else:
            nodes.append(h.make_node("Flatten", ["input"], ["flat"], axis=1))
    if spelling.startswith("gemm"):
        if spelling == "gemm-transB":
            initializers[0].CopyFrom(nh.from_array(np.ascontiguousarray(operand.T), "operand"))
        attrs = {"transB": 1} if spelling == "gemm-transB" else {}
        if spelling != "gemm-nobias":
            dense_inputs = dense_inputs + ["bias"]
        nodes.append(h.make_node("Gemm", dense_inputs, ["output"], **attrs))
    else:
        nodes.append(h.make_node("MatMul", dense_inputs, ["output"]))
    graph = h.make_graph(
        nodes, spelling.replace("-", "_"),
        [h.make_tensor_value_info("input", 1, [1, channels, 1, 1])],
        [h.make_tensor_value_info("output", 1, output_shape)], initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


manifest = []
for index, (label, spelling, channels, outputs) in enumerate(GRAPHS):
    rng = np.random.default_rng(110360 + index)
    model = build(spelling, channels, outputs, 110360 + index)
    onnx.save(model, root / f"model{index:03}.onnx")
    binary, meta = compile_sequence(root / f"model{index:03}.onnx")
    if meta.get("profile") != "native16-input":
        raise SystemExit(f"model{index:03} ({label}) routed to {meta.get('profile')!r}, not native16-input")
    (root / f"model{index:03}.bin").write_bytes(binary)
    q = quant_from(meta["quantization"])
    cases = rng.integers(0, 256, (args.cases, 1, 1, channels), dtype=np.uint8)
    cases[0] = 0
    cases[1] = 255
    cases[2] = 128
    cases.tofile(root / f"input{index:03}.u8")
    outputs_ref = np.stack([native_input_reference(x, q, meta["input_zero_point"],
                                                   pads=meta["conv_pads"],
                                                   strides=tuple(meta["conv_strides"]),
                                                   dilations=tuple(meta["conv_dilations"]))
                            for x in cases])
    outputs_ref.tofile(root / f"expected{index:03}.i8")
    manifest.append(dict(index=index, label=label, spelling=spelling,
                         ops=["Conv"], source_ops=[node.op_type for node in model.graph.node],
                         input_shape=[1, 1, 1, channels], output_shape=[1, 1, 1, outputs],
                         profile=meta["profile"], cases=args.cases,
                         output_bytes=int(outputs_ref[0].size)))
    print(manifest[-1], flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} dense-lowering models in {root}")
