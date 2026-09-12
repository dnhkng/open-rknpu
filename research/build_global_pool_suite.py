# SPDX-License-Identifier: MIT
"""Global average pooling / `ReduceMean` lowered to the three-stage reduction (walk form).

Independently generated `GlobalAveragePool` and `ReduceMean(axes=[2,3], keepdims=1)` graphs
over `[1,3,8,8]`. `open_rknpu.normalize` rewrites the terminal spatial mean into three
2x2/stride-2 `AveragePool` stages - the 8->4->2->1 geometry the pooling engine already runs -
and the result compiles through the op-level chain walk of the sequence emitter, i.e. a v5
container. Expected bytes come from the reduction profile's own documented integer reference
(`open_rknpu.reduction.reduction_reference`), which rounds after every stage; this is the
explicit graph's quantized execution, deliberately not an unrounded global mean.

The 1-channel forms take the scheduler's pooling-sequence branch instead (a v3 reduction
container) and live in `research/global_pool_reduce_suite`. Board runner: the v5 runner
`research/run_v5_suite.py`.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

from open_rknpu.quantization import Quantization
from open_rknpu.reduction import reduction_reference
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/global_pool_suite")
parser.add_argument("--cases", type=int, default=16)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)

# (label, operator, input channels, output channels, kernel, activation). A 3-channel
# 8x8 graph takes the op-level chain walk (v5); the K1/K3 bound is the walk's.
GRAPHS = [
    ("globalavg-c3-o3-k1", "GlobalAveragePool", 3, 3, 1, None),
    ("reducemean-c3-o3-k1", "ReduceMean", 3, 3, 1, None),
    ("globalavg-c3-o4-k3-relu", "GlobalAveragePool", 3, 4, 3, "Relu"),
    ("reducemean-c3-o4-k3-relu", "ReduceMean", 3, 4, 3, "Relu"),
    ("globalavg-c3-o8-k1-relu", "GlobalAveragePool", 3, 8, 1, "Relu"),
    ("reducemean-c3-o8-k1-relu", "ReduceMean", 3, 8, 1, "Relu"),
    ("globalavg-c3-o16-k3-relu", "GlobalAveragePool", 3, 16, 3, "Relu"),
    ("reducemean-c3-o16-k1", "ReduceMean", 3, 16, 1, None),
    ("globalavg-c3-o6-k3", "GlobalAveragePool", 3, 6, 3, None),
    ("reducemean-c3-o6-k1-relu", "ReduceMean", 3, 6, 1, "Relu"),
    ("globalavg-c3-o2-k3-relu", "GlobalAveragePool", 3, 2, 3, "Relu"),
    ("reducemean-c3-o2-k1", "ReduceMean", 3, 2, 1, None),
]


def quant_from(params):
    values = dict(params)
    for key in ("weights", "weight_zero_points", "weight_scales", "biases",
                "channel_multipliers"):
        values[key] = np.array(values[key])
    return Quantization(**values)


def build(spec, seed):
    label, operator, in_channels, out_channels, kernel, activation = spec
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-.8, .8, (out_channels, in_channels, kernel, kernel)).astype(np.float32)
    bias = rng.uniform(-1, 1, out_channels).astype(np.float32)
    nodes = [h.make_node("Conv", ["input", "w", "b"], ["conv"], kernel_shape=[kernel, kernel],
                         pads=[kernel // 2] * 4)]
    previous = "conv"
    if activation is not None:
        nodes.append(h.make_node(activation, [previous], ["relu"]))
        previous = "relu"
    attrs = {} if operator == "GlobalAveragePool" else dict(axes=[2, 3], keepdims=1)
    nodes.append(h.make_node(operator, [previous], ["output"], **attrs))
    graph = h.make_graph(
        nodes, label.replace("-", "_"),
        [h.make_tensor_value_info("input", 1, [1, in_channels, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, out_channels, 1, 1])],
        [nh.from_array(weights, "w"), nh.from_array(bias, "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


manifest = []
for index, spec in enumerate(GRAPHS):
    label, operator, in_channels, out_channels, kernel, activation = spec
    model = build(spec, 110540 + index)
    onnx.save(model, root / f"model{index:03}.onnx")
    binary, meta = compile_sequence(root / f"model{index:03}.onnx")
    if meta.get("profile") != "chain-walk":
        raise SystemExit(f"model{index:03} ({label}) routed to {meta.get('profile')!r}, "
                         f"not the chain walk")
    if decode_sequence(binary)["format_version"] != 5:
        raise SystemExit(f"model{index:03} ({label}) is not a v5 container")
    (root / f"model{index:03}.bin").write_bytes(binary)
    rng = np.random.default_rng(110540 + index)
    cases = rng.integers(0, 256, (args.cases, 8, 8, in_channels), dtype=np.uint8)
    cases[0] = 0
    cases[1] = 255
    cases[2] = 128
    cases.tofile(root / f"input{index:03}.u8")
    quantization = quant_from(meta["quantizations"][0])
    outputs = np.stack([reduction_reference(case, quantization, "AveragePool", 3)
                        for case in cases])
    outputs.tofile(root / f"expected{index:03}.i8")
    manifest.append(dict(index=index, label=label, operator=operator, relu=activation == "Relu",
                         input_channels=in_channels, output_channels=out_channels,
                         kernel=kernel, input_shape=[1, 8, 8, in_channels],
                         output_shape=[1, 1, 1, out_channels], profile="chain-walk",
                         format_version=5, cases=args.cases,
                         output_bytes=int(outputs[0].size)))
    print(manifest[-1], flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} v5 global-pooling models in {root}")
