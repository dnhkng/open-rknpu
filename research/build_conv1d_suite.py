"""1-D convolution promoted to the `H=1` 2-D form.

Independently generated rank-3 `[N, C, L]` graphs. `open_rknpu.normalize` rewrites the whole
graph to `[N, C, 1, L]` (same memory layout, so the bytes the caller passes are unchanged and
the container reports `H=1`), and the result compiles as `native16-input`. Expected bytes come
from the native input profile's own documented integer reference
(`open_rknpu.native.native_input_reference`). Board runner: `tests/board_api.c`.
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
parser.add_argument("--directory", default="research/conv1d_suite")
parser.add_argument("--cases", type=int, default=16)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)

# (label, input channels, output channels, kernel, padding, stride, activation, length)
GRAPHS = [
    ("conv1d-c3-k3-p1-l16", 3, 4, 3, 1, 1, None, 16),
    ("conv1d-c3-k3-p1-l16-relu", 3, 4, 3, 1, 1, "Relu", 16),
    ("conv1d-c1-k3-p1-l32", 1, 8, 3, 1, 1, None, 32),
    ("conv1d-c1-k3-p1-l32-relu", 1, 16, 3, 1, 1, "Relu", 32),
    ("conv1d-c3-k5-p2-l40", 3, 8, 5, 2, 1, None, 40),
    ("conv1d-c3-k5-p2-l40-relu", 3, 8, 5, 2, 1, "Relu", 40),
    ("conv1d-c1-k5-p2-l64", 1, 4, 5, 2, 1, None, 64),
    ("conv1d-c3-k2-p0-l24", 3, 4, 2, 0, 1, None, 24),
    ("conv1d-c3-k3-p1-s2-l32", 3, 8, 3, 1, 2, None, 32),
    ("conv1d-c1-k3-p1-s2-l48", 1, 8, 3, 1, 2, None, 48),
    ("conv1d-c3-k3-p0-l20", 3, 4, 3, 0, 1, None, 20),
    ("conv1d-c1-k3-p1-l128", 1, 1, 3, 1, 1, None, 128),
]


def quant_from(params):
    params = dict(params)
    for key in ("weights", "weight_zero_points", "weight_scales", "biases", "channel_multipliers"):
        params[key] = np.array(params[key])
    return Quantization(**params)


def build(spec, seed):
    label, in_channels, out_channels, kernel, padding, stride, activation, length = spec
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-.8, .8, (out_channels, in_channels, kernel)).astype(np.float32)
    bias = rng.uniform(-1, 1, out_channels).astype(np.float32)
    out_length = (length + 2 * padding - kernel) // stride + 1
    conv_output = "output" if activation is None else "conv"
    nodes = [h.make_node("Conv", ["input", "w", "b"], [conv_output], kernel_shape=[kernel],
                         pads=[padding, padding], strides=[stride])]
    if activation is not None:
        nodes.append(h.make_node(activation, [conv_output], ["output"]))
    graph = h.make_graph(
        nodes, label.replace("-", "_"),
        [h.make_tensor_value_info("input", 1, [1, in_channels, length])],
        [h.make_tensor_value_info("output", 1, [1, out_channels, out_length])],
        [nh.from_array(weights, "w"), nh.from_array(bias, "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model, out_length


manifest = []
for index, spec in enumerate(GRAPHS):
    label = spec[0]
    model, out_length = build(spec, 110370 + index)
    onnx.save(model, root / f"model{index:03}.onnx")
    binary, meta = compile_sequence(root / f"model{index:03}.onnx")
    if meta.get("profile") != "native16-input":
        raise SystemExit(f"model{index:03} ({label}) routed to {meta.get('profile')!r}, not native16-input")
    (root / f"model{index:03}.bin").write_bytes(binary)
    _, in_channels, out_channels, _, _, _, _, length = spec
    rng = np.random.default_rng(110370 + index)
    cases = rng.integers(0, 256, (args.cases, 1, length, in_channels), dtype=np.uint8)
    cases[0] = 0
    cases[1] = 255
    cases[2] = 128
    cases.tofile(root / f"input{index:03}.u8")
    q = quant_from(meta["quantization"])
    outputs = np.stack([native_input_reference(x, q, meta["input_zero_point"],
                                               pads=meta["conv_pads"],
                                               strides=tuple(meta["conv_strides"]),
                                               dilations=tuple(meta["conv_dilations"]))
                        for x in cases])
    outputs.tofile(root / f"expected{index:03}.i8")
    manifest.append(dict(index=index, label=label,
                         ops=[node.op_type for node in model.graph.node],
                         input_shape=[1, 1, length, in_channels],
                         output_shape=[1, 1, out_length, out_channels],
                         profile=meta["profile"], kernel=spec[3], pads=meta["conv_pads"],
                         strides=list(meta["conv_strides"]), cases=args.cases,
                         output_bytes=int(outputs[0].size)))
    print(manifest[-1], flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} 1-D convolution models in {root}")
