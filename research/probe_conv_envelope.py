"""MIT. Probe the compiler's real convolution envelope with the shapes a pretrained model uses.

Written for the Silero VAD question (`docs/plans/primitive-roadmap.md`, "Real-model envelope"): the
model needs 1-D convs (`kernel_shape [k]`) and rectangular kernels, so this script builds
the smallest ONNX graphs that carry those shapes and reports what the scheduler accepts or
rejects, with the exact message. It is a diagnostic, not evidence of hardware support.

    PYTHONPATH=src python3 research/probe_conv_envelope.py
"""
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_rknpu.scheduler import compile_sequence

RNG = np.random.default_rng(7)
OUT = Path("/tmp/probe_conv_envelope")


def build(path, shape, kernel, strides, pads, relu=False):
    """One Conv (optionally + Relu) with explicit ONNX geometry.

    `shape` is [N, C, W] for a 1-D convolution or [N, C, H, W] for a 2-D one;
    `kernel`/`strides`/`pads` are the ONNX attribute lists for that rank.
    """
    spatial = shape[2:]
    in_channels = shape[1]
    out_spatial = [(size + pads[2 * i] + pads[2 * i + 1] - kernel[i]) // strides[i] + 1
                   for i, size in enumerate(spatial)]
    weights = RNG.uniform(-0.3, 0.3, (8, in_channels, *kernel)).astype(np.float32)
    bias = np.zeros(8, dtype=np.float32)
    nodes = [h.make_node("Conv", ["input", "w", "b"], ["output" if not relu else "conv"],
                         kernel_shape=list(kernel), pads=list(pads), strides=list(strides))]
    if relu:
        nodes.append(h.make_node("Relu", ["conv"], ["output"]))
    graph = h.make_graph(
        nodes, "probe",
        [h.make_tensor_value_info("input", 1, list(shape))],
        [h.make_tensor_value_info("output", 1, [1, 8, *out_spatial])],
        [nh.from_array(weights, "w"), nh.from_array(bias, "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


CASES = [
    # label, shape, kernel, strides, pads, relu
    ("1-D Conv k3, input [1,3,8]", [1, 3, 8], [3], [1], [1, 1], False),
    ("1-D Conv k256 stride128, input [1,1,512]", [1, 1, 512], [256], [128], [0, 0], False),
    ("2-D k3x3 C64 H8 W8 (baseline)", [1, 64, 8, 8], [3, 3], [1, 1], [1, 1, 1, 1], False),
    ("2-D k3x3 C128 H8 W8 (baseline)", [1, 128, 8, 8], [3, 3], [1, 1], [1, 1, 1, 1], False),
    ("2-D k3x3 C129 H8 W8 (native16-input)", [1, 129, 8, 8], [3, 3], [1, 1], [1, 1, 1, 1], False),
    ("2-D k1x3 (rectangular) C128 H8 W8", [1, 128, 8, 8], [1, 3], [1, 1], [0, 1, 0, 1], False),
    ("2-D k3x3 C129 H1 W3 (encoder first layer)", [1, 129, 1, 3], [3, 3], [1, 1], [1, 1, 1, 1], False),
    ("2-D k3x3 C64 H1 W6 stride2 (encoder)", [1, 64, 1, 6], [3, 3], [2, 2], [1, 1, 1, 1], False),
    ("2-D k3x3 C64 H8 W8 + Relu", [1, 64, 8, 8], [3, 3], [1, 1], [1, 1, 1, 1], True),
]

OUT.mkdir(exist_ok=True)
for index, (label, shape, kernel, strides, pads, relu) in enumerate(CASES):
    path = build(OUT / f"case{index:02}.onnx", shape, kernel, strides, pads, relu)
    try:
        data, meta = compile_sequence(str(path))
        print(f"ACCEPT  {label:44s} {len(data):5d} bytes  profile={meta.get('profile')}")
    except Exception as exc:
        print(f"REJECT  {label:44s} {type(exc).__name__}: {exc}")
