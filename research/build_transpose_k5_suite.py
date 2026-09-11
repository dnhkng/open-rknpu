"""Bounded depthwise K5 ConvTranspose suite (vendor-derived emission).

K5 uses the layout recovered from `capture_transpose_k5`: 25 taps x 32 bytes with
per-lane `(value, -weight_zero_point)` pairs, asymmetric per-channel weight
quantization, and the vendor's phase/feature register fields. Expected bytes come
from the documented integer reference (scatter of the centered stem output with
centered weights, then the two-stage requantization). Board runner: tests/board_api.c.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.quantization import Quantization, reference
from open_rknpu.scheduler import compile_sequence

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/transpose_k5_suite")
parser.add_argument("--cases", type=int, default=16)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(110900)

CASES = [
    ("vendor", [2, 2], [2, 2, 2, 2], [0, 0], 3),
    ("stride1", [1, 1], [2, 2, 2, 2], [0, 0], 3),
    ("pad1", [2, 2], [1, 1, 1, 1], [0, 0], 3),
    ("pad0", [2, 2], [0, 0, 0, 0], [0, 0], 3),
    ("output_padding", [2, 2], [2, 2, 2, 2], [1, 1], 3),
    ("rectangular", [1, 2], [4, 0, 1, 3], [0, 1], 3),
    ("c1", [2, 2], [2, 2, 2, 2], [0, 0], 1),
    ("c4", [2, 2], [2, 2, 2, 2], [0, 0], 4),
    ("c8", [2, 2], [2, 2, 2, 2], [0, 0], 8),
]


def transposed_reference(sample, q1, q, weights, strides, pads, output_padding):
    """Integer reference for the depthwise transposed task."""
    kh, kw = weights.shape[2], weights.shape[3]
    oh = 7 * strides[0] + kh - pads[0] - pads[2] + output_padding[0]
    ow = 7 * strides[1] + kw - pads[1] - pads[3] + output_padding[1]
    channels = weights.shape[0]
    a = reference(sample, q1).astype(np.int64) - q1.output_zero_point
    wq = np.asarray(q.weights).reshape(channels, kh, kw)
    centered = wq - np.asarray(q.weight_zero_points, np.int64)[:, None, None]
    base = np.asarray(q.biases, np.int64) + q1.output_zero_point * centered.sum(axis=(1, 2))
    acc = np.broadcast_to(base, (oh, ow, channels)).copy()
    for iy in range(8):
        for ix in range(8):
            for ky in range(kh):
                for kx in range(kw):
                    oy = iy * strides[0] + ky - pads[0]
                    ox = ix * strides[1] + kx - pads[1]
                    if 0 <= oy < oh and 0 <= ox < ow:
                        acc[oy, ox] += a[iy, ix] * centered[:, ky, kx]
    product = acc * q.channel_multipliers
    scaled = (product + 8191 + ((product >> 14) & 1)) >> 14
    product = scaled * q.multiplier
    if q.shift:
        product = product + (1 << (q.shift - 1)) - 1 + ((product >> q.shift) & 1)
        result = (product >> q.shift) + q.output_zero_point
    else:
        result = product + q.output_zero_point
    return np.clip(result, -128, 127).astype(np.int8)


manifest = []
for index, (name, strides, pads, output_padding, channels) in enumerate(CASES):
    # Independent graph: an exact 1x1 Conv stem followed by the K5 depthwise
    # ConvTranspose. Weight zero points stay asymmetric like the vendor capture.
    kernel = 3 if channels <= 4 else 1
    pad = kernel // 2
    stem_w = rng.uniform(-.3, .3, (channels, 3, kernel, kernel)).astype(np.float32)
    stem_b = rng.uniform(-1, 1, channels).astype(np.float32)
    weights = rng.uniform(-.25, .25, (channels, 1, 5, 5)).astype(np.float32)
    bias = rng.uniform(-.5, .5, channels).astype(np.float32)
    oh = 7 * strides[0] + 5 - pads[0] - pads[2] + output_padding[0]
    ow = 7 * strides[1] + 5 - pads[1] - pads[3] + output_padding[1]
    nodes = [h.make_node("Conv", ["input", "stem_w", "stem_b"], ["stem"], kernel_shape=[kernel, kernel],
                         pads=[pad] * 4),
             h.make_node("ConvTranspose", ["stem", "w", "b"], ["output"], group=channels,
                         kernel_shape=[5, 5], strides=strides, pads=pads, output_padding=output_padding)]
    graph = h.make_graph(nodes, "transpose_k5",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, channels, oh, ow])],
        [nh.from_array(stem_w, "stem_w"), nh.from_array(stem_b, "stem_b"),
         nh.from_array(weights, "w"), nh.from_array(bias, "b")])
    graph.value_info.append(h.make_tensor_value_info("stem", 1, [1, channels, 8, 8]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    model = onnx.shape_inference.infer_shapes(model)
    path = root / f"model{index:03}.onnx"
    onnx.save(model, path)
    binary, meta = compile_sequence(path)
    path.with_suffix(".bin").write_bytes(binary)
    q = Quantization(**{k: np.array(v) if isinstance(v, list) else v
                        for k, v in meta["transposed_quantization"].items()})
    q1 = Quantization(**{k: np.array(v) if isinstance(v, list) else v
                         for k, v in meta["first"]["quantization"].items()})
    cases = rng.integers(0, 256, (args.cases, 8, 8, 3), dtype=np.uint8)
    cases[0] = 0
    cases[1] = 255
    cases[2] = 128
    cases.tofile(root / f"input{index:03}.u8")
    expected = np.stack([transposed_reference(sample, q1, q, weights, strides, pads, output_padding)
                         for sample in cases])
    expected.tofile(root / f"expected{index:03}.i8")
    manifest.append({"index": index, "name": name, "kernel": 5, "strides": strides, "pads": pads,
                     "output_padding": output_padding, "channels": channels, "cases": args.cases,
                     "stem_kernel": kernel, "output_shape": [oh, ow, channels],
                     "profile": meta["transposed_profile"],
                     "weight_zero_points": [int(v) for v in q.weight_zero_points],
                     "output_scale": meta["output_scale"], "output_zero_point": meta["output_zero_point"]})
    print(index, name, "range", [oh, ow, channels], "wzp", manifest[-1]["weight_zero_points"], flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} K5 models in {root}")
