# SPDX-License-Identifier: MIT
"""Asymmetric and one-sided Conv padding, plus rectangular kernels.

Independently generated single-Conv graphs whose ONNX `pads` are not the symmetric
`K//2` (and one `1xK` rectangular kernel). The geometry emitter already accepts these
shapes: `open_rknpu.normalize` square-embeds a rectangular kernel and materializes the
one-sided pads, and `open_rknpu.strided` re-emits the geometry, so the whole family
compiles. Expected bytes come from the native input profile's own documented integer
reference (`open_rknpu.native.native_input_reference`, with the emitted `conv_pads`);
`tests/test_bounded_ops.py` also pins an independent explicit-pad oracle. `pads >= K`
and `K > 31` stay rejected. Board runner: `tests/board_api.c`.
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
from open_rknpu.sequence import decode_sequence

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/rect_pad_suite")
parser.add_argument("--cases", type=int, default=16)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)

# (label, kernel, ONNX pads [begin_h, begin_w, end_h, end_w], activation)
GRAPHS = [
    ("K3-symmetric", (3, 3), (1, 1, 1, 1), None),
    ("K3-symmetric-relu", (3, 3), (1, 1, 1, 1), "Relu"),
    ("K3-one-sided-top", (3, 3), (0, 1, 0, 1), None),
    ("K3-one-sided-left-relu", (3, 3), (1, 0, 1, 0), "Relu"),
    ("K3-one-sided-bottom", (3, 3), (0, 0, 0, 1), None),
    ("K3-asymmetric-relu", (3, 3), (1, 1, 0, 0), "Relu"),
    ("K3-two-sided-top", (3, 3), (2, 0, 0, 0), None),
    ("K5-symmetric", (5, 5), (2, 2, 2, 2), None),
    ("K5-asymmetric-relu", (5, 5), (1, 0, 2, 3), "Relu"),
    ("K5-one-sided-top", (5, 5), (4, 0, 0, 0), None),
    ("K1xK3-rectangular", (1, 3), (0, 1, 0, 1), None),
    ("K1xK3-rectangular-relu", (1, 3), (0, 0, 0, 1), "Relu"),
]


def quant_from(params):
    values = dict(params)
    for key in ("weights", "weight_zero_points", "weight_scales", "biases",
                "channel_multipliers"):
        values[key] = np.array(values[key])
    return Quantization(**values)


def build(spec, seed):
    label, (kh, kw), pads, activation = spec
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-.8, .8, (3, 3, kh, kw)).astype(np.float32)
    bias = rng.uniform(-1, 1, 3).astype(np.float32)
    out_h = 8 + pads[0] + pads[2] - kh + 1
    out_w = 8 + pads[1] + pads[3] - kw + 1
    nodes = [h.make_node("Conv", ["input", "w", "b"], ["conv"],
                         kernel_shape=[kh, kw], pads=list(pads))]
    output = "conv"
    if activation is not None:
        nodes.append(h.make_node(activation, ["conv"], ["output"]))
        output = "output"
    graph = h.make_graph(
        nodes, label.replace("-", "_"),
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info(output, 1, [1, 3, out_h, out_w])],
        [nh.from_array(weights, "w"), nh.from_array(bias, "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model, out_h, out_w


manifest = []
for index, spec in enumerate(GRAPHS):
    label = spec[0]
    model, out_h, out_w = build(spec, 110560 + index)
    onnx.save(model, root / f"model{index:03}.onnx")
    binary, meta = compile_sequence(root / f"model{index:03}.onnx")
    route = "strided" if "conv_pads" in meta else "pooling-sequence"
    if route == "strided" and meta.get("profile") is not None:
        raise SystemExit(f"model{index:03} ({label}) routed to {meta.get('profile')!r}")
    (root / f"model{index:03}.bin").write_bytes(binary)
    info = decode_sequence(binary)
    if info["output_shape_nhwc"] != [1, out_h, out_w, 3]:
        raise SystemExit(f"model{index:03} ({label}) decoded to "
                         f"{info['output_shape_nhwc']}, not [1, {out_h}, {out_w}, 3]")
    rng = np.random.default_rng(110560 + index)
    cases = rng.integers(0, 256, (args.cases, 8, 8, 3), dtype=np.uint8)
    cases[0] = 0
    cases[1] = 255
    cases[2] = 128
    cases.tofile(root / f"input{index:03}.u8")
    quantization = quant_from(meta["quantization"])
    outputs = np.stack([native_input_reference(case, quantization, meta["input_zero_point"],
                                               pads=meta.get("conv_pads"),
                                               strides=tuple(meta.get("conv_strides", (1, 1))),
                                               dilations=tuple(meta.get("conv_dilations", (1, 1))))
                        for case in cases])
    outputs.tofile(root / f"expected{index:03}.i8")
    manifest.append(dict(index=index, label=label, kernel=list(spec[1]), pads=list(spec[2]),
                         relu=spec[3] == "Relu", input_channels=3, output_channels=3,
                         output_height=out_h, output_width=out_w,
                         effective_pads=list(meta.get("conv_pads")
                                             or (quantization.kernel_size // 2,) * 4),
                         route=route, input_shape=[1, 8, 8, 3],
                         output_shape=[1, out_h, out_w, 3], cases=args.cases,
                         output_bytes=int(outputs[0].size)))
    print(manifest[-1], flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} asymmetric-padding models in {root}")
