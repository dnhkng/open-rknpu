"""Asymmetric depthwise weight-pair experiment.

Stores (offset weight code, per-channel weight zero point) in the depthwise
weight pair and uses asymmetric per-channel quantization, keeping register
0x4054 at its verified symmetric value. Expected bytes come from
`depthwise_reference`, which already subtracts the per-channel weight zero point.
Board runner: tests/board_api.c.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.depthwise import depthwise_reference
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization, reference

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/depthwise_asymmetric_suite")
parser.add_argument("--cases", type=int, default=16)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(110400)

def quant(params):
    values = dict(params)
    for key in ("weights", "weight_zero_points", "weight_scales", "biases", "channel_multipliers"):
        values[key] = np.array(values[key])
    return Quantization(**values)

CONFIGS = []
for channels in (1, 3, 4):
    for kernel in (1, 3, 5):
        for sign in ("positive", "negative", "mixed_zero"):
            CONFIGS.append((channels, kernel, sign))

manifest = []
for index, (channels, kernel, sign) in enumerate(CONFIGS):
    stem_kernel = 1
    w1 = rng.uniform(-.2, .2, (channels, 3, stem_kernel, stem_kernel)).astype(np.float32)
    b1 = rng.uniform(-2, 2, channels).astype(np.float32)
    if sign == "positive":
        w2 = np.abs(rng.uniform(.05, .8, (channels, 1, kernel, kernel))).astype(np.float32)
    elif sign == "negative":
        w2 = (-np.abs(rng.uniform(.05, .8, (channels, 1, kernel, kernel)))).astype(np.float32)
    else:
        w2 = rng.uniform(-.8, .8, (channels, 1, kernel, kernel)).astype(np.float32)
        w2.reshape(channels, -1)[0, 0] = 0.0
    b2 = rng.uniform(-3, 3, channels).astype(np.float32)
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["conv"], kernel_shape=[stem_kernel, stem_kernel])]
    nodes.append(h.make_node("Conv", ["conv", "w2", "b2"], ["output"], kernel_shape=[kernel, kernel],
                             pads=[kernel // 2] * 4, group=channels))
    graph = h.make_graph(nodes, "depthwise_asymmetric",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, channels, 8, 8])],
        [nh.from_array(w1, "w1"), nh.from_array(b1, "b1"), nh.from_array(w2, "w2"), nh.from_array(b2, "b2")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    model = onnx.shape_inference.infer_shapes(model)
    path = root / f"model{index:03}.onnx"
    onnx.save(model, path)
    binary, meta = compile_sequence(path, asymmetric_depthwise=True)
    (root / f"model{index:03}.bin").write_bytes(binary)
    q1 = quant(meta["first"]["quantization"]); q2 = quant(meta["depthwise"])
    cases = rng.integers(0, 256, (args.cases, 8, 8, 3), dtype=np.uint8)
    cases[0] = 0; cases[1] = 255; cases[2] = 128
    cases.tofile(root / f"input{index:03}.u8")
    outputs = np.stack([depthwise_reference(reference(x, q1), q2, q1.output_zero_point) for x in cases])
    outputs.tofile(root / f"expected{index:03}.i8")
    manifest.append({"index": index, "channels": channels, "kernel": kernel, "sign": sign,
                     "weight_zero_points": [int(v) for v in q2.weight_zero_points],
                     "asymmetric_pair": meta["depthwise_asymmetric_pair"], "cases": args.cases})
    print(manifest[-1], flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} asymmetric depthwise models in {root}")
