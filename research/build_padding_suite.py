"""Leading-Pad host-preprocessing suite (constant/reflect/edge/wrap).

The compiler folds a leading Pad into the graph input shape and records the
required preprocessing; the board expects padded input bytes produced by
`open_rknpu.padding.pad_input`. Expected bytes come from the native input
reference. Board runner: tests/board_api.c.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.padding import pad_input
from open_rknpu.quantization import Quantization
from open_rknpu.native import native_input_reference

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/padding_suite")
parser.add_argument("--cases", type=int, default=16)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(110410)

def quant(params):
    values = dict(params)
    for key in ("weights", "weight_zero_points", "weight_scales", "biases", "channel_multipliers"):
        values[key] = np.array(values[key])
    return Quantization(**values)

CONFIGS = [(mode, pads) for mode in ("constant", "reflect", "edge", "wrap")
           for pads in ((1, 1, 1, 1), (2, 2, 2, 2), (0, 1, 2, 1))]
manifest = []
for index, (mode, pads) in enumerate(CONFIGS):
    height = width = 8
    padded_h = height + pads[0] + pads[2]; padded_w = width + pads[1] + pads[3]
    w = rng.uniform(-.7, .8, (3, 3, 3, 3)).astype(np.float32)
    b = rng.uniform(-2, 2, (3,)).astype(np.float32)
    onnx_mode = {"wrap": "wrap", "edge": "edge", "reflect": "reflect", "constant": "constant"}[mode]
    pad_node = h.make_node("Pad", ["input", "pads", "value"], ["padded"], mode=onnx_mode)
    nodes = [pad_node, h.make_node("Conv", ["padded", "w", "b"], ["output"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])]
    graph = h.make_graph(nodes, "padding",
        [h.make_tensor_value_info("input", 1, [1, 3, height, width])],
        [h.make_tensor_value_info("output", 1, [1, 3, padded_h, padded_w])],
        [nh.from_array(w, "w"), nh.from_array(b, "b"),
         nh.from_array(np.array([0, 0, pads[0], pads[1], 0, 0, pads[2], pads[3]], np.int64), "pads"),
         nh.from_array(np.array(0, np.float32), "value")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)])
    model.ir_version = 8
    path = root / f"model{index:03}.onnx"; onnx.save(model, path)
    binary, meta = compile_sequence(path)
    (root / f"model{index:03}.bin").write_bytes(binary)
    assert meta["input_padding"]["mode"] == mode and meta["input_padding"]["pads"] == list(pads)
    q = quant(meta["quantization"])
    raw = rng.integers(0, 256, (args.cases, height, width, 3), dtype=np.uint8)
    raw[0] = 0; raw[1] = 255; raw[2] = 128
    padded = np.stack([pad_input(x, pads, mode, constant_values=0) for x in raw])
    padded.tofile(root / f"input{index:03}.u8")
    outputs = np.stack([native_input_reference(x, q, meta["input_zero_point"], pads=meta["conv_pads"],
                                               strides=tuple(meta["conv_strides"]),
                                               dilations=tuple(meta["conv_dilations"])) for x in padded])
    outputs.tofile(root / f"expected{index:03}.i8")
    manifest.append({"index": index, "mode": mode, "pads": list(pads),
                     "padded_shape": meta["input_padding"]["padded_shape"], "cases": args.cases})
    print(manifest[-1], flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} leading-Pad models in {root}")
