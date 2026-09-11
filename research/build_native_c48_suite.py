"""Native input C33..48 suite (three 16-lane planes).

Independently generated commands; expected bytes come from the documented native
input integer reference. Board runner: tests/board_api.c.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization
from open_rknpu.native import native_input_reference

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/native_c48_suite")
parser.add_argument("--cases", type=int, default=16)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(110350)

def quant_from(params):
    params = dict(params)
    for key in ("weights", "weight_zero_points", "weight_scales", "biases", "channel_multipliers"):
        params[key] = np.array(params[key])
    return Quantization(**params)

manifest = []
index = 0
for ic in (33, 40, 48):
    for oc, kernel, height, width in ((1, 1, 6, 5), (8, 3, 6, 5), (16, 3, 8, 8), (3, 1, 8, 8)):
        w = rng.uniform(-.7, .8, (oc, ic, kernel, kernel)).astype(np.float32)
        b = rng.uniform(-2, 2, (oc,)).astype(np.float32)
        node = h.make_node("Conv", ["input", "w", "b"], ["output"],
                           kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4)
        graph = h.make_graph([node], "native_c48",
            [h.make_tensor_value_info("input", 1, [1, ic, height, width])],
            [h.make_tensor_value_info("output", 1, [1, oc, height, width])],
            [nh.from_array(w, "w"), nh.from_array(b, "b")])
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)]); model.ir_version = 8
        path = root / f"model{index:03}.onnx"; onnx.save(model, path)
        binary, meta = compile_sequence(path)
        (root / f"model{index:03}.bin").write_bytes(binary)
        q = quant_from(meta["quantization"])
        cases = rng.integers(0, 256, (args.cases, height, width, ic), dtype=np.uint8)
        cases[0] = 0; cases[1] = 255; cases[2] = 128
        cases.tofile(root / f"input{index:03}.u8")
        outputs = np.stack([native_input_reference(x, q, meta["input_zero_point"],
                                                   pads=meta["conv_pads"],
                                                   strides=tuple(meta["conv_strides"]),
                                                   dilations=tuple(meta["conv_dilations"]))
                            for x in cases])
        outputs.tofile(root / f"expected{index:03}.i8")
        manifest.append({"index": index, "input_channels": ic, "output_channels": oc,
                         "kernel": kernel, "height": height, "width": width, "cases": args.cases,
                         "output_bytes": int(outputs[0].size)})
        print(manifest[-1], flush=True)
        index += 1
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} native C33..48 models in {root}")
