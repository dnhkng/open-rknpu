"""Per-channel Mul quantization: immutable per-channel constant, per-channel scales.

The verified EW Mul task carries one output multiplier/shift, so its operand
stream stores a single constant scale ([.02,.35,1.9] -> [1,23,127] bytes). This
suite lowers the same arithmetic (`Mul(input, [C,1,1])`) onto the verified 1x1
depthwise Conv profile, whose native per-output-channel weight scale quantizes
each channel of the constant on its own grid. Bias is zero, so the semantics are
unchanged; the output grid stays one per-tensor scale.

Each model is compiled twice: the shared-scale EW constant Mul and the
per-channel depthwise lowering. Expected bytes come from `depthwise_reference`
over the identity stem, and the manifest records the float-domain error of both
paths so the per-channel win is measurable. Board runner: tests/board_api.c.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.depthwise import depthwise_reference
from open_rknpu.elementwise import mul_reference
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization, reference

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/per_channel_mul_suite")
parser.add_argument("--cases", type=int, default=16)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(110500)


def quant(params):
    values = dict(params)
    for key in ("weights", "weight_zero_points", "weight_scales", "biases", "channel_multipliers"):
        values[key] = np.array(values[key])
    return Quantization(**values)


PATTERNS = [
    ("mild", [1.2, 1.4, 1.6]),
    ("wide", [0.02, 0.35, 1.9]),
    ("extreme", [0.005, 0.3, 2.4]),
    ("mixed_sign", [-0.4, 0.9, -2.1]),
    ("small_all", [0.01, 0.02, 0.03]),
    ("zero_channel", [0.0, 0.5, -1.5]),
]
GEOMETRIES = [(5, 5), (5, 8), (6, 7), (8, 8), (7, 5), (6, 6)]

manifest = []
for index in range(12):
    name, values = PATTERNS[index % len(PATTERNS)]
    height, width = GEOMETRIES[index % len(GEOMETRIES)]
    # Jitter each channel so every model is an independently generated command set
    # while keeping the pattern's magnitude class.
    factor = np.asarray(values, np.float32) * rng.uniform(0.9, 1.1, 3).astype(np.float32)
    if name == "zero_channel":
        factor[0] = 0.0
    factor = factor.astype(np.float32)
    model = h.make_model(h.make_graph(
        [h.make_node("Mul", ["input", "factor"], ["output"])], "per_channel_mul",
        [h.make_tensor_value_info("input", 1, [1, 3, height, width])],
        [h.make_tensor_value_info("output", 1, [1, 3, height, width])],
        [nh.from_array(factor.reshape(3, 1, 1), "factor")]), opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    model = onnx.shape_inference.infer_shapes(model)
    path = root / f"model{index:03}.onnx"
    onnx.save(model, path)

    binary, meta = compile_sequence(path, per_channel_mul=True)
    (root / f"model{index:03}.bin").write_bytes(binary)
    q1 = quant(meta["first"]["quantization"])
    q2 = quant(meta["depthwise"])

    cases = rng.integers(0, 256, (args.cases, height, width, 3), dtype=np.uint8)
    cases[0] = 0
    cases[1] = 255
    cases[2] = 128
    cases.tofile(root / f"input{index:03}.u8")
    outputs = np.stack([depthwise_reference(reference(x, q1), q2, q1.output_zero_point) for x in cases])
    outputs.tofile(root / f"expected{index:03}.i8")

    # Same function through the shared-scale EW constant Mul, for comparison only.
    shared, shared_meta = compile_sequence(path)
    qs = quant(shared_meta["branches"][0]["quantization"])
    operand = np.clip(np.rint(np.broadcast_to(factor.reshape(3, 1, 1), (1, 3, height, width))
                             / shared_meta["constant_scale"]), -128, 127).astype(np.int8)[0].transpose(1, 2, 0)
    ew = np.stack([mul_reference(reference(x, qs), operand) for x in cases]).astype(np.float64)
    ew = (ew - shared_meta["output_zero_point"]) * (128 * qs.output_scale * shared_meta["constant_scale"])
    per_channel = outputs.astype(np.float64)
    per_channel = (per_channel - q2.output_zero_point) * q2.output_scale
    exact = cases.astype(np.float64) * factor
    manifest.append({"index": index, "pattern": name, "geometry": [height, width],
                     "constant": [float(v) for v in factor], "cases": args.cases,
                     "shared_scale": float(shared_meta["constant_scale"]),
                     "shared_operand": [int(v) for v in np.asarray(operand)[0, 0, :]],
                     "per_channel_weight_scales": [float(v) for v in q2.weight_scales],
                     "shared_max_error": float(np.abs(ew - exact).max()),
                     "per_channel_max_error": float(np.abs(per_channel - exact).max()),
                     "output_scale": float(q2.output_scale), "output_zero_point": int(q2.output_zero_point),
                     "input_scale": float(q1.output_scale), "input_zero_point": int(q1.output_zero_point)})
    print(manifest[-1], flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} per-channel Mul models in {root}")
