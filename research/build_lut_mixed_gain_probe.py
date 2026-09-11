"""Measure the LUT negative-half gain for mixed-input stems.

For diagonal stems the gain is `max_weight_scale/BASE_WEIGHT_SCALE`; a mixed stem
that shares one `(max-min)` per channel measured 2.016 instead, so the rule is not
the per-channel weight scale. This probe emits mixed-stem LUT profiles with a known
table correction `C` and inverts the sigmoid on the board output to recover the
hardware gain `H = u_measured/(x*C)` per channel, then fits `H` against the stem's
weight statistics.

Development-only: the public emitter rejects mixed stems unless
`allow_mixed_stems=True` is passed explicitly.
"""
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.lut import compile_lut

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "lut_mixed_gain_probe"
OUT.mkdir(exist_ok=True)

# (name, weights[3,3], bias[3])
STEMS = [
    ("pair_20_5", [[0.02, 0.005, 0], [0.005, 0.02, 0], [0, 0.005, 0.02]], [0, 0, 0]),
    ("pair_20_10", [[0.02, 0.01, 0], [0.01, 0.02, 0], [0, 0.01, 0.02]], [0, 0, 0]),
    ("triple_10", [[0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.01, 0.01, 0.01]], [0, 0, 0]),
    ("pair_30_10", [[0.03, 0.01, 0], [0.01, 0.03, 0], [0, 0.01, 0.03]], [0, 0, 0]),
    ("pair_20_5_bias", [[0.02, 0.005, 0], [0.005, 0.02, 0], [0, 0.005, 0.02]], [0.8, -0.8, 0]),
    ("mixed_sign", [[0.02, -0.005, 0], [-0.005, 0.02, 0], [0, 0.005, -0.02]], [0, 0, 0]),
    ("skew_30_2_1", [[0.03, 0.002, 0.001], [0.001, 0.03, 0.002], [0.002, 0.001, 0.03]], [0, 0, 0]),
    ("diagonal_ref", [[0.02, 0, 0], [0, 0.02, 0], [0, 0, 0.02]], [0, 0, 0]),
]


def model_for(weights, bias):
    graph = h.make_graph(
        [h.make_node("Conv", ["input", "w", "b"], ["conv"], kernel_shape=[1, 1]),
         h.make_node("Sigmoid", ["conv"], ["output"])], "mixed",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
        [nh.from_array(np.asarray(weights, np.float32).reshape(3, 3, 1, 1), "w"),
         nh.from_array(np.asarray(bias, np.float32), "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def weights_of(weights):
    w = np.asarray(weights, np.float64)
    minimum = np.minimum(w.min(axis=1), 0)
    maximum = np.maximum(w.max(axis=1), 0)
    scales = (maximum - minimum) / 255
    return {"per_channel_scale": scales.tolist(),
            "max_scale": float(scales.max()),
            "abs_sum": np.abs(w).sum(axis=1).tolist(),
            "abs_max": np.abs(w).max(axis=1).tolist(),
            "nonzero": (np.abs(w) > 0).sum(axis=1).tolist()}


manifest = []
for index, (name, weights, bias) in enumerate(STEMS):
    # Emit with the correction the current rule would use; the probe recovers H.
    binary, meta = compile_lut(model_for(weights, bias), 1, 128,
                               allow_mixed_stems=True, negative_gain_override=1.0)
    (OUT / f"model{index:03}.bin").write_bytes(binary)
    pixels = np.arange(64, dtype=np.uint8) * 4
    image = np.repeat(pixels.reshape(8, 8, 1), 3, axis=2)
    (OUT / f"input{index:03}.u8").write_bytes(image.tobytes())
    manifest.append({"index": index, "name": name, "weights": weights, "bias": bias,
                     "statistics": weights_of(weights), "table_correction": 1.0,
                     "pixels": pixels.tolist()})
    print(index, name, manifest[-1]["statistics"], flush=True)
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("built", len(manifest), "mixed-stem probes in", OUT)
