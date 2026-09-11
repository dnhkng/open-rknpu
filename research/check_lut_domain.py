"""Probe how the LUT argument relates to the Conv task's declared output scale.

The verified profile samples the table as `fn((i-512)/64)` and declares the stem
output scale as 1/2048. Three mechanisms are consistent with that single data
point:

* `float`  - the LUT sees the dequantized stem value `x = r*s_decl`, so the
  declared scale cancels and the table is a plain sampling of `fn`;
* `raw`    - the LUT sees the wide requantized code `r` with a fixed conversion,
  so the table only works for the one declared scale it was built for;
* `acc`    - the LUT sees the raw accumulator (no requantization at all).

Changing only the declared scale separates them: `float` leaves the output
unchanged, `raw` shifts the table argument, `acc` is also unchanged but predicts
a different table argument when the stem weights change magnitude.

Development experiment only: the public emitter never reads these results at run
time. Board runner: /tmp/board_run.
"""
from pathlib import Path
import json
import struct
import sys
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.lut import compile_lut

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "lut_domain_probe"
OUT.mkdir(exist_ok=True)
FACTOR = 1 / 32
VARIANTS = [
    ("float_baseline", 1 / 2048, 0),
    ("half_scale", 1 / 1024, 0),
    ("quarter_scale", 1 / 4096, 0),
    ("zero_point32", 1 / 2048, 32),
]


def model_for(factor):
    w = np.eye(3, dtype=np.float32).reshape(3, 3, 1, 1) * factor
    b = np.zeros(3, np.float32)
    model = h.make_model(h.make_graph(
        [h.make_node("Conv", ["input", "w", "b"], ["conv"], kernel_shape=[1, 1]),
         h.make_node("Sigmoid", ["conv"], ["output"])], "lut_domain",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
        [nh.from_array(w, "w"), nh.from_array(b, "b")]), opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def table_value(index):
    return np.clip(np.rint((1 / (1 + np.exp(-((index - 512) / 64)))) * 32768), -32768, 32767).astype(np.int64)


def candidates(pixels, declared_scale):
    """Candidate outputs for the three mechanisms, for one pixel column."""
    from open_rknpu.quantization import quantize
    w = np.eye(3, dtype=np.float32).reshape(3, 3, 1, 1) * FACTOR
    b = np.zeros(3, np.float32)
    out = {}
    # float: table argument is the true dequantized stem value
    x = (pixels.astype(np.float64) - 128) * FACTOR
    idx = np.clip(np.rint(512 + 64 * x), 0, 1024).astype(int)
    out["float"] = (np.rint(table_value(idx) * 255 / 32768) - 128).astype(np.int8)
    # raw / acc: recompute the wide code with this declared scale and index it
    q = quantize(w, b, output_scale=declared_scale, output_zero_point=0,
                 input_scale=1.0, input_zero_point=128)
    centered = q.weights.reshape(3, -1) - q.weight_zero_points[:, None]
    acc = centered[:, 0:1] * (pixels.astype(np.int64) - 128)[None, :] + q.biases[:, None]
    product = acc * q.channel_multipliers[:, None]
    scaled = (product + 8191 + ((product >> 14) & 1)) >> 14
    code = (scaled * q.multiplier + (1 << (q.shift - 1)) - 1 + ((scaled * q.multiplier >> q.shift) & 1)) >> q.shift
    for name, scale in (("raw", 1 / 2048), ("declared", declared_scale)):
        idx = np.clip(np.rint(512 + 64 * code * scale), 0, 1024).astype(int)
        out[name] = (np.rint(table_value(idx) * 255 / 32768) - 128).astype(np.int8)
    idx = np.clip(np.rint(512 + 64 * acc / 8160), 0, 1024).astype(int)
    out["acc"] = (np.rint(table_value(idx) * 255 / 32768) - 128).astype(np.int8)
    return out


manifest = []
for index, (name, scale, zero) in enumerate(VARIANTS):
    model = model_for(FACTOR)
    binary, meta = compile_lut(model, 1, 128, declared_scale=scale, declared_zero_point=zero)
    (OUT / f"model{index:03}.bin").write_bytes(binary)
    # 64 pixels cover 0..252; a second batch covers the odd values.
    pixels = np.zeros((2, 8, 8, 3), np.uint8)
    ramp = np.arange(64) * 4
    pixels[0, :, :, :] = ramp.reshape(8, 8, 1)
    pixels[1, :, :, :] = (255 - np.arange(64) * 4).reshape(8, 8, 1)
    # one pixel per run simplifies reading the board output; keep the batch flat
    (OUT / f"input{index:03}.u8").write_bytes(pixels.tobytes())
    prediction = candidates(np.arange(256), scale)
    manifest.append({"index": index, "variant": name, "declared_scale": scale,
                     "declared_zero_point": zero, "meta": meta,
                     "predictions": {k: np.asarray(val).reshape(-1)[::32].astype(int).tolist() for k, val in prediction.items()}})
    print(index, name, "scale", scale, "zero", zero, flush=True)
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("built", len(manifest), "variants in", OUT)
