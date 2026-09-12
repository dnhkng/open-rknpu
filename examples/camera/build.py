"""SPDX-License-Identifier: MIT

Checklist row E5: build the deterministic artifacts for the camera/V4L2 -> NPU example.

The graph is the smallest camera-shaped CNN that stays inside the verified envelope
(`docs/support-matrix.md` and `docs/primitives.md`), an RGB 32x32 stem with one pooled
1x1 classifier head:

    input [1,3,32,32] UINT8
      -> Conv 3x3, 3->8, pads 1, stride 1   (native16-input first stage)
      -> Relu                                (folded into the Conv task)
      -> MaxPool 2x2, stride 2               (16x16)
      -> Conv 1x1, 8->4                      (classifier head)

The first Conv reads a 32x32 image, which is larger than the legacy 5..8 image profile, so
`parse_chain` marks it `native_input` and the walk lowers it through the board-verified
native16 image emitter (`research/native_input_suite/`, `research/walk_chain_suite/`). The
whole graph compiles as the `chain-walk` profile, three tasks, format v5.

The frame is a deterministic test scene stored in the three capture formats the harness
supports - YUYV 4:2:2, NV12 4:2:0 and RGB888 - each at 64x48, well above the 32x32 model
input so the nearest-neighbour downsample is exercised. `input.u8` is the `convert.py`
conversion of the NV12 frame (the format `rkisp_mainpath` produces), and `expected.i8` is
the profile's own integer reference - `open_rknpu.walk.chain_walk_reference` over the
emitter's recorded bands and the `parse_chain` op list of the *normalized* graph - applied
to exactly those bytes. `report.json` records the geometry, the bands, the frame formats
and the conversion rule.

Deterministic, host-only, no board and no network. Run from the repository root:

    PYTHONPATH=src python examples/camera/build.py
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper as h, numpy_helper as nh

from open_rknpu.normalize import normalize_model
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence
from open_rknpu.walk import chain_walk_reference, load_quantizations, parse_chain

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "build"

# Model geometry. 32x32 keeps the image inside the 6144-atom native16 tiling bound and
# makes the nearest-neighbour downsample from 64x48 non-trivial in both axes.
MODEL_HEIGHT = 32
MODEL_WIDTH = 32
HIDDEN = 8
CLASSES = 4
INPUT_SCALE = 1.0
INPUT_ZERO_POINT = 0

# Deterministic synthetic capture. 64x48 NV12 is 4,608 bytes; the scene is a generated
# test chart (gradients plus a saturated colour-bar strip), not random noise, so the same
# bytes are easy to describe and the conversion's clipping is exercised.
FRAME_WIDTH = 64
FRAME_HEIGHT = 48
BAR_ROWS = 8
BAR_COLORS = ((0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0),
              (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255))
BAR_COUNT = len(BAR_COLORS)

GRAPH = [
    "input [1,3,32,32] UINT8, scale 1.0 / zero point 0",
    "Conv 3x3, 3->8, pads 1, stride 1, group 1, fused Relu",
    "MaxPool 2x2, stride 2, pads 0",
    "Conv 1x1, 8->4, pads 0, stride 1, group 1",
]

FLOAT = TensorProto.FLOAT


def _load_convert():
    """Import `convert.py` by path so the reference conversion is the same code the CLI runs."""
    spec = importlib.util.spec_from_file_location("camera_convert", HERE / "convert.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONVERT = _load_convert()


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
def _initializer(name, array):
    return nh.from_array(np.asarray(array, np.float32), name)


def build_model():
    """The deterministic ONNX graph (float32 weights, seed fixed)."""
    rng = np.random.default_rng(5051)
    stem_w = rng.uniform(-0.3, 0.3, (HIDDEN, 3, 3, 3)).astype(np.float32)
    stem_b = rng.uniform(-1.0, 1.0, HIDDEN).astype(np.float32)
    head_w = rng.uniform(-0.3, 0.3, (CLASSES, HIDDEN, 1, 1)).astype(np.float32)
    head_b = rng.uniform(-1.0, 1.0, CLASSES).astype(np.float32)
    nodes = [
        h.make_node("Conv", ["input", "stem_w", "stem_b"], ["stem"], kernel_shape=[3, 3],
                    pads=[1, 1, 1, 1], strides=[1, 1], group=1),
        h.make_node("Relu", ["stem"], ["stem_relu"]),
        h.make_node("MaxPool", ["stem_relu"], ["pool"], kernel_shape=[2, 2], strides=[2, 2],
                    pads=[0, 0, 0, 0]),
        h.make_node("Conv", ["pool", "head_w", "head_b"], ["output"], kernel_shape=[1, 1],
                    pads=[0, 0, 0, 0], strides=[1, 1], group=1),
    ]
    inputs = [h.make_tensor_value_info("input", FLOAT, [1, 3, MODEL_HEIGHT, MODEL_WIDTH])]
    outputs = [h.make_tensor_value_info(
        "output", FLOAT, [1, CLASSES, MODEL_HEIGHT // 2, MODEL_WIDTH // 2])]
    constants = [_initializer("stem_w", stem_w), _initializer("stem_b", stem_b),
                 _initializer("head_w", head_w), _initializer("head_b", head_b)]
    graph = h.make_graph(nodes, "camera_e5", inputs, outputs, constants)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def compile_model(model):
    """Compile with `compile_sequence`; must land on the `chain-walk` profile."""
    binary, meta = compile_sequence(model, input_scale=INPUT_SCALE,
                                    input_zero_point=INPUT_ZERO_POINT)
    if meta.get("profile") != "chain-walk":
        raise ValueError("the camera graph compiled as %r, expected chain-walk"
                         % meta.get("profile"))
    return binary, meta


def integer_reference(model, meta, packed):
    """The profile's own integer reference for one packed NHWC UINT8 input."""
    spec = parse_chain(normalize_model(model).graph)
    if spec.get("error"):
        raise ValueError(spec["error"])
    inputs = np.frombuffer(packed, dtype=np.uint8).reshape(MODEL_HEIGHT, MODEL_WIDTH, 3)
    return chain_walk_reference(inputs, load_quantizations(meta), spec["ops"],
                                INPUT_ZERO_POINT)


# --------------------------------------------------------------------------- #
# The synthetic capture, encoded into the three formats
# --------------------------------------------------------------------------- #
def source_scene(width, height):
    """A deterministic RGB test chart: three integer ramps plus a saturated bar strip."""
    x = np.arange(width, dtype=np.int64)[None, :]
    y = np.arange(height, dtype=np.int64)[:, None]
    scene = np.empty((height, width, 3), np.uint8)
    scene[..., 0] = (x * 255 // max(width - 1, 1)).astype(np.uint8)
    scene[..., 1] = (y * 255 // max(height - 1, 1)).astype(np.uint8)
    scene[..., 2] = ((x + y) * 255 // max(width + height - 2, 1)).astype(np.uint8)
    bar = width // BAR_COUNT
    for index, color in enumerate(BAR_COLORS):
        row = slice(max(height - BAR_ROWS, 0), height)
        column = slice(index * bar, (index + 1) * bar if index + 1 < BAR_COUNT else width)
        scene[row, column] = color
    return scene


def _luma(r, g, b):
    return np.clip((77 * r + 150 * g + 29 * b + 128) >> 8, 0, 255).astype(np.uint8)


def _chroma_blue(r, g, b):
    return np.clip(((-43 * r - 85 * g + 128 * b + 128) >> 8) + 128, 0, 255).astype(np.uint8)


def _chroma_red(r, g, b):
    return np.clip(((128 * r - 107 * g - 21 * b + 128) >> 8) + 128, 0, 255).astype(np.uint8)


def encode_yuyv(scene, width, height):
    """Encode an RGB888 scene as YUYV 4:2:2 (`Y0 U Y1 V` per pixel pair)."""
    r = scene[..., 0].astype(np.int64)
    g = scene[..., 1].astype(np.int64)
    b = scene[..., 2].astype(np.int64)
    luma = _luma(r, g, b)
    blue = _chroma_blue(r, g, b).astype(np.int64)
    red = _chroma_red(r, g, b).astype(np.int64)
    buffer = np.empty((height, width, 2), np.uint8)
    buffer[:, 0::2, 0] = luma[:, 0::2]
    buffer[:, 0::2, 1] = ((blue[:, 0::2] + blue[:, 1::2] + 1) // 2).astype(np.uint8)
    buffer[:, 1::2, 0] = luma[:, 1::2]
    buffer[:, 1::2, 1] = ((red[:, 0::2] + red[:, 1::2] + 1) // 2).astype(np.uint8)
    return buffer.tobytes()


def encode_nv12(scene, width, height):
    """Encode an RGB888 scene as NV12 4:2:0 (Y plane, then interleaved half-size U/V)."""
    r = scene[..., 0].astype(np.int64)
    g = scene[..., 1].astype(np.int64)
    b = scene[..., 2].astype(np.int64)
    luma = _luma(r, g, b)
    blue = _chroma_blue(r, g, b).astype(np.int64)
    red = _chroma_red(r, g, b).astype(np.int64)
    blocks = blue.reshape(height // 2, 2, width // 2, 2)
    blue_blocks = ((blocks.sum(axis=(1, 3)) + 2) // 4).astype(np.uint8)
    blocks = red.reshape(height // 2, 2, width // 2, 2)
    red_blocks = ((blocks.sum(axis=(1, 3)) + 2) // 4).astype(np.uint8)
    chroma = np.empty((height // 2, width // 2, 2), np.uint8)
    chroma[..., 0] = blue_blocks
    chroma[..., 1] = red_blocks
    return luma.tobytes() + chroma.tobytes()


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def describe_bands(meta):
    """The per-op band table: every emitter quantization plus the pools that preserve it."""
    bands = []
    for index, (kind, quantization) in enumerate(zip(meta["walk_ops"], meta["quantizations"])):
        if kind == "conv":
            weights = np.asarray(quantization["weights"])
            bands.append({
                "index": index,
                "op": "conv",
                "kernel_size": int(quantization["kernel_size"]),
                "weights_per_output": int(weights.shape[1]),
                "output_channels": int(weights.shape[0]),
                "input_scale": float(quantization["input_scale"]),
                "input_zero_point": int(quantization["input_zero_point"]),
                "output_scale": float(quantization["output_scale"]),
                "output_zero_point": int(quantization["output_zero_point"]),
                "relu": bool(quantization["relu"]),
                "multiplier": int(quantization["multiplier"]),
                "shift": int(quantization["shift"]),
            })
        else:
            bands.append({"index": index, "op": kind, "band": "preserves the input grid"})
    return bands


def write_artifacts(out):
    """Build every artifact into `out` and return the report dictionary."""
    out.mkdir(parents=True, exist_ok=True)
    model = build_model()
    binary, meta = compile_model(model)
    info = decode_sequence(binary)

    scene = source_scene(FRAME_WIDTH, FRAME_HEIGHT)
    frames = {
        "yuyv": encode_yuyv(scene, FRAME_WIDTH, FRAME_HEIGHT),
        "nv12": encode_nv12(scene, FRAME_WIDTH, FRAME_HEIGHT),
        "rgb": scene.tobytes(),
    }
    for fmt, data in frames.items():
        expected = CONVERT.frame_bytes(fmt, FRAME_WIDTH, FRAME_HEIGHT)
        if len(data) != expected:
            raise ValueError("encoded %s frame is %d bytes, expected %d"
                             % (fmt, len(data), expected))

    input_bytes = CONVERT.convert_frame(frames["nv12"], "nv12", FRAME_WIDTH, FRAME_HEIGHT,
                                        MODEL_WIDTH, MODEL_HEIGHT)
    expected_output = integer_reference(model, meta, input_bytes)

    artifacts = {"model.bin": binary}
    for fmt, data in frames.items():
        artifacts["frame.%s" % fmt] = data
    artifacts["input.u8"] = input_bytes
    artifacts["expected.i8"] = expected_output.tobytes()
    for name, data in artifacts.items():
        (out / name).write_bytes(data)

    report = {
        "profile": meta["profile"],
        "format_version": info["format_version"],
        "task_count": info["task_count"],
        "graph": GRAPH,
        "model": {
            "input_shape_nhwc": [1, MODEL_HEIGHT, MODEL_WIDTH, 3],
            "output_shape_nhwc": [1, MODEL_HEIGHT // 2, MODEL_WIDTH // 2, CLASSES],
            "input_bytes": int(info["input_bytes"]),
            "output_bytes": int(info["output_bytes"]),
            "arena_bytes": int(info["arena_bytes"]),
            "input_scale": INPUT_SCALE,
            "input_zero_point": INPUT_ZERO_POINT,
            "output_scale": float(meta["output_scale"]),
            "output_zero_point": int(meta["output_zero_point"]),
        },
        "bands": describe_bands(meta),
        "frame": {
            "width": FRAME_WIDTH,
            "height": FRAME_HEIGHT,
            "scene": ("deterministic test chart: R horizontal ramp, G vertical ramp, B diagonal "
                      "ramp, plus a %d-column saturated bar strip in the bottom %d rows"
                      % (BAR_COUNT, BAR_ROWS)),
            "formats": {fmt: {"bytes": len(data), "sha256": _sha256(data)}
                        for fmt, data in frames.items()},
            "primary": "frame.nv12",
            "primary_reason": "NV12 is what rkisp_mainpath produces on the reference board",
        },
        "conversion": CONVERT.conversion_rule(),
        "reference": "open_rknpu.walk.chain_walk_reference on input.u8",
        "artifacts": {name: {"bytes": len(data), "sha256": _sha256(data)}
                      for name, data in sorted(artifacts.items())},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the E5 camera example artifacts.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output directory (default: examples/camera/build)")
    args = parser.parse_args(argv)
    report = write_artifacts(args.out)
    model = report["model"]
    print("built %s: profile=%s tasks=%d %s -> %s"
          % (args.out, report["profile"], report["task_count"],
             "x".join(str(v) for v in model["input_shape_nhwc"][1:]),
             "x".join(str(v) for v in model["output_shape_nhwc"][1:])))
    for name in sorted(report["artifacts"]):
        entry = report["artifacts"][name]
        print("  %-14s %7d bytes  %s" % (name, entry["bytes"], entry["sha256"]))
    print("frame %dx%d  input.u8 from %s  expected_bytes=%d"
          % (report["frame"]["width"], report["frame"]["height"], report["frame"]["primary"],
             model["output_bytes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
