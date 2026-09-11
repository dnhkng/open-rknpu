"""General LUT stems: any C3 1x1 Conv whose analytic range fits the table domain.

The profile samples the 1026-entry table as `fn((index-512)/64)`, so the LUT
argument spans (-8, 8) in the stem's dequantized output units. Any 1x1 C3 stem
whose analytic output interval stays inside that window is accepted; outside it
the compiler raises instead of silently clipping. Expected bytes come from
`open_rknpu.lut.lut_reference`. Board runner: tests/board_api.c.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.lut import lut_reference, stem_range
from open_rknpu.scheduler import compile_sequence

parser = argparse.ArgumentParser()
parser.add_argument("--directory", default="research/lut_domain_suite")
parser.add_argument("--cases", type=int, default=16)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / args.directory
root.mkdir(parents=True, exist_ok=True)
for pattern in ("model*.onnx", "model*.bin", "input*.u8", "expected*.i8"):
    for stale in root.glob(pattern):
        stale.unlink()
rng = np.random.default_rng(110700)

IDENTITY = np.eye(3, dtype=np.float32)
# Every stem keeps one weight range per output channel (the negative table half
# advances with that scale) and an analytic range inside the table domain.
# The accepted family: a diagonal 1x1 Conv with one scalar gain `g` for every
# channel (sign flips allowed) and any bias, with the analytic range inside the
# sign-split table domain. Mixed input weights stay a retained blocker.
STEMS = {
    "Sigmoid": [
        (IDENTITY * (1 / 32), np.zeros(3, np.float32)),
        (IDENTITY * (1 / 64), np.array([1.5, -1.5, 0.0], np.float32)),
        (IDENTITY * (1 / 16), np.zeros(3, np.float32)),
        (-IDENTITY * (1 / 32), np.array([0.5, -0.5, 1.0], np.float32)),
        (np.diag(np.array([-1 / 32, 1 / 32, -1 / 32], np.float32)), np.array([0.5, 0.5, -0.5], np.float32)),
        (np.diag(np.array([1 / 64, -1 / 64, 1 / 64], np.float32)), np.array([-0.5, 0.0, 0.5], np.float32)),
    ],
    "Tanh": [
        (IDENTITY * (1 / 32), np.zeros(3, np.float32)),
        (IDENTITY * (1 / 64), np.array([-1.5, 1.5, 0.0], np.float32)),
        (IDENTITY * (1 / 16), np.zeros(3, np.float32)),
        (IDENTITY * (-1 / 32), np.array([1.0, -0.5, 0.5], np.float32)),
        (np.diag(np.array([1 / 32, -1 / 32, 1 / 32], np.float32)), np.array([-0.5, 0.5, 0.5], np.float32)),
        (np.diag(np.array([-1 / 64, 1 / 64, -1 / 64], np.float32)), np.array([0.5, 0.0, -0.5], np.float32)),
    ],
}
CONFIGS = []
for kind in ("Sigmoid", "Tanh"):
    for weights, bias in STEMS[kind]:
        CONFIGS.append((kind, weights, bias))

manifest = []
for index, (kind, weights, bias) in enumerate(CONFIGS):
    weights = np.asarray(weights, np.float32).reshape(3, 3)
    bias = np.asarray(bias, np.float32)
    low, high = stem_range(weights.reshape(3, 3, 1, 1), bias)
    assert -8 <= low and high <= 8, (index, low, high)
    model = h.make_model(h.make_graph(
        [h.make_node("Conv", ["input", "w", "b"], ["conv"], kernel_shape=[1, 1]),
         h.make_node(kind, ["conv"], ["output"])], "lut_domain",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
        [nh.from_array(weights.reshape(3, 3, 1, 1), "w"), nh.from_array(bias, "b")]),
        opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    path = root / f"model{index:03}.onnx"
    onnx.save(model, path)
    binary, meta = compile_sequence(path, 1, 128)
    path.with_suffix(".bin").write_bytes(binary)
    cases = rng.integers(0, 256, (args.cases, 8, 8, 3), dtype=np.uint8)
    cases[0] = 0
    cases[1] = 255
    cases[2] = 128
    cases.reshape(-1)[3:3 + 256] = np.arange(256, dtype=np.uint8)
    cases.tofile(root / f"input{index:03}.u8")
    expected = np.stack([lut_reference(x, weights, bias, kind) for x in cases])
    expected.tofile(root / f"expected{index:03}.i8")
    manifest.append({"index": index, "kind": kind, "cases": args.cases,
                     "stem_range": [low, high], "meta_range": meta["stem_range"],
                     "weights": weights.tolist(), "bias": bias.tolist(),
                     "output_scale": meta["output_scale"], "output_zero_point": meta["output_zero_point"]})
    print(index, kind, f"range=[{low:.3f},{high:.3f}]", flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} generalized LUT models in {root}")
