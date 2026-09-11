"""Development-only C48 input-channel order oracle.

Only the (0,0) tap is nonzero, and its weight for input channel c is `1 + 2*c`.
The captured buffer therefore reveals the input-channel order inside one tap.
Development oracle only; never read by the open emitters.
"""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from rknn.api import RKNN

root = Path(__file__).resolve().parent
out = root / "fixtures/native_c48_channels"
out.mkdir(exist_ok=True)
weights = np.zeros((1, 48, 3, 3), np.float32)
for c in range(48):
    weights[0, c, 0, 0] = 1 + 2 * c
model = h.make_model(h.make_graph(
    [h.make_node("Conv", ["input", "w", "b"], ["output"], kernel_shape=[3, 3], pads=[1] * 4)],
    "native_c48_channels",
    [h.make_tensor_value_info("input", 1, [1, 48, 6, 5])],
    [h.make_tensor_value_info("output", 1, [1, 1, 6, 5])],
    [nh.from_array(weights, "w"), nh.from_array(np.zeros(1, np.float32), "b")]),
    opset_imports=[h.make_opsetid("", 13)])
model.ir_version = 8
onnx.save(model, out / "model.onnx")
rng = np.random.default_rng(480021)
paths = []
for i in range(8):
    path = out / f"calibration{i}.npy"
    np.save(path, rng.uniform(0, 255, (1, 48, 6, 5)).astype(np.float32))
    paths.append(str(path))
(out / "dataset.txt").write_text("\n".join(paths) + "\n")
r = RKNN(verbose=False)
try:
    assert r.config(target_platform="rv1103", mean_values=[[0] * 48], std_values=[[1] * 48]) == 0
    assert r.load_onnx(model=str(out / "model.onnx")) == 0
    assert r.build(do_quantization=True, dataset=str(out / "dataset.txt")) == 0
    assert r.export_rknn(str(out / "model.rknn")) == 0
finally:
    r.release()
print("built", out / "model.rknn", (out / "model.rknn").stat().st_size, "bytes")
