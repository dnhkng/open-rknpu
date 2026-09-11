"""Development-only C65/C128 input-channel oracle: what does the vendor do past C64?

Only the (0,0) tap is nonzero and its weight for input channel c is `1 + 2*c`, so
a captured weight buffer reveals the input-channel order inside one tap and a
captured output surface reveals which channels each task contributed. The point of
the fixture is the *task split*: the open loader caps input channels at 64, and the
documented blocker says channels above 64 need an INT32 partial-sum surface plus a
final bias/requantization task. A vendor container for C65/C128 either

* routes the extra channels through the same 64-channel task descriptor (so the
  cap is ours, not the hardware's), or
* emits more than one CNA task with an intermediate surface (the partial-sum
  mechanism we have not decoded), or
* refuses the model.

Development oracle only; never read by the open emitters.

    python  # vendor oracle environment: set VENDOR_PYTHON \
        research/build_native_c128_oracle.py
"""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from rknn.api import RKNN

ROOT = Path(__file__).resolve().parent


def build(channels):
    name = "native_c%d_channels" % channels
    out = ROOT / "fixtures" / name
    out.mkdir(parents=True, exist_ok=True)
    weights = np.zeros((1, channels, 3, 3), np.float32)
    for c in range(channels):
        weights[0, c, 0, 0] = 1 + 2 * c
    model = h.make_model(h.make_graph(
        [h.make_node("Conv", ["input", "w", "b"], ["output"], kernel_shape=[3, 3], pads=[1] * 4)],
        name, [h.make_tensor_value_info("input", 1, [1, channels, 6, 5])],
        [h.make_tensor_value_info("output", 1, [1, 1, 6, 5])],
        [nh.from_array(weights, "w"), nh.from_array(np.zeros(1, np.float32), "b")]),
        opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, out / "model.onnx")
    rng = np.random.default_rng(650128 + channels)
    paths = []
    for index in range(8):
        path = out / ("calibration%d.npy" % index)
        np.save(path, rng.uniform(0, 255, (1, channels, 6, 5)).astype(np.float32))
        paths.append(str(path))
    (out / "dataset.txt").write_text("\n".join(paths) + "\n")
    runtime = RKNN(verbose=False)
    try:
        assert runtime.config(target_platform="rv1103", mean_values=[[0] * channels],
                              std_values=[[1] * channels]) == 0
        assert runtime.load_onnx(model=str(out / "model.onnx")) == 0
        assert runtime.build(do_quantization=True, dataset=str(out / "dataset.txt")) == 0
        assert runtime.export_rknn(str(out / "model.rknn")) == 0
    finally:
        runtime.release()
    print(name, (out / "model.rknn").stat().st_size, "bytes")


for channels in (65, 128):
    build(channels)


def build_markers(name, channels, out_channels):
    """Tap-0-only marker model: every (o, c) cell of tap (0,0) is nonzero."""
    out = ROOT / "fixtures" / name
    out.mkdir(parents=True, exist_ok=True)
    weights = np.zeros((out_channels, channels, 3, 3), np.float32)
    weights[:, :, 0, 0] = 1.0
    model = h.make_model(h.make_graph(
        [h.make_node("Conv", ["input", "w", "b"], ["output"], kernel_shape=[3, 3], pads=[1] * 4)],
        name, [h.make_tensor_value_info("input", 1, [1, channels, 6, 5])],
        [h.make_tensor_value_info("output", 1, [1, out_channels, 6, 5])],
        [nh.from_array(weights, "w"), nh.from_array(np.zeros(out_channels, np.float32), "b")]),
        opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, out / "model.onnx")
    rng = np.random.default_rng(651700 + channels + out_channels)
    paths = []
    for index in range(8):
        path = out / ("calibration%d.npy" % index)
        np.save(path, rng.uniform(0, 255, (1, channels, 6, 5)).astype(np.float32))
        paths.append(str(path))
    (out / "dataset.txt").write_text("\n".join(paths) + "\n")
    runtime = RKNN(verbose=False)
    try:
        assert runtime.config(target_platform="rv1103", mean_values=[[0] * channels],
                              std_values=[[1] * channels]) == 0
        assert runtime.load_onnx(model=str(out / "model.onnx")) == 0
        assert runtime.build(do_quantization=True, dataset=str(out / "dataset.txt")) == 0
        assert runtime.export_rknn(str(out / "model.rknn")) == 0
    finally:
        runtime.release()
    print(name, (out / "model.rknn").stat().st_size, "bytes")


build_markers("native_c65_oci", 65, 17)
