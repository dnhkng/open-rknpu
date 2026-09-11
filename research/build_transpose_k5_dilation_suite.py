"""MIT. Verify depthwise K3 dilation2 ConvTranspose through a direct sparse K5 rewrite.

The dilated K3 support spans five taps, so the same operation is a dense K5
ConvTranspose with zeros at the odd positions. This suite compiles that direct
emission (no vendor capture, no RKNN) and checks it on the board. Expected bytes
come from an explicit integer loop over the quantized expanded weights, the same
independent reference used for the retained K2-dilation2 suite.
"""
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.quantization import Quantization, reference

root = Path(__file__).resolve().parent / "transpose_k5_dilation_suite"
root.mkdir(parents=True, exist_ok=True)
for pattern in ("model*.onnx", "model*.bin", "input*.u8", "expected*.i8"):
    for stale in root.glob(pattern):
        stale.unlink()
rng = np.random.default_rng(111200)

# (channels, strides, pads, output_padding, auto_pad)
CASES = [
    (3, (1, 1), (0, 0, 0, 0), (0, 0), "NOTSET"),
    (3, (1, 1), (1, 1, 1, 1), (0, 0), "NOTSET"),
    (3, (2, 2), (0, 0, 0, 0), (0, 0), "NOTSET"),
    (3, (2, 2), (1, 1, 1, 1), (1, 0), "NOTSET"),
    (1, (1, 1), (2, 2, 2, 2), (0, 0), "NOTSET"),
    (4, (2, 2), (2, 2, 2, 2), (1, 0), "NOTSET"),
    (3, (1, 2), (1, 0, 0, 1), (0, 1), "NOTSET"),
    (3, (2, 2), (0, 0, 0, 0), (1, 1), "SAME_UPPER"),
]

manifest = []
for index, (channels, strides, pads, output_padding, auto) in enumerate(CASES):
    constants = [nh.from_array(rng.uniform(-.4, .4, (channels, 3, 1, 1)).astype(np.float32), "w1"),
                 nh.from_array(rng.uniform(-.5, .5, (channels,)).astype(np.float32), "b1"),
                 nh.from_array(rng.uniform(-.4, .4, (channels, 1, 3, 3)).astype(np.float32), "w2"),
                 nh.from_array(rng.uniform(-.5, .5, (channels,)).astype(np.float32), "b2")]
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["stem"], kernel_shape=[1, 1]),
             h.make_node("ConvTranspose", ["stem", "w2", "b2"], ["output"], kernel_shape=[3, 3],
                         strides=list(strides), pads=list(pads), dilations=[2, 2], group=channels,
                         output_padding=list(output_padding), auto_pad=auto)]
    sy, sx = strides
    resolved = list(pads)
    if auto == "SAME_UPPER":
        totals = [5 + output_padding[i] - strides[i] for i in range(2)]
        begins = [v // 2 for v in totals]
        resolved = [begins[0], begins[1], totals[0] - begins[0], totals[1] - begins[1]]
    oh = 7 * sy + 5 - resolved[0] - resolved[2] + output_padding[0]
    ow = 7 * sx + 5 - resolved[1] - resolved[3] + output_padding[1]
    graph = h.make_graph(nodes, "transpose_k5_dilation",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, channels, oh, ow])], constants)
    graph.value_info.append(h.make_tensor_value_info("stem", 1, [1, channels, 8, 8]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    path = root / f"model{index:03}.onnx"
    onnx.save(model, path)
    binary, meta = compile_sequence(path)
    path.with_suffix(".bin").write_bytes(binary)
    assert meta["transposed_rewrite"] == "K3 dilation2 expanded to sparse K5", meta.get("transposed_rewrite")
    q = Quantization(**{key: np.array(value) if isinstance(value, list) else value
                        for key, value in meta["transposed_quantization"].items()})
    q1 = Quantization(**{key: np.array(value) if isinstance(value, list) else value
                         for key, value in meta["first"]["quantization"].items()})
    inputs = rng.integers(0, 256, (16, 8, 8, 3), dtype=np.uint8)
    inputs[0] = 0
    inputs[1] = 255
    inputs[2] = 128
    # The K5 tap table uses asymmetric per-channel weight zero points, so the
    # reference centers the weights and folds the stem zero point into the base.
    centered = np.asarray(q.weights).reshape(channels, 5, 5) \
        - np.asarray(q.weight_zero_points, np.int64)[:, None, None]
    bias_units = np.asarray(q.biases, np.int64) + q1.output_zero_point * centered.sum(axis=(1, 2))
    expected = []
    for sample in inputs:
        a = reference(sample, q1).astype(np.int64) - q1.output_zero_point
        acc = np.broadcast_to(bias_units, (oh, ow, channels)).copy()
        for iy in range(8):
            for ix in range(8):
                for ky in range(5):
                    for kx in range(5):
                        oy = iy * sy + ky - resolved[0]
                        ox = ix * sx + kx - resolved[1]
                        if 0 <= oy < oh and 0 <= ox < ow:
                            acc[oy, ox] += a[iy, ix] * centered[:, ky, kx]
        product = acc * q.channel_multipliers
        scaled = (product + 8191 + ((product >> 14) & 1)) >> 14
        product = scaled * q.multiplier
        if q.shift:
            product = product + (1 << (q.shift - 1)) - 1 + ((product >> q.shift) & 1)
            result = (product >> q.shift) + q.output_zero_point
        else:
            result = product + q.output_zero_point
        expected.append(np.clip(result, -128, 127).astype(np.int8))
    inputs.tofile(root / f"input{index:03}.u8")
    np.stack(expected).tofile(root / f"expected{index:03}.i8")
    manifest.append({"index": index, "kernel": 3, "dilation": 2, "channels": channels,
                     "strides": list(strides), "pads": list(pads), "output_padding": list(output_padding),
                     "auto_pad": auto, "resolved_pads": resolved, "output_shape": [oh, ow, channels],
                     "cases": 16, "output_bytes": int(channels * oh * ow),
                     "rewrite": meta["transposed_rewrite"],
                     "output_scale": meta["output_scale"], "output_zero_point": meta["output_zero_point"]})
    print(index, f"C{channels}", strides, resolved, f"op{output_padding}", auto, f"out {oh}x{ow}", flush=True)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Built {len(manifest)} K3-dilation2 transpose models in {root}")
