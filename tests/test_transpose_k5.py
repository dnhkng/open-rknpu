"""Depthwise K5 ConvTranspose: vendor-derived emission, phase field and reference.

The board suite is `research/transpose_k5_suite/` (9 models, 144 inferences,
109,872 exact bytes). K5 uses the layout recovered from the vendor capture: 25
taps x 32 bytes with per-lane `(value, -weight_zero_point)` pairs and asymmetric
per-channel weight quantization. The stale pre-2026 artifacts are kept under
`research/transpose_k5_suite/stale/` as the retained failing experiment.
"""
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from onnx.reference import ReferenceEvaluator
from open_rknpu.quantization import Quantization, reference
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "research" / "transpose_k5_suite"

CASES = [("vendor", [2, 2], [2, 2, 2, 2], [0, 0], 3),
         ("stride1", [1, 1], [2, 2, 2, 2], [0, 0], 3),
         ("pad1", [2, 2], [1, 1, 1, 1], [0, 0], 3),
         ("pad0", [2, 2], [0, 0, 0, 0], [0, 0], 3),
         ("output_padding", [2, 2], [2, 2, 2, 2], [1, 1], 3),
         ("rectangular", [1, 2], [4, 0, 1, 3], [0, 1], 3),
         ("c8", [2, 2], [2, 2, 2, 2], [0, 0], 8)]


def model_for(channels, strides, pads, output_padding, seed=7):
    rng = np.random.default_rng(seed)
    kernel = 3 if channels <= 4 else 1
    stem_w = rng.uniform(-.3, .3, (channels, 3, kernel, kernel)).astype(np.float32)
    stem_b = rng.uniform(-1, 1, channels).astype(np.float32)
    weights = rng.uniform(-.25, .25, (channels, 1, 5, 5)).astype(np.float32)
    bias = rng.uniform(-.5, .5, channels).astype(np.float32)
    oh = 7 * strides[0] + 5 - pads[0] - pads[2] + output_padding[0]
    ow = 7 * strides[1] + 5 - pads[1] - pads[3] + output_padding[1]
    nodes = [h.make_node("Conv", ["input", "stem_w", "stem_b"], ["stem"], kernel_shape=[kernel, kernel],
                         pads=[kernel // 2] * 4),
             h.make_node("ConvTranspose", ["stem", "w", "b"], ["output"], group=channels,
                         kernel_shape=[5, 5], strides=strides, pads=pads, output_padding=output_padding)]
    graph = h.make_graph(nodes, "k5",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, channels, oh, ow])],
        [nh.from_array(stem_w, "stem_w"), nh.from_array(stem_b, "stem_b"),
         nh.from_array(weights, "w"), nh.from_array(bias, "b")])
    graph.value_info.append(h.make_tensor_value_info("stem", 1, [1, channels, 8, 8]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def reference_output(sample, q1, q, channels, strides, pads, output_padding):
    oh = 7 * strides[0] + 5 - pads[0] - pads[2] + output_padding[0]
    ow = 7 * strides[1] + 5 - pads[1] - pads[3] + output_padding[1]
    a = reference(sample, q1).astype(np.int64) - q1.output_zero_point
    centered = np.asarray(q.weights).reshape(channels, 5, 5) - np.asarray(q.weight_zero_points, np.int64)[:, None, None]
    base = np.asarray(q.biases, np.int64) + q1.output_zero_point * centered.sum(axis=(1, 2))
    acc = np.broadcast_to(base, (oh, ow, channels)).copy()
    for iy in range(8):
        for ix in range(8):
            for ky in range(5):
                for kx in range(5):
                    oy = iy * strides[0] + ky - pads[0]
                    ox = ix * strides[1] + kx - pads[1]
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
    return np.clip(result, -128, 127).astype(np.int8)


class TransposeK5Tests(unittest.TestCase):
    def compile(self, model):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.onnx"
            onnx.save(model, path)
            return compile_sequence(path)

    def test_every_geometry_compiles_with_the_vendor_layout(self):
        for name, strides, pads, output_padding, channels in CASES:
            with self.subTest(name=name):
                binary, meta = self.compile(model_for(channels, strides, pads, output_padding))
                self.assertIn("k5", meta["transposed_profile"])
                info = decode_sequence(binary)
                self.assertEqual(info["task_count"], 2)
                self.assertEqual(info["output_shape_nhwc"][3], channels)
                # 25 taps of 32 bytes, each lane a (value, -zero_point) pair.
                start = 96 + 16 * info["task_count"]
                payload = binary[start:]
                table = np.frombuffer(payload[0xB00:0xB00 + 25 * 32], np.int8).reshape(25, 32)
                self.assertTrue(table[:, 0:16:2].any())
                self.assertEqual(int(np.count_nonzero(table[:, 0:16:2].any(axis=1))), 25)
                self.assertTrue(table[:, 1:16:2].any(), "K5 stores asymmetric weight zero points")

    def test_phase_field_uses_the_effective_kernel(self):
        for name, strides, pads, output_padding, channels in CASES:
            with self.subTest(name=name):
                binary, _ = self.compile(model_for(channels, strides, pads, output_padding))
                info = decode_sequence(binary)
                start = 96 + 16 * info["task_count"]
                task = info["tasks"][1]
                registers = {}
                for index in range(task["register_count"]):
                    word = struct.unpack_from("<Q", binary[start:], task["command_offset"] + index * 8)[0]
                    registers[word & 0xffff] = (word >> 16) & 0xffffffff
                self.assertEqual(registers[0x1038], 0x5050002)
                self.assertEqual(registers[0x1030], 25 * 32)
                self.assertEqual(registers[0x1034], 25 * 16)
                self.assertEqual(registers[0x1188], 25 * 8)
                expected = ((5 - 1 - pads[1]) << 8) | (5 - 1 - pads[0])
                self.assertEqual(registers[0x1068], expected)

    def test_reference_tracks_the_float_model(self):
        rng = np.random.default_rng(11)
        for name, strides, pads, output_padding, channels in CASES:
            with self.subTest(name=name):
                model = model_for(channels, strides, pads, output_padding, seed=13)
                _, meta = self.compile(model)
                q = Quantization(**{k: np.array(v) if isinstance(v, list) else v
                                    for k, v in meta["transposed_quantization"].items()})
                q1 = Quantization(**{k: np.array(v) if isinstance(v, list) else v
                                     for k, v in meta["first"]["quantization"].items()})
                sample = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
                got = reference_output(sample, q1, q, channels, strides, pads, output_padding)
                value = (got.astype(np.float64) - meta["output_zero_point"]) * meta["output_scale"]
                expected = ReferenceEvaluator(model).run(
                    None, {"input": sample[None].transpose(0, 3, 1, 2).astype(np.float32)})[0][0].transpose(1, 2, 0)
                span = float(expected.max() - expected.min())
                self.assertLess(float(np.abs(value - expected).max()), 0.35 * span)

    def test_stale_artifacts_are_retained(self):
        stale = SUITE / "stale"
        self.assertTrue((stale / "model000.bin").exists())
        self.assertTrue((stale / "failure.json").exists())
        table = np.frombuffer((stale / "model000.bin").read_bytes()[128 + 0xB00:128 + 0xB00 + 25 * 32],
                              np.int8).reshape(25, 32)
        self.assertEqual(int(np.count_nonzero(table[:, 0:16:2].any(axis=1))), 17)

    def test_board_suite_is_recorded(self):
        import json
        results = json.loads((SUITE / "board_results_0.json").read_text())
        self.assertEqual(len(results), 9)
        self.assertTrue(all(entry["passed"] for entry in results))
        self.assertEqual(sum(entry["inferences"] for entry in results), 144)
        manifest = json.loads((SUITE / "manifest.json").read_text())
        self.assertEqual([entry["kernel"] for entry in manifest], [5] * 9)


if __name__ == "__main__":
    unittest.main()
