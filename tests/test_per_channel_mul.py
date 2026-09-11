"""Per-channel Mul quantization through the public sequence path.

The verified EW constant Mul stores one operand scale for the whole constant
(`[.02,.35,1.9]` becomes `[1,23,127]`), while `--per-channel-mul` lowers the same
`Mul(input, [C,1,1])` onto the verified 1x1 depthwise profile, whose native
per-output-channel weight scale quantizes each channel on its own grid. The
board suite that pins exact bytes is `research/per_channel_mul_suite/`.
"""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.depthwise import depthwise_reference
from open_rknpu.elementwise import mul_reference
from open_rknpu.quantization import Quantization, reference
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence


def constant_mul(shape, factor, seed=5, spatial=False):
    rng = np.random.default_rng(seed)
    if spatial:
        values = rng.uniform(-2, 2, (1, 3, shape[2], shape[3])).astype(np.float32)
    else:
        values = np.asarray(factor, np.float32).reshape(-1, 1, 1)
    return h.make_model(h.make_graph(
        [h.make_node("Mul", ["input", "factor"], ["output"])], "constant_mul",
        [h.make_tensor_value_info("input", 1, list(shape))],
        [h.make_tensor_value_info("output", 1, list(shape))],
        [nh.from_array(values, "factor")]), opset_imports=[h.make_opsetid("", 13)])


def quant(params):
    values = dict(params)
    for key in ("weights", "weight_zero_points", "weight_scales", "biases", "channel_multipliers"):
        values[key] = np.array(values[key])
    return Quantization(**values)


class PerChannelMulTests(unittest.TestCase):
    def compile(self, model, **kwargs):
        model.ir_version = 8
        model = onnx.shape_inference.infer_shapes(model)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.onnx"
            onnx.save(model, path)
            return compile_sequence(path, **kwargs)

    def errors(self, model, factor, cases=48, seed=6):
        """Maximum float-domain error of the per-channel and shared-scale paths."""
        rng = np.random.default_rng(seed)
        shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
        _, height, width = shape[1:]
        x = rng.integers(0, 256, (cases, height, width, 3), dtype=np.uint8)
        exact = x.astype(np.float64) * np.asarray(factor, np.float64).reshape(1, 1, 1, 3)
        _, meta = self.compile(model, per_channel_mul=True)
        q1, q2 = quant(meta["first"]["quantization"]), quant(meta["depthwise"])
        down = np.stack([depthwise_reference(reference(v, q1), q2, q1.output_zero_point) for v in x])
        down = (down.astype(np.float64) - q2.output_zero_point) * q2.output_scale
        _, shared = self.compile(model)
        qs = quant(shared["branches"][0]["quantization"])
        operand = np.clip(np.rint(np.broadcast_to(np.asarray(factor, np.float32).reshape(3, 1, 1),
                                                  (1, 3, height, width)) / shared["constant_scale"]),
                          -128, 127).astype(np.int8)[0].transpose(1, 2, 0)
        wide = np.stack([mul_reference(reference(v, qs), operand) for v in x]).astype(np.float64)
        wide = (wide - shared["output_zero_point"]) * (128 * qs.output_scale * shared["constant_scale"])
        return float(np.abs(down - exact).max()), float(np.abs(wide - exact).max()), meta, q2

    def test_per_channel_scales_beat_the_shared_operand_scale(self):
        factor = [0.02, 0.35, 1.9]
        model = constant_mul([1, 3, 8, 8], factor)
        per_channel, shared, meta, q2 = self.errors(model, factor)
        self.assertEqual(meta["profile"], "per-channel-constant-mul")
        self.assertEqual(meta["constant_data_mode"], "per-channel-weight-scales")
        self.assertEqual([round(v, 5) for v in meta["per_channel_constant"]], [round(v, 5) for v in factor])
        # Each channel of the constant keeps its own grid: the operand bytes are 127
        # everywhere instead of [1,23,127] under one shared scale.
        np.testing.assert_allclose(q2.weights.ravel(), [127, 127, 127], atol=1)
        self.assertAlmostEqual(q2.weight_scales[0] / q2.weight_scales[2], factor[0] / factor[2], places=3)
        self.assertLess(per_channel, shared / 2.5)

    def test_geometry_and_magnitude_variants(self):
        for shape, factor in (([1, 3, 5, 5], [1.2, 1.4, 1.6]),
                              ([1, 3, 6, 7], [0.005, 0.3, 2.4]),
                              ([1, 3, 8, 8], [-0.4, 0.9, -2.1]),
                              ([1, 3, 5, 8], [0.01, 0.02, 0.03]),
                              ([1, 3, 6, 6], [0.0, 0.5, -1.5])):
            with self.subTest(shape=shape, factor=factor):
                model = constant_mul(shape, factor)
                per_channel, shared, meta, _ = self.errors(model, factor)
                self.assertLess(per_channel, shared)
                binary, _ = self.compile(model, per_channel_mul=True)
                self.assertEqual(decode_sequence(binary)["output_shape_nhwc"],
                                 [1, shape[2], shape[3], 3])

    def test_shared_scale_path_is_unchanged_by_default(self):
        model = constant_mul([1, 3, 8, 8], [0.02, 0.35, 1.9])
        binary, meta = self.compile(model)
        self.assertEqual(meta["constant_data_mode"], "per-channel")
        self.assertNotIn("profile", meta)
        self.assertEqual(decode_sequence(binary)["task_count"], 2)

    def test_rejects_profiles_outside_the_depthwise_boundary(self):
        for model, factor in ((constant_mul([1, 3, 8, 8], [1.5, 1.5, 1.5]), [1.5, 1.5, 1.5]),
                              (constant_mul([1, 3, 8, 8], None, spatial=True), [0.5, 0.5, 0.5]),
                              (constant_mul([1, 3, 3, 3], [0.1, 0.2, 0.3]), [0.1, 0.2, 0.3])):
            with self.subTest(factor=factor):
                with self.assertRaises(ValueError):
                    self.compile(model, per_channel_mul=True)


if __name__ == "__main__":
    unittest.main()
