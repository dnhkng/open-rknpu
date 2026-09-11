"""Asymmetric depthwise weight zero points via the public sequence path."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence

ROOT = Path(__file__).resolve().parents[1]


def one_sided_depthwise(channels=3, kernel=3, sign="positive", seed=4):
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-.2, .2, (channels, 3, 1, 1)).astype(np.float32)
    b1 = rng.uniform(-2, 2, channels).astype(np.float32)
    magnitude = rng.uniform(.05, .8, (channels, 1, kernel, kernel)).astype(np.float32)
    w2 = (magnitude if sign == "positive" else -magnitude).astype(np.float32)
    b2 = rng.uniform(-3, 3, channels).astype(np.float32)
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["conv"], kernel_shape=[1, 1]),
             h.make_node("Conv", ["conv", "w2", "b2"], ["output"], kernel_shape=[kernel, kernel],
                         pads=[kernel // 2] * 4, group=channels)]
    graph = h.make_graph(nodes, "depthwise_asymmetric",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, channels, 8, 8])],
        [nh.from_array(w1, "w1"), nh.from_array(b1, "b1"), nh.from_array(w2, "w2"), nh.from_array(b2, "b2")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


class DepthwiseAsymmetricTests(unittest.TestCase):
    def compile(self, model, **kwargs):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.onnx"
            onnx.save(model, path)
            return compile_sequence(path, **kwargs)

    def test_asymmetric_flag_uses_nonzero_weight_zero_points(self):
        model = one_sided_depthwise(sign="positive")
        _, asymmetric = self.compile(model, asymmetric_depthwise=True)
        _, symmetric = self.compile(model)
        self.assertTrue(asymmetric["depthwise_asymmetric_pair"])
        self.assertTrue(any(zp != 0 for zp in asymmetric["depthwise"]["weight_zero_points"]))
        self.assertFalse(symmetric["depthwise_asymmetric_pair"])
        self.assertEqual(set(symmetric["depthwise"]["weight_zero_points"]), {0})

    def test_asymmetric_negative_and_kernels(self):
        for sign in ("positive", "negative"):
            for kernel in (1, 3, 5):
                with self.subTest(sign=sign, kernel=kernel):
                    binary, meta = self.compile(one_sided_depthwise(kernel=kernel, sign=sign),
                                                asymmetric_depthwise=True)
                    self.assertEqual(decode_sequence(binary)["output_shape_nhwc"], [1, 8, 8, 3])
                    self.assertTrue(any(zp != 0 for zp in meta["depthwise"]["weight_zero_points"]))


if __name__ == "__main__":
    unittest.main()
