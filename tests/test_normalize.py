"""Semantic checks for ONNX normalization, including unsafe broadcast cases."""
import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from onnx.reference import ReferenceEvaluator
from open_rknpu.normalize import normalize_model


class NormalizeTests(unittest.TestCase):
    def model(self, bias_shape=(2, 1, 1), branch=False, overridable=False):
        rng = np.random.default_rng(1103)
        nodes = [h.make_node("Conv", ["x", "w"], ["conv"], auto_pad="SAME_UPPER"),
                 h.make_node("Add", ["conv", "b"], ["y"])]
        inputs = [h.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3, 5, 2])]
        if overridable:
            inputs.append(h.make_tensor_value_info("b", onnx.TensorProto.FLOAT, bias_shape))
        outputs = [h.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 2, 5, 2])]
        if branch:
            outputs.append(h.make_tensor_value_info("conv", onnx.TensorProto.FLOAT, [1, 2, 5, 2]))
        return h.make_model(h.make_graph(nodes, "bias", inputs, outputs, [
            nh.from_array(rng.normal(size=(2, 3, 3, 3)).astype(np.float32), "w"),
            nh.from_array(rng.normal(size=bias_shape).astype(np.float32), "b")]),
            opset_imports=[h.make_opsetid("", 12)])

    def check_equal(self, model):
        normalized = normalize_model(model)
        x = np.random.default_rng(42).normal(size=(1, 3, 5, 2)).astype(np.float32)
        expected = ReferenceEvaluator(model).run(None, {"x": x})
        actual = ReferenceEvaluator(normalized).run(None, {"x": x})
        for a, b in zip(expected, actual):
            np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
        return normalized

    def test_fold_channel_bias_and_padding(self):
        for shape in ((2, 1, 1), (1, 2, 1, 1)):
            normalized = self.check_equal(self.model(shape))
            self.assertEqual([n.op_type for n in normalized.graph.node], ["Conv"])
            self.assertEqual(normalize_model(normalized).SerializeToString(), normalized.SerializeToString())

    def test_preserve_width_bias_branch_and_overridable_bias(self):
        for model in (self.model((2,)), self.model(branch=True), self.model(overridable=True)):
            normalized = self.check_equal(model)
            self.assertEqual([n.op_type for n in normalized.graph.node], ["Conv", "Add"])

    def test_constant_reshape_zero_and_inferred_dimension(self):
        model = h.make_model(h.make_graph(
            [h.make_node("Reshape", ["data", "shape"], ["y"])], "constant", [],
            [h.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 12])],
            [nh.from_array(np.arange(24, dtype=np.float32).reshape(2, 3, 4), "data"),
             nh.from_array(np.array([0, -1], dtype=np.int64), "shape")]),
            opset_imports=[h.make_opsetid("", 14)])
        normalized = normalize_model(model)
        self.assertEqual(len(normalized.graph.node), 0)
        np.testing.assert_array_equal(ReferenceEvaluator(model).run(None, {})[0],
                                      ReferenceEvaluator(normalized).run(None, {})[0])
