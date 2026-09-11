# SPDX-License-Identifier: MIT
"""Deep coverage of the bounded QLinearConv / Q-DQ importer (`quantized_import`).

`open_rknpu.quantized_import` lowers an already-quantized ONNX graph onto the
native16 Conv container while *preserving* the supplied INT8 weight codes: the
source weight scale/zero point become the native per-channel conversion, the
source output scale/zero point become the native output band, and only the
integer accumulator bias is derived. This module pins:

* accepted `QLinearConv` graphs (UINT8 activation in, INT8 activation out,
  scalar and per-output weight scale/zero point, with and without an INT32
  bias) and accepted `DQ -> Conv -> Q` graphs;
* that the declared quantization metadata reproduces the graph's parameters
  (weight codes, per-channel scales/zero points, both bands, the multiplier and
  shift, and the derived accumulator bias);
* that the supplied INT8 weight bytes really land in the container payload at
  the documented native16 offset and layout (independently repacked here);
* the integer output against an *independent* float64 implementation of the
  QLinearConv semantics - dequantize, real convolution, add the real bias,
  requantize onto the source output band and saturate - with 0 LSB on an
  integer-exact graph and a documented <= 1 LSB tolerance otherwise;
* every documented rejection: non-QLinearConv input, wrong activation I/O,
  non-constant / wrong-dtype weights, non-positive or non-finite scales, an
  out-of-range INT32 accumulator, the Q/DQ node-list and connection checks, a
  bad Q/DQ scale, and a Q/DQ float bias outside the INT32 accumulator range.

Nothing here reads a board, a vendor artifact or a network; all arithmetic is
recomputed from the compiled metadata.
"""
import unittest

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

from open_rknpu.native import native_input_reference
from open_rknpu.quantization import Quantization
from open_rknpu.quantized_import import (_preserved_quantization, compile_qdq_conv,
                                         compile_qlinearconv)
from open_rknpu.sequence import decode_sequence


def quantization_from(metadata):
    """Rebuild a live `Quantization` from emitted metadata."""
    values = {key: (np.array(value) if isinstance(value, list) else value)
              for key, value in metadata.items()}
    return Quantization(**values)


def to_real(codes, scale, zero_point):
    return (np.asarray(codes, np.float64) - zero_point) * scale


def requantize(values, scale, zero_point):
    return np.clip(np.rint(np.asarray(values, np.float64) / scale) + zero_point,
                   -128, 127).astype(np.int8)


def conv2d(x, weight, bias, pads, strides=(1, 1)):
    """NHWC cross-correlation in float64 (ONNX Conv)."""
    top, left, bottom, right = pads
    padded = np.pad(x, ((top, bottom), (left, right), (0, 0)))
    kh, kw = weight.shape[2:]
    sy, sx = strides
    oh = (padded.shape[0] - kh) // sy + 1
    ow = (padded.shape[1] - kw) // sx + 1
    out = np.zeros((oh, ow, weight.shape[0]))
    for oy in range(oh):
        for ox in range(ow):
            patch = padded[oy * sy:oy * sy + kh, ox * sx:ox * sx + kw, :]
            out[oy, ox] = np.tensordot(patch, weight, axes=([0, 1, 2], [2, 3, 1])) + bias
    return out


def qlinear_expected(sample, qw, wscale, wzp, qbias, xscale, xzp, yscale, yzp, pads):
    """Independent float64 QLinearConv: integer codes are a linear real band."""
    weights = ((qw.astype(np.float64) - wzp[:, None, None, None].astype(np.float64))
               * wscale[:, None, None, None].astype(np.float64))
    bias = qbias.astype(np.float64) * xscale * wscale.astype(np.float64)
    accumulator = conv2d(to_real(sample, xscale, xzp), weights, bias, pads)
    return requantize(accumulator, yscale, yzp)


def build_qlinear(outputs=3, channels=3, kernel=1, per_channel=False, bias=False, seed=3,
                  xscale=1 / 255, xzp=128, yscale=0.03, yzp=0, weight_scale=0.02):
    """A `QLinearConv` graph plus the source parameters it declares."""
    rng = np.random.default_rng(seed)
    qw = rng.integers(-6, 7, (outputs, channels, kernel, kernel)).astype(np.int8)
    if per_channel:
        wscale = np.linspace(weight_scale, weight_scale * 3, outputs).astype(np.float32)
        wzp = rng.integers(-3, 4, outputs).astype(np.int8)
    else:
        wscale = np.full(outputs, weight_scale, np.float32)
        wzp = np.zeros(outputs, np.int8)
    qbias = rng.integers(-2000, 2000, outputs).astype(np.int32)
    names = ["input", "xscale", "xzp", "w", "wscale", "wzp", "yscale", "yzp"]
    constants = [nh.from_array(np.array(xscale, np.float32), "xscale"),
                 nh.from_array(np.array(xzp, np.uint8), "xzp"),
                 nh.from_array(qw, "w"),
                 nh.from_array(wscale, "wscale"),
                 nh.from_array(wzp, "wzp"),
                 nh.from_array(np.array(yscale, np.float32), "yscale"),
                 nh.from_array(np.array(yzp, np.int8), "yzp")]
    if bias:
        names.append("bias")
        constants.append(nh.from_array(qbias, "bias"))
    side = 8 + 2 * (kernel // 2) - (kernel - 1)
    node = h.make_node("QLinearConv", names, ["output"], kernel_shape=[kernel, kernel],
                       pads=[kernel // 2] * 4, strides=[1, 1])
    graph = h.make_graph(
        [node], "qlinearconv",
        [h.make_tensor_value_info("input", onnx.TensorProto.UINT8, [1, channels, 8, 8])],
        [h.make_tensor_value_info("output", onnx.TensorProto.INT8, [1, outputs, side, side])],
        constants)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    params = dict(qw=qw, wscale=wscale, wzp=wzp, qbias=qbias, xscale=xscale, xzp=xzp,
                  yscale=yscale, yzp=yzp, kernel=kernel, bias=bias)
    return model, params


def build_qdq(outputs=3, channels=3, kernel=1, per_channel=False, bias=True, seed=5,
              weight_scale=0.02, xscale=1 / 255, xzp=128, yscale=0.03, yzp=0, bias_range=120.0):
    """A `DQ -> Conv -> Q` graph plus the source parameters it declares."""
    rng = np.random.default_rng(seed)
    qw = rng.integers(-6, 7, (outputs, channels, kernel, kernel)).astype(np.int8)
    if per_channel:
        ws = np.linspace(weight_scale, weight_scale * 2, outputs).astype(np.float32)
        wz = rng.integers(-3, 4, outputs).astype(np.int8)
    else:
        ws = np.full(outputs, weight_scale, np.float32)
        wz = np.zeros(outputs, np.int8)
    float_bias = (rng.uniform(-bias_range, bias_range, outputs).astype(np.float32) if bias
                  else np.zeros(outputs, np.float32))
    constants = [nh.from_array(np.array(xscale, np.float32), "xscale"),
                 nh.from_array(np.array(xzp, np.uint8), "xzp"),
                 nh.from_array(qw, "w"),
                 nh.from_array(ws, "ws"),
                 nh.from_array(wz, "wz"),
                 nh.from_array(np.array(yscale, np.float32), "yscale"),
                 nh.from_array(np.array(yzp, np.int8), "yzp")]
    if bias:
        constants.append(nh.from_array(float_bias, "bias"))
    conv_inputs = ["xdq", "wdq"] + (["bias"] if bias else [])
    nodes = [h.make_node("DequantizeLinear", ["input", "xscale", "xzp"], ["xdq"]),
             h.make_node("DequantizeLinear", ["w", "ws", "wz"], ["wdq"], axis=0),
             h.make_node("Conv", conv_inputs, ["conv"], kernel_shape=[kernel, kernel],
                         pads=[kernel // 2] * 4, strides=[1, 1]),
             h.make_node("QuantizeLinear", ["conv", "yscale", "yzp"], ["output"])]
    side = 8 + 2 * (kernel // 2) - (kernel - 1)
    graph = h.make_graph(
        nodes, "qdq",
        [h.make_tensor_value_info("input", onnx.TensorProto.UINT8, [1, channels, 8, 8])],
        [h.make_tensor_value_info("output", onnx.TensorProto.INT8, [1, outputs, side, side])],
        constants)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    params = dict(qw=qw, ws=ws, wz=wz, bias=float_bias, xscale=xscale, xzp=xzp,
                  yscale=yscale, yzp=yzp, kernel=kernel)
    return model, params


class ImportCase(unittest.TestCase):
    """Shared assertion that reports the observed per-channel LSB delta."""

    def assert_codes(self, got, expected, atol, label):
        got = np.asarray(got, np.int64)
        expected = np.asarray(expected, np.int64)
        self.assertEqual(got.shape, expected.shape, label)
        delta = int(np.abs(got - expected).max()) if got.size else 0
        self.assertLessEqual(delta, atol, "%s: max delta %d LSB" % (label, delta))
        return delta


class QLinearConvSemanticsTests(ImportCase):
    """Accepted QLinearConv graphs: metadata, container bytes and arithmetic."""

    def check_compile(self, model, params, atol, label):
        binary, meta = compile_qlinearconv(model)
        q = quantization_from(meta["quantization"])
        outputs = params["qw"].shape[0]
        np.testing.assert_array_equal(q.weights, params["qw"].reshape(outputs, -1))
        np.testing.assert_array_equal(np.asarray(q.weight_scales, np.float32), params["wscale"])
        np.testing.assert_array_equal(np.asarray(q.weight_zero_points, np.int32).astype(np.int8),
                                      params["wzp"])
        self.assertEqual(meta["input_scale"], np.float32(params["xscale"]))
        self.assertEqual(meta["input_zero_point"], params["xzp"])
        self.assertEqual(meta["output_scale"], np.float32(params["yscale"]))
        self.assertEqual(meta["output_zero_point"], params["yzp"])
        self.assertEqual(meta["quantized_import"], "QLinearConv weights preserved bit-for-bit")
        sample = np.random.default_rng(17).integers(
            0, 256, (8, 8, params["qw"].shape[1]), dtype=np.uint8)
        got = native_input_reference(sample, q, meta["input_zero_point"])
        qbias = (params["qbias"] if params["bias"]
                 else np.zeros(outputs, np.int32))
        expected = qlinear_expected(sample, params["qw"], params["wscale"], params["wzp"],
                                    qbias, params["xscale"], params["xzp"],
                                    params["yscale"], params["yzp"], (params["kernel"] // 2,) * 4)
        delta = self.assert_codes(got, expected, atol, label)
        return binary, meta, q, sample, expected, delta

    def test_scalar_weight_quantization_without_bias(self):
        model, params = build_qlinear(per_channel=False, bias=False)
        self.check_compile(model, params, 1, "scalar no bias")

    def test_per_output_weight_quantization_with_bias(self):
        model, params = build_qlinear(per_channel=True, bias=True)
        self.check_compile(model, params, 1, "per-output with bias")

    def test_spatial_3x3_weight_quantization(self):
        model, params = build_qlinear(kernel=3, per_channel=True, bias=True, seed=8)
        self.check_compile(model, params, 1, "3x3 per-output")

    def test_identity_weights_are_integer_exact(self):
        """A delta kernel at unit scales keeps the whole pipeline on integers."""
        qw = np.zeros((3, 3, 1, 1), np.int8)
        for channel in range(3):
            qw[channel, channel, 0, 0] = 1
        model, params = build_qlinear(per_channel=False, bias=False, xscale=1.0, xzp=0,
                                      yscale=1.0, yzp=0, weight_scale=1.0)
        params["qw"] = qw
        for tensor in model.graph.initializer:
            if tensor.name == "w":
                tensor.CopyFrom(nh.from_array(qw, "w"))
        _, _, q, sample, expected, delta = self.check_compile(model, params, 0, "identity")
        self.assertEqual(delta, 0)
        np.testing.assert_array_equal(q.weights, qw.reshape(3, -1))
        self.assert_codes(expected, np.clip(sample.astype(np.int64), -128, 127), 0, "identity codes")

    def test_supplied_int8_weight_bytes_survive_into_the_container(self):
        model, params = build_qlinear(per_channel=False, bias=False, seed=21)
        binary, meta, _, _, _, _ = self.check_compile(model, params, 1, "weight bytes")
        info = decode_sequence(binary)
        self.assertEqual(info["task_count"], 1)
        payload = binary[96 + 16 * info["task_count"]:]
        align = lambda value, multiple=64: (value + multiple - 1) // multiple * multiple
        weight_offset = align(130 * 8) * info["task_count"]
        qw, wzp = params["qw"], params["wzp"]
        rows = [[int(qw[o, c, 0, 0]) if c < qw.shape[1] else int(wzp[o]) for c in range(16)]
                for o in range(qw.shape[0])]
        expected = np.array(rows, np.int8).tobytes()
        self.assertEqual(payload[weight_offset:weight_offset + len(expected)], expected)
        self.assertEqual(int(meta["quantization"]["weights"][0][0]), int(qw[0, 0, 0, 0]))

    def test_output_band_is_not_ignored(self):
        model, params = build_qlinear()
        _, meta = compile_qlinearconv(model)
        sample = np.random.default_rng(4).integers(0, 256, (8, 8, 3), dtype=np.uint8)
        base = native_input_reference(sample, quantization_from(meta["quantization"]),
                                      meta["input_zero_point"])
        shifted = native_input_reference(
            sample,
            quantization_from(dict(meta["quantization"], output_zero_point=int(params["yzp"]) + 3)),
            meta["input_zero_point"])
        self.assertFalse(np.array_equal(shifted, base))


class QLinearConvRejectionTests(ImportCase):
    """Every documented QLinearConv rejection, with a compiling neighbour."""

    def test_requires_exactly_one_standard_qlinearconv(self):
        compile_qlinearconv(build_qlinear()[0])  # the valid neighbour compiles
        broken = onnx.ModelProto()
        broken.graph.CopyFrom(h.make_graph(
            [h.make_node("Relu", ["input"], ["output"])], "m",
            [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
            [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])]))
        with self.assertRaisesRegex(ValueError, "one standard QLinearConv"):
            compile_qlinearconv(broken)

    def test_activation_input_output_and_arity_are_checked(self):
        model, _ = build_qlinear()
        model.graph.node[0].input[0] = "w"
        with self.assertRaisesRegex(ValueError, "one activation input/output"):
            compile_qlinearconv(model)
        model, _ = build_qlinear()
        del model.graph.node[0].input[-1]
        with self.assertRaisesRegex(ValueError, "one activation input/output"):
            compile_qlinearconv(model)

    def test_missing_or_wrong_dtype_weights_are_rejected(self):
        for name, dtype in (("w", np.float32), ("wscale", np.int8), ("wzp", np.float32)):
            with self.subTest(name=name):
                model, _ = build_qlinear()
                for tensor in model.graph.initializer:
                    if tensor.name == name:
                        tensor.CopyFrom(nh.from_array(np.zeros(tensor.dims, dtype), name))
                with self.assertRaisesRegex(ValueError, "constant INT8"):
                    compile_qlinearconv(model)

    def test_missing_weight_tensor_is_rejected(self):
        model, _ = build_qlinear()
        kept = [tensor for tensor in model.graph.initializer if tensor.name != "w"]
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
        with self.assertRaisesRegex(ValueError, "constant INT8"):
            compile_qlinearconv(model)

    def test_nonpositive_or_nonfinite_scales_are_rejected(self):
        for name, value in (("xscale", 0.0), ("yscale", -0.5)):
            with self.subTest(name=name):
                model, _ = build_qlinear()
                for tensor in model.graph.initializer:
                    if tensor.name == name:
                        tensor.CopyFrom(nh.from_array(np.array(value, np.float32), name))
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    compile_qlinearconv(model)

    def test_missing_activation_scale_initializer_is_rejected(self):
        model, _ = build_qlinear()
        kept = [tensor for tensor in model.graph.initializer if tensor.name != "xscale"]
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
        with self.assertRaisesRegex(ValueError, "scalar activation quantization constants"):
            compile_qlinearconv(model)

    def test_weight_quantization_must_match_the_output_channel_count(self):
        model, _ = build_qlinear(outputs=3)
        for tensor in model.graph.initializer:
            if tensor.name == "wscale":
                tensor.CopyFrom(nh.from_array(np.zeros(2, np.float32), "wscale"))
        with self.assertRaisesRegex(ValueError, "scalar or per output channel"):
            compile_qlinearconv(model)


class QDQConvSemanticsTests(ImportCase):
    """Accepted DQ -> Conv -> Q graphs and the independent integer output."""

    def test_scalar_and_per_channel_qdq_semantics(self):
        for per_channel in (False, True):
            for kernel in (1, 3):
                with self.subTest(per_channel=per_channel, kernel=kernel):
                    model, params = build_qdq(per_channel=per_channel, kernel=kernel)
                    _, meta = compile_qdq_conv(model)
                    q = quantization_from(meta["quantization"])
                    np.testing.assert_array_equal(
                        q.weights, params["qw"].reshape(params["qw"].shape[0], -1))
                    np.testing.assert_array_equal(np.asarray(q.weight_scales, np.float32),
                                                  np.asarray(params["ws"], np.float32))
                    self.assertEqual(meta["output_scale"], np.float32(params["yscale"]))
                    self.assertEqual(meta["output_zero_point"], params["yzp"])
                    expected_bias = np.rint(
                        params["bias"].astype(np.float64)
                        / (params["xscale"] * params["ws"].astype(np.float64))).astype(np.int64)
                    np.testing.assert_array_equal(q.biases, expected_bias)
                    sample = np.random.default_rng(6).integers(
                        0, 256, (8, 8, params["qw"].shape[1]), dtype=np.uint8)
                    got = native_input_reference(sample, q, meta["input_zero_point"])
                    expected = qlinear_expected(
                        sample, params["qw"], params["ws"], params["wz"], expected_bias,
                        params["xscale"], params["xzp"], params["yscale"], params["yzp"],
                        (params["kernel"] // 2,) * 4)
                    self.assert_codes(got, expected, 1, "qdq %s k%d" % (per_channel, kernel))
                    self.assertIn("bit-for-bit", meta["quantized_import"])

    def test_qdq_float_bias_is_rounded_once(self):
        model, params = build_qdq(bias=True, per_channel=True)
        _, meta = compile_qdq_conv(model)
        expected = np.rint(params["bias"].astype(np.float64)
                           / (params["xscale"] * params["ws"].astype(np.float64))).astype(np.int64)
        np.testing.assert_array_equal(quantization_from(meta["quantization"]).biases, expected)
        self.assertEqual(meta["quantized_import"],
                         "Q/DQ Conv INT8 weights preserved bit-for-bit; "
                         "float bias rounded once to INT32")

    def test_missing_conv_bias_is_a_zero_bias(self):
        model, params = build_qdq(bias=False)
        _, meta = compile_qdq_conv(model)
        np.testing.assert_array_equal(quantization_from(meta["quantization"]).biases,
                                      np.zeros(params["qw"].shape[0], np.int64))

    def test_output_band_is_not_ignored(self):
        model, _ = build_qdq(per_channel=True, bias_range=1.0, yscale=0.5)
        _, meta = compile_qdq_conv(model)
        sample = np.random.default_rng(7).integers(0, 256, (8, 8, 3), dtype=np.uint8)
        base = native_input_reference(sample, quantization_from(meta["quantization"]),
                                      meta["input_zero_point"])
        self.assertGreater(len(np.unique(base)), 1, "expected a non-saturated output grid")
        changed = quantization_from(dict(meta["quantization"], output_zero_point=3))
        self.assertFalse(np.array_equal(
            native_input_reference(sample, changed, meta["input_zero_point"]), base))


class QDQConvRejectionTests(ImportCase):
    """Every documented Q/DQ rejection, each paired with a compiling graph."""

    def test_node_list_must_be_dq_dq_conv_q(self):
        model, _ = build_qdq()
        compile_qdq_conv(model)
        del model.graph.node[0]
        with self.assertRaisesRegex(ValueError, "input DQ, constant-weight DQ, Conv, output Q"):
            compile_qdq_conv(model)

    def test_graph_connections_are_checked(self):
        model, _ = build_qdq()
        model.graph.node[2].input[1] = "xdq"
        with self.assertRaisesRegex(ValueError, "invalid Q/DQ Conv graph connections"):
            compile_qdq_conv(model)

    def test_missing_or_wrong_dtype_qdq_weights_are_rejected(self):
        for name, dtype in (("w", np.float32), ("ws", np.int8), ("wz", np.float32)):
            with self.subTest(name=name):
                model, _ = build_qdq()
                for tensor in model.graph.initializer:
                    if tensor.name == name:
                        tensor.CopyFrom(nh.from_array(np.zeros(tensor.dims, dtype), name))
                with self.assertRaisesRegex(ValueError, "Q/DQ weights must be constant INT8"):
                    compile_qdq_conv(model)
        model, _ = build_qdq()
        kept = [tensor for tensor in model.graph.initializer if tensor.name != "w"]
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
        with self.assertRaisesRegex(ValueError, "Q/DQ weights must be constant INT8"):
            compile_qdq_conv(model)

    def test_unsupported_qdq_attributes_are_rejected(self):
        model, _ = build_qdq()
        model.graph.node[0].attribute.append(h.make_attribute("axis", 0))
        with self.assertRaisesRegex(ValueError, "unsupported Q/DQ attributes"):
            compile_qdq_conv(model)
        model, _ = build_qdq()
        for node in model.graph.node:
            if node.op_type == "DequantizeLinear" and node.input[0] == "w":
                node.attribute.append(h.make_attribute("block_size", 1))
        with self.assertRaisesRegex(ValueError, "unsupported Q/DQ attributes"):
            compile_qdq_conv(model)

    def test_per_channel_requires_axis_zero(self):
        model, _ = build_qdq(per_channel=True)
        for node in model.graph.node:
            if node.op_type == "DequantizeLinear" and node.input[0] == "w":
                node.attribute[0].i = 1
        with self.assertRaisesRegex(ValueError, "per-channel weight Q/DQ requires axis0"):
            compile_qdq_conv(model)

    def test_nonpositive_qdq_scale_is_rejected(self):
        model, _ = build_qdq()
        for tensor in model.graph.initializer:
            if tensor.name == "ws":
                tensor.CopyFrom(nh.from_array(np.array([-0.02, 0.02, 0.02], np.float32), "ws"))
        with self.assertRaisesRegex(ValueError, "Q/DQ scales must be finite and positive"):
            compile_qdq_conv(model)

    def test_qdq_bias_outside_int32_is_rejected(self):
        for bias in (1e10, -1e10):
            with self.subTest(bias=bias):
                model, _ = build_qdq()
                for tensor in model.graph.initializer:
                    if tensor.name == "bias":
                        tensor.CopyFrom(nh.from_array(
                            np.array([bias, 0.0, 0.0], np.float32), "bias"))
                with self.assertRaisesRegex(ValueError, "outside INT32 accumulator range"):
                    compile_qdq_conv(model)


class PreservedQuantizationBoundaryTests(ImportCase):
    """The three documented `_preserved_quantization` range rejections."""

    def test_int32_accumulator_overflow(self):
        weights = np.array([[[[1]]]], np.int8)
        with self.assertRaisesRegex(ValueError, "INT32 accumulator overflow"):
            _preserved_quantization(weights, np.array([1.0], np.float32), np.array([0], np.int8),
                                    np.array([2 ** 31 - 1], np.int32), 1.0, 128, 1.0, 0)
        quantization = _preserved_quantization(
            weights, np.array([1.0], np.float32), np.array([0], np.int8),
            np.array([0], np.int32), 1.0, 128, 1.0, 0)
        self.assertEqual(int(quantization.biases[0]), 0)
        self.assertEqual(quantization.kernel_size, 1)

    def test_per_channel_conversion_range(self):
        weights = np.array([[[[1]]]], np.int8)
        with self.assertRaisesRegex(ValueError, "per-channel conversion range"):
            _preserved_quantization(weights, np.array([1.0, 1e-9], np.float32),
                                    np.array([0, 0], np.int8), np.array([0, 0], np.int32),
                                    1.0, 128, 1.0, 0)
        quantization = _preserved_quantization(
            np.array([[[[1]]], [[[1]]]], np.int8), np.array([1.0, 0.5], np.float32),
            np.array([0, 0], np.int8), np.array([0, 0], np.int32), 1.0, 128, 1.0, 0)
        np.testing.assert_array_equal(quantization.channel_multipliers, np.array([16384, 8192]))
        self.assertTrue(np.all(quantization.channel_multipliers >= 1))

    def test_source_output_scale_conversion_range(self):
        weights = np.array([[[[1]]]], np.int8)
        with self.assertRaisesRegex(ValueError, "source output scale exceeds native conversion range"):
            _preserved_quantization(weights, np.array([1.0], np.float32), np.array([0], np.int8),
                                    np.array([0], np.int32), 1.0, 128, 1e-6, 0)
        quantization = _preserved_quantization(
            weights, np.array([1.0], np.float32), np.array([0], np.int8),
            np.array([0], np.int32), 1.0, 128, 1.0, 0)
        self.assertGreaterEqual(quantization.multiplier, 1)
        self.assertLessEqual(quantization.multiplier, 32767)
        self.assertGreaterEqual(quantization.shift, 0)


if __name__ == "__main__":
    unittest.main()
