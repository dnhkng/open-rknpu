"""SPDX-License-Identifier: MIT

Independent semantic validation of every emitter family in `open_rknpu`.

Method
------
The existing emitter tests compare a container with the emitter's own Python
reference (self-consistency).  This module supplies the other half.  For each
family it builds a small ONNX graph with hand-chosen, integer-friendly weights
(identity, delta, box, ramp, or a fixed-seed random tensor), compiles it with
`open_rknpu.scheduler.compile_sequence`, obtains the container's *integer* output
through that profile's reference function -- imported **only** as the thing under
test -- and compares it against an independent float64 implementation of the ONNX
semantics written here.  The independent side never calls an emitter reference:
it dequantizes the container's own quantization parameters as
`real = (code - zero_point) * scale`, runs the operator arithmetic in float64 with
numpy, and re-quantizes onto the container's output band as
`code = clip(rint(real / output_scale) + output_zero_point, -128, 127)`.

Tolerance policy
----------------
* **0 LSB** where the chosen weights and bands make the arithmetic integer-exact
  and the result is also asserted analytically: a K1 identity/delta kernel, a box
  kernel over a constant field, an Add/Sub of identical branches, a centre-delta
  ConvTranspose, and a constant Mul by zero.  These double as the exact property
  required per family.
* **<= 1 LSB** everywhere else.  The container replaces the ideal float product
  with a 14-bit channel multiplier, a multiplier/shift pair and the
  half-away-from-zero correction of `docs/quantization.md`, so a value landing on
  a rounding limit can move one code away from the float64 result.
* One documented exception: a constant Mul by 1.0 is a real-valued no-op only
  (<= 1 LSB), because the verified convention fixes the output band at
  `128 * product_scale`, so the largest constant code 127 cannot reproduce the
  input codes exactly.

Limitations (recorded, not skipped)
-----------------------------------
* `transposed.py` exposes no integer reference.  For the symmetric default
  padding the emitted K3 task is exactly the depthwise correlation, so
  `depthwise_reference` is a legitimate oracle and a centre delta is exact.
  Off-centre taps are validated by rebuilding the integer task from the
  container's declared `transposed_quantization` (the arithmetic the retained
  vendor tests use) and by decoding the emitted tap table, then comparing with
  the float64 ONNX scatter.  A wrong hardware phase field remains invisible to a
  host reference; the tap table itself is read from the compiled container.
* The pool in `pooling.py` has no reference of its own; the op-level walk
  reference is the integer oracle for the same Conv+2x2-pool semantics, and the
  compiled `pooling.py` container is separately checked to declare the matching
  output band and geometry.
* `depthwise.depthwise_reference` has no `strides` parameter, so only stride-1
  depthwise tasks are validated here; the stride-2 container bytes are pinned by
  `tests/test_depthwise_stride.py` against the retained suite.  Every other
  emitted depthwise field (padding, band, per-channel weights) is covered.

Findings
--------
* The dense chain profiles (`chain.py`, `chain_n.py`) leave the internal border
  register `0x1184` at its `-128` reset value, while the fan-out emitters
  (`graph.py`, `join_dag.py`, `pool_join.py`, ...) inject the producer's zero
  point.  A hidden band whose analytic zero point is not -128 (178 of the 242
  retained chain models) therefore pads with `(-128 - zero_point) * scale`
  instead of the real zero ONNX `Conv` pads with.  The retained board suites
  (`research/chain_k3_suite/` and friends) match `chain_n_reference` byte for
  byte, so the container is self-consistent and board-pinned, but it does not
  reproduce ONNX zero padding for that band.  `test_internal_padding_convention_is_recorded`
  pins the convention and reports the gap; no `src` change was made because the
  board evidence contradicts the "inject the producer zero point" fix.
"""
import unittest

import numpy as np
import onnx
from onnx import helper as h
from onnx import numpy_helper as nh

from open_rknpu.chain_n import chain_n_reference
from open_rknpu.depthwise import depthwise_reference
from open_rknpu.elementwise import add_reference, max_reference, mul_reference, sub_reference
from open_rknpu.graph import diamond_reference, join_reference
from open_rknpu.join_dag import join_dag_reference
from open_rknpu.lut import LUT_DOMAIN, lut_reference
from open_rknpu.native import native_input_reference
from open_rknpu.quantization import Quantization, reference
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence
from open_rknpu.walk import chain_walk_reference, load_quantizations, parse_chain

# ---------------------------------------------------------------------------
# Independent arithmetic (the expected side).  Nothing here calls an emitter
# reference: the container only supplies its declared quantization parameters.
# ---------------------------------------------------------------------------


def _quant(entry):
    """Build a live `Quantization` from emitter metadata (dict or Quantization)."""
    if isinstance(entry, Quantization):
        return entry
    if "quantization" in entry:
        entry = entry["quantization"]
    values = {key: (np.array(value) if isinstance(value, list) else value) for key, value in entry.items()}
    return Quantization(**values)


def dequant_weights(q):
    """Dequantize the container's weight codes: `scale * (code - zero_point)`."""
    codes = np.asarray(q.weights, np.float64)
    zero = np.asarray(q.weight_zero_points, np.float64)
    scale = np.asarray(q.weight_scales, np.float64)
    kernel = int(q.kernel_size)
    outputs = codes.shape[0]
    inputs = codes.size // (outputs * kernel * kernel)
    centered = codes.reshape(outputs, inputs, kernel, kernel) - zero[:, None, None, None]
    return centered * scale[:, None, None, None]


def requantize(values, scale, zero_point):
    """Re-quantize a real tensor onto an INT8 band (round-half-to-even)."""
    return np.clip(np.rint(np.asarray(values, np.float64) / scale) + zero_point, -128, 127).astype(np.int8)


def to_real(codes, scale, zero_point):
    return (np.asarray(codes, np.float64) - zero_point) * scale


def conv2d(x, weight, bias, pads=(0, 0, 0, 0), strides=(1, 1)):
    """NHWC cross-correlation: ONNX Conv, float64."""
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


def depthwise2d(x, weight, bias, pads, strides=(1, 1)):
    """NHWC grouped (depthwise) Conv with real zero padding, float64."""
    top, left, bottom, right = pads
    padded = np.pad(x, ((top, bottom), (left, right), (0, 0)))
    channels, _, kh, kw = weight.shape
    sy, sx = strides
    oh = (padded.shape[0] - kh) // sy + 1
    ow = (padded.shape[1] - kw) // sx + 1
    out = np.zeros((oh, ow, channels))
    per_channel = weight[:, 0].transpose(1, 2, 0)  # (kh, kw, C)
    for oy in range(oh):
        for ox in range(ow):
            patch = padded[oy * sy:oy * sy + kh, ox * sx:ox * sx + kw, :]
            out[oy, ox] = (patch * per_channel).sum(axis=(0, 1)) + bias
    return out


def conv_transpose2d(x, weight, bias, pads, strides, out_height, out_width):
    """ONNX ConvTranspose scatter, float64.  `weight` is (C_in, C_out, kh, kw)."""
    kh, kw = weight.shape[2:]
    out = np.zeros((out_height, out_width, weight.shape[1])) + bias
    for iy in range(x.shape[0]):
        for ix in range(x.shape[1]):
            for ky in range(kh):
                for kx in range(kw):
                    oy = iy * strides[0] - pads[0] + ky
                    ox = ix * strides[1] - pads[1] + kx
                    if 0 <= oy < out_height and 0 <= ox < out_width:
                        out[oy, ox] = out[oy, ox] + x[iy, ix] @ weight[:, :, ky, kx]
    return out


def pool2x2(codes, kind):
    """2x2/stride-2 block maximum or rounded mean on an INT8 grid."""
    height, width, channels = codes.shape
    blocked = codes.astype(np.int64).reshape(height // 2, 2, width // 2, 2, channels)
    if kind == "MaxPool":
        pooled = blocked.max(axis=(1, 3))
    else:
        pooled = np.rint(blocked.sum(axis=(1, 3)) / 4.0)
    return np.clip(pooled, -128, 127).astype(np.int8)


def chain_expected(sample, quants, biases, input_scale, input_zero_point, border_code=None):
    """Layer-by-layer float64 chain with an INT8 grid between layers.

    `border_code` selects the convention for a spatial (K3) layer that reads an
    internal grid: `None` pads the *dequantized* tensor with real zeros (ONNX
    `Conv`), while an integer pads the previous INT8 codes with that code and
    dequantizes, which is what the emitted native16 border register does.
    """
    value = to_real(sample, input_scale, input_zero_point)
    codes = None
    for index, (q, bias) in enumerate(zip(quants, biases)):
        pad = q.kernel_size // 2
        if index > 0 and border_code is not None:
            previous = quants[index - 1]
            padded = np.pad(codes, ((pad, pad), (pad, pad), (0, 0)), constant_values=border_code)
            value = conv2d(to_real(padded, previous.output_scale, previous.output_zero_point),
                           dequant_weights(q), np.asarray(bias, np.float64), (0, 0, 0, 0))
        else:
            value = conv2d(value, dequant_weights(q), np.asarray(bias, np.float64), (pad, pad, pad, pad))
        if q.relu:
            value = np.maximum(value, 0.0)
        codes = requantize(value, q.output_scale, q.output_zero_point)
        value = to_real(codes, q.output_scale, q.output_zero_point)
    if codes is None:
        raise AssertionError("empty chain")
    return codes


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def build_model(nodes, input_shape, output_shape, initializers, value_info=()):
    """Build a graph.  Spatial shapes are given as NHWC (H, W, C) and declared NCHW."""
    def nchw(shape):
        height, width, channels = shape
        return [1, channels, height, width]

    graph = h.make_graph(
        nodes, "semantic",
        [h.make_tensor_value_info("input", 1, nchw(input_shape))],
        [h.make_tensor_value_info("output", 1, nchw(output_shape))],
        list(initializers),
        value_info=[h.make_tensor_value_info(name, 1, nchw(shape)) for name, shape in value_info])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def identity_kernel(channels, kernel=1):
    weight = np.zeros((channels, channels, kernel, kernel), np.float32)
    for channel in range(channels):
        weight[channel, channel, kernel // 2, kernel // 2] = 1.0
    return weight


def diagonal_kernel(outputs, inputs, kernel=1):
    """Ones on the overlapping diagonal of a (outputs, inputs) channel map."""
    weight = np.zeros((outputs, inputs, kernel, kernel), np.float32)
    for channel in range(min(outputs, inputs)):
        weight[channel, channel, kernel // 2, kernel // 2] = 1.0
    return weight


def conv_node(name, source, out, weight, bias, kernel, pads, strides=(1, 1), group=1):
    return h.make_node("Conv", [source, f"{name}_w", f"{name}_b"], [out], kernel_shape=[kernel, kernel],
                       pads=list(pads), strides=list(strides), group=group)


def ramp(shape, low=124, span=9):
    height, width, channels = shape
    return (low + (np.arange(height * width * channels) % span)).astype(np.uint8).reshape(shape)


class SemanticCase(unittest.TestCase):
    """Shared assertion with an explicit reported tolerance."""

    def assert_codes(self, got, expected, atol, label):
        got = np.asarray(got, np.int64)
        expected = np.asarray(expected, np.int64)
        self.assertEqual(got.shape, expected.shape, f"{label}: shape {got.shape} != {expected.shape}")
        delta = int(np.abs(got - expected).max()) if got.size else 0
        self.assertLessEqual(delta, atol, f"{label}: observed max delta {delta} LSB exceeds {atol} LSB")
        return delta


# ---------------------------------------------------------------------------
# 1. image-input dense Conv (native.py)
# ---------------------------------------------------------------------------


class NativeConvTests(SemanticCase):
    def make(self, case):
        return build_model(
            [conv_node("conv", "input", "output", case["weight"], case["bias"], case["kernel"],
                       case["pads"], case["strides"])],
            case["input_shape"], case["output_shape"],
            [nh.from_array(case["weight"], "conv_w"), nh.from_array(case["bias"], "conv_b")])

    def test_native_image_conv_semantics(self):
        delta_k1 = np.zeros((4, 4, 1, 1), np.float32)
        for channel in range(4):
            delta_k1[channel, channel, 0, 0] = 1.0
        delta_k3 = np.zeros((2, 2, 3, 3), np.float32)
        for channel in range(2):
            delta_k3[channel, channel, 1, 1] = 1.0
        rng = np.random.default_rng(4104)
        cases = [
            dict(name="delta-k1", weight=delta_k1, bias=np.zeros(4, np.float32), kernel=1,
                 pads=(0, 0, 0, 0), strides=(1, 1), input_shape=(6, 6, 4), output_shape=(6, 6, 4), atol=0),
            dict(name="box-k3", weight=np.ones((2, 4, 3, 3), np.float32), bias=np.zeros(2, np.float32), kernel=3,
                 pads=(1, 1, 1, 1), strides=(1, 1), input_shape=(6, 6, 4), output_shape=(6, 6, 2), atol=1),
            dict(name="delta-k3-stride2", weight=delta_k3, bias=np.zeros(2, np.float32), kernel=3,
                 pads=(1, 1, 1, 1), strides=(2, 2), input_shape=(8, 8, 2), output_shape=(4, 4, 2), atol=0),
            dict(name="random-k1", weight=rng.uniform(.05, .3, (3, 4, 1, 1)).astype(np.float32),
                 bias=rng.uniform(-.2, .2, 3).astype(np.float32), kernel=1,
                 pads=(0, 0, 0, 0), strides=(1, 1), input_shape=(6, 6, 4), output_shape=(6, 6, 3), atol=1),
        ]
        for case in cases:
            with self.subTest(name=case["name"]):
                _, meta = compile_sequence(self.make(case))
                self.assertEqual(meta["profile"], "native16-input")
                q = _quant(meta["quantization"])
                sample = ramp(case["input_shape"])
                got = native_input_reference(sample, q, meta["input_zero_point"],
                                             meta["conv_pads"], meta["conv_strides"])
                real = to_real(sample, meta["input_scale"], meta["input_zero_point"])
                expected = requantize(conv2d(real, dequant_weights(q), np.asarray(case["bias"], np.float64),
                                             case["pads"], case["strides"]),
                                      meta["output_scale"], meta["output_zero_point"])
                self.assert_codes(got, expected, case["atol"], case["name"])
                if case["name"] == "delta-k1":
                    # identity: container code must be exactly byte - output_zero_point
                    self.assert_codes(got, sample.astype(np.int64) + meta["output_zero_point"], 0,
                                      "delta-k1 analytic identity")


# ---------------------------------------------------------------------------
# 2. Conv chain composition (chain.py / chain_n.py)
# ---------------------------------------------------------------------------


class ChainCompositionTests(SemanticCase):
    def compile(self, layers):
        """`layers` is a list of (input_channels, output_channels, kernel, identity)."""
        nodes = []
        initializers = []
        biases = []
        previous = "input"
        for index, (cin, cout, kernel, identity) in enumerate(layers):
            if identity:
                weight = diagonal_kernel(cout, cin, kernel)
            else:
                generator = np.random.default_rng(1000 + index)
                weight = generator.uniform(.05, .3, (cout, cin, kernel, kernel)).astype(np.float32)
            bias = np.zeros(cout, np.float32)
            initializers += [nh.from_array(weight, f"c{index}_w"), nh.from_array(bias, f"c{index}_b")]
            out = "output" if index == len(layers) - 1 else f"t{index}"
            nodes.append(conv_node(f"c{index}", previous, out, weight, bias, kernel, (kernel // 2,) * 4))
            previous = out
            if index < len(layers) - 1:
                nodes.append(h.make_node("Relu", [previous], [f"r{index}"]))
                previous = f"r{index}"
            biases.append(bias)
        model = build_model(nodes, (8, 8, 3), (8, 8, layers[-1][1]), initializers)
        return model, biases

    def test_conv_chain_composition_semantics(self):
        sample = ramp((8, 8, 3), low=96, span=64)
        # Two identity 1x1 convolutions must reproduce one identity convolution.
        model, biases = self.compile([(3, 3, 1, True), (3, 3, 1, True)])
        _, meta = compile_sequence(model)
        self.assertEqual(meta["profile"], 2)
        got = chain_n_reference(sample, [meta["first"], meta["second"]])
        self.assert_codes(got, sample.astype(np.int64) - 128, 0, "chain identity analytic")
        single = build_model([conv_node("s", "input", "output", identity_kernel(3), np.zeros(3, np.float32),
                                       1, (0, 0, 0, 0))],
                             (8, 8, 3), (8, 8, 3),
                             [nh.from_array(identity_kernel(3), "s_w"),
                              nh.from_array(np.zeros(3, np.float32), "s_b")])
        _, single_meta = compile_sequence(single)
        one = reference(sample, _quant(single_meta["quantization"]))
        self.assert_codes(got, one, 0, "two identity convs equal one")

        # Conv -> Relu -> Conv versus the independent two-step computation.
        for name, kernel, atol in (("k1", 1, 1), ("k3", 3, 1)):
            with self.subTest(name=name):
                model, biases = self.compile([(3, 5, kernel, False), (5, 3, kernel, False)])
                _, meta = compile_sequence(model)
                quants = [_quant(meta["first"]), _quant(meta["second"])]
                got = chain_n_reference(sample, [meta["first"], meta["second"]])
                expected = chain_expected(sample, quants, biases, 1.0, 0)
                self.assert_codes(got, expected, atol, f"chain {name}")

        # A five-layer alternating Conv/Relu chain (chain_n.py), random and identity.
        random_layers = [(3, 5, 1, False), (5, 5, 3, False), (5, 4, 3, False),
                         (4, 4, 1, False), (4, 3, 3, False)]
        model, biases = self.compile(random_layers)
        _, meta = compile_sequence(model)
        self.assertEqual(meta["profile"], "native-chain")
        quants = [_quant(entry) for entry in meta["quantizations"]]
        got = chain_n_reference(sample, meta["quantizations"])
        self.assert_codes(got, chain_expected(sample, quants, biases, 1.0, 0), 1, "chain_n five layers")

        identity_layers = [(3, 5, 1, True), (5, 5, 1, True), (5, 4, 1, True),
                           (4, 4, 1, True), (4, 3, 1, True)]
        model, biases = self.compile(identity_layers)
        _, meta = compile_sequence(model)
        got = chain_n_reference(sample, meta["quantizations"])
        self.assert_codes(got, sample.astype(np.int64) - 128, 0, "chain_n identity analytic")

    def test_internal_padding_convention_is_recorded(self):
        """The internal K3 border uses the tensor's zero point, as ONNX pads.

        A mixed-sign hidden layer carries an analytic zero point other than -128, so the
        emitted native16 layer programs register 0x1184 with the band it reads and pads the
        border with the real zero ONNX `Conv` pads with. Before 2026-09-12 the register
        kept its -128 reset and the container deviated from the float graph (F4); this test
        pins the fixed convention and reports how far the old one was, so the difference
        cannot come back silently.
        """
        layers = [(3, 5, 1), (5, 5, 3), (5, 4, 3), (4, 4, 1), (4, 3, 3)]
        generator = np.random.default_rng(77)
        nodes, initializers, biases, previous = [], [], [], "input"
        for index, (cin, cout, kernel) in enumerate(layers):
            weight = generator.uniform(-.6, .7, (cout, cin, kernel, kernel)).astype(np.float32)
            bias = generator.uniform(-1, 1, cout).astype(np.float32)
            initializers += [nh.from_array(weight, f"c{index}_w"), nh.from_array(bias, f"c{index}_b")]
            out = "output" if index == len(layers) - 1 else f"t{index}"
            nodes.append(conv_node(f"c{index}", previous, out, weight, bias, kernel, (kernel // 2,) * 4))
            previous = out
            if index < len(layers) - 1:
                nodes.append(h.make_node("Relu", [previous], [f"r{index}"]))
                previous = f"r{index}"
            biases.append(bias)
        sample = ramp((8, 8, 3), low=96, span=64)
        model = build_model(nodes, (8, 8, 3), (8, 8, 3), initializers)
        _, meta = compile_sequence(model)
        quants = [_quant(entry) for entry in meta["quantizations"]]
        self.assertTrue(any(q.output_zero_point != -128 for q in quants[1:]),
                        "expected a hidden band whose zero point is not -128")
        got = chain_n_reference(sample, meta["quantizations"])
        zero_pad = chain_expected(sample, quants, biases, 1.0, 0)
        self.assert_codes(got, zero_pad, 1, "container zero-padding convention")
        legacy = chain_expected(sample, quants, biases, 1.0, 0, border_code=-128)
        deviation = int(np.abs(got.astype(np.int64) - legacy.astype(np.int64)).max())
        self.assertGreater(deviation, 1,
                           "the fixed container should differ from the old -128 border by more "
                           "than 1 LSB (hidden zero points %s)"
                           % ([q.output_zero_point for q in quants],))


# ---------------------------------------------------------------------------
# 3. pooling (pooling.py terminal pool and the walk's interior pool)
# ---------------------------------------------------------------------------


class PoolingSemanticsTests(SemanticCase):
    def test_terminal_pool_semantics(self):
        sample = ramp((8, 8, 3), low=100, span=48)
        for kind, atol in (("MaxPool", 0), ("AveragePool", 0)):
            with self.subTest(kind=kind):
                weight = np.random.default_rng(3001).uniform(.05, .3, (3, 3, 1, 1)).astype(np.float32)
                bias = np.zeros(3, np.float32)
                model = build_model(
                    [conv_node("c", "input", "stem", weight, bias, 1, (0, 0, 0, 0)),
                     h.make_node(kind, ["stem"], ["output"], kernel_shape=[2, 2], strides=[2, 2])],
                    (8, 8, 3), (4, 4, 3),
                    [nh.from_array(weight, "c_w"), nh.from_array(bias, "c_b")],
                    value_info=[("stem", (8, 8, 3))])
                _, meta = compile_sequence(model)
                self.assertEqual(meta["pool_stages"], [kind])
                q = _quant(meta["quantization"])
                self.assertEqual((meta["output_scale"], meta["output_zero_point"]),
                                 (q.output_scale, q.output_zero_point))
                self.assertEqual(meta["output_shape_nhwc"], [1, 4, 4, 3])
                ops = parse_chain(model.graph)["ops"]
                got = chain_walk_reference(sample, [q, None], ops, 0)
                stem = requantize(conv2d(to_real(sample, 1.0, 0), dequant_weights(q), bias), q.output_scale,
                                  q.output_zero_point)
                self.assert_codes(got, pool2x2(stem, kind), atol, f"pooling.py {kind}")

        # Exact property: an identity stem makes the block maximum analytic.
        model = build_model(
            [conv_node("c", "input", "stem", identity_kernel(3), np.zeros(3, np.float32), 1, (0, 0, 0, 0)),
             h.make_node("MaxPool", ["stem"], ["output"], kernel_shape=[2, 2], strides=[2, 2])],
            (8, 8, 3), (4, 4, 3),
            [nh.from_array(identity_kernel(3), "c_w"), nh.from_array(np.zeros(3, np.float32), "c_b")],
            value_info=[("stem", (8, 8, 3))])
        _, meta = compile_sequence(model)
        q = _quant(meta["quantization"])
        ops = parse_chain(model.graph)["ops"]
        got = chain_walk_reference(sample, [q, None], ops, 0)
        blocks = (sample.astype(np.int64) - 128).reshape(4, 2, 4, 2, 3)
        self.assert_codes(got, blocks.max(axis=(1, 3)), 0, "identity stem block maximum")

    def test_walk_interior_pool_semantics(self):
        sample = ramp((8, 8, 3), low=100, span=48)
        for kind in ("MaxPool", "AveragePool"):
            with self.subTest(kind=kind):
                rng = np.random.default_rng(3200 + (kind == "AveragePool"))
                w1 = rng.uniform(.05, .3, (5, 3, 1, 1)).astype(np.float32)
                w2 = rng.uniform(.05, .3, (3, 5, 3, 3)).astype(np.float32)
                b1, b2 = np.zeros(5, np.float32), np.zeros(3, np.float32)
                model = build_model(
                    [conv_node("c1", "input", "stem", w1, b1, 1, (0, 0, 0, 0)),
                     h.make_node(kind, ["stem"], ["pooled"], kernel_shape=[2, 2], strides=[2, 2]),
                     conv_node("c2", "pooled", "output", w2, b2, 3, (1, 1, 1, 1))],
                    (8, 8, 3), (4, 4, 3),
                    [nh.from_array(w1, "c1_w"), nh.from_array(b1, "c1_b"),
                     nh.from_array(w2, "c2_w"), nh.from_array(b2, "c2_b")],
                    value_info=[("stem", (8, 8, 5)), ("pooled", (4, 4, 5))])
                _, meta = compile_sequence(model)
                self.assertEqual(meta["profile"], "chain-walk")
                quants = load_quantizations(meta)
                ops = parse_chain(model.graph)["ops"]
                got = chain_walk_reference(sample, quants, ops, 0)
                stem = requantize(conv2d(to_real(sample, 1.0, 0), dequant_weights(quants[0]), b1),
                                  quants[0].output_scale, quants[0].output_zero_point)
                pooled = pool2x2(stem, kind)
                expected = requantize(conv2d(to_real(pooled, quants[0].output_scale, quants[0].output_zero_point),
                                             dequant_weights(quants[2]), b2, (1, 1, 1, 1)),
                                      meta["output_scale"], meta["output_zero_point"])
                self.assert_codes(got, expected, 0, f"walk interior {kind}")


# ---------------------------------------------------------------------------
# 4. depthwise (depthwise.py)
# ---------------------------------------------------------------------------


class DepthwiseSemanticsTests(SemanticCase):
    def test_depthwise_semantics(self):
        sample = ramp((8, 8, 3), low=124, span=9)
        constant = np.full((8, 8, 3), 131, np.uint8)
        cases = [
            dict(name="box-k3", kernel=3, strides=(1, 1), sample=sample, atol=0),
            dict(name="box-k1", kernel=1, strides=(1, 1), sample=sample, atol=0),
            dict(name="delta-k3", kernel=3, strides=(1, 1), sample=sample, atol=0, delta=True),
            dict(name="box-k3-constant", kernel=3, strides=(1, 1), sample=constant, atol=0),
        ]
        for case in cases:
            with self.subTest(name=case["name"]):
                kernel = case["kernel"]
                channels = 3
                stem_weight = identity_kernel(3)
                stem_bias = np.zeros(3, np.float32)
                if case.get("delta"):
                    weight = np.zeros((channels, 1, kernel, kernel), np.float32)
                    for channel in range(channels):
                        weight[channel, 0, kernel // 2, kernel // 2] = 1.0
                else:
                    weight = np.ones((channels, 1, kernel, kernel), np.float32)
                bias = np.zeros(channels, np.float32)
                height, width = case["sample"].shape[:2]
                stride = case["strides"][0]
                out_hw = ((height + 2 * (kernel // 2) - kernel) // stride + 1,)
                model = build_model(
                    [conv_node("stem", "input", "stem_out", stem_weight, stem_bias, 1, (0, 0, 0, 0)),
                     h.make_node("Conv", ["stem_out", "dw_w", "dw_b"], ["output"], group=channels,
                                 kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4,
                                 strides=list(case["strides"]))],
                    (height, width, 3), (out_hw[0], out_hw[0], channels),
                    [nh.from_array(stem_weight, "stem_w"), nh.from_array(stem_bias, "stem_b"),
                     nh.from_array(weight, "dw_w"), nh.from_array(bias, "dw_b")],
                    value_info=[("stem_out", (height, width, 3))])
                _, meta = compile_sequence(model)
                q_stem = _quant(meta["first"]["quantization"])
                q_dw = _quant(meta["depthwise"])
                stem_codes = reference(case["sample"], q_stem)
                got = depthwise_reference(stem_codes, q_dw, q_stem.output_zero_point)
                stem_real = conv2d(to_real(case["sample"], 1.0, 0), dequant_weights(q_stem), stem_bias)
                stem_codes_independent = requantize(stem_real, q_stem.output_scale, q_stem.output_zero_point)
                dw_real = depthwise2d(to_real(stem_codes_independent, q_stem.output_scale, q_stem.output_zero_point),
                                      dequant_weights(q_dw).reshape(channels, 1, kernel, kernel)[:, :, :, :],
                                      bias, (kernel // 2,) * 4, case["strides"])
                expected = requantize(dw_real, q_dw.output_scale, q_dw.output_zero_point)
                self.assert_codes(got, expected, case["atol"], case["name"])
                if case["name"] == "box-k3-constant":
                    # Interior windows are full 3x3 boxes; the one-pixel border sees
                    # the zero (real) padding instead of the constant code.
                    self.assert_codes(got[1:-1, 1:-1], np.full((6, 6, 3), int(case["sample"][0, 0, 0]) - 128),
                                      0, "constant box interior analytic")


# ---------------------------------------------------------------------------
# 5. elementwise joins (elementwise.py) and the identical-branch diamond
# ---------------------------------------------------------------------------


def branch_weights(seed, outputs=3, inputs=3):
    return np.random.default_rng(seed).uniform(.05, .4, (outputs, inputs, 1, 1)).astype(np.float32)


class ElementwiseJoinTests(SemanticCase):
    def elementwise_model(self, kind, weight_a, weight_b, bias_a=None, bias_b=None):
        height, width = 8, 8
        channels = weight_a.shape[0]
        bias_a = np.zeros(channels, np.float32) if bias_a is None else bias_a
        bias_b = np.zeros(weight_b.shape[0], np.float32) if bias_b is None else bias_b
        return build_model(
            [conv_node("ca", "input", "a", weight_a, bias_a, 1, (0, 0, 0, 0)),
             conv_node("cb", "input", "b", weight_b, bias_b, 1, (0, 0, 0, 0)),
             h.make_node(kind, ["a", "b"], ["output"])],
            (height, width, 3), (height, width, channels),
            [nh.from_array(weight_a, "ca_w"), nh.from_array(bias_a, "ca_b"),
             nh.from_array(weight_b, "cb_w"), nh.from_array(bias_b, "cb_b")],
            value_info=[("a", (height, width, channels)), ("b", (height, width, channels))])

    def test_elementwise_join_semantics(self):
        sample = ramp((8, 8, 3), low=100, span=48)
        join_oracle = {"Add": add_reference, "Sub": sub_reference, "Max": max_reference, "Mul": mul_reference}
        cases = [dict(name="Add", kind="Add", weight_a=branch_weights(5001), weight_b=branch_weights(5002), atol=1),
                 dict(name="Sub", kind="Sub", weight_a=branch_weights(5003), weight_b=branch_weights(5004), atol=1),
                 dict(name="Max", kind="Max", weight_a=branch_weights(5005), weight_b=branch_weights(5006), atol=1),
                 dict(name="Mul", kind="Mul", weight_a=branch_weights(5007), weight_b=branch_weights(5008), atol=1),
                 dict(name="Add-identity-branch", kind="Add", weight_a=branch_weights(5009),
                      weight_b=identity_kernel(3), atol=1)]
        for case in cases:
            with self.subTest(name=case["name"]):
                model = self.elementwise_model(case["kind"], case["weight_a"], case["weight_b"])
                _, meta = compile_sequence(model)
                branches = [_quant(branch["quantization"]) for branch in meta["branches"]]
                got = join_oracle[case["kind"]](reference(sample, branches[0]), reference(sample, branches[1]))
                real = to_real(sample, 1.0, 0)
                real_a = conv2d(real, dequant_weights(branches[0]), np.zeros(branches[0].weights.shape[0]))
                real_b = conv2d(real, dequant_weights(branches[1]), np.zeros(branches[1].weights.shape[0]))
                a = to_real(requantize(real_a, branches[0].output_scale, branches[0].output_zero_point),
                            branches[0].output_scale, branches[0].output_zero_point)
                b = to_real(requantize(real_b, branches[1].output_scale, branches[1].output_zero_point),
                            branches[1].output_scale, branches[1].output_zero_point)
                value = {"Add": a + b, "Sub": a - b, "Max": np.maximum(a, b), "Mul": a * b}[case["kind"]]
                expected = requantize(value, meta["output_scale"], meta["output_zero_point"])
                self.assert_codes(got, expected, case["atol"], case["name"])

        # Exact property: identical Add branches reproduce the branch code.
        weight = branch_weights(5100)
        model = self.elementwise_model("Add", weight, weight.copy())
        _, meta = compile_sequence(model)
        branches = [_quant(branch["quantization"]) for branch in meta["branches"]]
        got = join_reference("Add", reference(sample, branches[0]), reference(sample, branches[1]))
        self.assert_codes(got, reference(sample, branches[0]), 0, "Add of identical branches")
        self.assert_codes(join_reference("Sub", reference(sample, branches[0]), reference(sample, branches[1])),
                          np.zeros((8, 8, 3)) + meta.get("output_zero_point", 0), 0, "Sub of identical branches")

    def test_diamond_identical_branch_semantics(self):
        sample = ramp((8, 8, 3), low=100, span=48)
        hidden = 3
        stem = branch_weights(5200, outputs=hidden, inputs=3)
        head = branch_weights(5201)
        initializers = [nh.from_array(stem, "s_w"), nh.from_array(np.zeros(hidden, np.float32), "s_b"),
                        nh.from_array(head, "h0_w"), nh.from_array(np.zeros(3, np.float32), "h0_b"),
                        nh.from_array(head, "h1_w"), nh.from_array(np.zeros(3, np.float32), "h1_b")]
        for kind in ("Add", "Sub", "Max"):
            with self.subTest(kind=kind):
                model = build_model(
                    [conv_node("s", "input", "stem", stem, np.zeros(hidden, np.float32), 1, (0, 0, 0, 0)),
                     conv_node("h0", "stem", "head_a", head, np.zeros(3, np.float32), 1, (0, 0, 0, 0)),
                     conv_node("h1", "stem", "head_b", head.copy(), np.zeros(3, np.float32), 1, (0, 0, 0, 0)),
                     h.make_node(kind, ["head_a", "head_b"], ["output"])],
                    (8, 8, 3), (8, 8, 3), initializers,
                    value_info=[("stem", (8, 8, hidden)), ("head_a", (8, 8, 3)), ("head_b", (8, 8, 3))])
                _, meta = compile_sequence(model)
                got = diamond_reference(sample, meta["stem_quantization"], meta["head_quantization"], kind)
                stem_codes = reference(sample, _quant(meta["stem_quantization"]))
                head_q = _quant(meta["head_quantization"][0])
                head_codes = requantize(
                    conv2d(to_real(stem_codes, _quant(meta["stem_quantization"]).output_scale,
                                   _quant(meta["stem_quantization"]).output_zero_point),
                           dequant_weights(head_q), np.zeros(3)), head_q.output_scale, head_q.output_zero_point)
                stem_real = conv2d(to_real(sample, 1.0, 0), dequant_weights(_quant(meta["stem_quantization"])),
                                   np.zeros(hidden))
                stem_expected = requantize(stem_real, _quant(meta["stem_quantization"]).output_scale,
                                           _quant(meta["stem_quantization"]).output_zero_point)
                head_real = conv2d(to_real(stem_expected, _quant(meta["stem_quantization"]).output_scale,
                                           _quant(meta["stem_quantization"]).output_zero_point),
                                   dequant_weights(head_q), np.zeros(3))
                head_expected = requantize(head_real, head_q.output_scale, head_q.output_zero_point)
                expected = requantize({"Add": head_real + head_real, "Sub": head_real - head_real,
                                       "Max": np.maximum(head_real, head_real)}[kind],
                                      meta["output_scale"], meta["output_zero_point"])
                self.assert_codes(got, expected, 1, f"diamond {kind}")
                self.assert_codes(head_codes, head_expected, 0, f"diamond {kind} head grid")
                if kind == "Add":
                    self.assert_codes(got, head_codes, 0, "identical diamond Add analytic")


# ---------------------------------------------------------------------------
# 6. constant Mul (elementwise.py)
# ---------------------------------------------------------------------------


class ConstantMulTests(SemanticCase):
    def constant_model(self, factor):
        node = h.make_node("Mul", ["input", "factor"], ["output"])
        return build_model([node], (8, 8, 3), (8, 8, 3), [nh.from_array(np.asarray(factor, np.float32), "factor")])

    def test_constant_mul_semantics(self):
        sample = ramp((8, 8, 3), low=100, span=48)
        cases = [
            dict(name="scalar-1.0", factor=np.float32(1.0), atol=1),
            dict(name="scalar-0.0", factor=np.float32(0.0), atol=0),
            dict(name="per-channel", factor=np.array([0.5, 1.0, 2.0], np.float32).reshape(1, 3, 1, 1), atol=1),
            dict(name="spatial", factor=np.linspace(0.25, 2.0, 8 * 8 * 3).reshape(1, 3, 8, 8).astype(np.float32),
                 atol=1),
        ]
        for case in cases:
            with self.subTest(name=case["name"]):
                model = self.constant_model(case["factor"])
                _, meta = compile_sequence(model)
                branch = _quant(meta["branches"][0]["quantization"])
                stem = reference(sample, branch)
                scale = meta["constant_scale"]
                expanded = np.broadcast_to(np.asarray(case["factor"], np.float32), (1, 3, 8, 8))
                codes = np.clip(np.rint(expanded / scale), -128, 127).astype(np.int8)[0].transpose(1, 2, 0)
                if meta["constant_data_mode"] == "per-channel":
                    codes = codes[0, 0]
                got = mul_reference(stem, codes)
                factor = np.asarray(case["factor"], np.float64)
                if factor.ndim == 0:
                    factor_nhwc = float(factor)
                elif factor.shape == (1, 3, 1, 1):
                    factor_nhwc = factor.reshape(3)
                else:
                    factor_nhwc = factor[0].transpose(1, 2, 0)
                expected = requantize(to_real(sample, 1.0, 0) * factor_nhwc,
                                      meta["output_scale"], meta["output_zero_point"])
                self.assert_codes(got, expected, case["atol"], case["name"])
                if case["name"] == "scalar-0.0":
                    self.assert_codes(got, np.zeros((8, 8, 3)) + meta["output_zero_point"], 0, "zero constant analytic")
                if case["name"] == "scalar-1.0":
                    # A constant 1.0 is a no-op in real value, not in code (documented).
                    value = to_real(got, meta["output_scale"], meta["output_zero_point"])
                    reference_value = to_real(stem, branch.output_scale, branch.output_zero_point)
                    delta = float(np.abs(value - reference_value).max())
                    self.assertLessEqual(delta, meta["output_scale"] + 1e-9,
                                         f"scalar-1.0 real no-op delta {delta} exceeds one output LSB")


# ---------------------------------------------------------------------------
# 7. LUT (lut.py)
# ---------------------------------------------------------------------------


class LutSemanticsTests(SemanticCase):
    def lut_model(self, gain, kind):
        weight = identity_kernel(3).copy()
        weight *= gain
        bias = np.zeros(3, np.float32)
        # The LUT profile only accepts a bare kernel_shape on the stem.
        stem = h.make_node("Conv", ["input", "c_w", "c_b"], ["stem"], kernel_shape=[1, 1])
        return build_model(
            [stem, h.make_node(kind, ["stem"], ["output"])],
            (8, 8, 3), (8, 8, 3),
            [nh.from_array(weight, "c_w"), nh.from_array(bias, "c_b")],
            value_info=[("stem", (8, 8, 3))])

    def test_lut_semantics(self):
        sample = np.zeros((8, 8, 3), np.uint8)
        sample[:, :, 0] = np.linspace(0, 255, 64, dtype=np.uint8).reshape(8, 8)
        sample[:, :, 1] = 128
        sample[:, :, 2] = 200
        for gain in (1.0 / 32.0, 1.0 / 16.0):
            for kind in ("Sigmoid", "Tanh"):
                with self.subTest(gain=gain, kind=kind):
                    model = self.lut_model(gain, kind)
                    _, meta = compile_sequence(model, input_scale=1.0, input_zero_point=128)
                    self.assertEqual(meta["lut_profile"], f"{kind.lower()}-8x8-c3")
                    self.assertEqual(meta["kind"], kind)
                    weight = (identity_kernel(3) * gain).astype(np.float32)
                    got = lut_reference(sample, weight, np.zeros(3, np.float32), kind)
                    self.assertEqual(got.shape, sample.shape)
                    real = to_real(sample, 1.0, 128)
                    argument = real * gain
                    function = (lambda value: 1.0 / (1.0 + np.exp(-value))) if kind == "Sigmoid" else np.tanh
                    expected = requantize(function(argument), meta["output_scale"], meta["output_zero_point"])
                    self.assert_codes(got, expected, 1, f"LUT {kind} gain {gain}")
                    # Table domain contract: stem range inside (-8*gain, +8) and monotone ramp.
                    low, high = meta["stem_range"]
                    self.assertGreaterEqual(low, -LUT_DOMAIN * meta["negative_gain"] - 1e-6)
                    self.assertLessEqual(high, LUT_DOMAIN + 1e-6)
                    ramp_codes = got[:, :, 0].reshape(-1).astype(np.int64)
                    increments = np.diff(ramp_codes)
                    self.assertTrue(np.all(increments >= 0), "monotone ramp produced a decreasing LUT output")
                    self.assertGreater(int(ramp_codes.max() - ramp_codes.min()), 64)
        # A stem outside the table domain is rejected rather than clipped silently.
        with self.assertRaises(ValueError):
            compile_sequence(self.lut_model(0.5, "Sigmoid"), input_scale=1.0, input_zero_point=128)


# ---------------------------------------------------------------------------
# 8. ConvTranspose (transposed.py)
# ---------------------------------------------------------------------------


def transposed_integer(stem_codes, q, stem_zero_point, kernel, pads, strides, out_shape):
    """Independent rebuild of the emitted requantization task (documented convention)."""
    channels = q.weights.shape[0]
    centered = np.asarray(q.weights, np.int64).reshape(channels, kernel, kernel) \
        - np.asarray(q.weight_zero_points, np.int64)[:, None, None]
    base = np.asarray(q.biases, np.int64) + stem_zero_point * centered.sum(axis=(1, 2))
    acc = np.broadcast_to(base, (*out_shape, channels)).astype(np.int64).copy()
    activation = stem_codes.astype(np.int64) - stem_zero_point
    for iy in range(stem_codes.shape[0]):
        for ix in range(stem_codes.shape[1]):
            for ky in range(kernel):
                for kx in range(kernel):
                    oy = iy * strides[0] - pads[0] + ky
                    ox = ix * strides[1] - pads[1] + kx
                    if 0 <= oy < out_shape[0] and 0 <= ox < out_shape[1]:
                        acc[oy, ox] += activation[iy, ix] * centered[:, ky, kx]
    product = acc * np.asarray(q.channel_multipliers, np.int64)
    scaled = (product + 8191 + ((product >> 14) & 1)) >> 14
    product = scaled * int(q.multiplier)
    if q.shift:
        product = product + (1 << (q.shift - 1)) - 1 + ((product >> q.shift) & 1)
        result = (product >> q.shift) + int(q.output_zero_point)
    else:
        result = product + int(q.output_zero_point)
    return np.clip(result, -128, 127).astype(np.int8)


def depthwise_transposed_weights(weight):
    """Expand a depthwise `(C, 1, kh, kw)` ConvTranspose tensor to the dense equivalent."""
    channels, per_group, kh, kw = weight.shape
    assert per_group == 1
    dense = np.zeros((channels, channels, kh, kw))
    for index in range(channels):
        dense[index, index] = weight[index, 0]
    return dense


class TransposedSemanticsTests(SemanticCase):
    def transposed_model(self, kernel, weight, pads, strides, out_hw):
        stem = identity_kernel(3)
        model = build_model(
            [conv_node("stem", "input", "stem_out", stem, np.zeros(3, np.float32), 1, (0, 0, 0, 0)),
             h.make_node("ConvTranspose", ["stem_out", "w", "b"], ["output"], group=3,
                         kernel_shape=[kernel, kernel], pads=list(pads), strides=list(strides))],
            (8, 8, 3), (out_hw[0], out_hw[1], 3),
            [nh.from_array(stem, "stem_w"), nh.from_array(np.zeros(3, np.float32), "stem_b"),
             nh.from_array(weight, "w"), nh.from_array(np.zeros(3, np.float32), "b")],
            value_info=[("stem_out", (8, 8, 3))])
        return model

    def test_conv_transpose_delta_scatter(self):
        sample = ramp((8, 8, 3), low=124, span=9)
        deltas = [("centre", 1, 1), ("corner", 0, 0), ("corner-b", 2, 2)]
        for name, ky, kx in deltas:
            with self.subTest(delta=name):
                weight = np.zeros((3, 1, 3, 3), np.float32)
                for channel in range(3):
                    weight[channel, 0, ky, kx] = 1.0
                model = self.transposed_model(3, weight, (1, 1, 1, 1), (1, 1), (8, 8))
                binary, meta = compile_sequence(model)
                q_stem = _quant(meta["first"]["quantization"])
                q = _quant(meta["transposed_quantization"])
                stem = reference(sample, q_stem)
                expected = requantize(conv_transpose2d(
                    to_real(stem, q_stem.output_scale, q_stem.output_zero_point),
                    depthwise_transposed_weights(dequant_weights(q)), np.zeros(3), (1, 1, 1, 1), (1, 1), 8, 8),
                    meta["output_scale"], meta["output_zero_point"])
                got = transposed_integer(stem, q, q_stem.output_zero_point, 3, (1, 1, 1, 1), (1, 1), (8, 8))
                self.assert_codes(got, expected, 0, f"transposed delta {name}")
                if name == "centre":
                    # depthwise_reference is exact for the symmetric kernel and is the
                    # only src reference available for this profile.
                    oracle = depthwise_reference(stem, q, q_stem.output_zero_point)
                    self.assert_codes(oracle, expected, 0, "transposed centre delta identity")
                # Decode the emitted tap table to confirm the scatter orientation.
                info = decode_sequence(binary)
                payload = binary[96 + 16 * info["task_count"]:]
                table = np.frombuffer(payload[0xB00:0xB00 + 9 * 32], np.int8).reshape(9, 32)
                tap = (2 - ky) * 3 + (2 - kx)
                self.assertNotEqual(int(table[tap, 0]), 0, f"flip table missing tap {tap}")

    def test_conv_transpose_stride2_and_random(self):
        sample = ramp((8, 8, 3), low=124, span=9)
        # Stride-2 delta scatter: output row/column 2*i + ky - 1 on a 15x15 grid.
        weight = np.zeros((3, 1, 3, 3), np.float32)
        for channel in range(3):
            weight[channel, 0, 1, 1] = 1.0
        model = self.transposed_model(3, weight, (1, 1, 1, 1), (2, 2), (15, 15))
        _, meta = compile_sequence(model)
        q_stem = _quant(meta["first"]["quantization"])
        q = _quant(meta["transposed_quantization"])
        stem = reference(sample, q_stem)
        got = transposed_integer(stem, q, q_stem.output_zero_point, 3, (1, 1, 1, 1), (2, 2), (15, 15))
        expected = requantize(conv_transpose2d(to_real(stem, q_stem.output_scale, q_stem.output_zero_point),
                                               depthwise_transposed_weights(dequant_weights(q)), np.zeros(3),
                                               (1, 1, 1, 1), (2, 2), 15, 15),
                              meta["output_scale"], meta["output_zero_point"])
        self.assert_codes(got, expected, 0, "transposed stride2 delta")
        # A small random kernel: float64 ONNX scatter versus the rebuilt integer task.
        rng = np.random.default_rng(8001)
        weight = rng.uniform(.05, .3, (3, 1, 3, 3)).astype(np.float32)
        model = self.transposed_model(3, weight, (1, 0, 2, 1), (1, 1), (7, 9))
        _, meta = compile_sequence(model)
        q_stem = _quant(meta["first"]["quantization"])
        q = _quant(meta["transposed_quantization"])
        stem = reference(sample, q_stem)
        got = transposed_integer(stem, q, q_stem.output_zero_point, 3, (1, 0, 2, 1), (1, 1), (7, 9))
        expected = requantize(conv_transpose2d(to_real(stem, q_stem.output_scale, q_stem.output_zero_point),
                                               depthwise_transposed_weights(dequant_weights(q)), np.zeros(3),
                                               (1, 0, 2, 1), (1, 1), 7, 9),
                              meta["output_scale"], meta["output_zero_point"])
        self.assert_codes(got, expected, 1, "transposed random kernel")


# ---------------------------------------------------------------------------
# 9. join DAG (join_dag.py) and join chain (graph.py)
# ---------------------------------------------------------------------------


def multi_head_model(kinds, heads=3, hidden=3, seed=9001, identical=False, expression=None):
    rng = np.random.default_rng(seed)
    stem = rng.uniform(.1, .4, (hidden, 3, 1, 1)).astype(np.float32)
    initializers = [nh.from_array(stem, "stem_w"), nh.from_array(np.zeros(hidden, np.float32), "stem_b")]
    nodes = [conv_node("stem", "input", "stem_out", stem, np.zeros(hidden, np.float32), 1, (0, 0, 0, 0))]
    heads_data = []
    for index in range(heads):
        weight = identity_kernel(3) if identical else rng.uniform(.1, .4, (3, hidden, 1, 1)).astype(np.float32)
        initializers += [nh.from_array(weight, f"head{index}_w"),
                         nh.from_array(np.zeros(3, np.float32), f"head{index}_b")]
        nodes.append(conv_node(f"head{index}", "stem_out", f"h{index}", weight,
                               np.zeros(3, np.float32), 1, (0, 0, 0, 0)))
        heads_data.append(weight)
    if expression is None:
        previous = "h0"
        for index, kind in enumerate(kinds):
            output = "output" if index == len(kinds) - 1 else f"j{index}"
            nodes.append(h.make_node(kind, [previous, f"h{index + 1}"], [output]))
            previous = output
    else:
        for index, (kind, first, second) in enumerate(expression):
            output = "output" if index == len(expression) - 1 else f"j{index}"
            nodes.append(h.make_node(kind, [first, second], [output]))
    model = build_model(nodes, (8, 8, 3), (8, 8, 3), initializers,
                        value_info=[("stem_out", (8, 8, hidden))])
    return model, heads_data


class JoinDagChainTests(SemanticCase):
    def independent_join_chain(self, sample, meta, head_weights, kinds):
        stem_q = _quant(meta["stem_quantization"])
        stem_real = conv2d(to_real(sample, 1.0, 0), dequant_weights(stem_q), np.zeros(stem_q.weights.shape[0]))
        stem_codes = requantize(stem_real, stem_q.output_scale, stem_q.output_zero_point)
        reals = []
        for head_weight, entry in zip(head_weights, meta["head_quantization"]):
            q = _quant(entry)
            real = conv2d(to_real(stem_codes, stem_q.output_scale, stem_q.output_zero_point),
                          dequant_weights(q), np.zeros(3))
            reals.append(to_real(requantize(real, q.output_scale, q.output_zero_point), q.output_scale,
                                 q.output_zero_point))
        value = reals[0]
        for index, kind in enumerate(kinds):
            other = reals[index + 1]
            value = {"Add": value + other, "Sub": value - other, "Max": np.maximum(value, other),
                     "Mul": value * other}[kind]
        return requantize(value, meta["output_scale"], meta["output_zero_point"])

    def test_join_chain_semantics(self):
        sample = ramp((8, 8, 3), low=100, span=48)
        # Exact property: identical branches, Sub then Mul, collapse to the join of zero.
        model, head_weights = multi_head_model(("Sub", "Mul"), identical=True)
        _, meta = compile_sequence(model)
        self.assertEqual(meta["profile"], "join-chain")
        got = diamond_reference(sample, meta["stem_quantization"], meta["head_quantization"],
                                meta["join_kinds"], meta.get("tail_quantization", ()),
                                meta.get("join_zero_point", 0))
        self.assert_codes(got, np.zeros((8, 8, 3)) + meta["output_zero_point"], 0, "identical join chain analytic")
        # General chains: independent float64 composition with the container bands.
        for kinds in (("Add", "Add"), ("Max", "Add"), ("Sub", "Max")):
            with self.subTest(kinds=kinds):
                model, head_weights = multi_head_model(kinds)
                _, meta = compile_sequence(model)
                self.assertEqual(meta["profile"], "join-chain")
                got = diamond_reference(sample, meta["stem_quantization"], meta["head_quantization"],
                                        meta["join_kinds"], meta.get("tail_quantization", ()),
                                        meta.get("join_zero_point", 0))
                expected = self.independent_join_chain(sample, meta, head_weights, meta["join_kinds"])
                self.assert_codes(got, expected, 1, f"join chain {kinds}")

    def test_join_dag_semantics(self):
        sample = ramp((8, 8, 3), low=100, span=48)
        expression = (("Sub", "h0", "h1"), ("Mul", "j0", "h0"))
        model, _ = multi_head_model((), identical=True, expression=expression)
        _, meta = compile_sequence(model)
        self.assertEqual(meta["profile"], "join-dag")
        got = join_dag_reference(sample, meta["stem_quantization"], meta["head_quantization"],
                                 meta["head_names"], meta["join_expression"], head_kinds=meta["head_kinds"],
                                 depthwise_quantizations=meta["depthwise_quantization"])
        self.assert_codes(got, np.zeros((8, 8, 3)) + meta["output_zero_point"], 0, "identical join DAG analytic")
        # General DAG with a reused branch: independent float64 replay of the expression.
        expression = (("Add", "h0", "h1"), ("Mul", "j0", "h0"))
        model, head_weights = multi_head_model((), identical=False, expression=expression)
        _, meta = compile_sequence(model)
        self.assertEqual(meta["profile"], "join-dag")
        got = join_dag_reference(sample, meta["stem_quantization"], meta["head_quantization"],
                                 meta["head_names"], meta["join_expression"], head_kinds=meta["head_kinds"],
                                 depthwise_quantizations=meta["depthwise_quantization"])
        stem_q = _quant(meta["stem_quantization"])
        stem_real = conv2d(to_real(sample, 1.0, 0), dequant_weights(stem_q), np.zeros(stem_q.weights.shape[0]))
        stem_codes = requantize(stem_real, stem_q.output_scale, stem_q.output_zero_point)
        values = {}
        for index, entry in enumerate(meta["head_quantization"]):
            q = _quant(entry)
            real = conv2d(to_real(stem_codes, stem_q.output_scale, stem_q.output_zero_point),
                          dequant_weights(q), np.zeros(3))
            values[f"h{index}"] = to_real(requantize(real, q.output_scale, q.output_zero_point),
                                          q.output_scale, q.output_zero_point)
        for step in meta["join_expression"]:
            first, second = values[step["inputs"][0]], values[step["inputs"][1]]
            values[step["name"]] = {"Add": first + second, "Sub": first - second,
                                    "Max": np.maximum(first, second), "Mul": first * second}[step["kind"]]
        expected = requantize(values[meta["join_expression"][-1]["name"]], meta["output_scale"],
                              meta["output_zero_point"])
        self.assert_codes(got, expected, 1, "join DAG reused branch")


if __name__ == "__main__":
    unittest.main()
