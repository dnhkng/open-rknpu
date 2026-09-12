"""SPDX-License-Identifier: MIT

Chain walk: a constant elementwise stage inside an op-level Conv chain.

`open_rknpu.walk` lowers `Conv[/Relu] -> Add|Sub|Max|Mul(constant) -> (pool ->)*
Conv[/Relu] ...` one op at a time. The elementwise stage reuses the standalone
elementwise emitter's 78-word DPU program (`open_rknpu.elementwise` registers, through
`_join_fields`, exactly as the join walk does) with its second operand read from a
payload constant grid, so these tests pin four things:

* the accepted stages compile and the container decodes with the expected task count;
* the band bookkeeping: the feeding Conv is re-quantized onto the emitter's shared
  zero-point-zero operand scale, the stage's output band is the emitter's, and the
  constant payload is the code grid the emitter wrote;
* the integer reference composes the verified per-op references on exactly those
  bands, reproducing the expected bytes;
* every rejected form keeps its specific `ValueError` (message pinned verbatim).
"""
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

from open_rknpu.chain import native_reference
from open_rknpu.elementwise import add_reference, max_reference, mul_reference, sub_reference
from open_rknpu.quantization import reference
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence, payload_base
from open_rknpu.walk import (chain_walk_reference, compile_chain_walk, load_quantizations,
                             parse_chain)

EW_ORACLE = {"Add": add_reference, "Sub": sub_reference, "Max": max_reference, "Mul": mul_reference}
SURFACE = ((8 * 8 + 3) // 4) * 64
CONV0 = [nh.from_array(np.random.default_rng(1).uniform(-.6, .6, (8, 3, 1, 1)).astype(np.float32), "w0"),
         nh.from_array(np.random.default_rng(2).uniform(-1, 1, (8,)).astype(np.float32), "b0")]
CONV1 = [nh.from_array(np.random.default_rng(3).uniform(-.6, .6, (3, 8, 1, 1)).astype(np.float32), "w1"),
         nh.from_array(np.random.default_rng(4).uniform(-1, 1, (3,)).astype(np.float32), "b1")]


def constant_of(mode, channels=8, height=8, width=8, seed=4):
    """A deterministic constant in one of the three accepted broadcast modes."""
    rng = np.random.default_rng(seed)
    if mode == "scalar":
        return np.float32(0.5)
    if mode == "per-channel":
        return rng.uniform(.2, 1.2, (1, channels, 1, 1)).astype(np.float32)
    return rng.uniform(.2, 1.2, (1, channels, height, width)).astype(np.float32)


def graph_model(nodes, inits, out_name="out", out_shape=(1, 3, 8, 8), height=8, width=8):
    graph = h.make_graph(nodes, "walk_elementwise",
                         [h.make_tensor_value_info("input", 1, [1, 3, height, width])],
                         [h.make_tensor_value_info(out_name, 1, list(out_shape))],
                         list(inits))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def chain_model(kind="Add", mode="per-channel", pooled=False, height=8, width=8,
                second_kernel=1, relu_first=False):
    """`Conv[/Relu] -> kind(constant) -> [MaxPool] -> Conv`, deterministic."""
    nodes = [h.make_node("Conv", ["input", "w0", "b0"], ["c0"], kernel_shape=[1, 1])]
    inits = list(CONV0) + [nh.from_array(constant_of(mode, 8, height, width), "k")]
    source = "c0"
    if relu_first:
        nodes.append(h.make_node("Relu", [source], ["r0"]))
        source = "r0"
    nodes.append(h.make_node(kind, [source, "k"], ["e0"]))
    source, out_height, out_width = "e0", height, width
    if pooled:
        nodes.append(h.make_node("MaxPool", [source], ["p0"],
                                 kernel_shape=[2, 2], strides=[2, 2]))
        source, out_height, out_width = "p0", height // 2, width // 2
    nodes.append(h.make_node("Conv", [source, "w1", "b1"], ["out"],
                             kernel_shape=[second_kernel] * 2,
                             pads=[second_kernel // 2] * 4))
    inits += [nh.from_array(np.random.default_rng(5).uniform(
        -.6, .6, (3, 8, second_kernel, second_kernel)).astype(np.float32), "w1"),
        nh.from_array(np.random.default_rng(6).uniform(-1, 1, (3,)).astype(np.float32), "b1")]
    return graph_model(nodes, inits, "out", (1, 3, out_height, out_width), height, width)


def standalone_model(kind, height=8, width=8, seed=6):
    """A two-branch elementwise graph: the emitter whose 78-word program is reused."""
    rng = np.random.default_rng(seed)
    wa = rng.uniform(.05, .3, (8, 3, 1, 1)).astype(np.float32)
    wb = rng.uniform(.05, .4, (8, 3, 1, 1)).astype(np.float32)
    nodes = [h.make_node("Conv", ["input", "wa", "ba"], ["a"], kernel_shape=[1, 1]),
             h.make_node("Conv", ["input", "wb", "bb"], ["b"], kernel_shape=[1, 1]),
             h.make_node(kind, ["a", "b"], ["out"])]
    inits = [nh.from_array(wa, "wa"), nh.from_array(np.zeros(8, np.float32), "ba"),
             nh.from_array(wb, "wb"), nh.from_array(np.zeros(8, np.float32), "bb")]
    return graph_model(nodes, inits, "out", (1, 8, height, width), height, width)


def saved(model):
    folder = tempfile.mkdtemp()
    path = Path(folder) / "walk_elementwise.onnx"
    onnx.save(model, path)
    return path


def elementwise_task(binary):
    """The decoded 78-word elementwise program of a container: `{register: value}`."""
    info = decode_sequence(binary)
    base = payload_base(info)
    task = next(task for task in info["tasks"]
                if (task["register_count"], task["enable"]) == (78, 24))
    values = {}
    for index in range(78):
        word = struct.unpack_from("<Q", binary, base + task["command_offset"] + index * 8)[0]
        values[word & 0xFFFF] = (word >> 16) & 0xFFFFFFFF
    return values


def constant_codes(constant, scale, channels=8, height=8, width=8):
    """The code grid an emitter writes, computed independently of the walk."""
    expanded = np.broadcast_to(np.asarray(constant, np.float32),
                               (1, channels, height, width))[0]
    return np.clip(np.rint(expanded / scale), -128, 127).astype(np.int8).transpose(1, 2, 0)


def pool2x2(codes):
    height, width, channels = codes.shape
    blocked = codes.astype(np.int64).reshape(height // 2, 2, width // 2, 2, channels)
    return np.clip(blocked.max(axis=(1, 3)), -128, 127).astype(np.int8)


class WalkElementwiseAcceptanceTests(unittest.TestCase):
    def test_every_stage_compiles_and_decodes(self):
        for kind in ("Add", "Sub", "Max", "Mul"):
            for pooled in (False, True):
                with self.subTest(kind=kind, pooled=pooled):
                    binary, meta = compile_chain_walk(chain_model(kind=kind, pooled=pooled))
                    info = decode_sequence(binary)
                    self.assertEqual(meta["profile"], "chain-walk")
                    self.assertEqual(meta["walk_elementwise"], [kind])
                    self.assertEqual(meta["walk_ops"],
                                     ["conv", "ew", "MaxPool", "conv"] if pooled
                                     else ["conv", "ew", "conv"])
                    self.assertEqual(info["task_count"], 4 if pooled else 3)
                    self.assertEqual(list(meta["output_shape"]),
                                     [1, 3, 4, 4] if pooled else [1, 3, 8, 8])

    def test_dispatch_accepts_the_maintainer_probes(self):
        # `Conv -> Add(constant) -> MaxPool -> Conv` and `Conv -> Add(const) -> Conv`
        # used to be rejected by the depthwise dispatcher. They are the walk's now.
        for pooled in (False, True):
            for mode in ("scalar", "per-channel", "spatial"):
                with self.subTest(pooled=pooled, mode=mode):
                    _, meta = compile_sequence(saved(chain_model(mode=mode, pooled=pooled)))
                    self.assertEqual(meta["profile"], "chain-walk")
                    self.assertEqual(meta["walk_elementwise"], ["Add"])
        # Mul is only reachable through dispatch for a spatial constant: normalize
        # folds a scalar/per-channel Mul into the Conv weights before the walk sees it.
        _, meta = compile_sequence(saved(chain_model(kind="Mul", mode="spatial")))
        self.assertEqual(meta["profile"], "chain-walk")
        self.assertEqual(meta["walk_elementwise"], ["Mul"])

    def test_walk_ops_survive_a_k3_successor(self):
        binary, meta = compile_chain_walk(chain_model(second_kernel=3))
        self.assertEqual(meta["walk_ops"], ["conv", "ew", "conv"])
        self.assertEqual(decode_sequence(binary)["task_count"], 3)

    def test_relu_before_the_stage_is_folded_into_the_feeding_conv(self):
        # The class is Conv[/Relu] -> stage: the Relu is folded into the feeding Conv
        # exactly as everywhere else in the walk, and the reference still composes.
        model = chain_model(relu_first=True)
        binary, meta = compile_chain_walk(model)
        self.assertEqual(meta["walk_ops"], ["conv", "ew", "conv"])
        self.assertEqual(meta["walk_activations"], ["Relu", None])
        self.assertEqual(decode_sequence(binary)["task_count"], 3)
        image = np.random.default_rng(11).integers(0, 256, (8, 8, 3), dtype=np.uint8)
        spec = parse_chain(model.graph)
        bands = load_quantizations(meta)
        first = reference(image, bands[0])
        codes = constant_codes(spec["ops"][1]["constant"], bands[1]["constant_scale"])
        expected = native_reference(EW_ORACLE["Add"](first, codes), bands[-1], 0)
        np.testing.assert_array_equal(
            chain_walk_reference(image, bands, spec["ops"]), expected)


class WalkElementwiseReferenceTests(unittest.TestCase):
    def test_reference_composes_the_verified_emitters(self):
        # The walk's reference must be exactly `reference` (first Conv), the emitted
        # constant codes, the standalone `*_reference` for the op and `native_reference`
        # (later Conv) - i.e. the verified per-op arithmetic on the emitted bands.
        image = np.random.default_rng(9).integers(0, 256, (8, 8, 3), dtype=np.uint8)
        for kind in ("Add", "Sub", "Max", "Mul"):
            for mode in ("scalar", "per-channel", "spatial"):
                for pooled in (False, True):
                    with self.subTest(kind=kind, mode=mode, pooled=pooled):
                        model = chain_model(kind=kind, mode=mode, pooled=pooled)
                        _, meta = compile_chain_walk(model)
                        spec = parse_chain(model.graph)
                        bands = load_quantizations(meta)
                        first = reference(image, bands[0])
                        codes = constant_codes(spec["ops"][1]["constant"],
                                               bands[1]["constant_scale"])
                        grid = EW_ORACLE[kind](first, codes)
                        if pooled:
                            grid = pool2x2(grid)
                        expected = native_reference(grid, bands[-1], 0)
                        got = chain_walk_reference(image, bands, spec["ops"])
                        np.testing.assert_array_equal(got, expected)

    def test_decoded_band_matches_the_reference(self):
        for kind in ("Add", "Sub", "Max", "Mul"):
            with self.subTest(kind=kind):
                binary, meta = compile_chain_walk(chain_model(kind=kind))
                info = decode_sequence(binary)
                bands = load_quantizations(meta)
                # The container's header band is the last Conv's band, which is the
                # band the reference reads.
                self.assertEqual(info["output_scale"], meta["output_scale"])
                self.assertEqual(info["output_zero_point"], meta["output_zero_point"])
                self.assertEqual(bands[-1].output_scale, meta["output_scale"])
                self.assertEqual(bands[-1].output_zero_point, meta["output_zero_point"])
                # The elementwise stage's output is zero-centred on the feeding Conv's
                # operand scale, which is what the next Conv's reference assumes.
                self.assertEqual(bands[0].output_zero_point, 0)
                self.assertEqual(bands[1]["output_zero_point"], 0)
                self.assertEqual(bands[1]["scale"], bands[0].output_scale)

    def test_constant_payload_is_the_emitted_code_grid(self):
        for mode in ("scalar", "per-channel", "spatial"):
            with self.subTest(mode=mode):
                model = chain_model(mode=mode)
                binary, meta = compile_chain_walk(model)
                spec = parse_chain(model.graph)
                band = load_quantizations(meta)[1]
                task = elementwise_task(binary)
                base = payload_base(decode_sequence(binary))
                block = np.frombuffer(binary, np.int8, count=SURFACE,
                                      offset=base + task[0x5038]).reshape(8, 8, 16)
                wanted = constant_codes(spec["ops"][1]["constant"], band["constant_scale"])
                np.testing.assert_array_equal(block[:, :, :8], wanted)
                self.assertFalse(block[:, :, 8:].any())
                self.assertEqual(task[0x5034], 0x40000004)
                self.assertEqual(task[0x5040], SURFACE)

    def test_band_rule_is_the_standalone_emitters(self):
        # Add/Sub/Max share one operand scale and double it; Mul keeps the two operand
        # scales and folds them into 128 * product.
        for kind in ("Add", "Sub", "Max", "Mul"):
            with self.subTest(kind=kind):
                _, meta = compile_chain_walk(chain_model(kind=kind))
                band = load_quantizations(meta)[1]
                scale = band["scale"]
                if kind == "Mul":
                    self.assertAlmostEqual(
                        band["output_scale"],
                        float(np.float32(128 * scale * band["constant_scale"])), places=6)
                else:
                    self.assertEqual(band["constant_scale"], scale)
                    self.assertAlmostEqual(band["output_scale"],
                                           float(np.float32(2 * scale)), places=6)

    def test_register_program_is_the_standalone_emitters(self):
        # Only the three address registers (primary operand, secondary operand, output)
        # may differ: every functional field is the standalone elementwise emitter's.
        for kind in ("Add", "Sub", "Max", "Mul"):
            with self.subTest(kind=kind):
                standalone, _ = compile_sequence(saved(standalone_model(kind)))
                reference_registers = elementwise_task(standalone)
                walked, _ = compile_chain_walk(chain_model(kind=kind))
                registers = elementwise_task(walked)
                self.assertEqual(set(registers), set(reference_registers))
                for register, value in reference_registers.items():
                    if register in (0x4020, 0x5018, 0x5038):
                        continue
                    self.assertEqual(registers[register], value, "#%x" % register)

    def test_zero_constant_add_preserves_the_real_value(self):
        # An exact property: adding real zero to a zero-point-zero grid leaves the
        # dequantized value unchanged (the codes halve and the scale doubles).
        model = chain_model(kind="Add", mode="spatial")
        model.graph.initializer[2].CopyFrom(nh.from_array(np.zeros((1, 8, 8, 8), np.float32), "k"))
        model = onnx.shape_inference.infer_shapes(model)
        _, meta = compile_chain_walk(model)
        spec = parse_chain(model.graph)
        bands = load_quantizations(meta)
        image = np.random.default_rng(2).integers(0, 256, (8, 8, 3), dtype=np.uint8)
        first = reference(image, bands[0])
        # The reference walks the declared op list, so the first two ops stop at the
        # stage's own grid.
        stage = chain_walk_reference(image, bands[:2], spec["ops"][:2])
        np.testing.assert_array_equal(stage, np.rint(first.astype(np.int32) / 2).astype(np.int8))
        self.assertEqual(bands[1]["constant_scale"], bands[1]["scale"])
        np.testing.assert_array_equal(
            chain_walk_reference(image, bands, spec["ops"]),
            native_reference(stage, bands[-1], 0))


class WalkElementwiseRejectionTests(unittest.TestCase):
    """Every out-of-envelope form, with the message pinned verbatim."""

    def _message(self, model):
        with self.assertRaises(ValueError) as caught:
            compile_chain_walk(model)
        return str(caught.exception)

    def _conv(self, source, name, out_channels, in_channels, output):
        return h.make_node("Conv", [source, name + "_w", name + "_b"], [output], kernel_shape=[1, 1])

    def _inits(self, name, out_channels, in_channels):
        rng = np.random.default_rng(len(name))
        return [nh.from_array(rng.uniform(-.5, .5, (out_channels, in_channels, 1, 1)).astype(np.float32),
                              name + "_w"),
                nh.from_array(rng.uniform(-1, 1, (out_channels,)).astype(np.float32), name + "_b")]

    def test_two_external_operands(self):
        model = graph_model(
            [self._conv("input", "a", 8, 3, "c0"),
             h.make_node("Add", ["input", "input"], ["e0"]),
             self._conv("e0", "b", 3, 8, "out")],
            self._inits("a", 8, 3) + self._inits("b", 3, 8))
        self.assertEqual(
            self._message(model),
            "walk elementwise stage 1 (Add) requires one operand to be the chain tensor "
            "(two external operands are unsupported)")

    def test_non_constant_operand(self):
        model = graph_model(
            [self._conv("input", "a", 8, 3, "c0"),
             h.make_node("Add", ["c0", "input"], ["e0"]),
             self._conv("e0", "b", 3, 8, "out")],
            self._inits("a", 8, 3) + self._inits("b", 3, 8))
        self.assertEqual(
            self._message(model),
            "walk elementwise stage 1 (Add) requires the non-chain operand to be an "
            "immutable initializer constant")

    def test_successor_must_reach_a_conv(self):
        message = ("walk elementwise stage 1 (Add) must be followed (through 2x2 pools) "
                   "by a Conv[/Relu]")
        relu = graph_model(
            [self._conv("input", "a", 8, 3, "c0"),
             h.make_node("Add", ["c0", "k"], ["e0"]),
             h.make_node("Relu", ["e0"], ["r0"]),
             self._conv("r0", "b", 3, 8, "out")],
            self._inits("a", 8, 3) + self._inits("b", 3, 8) +
            [nh.from_array(constant_of("per-channel"), "k")])
        self.assertEqual(self._message(relu), message)
        # A terminal elementwise stage is the same bound.
        terminal = graph_model(
            [self._conv("input", "a", 8, 3, "c0"),
             h.make_node("Add", ["c0", "k"], ["e0"])],
            self._inits("a", 8, 3) + [nh.from_array(constant_of("per-channel"), "k")],
            out_name="e0", out_shape=(1, 8, 8, 8))
        self.assertEqual(self._message(terminal), message)

    def test_one_elementwise_stage_per_chain(self):
        model = graph_model(
            [self._conv("input", "a", 8, 3, "c0"),
             h.make_node("Add", ["c0", "k"], ["e0"]),
             self._conv("e0", "b", 8, 8, "c1"),
             h.make_node("Max", ["c1", "k"], ["e1"]),
             self._conv("e1", "c", 3, 8, "out")],
            self._inits("a", 8, 3) + self._inits("b", 8, 8) + self._inits("c", 3, 8) +
            [nh.from_array(constant_of("per-channel"), "k")])
        self.assertEqual(self._message(model),
                         "walk elementwise stage 2 (Max) supports one elementwise stage per chain")

    def test_constant_shape_bound(self):
        model = chain_model()
        model.graph.initializer[2].CopyFrom(nh.from_array(np.zeros((1, 1, 8, 8), np.float32), "k"))
        model = onnx.shape_inference.infer_shapes(model)
        self.assertEqual(
            self._message(model),
            "walk elementwise stage 1 (Add) constant shape (1, 1, 8, 8) is outside the "
            "broadcast bound (scalar, [1,C,1,1] or [1,C,H,W])")

    def test_non_finite_constant(self):
        model = graph_model(
            [self._conv("input", "a", 8, 3, "c0"),
             h.make_node("Add", ["c0", "k"], ["e0"]),
             self._conv("e0", "b", 3, 8, "out")],
            self._inits("a", 8, 3) + self._inits("b", 3, 8) +
            [nh.from_array(np.float32(np.inf), "k")])
        self.assertEqual(self._message(model),
                         "walk elementwise stage 1 (Add) requires a finite float32 constant")

    def test_sub_constant_must_be_the_subtrahend(self):
        model = graph_model(
            [self._conv("input", "a", 8, 3, "c0"),
             h.make_node("Sub", ["k", "c0"], ["e0"]),
             self._conv("e0", "b", 3, 8, "out")],
            self._inits("a", 8, 3) + self._inits("b", 3, 8) +
            [nh.from_array(constant_of("per-channel"), "k")])
        self.assertEqual(
            self._message(model),
            "walk elementwise stage 1 (Sub) requires the chain tensor minus the constant "
            "(chain tensor first)")

    def test_elementwise_must_follow_a_conv(self):
        model = graph_model(
            [self._conv("input", "a", 8, 3, "c0"),
             h.make_node("MaxPool", ["c0"], ["p0"], kernel_shape=[2, 2], strides=[2, 2]),
             h.make_node("Add", ["p0", "k"], ["e0"]),
             self._conv("e0", "b", 3, 8, "out")],
            self._inits("a", 8, 3) + self._inits("b", 3, 8) +
            [nh.from_array(constant_of("per-channel"), "k")])
        self.assertEqual(self._message(model),
                         "walk elementwise stage 1 (Add) must directly follow the chain Conv[/Relu]")

    def test_non_chain_graph_keeps_the_generic_message(self):
        model = graph_model(
            [self._conv("input", "a", 8, 3, "c0"),
             h.make_node("Sigmoid", ["c0"], ["out"])],
            self._inits("a", 8, 3), out_shape=(1, 8, 8, 8))
        self.assertEqual(self._message(model), "unsupported chain walk graph")


if __name__ == "__main__":
    unittest.main()
