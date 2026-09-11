# SPDX-License-Identifier: MIT
"""Pins the staged 2x2 pooling reduction and the elementwise DAG emitters.

`reduction.py` composes one dense Conv (or Conv+Relu) with three explicit 2x2
stride-2 pool stages down to a 1x1 output, while `elementwise_chain.py`,
`elementwise_multi.py` and `native_elementwise.py` emit fan-out DAGs whose
integer stages run at fixed bands.  The modules here pin:

* the accepted reduction profiles (MaxPool profile 5 / AveragePool profile 6,
  three levels, 4 tasks, 1x1x3 output) and their staged-pool rejections;
* the elementwise chain `Mul(...Mul(Op(a,b),a)...,a)`, the multi-input v5 DAG
  `Mul(...Mul(Op(a,b),c_k)...,c)` and the native16 two-input DAG, with task
  counts, chains and output shapes;
* that every accepted graph's compiled integer reference is reproduced by an
  independent float64 implementation that dequantizes the compiled branch codes
  with their container bands, runs the elementwise/pool arithmetic, and
  requantizes with the output band (<= 1 LSB);
* the multi-input external-input ordering contract, and the documented shape,
  input-count, distinct-join and operand-zero-point rejections.

No board is used: the integer references are the emitters' own, and the
independent values are recomputed here from the compiled quantization metadata.
"""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from onnx.reference import ReferenceEvaluator

from open_rknpu.compiler import compile_model
from open_rknpu.elementwise import add_reference, max_reference, mul_reference, sub_reference
from open_rknpu.elementwise_chain import chain_reference, compile_elementwise_dag
from open_rknpu.elementwise_multi import compile_multi_input_dag, multi_input_reference
from open_rknpu.model import decode, encode
from open_rknpu.native_elementwise import compile_native_elementwise
from open_rknpu.quantization import Quantization, reference
from open_rknpu.reduction import compile_reduction
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence


def save_model(model):
    folder = tempfile.TemporaryDirectory()
    path = Path(folder.name) / "model.onnx"
    onnx.save(model, path)
    return folder, path

def save_and_compile(model, function):
    folder, path = save_model(model)
    try:
        return function(path)
    finally:
        folder.cleanup()

def compile_legacy(model):
    return save_and_compile(model, compile_model)

def compile_sequence_model(model, *args, **kwargs):
    folder, path = save_model(model)
    try:
        return compile_sequence(path, *args, **kwargs)
    finally:
        folder.cleanup()

def quantization_from(params):
    return Quantization(**{key: np.array(value) if isinstance(value, list) else value
                           for key, value in params.items()})

def quantization_of(meta):
    return quantization_from(meta["quantization"])

def dequantize(codes, band):
    return (codes.astype(np.float64) - band["zero_point"]) * band["scale"]

def requantize(values, band):
    return np.clip(np.rint(values / band["scale"]) + band["zero_point"], -128, 127).astype(np.int8)

def stage_reference(codes, kind, levels):
    """Integer staged pooling reference: MaxPool is exact, AveragePool rounds twice."""
    result = codes.astype(np.int64)
    size = result.shape[0]
    for _ in range(levels):
        size //= 2
        folded = result.reshape(size, 2, size, 2, result.shape[2])
        result = folded.max(axis=(1, 3)) if kind == "MaxPool" else np.rint(folded.sum(axis=(1, 3)) / 4)
    return result.astype(np.int8)

def float_reference(model, inputs):
    values = inputs.astype(np.float32)[None].transpose(0, 3, 1, 2)
    return ReferenceEvaluator(model).run(None, {"input": values})[0][0].transpose(1, 2, 0).astype(np.float64)

# --- builders -----------------------------------------------------------------

def reduction_model(kind="MaxPool", pools=3, kernel=2, stride=2, seed=0):
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-0.4, 0.4, (3, 3, 1, 1)).astype(np.float32)
    bias = rng.uniform(-1, 1, 3).astype(np.float32)
    nodes = [h.make_node("Conv", ["input", "w", "b"], ["c0"], kernel_shape=[1, 1])]
    previous = "c0"
    for index in range(pools):
        output = "output" if index == pools - 1 else f"c{index + 1}"
        nodes.append(h.make_node(kind, [previous], [output], kernel_shape=[kernel, kernel],
                                 strides=[stride, stride]))
        previous = output
    graph = h.make_graph(nodes, "reduction", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, [1, 3, 1, 1])],
                         [nh.from_array(weights, "w"), nh.from_array(bias, "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model

def chain_model(first="Add", extra=1, seed=31, distinct=True, inputs=2):
    rng = np.random.default_rng(seed)
    wa = rng.uniform(-0.7, 0.8, (3, 3, 1, 1)).astype(np.float32)
    ba = rng.uniform(-2, 2, (3,)).astype(np.float32)
    wb = rng.uniform(-0.7, 0.8, (3, 3, 1, 1)).astype(np.float32)
    bb = rng.uniform(-2, 2, (3,)).astype(np.float32)
    second_join = "B" if distinct else "A"
    nodes = [h.make_node("Conv", ["a", "wa", "ba"], ["A"], kernel_shape=[1, 1]),
             h.make_node("Conv", ["b", "wb", "bb"], ["B"], kernel_shape=[1, 1]),
             h.make_node(first, ["A", second_join], ["C"])]
    previous = "C"
    for index in range(extra):
        output = "out" if index == extra - 1 else f"o{index}"
        nodes.append(h.make_node("Mul", [previous, "A"], [output]))
        previous = output
    shapes = [h.make_tensor_value_info("a", 1, [1, 3, 8, 8]), h.make_tensor_value_info("b", 1, [1, 3, 8, 8])]
    if inputs == "mismatch":
        shapes = [h.make_tensor_value_info("a", 1, [1, 3, 8, 8]), h.make_tensor_value_info("b", 1, [1, 3, 7, 8])]
    elif inputs == 3:
        shapes.append(h.make_tensor_value_info("extra", 1, [1, 3, 8, 8]))
    graph = h.make_graph(nodes, "chain", shapes, [h.make_tensor_value_info("out", 1, [1, 3, 8, 8])],
                         [nh.from_array(wa, "wa"), nh.from_array(ba, "ba"),
                          nh.from_array(wb, "wb"), nh.from_array(bb, "bb")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)

def multi_model(first="Add", extra=1, seed=13, order=("a", "b"), weights_order=("wa", "wb")):
    rng = np.random.default_rng(seed)

    def branch():
        return rng.uniform(-0.7, 0.8, (3, 3, 1, 1)).astype(np.float32), rng.uniform(-2, 2, (3,)).astype(np.float32)

    wa, ba = branch()
    wb, bb = branch()
    initializers = [nh.from_array(wa, "wa"), nh.from_array(ba, "ba"),
                    nh.from_array(wb, "wb"), nh.from_array(bb, "bb")]
    bias_name = {"wa": "ba", "wb": "bb"}
    inputs = list(order) + [f"c{index}" for index in range(extra)]
    nodes = [h.make_node("Conv", [order[0], weights_order[0], bias_name[weights_order[0]]], ["A"], kernel_shape=[1, 1]),
             h.make_node("Conv", [order[1], weights_order[1], bias_name[weights_order[1]]], ["B"], kernel_shape=[1, 1]),
             h.make_node(first, ["A", "B"], ["C"])]
    previous = "C"
    for index in range(extra):
        weight, bias = branch()
        initializers += [nh.from_array(weight, f"wc{index}"), nh.from_array(bias, f"bc{index}")]
        output = "out" if index == extra - 1 else f"S{index + 1}"
        nodes.append(h.make_node("Conv", [f"c{index}", f"wc{index}", f"bc{index}"], [f"D{index}"], kernel_shape=[1, 1]))
        nodes.append(h.make_node("Mul", [previous, f"D{index}"], [output]))
        previous = output
    graph = h.make_graph(nodes, "multi", [h.make_tensor_value_info(name, 1, [1, 3, 8, 8]) for name in inputs],
                         [h.make_tensor_value_info("out", 1, [1, 3, 8, 8])], initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)

def native_model(kind="Add", batch=1, channels=3, out_channels=3, height=8, width=8, seed=0,
                 second_shape=None, node_count=2):
    rng = np.random.default_rng(seed)
    initializers = []
    nodes = []
    for index in range(2):
        weights = rng.uniform(-0.4, 0.4, (out_channels, channels, 1, 1)).astype(np.float32)
        bias = rng.uniform(-1, 1, out_channels).astype(np.float32)
        initializers += [nh.from_array(weights, f"w{index}"), nh.from_array(bias, f"b{index}")]
        nodes.append(h.make_node("Conv", [f"i{index}", f"w{index}", f"b{index}"], [f"c{index}"], kernel_shape=[1, 1]))
    nodes.append(h.make_node(kind, ["c0", "c1"], ["output"]))
    output_name = "output"
    if node_count == 1:
        nodes = nodes[:1]
        output_name = "c0"
    first_shape = [batch, channels, height, width]
    second = list(second_shape) if second_shape is not None else first_shape
    graph = h.make_graph(
        nodes, "native",
        [h.make_tensor_value_info("i0", 1, first_shape), h.make_tensor_value_info("i1", 1, second)],
        [h.make_tensor_value_info(output_name, 1, [batch, out_channels, height, width])], initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)

def raw_reduction_model(nodes, output_shape=(1, 3, 1, 1), output_type=1):
    weights = np.random.default_rng(0).uniform(-0.4, 0.4, (3, 3, 1, 1)).astype(np.float32)
    bias = np.zeros(3, np.float32)
    graph = h.make_graph(nodes, "reduction", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", output_type, list(output_shape))],
                         [nh.from_array(weights, "w"), nh.from_array(bias, "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model

def raw_dag_model(nodes, inputs, outputs, initializers):
    graph = h.make_graph(nodes, "validation", inputs, outputs, initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)

def independent_chain(branch_codes, bands, first_stage, conversions, output_band):
    real = [dequantize(codes, band) for codes, band in zip(branch_codes, bands)]
    if first_stage == "Add":
        stage = real[0] + real[1]
    elif first_stage == "Sub":
        stage = real[0] - real[1]
    elif first_stage == "Max":
        stage = np.maximum(real[0], real[1])
    elif first_stage == "Mul":
        stage = real[0] * real[1]
    else:
        raise ValueError("unknown stage: " + str(first_stage))
    for _conversion, _shift in conversions:
        stage = stage * real[0]
    return requantize(stage, output_band)

# --- reduction ----------------------------------------------------------------

class ReductionProfileTests(unittest.TestCase):
    def test_profiles_and_independent_reference(self):
        rng = np.random.default_rng(5)
        samples = rng.integers(0, 256, (24, 8, 8, 3), dtype=np.uint8)
        for kind, profile in (("MaxPool", 5), ("AveragePool", 6)):
            with self.subTest(kind=kind):
                model = reduction_model(kind)
                payload, meta = compile_legacy(model)
                self.assertEqual(meta["profile"], profile)
                self.assertEqual(meta["pool"], kind)
                self.assertEqual(meta["pool_levels"], 3)
                self.assertEqual(meta["output_shape_nhwc"], [1, 1, 1, 3])
                info = decode(encode(payload, meta))
                self.assertEqual((info["profile"], info["task_count"]), (profile, 4))
                self.assertEqual(info["output_shape_nhwc"], [1, 1, 1, 3])
                self.assertEqual(info["operators"], ["Conv", kind] * 1 + [kind, kind])
                q = quantization_of(meta)
                band = {"scale": meta["output_scale"], "zero_point": meta["output_zero_point"]}
                got = np.stack([stage_reference(reference(x, q), kind, 3) for x in samples])
                want = np.stack([requantize(float_reference(model, x), band) for x in samples])
                difference = np.abs(got.astype(np.int64) - want.astype(np.int64))
                self.assertLessEqual(int(difference.max()), 1)

    def test_rejects_wrong_pool_count_and_kernel(self):
        # Three levels with a non-2x2/stride-2 pool attribute are rejected by the
        # staged emitter itself.
        with self.assertRaisesRegex(ValueError, "unsupported staged pooling graph"):
            compile_legacy(reduction_model(kernel=3))
        with self.assertRaisesRegex(ValueError, "unsupported staged pooling graph"):
            compile_legacy(reduction_model(stride=1))
        # Two or four pool stages never reach the staged emitter.
        for pools in (2, 4):
            with self.subTest(pools=pools):
                with self.assertRaises(ValueError):
                    compile_legacy(reduction_model(pools=pools))

    def test_reduction_validation_bounds(self):
        conv = h.make_node("Conv", ["input", "w", "b"], ["c0"], kernel_shape=[1, 1])

        def pool(source, destination, kind="MaxPool", **attrs):
            return h.make_node(kind, [source], [destination], **{**dict(kernel_shape=[2, 2], strides=[2, 2]), **attrs})

        cases = {
            "too_many_nodes": [conv, pool("c0", "c1"), pool("c1", "c2"), pool("c2", "c3"),
                               pool("c3", "c4"), pool("c4", "output")],
            "mixed_kind": [conv, pool("c0", "c1"), pool("c1", "c2", kind="AveragePool"), pool("c2", "output")],
            "wrong_attrs": [conv, pool("c0", "c1"), pool("c1", "c2"),
                            pool("c2", "output", kernel_shape=[3, 3])],
            "broken_chain": [conv, pool("c0", "c1"), pool("c0", "c2"), pool("c2", "output")],
        }
        for name, nodes in cases.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(ValueError, "unsupported staged pooling graph"):
                    save_and_compile(raw_reduction_model(nodes), compile_reduction)
        with self.assertRaisesRegex(ValueError, "unsupported staged pooling graph"):
            save_and_compile(raw_reduction_model([conv, pool("c0", "c1"), pool("c1", "c2"), pool("c2", "output")],
                                                 output_shape=(1, 3, 2, 2)), compile_reduction)
        with self.assertRaisesRegex(ValueError, "staged pooling requires float32 output"):
            save_and_compile(raw_reduction_model([conv, pool("c0", "c1"), pool("c1", "c2"), pool("c2", "output")],
                                                 output_type=7), compile_reduction)

# --- elementwise chain --------------------------------------------------------

class ElementwiseChainProfileTests(unittest.TestCase):
    def test_profiles_and_independent_reference(self):
        rng = np.random.default_rng(7)
        for first in ("Add", "Sub", "Max", "Mul"):
            for extra in (1, 2):
                with self.subTest(first=first, extra=extra):
                    binary, meta = compile_sequence_model(chain_model(first, extra))
                    info = decode_sequence(binary)
                    expected_chain = "Mul(" * extra + f"{first}(a,b)" + ",a)" * extra
                    self.assertEqual(meta["first_stage"], first)
                    self.assertEqual(meta["stages"], extra + 1)
                    self.assertEqual(meta["elementwise_chain"], expected_chain)
                    self.assertEqual(info["task_count"], 3 + extra)
                    self.assertEqual(info["input_tensor_count"], 2)
                    self.assertEqual(info["output_shape_nhwc"], [1, 8, 8, 3])
                    bands = [{"scale": float(branch["output_scale"]), "zero_point": int(branch["output_zero_point"])}
                             for branch in meta["branches"]]
                    output_band = {"scale": meta["output_scale"], "zero_point": meta["output_zero_point"]}
                    a = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
                    b = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
                    got = chain_reference(a, b, meta)
                    q0, q1 = quantization_of(meta["branches"][0]), quantization_of(meta["branches"][1])
                    want = independent_chain([reference(a, q0), reference(b, q1)], bands, first,
                                             meta["stage_conversions"], output_band)
                    self.assertLessEqual(int(np.abs(got.astype(np.int64) - want.astype(np.int64)).max()), 1)

    def test_rejects_non_matching_inputs_and_joins(self):
        with self.assertRaisesRegex(ValueError, "external tensors must be float32"):
            compile_sequence_model(chain_model(inputs="mismatch"))
        with self.assertRaisesRegex(ValueError, "requires two inputs and one output"):
            compile_sequence_model(chain_model(inputs=3))
        with self.assertRaisesRegex(ValueError, "first stage must combine the two Conv branches"):
            compile_sequence_model(chain_model(distinct=False))

    def test_rejects_operand_zero_points_and_output_override(self):
        with self.assertRaisesRegex(ValueError, "Mul operand zero points require a Mul profile"):
            compile_sequence_model(chain_model(), mul_operand_zero_points=(1, 0))
        with self.assertRaisesRegex(ValueError, "output override unsupported"):
            compile_sequence_model(chain_model(), output_range={"scale": 1.0, "zero_point": 0})

# --- multi-input DAG ----------------------------------------------------------

class MultiInputDagProfileTests(unittest.TestCase):
    def test_profiles_and_independent_reference(self):
        rng = np.random.default_rng(11)
        for first in ("Add", "Sub", "Max", "Mul"):
            for extra in (1, 2, 3):
                with self.subTest(first=first, extra=extra):
                    binary, meta = compile_sequence_model(multi_model(first, extra))
                    info = decode_sequence(binary)
                    expected_chain = "Mul(" * extra + f"{first}(a,b)" + ",c)" * extra
                    self.assertEqual(info["format_version"], 5)
                    self.assertEqual(info["input_tensor_count"], 2 + extra)
                    self.assertEqual(info["task_count"], 3 + 2 * extra)
                    self.assertEqual(meta["extra_inputs"], extra)
                    self.assertEqual(meta["elementwise_chain"], expected_chain)
                    self.assertEqual(info["output_shape_nhwc"], [1, 8, 8, 3])
                    bands = [{"scale": float(branch["output_scale"]), "zero_point": int(branch["output_zero_point"])}
                             for branch in meta["branches"]]
                    third_band = {"scale": float(meta["third"]["output_scale"]),
                                  "zero_point": int(meta["third"]["output_zero_point"])}
                    values = [rng.integers(0, 256, (8, 8, 3), dtype=np.uint8) for _ in range(2 + extra)]
                    got = multi_input_reference(values[0], values[1], values[2:], meta)
                    codes = [reference(values[0], quantization_of(meta["branches"][0])),
                             reference(values[1], quantization_of(meta["branches"][1]))]
                    third = quantization_from(meta["third"])
                    real = [dequantize(codes[0], bands[0]), dequantize(codes[1], bands[1])]
                    stage = {"Add": real[0] + real[1], "Sub": real[0] - real[1],
                             "Max": np.maximum(real[0], real[1]), "Mul": real[0] * real[1]}[first]
                    for value in values[2:]:
                        stage = stage * dequantize(reference(value, third), third_band)
                    want = requantize(stage, {"scale": meta["output_scale"], "zero_point": meta["output_zero_point"]})
                    self.assertLessEqual(int(np.abs(got.astype(np.int64) - want.astype(np.int64)).max()), 1)

    def test_external_input_ordering_contract(self):
        rng = np.random.default_rng(19)
        a = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        b = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        c = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        _, meta = compile_sequence_model(multi_model("Add", 1))
        declared = multi_input_reference(a, b, [c], meta)
        # Values are consumed in graph order: swapping a and b changes the result.
        self.assertFalse(np.array_equal(declared, multi_input_reference(b, a, [c], meta)))
        # Permuting the declared order together with the matching branch weights is a
        # relabelling of the same DAG, so the reference must agree byte for byte.
        swapped_binary, swapped = compile_sequence_model(multi_model("Add", 1, order=("b", "a"),
                                                                   weights_order=("wb", "wa")))
        info = decode_sequence(swapped_binary)
        self.assertEqual([t["index"] for t in info["input_tensors"]], [0, 1, 2])
        np.testing.assert_array_equal(declared, multi_input_reference(b, a, [c], swapped))

    def test_rejects_bad_branch_counts(self):
        with self.assertRaisesRegex(ValueError, "Mul operand zero points require a Mul profile"):
            compile_sequence_model(multi_model("Add", 2), mul_operand_zero_points=(0, 1))
        # One graph input per Conv is required.
        extra_input = multi_model("Add", 1)
        extra_input.graph.input.append(h.make_tensor_value_info("extra", 1, [1, 3, 8, 8]))
        with self.assertRaisesRegex(ValueError, "one input per Conv and one output"):
            compile_sequence_model(extra_input)

# --- native elementwise -------------------------------------------------------

class NativeElementwiseProfileTests(unittest.TestCase):
    def test_profiles_and_independent_reference(self):
        rng = np.random.default_rng(23)
        functions = {"Add": add_reference, "Sub": sub_reference, "Max": max_reference, "Mul": mul_reference}
        for kind, function in functions.items():
            for batch in (1, 2):
                with self.subTest(kind=kind, batch=batch):
                    binary, meta = compile_native_elementwise(native_model(kind, batch=batch))
                    info = decode_sequence(binary)
                    self.assertEqual(meta["elementwise_profile"], kind.lower() + "-native16-input")
                    self.assertEqual(info["task_count"], 3 * batch)
                    self.assertEqual(info["input_tensor_count"], 2)
                    self.assertEqual(info["output_shape_nhwc"], [batch, 8, 8, 3])
                    self.assertEqual(meta["output_shape_nhwc"], [batch, 8, 8, 3])
                    q0, q1 = quantization_of(meta["branches"][0]), quantization_of(meta["branches"][1])
                    band = {"scale": meta["output_scale"], "zero_point": meta["output_zero_point"]}
                    a = rng.integers(0, 256, (batch, 8, 8, 3), dtype=np.uint8)
                    b = rng.integers(0, 256, (batch, 8, 8, 3), dtype=np.uint8)
                    got = np.stack([function(reference(x, q0), reference(y, q1)) for x, y in zip(a, b)])
                    band_q0 = {"scale": q0.output_scale, "zero_point": q0.output_zero_point}
                    band_q1 = {"scale": q1.output_scale, "zero_point": q1.output_zero_point}
                    real_a = dequantize(reference(a, q0), band_q0)
                    real_b = dequantize(reference(b, q1), band_q1)
                    stage = {"Add": real_a + real_b, "Sub": real_a - real_b,
                             "Max": np.maximum(real_a, real_b), "Mul": real_a * real_b}[kind]
                    want = requantize(stage, band)
                    self.assertLessEqual(int(np.abs(got.astype(np.int64) - want.astype(np.int64)).max()), 1)

    def test_rejects_mismatched_input_shapes(self):
        for second in ([1, 3, 7, 8], [2, 3, 8, 8], [1, 2, 8, 8]):
            with self.subTest(second=second):
                with self.assertRaisesRegex(ValueError, "matching static input shapes"):
                    compile_native_elementwise(native_model(second_shape=second))

    def test_rejects_bad_operand_zero_points(self):
        with self.assertRaisesRegex(ValueError, "apply only to Mul"):
            compile_native_elementwise(native_model("Add"), operand_zero_points=(1, 0))
        with self.assertRaisesRegex(ValueError, "must be two INT8 values"):
            compile_native_elementwise(native_model("Mul"), operand_zero_points=(0, 200))
        with self.assertRaisesRegex(ValueError, "must be two INT8 values"):
            compile_native_elementwise(native_model("Mul"), operand_zero_points=(0,))

    def test_rejects_wrong_node_count(self):
        with self.assertRaisesRegex(ValueError, "requires two Conv branches"):
            compile_native_elementwise(native_model(node_count=1))

    def test_rejects_invalid_connections_and_kernels(self):
        weight = np.random.default_rng(0).uniform(-0.4, 0.4, (3, 3, 1, 1)).astype(np.float32)
        weight3 = np.random.default_rng(1).uniform(-0.4, 0.4, (3, 3, 3, 3)).astype(np.float32)
        bias = np.zeros(3, np.float32)
        inputs = [h.make_tensor_value_info("i0", 1, [1, 3, 8, 8]),
                  h.make_tensor_value_info("i1", 1, [1, 3, 8, 8])]
        output = [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])]
        initializers = [nh.from_array(weight, "w0"), nh.from_array(bias, "b0"),
                        nh.from_array(weight, "w1"), nh.from_array(bias, "b1"),
                        nh.from_array(weight3, "w3")]
        # The join reads one branch twice instead of both branches.
        bad_join = raw_dag_model(
            [h.make_node("Conv", ["i0", "w0", "b0"], ["c0"], kernel_shape=[1, 1]),
             h.make_node("Conv", ["i1", "w1", "b1"], ["c1"], kernel_shape=[1, 1]),
             h.make_node("Add", ["c0", "c0"], ["output"])], inputs, output, initializers)
        with self.assertRaisesRegex(ValueError, "invalid native elementwise graph connections"):
            compile_native_elementwise(bad_join)
        # The native elementwise branches are restricted to 1x1 Conv.
        three_by_three = raw_dag_model(
            [h.make_node("Conv", ["i0", "w3", "b0"], ["c0"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
             h.make_node("Conv", ["i1", "w3", "b1"], ["c1"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
             h.make_node("Add", ["c0", "c1"], ["output"])], inputs, output, initializers)
        with self.assertRaisesRegex(ValueError, "same-shape 1x1 Conv"):
            compile_native_elementwise(three_by_three)

class DagDirectValidationTests(unittest.TestCase):
    """Direct emitter calls for bound shapes the scheduler would route elsewhere."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.weight = rng.uniform(-0.7, 0.8, (3, 3, 1, 1)).astype(np.float32)
        self.weight3 = rng.uniform(-0.7, 0.8, (3, 3, 3, 3)).astype(np.float32)
        self.bias = np.zeros(3, np.float32)
        self.two_inputs = [h.make_tensor_value_info("a", 1, [1, 3, 8, 8]),
                           h.make_tensor_value_info("b", 1, [1, 3, 8, 8])]
        self.three_inputs = self.two_inputs + [h.make_tensor_value_info("c", 1, [1, 3, 8, 8])]
        self.output = [h.make_tensor_value_info("out", 1, [1, 3, 8, 8])]
        self.initializers = [nh.from_array(self.weight, "wa"), nh.from_array(self.bias, "ba"),
                             nh.from_array(self.weight, "wb"), nh.from_array(self.bias, "bb"),
                             nh.from_array(self.weight3, "w3")]

    def test_chain_validation_bounds(self):
        conv_a = h.make_node("Conv", ["a", "wa", "ba"], ["A"], kernel_shape=[1, 1])
        conv_b = h.make_node("Conv", ["b", "wb", "bb"], ["B"], kernel_shape=[1, 1])
        cases = {
            "bad_first_stage": ([conv_a, conv_b, h.make_node("Relu", ["A"], ["C"]),
                                 h.make_node("Mul", ["C", "A"], ["out"])],
                                "requires Conv, Conv"),
            "later_stage_not_first_branch": ([conv_a, conv_b, h.make_node("Add", ["A", "B"], ["C"]),
                                              h.make_node("Mul", ["C", "B"], ["out"])],
                                             "must multiply the previous stage by the first branch"),
            "non_1x1_branch": ([h.make_node("Conv", ["a", "w3", "ba"], ["A"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
                                conv_b, h.make_node("Add", ["A", "B"], ["C"]),
                                h.make_node("Mul", ["C", "A"], ["out"])],
                               "branches must be 1x1 Conv"),
            "inputs_out_of_order": ([h.make_node("Conv", ["b", "wa", "ba"], ["A"], kernel_shape=[1, 1]),
                                     h.make_node("Conv", ["a", "wb", "bb"], ["B"], kernel_shape=[1, 1]),
                                     h.make_node("Add", ["A", "B"], ["C"]),
                                     h.make_node("Mul", ["C", "A"], ["out"])],
                                    "consume the two graph inputs in order"),
        }
        for name, (nodes, message) in cases.items():
            with self.subTest(case=name):
                model = raw_dag_model(nodes, self.two_inputs, self.output, self.initializers)
                with self.assertRaisesRegex(ValueError, message):
                    compile_elementwise_dag(model)

    def test_multi_input_validation_bounds(self):
        conv_a = h.make_node("Conv", ["a", "wa", "ba"], ["A"], kernel_shape=[1, 1])
        conv_b = h.make_node("Conv", ["b", "wb", "bb"], ["B"], kernel_shape=[1, 1])
        cases = {
            "bad_pattern": ([conv_a, conv_b, h.make_node("Add", ["A", "B"], ["C"]),
                             h.make_node("Add", ["C", "A"], ["out"])],
                            self.two_inputs, "requires Conv, Conv"),
            "inputs_out_of_order": ([h.make_node("Conv", ["b", "wa", "ba"], ["A"], kernel_shape=[1, 1]),
                                     h.make_node("Conv", ["a", "wb", "bb"], ["B"], kernel_shape=[1, 1]),
                                     h.make_node("Add", ["A", "B"], ["C"]),
                                     h.make_node("Conv", ["c", "wa", "ba"], ["D"], kernel_shape=[1, 1]),
                                     h.make_node("Mul", ["C", "D"], ["out"])],
                                    self.three_inputs, "consume the graph inputs in order"),
            "non_1x1_branch": ([h.make_node("Conv", ["a", "w3", "ba"], ["A"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
                                conv_b, h.make_node("Add", ["A", "B"], ["C"]),
                                h.make_node("Conv", ["c", "wa", "ba"], ["D"], kernel_shape=[1, 1]),
                                h.make_node("Mul", ["C", "D"], ["out"])],
                               self.three_inputs, "branches must be 1x1 Conv"),
        }
        for name, (nodes, inputs, message) in cases.items():
            with self.subTest(case=name):
                model = raw_dag_model(nodes, inputs, self.output, self.initializers)
                with self.assertRaisesRegex(ValueError, message):
                    compile_multi_input_dag(model)

if __name__ == "__main__":
    unittest.main()
