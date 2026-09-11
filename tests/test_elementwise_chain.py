"""Two-stage elementwise DAG: Mul(Add(a,b), a) with fan-out."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence
from open_rknpu.elementwise_chain import chain_reference

ROOT = Path(__file__).resolve().parents[1]


def dag_model(seed=31, swap=False, first="Add"):
    rng = np.random.default_rng(seed)
    wa = rng.uniform(-.7, .8, (3, 3, 1, 1)).astype(np.float32); ba = rng.uniform(-2, 2, (3,)).astype(np.float32)
    wb = rng.uniform(-.7, .8, (3, 3, 1, 1)).astype(np.float32); bb = rng.uniform(-2, 2, (3,)).astype(np.float32)
    second = "B" if swap else "A"
    nodes = [h.make_node("Conv", ["a", "wa", "ba"], ["A"], kernel_shape=[1, 1]),
             h.make_node("Conv", ["b", "wb", "bb"], ["B"], kernel_shape=[1, 1]),
             h.make_node(first, ["A", "B"], ["C"]),
             h.make_node("Mul", ["C", second], ["out"])]
    graph = h.make_graph(nodes, "elementwise_chain",
        [h.make_tensor_value_info("a", 1, [1, 3, 8, 8]), h.make_tensor_value_info("b", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("out", 1, [1, 3, 8, 8])],
        [nh.from_array(wa, "wa"), nh.from_array(ba, "ba"), nh.from_array(wb, "wb"), nh.from_array(bb, "bb")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def deep_model(first="Mul", extra=3, seed=17):
    rng = np.random.default_rng(seed)
    wa = rng.uniform(-.7, .8, (3, 3, 1, 1)).astype(np.float32); ba = rng.uniform(-2, 2, (3,)).astype(np.float32)
    wb = rng.uniform(-.7, .8, (3, 3, 1, 1)).astype(np.float32); bb = rng.uniform(-2, 2, (3,)).astype(np.float32)
    nodes = [h.make_node("Conv", ["a", "wa", "ba"], ["A"], kernel_shape=[1, 1]),
             h.make_node("Conv", ["b", "wb", "bb"], ["B"], kernel_shape=[1, 1]),
             h.make_node(first, ["A", "B"], ["C"])]
    previous = "C"
    for stage in range(extra):
        out = f"out{stage}" if stage < extra - 1 else "out"
        nodes.append(h.make_node("Mul", [previous, "A"], [out]))
        previous = out
    graph = h.make_graph(nodes, "elementwise_deep",
        [h.make_tensor_value_info("a", 1, [1, 3, 8, 8]), h.make_tensor_value_info("b", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("out", 1, [1, 3, 8, 8])],
        [nh.from_array(wa, "wa"), nh.from_array(ba, "ba"), nh.from_array(wb, "wb"), nh.from_array(bb, "bb")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def compile_model(model):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "model.onnx"
        onnx.save(model, path)
        return compile_sequence(path)


class ElementwiseChainTests(unittest.TestCase):
    def test_container_shape_and_tasks(self):
        binary, meta = compile_model(dag_model())
        info = decode_sequence(binary)
        self.assertEqual(info["task_count"], 4)
        self.assertEqual([t["register_count"] for t in info["tasks"]], [126, 126, 78, 78])
        self.assertEqual(info["input_tensor_count"], 2)
        self.assertEqual(info["shape_nhwc"], [1, 16, 8, 3])
        self.assertEqual(info["output_shape_nhwc"], [1, 8, 8, 3])
        self.assertEqual(meta["elementwise_chain"], "Mul(Add(a,b),a)")

    def test_reference_composition(self):
        binary, meta = compile_model(dag_model())
        rng = np.random.default_rng(7)
        a = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        b = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        output = chain_reference(a, b, meta)
        self.assertEqual(output.shape, (8, 8, 3))
        self.assertEqual(output.dtype, np.int8)

    def test_all_first_stages(self):
        for first in ("Add", "Mul", "Sub", "Max"):
            with self.subTest(first=first):
                binary, meta = compile_model(dag_model(first=first))
                self.assertEqual(decode_sequence(binary)["task_count"], 4)
                self.assertEqual(meta["elementwise_chain"], f"Mul({first}(a,b),a)")
                rng = np.random.default_rng(3)
                output = chain_reference(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8),
                                         rng.integers(0, 256, (8, 8, 3), dtype=np.uint8), meta)
                self.assertEqual(output.shape, (8, 8, 3))

    def test_deep_chains(self):
        for first in ("Add", "Mul"):
            for extra in (1, 2, 3):
                with self.subTest(first=first, extra=extra):
                    binary, meta = compile_model(deep_model(first=first, extra=extra))
                    info = decode_sequence(binary)
                    self.assertEqual(info["task_count"], 3 + extra)
                    self.assertEqual(meta["stages"], extra + 1)
                    self.assertEqual(meta["elementwise_chain"],
                                     f"{first}(a,b)" if extra == 0 else
                                     "Mul(" * extra + f"{first}(a,b)" + ",a)" * extra)
                    self.assertEqual(len(meta["stage_conversions"]), extra)
                    rng = np.random.default_rng(11)
                    output = chain_reference(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8),
                                             rng.integers(0, 256, (8, 8, 3), dtype=np.uint8), meta)
                    self.assertEqual(output.shape, (8, 8, 3))

    def test_second_operand_must_be_first_branch(self):
        with self.assertRaises(ValueError):
            compile_model(dag_model(swap=True))

    def test_output_override_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.onnx"
            onnx.save(dag_model(), path)
            with self.assertRaises(ValueError):
                compile_sequence(path, output_range={"scale": 1.0, "zero_point": 0})


if __name__ == "__main__":
    unittest.main()
