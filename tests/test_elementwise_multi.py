"""Multi-input elementwise DAG (format v5): Mul(...Mul(Op(a,b),c_k)...,c)."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence
from open_rknpu.elementwise_multi import multi_input_reference

ROOT = Path(__file__).resolve().parents[1]


def multi_model(first="Add", extra=1, seed=13):
    rng = np.random.default_rng(seed)
    def branch():
        return rng.uniform(-.7, .8, (3, 3, 1, 1)).astype(np.float32), rng.uniform(-2, 2, (3,)).astype(np.float32)
    wa, ba = branch(); wb, bb = branch()
    initializers = [nh.from_array(wa, "wa"), nh.from_array(ba, "ba"),
                    nh.from_array(wb, "wb"), nh.from_array(bb, "bb")]
    inputs = ["a", "b"] + [f"c{k}" for k in range(extra)]
    nodes = [h.make_node("Conv", ["a", "wa", "ba"], ["A"], kernel_shape=[1, 1]),
             h.make_node("Conv", ["b", "wb", "bb"], ["B"], kernel_shape=[1, 1]),
             h.make_node(first, ["A", "B"], ["C"])]
    previous = "C"
    for k in range(extra):
        w, b = branch()
        initializers += [nh.from_array(w, f"wc{k}"), nh.from_array(b, f"bc{k}")]
        out = f"S{k + 1}" if k < extra - 1 else "out"
        nodes.append(h.make_node("Conv", [f"c{k}", f"wc{k}", f"bc{k}"], [f"D{k}"], kernel_shape=[1, 1]))
        nodes.append(h.make_node("Mul", [previous, f"D{k}"], [out]))
        previous = out
    graph = h.make_graph(nodes, "multi_input",
        [h.make_tensor_value_info(v, 1, [1, 3, 8, 8]) for v in inputs],
        [h.make_tensor_value_info("out", 1, [1, 3, 8, 8])], initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def compile_model(model):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "model.onnx"
        onnx.save(model, path)
        return compile_sequence(path)


class ElementwiseMultiTests(unittest.TestCase):
    def test_three_to_five_inputs(self):
        for extra in (1, 2, 3):
            with self.subTest(inputs=2 + extra):
                binary, meta = compile_model(multi_model(extra=extra))
                info = decode_sequence(binary)
                self.assertEqual(info["format_version"], 5)
                self.assertEqual(info["input_tensor_count"], 2 + extra)
                self.assertEqual(info["task_count"], 3 + 2 * extra)
                self.assertEqual(meta["extra_inputs"], extra)
                rng = np.random.default_rng(5)
                values = [rng.integers(0, 256, (8, 8, 3), dtype=np.uint8) for _ in range(2 + extra)]
                output = multi_input_reference(values[0], values[1], values[2:], meta)
                self.assertEqual(output.shape, (8, 8, 3))

    def test_first_stages(self):
        for first in ("Add", "Mul", "Max"):
            with self.subTest(first=first):
                binary, meta = compile_model(multi_model(first=first, extra=2))
                self.assertEqual(meta["elementwise_chain"], f"Mul(Mul({first}(a,b),c),c)")

    def test_arena_grows_with_inputs(self):
        sizes = [decode_sequence(compile_model(multi_model(extra=extra))[0])["arena_bytes"]
                 for extra in (1, 2, 3)]
        self.assertEqual(sizes, sorted(sizes))
        self.assertLess(sizes[0], sizes[-1])


if __name__ == "__main__":
    unittest.main()
