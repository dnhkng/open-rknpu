"""N-layer native Conv chain: container shape, layer count and rejections."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence
from open_rknpu.chain_n import chain_n_reference

ROOT = Path(__file__).resolve().parents[1]


def chain_model(layers, seed=5):
    rng = np.random.default_rng(seed)
    nodes = []; tensors = []; previous = "input"
    for index, (channels, kernel) in enumerate(layers):
        inputs = 3 if index == 0 else layers[index - 1][0]
        w = rng.uniform(-.7, .8, (channels, inputs, kernel, kernel)).astype(np.float32)
        b = rng.uniform(-2, 2, (channels,)).astype(np.float32)
        tensors += [nh.from_array(w, f"w{index}"), nh.from_array(b, f"b{index}")]
        nodes.append(h.make_node("Conv", [previous, f"w{index}", f"b{index}"], [f"c{index}"],
                                 kernel_shape=[kernel, kernel], pads=[kernel // 2] * 4))
        previous = f"c{index}"
        if index < len(layers) - 1:
            nodes.append(h.make_node("Relu", [previous], [f"r{index}"]))
            previous = f"r{index}"
    graph = h.make_graph(nodes, "native_chain",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info(previous, 1, [1, layers[-1][0], 8, 8])], tensors)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def compile_model(model):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "model.onnx"
        onnx.save(model, path)
        return compile_sequence(path)


class NativeChainTests(unittest.TestCase):
    def test_three_and_four_layer_chains(self):
        configs = [[(5, 1), (5, 3), (3, 1)], [(16, 3), (12, 3), (3, 1)],
                   [(4, 3), (8, 3), (6, 1), (3, 3)], [(6, 1), (6, 1), (6, 1), (3, 1)]]
        for layers in configs:
            with self.subTest(layers=len(layers), hidden=[c for c, _ in layers]):
                binary, meta = compile_model(chain_model(layers))
                info = decode_sequence(binary)
                self.assertEqual(info["task_count"], len(layers))
                self.assertEqual(info["input_layout"], "packed")  # external input is packed NHWC; layers are native16
                self.assertEqual(info["output_shape_nhwc"], [1, 8, 8, 3])
                self.assertEqual(meta["profile"], "native-chain")
                self.assertEqual(meta["layers"], len(layers))
                self.assertEqual(meta["hidden_channels"], [c for c, _ in layers])
                self.assertEqual(meta["kernels"], [k for _, k in layers])

    def test_reference_composition_runs(self):
        layers = [(5, 1), (5, 3), (3, 1)]
        binary, meta = compile_model(chain_model(layers))
        # The reference is composed from the same per-layer quantizations the
        # emitter used; this checks the plumbing and output contract.
        for value in (0, 128, 255):
            x = np.full((8, 8, 3), value, np.uint8)
            output = chain_n_reference(x, meta["quantizations"])
            self.assertEqual(output.shape, (8, 8, 3))
            self.assertEqual(output.dtype, np.int8)

    def test_multi_output_v5_container(self):
        from open_rknpu.chain_n import compile_chain_n
        for layers in ([(5, 1), (5, 3), (3, 1)], [(4, 3), (8, 3), (6, 1), (3, 3)]):
            with self.subTest(layers=len(layers)):
                binary, meta = compile_chain_n(chain_model(layers), expose_intermediates=True)
                info = decode_sequence(binary)
                self.assertEqual(info["format_version"], 5)
                self.assertEqual(info["output_tensor_count"], len(layers))
                self.assertEqual([t["name"] for t in info["output_tensors"]],
                                 [f"layer{i + 1}" for i in range(len(layers) - 1)] + ["output"])
                self.assertEqual(meta["exposed_intermediates"], True)
                self.assertEqual([t["shape_nhwc"][3] for t in info["output_tensors"]] if "shape_nhwc" in info["output_tensors"][0]
                                 else [t["channels"] for t in info["output_tensors"]],
                                 [c for c, _ in layers])

    def test_intermediate_reuse_shrinks_arena(self):
        from open_rknpu.chain_n import compile_chain_n
        model = chain_model([(4, 3), (8, 3), (6, 1), (3, 3)])
        distinct, _ = compile_chain_n(model)
        reused, meta = compile_chain_n(model, reuse_intermediates=True)
        self.assertTrue(meta["reused_intermediates"])
        self.assertLess(decode_sequence(reused)["arena_bytes"], decode_sequence(distinct)["arena_bytes"])
        # A three-layer chain has only two intermediates, so nothing to reuse.
        _, meta3 = compile_chain_n(chain_model([(5, 1), (5, 3), (3, 1)]), reuse_intermediates=True)
        self.assertFalse(meta3["reused_intermediates"])

    def test_rejections(self):
        # Even node count (Conv, Conv) is not an alternating chain.
        rng = np.random.default_rng(7)
        tensors = [nh.from_array(rng.uniform(-.7, .8, (4, 3, 1, 1)).astype(np.float32), "w0"),
                   nh.from_array(np.zeros(4, np.float32), "b0"),
                   nh.from_array(rng.uniform(-.7, .8, (3, 4, 1, 1)).astype(np.float32), "w1"),
                   nh.from_array(np.zeros(3, np.float32), "b1")]
        nodes = [h.make_node("Conv", ["input", "w0", "b0"], ["c0"], kernel_shape=[1, 1]),
                 h.make_node("Conv", ["c0", "w1", "b1"], ["out"], kernel_shape=[1, 1])]
        graph = h.make_graph(nodes, "bad", [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                             [h.make_tensor_value_info("out", 1, [1, 3, 8, 8])], tensors)
        broken = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
        broken.ir_version = 8
        with self.assertRaises(ValueError):
            compile_model(broken)
        # Hidden channels outside 3..16.
        with self.assertRaises(ValueError):
            compile_model(chain_model([(2, 1), (2, 1), (3, 1)]))


if __name__ == "__main__":
    unittest.main()
