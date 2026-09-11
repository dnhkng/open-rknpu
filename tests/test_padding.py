"""Input padding helper and leading-Pad folding."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.padding import pad_input
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence

ROOT = Path(__file__).resolve().parents[1]


def padded_conv(mode="reflect", pads=(1, 1, 1, 1)):
    w = np.arange(81, dtype=np.float32).reshape(3, 3, 3, 3) / 81
    b = np.zeros(3, np.float32)
    height = width = 8
    nodes = [h.make_node("Pad", ["input", "pads", "value"], ["padded"], mode=mode),
             h.make_node("Conv", ["padded", "w", "b"], ["output"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])]
    graph = h.make_graph(nodes, "padding",
        [h.make_tensor_value_info("input", 1, [1, 3, height, width])],
        [h.make_tensor_value_info("output", 1, [1, 3, height + pads[0] + pads[2], width + pads[1] + pads[3]])],
        [nh.from_array(w, "w"), nh.from_array(b, "b"),
         nh.from_array(np.array([0, 0, pads[0], pads[1], 0, 0, pads[2], pads[3]], np.int64), "pads"),
         nh.from_array(np.array(0, np.float32), "value")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)])
    model.ir_version = 8
    return model


class PaddingTests(unittest.TestCase):
    def test_pad_input_matches_numpy(self):
        data = np.arange(8 * 6 * 3, dtype=np.uint8).reshape(8, 6, 3)
        for mode, np_mode in (("constant", "constant"), ("reflect", "reflect"), ("edge", "edge"), ("wrap", "wrap")):
            with self.subTest(mode=mode):
                kwargs = {"constant_values": 7} if mode == "constant" else {}
                expected = np.pad(data, ((1, 2), (2, 1), (0, 0)), mode=np_mode, **kwargs)
                actual = pad_input(data, (1, 2, 2, 1), mode, constant_values=7)
                np.testing.assert_array_equal(actual, expected)
        batch = data[None]
        expected = np.pad(batch, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
        np.testing.assert_array_equal(pad_input(batch, (1, 1, 1, 1), "edge"), expected)

    def test_rejects_bad_mode_and_negative_pads(self):
        data = np.zeros((8, 8, 3), np.uint8)
        with self.assertRaises(ValueError):
            pad_input(data, (1, 1, 1, 1), "unknown")
        with self.assertRaises(ValueError):
            pad_input(data, (-1, 0, 0, 0), "edge")

    def test_leading_pad_is_folded_and_recorded(self):
        for mode in ("constant", "reflect", "edge", "wrap"):
            with self.subTest(mode=mode):
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "model.onnx"
                    onnx.save(padded_conv(mode=mode, pads=(1, 1, 1, 1)), path)
                    binary, meta = compile_sequence(path)
                info = decode_sequence(binary)
                padding = meta["input_padding"]
                self.assertEqual(padding["mode"], mode)
                self.assertEqual(padding["pads"], [1, 1, 1, 1])
                self.assertEqual(padding["padded_shape"], [1, 3, 10, 10])
                self.assertTrue(padding["requires_host_preprocessing"])
                self.assertEqual(info["shape_nhwc"], [1, 10, 10, 3])
                self.assertEqual(info["output_shape_nhwc"], [1, 10, 10, 3])

    def test_negative_pad_rejected(self):
        model = padded_conv(mode="edge", pads=(1, 1, 1, 1))
        pads = np.array([0, 0, 1, 1, 0, 0, -1, 1], np.int64)
        for tensor in model.graph.initializer:
            if tensor.name == "pads":
                tensor.CopyFrom(nh.from_array(pads, "pads"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.onnx"
            onnx.save(model, path)
            with self.assertRaises(ValueError):
                compile_sequence(path)


if __name__ == "__main__":
    unittest.main()
