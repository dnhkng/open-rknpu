"""MIT. Depthwise K3 dilation2 ConvTranspose through a direct sparse K5 emission."""
from pathlib import Path
import json
import unittest
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from open_rknpu.scheduler import compile_sequence

SUITE = Path(__file__).resolve().parents[1] / "research" / "transpose_k5_dilation_suite"


def build(channels=3, strides=(1, 1), pads=(0, 0, 0, 0), output_padding=(0, 0), auto="NOTSET",
          kernel=3, dilation=2, dense=False):
    """A 1x1 stem plus a dilated ConvTranspose."""
    rng = np.random.default_rng(channels + strides[0] * 3 + strides[1])
    oc = channels + 1 if dense else 1
    weight_shape = (channels, oc, kernel, kernel)
    constants = [nh.from_array(rng.uniform(-.4, .4, (channels, 3, 1, 1)).astype(np.float32), "w1"),
                 nh.from_array(rng.uniform(-.5, .5, (channels,)).astype(np.float32), "b1"),
                 nh.from_array(rng.uniform(-.4, .4, weight_shape).astype(np.float32), "w2"),
                 nh.from_array(rng.uniform(-.5, .5, (channels * oc,)).astype(np.float32), "b2")]
    effective = (kernel - 1) * dilation + 1
    attributes = dict(kernel_shape=[kernel, kernel], strides=list(strides), pads=list(pads),
                      dilations=[dilation, dilation], group=1 if dense else channels,
                      output_padding=list(output_padding), auto_pad=auto)
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["stem"], kernel_shape=[1, 1]),
             h.make_node("ConvTranspose", ["stem", "w2", "b2"], ["output"], **attributes)]
    sy, sx = strides
    resolved = list(pads)
    if auto == "SAME_UPPER":
        totals = [effective + output_padding[i] - strides[i] for i in range(2)]
        begins = [v // 2 for v in totals]
        resolved = [begins[0], begins[1], totals[0] - begins[0], totals[1] - begins[1]]
    out_channels = channels * oc
    oh = 7 * sy + effective - resolved[0] - resolved[2] + output_padding[0]
    ow = 7 * sx + effective - resolved[1] - resolved[3] + output_padding[1]
    graph = h.make_graph(nodes, "transpose_dilation",
        [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
        [h.make_tensor_value_info("output", 1, [1, out_channels, oh, ow])], constants)
    graph.value_info.append(h.make_tensor_value_info("stem", 1, [1, channels, 8, 8]))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def dilated_reference(sample, weights, strides, pads, output_padding, dilation):
    """Float-domain depthwise transposed convolution with explicit dilation."""
    channels, _, kh, kw = weights.shape
    height, width = 8, 8
    oh = (height - 1) * strides[0] - pads[0] - pads[2] + (kh - 1) * dilation + 1 + output_padding[0]
    ow = (width - 1) * strides[1] - pads[1] - pads[3] + (kw - 1) * dilation + 1 + output_padding[1]
    out = np.zeros((oh, ow, channels), np.float64)
    for iy in range(height):
        for ix in range(width):
            for ky in range(kh):
                for kx in range(kw):
                    oy = iy * strides[0] + ky * dilation - pads[0]
                    ox = ix * strides[1] + kx * dilation - pads[1]
                    if 0 <= oy < oh and 0 <= ox < ow:
                        out[oy, ox] += sample[iy, ix] * weights[:, 0, ky, kx]
    return out


class TransposeK5DilationTests(unittest.TestCase):
    def test_recompiles_byte_identically(self):
        manifest = json.loads((SUITE / "manifest.json").read_text())
        self.assertEqual(len(manifest), 8)
        for entry in manifest:
            path = SUITE / f"model{entry['index']:03}.onnx"
            with self.subTest(model=entry["index"]):
                data, meta = compile_sequence(path)
                self.assertEqual(data, path.with_suffix(".bin").read_bytes())
                self.assertEqual(meta["transposed_rewrite"], "K3 dilation2 expanded to sparse K5")
                self.assertEqual(meta["output_shape_nhwc"][1:], entry["output_shape"])

    def test_board_suite_is_recorded(self):
        results = json.loads((SUITE / "board_results_0.json").read_text())
        self.assertEqual(len(results), 8)
        self.assertTrue(all(entry["passed"] for entry in results))
        self.assertEqual(sum(entry["inferences"] for entry in results), 128)
        self.assertEqual(sum(entry["bytes"] for entry in results), 82432)

    def test_expansion_is_the_same_operator(self):
        """A zero-filled K5 with dilation1 equals the dilated K3 in the float domain."""
        rng = np.random.default_rng(7)
        for strides, pads, output_padding in ((( 1, 1), (0, 0, 0, 0), (0, 0)),
                                              ((1, 1), (1, 1, 1, 1), (0, 0)),
                                              ((2, 2), (1, 1, 1, 1), (1, 0))):
            kernel = rng.uniform(-.4, .4, (3, 1, 3, 3))
            sample = rng.uniform(0, 1, (8, 8, 3))
            dilated = dilated_reference(sample, kernel, strides, pads, output_padding, 2)
            expanded = np.zeros((3, 1, 5, 5))
            expanded[:, :, ::2, ::2] = kernel
            dense = dilated_reference(sample, expanded, strides, pads, output_padding, 1)
            with self.subTest(strides=strides, pads=pads):
                np.testing.assert_allclose(dilated, dense, atol=1e-9, rtol=0)

    def test_rejections(self):
        with self.assertRaises(ValueError):
            compile_sequence(build(dense=True))
        with self.assertRaises(ValueError):
            compile_sequence(build(dilation=3))
        with self.assertRaises(ValueError):
            compile_sequence(build(kernel=2, dilation=3))
        with self.assertRaises(ValueError):
            compile_sequence(build(kernel=7))
        with self.assertRaises(ValueError):
            compile_sequence(build(strides=(3, 3)))
        with self.assertRaises(ValueError):
            compile_sequence(build(), calibration_ranges={})


if __name__ == "__main__":
    unittest.main()
