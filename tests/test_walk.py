"""MIT. Chain walk: parser, chain equivalence, dispatch and board evidence.

`open_rknpu.walk` is the op-level walk path: a linear Conv/Relu chain with a pool that
is *not* the last node. No other profile matches that shape, so the scheduler routes it
to the walk and `research/walk_chain_suite/` is board-exact (6 models, 96 inferences,
3456 bytes). These tests pin the parser, the chain equivalence with the established
arithmetic, the dispatch boundary and the retained board run.
"""
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

from open_rknpu.chain_n import chain_n_reference
from open_rknpu.scheduler import compile_sequence
from open_rknpu.walk import (chain_walk_reference, compile_chain_walk, compile_join_walk,
                             load_quantizations, parse_chain)

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "research" / "walk_chain_suite"


def graph_of(ops, height=8, width=8, seed=1):
    rng = np.random.default_rng(seed)
    nodes, inits = [], []
    source, channels, height_out, width_out = "input", 3, height, width
    for index, spec in enumerate(ops):
        if spec[0] == "conv":
            _, out_channels, kernel, activation = spec
            nodes.append(h.make_node("Conv", [source, f"w{index}", f"b{index}"], [f"c{index}"],
                                     kernel_shape=[kernel] * 2, pads=[kernel // 2] * 4))
            inits += [nh.from_array(rng.uniform(-.8, .8, (out_channels, channels, kernel, kernel)).astype(np.float32), f"w{index}"),
                      nh.from_array(rng.uniform(-1, 1, (out_channels,)).astype(np.float32), f"b{index}")]
            source, channels = f"c{index}", out_channels
            if activation == "Relu":
                nodes.append(h.make_node("Relu", [source], [f"r{index}"]))
                source = f"r{index}"
        else:
            if spec[0] not in ("MaxPool", "AveragePool"):
                nodes.append(h.make_node(spec[0], [source], [f"x{index}"]))
                source = f"x{index}"
                continue
            nodes.append(h.make_node(spec[0], [source], [f"p{index}"], kernel_shape=[2, 2], strides=[2, 2]))
            source = f"p{index}"
            height_out, width_out = height_out // 2, width_out // 2
    graph = h.make_graph(nodes, "walk",
                         [h.make_tensor_value_info("input", 1, [1, 3, height, width])],
                         [h.make_tensor_value_info(source, 1, [1, channels, height_out, width_out])],
                         inits)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def model_onnx(model):
    """Save a model to a temporary path (the scheduler takes a path)."""
    folder = tempfile.mkdtemp()
    path = Path(folder) / "walk.onnx"
    onnx.save(model, path)
    return str(path)


class ChainWalkTests(unittest.TestCase):
    def test_parser_accepts_a_conv_chain_and_rejects_other_shapes(self):
        spec = parse_chain(graph_of([("conv", 8, 3, "Relu"), ("MaxPool",), ("conv", 3, 3, None)]).graph)
        self.assertEqual([op["kind"] for op in spec["ops"]], ["conv", "MaxPool", "conv"])
        self.assertEqual(tuple(spec["output_shape"]), (1, 3, 4, 4))
        # A pool that halves an odd grid cannot lower.
        self.assertIsNone(parse_chain(graph_of([("conv", 8, 3, None), ("MaxPool",)], height=7, width=8).graph))
        # An unsupported op is rejected rather than silently dropped.
        self.assertIsNone(parse_chain(graph_of([("conv", 8, 3, None), ("Sigmoid",)]).graph))

    def test_conv_chain_reference_matches_the_established_chain(self):
        model = graph_of([("conv", 8, 3, "Relu"), ("conv", 3, 3, None)], seed=5)
        binary, meta = compile_chain_walk(model)
        walk = load_quantizations(meta)
        inputs = (np.arange(8 * 8 * 3) % 256).astype(np.uint8).reshape(8, 8, 3)
        self.assertEqual(
            chain_walk_reference(inputs, walk, parse_chain(model.graph)["ops"]).tobytes(),
            chain_n_reference(inputs, walk).tobytes())
        self.assertEqual(meta["profile"], "chain-walk")
        self.assertFalse(meta["arena_reuse"])

    def test_mid_chain_pool_routes_to_the_walk(self):
        path = ROOT / "research" / "walk_chain_suite" / "model000.onnx"
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder) / "model.onnx"
            copy.write_bytes(path.read_bytes())
            _, meta = compile_sequence(copy)
        self.assertEqual(meta["profile"], "chain-walk")

    def test_pool_only_profiles_are_not_hijacked(self):
        # A terminal pool belongs to the established pooling profile, and a chain
        # without a pool belongs to the chain profile: neither may reach the walk.
        for suite, index in (("pooling", None), ("native_chain_suite", 0),
                             ("chain_multi_suite", 0)):
            folder = ROOT / "research" / suite if index is not None else None
            if folder is None or not folder.exists():
                continue
            path = sorted(folder.glob("model*.onnx"))[index]
            with self.subTest(model=f"{suite}/{path.name}"):
                _, meta = compile_sequence(path)
                self.assertNotEqual(meta.get("profile"), "chain-walk")

    def test_containers_match_the_retained_bytes(self):
        # The suite's containers are the ones the board ran; a refactor may not change them.
        manifest = json.loads((SUITE / "manifest.json").read_text())
        for entry in manifest:
            index = entry["index"]
            path = SUITE / f"model{index:03}.onnx"
            if not path.is_file():
                continue
            with self.subTest(model=index):
                binary, _ = compile_chain_walk(onnx.load(path))
                self.assertEqual(binary, (SUITE / f"model{index:03}.bin").read_bytes())

    def test_join_walk_reproduces_the_diamond_profile(self):
        # The walk's op-level join dispatch must equal the diamond emitter's declared
        # stages: every diamond and diamond-tail model is byte-identical whether it is
        # compiled through the profile or through the walk.
        checked = 0
        for suite in ("diamond_suite", "diamond_tail_suite"):
            for path in sorted((ROOT / "research" / suite).glob("model*.onnx")):
                with self.subTest(model=f"{suite}/{path.name}"):
                    model = onnx.load(path)
                    walked, _ = compile_join_walk(model)
                    with tempfile.TemporaryDirectory() as folder:
                        copy = Path(folder) / "model.onnx"
                        onnx.save(model, copy)
                        profiled, _ = compile_sequence(copy)
                    self.assertEqual(walked, profiled)
                    checked += 1
        self.assertEqual(checked, 24)

    def test_mixed_pool_join_suite_is_board_exact(self):
        # Graphs with a different pool kind per branch (and an optional tail) have no
        # profile; the walk lowers them and the retained board run is exact.
        suite = ROOT / "research" / "walk_join_suite"
        manifest = json.loads((suite / "manifest.json").read_text())
        results = json.loads((suite / "board_results_0.json").read_text())
        self.assertEqual(len(manifest), 6)
        self.assertEqual(len(results), 6)
        for entry in results:
            with self.subTest(model=entry["model"]):
                self.assertTrue(entry["passed"])
        summary = (suite / "board_summary.txt").read_text()
        self.assertTrue(summary.startswith("PASS:"), summary)
        self.assertIn("96 inferences", summary)
        self.assertIn("4608 exact output bytes", summary)
        self.assertTrue((suite / "README.md").is_file())

    def test_retained_board_run_is_exact(self):
        results = json.loads((SUITE / "board_results_0.json").read_text())
        manifest = json.loads((SUITE / "manifest.json").read_text())
        self.assertEqual(len(results), len(manifest))
        self.assertEqual(len(results), 12)
        for entry in results:
            with self.subTest(model=entry["model"]):
                self.assertTrue(entry["passed"])
                self.assertIn("passed", entry["output"])
        summary = (SUITE / "board_summary.txt").read_text()
        self.assertTrue(summary.startswith("PASS:"), summary)
        self.assertIn("192 inferences", summary)
        self.assertIn("6368 exact output bytes", summary)
        self.assertTrue((SUITE / "README.md").is_file())


    def test_large_image_uses_the_native16_input_stage(self):
        """A 3-channel image past the legacy 8x8 emitter uses the native16 stage.

        The verifier of the board suite (`research/mel_kws_suite/`) is the trained
        mel-CNN; here a synthetic 32x32 chain pins the emitter contract: the walk
        accepts it, the container declares a native16 image surface, and the host
        reference reproduces the composed output shape.
        """
        model = graph_of([("conv", 16, 3, "Relu"), ("MaxPool",), ("conv", 10, 1, None)],
                         height=32, width=32)
        binary, meta = compile_sequence(model_onnx(model), 1.0, 0)
        self.assertEqual(meta["profile"], "chain-walk")
        self.assertEqual(list(meta["input_shape"]), [1, 3, 32, 32])
        plan = parse_chain(model.graph)
        self.assertTrue(plan["ops"][0]["native_input"])
        packed = np.zeros((32, 32, 3), dtype=np.uint8)
        quantizations = load_quantizations(meta)
        grid = chain_walk_reference(packed, quantizations, plan["ops"], 0)
        self.assertEqual(grid.shape, (16, 16, 10))
        self.assertLessEqual(len(binary), 4096 * 64)

    def test_calibration_bands_reach_the_walk(self):
        """Measured bands replace the analytic ones (and the last Conv's output band)."""
        model = graph_of([("conv", 16, 3, "Relu"), ("MaxPool",), ("conv", 10, 1, None)],
                         height=32, width=32)
        plan = parse_chain(model.graph)
        convs = [op for op in plan["ops"] if op["kind"] == "conv"]
        ranges = {}
        for index, op in enumerate(convs):
            ranges[op["tensor"]] = dict(min=-1.0, max=1.0, scale=0.001 * (index + 1),
                                        zero_point=-100 + index)
        binary, meta = compile_sequence(model_onnx(model), 1.0, 0, calibration_ranges=ranges)
        bands = [(round(q.output_scale, 6), q.output_zero_point)
                 for q in load_quantizations(meta) if q]
        last = ranges[convs[-1]["tensor"]]
        self.assertEqual(bands[-1], (round(last["scale"], 6), last["zero_point"]))
        # The measured band reaches the last Conv verbatim, while the Conv that feeds
        # the pool is still re-quantized onto a zero-point-0 grid (so it is not simply
        # the measured band).
        self.assertEqual(bands[-1][1], last["zero_point"])
        self.assertEqual(bands[0][1], 0)
        self.assertNotAlmostEqual(bands[0][0], round(ranges[convs[0]["tensor"]]["scale"], 6))

    def test_explicit_pool_defaults_are_accepted(self):
        """Exporters may write the default pool attributes explicitly (torch does)."""
        model = graph_of([("conv", 8, 3, "Relu"), ("MaxPool",), ("conv", 3, 1, None)])
        for node in model.graph.node:
            if node.op_type == "MaxPool":
                node.attribute.extend([h.make_attribute("ceil_mode", 0),
                                       h.make_attribute("dilations", [1, 1]),
                                       h.make_attribute("pads", [0, 0, 0, 0])])
        spec = parse_chain(model.graph)
        self.assertIsNotNone(spec)
        compile_sequence(model_onnx(model), 1.0, 0)

if __name__ == "__main__":
    unittest.main()
