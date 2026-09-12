"""SPDX-License-Identifier: MIT

Inventory and replay of the integer references the scheduler-selectable profiles expose.

This module closes checklist row F13: the three profiles that had no integer reference of
their own now have one, so a user can self-verify those containers, and the retained board
suites are replayed byte-for-byte through the new functions.

Three parts:

* **Inventory.** `SCHEDULER_REFERENCES` maps every emitter entry point the scheduler can
  select to the `module.function` integer reference that reproduces its container, or
  `None` for the two documented gaps. The entry-point set is parsed out of
  `src/open_rknpu/scheduler.py` (so a new or removed scheduler branch fails the test), the
  per-module `*reference` function set is scanned out of the sources (so a profile gaining
  or losing a reference fails the test), and every pinned reference must resolve and be
  callable.
* **Replay.** The newly covered profiles - depthwise and dense `ConvTranspose` (including
  the off-centre taps and the sparse K2/K3/K5 rewrites), the `pooling` profile and the
  three-stage `reduction` profile - are replayed against a sampled, deterministic set of
  retained board suites with byte equality.
* **Rejections.** The exact `ValueError` message of every new reference guard is pinned.

No board, no network: the container's own quantization parameters and the recorded
`inputNNN.u8`/`expectedNNN.i8` are all the replay needs.
"""
from pathlib import Path
import ast
import importlib
import json
import re
import unittest

import numpy as np
import onnx

from open_rknpu.compiler import compile_model
from open_rknpu.depthwise import depthwise_reference
from open_rknpu.network import network_reference
from open_rknpu.pooling import pool_reference
from open_rknpu.quantization import Quantization, reference
from open_rknpu.reduction import reduction_reference
from open_rknpu.scheduler import compile_sequence
from open_rknpu.transposed import transposed_reference

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
SRC = ROOT / "src" / "open_rknpu"


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

# Scheduler-selectable emitter entry points, in the emitted profile's own terms. The key
# is `module.compile_*`; the value is the integer reference function (or a tuple of them)
# that replays the container from its quantization parameters, or `None` when the profile
# has no reference of its own. `open_rknpu.pooling.pool_registers` is the pool task the
# sequence lowering emits for `Conv[/Relu] + 2x2 pool(s)`.
SCHEDULER_REFERENCES = {
    "open_rknpu.activation.compile_leaky": ("open_rknpu.activation.leaky_reference",),
    "open_rknpu.activation.compile_prelu": ("open_rknpu.activation.prelu_reference",),
    "open_rknpu.chain.compile_chain": ("open_rknpu.chain.native_reference",),
    "open_rknpu.chain_n.compile_chain_n": ("open_rknpu.chain_n.chain_n_reference",),
    "open_rknpu.depthwise.compile_depthwise": ("open_rknpu.depthwise.depthwise_reference",),
    "open_rknpu.depthwise.compile_depthwise_pointwise": (
        "open_rknpu.depthwise.depthwise_reference", "open_rknpu.chain.native_reference"),
    "open_rknpu.depthwise_join.compile_depthwise_join": (
        "open_rknpu.depthwise_join.depthwise_join_reference",),
    "open_rknpu.elementwise.compile_constant_mul": ("open_rknpu.elementwise.mul_reference",),
    "open_rknpu.elementwise.compile_elementwise": (
        "open_rknpu.elementwise.add_reference", "open_rknpu.elementwise.sub_reference",
        "open_rknpu.elementwise.max_reference", "open_rknpu.elementwise.mul_reference"),
    "open_rknpu.elementwise.compile_mul_add": ("open_rknpu.elementwise.mul_reference",),
    "open_rknpu.elementwise.compile_mul_clip": ("open_rknpu.elementwise.mul_reference",),
    "open_rknpu.elementwise.compile_mul_relu": ("open_rknpu.elementwise.mul_reference",),
    "open_rknpu.elementwise.compile_per_channel_constant_mul": (
        "open_rknpu.elementwise.mul_reference",),
    "open_rknpu.elementwise.compile_runtime_scale_mul": (
        "open_rknpu.elementwise.runtime_scale_reference",),
    "open_rknpu.elementwise.compile_standalone_mul": ("open_rknpu.elementwise.mul_reference",),
    "open_rknpu.elementwise_chain.compile_elementwise_dag": (
        "open_rknpu.elementwise_chain.chain_reference",),
    "open_rknpu.elementwise_multi.compile_multi_input_dag": (
        "open_rknpu.elementwise_multi.multi_input_reference",),
    "open_rknpu.graph.compile_diamond": ("open_rknpu.graph.diamond_reference",),
    "open_rknpu.graph.compile_join_chain": (
        "open_rknpu.graph.diamond_reference", "open_rknpu.graph.join_chain_scale_reference",
        "open_rknpu.graph.join_chain_residual_reference"),
    "open_rknpu.graph.compile_two_head": None,
    "open_rknpu.join_dag.compile_join_dag": ("open_rknpu.join_dag.join_dag_reference",),
    "open_rknpu.layout.compile_spatial_reshape": None,
    "open_rknpu.lut.compile_lut": ("open_rknpu.lut.lut_reference",),
    "open_rknpu.native.compile_native_input": ("open_rknpu.native.native_input_reference",),
    "open_rknpu.pool_join.compile_pool_join": ("open_rknpu.pool_join.pool_join_reference",),
    "open_rknpu.pooled_branches.compile_pooled_branches": (
        "open_rknpu.pooled_branches.pooled_branches_reference",),
    "open_rknpu.pooling.pool_registers": ("open_rknpu.pooling.pool_reference",),
    "open_rknpu.quantized_import.compile_qdq_conv": ("open_rknpu.native.native_input_reference",),
    "open_rknpu.quantized_import.compile_qlinearconv": (
        "open_rknpu.native.native_input_reference",),
    "open_rknpu.strided.compile_strided": ("open_rknpu.native.native_input_reference",),
    "open_rknpu.tiled_chain.compile_tiled_chain": ("open_rknpu.chain_n.chain_n_reference",),
    "open_rknpu.transposed.compile_transposed": ("open_rknpu.transposed.transposed_reference",),
    "open_rknpu.walk.compile_chain_walk": ("open_rknpu.walk.chain_walk_reference",),
    "open_rknpu.walk.compile_join_walk": ("open_rknpu.walk.join_walk_reference",),
}

# Profiles reached through `open_rknpu.compiler.compile_model` rather than the scheduler:
# the legacy single-program container profiles 3..8.
LEGACY_REFERENCES = {
    "open_rknpu.pooling.compile_pool": ("open_rknpu.pooling.pool_reference",),
    "open_rknpu.reduction.compile_reduction": ("open_rknpu.reduction.reduction_reference",),
    "open_rknpu.network.compile_network": ("open_rknpu.network.network_reference",),
}

# The only profiles without a reference of their own, with the retained evidence that
# would be needed to add one. Both are container-only today: their suites pin the emitted
# bytes but never compare an integer result.
DOCUMENTED_GAPS = {
    "open_rknpu.graph.compile_two_head": (
        "two-head fan-out has no integer reference; `research/two_head_suite/` records "
        "containers only"),
    "open_rknpu.layout.compile_spatial_reshape": (
        "terminal Reshape is a byte permutation; `research/spatial_reshape_suite/` "
        "records containers only"),
}

# Every `def *reference*(` in `src/open_rknpu`, pinned so a profile that gains or loses a
# reference function fails here. `pooling`, `reduction` and `transposed` are the F13
# additions.
MODULE_REFERENCES = {
    "open_rknpu.activation": {"leaky_reference", "prelu_reference"},
    "open_rknpu.chain": {"native_reference"},
    "open_rknpu.chain_n": {"chain_n_reference", "chain_n_reference_layers"},
    "open_rknpu.depthwise": {"depthwise_reference"},
    "open_rknpu.depthwise_join": {"depthwise_join_reference"},
    "open_rknpu.elementwise": {"add_reference", "sub_reference", "max_reference",
                               "mul_reference", "mul_requant_reference",
                               "runtime_scale_reference"},
    "open_rknpu.elementwise_chain": {"add_reference", "sub_reference", "max_reference",
                                     "chain_reference"},
    "open_rknpu.elementwise_multi": {"multi_input_reference"},
    "open_rknpu.graph": {"join_reference", "diamond_reference", "join_chain_scale_reference",
                         "join_chain_residual_reference"},
    "open_rknpu.join_dag": {"join_dag_reference"},
    "open_rknpu.lut": {"lut_reference"},
    "open_rknpu.native": {"native_input_reference"},
    "open_rknpu.network": {"network_reference"},
    "open_rknpu.pool_join": {"pool_join_reference"},
    "open_rknpu.pooled_branches": {"pooled_branches_reference"},
    "open_rknpu.pooling": {"pool_reference", "pool_codes_reference"},
    "open_rknpu.quantization": {"reference"},
    "open_rknpu.reduction": {"reduction_reference"},
    "open_rknpu.transposed": {"transposed_reference"},
    "open_rknpu.walk": {"chain_walk_reference", "join_walk_reference"},
}

REFERENCE_DEF = re.compile(r"^def ([A-Za-z_][A-Za-z0-9_]*)\(", re.M)


def scheduler_entry_points():
    """The emitter entry points `compile_sequence` dispatches to, from its own source."""
    tree = ast.parse((SRC / "scheduler.py").read_text())
    entries = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            for alias in node.names:
                if alias.name == "pool_registers":
                    entries.add("open_rknpu.pooling.pool_registers")
                elif alias.name.startswith("compile_") and alias.name != "compile_model":
                    entries.add(f"open_rknpu.{node.module}.{alias.name}")
    return entries


def discovered_references():
    references = {}
    for path in sorted(SRC.glob("*.py")):
        names = {name for name in REFERENCE_DEF.findall(path.read_text())
                 if "reference" in name}
        if names:
            references[f"open_rknpu.{path.stem}"] = names
    return references


def quantization_from(params):
    return Quantization(**{key: np.array(value) if isinstance(value, list) else value
                           for key, value in params.items()})


def recorded_cases(folder, index, shape, limit=None):
    data = np.frombuffer((folder / f"input{index:03}.u8").read_bytes(), np.uint8)
    return data.reshape(-1, *shape)[:limit]


def recorded_expected(folder, index):
    return np.fromfile(folder / f"expected{index:03}.i8", dtype=np.int8)


class InventoryTests(unittest.TestCase):
    def test_scheduler_entry_points_match_the_pinned_inventory(self):
        self.assertEqual(scheduler_entry_points(), set(SCHEDULER_REFERENCES))

    def test_documented_gaps_are_exactly_the_profiles_without_a_reference(self):
        gaps = {key for key, reference in SCHEDULER_REFERENCES.items() if reference is None}
        self.assertEqual(gaps, set(DOCUMENTED_GAPS))
        for reason in DOCUMENTED_GAPS.values():
            self.assertTrue(reason)

    def test_every_pinned_reference_resolves_to_a_callable(self):
        for key, references in {**SCHEDULER_REFERENCES, **LEGACY_REFERENCES}.items():
            for target in references or ():
                module, _, name = target.rpartition(".")
                self.assertTrue(callable(getattr(importlib.import_module(module), name)),
                                f"{key} pins missing reference {target}")

    def test_reference_function_inventory_is_unchanged(self):
        self.assertEqual(discovered_references(), MODULE_REFERENCES)


# ---------------------------------------------------------------------------
# ConvTranspose replay: depthwise, dense, off-centre taps and the sparse rewrites
# ---------------------------------------------------------------------------

# (suite, model index, cases) sampled across every retained transposed suite. The geometry
# comes from the suite manifest when it records it (`resolved_pads` for the auto_pad and
# output_shape modes) and from the ONNX node otherwise.
TRANSPOSED_SAMPLES = (
    ("transpose_channels_suite", 0, 2),           # depthwise C1 K3 stride1 pads1
    ("transpose_channels_suite", 3, 2),           # depthwise C4 K2 stride2/1
    ("transpose_k5_suite", 0, 2),                 # depthwise K5, asymmetric zero points
    ("transpose_k5_suite", 5, 2),                 # depthwise K5, per-axis stride + padding
    ("transpose_k5_dilation_suite", 6, 2),        # K3 dilation2 -> sparse K5, unequal stride
    ("transpose_k5_dilation_suite", 7, 2),        # K3 dilation2 -> sparse K5, SAME_UPPER
    ("transpose_dilation_suite", 3, 2),           # K2 dilation2 -> sparse K3, SAME_UPPER
    ("transpose_dense_suite", 0, 2),              # dense C1->C5 (ic=1 discriminator)
    ("transpose_dense_suite", 7, 2),              # dense C16->C16
    ("transpose_dilation_dense_suite", 2, 2),     # dense K3 dilation2 -> sparse K5
    ("transpose_grouped_suite", 0, 2),            # group2 expanded to dense
    ("transpose_unequal_suite", 5, 2),            # off-centre unequal stride, SAME_LOWER
    ("transpose_rectangular_suite", 3, 2),        # rectangular kernel -> sparse K3
    ("transpose_overlap_suite", 0, 2),            # K3 stride2 overlapping taps
    ("transpose_padding_suite", 3, 2),            # K2 output_padding
    ("transpose_output_shape_suite", 1, 2),       # output_shape + SAME_LOWER
    ("transpose_output_quantization_suite", 0, 2),# independent output conversion
    ("transpose_stride1_suite", 3, 2),            # K3 stride1 asymmetric padding
    ("transpose_no_bias_suite", 1, 2),            # missing bias synthesized as zero
)


class TransposedReferenceReplayTests(unittest.TestCase):
    def geometry(self, suite, index, entry):
        model = onnx.load(suite / f"model{index:03}.onnx")
        node = model.graph.node[-1]
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        pads = list(entry.get("resolved_pads") or entry.get("pads")
                    or attrs.get("pads", [0, 0, 0, 0]))
        strides = list(entry.get("strides") or attrs.get("strides", [1, 1]))
        output_padding = list(entry.get("output_padding") or attrs.get("output_padding", [0, 0]))
        return model, pads, strides, output_padding

    def replay(self, suite_name, index, cases):
        suite = RESEARCH / suite_name
        entry = next(e for e in json.loads((suite / "manifest.json").read_text())
                     if e["index"] == index)
        kwargs = {}
        if entry.get("compile_output_range"):
            kwargs = dict(input_scale=entry.get("input_scale", 1.0),
                          input_zero_point=entry.get("input_zero_point", 0),
                          output_range=entry["compile_output_range"])
        _, meta = compile_sequence(suite / f"model{index:03}.onnx", **kwargs)
        stem = quantization_from(meta["first"]["quantization"])
        transposed = quantization_from(meta["transposed_quantization"])
        model, pads, strides, output_padding = self.geometry(suite, index, entry)
        shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
        _, channels, height, width = shape
        output_shape = entry.get("output_shape") or meta["output_shape_nhwc"][1:]
        samples = recorded_cases(suite, index, (height, width, channels), cases)
        produced = np.stack([transposed_reference(sample, stem, transposed, pads, strides,
                                                  output_padding, output_shape)
                             for sample in samples])
        expected = recorded_expected(suite, index).reshape(-1, *produced.shape[1:])[:produced.shape[0]]
        return produced, expected

    def test_retained_transposed_suites_replay_byte_for_byte(self):
        for suite_name, index, cases in TRANSPOSED_SAMPLES:
            with self.subTest(suite=suite_name, model=f"{index:03}"):
                produced, expected = self.replay(suite_name, index, cases)
                self.assertEqual(produced.astype(np.int8).tobytes(), expected.tobytes())

    def test_off_centre_taps_are_not_the_symmetric_oracle(self):
        """A K3 stride-1 corner tap changes the result versus the centre-only oracle.

        `depthwise_reference` is exact for the symmetric centre tap; this pins that the
        transposed reference is driven by the *scattered* weights, so an off-centre tap
        cannot silently agree with a centre-only implementation.
        """
        produced, expected = self.replay("transpose_channels_suite", 0, 2)
        self.assertEqual(produced.astype(np.int8).tobytes(), expected.tobytes())
        suite = RESEARCH / "transpose_channels_suite"
        _, meta = compile_sequence(suite / "model000.onnx")
        stem = quantization_from(meta["first"]["quantization"])
        transposed = quantization_from(meta["transposed_quantization"])
        model = onnx.load(suite / "model000.onnx")
        shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
        _, channels, height, width = shape
        sample = recorded_cases(suite, 0, (height, width, channels), 1)[0]
        centre_oracle = depthwise_reference(reference(sample, stem), transposed,
                                            stem.output_zero_point)
        self.assertNotEqual(centre_oracle.tobytes(), expected[0].tobytes(),
                            "the sampled transposed kernel must have off-centre taps")

    def test_learned_k2_stride2_suite_replays_byte_for_byte(self):
        """`transpose_learned_suite` retains no ONNX, only `modelNNN.json` quantization.

        The 6 board-passing containers (96 inferences) were built directly from a learned
        K2/stride-2 depthwise kernel; their `weights`/`first.quantization`/
        `quantization` fields are the recorded parameters the reference is defined over.
        """
        suite = RESEARCH / "transpose_learned_suite"
        for index in range(6):
            with self.subTest(model=f"{index:03}"):
                record = json.loads((suite / f"model{index:03}.json").read_text())
                stem = quantization_from(record["first"]["quantization"])
                transposed = quantization_from(record["quantization"])
                samples = recorded_cases(suite, index, (8, 8, 3))
                produced = np.stack([transposed_reference(sample, stem, transposed,
                                                          strides=(2, 2), output_shape=(16, 16, 3))
                                     for sample in samples])
                expected = recorded_expected(suite, index).reshape(produced.shape)
                self.assertEqual(produced.tobytes(), expected.tobytes())


# ---------------------------------------------------------------------------
# pooling / reduction replay
# ---------------------------------------------------------------------------

# (suite, model indices, cases per model)
SCHEDULED_POOL_SAMPLES = ("scheduled_pool_suite", (0, 3, 6, 9), 4)
API_POOL_SAMPLES = (("pool_api_suite", "pool", 1, (0, 2)), ("reduction_api_suite", "reduce", 3, (0, 2)))


class PoolingReferenceReplayTests(unittest.TestCase):
    def test_scheduled_pool_suite_replays_byte_for_byte(self):
        suite = RESEARCH / "scheduled_pool_suite"
        for index in SCHEDULED_POOL_SAMPLES[1]:
            with self.subTest(model=f"{index:03}"):
                _, meta = compile_sequence(suite / f"model{index:03}.onnx", 0.5, 128)
                quantization = quantization_from(meta["quantization"])
                stages = meta["pool_stages"]
                model = onnx.load(suite / f"model{index:03}.onnx")
                shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
                _, channels, height, width = shape
                samples = recorded_cases(suite, index, (height, width, channels),
                                         SCHEDULED_POOL_SAMPLES[2])
                produced = np.stack([pool_reference(sample, quantization, stages[0], len(stages))
                                     for sample in samples])
                expected = recorded_expected(suite, index).reshape(-1, *produced.shape[1:])[:produced.shape[0]]
                self.assertEqual(produced.tobytes(), expected.tobytes())

    def test_legacy_pool_and_reduction_api_suites_replay_byte_for_byte(self):
        for suite_name, prefix, levels, indices in API_POOL_SAMPLES:
            suite = RESEARCH / suite_name
            manifest = {entry["index"]: entry
                        for entry in json.loads((suite / "manifest.json").read_text())}
            reference = pool_reference if levels == 1 else reduction_reference
            for index in indices:
                entry = manifest[index]
                kind = "MaxPool" if entry["pool"] == "max" else "AveragePool"
                with self.subTest(suite=suite_name, model=f"{index:03}"):
                    _, meta = compile_model(
                        RESEARCH / f"generated/{prefix}_{entry['pool']}_r{entry['relu']}.onnx")
                    quantization = quantization_from(meta["quantization"])
                    samples = recorded_cases(suite, index, (8, 8, 3), 4)
                    produced = np.stack([reference(sample, quantization, kind, levels)
                                         for sample in samples])
                    expected = recorded_expected(suite, index).reshape(-1, *produced.shape[1:])[:produced.shape[0]]
                    self.assertEqual(produced.tobytes(), expected.tobytes())

    def test_network_suite_replays_byte_for_byte(self):
        suite = RESEARCH / "network_suite"
        for index in (0, 3, 6, 9):
            with self.subTest(model=f"{index:03}"):
                _, meta = compile_model(suite / f"model{index:03}.onnx")
                quantizations = [quantization_from(meta["first"]["quantization"]),
                                 quantization_from(meta["second"])]
                samples = recorded_cases(suite, index, (8, 8, 3), 2)
                produced = np.stack([network_reference(sample, quantizations, meta["pool"], 3)
                                     for sample in samples])
                expected = recorded_expected(suite, index).reshape(-1, *produced.shape[1:])[:produced.shape[0]]
                self.assertEqual(produced.tobytes(), expected.tobytes())

    def test_mnist_max_pool_prefix_replays_byte_for_byte(self):
        suite = RESEARCH / "mnist_pool_suite"
        manifest = json.loads((RESEARCH / "mnist_first_suite" / "manifest.json").read_text())[0]
        _, meta = compile_sequence(suite / "model000.onnx", manifest["input_scale"],
                                   manifest["input_zero_point"])
        quantization = quantization_from(meta["quantization"])
        stages = meta["pool_stages"]
        samples = recorded_cases(suite, 0, (28, 28, 1), 4)
        produced = np.stack([pool_reference(sample, quantization, stages[0], len(stages))
                             for sample in samples])
        expected = recorded_expected(suite, 0).reshape(-1, *produced.shape[1:])[:produced.shape[0]]
        self.assertEqual(produced.tobytes(), expected.tobytes())


# ---------------------------------------------------------------------------
# Exact rejection messages
# ---------------------------------------------------------------------------


def fake_quantization(outputs, inputs, kernel=1):
    """A minimal live `Quantization` so the reference guards can be exercised directly."""
    return Quantization(np.ones((outputs, inputs * kernel * kernel), np.int64),
                        np.zeros(outputs, np.int64), np.ones(outputs, np.float32),
                        np.zeros(outputs, np.int64), np.full(outputs, 16384, np.int64),
                        1, 0, 1.0, 0, kernel)


class ReferenceRejectionTests(unittest.TestCase):
    def setUp(self):
        self.sample = np.full((8, 8, 3), 128, np.uint8)
        self.quantization = fake_quantization(3, 1, 3)

    def test_pool_reference_exact_messages(self):
        with self.assertRaisesRegex(ValueError,
                                    r"^pool reference supports MaxPool or AveragePool$"):
            pool_reference(self.sample, self.quantization, "MinPool")
        with self.assertRaisesRegex(
                ValueError, r"^pool reference requires at least one 2x2 pooling level$"):
            pool_reference(self.sample, self.quantization, "MaxPool", 0)

    def test_reduction_reference_exact_message(self):
        with self.assertRaisesRegex(
                ValueError, r"^reduction reference models exactly three 2x2 pooling levels$"):
            reduction_reference(self.sample, self.quantization, "MaxPool", 2)

    def test_network_reference_exact_messages(self):
        with self.assertRaisesRegex(
                ValueError, r"^network reference supports MaxPool or AveragePool$"):
            network_reference(self.sample, [], "MinPool")
        with self.assertRaisesRegex(
                ValueError, r"^network reference models exactly three 2x2 pooling levels$"):
            network_reference(self.sample, [], "MaxPool", 2)

    def test_transposed_reference_exact_messages(self):
        with self.assertRaisesRegex(
                ValueError, r"^transposed reference requires four pad values$"):
            transposed_reference(self.sample, self.quantization, self.quantization,
                                 pads=(0, 0, 0))
        with self.assertRaisesRegex(
                ValueError, r"^transposed reference supports per-axis stride 1 or 2$"):
            transposed_reference(self.sample, self.quantization, self.quantization,
                                 strides=(3, 1))
        with self.assertRaisesRegex(
                ValueError,
                r"^transposed reference requires output_padding below its axis stride$"):
            transposed_reference(self.sample, self.quantization, self.quantization,
                                 strides=(2, 2), output_padding=(2, 0))
        with self.assertRaisesRegex(
                ValueError, r"^transposed reference supports K2/K3/K5 kernels$"):
            transposed_reference(self.sample, self.quantization,
                                 fake_quantization(3, 1, 4))
        with self.assertRaisesRegex(
                ValueError, r"^transposed reference requires the stem channels to match "
                            r"the weight layout$"):
            # A depthwise C3 transposed cannot read a 2-channel stem grid.
            transposed_reference(self.sample, fake_quantization(2, 3, 1),
                                 self.quantization, pads=(1, 1, 1, 1))


if __name__ == "__main__":
    unittest.main()
