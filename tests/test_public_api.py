# SPDX-License-Identifier: MIT
"""Machine-check the public surface that ``docs/api-stability.md`` promises.

The stability page tells a user what will not break within 0.x.  A promise nobody tests
rots, so this module pins the supported names, their parameter *names and defaults* (not
annotations, which may evolve), the ``open-rknpu`` console entry point, and the ``meta``
keys ``compile_sequence`` returns for a tiny Conv.

Deliberately out of scope: the internal modules (the emitter modules, ``graph``,
``compose``, ``liveness``, ``normalize`` and the ``walk`` planner internals).  They are
listed in the stability page as changeable in any release, so pinning them here would
contradict that page.  ``open_rknpu`` has no ``__all__``; the stability page states that,
and this module asserts the supported names are reachable by their documented module path
instead.

No board, no network, no sleep: three in-memory models compiled into temporary files.
"""
import importlib
import inspect
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# The supported surface (docs/api-stability.md)
# --------------------------------------------------------------------------- #

# qualname -> (parameter names in order, {name: default} for the parameters that have one)
CORE = {
    "open_rknpu.scheduler.compile_sequence": (
        ("path", "input_scale", "input_zero_point", "output_range",
         "mul_operand_zero_points", "mutable_weights", "mutable_constants",
         "calibration_ranges", "expose_intermediates", "reuse_intermediates",
         "asymmetric_depthwise", "per_channel_mul", "submission", "tiles"),
        dict(input_scale=1.0, input_zero_point=0, output_range=None,
             mul_operand_zero_points=(0, 0), mutable_weights=False, mutable_constants=False,
             calibration_ranges=None, expose_intermediates=False, reuse_intermediates=False,
             asymmetric_depthwise=False, per_channel_mul=False, submission=None, tiles=None),
    ),
    "open_rknpu.calibration.measure": (
        ("model_path", "directory", "method", "percentile", "bins"),
        dict(method="minmax", percentile=99.99, bins=2048),
    ),
    "open_rknpu.sequence.encode_sequence": (
        ("payload", "input_shape", "output_shape", "input_stride", "arena_bytes",
         "input_offset", "output_offset", "tasks", "input_scale", "input_zero_point",
         "output_scale", "output_zero_point", "serial", "input_layout", "batch",
         "input_tensor_count", "constants"),
        dict(input_scale=1.0, input_zero_point=0, output_scale=1.0, output_zero_point=0,
             serial=False, input_layout="packed", batch=1, input_tensor_count=1,
             constants=()),
    ),
    "open_rknpu.sequence.decode_sequence": (("data",), {}),
    "open_rknpu.model.encode": (("payload", "metadata"), {}),
    "open_rknpu.model.decode": (("data",), {}),
    "open_rknpu.quantization.Quantization": (
        ("weights", "weight_zero_points", "weight_scales", "biases", "channel_multipliers",
         "multiplier", "shift", "output_scale", "output_zero_point", "kernel_size", "relu",
         "input_scale", "input_zero_point"),
        dict(kernel_size=1, relu=False, input_scale=1.0, input_zero_point=0),
    ),
    "open_rknpu.quantization.reference": (
        ("inputs", "quantization", "rounding"), dict(rounding="separate"),
    ),
    # F11: the supported mutable-parameter helper API (docs/api-stability.md).
    "open_rknpu.mutable.compile_mutable": (
        ("model", "mutable_weights", "mutable_constants", "kwargs"),
        dict(mutable_weights=False, mutable_constants=False),
    ),
    "open_rknpu.mutable.constant_regions": (("binary",), {}),
    "open_rknpu.mutable.constant_payload": (("binary", "name", "index"), dict(name=None, index=0)),
    "open_rknpu.mutable.replace_constant": (
        ("binary", "replacement", "name", "index"), dict(name=None, index=0),
    ),
    "open_rknpu.mutable.graft_region": (
        ("host", "donor", "name", "index"), dict(name=None, index=0),
    ),
    "open_rknpu.mutable.program_bytes": (("binary",), {}),
}

# The per-profile references documented next to their emitters (docs/api.md table).
REFERENCES = {
    "open_rknpu.native.native_input_reference": (
        ("inputs", "q", "zero_point", "pads", "strides", "dilations", "upper_code"),
        dict(pads=None, strides=(1, 1), dilations=(1, 1), upper_code=None),
    ),
    "open_rknpu.chain.native_reference": (
        ("inputs", "q", "input_zero_point"), dict(input_zero_point=-128),
    ),
    "open_rknpu.lut.lut_reference": (
        ("inputs", "weights", "bias", "kind", "input_scale", "input_zero_point"),
        dict(input_scale=1.0, input_zero_point=128),
    ),
    "open_rknpu.lut.stem_range": (
        ("weights", "bias", "input_scale", "input_zero_point"),
        dict(input_scale=1.0, input_zero_point=128),
    ),
    "open_rknpu.activation.leaky_reference": (("inputs", "q", "alpha"), {}),
    "open_rknpu.activation.prelu_reference": (("inputs", "q", "slopes"), {}),
    "open_rknpu.graph.diamond_reference": (
        ("inputs", "stem_quantization", "head_quantizations", "kind", "tail_quantizations",
         "join_zero_point", "head_kinds", "depthwise_quantizations"),
        dict(tail_quantizations=(), join_zero_point=0, head_kinds=None,
             depthwise_quantizations=None),
    ),
    "open_rknpu.join_dag.join_dag_reference": (
        ("inputs", "stem_quantization", "head_quantizations", "head_names", "expression",
         "head_kinds", "depthwise_quantizations", "branch_names", "branch_quantization", "pool"),
        dict(head_kinds=None, depthwise_quantizations=None, branch_names=None,
             branch_quantization=None, pool=None),
    ),
    "open_rknpu.graph.join_chain_scale_reference": (
        ("inputs", "stem_quantization", "head_quantizations", "kinds", "operand_codes",
         "head_kinds", "depthwise_quantizations", "runtime_scale", "output_zero_point"),
        dict(head_kinds=None, depthwise_quantizations=None, runtime_scale=None,
             output_zero_point=0),
    ),
    "open_rknpu.walk.chain_walk_reference": (
        ("inputs", "quantizations", "ops", "input_zero_point"), dict(input_zero_point=0),
    ),
    "open_rknpu.walk.join_walk_reference": (
        ("inputs", "quantizations", "plan", "join_zero_point"), dict(join_zero_point=0),
    ),
    "open_rknpu.walk.load_quantizations": (("meta",), {}),
    "open_rknpu.depthwise.depthwise_reference": (("inputs", "q", "input_zero_point"), {}),
    "open_rknpu.pooling.pool_reference": (
        ("inputs", "quantization", "kind", "levels"), dict(levels=1),
    ),
    "open_rknpu.reduction.reduction_reference": (
        ("inputs", "quantization", "kind", "levels"), dict(levels=3),
    ),
    "open_rknpu.transposed.transposed_reference": (
        ("inputs", "stem_quantization", "quantization", "pads", "strides",
         "output_padding", "output_shape"),
        dict(pads=(0, 0, 0, 0), strides=(1, 1), output_padding=(0, 0), output_shape=None),
    ),
    "open_rknpu.elementwise.add_reference": (("a", "b"), {}),
    "open_rknpu.elementwise.sub_reference": (("a", "b"), {}),
    "open_rknpu.elementwise.max_reference": (("a", "b"), {}),
    "open_rknpu.elementwise.mul_reference": (("a", "b"), {}),
    "open_rknpu.elementwise.mul_requant_reference": (
        ("a", "b", "multiplier", "shift", "zero_point", "operand_zero_points"),
        dict(operand_zero_points=(0, 0)),
    ),
    "open_rknpu.elementwise.runtime_scale_reference": (
        ("image", "operand_codes", "stem_quantization"), {},
    ),
}

SUPPORTED = dict(CORE)
SUPPORTED.update(REFERENCES)

# Modules the stability page marks internal; a subset is enough to assert the split is not
# accidentally flattened into a package-wide ``__all__``.
INTERNAL = ("compose", "liveness", "normalize", "walk", "graph", "native", "scheduler")

# The documented profile-identifier keys (docs/api-stability.md, "meta keys").
PROFILE_KEYS = ("profile", "sequence_profile", "lut_profile", "transposed_profile",
                "depthwise_profile", "elementwise_profile")


def dotted(qualname):
    module_name, _, attribute = qualname.rpartition(".")
    return importlib.import_module(module_name), attribute


def tiny_conv(output_channels=4):
    rng = np.random.default_rng(0)
    weight = rng.uniform(-0.3, 0.3, (output_channels, 3, 3, 3)).astype(np.float32)
    bias = rng.uniform(-0.3, 0.3, (output_channels,)).astype(np.float32)
    nodes = [h.make_node("Conv", ["input", "w", "b"], ["output"], kernel_shape=[3, 3],
                         pads=[1, 1, 1, 1])]
    graph = h.make_graph(nodes, "conv",
                         [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, [1, output_channels, 8, 8])],
                         [nh.from_array(weight, "w"), nh.from_array(bias, "b")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def tiny_chain():
    """A Conv-Relu-Conv graph: the smallest model that reports ``profile`` (legacy 2)."""
    rng = np.random.default_rng(1)
    first = rng.uniform(-0.3, 0.3, (4, 3, 3, 3)).astype(np.float32)
    first_bias = rng.uniform(-0.3, 0.3, (4,)).astype(np.float32)
    last = rng.uniform(-0.3, 0.3, (3, 4, 3, 3)).astype(np.float32)
    last_bias = rng.uniform(-0.3, 0.3, (3,)).astype(np.float32)
    nodes = [h.make_node("Conv", ["input", "w1", "b1"], ["c0"], kernel_shape=[3, 3],
                         pads=[1, 1, 1, 1]),
             h.make_node("Relu", ["c0"], ["r0"]),
             h.make_node("Conv", ["r0", "w2", "b2"], ["output"], kernel_shape=[3, 3],
                         pads=[1, 1, 1, 1])]
    graph = h.make_graph(nodes, "chain",
                         [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
                         [nh.from_array(first, "w1"), nh.from_array(first_bias, "b1"),
                          nh.from_array(last, "w2"), nh.from_array(last_bias, "b2")],
                         value_info=[h.make_tensor_value_info("c0", 1, [1, 4, 8, 8]),
                                     h.make_tensor_value_info("r0", 1, [1, 4, 8, 8])])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def tiny_lut():
    """A diagonal 1x1 Conv plus Sigmoid: the smallest model that reports ``lut_profile``."""
    weight = np.zeros((3, 3, 1, 1), np.float32)
    for channel in range(3):
        weight[channel, channel, 0, 0] = np.float32(1 / 32)
    bias = np.zeros((3,), np.float32)
    nodes = [h.make_node("Conv", ["input", "w", "b"], ["stem"], kernel_shape=[1, 1]),
             h.make_node("Sigmoid", ["stem"], ["output"])]
    graph = h.make_graph(nodes, "lut",
                         [h.make_tensor_value_info("input", 1, [1, 3, 8, 8])],
                         [h.make_tensor_value_info("output", 1, [1, 3, 8, 8])],
                         [nh.from_array(weight, "w"), nh.from_array(bias, "b")],
                         value_info=[h.make_tensor_value_info("stem", 1, [1, 3, 8, 8])])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)])
    model.ir_version = 8
    return onnx.shape_inference.infer_shapes(model)


def compile_graph(model, **kwargs):
    from open_rknpu.scheduler import compile_sequence
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "model.onnx"
        onnx.save(model, path)
        return compile_sequence(path, **kwargs)


def project_scripts():
    """The ``[project.scripts]`` table, parsed without ``tomllib`` (Python 3.10 has none)."""
    section = None
    scripts = {}
    for raw in (ROOT / "pyproject.toml").read_text().splitlines():
        line = re.sub(r"\s+#.*$", "", raw).strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section == "project.scripts" and "=" in line:
            key, value = line.split("=", 1)
            scripts[key.strip()] = value.strip().strip('"').strip("'")
    return scripts


class PublicSurfaceTest(unittest.TestCase):
    """docs/api-stability.md is a contract; this class is the executable copy."""

    def test_supported_names_import_from_their_documented_module(self):
        for qualname in SUPPORTED:
            with self.subTest(name=qualname):
                module, attribute = dotted(qualname)
                self.assertTrue(hasattr(module, attribute), qualname)
                self.assertTrue(callable(getattr(module, attribute)), qualname)

    def test_supported_signatures_match_names_and_defaults(self):
        for qualname, (names, defaults) in SUPPORTED.items():
            with self.subTest(name=qualname):
                module, attribute = dotted(qualname)
                signature = inspect.signature(getattr(module, attribute))
                parameters = signature.parameters
                self.assertEqual(tuple(parameters), tuple(names), qualname)
                measured = {name: parameter.default for name, parameter in parameters.items()
                            if parameter.default is not inspect.Parameter.empty}
                self.assertEqual(measured, defaults, qualname)

    def test_console_script_is_declared_and_points_at_cli_main(self):
        self.assertEqual(project_scripts().get("open-rknpu"), "open_rknpu.cli:main")
        from open_rknpu.cli import main
        self.assertTrue(callable(main))
        self.assertEqual(tuple(inspect.signature(main).parameters), ())

    def test_compile_sequence_meta_keys_for_a_tiny_conv(self):
        binary, meta = compile_graph(tiny_conv())
        self.assertIsInstance(binary, bytes)
        self.assertEqual(meta["shape_nhwc"], [1, 8, 8, 3])
        self.assertEqual(meta["output_shape_nhwc"], [1, 8, 8, 4])
        self.assertEqual(meta["submission"], "serial")
        self.assertEqual(meta["input_scale"], 1.0)
        self.assertEqual(meta["input_zero_point"], 0)
        self.assertIsInstance(meta["output_scale"], float)
        self.assertIsInstance(meta["output_zero_point"], int)
        self.assertIsInstance(meta["quantization"], dict)
        for key in ("weights", "weight_zero_points", "weight_scales", "biases",
                    "channel_multipliers", "multiplier", "shift", "output_scale",
                    "output_zero_point"):
            self.assertIn(key, meta["quantization"])

    def test_compile_sequence_reports_a_profile_identifier(self):
        # A bare dense Conv reports none (only ``pool_stages``); the promise is pinned on
        # tiny Conv profiles that define one, as docs/api-stability.md documents.
        _, chain = compile_graph(tiny_chain())
        self.assertEqual(chain["profile"], 2)
        self.assertEqual(chain["sequence_profile"], "Conv-Relu-Conv")
        _, lut = compile_graph(tiny_lut(), input_zero_point=128)
        self.assertEqual(lut["lut_profile"], "sigmoid-8x8-c3")
        for meta in (chain, lut):
            self.assertIn("submission", meta)
            self.assertTrue(any(key in meta for key in PROFILE_KEYS))

    def test_internal_modules_are_not_in_a_package_wide_all(self):
        import open_rknpu
        if hasattr(open_rknpu, "__all__"):
            for name in INTERNAL:
                self.assertNotIn(name, open_rknpu.__all__)
        else:
            # docs/api-stability.md states the package exports no ``__all__``; the
            # supported names are reachable by their documented module path instead.
            self.assertFalse(hasattr(open_rknpu, "__all__"))
            for qualname in SUPPORTED:
                module, attribute = dotted(qualname)
                self.assertTrue(hasattr(module, attribute), qualname)


if __name__ == "__main__":
    unittest.main()
