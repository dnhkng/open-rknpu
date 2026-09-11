"""SPDX-License-Identifier: MIT

Sampled replay of the published board suites: containers, then recorded cases.

The full sweep (`research/verify_suites.py`, `research/campaign_sweep.py`) takes minutes
and is a CI release job. This module keeps the same regression cheap enough for the unit
suite by replaying a fixed, deterministic sample:

* containers - the first three `model*.onnx` of each of the 14 campaign suites, plus a
  handful from the walk/join/depthwise/LUT/native suites. Each is recompiled with
  `compile_sequence` and the bytes must equal the retained `model*.bin`. The only
  permitted exceptions are the 12 pre-existing drifts pinned by
  `research/campaign_sweep.py::EXPECTED_DRIFT`; that set is read out of the sweep
  script's AST rather than imported, because importing it would run the whole sweep.
* recorded cases - five models that also ship `input*.u8`/`expected*.i8` are replayed
  through the *same* integer reference the suite generator used (read from
  `research/build_<suite>.py`): `chain_n_reference_layers` for `chain_multi_suite`,
  `chain_walk_reference` for `walk_chain_suite`, `join_walk_reference` for
  `walk_join_suite`, `depthwise_join_reference` for `depthwise_join_suite`, and
  `diamond_reference` for `diamond_suite`. Every other sampled suite is container-only
  here (its builder does have a reference, so nothing is silently skipped); replaying
  them all is what the full sweep does, not this module.

No board, no network, and no compiler state leaks between cases.
"""
from pathlib import Path
import ast
import json
import time
import unittest

import numpy as np
import onnx

from open_rknpu.chain_n import chain_n_reference_layers
from open_rknpu.graph import diamond_reference
from open_rknpu.depthwise_join import depthwise_join_reference
from open_rknpu.quantization import Quantization
from open_rknpu.scheduler import compile_sequence
from open_rknpu.walk import (chain_walk_reference, join_walk_reference, load_quantizations,
                             parse_chain, parse_join_walk)

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"


def _sweep_literal(name):
    """Read a module-level literal out of campaign_sweep.py without executing it."""
    tree = ast.parse((RESEARCH / "campaign_sweep.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(target, "id", None) == name
                                                for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("research/campaign_sweep.py no longer defines " + name)


CAMPAIGN = tuple(_sweep_literal("CAMPAIGN"))
EXPECTED_DRIFT = frozenset(_sweep_literal("EXPECTED_DRIFT"))
CAMPAIGN_SAMPLE = 3
# (suite, models, how to reproduce the compile flags the builder used)
EXTRA_SAMPLES = (
    ("walk_chain_suite", 3, "default"),
    ("walk_join_suite", 3, "default"),
    ("diamond_suite", 3, "default"),
    ("depthwise_suite", 3, "manifest-band"),
    ("lut_domain_suite", 2, "lut"),
    ("native_input_suite", 2, "manifest-band"),
)


def _read_manifest(suite):
    path = RESEARCH / suite / "manifest.json"
    entries = json.loads(path.read_text()) if path.exists() else []
    return {entry["index"]: entry for entry in entries
            if isinstance(entry, dict) and "index" in entry}


def _sample_flags(suite, index, how):
    if how == "default":
        return {}
    if how == "lut":
        return dict(input_scale=1.0, input_zero_point=128)
    entry = _read_manifest(suite)[index]
    return dict(input_scale=entry["input_scale"], input_zero_point=entry["input_zero_point"])


def _quantize(params):
    if isinstance(params, Quantization):
        return params
    values = dict(params)
    for key in ("weights", "weight_zero_points", "weight_scales", "biases", "channel_multipliers"):
        values[key] = np.array(values[key])
    return Quantization(**values)


def _cases(folder, index, shape):
    data = np.frombuffer((folder / f"input{index:03}.u8").read_bytes(), np.uint8)
    return data.reshape(-1, *shape)


def _replay_chain_multi(model, meta, folder, index):
    shape = meta["shape_nhwc"]
    cases = _cases(folder, index, shape[1:])
    return b"".join(b"".join(layer.tobytes()
                             for layer in chain_n_reference_layers(case, meta["quantizations"]))
                    for case in cases)


def _replay_walk_chain(model, meta, folder, index):
    spec = parse_chain(model.graph)
    quantizations = load_quantizations(meta)
    batch, channels, height, width = spec["input_shape"]
    cases = _cases(folder, index, (height, width, channels))
    return np.stack([chain_walk_reference(case, quantizations, spec["ops"])
                     for case in cases]).tobytes()


def _replay_walk_join(model, meta, folder, index):
    plan = parse_join_walk(model.graph)
    quantizations = {name: (_quantize(value) if value else None)
                     for name, value in meta["quantizations"].items()}
    cases = _cases(folder, index, (8, 8, 3))
    return np.stack([join_walk_reference(case, quantizations, plan,
                                         join_zero_point=meta["join_zero_point"])
                     for case in cases]).tobytes()


def _replay_depthwise_join(model, meta, folder, index):
    cases = _cases(folder, index, (8, 8, 3))
    return np.stack([depthwise_join_reference(case, meta["stem_quantization"],
                                              meta["head_quantization"][0],
                                              meta["depthwise_quantization"], meta["join"])
                     for case in cases]).tobytes()


def _replay_diamond(model, meta, folder, index):
    cases = _cases(folder, index, (8, 8, 3))
    return np.stack([diamond_reference(case, meta["stem_quantization"], meta["head_quantization"],
                                       meta["join"]) for case in cases]).tobytes()


# (suite, model index, compile kwargs, the builder's reference call)
REFERENCE_REPLAYS = (
    ("chain_multi_suite", 0, dict(expose_intermediates=True), _replay_chain_multi),
    ("walk_chain_suite", 0, {}, _replay_walk_chain),
    ("walk_join_suite", 0, {}, _replay_walk_join),
    ("depthwise_join_suite", 0, {}, _replay_depthwise_join),
    ("diamond_suite", 0, {}, _replay_diamond),
)


class SuiteReplayTests(unittest.TestCase):
    def test_campaign_and_extra_sample_compile_to_the_published_bytes(self):
        self.assertEqual(len(CAMPAIGN), 14)
        self.assertEqual(len(EXPECTED_DRIFT), 12)
        same = drift = 0
        for suite in CAMPAIGN:
            folder = RESEARCH / suite
            for path in sorted(folder.glob("model*.onnx"))[:CAMPAIGN_SAMPLE]:
                with self.subTest(model=f"{suite}/{path.name}"):
                    same, drift = self.check(path, {}, same, drift)
        for suite, count, how in EXTRA_SAMPLES:
            folder = RESEARCH / suite
            for path in sorted(folder.glob("model*.onnx"))[:count]:
                index = int(path.stem[5:])
                with self.subTest(model=f"{suite}/{path.name}"):
                    same, drift = self.check(path, _sample_flags(suite, index, how), same, drift)
        self.assertGreaterEqual(same, 50)
        self.assertEqual(same + drift, CAMPAIGN_SAMPLE * len(CAMPAIGN)
                         + sum(count for _, count, _ in EXTRA_SAMPLES))

    def check(self, path, kwargs, same, drift):
        artifact = path.with_suffix(".bin")
        if not artifact.exists():
            return same, drift
        key = f"{path.parent.name}/{path.name}"
        binary, _ = compile_sequence(path, **kwargs)
        if bytes(binary) == artifact.read_bytes():
            return same + 1, drift
        self.assertIn(key, EXPECTED_DRIFT, "unexpected drift from the published container")
        return same, drift + 1

    def test_recorded_cases_replay_through_the_builder_reference(self):
        for suite, index, kwargs, reference in REFERENCE_REPLAYS:
            with self.subTest(suite=suite, model=f"{index:03}"):
                folder = RESEARCH / suite
                model_path = folder / f"model{index:03}.onnx"
                binary, meta = compile_sequence(model_path, **kwargs)
                self.assertEqual(binary, (folder / f"model{index:03}.bin").read_bytes())
                model = onnx.load(model_path)
                expected = (folder / f"expected{index:03}.i8").read_bytes()
                actual = reference(model, meta, folder, index)
                self.assertEqual(len(actual), len(expected))
                self.assertEqual(actual, expected)

    def test_sample_stays_within_the_time_budget(self):
        # The module is meant to be cheap; a runaway sample should fail loudly here
        # rather than silently slowing the whole unit suite.
        started = time.perf_counter()
        for suite, index, kwargs, reference in REFERENCE_REPLAYS:
            folder = RESEARCH / suite
            model_path = folder / f"model{index:03}.onnx"
            _, meta = compile_sequence(model_path, **kwargs)
            reference(onnx.load(model_path), meta, folder, index)
        self.assertLess(time.perf_counter() - started, 10.0)


if __name__ == "__main__":
    unittest.main()
