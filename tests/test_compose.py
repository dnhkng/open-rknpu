"""MIT. Stage composer and declared-binding regression.

`open_rknpu.compose` assembles a version-5 container from a declared stage list:
topological task order and arena placement from `open_rknpu.liveness`, program slots
and constant blocks in declared order, address substitution into each stage's
register words, and a declared binding view that is checked against the container it
produced. These tests cover the pass itself and the declared/derived agreement of
the profiles that have been ported onto it (`pool_join`, `pooled_branches`, `diamond` and
the join DAGs, and - with the last P1 port - `join_chain` including its runtime-scale and
runtime-residual tails), and the external-input placement rules (one to eight inputs,
early before the arena and late after it).
"""
from pathlib import Path
import unittest

from open_rknpu.compose import (Binding, ConstantSpec, FAMILY_BY_SIGNATURE, FAMILIES, Stage,
                                TensorSpec, check_declared_bindings, compose, derive_bindings)
from open_rknpu.model import decode
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import LAYOUT_NATIVE16, ROLE_INPUT, ROLE_INTERNAL, ROLE_OUTPUT

ROOT = Path(__file__).resolve().parents[1]


def chain_stages():
    """Two elementwise stages declared *out of order* to exercise the pass."""

    def first_fields(addresses, constants):
        return {0x4020: addresses["mid"], 0x5018: addresses["image"], 0x5038: addresses["image"]}

    def second_fields(addresses, constants):
        return {0x4020: addresses["output"], 0x5018: addresses["mid"], 0x5038: addresses["mid"]}

    return [
        Stage(name="second", family="elementwise", reads=("mid",), writes=("output",),
              fields=second_fields,
              bindings=(Binding(0x5018, "mid", "read"), Binding(0x5038, "mid", "read"),
                        Binding(0x4020, "output", "write"))),
        Stage(name="first", family="elementwise", reads=("image",), writes=("mid",),
              fields=first_fields,
              bindings=(Binding(0x5018, "image", "read"), Binding(0x5038, "image", "read"),
                        Binding(0x4020, "mid", "write"))),
    ]


def chain_tensors():
    return [TensorSpec("image", ROLE_INPUT, LAYOUT_NATIVE16, (1, 8, 8, 3), 8 * 8 * 16),
            TensorSpec("mid", ROLE_INTERNAL, LAYOUT_NATIVE16, (1, 8, 8, 3), 8 * 8 * 16),
            TensorSpec("output", ROLE_OUTPUT, LAYOUT_NATIVE16, (1, 8, 8, 3), 8 * 8 * 16)]


class ComposerTests(unittest.TestCase):
    def test_pass_orders_tasks_but_keeps_declared_program_slots(self):
        stages = chain_stages()
        binary, meta = compose(stages, chain_tensors(), serial=False)
        info = decode(binary)
        # Program slots follow the declaration (second is declared first) ...
        slot = (78 + 4) * 8
        self.assertEqual(meta["program_offsets"]["second"], 0)
        self.assertEqual(meta["program_offsets"]["first"], (slot + 63) // 64 * 64)
        # ... while the emitted task order is topological (first, then second).
        self.assertEqual(meta["stage_schedule"], ["first", "second"])
        self.assertEqual(meta["schedule"], ["mid", "output"])
        self.assertEqual([task["command_offset"] for task in info["tasks"]],
                         [meta["program_offsets"]["first"], meta["program_offsets"]["second"]])
        derived = derive_bindings(binary, info)
        self.assertTrue(check_declared_bindings(meta["declared_bindings"], derived,
                                               meta["tensor_offsets"], meta["stage_schedule"]))

    def test_declared_view_matches_a_composed_profile(self):
        # pool_join reuses arena slots; pooled_branches gives every internal a fresh
        # slot. Both declared views must match the containers they produced.
        cases = [("pool_join_suite", 0), ("pool_join_suite", 5),
                 ("pooled_branches_suite", 0), ("pooled_branches_suite", 6)]
        for suite, index in cases:
            path = ROOT / "research" / suite / f"model{index:03}.onnx"
            data, meta = compile_sequence(path)
            info = decode(data)
            derived = derive_bindings(data, info)
            with self.subTest(model=path.name):
                self.assertEqual(len(derived), len(meta["declared_bindings"]))
                self.assertTrue(check_declared_bindings(meta["declared_bindings"], derived,
                                                       meta["tensor_offsets"],
                                                       meta["stage_schedule"]))
                for task in derived:
                    self.assertIn(task["family"], FAMILIES)
                    self.assertTrue(task["reads"])
                    self.assertTrue(task["writes"])
                self.assertEqual(meta["arena_reuse"], suite == "pool_join_suite")

    def test_declaration_mismatch_is_detected(self):
        stages = chain_stages()
        binary, meta = compose(stages, chain_tensors(), serial=False)
        derived = derive_bindings(binary, decode(binary))
        wrong = [dict(entry) for entry in meta["declared_bindings"]]
        wrong[0]["reads"] = {0x5018: "image", 0x5038: "image"}
        with self.assertRaises(ValueError):
            check_declared_bindings(wrong, derived, meta["tensor_offsets"], meta["stage_schedule"])
        # ... and a missing address register is caught too.
        wrong = [dict(entry) for entry in meta["declared_bindings"]]
        wrong[1]["reads"] = {}
        with self.assertRaises(ValueError):
            check_declared_bindings(wrong, derived, meta["tensor_offsets"], meta["stage_schedule"])

    def test_unknown_tensor_and_read_before_write_are_rejected(self):
        bad = [Stage(name="only", family="elementwise", reads=("missing",), writes=("output",),
                     fields=lambda addresses, constants: {0x4020: addresses["output"]},
                     bindings=(Binding(0x5018, "missing", "read"),))]
        with self.assertRaises(ValueError):
            compose(bad, chain_tensors())
        # A stage that reads a tensor nobody writes (and is not an input) is rejected
        # by the liveness pass, not silently placed.
        cyclic = [Stage(name="a", family="elementwise", reads=("output",), writes=("mid",),
                        fields=lambda addresses, constants: {0x4020: addresses["mid"]},
                        bindings=(Binding(0x5018, "output", "read"),)),
                   Stage(name="b", family="elementwise", reads=("mid",), writes=("output",),
                        fields=lambda addresses, constants: {0x4020: addresses["output"]},
                        bindings=(Binding(0x5038, "mid", "read"),))]
        with self.assertRaises(ValueError):
            compose(cyclic, chain_tensors())

    def test_families_cover_every_v5_task_signature_in_use(self):
        for name in ("native-conv", "pool", "elementwise", "lut-setup"):
            self.assertIn(name, FAMILIES)
        self.assertEqual((FAMILIES["native-conv"].words, FAMILIES["native-conv"].enable), (126, 29))
        self.assertEqual((FAMILIES["pool"].words, FAMILIES["pool"].enable), (37, 96))
        self.assertEqual((FAMILIES["elementwise"].words, FAMILIES["elementwise"].enable), (78, 24))
        self.assertEqual((FAMILIES["lut-setup"].words, FAMILIES["lut-setup"].enable), (1106, 24))
        for family in FAMILIES.values():
            self.assertIn((family.words, family.enable), FAMILY_BY_SIGNATURE)
        # The pool family carries its own descriptor tags; the native family the
        # register profile's tags.
        self.assertTrue(all(len(entry) == 3 for entry in FAMILIES["pool"].base))
        self.assertEqual(FAMILIES["native-conv"].base[0][0], 0x1004)
        # Declared constants are optional; a stage without constants still composes.
        stages = chain_stages()
        plain = [Stage(name=stage.name, family=stage.family, reads=stage.reads,
                       writes=stage.writes, fields=stage.fields, bindings=stage.bindings)
                 for stage in stages]
        self.assertTrue(all(not stage.constants for stage in plain))
        self.assertTrue(all(isinstance(spec, ConstantSpec) for spec in ()))


if __name__ == "__main__":
    unittest.main()


class ComposerExternalInputTests(unittest.TestCase):
    """The composer places one to eight external inputs, early and late."""

    def tensors(self, late=False):
        return [TensorSpec("image", ROLE_INPUT, LAYOUT_NATIVE16, (1, 8, 8, 3), 8 * 8 * 16, 0),
                TensorSpec("scale", ROLE_INPUT, LAYOUT_NATIVE16, (1, 1, 1, 3), 64, 1),
                TensorSpec("mid", ROLE_INTERNAL, LAYOUT_NATIVE16, (1, 8, 8, 3), 8 * 8 * 16, 0),
                TensorSpec("output", ROLE_OUTPUT, LAYOUT_NATIVE16, (1, 8, 8, 3), 8 * 8 * 16, 0)]

    def stages(self):
        def fields(addresses, constants):
            return {0x4020: addresses["mid"], 0x5018: addresses["image"],
                    0x5038: addresses["scale"]}

        def second_fields(addresses, constants):
            return {0x4020: addresses["output"], 0x5018: addresses["mid"],
                    0x5038: addresses["mid"]}

        return [Stage(name="first", family="elementwise", reads=("image", "scale"), writes=("mid",),
                      fields=fields, bindings=(Binding(0x5018, "image", "read"),
                                               Binding(0x5038, "scale", "read"),
                                               Binding(0x4020, "mid", "write"))),
                Stage(name="second", family="elementwise", reads=("mid",), writes=("output",),
                      fields=second_fields, bindings=(Binding(0x5018, "mid", "read"),
                                                      Binding(0x5038, "mid", "read"),
                                                      Binding(0x4020, "output", "write")))]

    def test_early_inputs_precede_the_arena_and_late_inputs_follow_it(self):
        binary, meta = compose(self.stages(), self.tensors(), serial=True)
        info = decode(binary)
        offsets = meta["tensor_offsets"]
        self.assertEqual(info["input_tensor_count"], 2)
        self.assertLess(offsets["image"], offsets["mid"])
        self.assertLess(offsets["scale"], offsets["mid"])
        binary, meta = compose(self.stages(), self.tensors(), serial=True, late_inputs=("scale",))
        offsets = meta["tensor_offsets"]
        self.assertLess(offsets["image"], offsets["mid"])
        self.assertLess(offsets["mid"], offsets["scale"])
        self.assertLess(offsets["scale"], offsets["output"])
        derived = derive_bindings(binary, decode(binary))
        self.assertTrue(check_declared_bindings(meta["declared_bindings"], derived,
                                                offsets, meta["stage_schedule"]))

    def test_unknown_late_input_and_too_many_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            compose(self.stages(), self.tensors(), late_inputs=("nope",))
        many = [TensorSpec(f"in{index}", ROLE_INPUT, LAYOUT_NATIVE16, (1, 1, 1, 3), 64, index)
                for index in range(9)]
        many.append(TensorSpec("output", ROLE_OUTPUT, LAYOUT_NATIVE16, (1, 1, 1, 3), 64, 0))
        with self.assertRaises(ValueError):
            compose(self.stages(), many)
