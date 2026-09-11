"""MIT. The DAG emitters build their containers through `open_rknpu.compose`.

PIPELINING_PLAN P1 / COMPLETION_PLAN P1 left `join_dag`, `join_chain` and `diamond`
assembling containers by hand. `compile_diamond` and `compile_join_dag` (which also
serves the pooled join DAG) now declare stages and call the composer; this module
pins the strongest acceptance the plan asks for: the composed container is
**byte-identical** to the retained board-verified `.bin` of every model in the
affected suites, and the composer's declared binding view matches each container.
`compile_join_chain` (including its runtime per-channel scale and runtime residual
tails) and `compile_two_head` are covered the same way; the ten models whose retained
`.bin` already differed from a fresh compile before the port are pinned by the
pre-port container hashes in `research/composer_port_reference.json`.
"""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest


from open_rknpu.compose import check_declared_bindings, derive_bindings
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence, relink_for_batched

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "research" / "composer_port_reference.json").read_text())
COMPOSED_SUITES = ("diamond_suite", "diamond_tail_suite", "join_dag_suite", "pooled_dag_suite")


def compile_path(path, **kwargs):
    with tempfile.TemporaryDirectory() as folder:
        copy = Path(folder) / "model.onnx"
        copy.write_bytes(path.read_bytes())
        return compile_sequence(copy, **kwargs)


class ComposedEmitterTests(unittest.TestCase):
    def test_composed_dags_are_byte_identical_to_the_board_evidence(self):
        checked = 0
        for suite in COMPOSED_SUITES:
            for model in sorted((ROOT / "research" / suite).glob("model*.onnx")):
                retained = model.with_suffix(".bin")
                if not retained.is_file():
                    continue
                with self.subTest(container=f"{suite}/{model.name}"):
                    binary, _ = compile_path(model)
                    self.assertEqual(binary, retained.read_bytes())
                    checked += 1
        self.assertGreaterEqual(checked, 48)

    def test_declared_bindings_match_the_composed_containers(self):
        # Both a join DAG with a terminal pool and a diamond tail exercise every
        # family the composer now emits for these profiles.
        for suite, index in (("diamond_tail_suite", "000"), ("pooled_dag_suite", "000"),
                             ("join_dag_suite", "000"), ("diamond_suite", "000")):
            path = ROOT / "research" / suite / f"model{index}.onnx"
            with self.subTest(suite=suite):
                binary, meta = compile_path(path)
                info = decode_sequence(binary)
                derived = derive_bindings(binary, info)
                self.assertEqual(len(derived), info["task_count"])
                self.assertTrue(check_declared_bindings(meta["declared_bindings"], derived,
                                                        meta["tensor_offsets"],
                                                        meta["stage_schedule"]))

    def test_batched_compose_matches_the_submission_post_pass(self):
        path = ROOT / "research" / "diamond_tail_suite" / "model000.onnx"
        serial, _ = compile_path(path)
        batched, meta = compile_path(path, submission="batched")
        self.assertEqual(batched, relink_for_batched(serial))
        self.assertFalse(decode_sequence(batched)["serial"])
        self.assertEqual(meta["engine_runs"], [decode_sequence(batched)["task_count"]])

    def test_join_chain_port_reproduces_every_pre_port_container(self):
        # The join-chain emitter (fan-out heads, join tree, tail layers, runtime
        # per-channel scale, runtime residual) was ported without changing one byte:
        # every container hashes to the value captured from the hand-assembled
        # emitter before the port.
        containers = REFERENCE["containers"]
        self.assertGreaterEqual(len(containers), 100)
        for name, digest in sorted(containers.items()):
            with self.subTest(container=name):
                binary, _ = compile_path(ROOT / "research" / name)
                self.assertEqual(hashlib.sha256(binary).hexdigest(), digest)

    def test_join_chain_suites_still_match_their_retained_evidence(self):
        # Where the retained board artefact equals a fresh compile, pin the bytes as
        # well (the ten drifted models are covered by the hash reference above).
        for suite in ("join_chain_suite", "depthwise_chain_suite"):
            for model in sorted((ROOT / "research" / suite).glob("model*.onnx")):
                retained = model.with_suffix(".bin")
                if not retained.is_file():
                    continue
                with self.subTest(container=f"{suite}/{model.name}"):
                    binary, _ = compile_path(model)
                    self.assertEqual(binary, retained.read_bytes())


if __name__ == "__main__":
    unittest.main()
