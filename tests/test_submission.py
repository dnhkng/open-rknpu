"""MIT. Batched-submission rule and job-shape evidence (docs/plans/pipelining-plan.md S1/S2/S8).

Measured on RV1103: a job submitted in one ioctl runs its whole task list only when

* every task links to the next program (register `0x10`, payload-relative link value)
  and the last task is terminal - an unlinked (terminal) list runs its first task and
  times out;
* every transition carries the control word measured for its engine hand-off -
  `0x40` CNA->CNA and DPU->DPU, `0x14` CNA->DPU (the value the vendor compiler writes
  in front of a pool task), terminal `0x28`. The old "one engine per job" rule was a
  consequence of writing `0x40` everywhere; with the right control a mixed-engine
  DAG runs in one job, which `pool_join_suite`-style graphs now use;
* no hand-off from a DPU task back to a CNA task passed in the sweep, so a graph whose
  scheduled order needs one is refused instead of emitted;
* depth is bounded only by the loader's 64-task table.

Batched submission is nevertheless *slower* than serial for the small per-task costs
of an 8x8 profile (crossover near eight tasks per same-engine run), so serial remains
the default; `open_rknpu.compose` emits the linked form on request.
"""
from pathlib import Path
import json
import struct
import sys
import unittest

from open_rknpu.compose import (Binding, ENGINE_CLASS, Stage, TensorSpec, amount_control,
                                batched_layout, compose)
from open_rknpu.sequence import (LAYOUT_NATIVE16, LAYOUT_PACKED_U8, ROLE_INPUT,
                                 ROLE_INTERNAL, ROLE_OUTPUT, decode_sequence as decode)
from open_rknpu.scheduler import batched_supported, compile_sequence

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "research" / "submission_probe"
GROUP = ROOT / "research" / "group_probe"
FIELD = ROOT / "research" / "job_field_probe"


def committed(suite, index=0):
    return (ROOT / "research" / suite / f"model{index:03}.bin").read_bytes()


def controls(data):
    """The `(link, control)` tail words of every task in a container."""
    info = decode(data)
    if info["format_version"] == 5:
        base = 112 + 16 * info["task_count"] + 64 * info.get("tensor_count", 0)
    else:
        base = 96 + 16 * info["task_count"] + 40 * info.get("constant_count", 0)
    out = []
    for task in info["tasks"]:
        offset = base + task["command_offset"] + task["register_count"] * 8
        word0, word1 = struct.unpack_from("<QQ", data, offset)
        out.append(((word0 >> 16) & 0xFFFFFFFF, (word1 >> 16) & 0xFFFFFFFF))
    return out


class BatchedRuleTests(unittest.TestCase):
    def test_engine_classes_follow_the_task_enable_masks(self):
        self.assertEqual(ENGINE_CLASS[29], "CNA")
        self.assertEqual(ENGINE_CLASS[96], "DPU")
        self.assertEqual(ENGINE_CLASS[24], "DPU")

    def test_tail_control_is_the_successors_fetch_amount(self):
        # PC_DATA_AMOUNT = (words + 4 + 2 - 1)/2 - 1 on this SoC; the vendor captures
        # write exactly that value for the program the tail links to.
        self.assertEqual(amount_control(126), 0x40)     # Conv program
        self.assertEqual(amount_control(37), 0x14)      # pool program
        self.assertEqual(amount_control(78), 0x28)      # elementwise program
        self.assertEqual(amount_control(1106), 0x22A)   # LUT setup program

    def test_linked_single_engine_container_is_accepted(self):
        data = committed("chain_output_quantization_suite")
        info = decode(data)
        self.assertEqual(info["task_count"], 2)
        self.assertIsNone(batched_layout(data, info))
        self.assertTrue(batched_supported(data, info))

    def test_unlinked_twins_degrade_to_one_run_per_task(self):
        from open_rknpu.compose import engine_runs
        # The twins in research/submission_probe are flag flips of serial containers, so
        # their tails are terminal. A job submitted as one ioctl would run only the first
        # task (the rule retained in `group_probe/job_results.json`); since S10 the
        # runtime submits one job per maximal linked run, so such a container is simply
        # one run per task - valid, and byte-identical to its serial use.
        for name in ("mnist_pool_suite-000-batched.bin",
                     "runtime_scale_suite-000-batched.bin",
                     "pool_join_suite-000-batched.bin"):
            path = PROBE / name
            with self.subTest(twin=name):
                self.assertTrue(path.is_file(), name)
                data = path.read_bytes()
                info = decode(data)
                self.assertIsNone(batched_layout(data, info))
                self.assertEqual(engine_runs(data, info),
                                 [(position, 1) for position in range(info["task_count"])])
        # The linked twin is a single job.
        linked = decode((PROBE / "native_chain_suite-000-batched-linked.bin").read_bytes())
        self.assertIsNotNone(batched_layout(
            (PROBE / "native_chain_suite-000-batched-linked.bin").read_bytes(), linked))

    def test_linked_mixed_twin_uses_the_measured_hand_off(self):
        # research/job_field_probe passed this twin on the board: conv (CNA) then pool
        # (DPU) in one job, linked with control 0x14.
        data = (FIELD / "twins" / "pair_cna_dpu_0x14.bin").read_bytes()
        info = decode(data)
        self.assertIsNone(batched_layout(data, info))
        self.assertEqual(controls(data)[0][1], 0x14)
        # The same list with the single-engine control 0x40 is refused.
        offset = (96 + 16 * info["task_count"] + 40 * info.get("constant_count", 0)
                  + info["tasks"][0]["command_offset"]
                  + info["tasks"][0]["register_count"] * 8 + 8)
        wrong = bytearray(data)
        struct.pack_into("<Q", wrong, offset, 0x101 << 48 | 0x40 << 16 | 0x14)
        self.assertIn("fetch amount 0x14", batched_layout(bytes(wrong), info))

    def test_job_field_evidence_is_retained(self):
        summary = json.loads((FIELD / "summary.json").read_text())
        self.assertEqual(summary["cna_dpu"], 0x14)
        self.assertEqual(summary["dpu_dpu"], 0x40)
        # The raw sweep record: the first passing candidate for each transition it
        # probed. Its DPU->CNA conclusion is withdrawn (the run's prefix used 0x14 at a
        # Conv->elementwise step, which is not that transition's amount) - the amount
        # rule in `test_tail_control_is_the_successors_fetch_amount` supersedes it.
        self.assertIsNone(summary["dpu_cna"])
        self.assertIn("WITHDRAWN", (FIELD / "README.md").read_text())
        self.assertIn("0x14", summary["candidate_order"])
        records = json.loads((FIELD / "board_results.json").read_text())
        passing = [record["case"] for record in records if record["passed"]]
        self.assertEqual(passing, ["pair-serial-committed", "pair-batched-cna-dpu-0x14",
                                   "join6-serial-committed", "join6-batched-dpu-dpu-0x40",
                                   "mixedhead7-serial-committed"])
        core = json.loads((FIELD / "core_fields.json").read_text())
        self.assertEqual(len(core), 5)
        for record in core:
            self.assertEqual(record["mismatches"], 0)
            self.assertFalse(record["changed"])

    def _dpu_then_cna_stages(self):
        """A declaration whose scheduled order is elementwise (DPU) then Conv (CNA)."""
        add = Stage(name="add", family="elementwise", reads=("input0", "input0"),
                    writes=("mid",),
                    fields=lambda a, c: {0x5018: a["input0"], 0x5038: a["input0"],
                                         0x4020: a["mid"]},
                    bindings=(Binding(0x5018, "input0", "read"),
                              Binding(0x5038, "input0", "read"),
                              Binding(0x4020, "mid", "write")))
        conv = Stage(name="conv", family="native-conv", reads=("mid",), writes=("out",),
                     fields=lambda a, c: {0x1070: a["mid"], 0x4020: a["out"]},
                     bindings=(Binding(0x1070, "mid", "read"),
                               Binding(0x4020, "out", "write")))
        tensors = [TensorSpec("input0", ROLE_INPUT, LAYOUT_PACKED_U8, (1, 8, 8, 3), 384),
                   TensorSpec("mid", ROLE_INTERNAL, LAYOUT_NATIVE16, (1, 8, 8, 16), 1024),
                   TensorSpec("out", ROLE_OUTPUT, LAYOUT_NATIVE16, (1, 8, 8, 16), 1024)]
        return [add, conv], tensors

    def test_composer_links_mixed_engine_runs_into_one_job(self):
        # A three-CNA then three-DPU graph is one valid job now.
        data, meta = compile_sequence(ROOT / "research" / "pool_join_suite" / "model000.onnx",
                                      submission="batched")
        info = decode(data)
        self.assertFalse(info["serial"])
        self.assertEqual(meta["submission"], "batched")
        self.assertIsNone(batched_layout(data, info))
        # Each tail carries the successor's fetch amount (S8/S10).
        self.assertEqual([control for _, control in controls(data)][:-1],
                         [amount_control(task["register_count"])
                          for task in info["tasks"][1:]])
        # The output is the serial container's output with the linked tails and the flag.
        serial = committed("pool_join_suite")
        self.assertEqual(len(serial), len(data))
        for position, task in enumerate(info["tasks"]):
            base = 112 + 16 * info["task_count"] + 64 * info["tensor_count"]
            start = base + task["command_offset"]
            tail = start + task["register_count"] * 8
            self.assertEqual(data[start:tail], serial[start:tail])
        # A DPU -> CNA order links like any other: the control is the successor's fetch
        # amount, so the pair is one job of two tasks (S10).
        stages, tensors = self._dpu_then_cna_stages()
        binary, meta = compose(stages, tensors, serial=False)
        self.assertEqual(meta["engine_runs"], [2])
        self.assertIsNone(batched_layout(binary, decode(binary)))
        self.assertEqual(controls(binary)[0][1], amount_control(126))
        # The DAG emitters threaded in S10 all link their runs too.
        for suite in ("join_dag_suite", "pooled_dag_suite", "diamond_tail_suite",
                      "depthwise_join_suite"):
            data, meta = compile_sequence(ROOT / "research" / suite / "model000.onnx",
                                          submission="batched")
            with self.subTest(suite=suite):
                self.assertEqual(meta["engine_runs"], [len(decode(data)["tasks"])])
                self.assertIsNone(batched_layout(data, decode(data)))
        # A profile whose emitter never learned about the mode still works: the
        # post-pass relinks the container (`sequence.relink_for_batched`).
        data, meta = compile_sequence(ROOT / "research" / "add_suite" / "model000.onnx",
                                      submission="batched")
        info = decode(data)
        self.assertEqual(meta["engine_runs"], [info["task_count"]])
        self.assertIsNone(batched_layout(data, info))
        # The N-layer chain emitter does act on it: linked, single-engine, batched.
        data, meta = compile_sequence(ROOT / "research" / "chain_multi_suite" / "model000.onnx",
                                      submission="batched")
        info = decode(data)
        self.assertFalse(info["serial"])
        self.assertEqual(meta["submission"], "batched")
        self.assertIsNone(batched_layout(data, info))
        # The composer's own single-engine chain emits a linked batched container.
        sys.path.insert(0, str(ROOT / "tests"))
        from test_compose import chain_stages, chain_tensors
        binary, meta = compose(chain_stages(), chain_tensors(), serial=False)
        self.assertEqual(meta["submission"], "batched")
        info = decode(binary)
        self.assertIsNone(batched_layout(binary, info))
        links = [link for link, _ in controls(binary)]
        self.assertEqual(links[0], info["tasks"][1]["command_offset"])
        self.assertEqual(links[-1], 0)

    def test_mixed_engine_one_job_evidence(self):
        probe = ROOT / "research" / "mixed_batched_probe"
        manifest = json.loads((probe / "manifest.json").read_text())
        self.assertGreaterEqual(len(manifest), 4)
        for entry in manifest:
            container = (probe / entry["file"]).read_bytes()
            data, _ = compile_sequence(ROOT / "research" / entry["suite"]
                                       / f"model{entry['index']:03}.onnx",
                                       submission="batched")
            info = decode(container)
            with self.subTest(suite=entry["suite"], model=entry["index"]):
                self.assertEqual(container, data)      # retained == current emitter
                self.assertFalse(info["serial"])
                self.assertIsNone(batched_layout(container, info))
                self.assertEqual(info["task_count"], entry["tasks"])
                # One job, more than one engine.
                self.assertGreater(len({ENGINE_CLASS.get(task["enable"])
                                        for task in info["tasks"]}), 1)
        results = json.loads((probe / "board_results.json").read_text())
        self.assertTrue(results)
        for record in results:
            with self.subTest(suite=record["suite"], model=record["index"],
                              mode=record["mode"]):
                self.assertTrue(record["passed"])
                self.assertEqual(record["mismatches"], 0)
                self.assertGreaterEqual(record["cases"], 4)
        # At this depth the one-job form is the faster mode on every container.
        for suite, index in {(record["suite"], record["index"]) for record in results}:
            serial = next(r for r in results if (r["suite"], r["index"], r["mode"]) ==
                          (suite, index, "serial"))
            batched = next(r for r in results if (r["suite"], r["index"], r["mode"]) ==
                           (suite, index, "batched"))
            self.assertLess(batched["min_us"], serial["min_us"])

    def test_every_transition_links_with_the_successors_amount(self):
        from open_rknpu.compose import engine_runs
        # A mixed head: Conv x4, elementwise x2, Conv. Each tail carries the *next*
        # program's fetch amount, so the whole list is one job even though the
        # transitions cross engines (S8/S10; measured in research/grouped_probe/).
        data, meta = compile_sequence(ROOT / "research" / "mixed_head_suite" / "model001.onnx",
                                      submission="batched")
        info = decode(data)
        self.assertEqual(meta["engine_runs"], [7])
        self.assertEqual(engine_runs(data, info), [(0, 7)])
        self.assertIsNone(batched_layout(data, info))
        expected = [amount_control(task["register_count"]) for task in info["tasks"][1:]]
        self.assertEqual([control for _, control in controls(data)][:-1], expected)
        self.assertEqual(expected[:4], [0x40, 0x40, 0x40, 0x28])
        self.assertEqual(expected[-1], 0x40)                        # elementwise -> Conv
        self.assertEqual(controls(data)[-1][1], 0x28)               # terminal sentinel
        # The join-chain emitter (graph.py) links the same way without the composer.
        data, meta = compile_sequence(ROOT / "research" / "join_chain_suite" / "model005.onnx",
                                      submission="batched")
        info = decode(data)
        self.assertEqual(meta["engine_runs"], [info["task_count"]])
        self.assertEqual(engine_runs(data, info), [(0, info["task_count"])])
        self.assertIsNone(batched_layout(data, info))
        # A profile whose emitter ignores the request is relinked by the post-pass.
        data, meta = compile_sequence(ROOT / "research" / "add_suite" / "model000.onnx",
                                      submission="batched")
        self.assertEqual(meta["engine_runs"], [decode(data)["task_count"]])
        self.assertIsNone(batched_layout(data, decode(data)))

    def test_grouped_probe_evidence(self):
        probe = ROOT / "research" / "grouped_probe"
        manifest = json.loads((probe / "manifest.json").read_text())
        self.assertEqual(len(manifest), 8)
        for entry in manifest:
            with self.subTest(suite=entry["suite"], model=entry["index"]):
                self.assertEqual(entry["runs"], [[0, entry["tasks"]]])
                self.assertEqual(entry["meta_runs"], [entry["tasks"]])
                data, meta = compile_sequence(
                    ROOT / "research" / entry["suite"] / f"model{entry['index']:03}.onnx",
                    submission="batched")
                self.assertEqual(data, (probe / entry["file"]).read_bytes())
                self.assertEqual(meta["engine_runs"], [entry["tasks"]])
        results = json.loads((probe / "board_results.json").read_text())
        discovery = [r for r in results if r["kind"] == "discovery"]
        published = [r for r in results if r["kind"] == "published"]
        self.assertEqual(len(discovery), 4)
        self.assertEqual(len(published), len(manifest) * 2)
        # The amount rule: a Conv->elementwise transition with the successor's fetch
        # amount passes; with 0x14 it is the retained counter-example.
        by_case = {record["case"]: record for record in discovery}
        self.assertTrue(by_case["conv->elementwise amount 0x28"]["passed"])
        self.assertFalse(by_case["conv->elementwise 0x14 (counter-example)"]["passed"])
        self.assertTrue(by_case["elementwise->elementwise amount 0x28"]["passed"])
        self.assertTrue(by_case["conv->pool amount 0x14"]["passed"])
        for record in published:
            with self.subTest(suite=record["suite"], mode=record["mode"]):
                self.assertTrue(record["passed"])
                self.assertEqual(record["mismatches"], 0)
                self.assertEqual(record["engine_runs"],
                                 1 if record["mode"] == "one-job"
                                 else record["tasks"])
        # Every published container is one job and beats its serial minimum.
        for entry in manifest:
            key = (entry["suite"], entry["index"])
            serial = next(r for r in published if (r["suite"], r["index"], r["mode"]) ==
                          key + ("serial",))
            onejob = next(r for r in published if (r["suite"], r["index"], r["mode"]) ==
                          key + ("one-job",))
            self.assertEqual(onejob["engine_runs"], 1)
            self.assertLessEqual(onejob["min_us"], serial["min_us"])


    def test_every_compilable_profile_can_be_submitted_as_one_job(self):
        from open_rknpu.sequence import relink_for_batched
        from open_rknpu.sequence import decode_sequence
        # Emitters that build whole containers by hand (or one task at a time) never
        # learned about the submission mode; `compile_sequence` relinks whatever they
        # produced, so the mode is a property of the container format, not of each
        # emitter (docs/plans/pipelining-plan.md S10 residual).
        for suite, index in (("add_suite", 0), ("application_suite", 0),
                             ("mul_batch_suite", 2), ("depthwise_chain_suite", 0),
                             ("transpose_k5_suite", 0)):
            source = ROOT / "research" / suite / f"model{index:03}.onnx"
            with self.subTest(suite=suite):
                serial, _ = compile_sequence(source)
                batched, meta = compile_sequence(source, submission="batched")
                self.assertNotEqual(batched, serial)
                self.assertEqual(relink_for_batched(serial), batched)
                self.assertEqual(relink_for_batched(batched), batched)   # idempotent
                info = decode_sequence(batched)
                self.assertEqual(meta["engine_runs"], [info["task_count"]])
                self.assertIsNone(batched_layout(batched, info))
        # The submission mode is never the reason a profile is refused.
        for suite in ("lut_domain_suite", "native_extreme_kernels_suite"):
            source = ROOT / "research" / suite / "model000.onnx"
            with self.subTest(refused=suite):
                with self.assertRaises(ValueError) as plain:
                    compile_sequence(source)
                with self.assertRaises(ValueError) as batched:
                    compile_sequence(source, submission="batched")
                self.assertEqual(str(plain.exception), str(batched.exception))

    def test_serial_emission_is_unchanged(self):
        for suite in ("pool_join_suite", "pooled_branches_suite"):
            for index in (0, 5):
                path = ROOT / "research" / suite / f"model{index:03}.onnx"
                data, meta = compile_sequence(path)
                with self.subTest(suite=suite, model=index):
                    self.assertEqual(data, committed(suite, index))
                    self.assertEqual(meta["submission"], "serial")

    def test_deep_chain_twins_and_double_buffering(self):
        suite = ROOT / "research" / "deep_chain_suite"
        for entry in json.loads((suite / "manifest.json").read_text()):
            index = entry["index"]
            batched = (suite / "batched" / f"model{index:03}.bin").read_bytes()
            serial = (suite / f"model{index:03}.bin").read_bytes()
            reuse_batched = (suite / "reuse_batched" / f"model{index:03}.bin").read_bytes()
            with self.subTest(layers=entry["layers"]):
                self.assertFalse(decode(batched)["serial"])
                self.assertTrue(decode(serial)["serial"])
                self.assertIsNone(batched_layout(batched, decode(batched)))
                self.assertIsNone(batched_layout(reuse_batched, decode(reuse_batched)))
                self.assertEqual(len(serial), len(batched))
                self.assertGreaterEqual(entry["layers"], 8)
                # Double buffering must shrink the arena and keep it flat with depth.
                self.assertLess(entry["reuse_arena_bytes"], entry["arena_bytes"])
                self.assertTrue(entry["reuse_double_buffered"])
        results = json.loads((suite / "batched_results.json").read_text())
        for record in results:
            with self.subTest(layers=record["layers"]):
                for mode in ("serial", "batched", "reuse", "reuse_batched"):
                    self.assertTrue(record[mode]["passed"], (record["layers"], mode))
                self.assertGreater(record["speedup"], 1.5)

    def test_job_shape_and_depth_evidence(self):
        jobs = json.loads((GROUP / "job_results.json").read_text())
        by_name = {entry["name"]: entry for entry in jobs}
        self.assertTrue(by_name["C3-3xCNA-linked"]["passed"], "3-task linked job")
        self.assertTrue(by_name["C4-4xCNA-loop-linked"]["passed"], "4-task linked job")
        self.assertFalse(by_name["C1-2xCNA-terminal"]["passed"], "unlinked list")
        self.assertFalse(by_name["C6-6mixed-linked"]["passed"], "mixed linked list")
        self.assertFalse(by_name["C5-6mixed-linked-union"]["passed"], "mixed + union mask")
        depth = json.loads((GROUP / "depth_results.json").read_text())
        self.assertEqual([entry["tasks"] for entry in depth], [8, 32, 64])
        for entry in depth:
            self.assertTrue(entry["passed"], entry)
        self.assertTrue((GROUP / "crossover.json").exists())
        self.assertTrue((GROUP / "README.md").is_file())


if __name__ == "__main__":
    unittest.main()
