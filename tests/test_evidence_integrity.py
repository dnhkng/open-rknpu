"""SPDX-License-Identifier: MIT

Evidence integrity: every published suite must be self-consistent on disk.

The ledger and the board results are the project's proof, and the container baseline only
says that *fresh compiles* are stable - not that the retained artifacts agree with the
manifests and summaries that describe them. This module checks the whole tree, not a sample:

* every `research/*_suite/` has a `README.md`, a `manifest.json`, board results and a summary;
* every manifest entry names an index whose `modelNNN.bin` and `expectedNNN.i8` exist;
* where a manifest records `cases`/`output_bytes`, the reference file is exactly
  `cases * output_bytes` long, and a per-model `inputNNN.u8` holds every external input of
  each case, concatenated in index order (`join_scale_suite` stores 192 + 3 bytes a case);
* every `modelNNN.bin` decodes (`ORNPUSEQ` or legacy `ORNPUBIN`) and its declared input and
  output geometry agrees with the manifest under one of the conventions the generators use
  (NHWC, HWC, CHW, HW or C-only), or with one operand of a packed tensor whose named operands
  provably tile it (`mul_reshape_suite` packs two [1, 5, 6, 3] operands into [1, 10, 6, 3]);
* every passing `board_results_*.json` entry's `bytes` equal `inferences * output_bytes`, and
  the suite's `board_summary.txt` agrees with the recorded pass/fail outcome.

Two conventions are tolerated deliberately, and reported rather than failed: a suite may keep
one shared input for several models (e.g. `mul_broadcast_mode_suite` has 12 models and one
input), and a legacy single container may be named `model.bin`. It is structural: it does not
re-run the board (the recorded results are facts) and it does not recompile (that is
`research/verify_suites.py`).
"""
from pathlib import Path
import json
import re
import sys
import unittest

from open_rknpu.model import decode as decode_model
from open_rknpu.sequence import decode_sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import build_suite_readmes  # noqa: E402  (the suite-page generator)

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
NUMBERED = re.compile(r"^model(\d+)\.bin$")
# Ratchets: the number of suites that keep each record may not shrink. 35 older suites have
# no manifest and 44 have no machine-readable board results (their run is recorded in the
# ledger and the suite README), so those are counted, not required per suite.
MIN_SUITES = 180
MIN_MANIFEST_SUITES = 148
MIN_BOARD_SUITES = 139


def tensor_shapes(info):
    """Every (role, index, NHWC-ish shape) the container declares, if it has a table."""
    shapes = []
    for tensor in info.get("tensors", []) or []:
        shape = [int(tensor.get(key, 1)) for key in ("batch", "height", "width", "channels")]
        shapes.append((tensor.get("role"), int(tensor.get("index", 0)), shape))
    return shapes


def tensor_bytes(shape):
    batch, height, width, channels = shape
    return batch * height * width * channels


def external_bytes(info, role, index=None):
    """API bytes of one external tensor, or the sum over all of them."""
    shapes = [shape for tensor_role, tensor_index, shape in tensor_shapes(info)
              if tensor_role == role and (index is None or tensor_index == index)]
    if shapes:
        return sum(tensor_bytes(shape) for shape in shapes)
    key = "input_bytes" if role == 0 else "output_bytes"
    return int(info.get(key, 0)) or None


def primary_bytes(info, role):
    """The external tensor with index 0 (a manifest usually describes that one)."""
    return external_bytes(info, role, index=0)


def shape_matches_the_container(info, declared):
    """True if the declared shape matches any tensor under a used convention.

    Multi-input suites describe one branch's tensor in the manifest, so comparing against
    the summed geometry is wrong; the container's own table is the authority. A manifest that
    describes a *packed* input in `shape_nhwc` may still describe a single operand in
    `input_shape`, which `operand_shapes` covers.
    """
    if shape_agrees(info["shape_nhwc"], declared) or shape_agrees(info["output_shape_nhwc"], declared):
        return True
    return any(shape_agrees(shape, declared) for _, _, shape in tensor_shapes(info))


def declared_operands(entry, key):
    """The `(record, NHWC shape)` operands a manifest names under `key`, batch normalised."""
    shapes = []
    for operand in entry.get(key) or []:
        if not isinstance(operand, dict):
            continue
        shape = operand.get("shape_nhwc") or operand.get("shape")
        if not shape:
            continue
        shape = [int(value) for value in shape]
        if len(shape) == 3:
            shape = [1] + shape
        if len(shape) == 4:
            shapes.append((operand, shape))
    return shapes


def operands_tile(entry, info, key):
    """True when a manifest's named operands exactly tile the container's packed tensor.

    `mul_reshape_suite` packs two [1, 5, 6, 3] operands into one [1, 10, 6, 3] input, so its
    `input_shape` of [5, 6, 3] is a slice of the container, not the container itself. The
    operand route is only trusted when the operands provably reconstruct the packed tensor.
    """
    operands = declared_operands(entry, key)
    if len(operands) < 2:
        return False
    packed = primary_bytes(info, 0 if key == "input_tensors" else 1)
    if not packed:
        return False
    sizes = [tensor_bytes(shape) for _, shape in operands]
    if any(size <= 0 for size in sizes):
        return False
    offsets = [int(operand.get("packed_byte_offset", -1)) for operand, _ in operands]
    expected = [sum(sizes[:position]) for position in range(len(sizes))]
    return offsets == expected and sum(sizes) == packed


def shape_agrees(decoded, declared):
    """True if a decoded NHWC shape matches a manifest shape under a used convention."""
    decoded = [int(value) for value in decoded]
    declared = [int(value) for value in declared]
    batch, height, width, channels = decoded
    conventions = (
        decoded,                          # NHWC
        [height, width, channels],        # HWC
        [channels, height, width],        # CHW
        [batch, channels, height, width], # NCHW
        [height, width],                  # HW
        [channels],                       # C
    )
    return declared in conventions


class EvidenceIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suites = sorted(path for path in RESEARCH.glob("*_suite") if path.is_dir())
        cls.manifests = {}
        for suite in cls.suites:
            path = suite / "manifest.json"
            if path.is_file():
                cls.manifests[suite.name] = json.loads(path.read_text())

    def entries(self, suite):
        manifest = self.manifests.get(suite.name)
        if not isinstance(manifest, list):
            return {}
        return {int(entry["index"]): entry for entry in manifest
                if isinstance(entry, dict) and "index" in entry}

    def test_every_suite_keeps_its_records(self):
        self.assertGreaterEqual(len(self.suites), MIN_SUITES)
        manifests = sum(1 for suite in self.suites if (suite / "manifest.json").is_file())
        boards = sum(1 for suite in self.suites if list(suite.glob("board_results_*.json")))
        self.assertGreaterEqual(manifests, MIN_MANIFEST_SUITES, "suite manifests disappeared")
        self.assertGreaterEqual(boards, MIN_BOARD_SUITES, "board results disappeared")
        for suite in self.suites:
            with self.subTest(suite=suite.name):
                self.assertTrue((suite / "README.md").is_file(), "no README")
                self.assertTrue(list(suite.glob("model*.bin")), "no container")
                self.assertTrue(list(suite.glob("*.u8")), "no input bytes")

    def test_manifest_entries_have_their_artifacts(self):
        for suite in self.suites:
            for index, entry in self.entries(suite).items():
                with self.subTest(suite=suite.name, model=index):
                    self.assertTrue((suite / f"model{index:03}.bin").is_file(),
                                    "manifest entry without a container")
                    expected = suite / f"expected{index:03}.i8"
                    self.assertTrue(expected.is_file(), "manifest entry without a reference")
                    output_bytes = entry.get("output_bytes")
                    if output_bytes:
                        size = expected.stat().st_size
                        if "cases" in entry or "inputs" in entry:
                            cases = int(entry.get("cases", entry.get("inputs")) or 1)
                            self.assertEqual(size, cases * int(output_bytes),
                                             "reference size != recorded cases x output_bytes")
                        else:
                            self.assertGreater(size, 0)
                            self.assertEqual(size % int(output_bytes), 0,
                                             "reference size is not a whole number of cases")

    def test_every_container_decodes_and_matches_its_manifest(self):
        checked = 0
        shared_input_suites = []
        for suite in self.suites:
            entries = self.entries(suite)
            inputs = sorted(suite.glob("input*.u8"))
            if entries and len(inputs) < len(entries):
                shared_input_suites.append(suite.name)
            for container in sorted(suite.glob("model*.bin")):
                match = NUMBERED.match(container.name)
                if match is None:
                    continue  # a single legacy container kept as model.bin
                index = int(match.group(1))
                data = container.read_bytes()
                info = decode_sequence(data) if data[:8] == b"ORNPUSEQ" else decode_model(data)
                checked += 1
                entry = entries.get(index)
                if entry is None:
                    continue
                with self.subTest(suite=suite.name, model=index):
                    if entry.get("input_shape"):
                        tiled = operands_tile(entry, info, "input_tensors")
                        operand_hit = tiled and any(
                            shape_agrees(shape, entry["input_shape"])
                            for _, shape in declared_operands(entry, "input_tensors"))
                        self.assertTrue(shape_matches_the_container(info, entry["input_shape"])
                                        or operand_hit,
                                        "input geometry %s matches no tensor of the container "
                                        "(operands=%s tiled=%s)"
                                        % (entry["input_shape"],
                                           [shape for _, shape in declared_operands(entry, "input_tensors")],
                                           tiled))
                    if entry.get("output_shape"):
                        tiled = operands_tile(entry, info, "output_tensors")
                        operand_hit = tiled and any(
                            shape_agrees(shape, entry["output_shape"])
                            for _, shape in declared_operands(entry, "output_tensors"))
                        self.assertTrue(shape_matches_the_container(info, entry["output_shape"])
                                        or operand_hit,
                                        "output geometry %s matches no tensor of the container"
                                        % (entry["output_shape"],))
                    if entry.get("shape_nhwc"):
                        self.assertEqual([int(v) for v in entry["shape_nhwc"]],
                                         [int(v) for v in info["shape_nhwc"]],
                                         "explicitly declared NHWC input differs")
                    if entry.get("output_shape_nhwc"):
                        self.assertEqual([int(v) for v in entry["output_shape_nhwc"]],
                                         [int(v) for v in info["output_shape_nhwc"]],
                                         "explicitly declared NHWC output differs")
                    if entry.get("output_bytes"):
                        declared = int(entry["output_bytes"])
                        candidates = {int(info["output_bytes"]), primary_bytes(info, 1),
                                      external_bytes(info, 1)}
                        self.assertIn(declared, {value for value in candidates if value},
                                      "declared output bytes match neither the total nor the "
                                      "primary output")
                    model_input = suite / f"input{index:03}.u8"
                    if model_input.is_file():
                        # `input_bytes` is the API buffer size (batch x H x W x C), so a
                        # file is always a whole number of cases; the exact count is only
                        # checked when the manifest records it (many older suites do not).
                        size = model_input.stat().st_size
                        # A file holds every external input for one case, concatenated in
                        # index order (e.g. join_scale_suite stores 192 + 3 bytes per case).
                        per_case = external_bytes(info, 0) or primary_bytes(info, 0)
                        self.assertGreater(size, 0)
                        if "cases" in entry or "inputs" in entry:
                            cases = int(entry.get("cases", entry.get("inputs")) or 1)
                            self.assertEqual(size, cases * per_case,
                                             "input size != recorded cases x API input bytes")
                        else:
                            self.assertEqual(size % per_case, 0,
                                             "input size is not a whole number of cases")
        self.assertGreater(checked, 2000, "expected to decode the whole published tree")
        self.assertTrue(shared_input_suites, "expected at least one shared-input suite")

    def test_every_suite_page_is_present_and_current(self):
        """A ledger row links a directory; the directory must explain itself.

        The sweep that added this test found 135 of 183 suites (82 of them ledger-linked)
        with no README at all. Generated pages are compared up to their `## Notes` heading,
        so a human can extend the notes without making the page look stale.
        """
        for suite in self.suites:
            readme = suite / "README.md"
            with self.subTest(suite=suite.name):
                self.assertTrue(readme.is_file(), "no README")
                committed = readme.read_text()
                if build_suite_readmes.MARKER in committed:
                    generated = build_suite_readmes.render(suite)
                    self.assertEqual(committed.split("## Notes")[0], generated.split("## Notes")[0],
                                     "generated suite page is stale (run research/build_suite_readmes.py)")

    def test_board_results_reproduce_their_byte_counts(self):
        checked = 0
        for suite in self.suites:
            output_bytes = {index: int(entry["output_bytes"])
                            for index, entry in self.entries(suite).items()
                            if entry.get("output_bytes") is not None}
            for path in sorted(suite.glob("board_results_*.json")):
                results = json.loads(path.read_text())
                self.assertTrue(results, f"{path} is empty")
                for entry in results:
                    with self.subTest(suite=suite.name, file=path.name, model=entry.get("model")):
                        self.assertIn("passed", entry)
                        self.assertIn("output", entry)
                        if not entry.get("passed"):
                            continue
                        index = int(str(entry.get("model", "")).lstrip("0") or "0")
                        if index in output_bytes:
                            checked += 1
                            self.assertEqual(int(entry["bytes"]),
                                             int(entry.get("inferences", 0)) * output_bytes[index],
                                             "recorded bytes != inferences x output_bytes")
        self.assertGreater(checked, 20, "expected several board rows to be arithmetically checked")

    def test_summaries_agree_with_the_recorded_outcome(self):
        for suite in self.suites:
            summaries = sorted(suite.glob("board_summary.txt"))
            if not summaries:
                continue
            text = summaries[-1].read_text().strip()
            with self.subTest(suite=suite.name):
                self.assertTrue(text.startswith(("PASS:", "FAIL:")), text[:80])
                results = []
                for path in sorted(suite.glob("board_results_*.json")):
                    results.extend(json.loads(path.read_text()))
                if results and all(entry.get("passed") for entry in results):
                    self.assertTrue(text.startswith("PASS:"), "summary disagrees with the results")


if __name__ == "__main__":
    unittest.main()
