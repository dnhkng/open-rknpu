"""SPDX-License-Identifier: MIT

The release gate's logic: sdist normalisation, the contents audit, and the MANIFEST.in
guards that keep the research evidence out of the published compiler.

`research/check_reproducible_build.py` builds the real sdist twice (~25 s) and runs in CI
via `make reproducible`; these tests cover its logic on synthetic archives so a bug in the
gate itself is caught by the fast suite. Set `OPEN_RKNPU_SLOW_BUILD=1` to also build the
real sdist twice here.
"""
from pathlib import Path
import gzip
import io
import os
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import check_reproducible_build as gate  # noqa: E402  (the gate under test)


def write_tar(path, members):
    """Write `{name: (bytes, mtime)}` as a gzipped tar with a real directory prefix."""
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tar:
                for name, (data, mtime) in members.items():
                    info = tarfile.TarInfo("open_rknpu-0.1.0/" + name)
                    info.size = len(data)
                    info.mtime = mtime
                    tar.addfile(info, io.BytesIO(data))
    return path


def required_contents():
    contents = {name: (b"placeholder\n", 1700000000) for name in gate.REQUIRED}
    contents["src/open_rknpu/__init__.py"] = (b'__version__ = "0.1.0"\n', 1700000000)
    return contents


class NormalizeTests(unittest.TestCase):
    def test_two_builds_with_different_timestamps_normalise_identically(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            first = write_tar(tmp / "a.tar.gz", required_contents())
            stale = {name: (data, 1) for name, (data, _mtime) in required_contents().items()}
            second = write_tar(tmp / "b.tar.gz", stale)
            self.assertNotEqual(first.read_bytes(), second.read_bytes())
            one = gate.normalize_sdist(first, tmp / "one.tar.gz")
            two = gate.normalize_sdist(second, tmp / "two.tar.gz")
            self.assertEqual(one.read_bytes(), two.read_bytes())

    def test_normalisation_preserves_every_member_and_its_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = write_tar(tmp / "a.tar.gz", required_contents())
            before = gate.tar_members(source)
            after = gate.tar_members(gate.normalize_sdist(source, tmp / "n.tar.gz"))
            self.assertEqual({name: data for name, (data, _mtime) in before.items()},
                             {name: data for name, (data, _mtime) in after.items()})
            self.assertEqual({mtime for _data, mtime in after.values()}, {gate.EPOCH})

    def test_normalisation_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = write_tar(tmp / "a.tar.gz", required_contents())
            once = gate.normalize_sdist(source, tmp / "one.tar.gz")
            twice = gate.normalize_sdist(once, tmp / "two.tar.gz")
            self.assertEqual(once.read_bytes(), twice.read_bytes())


class AuditTests(unittest.TestCase):
    def test_a_complete_compiler_sdist_passes(self):
        missing, denied, oversized, total = gate.audit(required_contents(), gate.REQUIRED, gate.REQUIRED_PREFIXES)
        self.assertEqual(missing, [])
        self.assertEqual(denied, [])
        self.assertEqual(oversized, [])
        self.assertGreater(total, 0)

    def test_missing_metadata_is_reported(self):
        contents = required_contents()
        del contents["THIRD_PARTY.md"]
        missing, _denied, _oversized, _total = gate.audit(contents, gate.REQUIRED, gate.REQUIRED_PREFIXES)
        self.assertIn("THIRD_PARTY.md", missing)

    def test_evidence_members_are_denied(self):
        for name in ("research/foo_suite/model000.bin", "tests/test_graph.py",
                     "docs/board.md", "examples/cookbook/01.py", "src/open_rknpu/__pycache__/x.pyc",
                     "hardware_refs/RV1106.pdf", "research/toolchain/bin/gcc"):
            with self.subTest(member=name):
                contents = required_contents()
                contents[name] = (b"x", 0)
                _missing, denied, _oversized, _total = gate.audit(contents, gate.REQUIRED, gate.REQUIRED_PREFIXES)
                self.assertIn(name, denied)

    def test_a_fixture_sized_member_is_reported(self):
        contents = required_contents()
        contents["src/open_rknpu/big.py"] = (b"0" * (gate.MAX_MEMBER_BYTES + 1), 0)
        _missing, _denied, oversized, _total = gate.audit(contents, gate.REQUIRED, gate.REQUIRED_PREFIXES)
        self.assertIn("src/open_rknpu/big.py", oversized)

    def test_changed_members_finds_a_single_difference(self):
        first = required_contents()
        second = dict(first)
        second["README.md"] = (b"different\n", first["README.md"][1])
        self.assertEqual(len(gate.changed_members(first, second)), 1)
        self.assertEqual(gate.changed_members(first, dict(first)), [])

    def test_changed_members_reports_a_removed_file(self):
        first = required_contents()
        second = dict(first)
        del second["LICENSE"]
        problems = gate.changed_members(first, second)
        self.assertTrue(problems and "member sets differ" in problems[0], problems)


class ManifestGuardTests(unittest.TestCase):
    def test_manifest_prunes_the_evidence_tree_and_keeps_attribution(self):
        manifest = (ROOT / "MANIFEST.in").read_text()
        for name in ("prune research", "prune tests", "prune docs", "prune examples"):
            with self.subTest(rule=name):
                self.assertIn(name, manifest)
        for name in ("README.md", "LICENSE", "THIRD_PARTY.md"):
            with self.subTest(rule=name):
                self.assertIn(f"include {name}", manifest)

    def test_pyproject_ships_the_runtime_and_both_licence_files(self):
        pyproject = (ROOT / "pyproject.toml").read_text()
        self.assertIn('license-files = ["LICENSE", "THIRD_PARTY.md"]', pyproject)
        for name in gate.WHEEL_RUNTIME:
            with self.subTest(runtime_file=name):
                self.assertIn(f"runtime/{name}", pyproject)
                self.assertTrue((ROOT / "runtime" / name).is_file(),
                                f"data-files names a missing runtime/{name}")

    def test_every_package_module_is_expected_in_the_wheel(self):
        modules = {f"open_rknpu/{path.name}" for path in (ROOT / "src" / "open_rknpu").glob("*.py")}
        self.assertGreaterEqual(len(modules), 20)
        self.assertIn("open_rknpu/cli.py", modules)


@unittest.skipUnless(os.environ.get("OPEN_RKNPU_SLOW_BUILD"),
                     "set OPEN_RKNPU_SLOW_BUILD=1 to build the real sdist twice")
class SlowBuildTests(unittest.TestCase):
    def test_the_real_sdist_is_reproducible_and_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            first = gate.build_sdist(tmp / "one")
            second = gate.build_sdist(tmp / "two")
            self.assertEqual(gate.changed_members(gate.tar_members(first),
                                                  gate.tar_members(second)), [])
            one = gate.normalize_sdist(first, tmp / "one-normalized.tar.gz")
            two = gate.normalize_sdist(second, tmp / "two-normalized.tar.gz")
            self.assertEqual(one.read_bytes(), two.read_bytes())
            missing, denied, oversized, _total = gate.audit(
                gate.tar_members(first), gate.REQUIRED, gate.REQUIRED_PREFIXES)
            self.assertEqual((missing, denied, oversized), ([], [], []))


if __name__ == "__main__":
    unittest.main()
