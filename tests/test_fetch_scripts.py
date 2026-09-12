"""SPDX-License-Identifier: MIT

The dataset and toolchain fetchers must fail loudly, offline, and on the right hash.

`examples/fetch_idx.py`, `examples/mel-kws/fetch_data.py` and `research/fetch_toolchain.py`
are the scripts that reach the network. A fresh clone has no data and CI has no network, so
`urllib.request.urlopen` is monkeypatched in every behavioural test here, and the real socket
is additionally replaced by one that raises. Nothing is downloaded and no `/tmp` cache is
touched.

What is pinned, and why:

* `fetch_idx.py` pins a 64-hex sha256 per IDX file and deletes a file whose digest does not
  match. The tests assert that the accepted path writes to the documented
  `research/pretrained/<dataset>/<split>/` location, that the mismatch path raises **that
  exact message** and removes the corrupt file, that an already-verified file is not
  re-downloaded, and that a connection error propagates instead of being swallowed.
* `fetch_data.py` is the FSDD fetcher; there is no `research/fetch_fsdd.sh` in this checkout,
  so the `examples/fetch_*` script named by the row is this file. It pins the archive sha256
  and then requires exactly 3,000 recordings in the zip before extracting. Both the archive
  mismatch and the wrong-recording-count messages are asserted verbatim. `/tmp/fsdd.zip` is
  redirected into the test's temp directory by patching the loaded module's `Path`, so the
  real cache is never read or written.
* `fetch_toolchain.py` pins every selected file by its **git blob id** (SHA-1 content
  address in `research/toolchain_tree.json`) and now recomputes that id before writing the
  bytes, so a truncated or substituted download stops the fetch instead of landing a bad
  compiler in the tree. The tests load the real script from a temp directory with a minimal
  manifest and assert the exact pinned `https://raw.githubusercontent.com/...` URLs it
  requests, that selected blobs land at the manifest paths with the manifest mode (and
  unselected ones do not), that a payload whose id cannot match the pin raises the exact
  message and writes nothing, and that the connection error surfaces.

Metadata tests read the scripts' pins (by loading the modules, which is reading them) and
assert the shape that keeps them meaningful: every download base is https except the one
documented legacy Fashion-MNIST host, and every sha256 pin is a full 64-hex digest. A missing
or truncated pin is treated as a regression.

None of the three scripts shells out to `sha256sum`/`curl`; all are pure Python, so the
Python paths are the only paths to test.
"""
from pathlib import Path
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import socket  # noqa: F401  (patched below to prove no socket is opened)
import tempfile
import unittest
import urllib.error
import urllib.request
import zipfile
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")

# The only plaintext download base the repository ships, and it is a deliberate legacy
# endpoint (the canonical Zalando Fashion-MNIST bucket). Any *other* http URL is a failure.
PLAINTEXT_BASE_ALLOWLIST = {"http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"}
TOOLCHAIN_PREFIX = "arm-rockchip830-linux-uclibcgnueabihf"


def load_module(relative_path, name):
    """Import a top-level script from the checkout without running its `__main__`."""
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def quiet(callable_):
    """Run `callable_` with stdout swallowed; import-time and fetch progress prints are noise."""
    with contextlib.redirect_stdout(io.StringIO()):
        return callable_()


class FetchIdxTests(unittest.TestCase):
    """`examples/fetch_idx.py`: the MNIST / Fashion-MNIST IDX fetcher."""

    def setUp(self):
        self.module = load_module("examples/fetch_idx.py", "fetch_idx_under_test")
        self.payloads = {
            name: bytes([index]) * 8 for index, name in enumerate(self.module.FILES)
        }
        self.spec = {
            "base": "https://example.invalid/mnist/",
            "directory": "research/pretrained/mnist",
            "sha256": {name: sha256(data) for name, data in self.payloads.items()},
        }
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls = []

    def fake_urlopen(self, url, timeout=None):
        self.calls.append(url)
        return io.BytesIO(self.payloads[url.rsplit("/", 1)[-1]])

    def test_correct_payload_is_written_where_the_script_says(self):
        """A digest that matches the pin is accepted and stored under `<directory>/<split>/`."""
        self.module.DATASETS = {"mnist": self.spec}
        self.module.ROOT = self.root
        with mock.patch("socket.socket", side_effect=AssertionError("a real socket was opened")), \
                mock.patch("urllib.request.urlopen", self.fake_urlopen):
            quiet(lambda: self.module.fetch("mnist", force=True))

        self.assertEqual(len(self.calls), len(self.module.FILES))
        for name in self.module.FILES:
            destination = self.root / self.spec["directory"] / self.module.SPLIT[name] / name
            self.assertEqual(destination.read_bytes(), self.payloads[name])
        self.assertEqual([url.rsplit("/", 1)[-1] for url in self.calls], list(self.module.FILES))
        self.assertTrue(all(url.startswith("https://") for url in self.calls))

    def test_corrupted_payload_raises_the_exact_message_and_deletes_the_file(self):
        """The pinned digest is the error's payload; a corrupt file must not be left behind."""
        corrupt = b"not the pinned bytes"
        self.module.DATASETS = {"mnist": self.spec}
        self.module.ROOT = self.root
        expected = self.spec["sha256"][self.module.FILES[0]]
        with mock.patch("urllib.request.urlopen", lambda url, timeout=None: io.BytesIO(corrupt)):
            with self.assertRaises(SystemExit) as caught:
                quiet(lambda: self.module.fetch("mnist", force=True))

        self.assertEqual(
            str(caught.exception),
            f"{self.module.FILES[0]}: sha256 {sha256(corrupt)} does not match the pinned {expected}",
        )
        destination = self.root / self.spec["directory"] / self.module.SPLIT[self.module.FILES[0]] \
            / self.module.FILES[0]
        self.assertFalse(destination.exists(), "the corrupt file must be removed before raising")

    def test_verified_files_are_not_downloaded(self):
        """The offline happy path: every already-verified file skips urlopen."""
        for name in self.module.FILES:
            destination = self.root / self.spec["directory"] / self.module.SPLIT[name] / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(self.payloads[name])
        self.module.DATASETS = {"mnist": self.spec}
        self.module.ROOT = self.root
        poisoned = mock.Mock(side_effect=AssertionError("urlopen must not be called"))
        with mock.patch("urllib.request.urlopen", poisoned):
            quiet(lambda: self.module.fetch("mnist", force=False))
        poisoned.assert_not_called()

    def test_connection_error_surfaces(self):
        """No silent skip: a transport failure reaches the caller."""
        self.module.DATASETS = {"mnist": self.spec}
        self.module.ROOT = self.root
        with mock.patch("urllib.request.urlopen",
                        side_effect=urllib.error.URLError("no route to host")):
            with self.assertRaises(urllib.error.URLError) as caught:
                quiet(lambda: self.module.fetch("mnist", force=True))
        self.assertIn("no route to host", str(caught.exception))


class FetchFsddTests(unittest.TestCase):
    """`examples/mel-kws/fetch_data.py`: the Free Spoken Digit Dataset fetcher."""

    def setUp(self):
        self.module = load_module("examples/mel-kws/fetch_data.py", "fetch_data_under_test")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dest = self.root / "fsdd"
        self.calls = []

    def redirect_path(self, value):
        """Send the hard-coded `/tmp/fsdd.zip` cache into this test's temp directory."""
        if str(value) == "/tmp/fsdd.zip":
            return self.root / "fsdd.zip"
        return Path(value)

    def make_archive(self, count):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
            for index in range(count):
                archive.writestr(
                    f"free-spoken-digit-dataset-master/recordings/{index}_jack_0.wav",
                    b"RIFF0000WAVEfmt ",
                )
        return buffer.getvalue()

    def patch_module(self, archive, sha256_pin):
        self.module.Path = self.redirect_path  # type: ignore[assignment]
        self.module.DEST = self.dest
        self.module.SHA256 = sha256_pin

        def fake_urlopen(url, timeout=None):
            self.calls.append(url)
            return io.BytesIO(archive)

        return mock.patch("urllib.request.urlopen", fake_urlopen)

    def test_correct_archive_is_verified_and_extracted(self):
        archive = self.make_archive(3000)
        pin = sha256(archive)
        with self.patch_module(archive, pin), \
                mock.patch("socket.socket", side_effect=AssertionError("a real socket was opened")):
            quiet(self.module.main)

        recordings = self.dest / "recordings"
        self.assertEqual(len(list(recordings.glob("*.wav"))), 3000)
        self.assertTrue((self.root / "fsdd.zip").is_file(), "the redirected cache must be used")
        source = json.loads((self.dest / "source.json").read_text())
        self.assertEqual(source["archive_sha256"], pin)
        self.assertEqual(source["url"], self.module.ARCHIVE)
        self.assertEqual(source["license"], self.module.LICENSE)
        self.assertEqual(self.calls, [self.module.ARCHIVE])
        self.assertTrue(self.module.ARCHIVE.startswith("https://"))

    def test_corrupted_archive_raises_the_exact_message(self):
        corrupt = b"corrupted archive bytes"
        with self.patch_module(corrupt, sha256(b"the real archive")):
            with self.assertRaises(SystemExit) as caught:
                quiet(self.module.main)
        self.assertEqual(
            str(caught.exception),
            f"archive sha256 {sha256(corrupt)} does not match the pinned "
            f"{sha256(b'the real archive')}",
        )
        self.assertFalse((self.dest / "recordings").exists())

    def test_wrong_recording_count_raises_the_exact_message(self):
        archive = self.make_archive(2)
        with self.patch_module(archive, sha256(archive)):
            with self.assertRaises(SystemExit) as caught:
                quiet(self.module.main)
        self.assertEqual(str(caught.exception), "expected 3000 recordings, found 2")

    def test_connection_error_surfaces(self):
        self.module.Path = self.redirect_path  # type: ignore[assignment]
        self.module.DEST = self.dest
        self.module.SHA256 = "0" * 64
        with mock.patch("urllib.request.urlopen",
                        side_effect=urllib.error.URLError("offline")):
            with self.assertRaises(urllib.error.URLError):
                quiet(self.module.main)

    def test_complete_dataset_is_not_re_downloaded(self):
        """The offline early return: 3,000 recordings on disk must not touch the network."""
        recordings = self.dest / "recordings"
        recordings.mkdir(parents=True)
        for index in range(3000):
            (recordings / f"{index}.wav").write_bytes(b"")
        self.module.Path = self.redirect_path  # type: ignore[assignment]
        self.module.DEST = self.dest
        self.module.SHA256 = "0" * 64
        poisoned = mock.Mock(side_effect=AssertionError("urlopen must not be called"))
        with mock.patch("urllib.request.urlopen", poisoned):
            quiet(self.module.main)
        poisoned.assert_not_called()


TOOLCHAIN_BLOB = b"blob-content"          # what the fake urlopen serves for every URL


def git_blob_sha(data):
    """The git object id of `data`: what `toolchain_tree.json` pins per file."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def toolchain_manifest(payload=TOOLCHAIN_BLOB, pin=None):
    """A minimal tree exercising every branch of the script's `selected()` filter.

    Every selected blob is served the same payload, so every pin is that payload's git
    blob id - exactly the content address the real manifest carries (and which the fetcher
    now verifies).
    """
    sha = pin if pin is not None else git_blob_sha(payload)
    return {
        "sha": "0" * 40,
        "url": "https://api.github.com/repos/LuckfoxTECH/luckfox-pico/git/trees/" + "0" * 40,
        "tree": [
            {"path": f"bin/{TOOLCHAIN_PREFIX}-gcc", "mode": "100755", "type": "blob",
             "sha": sha, "size": len(payload)},
            {"path": f"bin/{TOOLCHAIN_PREFIX}-ld", "mode": "120000", "type": "blob",
             "sha": sha, "size": len(payload)},
            {"path": "libexec/gcc/x/8.4.0/cc1", "mode": "100755", "type": "blob",
             "sha": sha, "size": len(payload)},
            {"path": f"{TOOLCHAIN_PREFIX}/bin/as", "mode": "100755", "type": "blob",
             "sha": sha, "size": len(payload)},
            {"path": f"bin/{TOOLCHAIN_PREFIX}-g++", "mode": "100755", "type": "blob",
             "sha": "4" * 40, "size": len(payload)},
        ],
    }


class FetchToolchainTests(unittest.TestCase):
    """`research/fetch_toolchain.py`: the Luckfox cross-toolchain fetcher (content-pinned).

    It is a top-level script, so the real source is copied into a temp directory next to a
    minimal `toolchain_tree.json` and imported there: every write lands in the temp tree and
    `urlopen` is the only thing patched.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls = []

    def load(self, responder, manifest=None):
        script = self.root / "fetch_toolchain.py"
        script.write_bytes((ROOT / "research" / "fetch_toolchain.py").read_bytes())
        (self.root / "toolchain_tree.json").write_text(
            json.dumps(manifest if manifest is not None else toolchain_manifest()))

        def fake_urlopen(url, timeout=None):
            self.calls.append(url)
            return io.BytesIO(responder(url))

        spec = importlib.util.spec_from_file_location("fetch_toolchain_under_test", script)
        module = importlib.util.module_from_spec(spec)
        with mock.patch("urllib.request.urlopen", fake_urlopen):
            quiet(lambda: spec.loader.exec_module(module))
        return module

    def test_requests_the_pinned_https_urls_and_skips_unselected_blobs(self):
        module = self.load(lambda url: TOOLCHAIN_BLOB)
        base = module.base
        self.assertTrue(base.startswith("https://"), base)
        expected = {
            base + f"bin/{TOOLCHAIN_PREFIX}-gcc",
            base + f"bin/{TOOLCHAIN_PREFIX}-ld",
            base + "libexec/gcc/x/8.4.0/cc1",
            base + f"{TOOLCHAIN_PREFIX}/bin/as",
        }
        self.assertEqual(set(self.calls), expected)
        self.assertTrue(all(url.startswith("https://") for url in self.calls))
        self.assertNotIn(base + f"bin/{TOOLCHAIN_PREFIX}-g++", self.calls)

    def test_selected_blobs_land_at_the_manifest_paths_with_the_manifest_mode(self):
        self.load(lambda url: TOOLCHAIN_BLOB)
        toolchain = self.root / "toolchain"
        binary = toolchain / "bin" / f"{TOOLCHAIN_PREFIX}-gcc"
        self.assertEqual(binary.read_bytes(), TOOLCHAIN_BLOB)
        self.assertEqual(os.stat(binary).st_mode & 0o777, 0o755)
        self.assertTrue((toolchain / "libexec/gcc/x/8.4.0/cc1").is_file())
        self.assertTrue((toolchain / TOOLCHAIN_PREFIX / "bin/as").is_file())
        symlink = toolchain / "bin" / f"{TOOLCHAIN_PREFIX}-ld"
        self.assertTrue(symlink.is_symlink())
        self.assertEqual(os.readlink(symlink), TOOLCHAIN_BLOB.decode())
        self.assertFalse((toolchain / "bin" / f"{TOOLCHAIN_PREFIX}-g++").exists())

    def test_a_payload_that_cannot_match_the_pin_is_refused(self):
        """The fetcher verifies the git blob id before writing anything.

        `toolchain_tree.json` pins every file by its content address; a truncated or
        substituted download must stop the fetch, not land a bad compiler in the tree.
        """
        manifest = toolchain_manifest(pin="0" * 40)   # cannot match TOOLCHAIN_BLOB
        with self.assertRaises(SystemExit) as caught:
            self.load(lambda url: TOOLCHAIN_BLOB, manifest=manifest)
        self.assertIn("does not match the manifest pin", str(caught.exception))
        self.assertFalse((self.root / "toolchain" / "bin" / f"{TOOLCHAIN_PREFIX}-gcc").exists(),
                         "a mismatching payload must not be written")

    def test_the_manifest_pin_matches_the_served_payload(self):
        """The accept path: the pin is recomputed from the bytes, not trusted."""
        self.load(lambda url: TOOLCHAIN_BLOB)
        self.assertEqual(git_blob_sha(TOOLCHAIN_BLOB), toolchain_manifest()["tree"][0]["sha"])
        self.assertTrue((self.root / "toolchain" / "bin" / f"{TOOLCHAIN_PREFIX}-gcc").is_file())

    def test_connection_error_surfaces(self):
        def offline(url, timeout=None):
            self.calls.append(url)
            raise urllib.error.URLError("offline")

        script = self.root / "fetch_toolchain.py"
        script.write_bytes((ROOT / "research" / "fetch_toolchain.py").read_bytes())
        (self.root / "toolchain_tree.json").write_text(json.dumps(toolchain_manifest()))
        spec = importlib.util.spec_from_file_location("fetch_toolchain_offline", script)
        module = importlib.util.module_from_spec(spec)
        with mock.patch("urllib.request.urlopen", offline):
            with self.assertRaises(urllib.error.URLError):
                quiet(lambda: spec.loader.exec_module(module))
        self.assertTrue(self.calls, "the offline run must still have attempted the pinned URLs")


class PinMetadataTests(unittest.TestCase):
    """The pins themselves: a missing or malformed pin is a regression, not a footnote."""

    def test_fetch_idx_pins_every_file_to_a_full_sha256(self):
        module = load_module("examples/fetch_idx.py", "fetch_idx_pins")
        self.assertEqual(set(module.DATASETS), {"mnist", "fashion"})
        for name, spec in module.DATASETS.items():
            self.assertEqual(set(spec["sha256"]), set(module.FILES), name)
            for filename, pin in spec["sha256"].items():
                self.assertRegex(pin, HEX64, f"{name}/{filename} is not a full sha256")

    def test_fetch_idx_download_bases_are_https_with_one_documented_exception(self):
        module = load_module("examples/fetch_idx.py", "fetch_idx_bases")
        plaintext = {
            spec["base"] for spec in module.DATASETS.values()
            if not spec["base"].startswith("https://")
        }
        self.assertEqual(plaintext, PLAINTEXT_BASE_ALLOWLIST)
        for spec in module.DATASETS.values():
            self.assertTrue(spec["base"].endswith("/"), spec["base"])
            self.assertTrue(spec["directory"].startswith("research/pretrained/"))

    def test_fsdd_archive_pin_is_a_full_sha256_on_https(self):
        module = load_module("examples/mel-kws/fetch_data.py", "fetch_data_pins")
        self.assertRegex(module.SHA256, HEX64)
        self.assertTrue(module.ARCHIVE.startswith("https://"), module.ARCHIVE)
        self.assertTrue(module.ARCHIVE.endswith(".zip"), module.ARCHIVE)
        self.assertTrue(module.LICENSE)

    def test_toolchain_manifest_pins_every_blob_url_and_git_sha(self):
        manifest = json.loads((ROOT / "research/toolchain_tree.json").read_text())
        self.assertTrue(manifest["url"].startswith("https://"), manifest["url"])
        self.assertRegex(manifest["sha"], GIT_SHA)
        blobs = [entry for entry in manifest["tree"] if entry["type"] == "blob"]
        self.assertGreater(len(blobs), 1000, "the pinned toolchain tree lost its blobs")
        for entry in blobs:
            self.assertTrue(entry["url"].startswith("https://"), entry["url"])
            self.assertRegex(entry["sha"], GIT_SHA, entry["path"])

    def test_toolchain_script_downloads_from_an_https_base(self):
        """It has no sha256 pin; this asserts the pin it does have (the git tree) is https."""
        source = (ROOT / "research/fetch_toolchain.py").read_text()
        self.assertIn('base = "https://raw.githubusercontent.com/', source)
        self.assertNotIn("http://", source)


if __name__ == "__main__":
    unittest.main()
