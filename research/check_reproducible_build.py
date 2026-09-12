#!/usr/bin/env python3
"""SPDX-License-Identifier: MIT

Reproducible source distribution, and an audit of what it carries.

Two independent properties matter before a release is uploaded, and neither is checked
by `python -m build` alone:

1. **Reproducibility.** Building the sdist twice from the same tree must yield the same
   *contents*, and after normalisation the same *bytes*. setuptools stamps the files it
   generates during the build (root `PKG-INFO`, `setup.cfg`, `src/*.egg-info/*`) with the
   wall clock, so raw archives differ even though no source changed. `normalize_sdist`
   rewrites an archive with every timestamp pinned to `SOURCE_DATE_EPOCH`, the uid/gid
   zeroed, the mode fixed and the members sorted, which is what release tooling must
   upload. `--normalize` applies that rewrite in place.
2. **Contents.** The sdist is the *compiler*: the package, the libc-only runtime that
   ships as package data, and the licence/attribution files. The test suite, the
   documentation and the research evidence tree (hundreds of megabytes of board
   fixtures, ONNX models and containers) belong to the Git repository, not to PyPI. An
   accidental `graft` or a missing `prune` would upload them.

The script fails on non-reproducible contents, on a normalised byte difference, on a
missing required member, on any evidence/denylisted member, and on a member large enough
to be a fixture. When a wheel is present in the audited directory it gets the same audit.

    PYTHONPATH=src python3 research/check_reproducible_build.py
    PYTHONPATH=src python3 research/check_reproducible_build.py --outdir dist --normalize
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EPOCH = 1700000000            # fixed timestamp: the build must not depend on the clock
MAX_MEMBER_BYTES = 1 << 20    # nothing legitimate in the published compiler is this large
MAX_TOTAL_BYTES = 4 << 20
REQUIRED = (
    "PKG-INFO", "pyproject.toml", "README.md", "LICENSE", "THIRD_PARTY.md",
    "AUTHORS", "CHANGELOG.md", "CITATION.cff", "SECURITY.md", "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md", "MANIFEST.in", "setup.cfg",
    "runtime/open_rknpu.c", "runtime/open_rknpu.h", "runtime/main.c",
    "runtime/Makefile", "runtime/sequence_format.md",
)
REQUIRED_PREFIXES = ("src/open_rknpu/",)
WHEEL_RUNTIME = ("open_rknpu.c", "open_rknpu.h", "main.c", "Makefile", "sequence_format.md")
DENY_PARTS = ("research/", "tests/", "docs/", "examples/", "hardware_refs/",
              "toolchain/", "vendor/", "pretrained/", "__pycache__/", ".git/", "dist/")
DENY_SUFFIXES = (".onnx", ".bin", ".u8", ".i8", ".pyc", ".pyo", ".o", ".a", ".so", ".pdf")


def build_sdist(outdir):
    """Run the real PEP 517 sdist build; return the archive path."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, SOURCE_DATE_EPOCH=str(EPOCH))
    result = subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--outdir", str(outdir)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=900,
    )
    if result.returncode != 0:
        raise SystemExit(f"sdist build failed:\n{result.stdout}\n{result.stderr}")
    archives = sorted(outdir.glob("*.tar.gz"))
    if not archives:
        raise SystemExit(f"the build produced no sdist in {outdir}")
    return archives[-1]


def tar_members(archive):
    """`{name without the version prefix: (bytes, mtime)}`."""
    contents = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = member.name.split("/", 1)[1] if "/" in member.name else member.name
            contents[name] = (tar.extractfile(member).read(), member.mtime)
    return contents


def wheel_members(archive):
    contents = {}
    with zipfile.ZipFile(archive) as wheel:
        for name in wheel.namelist():
            if name.endswith("/"):
                continue
            contents[name] = (wheel.read(name), None)
    return contents


def normalize_sdist(source, destination):
    """Rewrite an sdist with pinned metadata so two builds are byte-identical."""
    with tarfile.open(source, "r:gz") as tar:
        members = [(member.name, tar.extractfile(member).read())
                   for member in tar.getmembers() if member.isfile()]
    with open(destination, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as out:
                for name, data in sorted(members):
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mtime = EPOCH
                    info.mode = 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.type = tarfile.REGTYPE
                    out.addfile(info, io.BytesIO(data))
    return destination


def wheel_required(archive):
    """The exact members the published wheel must carry, derived from the checkout."""
    stem = archive.name.rsplit(".whl", 1)[0].rsplit("-py", 1)[0]
    data = f"{stem}.data/data/share/open-rknpu/runtime/"
    required = [f"{stem}.dist-info/{name}" for name in
                ("METADATA", "WHEEL", "RECORD", "entry_points.txt", "top_level.txt")]
    required += [f"{stem}.dist-info/licenses/{name}" for name in ("LICENSE", "THIRD_PARTY.md")]
    required += [data + name for name in WHEEL_RUNTIME]
    required += [f"open_rknpu/{path.name}"
                 for path in sorted((ROOT / "src" / "open_rknpu").glob("*.py"))]
    return required


def audit(contents, required, prefixes=()):
    """Return (required_missing, denied, oversized, total_uncompressed_bytes)."""
    names = set(contents)
    missing = [name for name in required if name not in names]
    for prefix in prefixes:
        if not any(name.startswith(prefix) for name in names):
            missing.append(prefix + "**")
    denied = sorted(name for name in names
                    if any(part in name for part in DENY_PARTS)
                    or name.endswith(DENY_SUFFIXES))
    oversized = sorted(name for name, (data, _mtime) in contents.items()
                       if len(data) > MAX_MEMBER_BYTES)
    total = sum(len(data) for data, _mtime in contents.values())
    return missing, denied, oversized, total


def changed_members(first, second):
    """The members whose bytes differ between two builds (empty means contents match)."""
    digest = lambda data: hashlib.sha256(data).hexdigest()  # noqa: E731 (a local shorthand)
    if sorted(first) != sorted(second):
        only_first = sorted(set(first) - set(second))
        only_second = sorted(set(second) - set(first))
        return [f"member sets differ: only in first={only_first} only in second={only_second}"]
    return [f"{name}: bytes differ between two builds" for name in sorted(first)
            if digest(first[name][0]) != digest(second[name][0])]


def audit_archive(archive):
    """Audit one sdist or wheel; return (failures, contents)."""
    is_sdist = archive.name.endswith(".tar.gz")
    kind = "sdist" if is_sdist else "wheel"
    contents = tar_members(archive) if is_sdist else wheel_members(archive)
    if is_sdist:
        missing, denied, oversized, total = audit(contents, REQUIRED, REQUIRED_PREFIXES)
    else:
        missing, denied, oversized, total = audit(contents, wheel_required(archive))
    print(f"{kind} {archive.name}: members={len(contents)} bytes={total}")
    failures = [f"{kind}: missing {name}" for name in missing]
    failures += [f"{kind}: denied member {name}" for name in denied]
    failures += [f"{kind}: oversized member {name}" for name in oversized]
    if total > MAX_TOTAL_BYTES:
        failures.append(f"{kind}: total {total} bytes exceeds {MAX_TOTAL_BYTES}")
    return failures, contents


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", default=None,
                        help="audit this directory's sdist/wheel instead of building twice")
    parser.add_argument("--normalize", action="store_true",
                        help="rewrite each sdist in --outdir with pinned metadata")
    parser.add_argument("--keep", action="store_true", help="keep the built archives")
    args = parser.parse_args()

    if args.normalize and not args.outdir:
        raise SystemExit("--normalize requires --outdir")
    if importlib.util.find_spec("build") is None:
        raise SystemExit("the 'build' package is required (pip install build)")

    failures = []
    if args.outdir:
        outdir = Path(args.outdir)
        archives = sorted(outdir.glob("*.tar.gz")) + sorted(outdir.glob("*.whl"))
        if not archives:
            raise SystemExit(f"no sdist or wheel in {outdir}")
        for archive in archives:
            if args.normalize and archive.name.endswith(".tar.gz"):
                with tempfile.TemporaryDirectory(prefix="open-rknpu-norm-") as tmp:
                    normalized = Path(tmp) / archive.name
                    normalize_sdist(archive, normalized)
                    shutil.move(str(normalized), archive)
                print(f"normalized {archive.name}")
            found, _contents = audit_archive(archive)
            failures += found
    else:
        workdir = Path(tempfile.mkdtemp(prefix="open-rknpu-repro-"))
        try:
            first = build_sdist(workdir / "one")
            second = build_sdist(workdir / "two")
            one, two = tar_members(first), tar_members(second)
            print(f"sdist: {first.name} members={len(one)} bytes={first.stat().st_size}")
            differences = changed_members(one, two)
            if differences:
                failures.append("two builds of the same tree differ in content")
                failures += differences
            else:
                print("contents identical across two builds")
            normalized_first = normalize_sdist(first, workdir / "first-normalized.tar.gz")
            normalized_second = normalize_sdist(second, workdir / "second-normalized.tar.gz")
            if normalized_first.read_bytes() == normalized_second.read_bytes():
                print("REPRODUCIBLE: normalized sdists are byte-identical "
                      f"(sha256 {hashlib.sha256(normalized_first.read_bytes()).hexdigest()})")
            else:
                failures.append("normalized sdists are not byte-identical")
            found, _contents = audit_archive(first)
            failures += found
            if args.keep:
                target = ROOT / "dist"
                target.mkdir(exist_ok=True)
                shutil.copy2(normalized_first, target / first.name)
                print(f"kept normalized {target / first.name}")
        finally:
            if not args.keep:
                shutil.rmtree(workdir, ignore_errors=True)

    for line in failures:
        print(f"FAIL {line}", file=sys.stderr)
    print(f"checks={'fail' if failures else 'pass'} failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
