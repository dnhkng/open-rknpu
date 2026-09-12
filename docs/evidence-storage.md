# Evidence storage

The project's claim is that every container in `research/` was compiled once, recorded,
and run on a real RV1103. That makes the evidence tree part of the product, not a local
scratch directory, and it has to be stored somewhere a reader can audit. This page records
the decision (checklist A6), the measurements behind it, and the trigger that would
change it.

## Decision

**The evidence stays in the Git repository.** No Git LFS, no external archive, no
download-on-demand for the per-suite containers, references, board results, manifests and
READMEs. The compiler package on PyPI is separate and deliberately lean; the repository is
where the proof lives.

## Measurements (2026-09-12, `main`)

| Quantity | Value | Limit that matters |
| --- | --- | --- |
| Tracked files | 16,517 | GitHub warns on repositories with very many files, but no hard count limit |
| Packed repository (`.git`) | 38.5 MiB | GitHub warns above 1 GiB, blocks above 5 GiB |
| Working tree, tracked contents | ~20 MB | - |
| Largest tracked file | 1.9 MB (`research/accumulator_boundary_suite/manifest.json`) | GitHub blocks files above 100 MB, warns above 50 MB |
| Evidence tree (`research/`) | 16,217 files, ~14 MB | - |
| Test suite (`tests/`) | 97 files, ~1.5 MB | - |
| Examples (`examples/`) | 99 files, ~3.8 MB | - |

The pack is three orders of magnitude below the point where GitHub asks a repository to
move binaries out of history, and the largest file is 26x below the warning threshold.

## What is *not* stored

The size above is only possible because the bulky, licence-encumbered or reproducible
inputs are excluded, and each is replaced by something a reader can run:

| Excluded | Why | Where it lives |
| --- | --- | --- |
| Cross toolchain (`research/toolchain/`, 81 MB) | Rockchip's re-distribution terms | `research/fetch_toolchain.sh`, pinned by sha256 |
| Datasets (FSDD recordings, fashion/MNIST derivatives) | Licence and size | `research/fetch_fsdd.sh`, `examples/*/build.py` |
| Vendor binaries, decompiled sources, Ghidra project | Not redistributable | `THIRD_PARTY.md`, `docs/provenance.md` |
| Rockchip documents (`hardware_refs/*.pdf`) | Copyright | referenced by title and section only |
| Build outputs (`dist/`, `build/`, `examples/*/build/`) | Reproducible | rebuilt by `make wheel`, `examples/*/build.py` |

`research/container_baseline.json` (2,341 sha256 or `ERR:` entries), the 130-row board
ledger and all 183 suite READMEs stay in-tree: they are small text, and they are the index
that makes the binaries meaningful.

## Why not Git LFS

LFS would move the `.bin`/`.onnx`/`.u8`/`.i8` files behind a pointer store that
`git clone` does not fetch by default. The verification sweep
(`research/verify_suites.py`), the campaign sweep, the board replay scripts and
`tests/test_evidence_integrity.py` all read those files directly, so every clone of the
published repository would need `git lfs pull` before it could reproduce anything - and a
reader without the LFS client would see empty pointer files instead of a verifiable
container. At 38.5 MiB there is nothing to gain.

## The trigger to revisit

Move the per-suite artifacts out of Git, keep the manifests and READMEs in-tree, and add a
fetch script, when **any** of these becomes true:

* the pack grows beyond **500 MiB** (roughly 6x today's evidence, e.g. a much larger
  model zoo or per-sample fixture corpora);
* a single tracked file approaches **50 MiB**;
* a dataset that cannot be regenerated or fetched under a clear licence is needed for a
  suite.

The migration is mechanical because the artifacts are already addressed by relative path
from the manifests: a fetch script would reconstruct `research/*_suite/*.bin|onnx|u8|i8`
from release assets, and `research/verify_suites.py --update` plus
`tests/test_evidence_integrity.py` would fail loudly if a fetched file were wrong. Do it in
a dedicated commit that keeps the current pack reachable in history for at least one
release.

## How the decision is enforced

* `tests/test_evidence_integrity.py` checks that every suite keeps its README, manifest,
  containers, inputs, expected outputs and board results, and that the recorded numbers
  agree with the files on disk - so the tree cannot be trimmed silently.
* `research/build_suite_readmes.py --check` regenerates the suite pages from the
  manifests and fails on drift.
* `.gitignore` keeps the excluded inputs out; `.gitattributes` marks the binary evidence
  as binary so diffs stay readable.
* `research/check_reproducible_build.py` audits the opposite direction: nothing from
  `research/` may leak into the published sdist or wheel.
