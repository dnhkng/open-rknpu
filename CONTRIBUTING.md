# Contributing

Thanks for looking. This project has an unusual constraint: **it is an evidence-driven
reverse-engineering effort**, so the review bar is "show me the bytes", not "the tests
pass".

## Setup

```sh
python -m pip install -e ".[dev]"     # NumPy, ONNX, pytest, ruff
make test                             # the host test suite
make coverage                         # tests under coverage, enforcing the floor
make lint                             # ruff on the maintained paths
```

No vendor SDK, no RKNN packages and no Rockchip binaries are required for anything under
`src/`, `tests/` or `examples/primitives/`. Hardware work additionally needs an RV1103/RV1106
board and an ARM uClibc cross toolchain (see [board](docs/board.md)).

## Rules of the road

1. **No vendor artifacts.** Never add `librknn*.so`, decompiled vendor sources, RKNN
   models, GPL kernel sources or captured binaries to the repository. The vendor library is
   an *oracle* used during reverse engineering; the compiler must not read it, and the
   repository must not ship it. Provenance notes are welcome in `research/`.
2. **No silent fallbacks.** A profile that cannot handle a graph must raise a specific
   `ValueError`. Accepting a shape and emitting a wrong program is the worst possible bug
   here.
3. **Exactness over tolerance.** Every profile ships a Python integer reference next to its
   emitter. Containers are compared byte-for-byte on the board; float tolerances are only
   used for the *model* accuracy story, never as hardware evidence.
4. **Evidence, not claims.** A new capability lands with: a generator
   (`research/build_*.py`), a suite with `board_results_*.json` and `board_summary.txt`, a
   ledger row, and the reproduce recipe in the suite `README.md`.
5. **Pin rejections too.** `research/container_baseline.json` records `ERR:<Type>` for
   models a profile deliberately refuses, so "it started compiling that" is also a test
   failure. Update the baseline only when a container change is intended.
6. **Keep the docs honest.** If a bound changes, update
   [primitives](docs/primitives.md), the ledger and the investigation log in the
   same change. `research/check_docs_links.py` must stay green.

## Style

* ruff with the repository config (`E9` + `F`, 120 columns). The legacy scripts under
  `research/` are exempt (they are experiment tooling); everything else is linted:
  `ruff check src tests examples research/verify_suites.py research/campaign_sweep.py
  research/check_docs_links.py research/probe_conv_envelope.py`.
* Dense, compact code is the house style — the register emitters are tables of fields and
  benefit from staying readable as tables. Do not run a wholesale formatter over them.
* MIT SPDX header/docstring at the top of every new file.
* Documentation: a short statement of *what is verified* and *how*, plus the exact
  reproduce command.

## Adding a primitive

Follow the eight-step recipe in
[architecture](docs/architecture.md#adding-a-primitive) and the evidence
requirements in
[verification](docs/verification.md#adding-evidence-for-a-new-primitive).

## Pull requests

* One capability per PR; keep unrelated refactors out.
* Include the output of `make test`, `make baseline` and, for hardware work, the board
  summary line from `board_summary.txt`.
* If you touched an emitter, say whether the container baseline is expected to be
  unchanged, and why.
* If you moved a file, re-run `make docs-check` — broken relative links are a test failure.

## URLs and metadata

`pyproject.toml` contains placeholder project URLs and no author list. Update them on your
fork before publishing a release.
