<!--
Thanks for the PR. Keep it to one capability, and keep unrelated refactors out.
The evidence bar is "show me the bytes" — see CONTRIBUTING.md.
-->

## What this changes

<!-- One or two sentences. Which primitive, profile, emitter or document? -->

## Why

<!-- The model or workflow this unblocks, or the bug it fixes. -->

## Checklist

- [ ] **Lint** — `make lint` is clean (ruff `E9`+`F`, 120 columns) on
      `src tests examples research/verify_suites.py research/campaign_sweep.py
      research/check_docs_links.py research/probe_conv_envelope.py`.
- [ ] **Tests** — `make test` passes, and new behaviour has a test. Exactness tests
      compare against an independent Python integer reference, not against the emitter.
- [ ] **Coverage** — `make coverage` still meets the floor (`fail_under = 95`).
- [ ] **Docs links** — `make docs-check` is green (`research/check_docs_links.py`).
      If I moved a file, I re-ran it.
- [ ] **Baseline note** — I ran `make baseline` / `PYTHONPATH=src python
      research/verify_suites.py`, and I state below whether
      `research/container_baseline.json` is expected to change and why. If a bound
      changed, `docs/primitives.md`, the ledger and `docs/investigation-log.md` are
      updated in this PR.
- [ ] **No vendor artifacts** — this PR adds no `librknn*.so`, decompiled vendor
      source, RKNN model, GPL kernel source or captured binary. Provenance notes go in
      `research/` instead.

## Evidence

<!--
  For a capability: the generator, the suite with board_results_*.json and
  board_summary.txt, the ledger row, and the reproduce command.
  For hardware work: paste the board summary line.
  For an emitter change: say what is byte-identical and what is not.
-->

## Container baseline

- [ ] Unchanged (no emitter touched, or the change provably cannot affect emitted bytes)
- [ ] Intentionally changed — described above and `research/container_baseline.json` updated
- [ ] Rejections changed (new `ERR:<Type>` rows) — described above
