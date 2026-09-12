# Branch protection

`main` is where the evidence contracts live: `research/container_baseline.json` (all 2,244
suite models), the 130-row board ledger, the cost-model baseline and the coverage table in
[verification.md](verification.md). A change merged without a review or with a red gate makes
one of those claims false, so `main` is protected by a repository **ruleset** whose required
checks are the CI jobs. The checks are code in
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml),
[`.github/workflows/docs.yml`](../.github/workflows/docs.yml) and
[`.github/workflows/mutation.yml`](../.github/workflows/mutation.yml); this page is the state
of the protection, why each gate exists, the exact settings to apply, and how to reproduce
every gate on a laptop.

Applying or changing the ruleset needs **repository-admin** rights
(Settings → Rules → Rulesets). Nothing here is enforced by a file in the repository: a
workflow cannot protect its own branch, and a required check that no job reports blocks every
pull request, which is why the contexts below must match the job names exactly.

## Required checks on `main`

Every row is a check that must pass before a pull request can merge. The `host` matrix runs
the whole Python suite; the other gates run inside `host (3.12)` so a single job carries the
evidence contracts and the matrix stays three jobs deep.

| Check (context) | Job | What it proves | Why it must block |
| --- | --- | --- | --- |
| `host (3.10)` | `ci` / `host` | the suite on the oldest supported interpreter | `requires-python >= 3.10` in `pyproject.toml`; a newer-only construct would otherwise fail at a user's first run |
| `host (3.12)` | `ci` / `host` | the suite on the release interpreter, plus every evidence gate below | this is the interpreter the published wheel is built and smoke-tested on |
| `host (3.13)` | `ci` / `host` | the suite on the newest tested interpreter | catches removals and deprecations ahead of the next release |
| `runtime` | `ci` / `runtime` | `make host-c`: the runtime, every `tests/board_*.c` harness and the host loader compile with `-Werror`; the `_Static_assert` ABI block is evaluated | CI used to build no C at all, so a syntax error in shipped code could reach users |
| `wheel` | `ci` / `wheel` | the built wheel is installed in a clean venv, imported from `site-packages` (not the checkout) and used to compile/decode a model | proves the distribution works as installed, not only in-tree |

### What runs inside `host (3.12)`

| Step in `ci.yml` | Contract it enforces | Local reproduction |
| --- | --- | --- |
| tests | the 1,008-test host suite | `make test` |
| coverage floor | compiler line coverage stays above `fail_under = 99` | `make coverage` |
| coverage table is current | the uncovered-line table in [verification.md](verification.md) matches the coverage data | `python research/coverage_doc_table.py --check` |
| C loader + runner under ASan/UBSan | `tests.test_host_loader` and `tests.test_runtime_cli` run with `-fsanitize=address,undefined` and `detect_leaks=1` | `OPEN_RKNPU_HOST_CFLAGS='-O1 -g -std=gnu99 -Wall -Wextra -Werror -D_GNU_SOURCE -Iruntime -fsanitize=address,undefined -fno-omit-frame-pointer' ASAN_OPTIONS=detect_leaks=1 PYTHONPATH=src python -m unittest tests.test_host_loader tests.test_runtime_cli` |
| published evidence integrity | every retained suite agrees with its manifest, references, board results and README | `make evidence` |
| container baseline | all 2,244 published models still compile to identical bytes, or stay deliberately rejected | `make baseline` |
| campaign sweep | the largest suites recompile to their published containers; the 12 known drifts stay known | `make campaign` |
| cost-model regression | tasks, engine blocks, registers, arena and payload do not inflate | `make perf` |
| distribution + reproducible sdist | `python -m build` + `twine check --strict`, then the sdist is normalised, audited and built twice byte-identically | `make reproducible` |
| low-level op examples | every `examples/primitives/` script runs and matches its reference | `make primitives` |
| documentation links | 0 broken relative links and 0 unresolved anchors | `make docs-check` |

### Post-merge and scheduled gates

| Gate | Trigger | What it proves |
| --- | --- | --- |
| `build site` (`docs` workflow) | push to `main` touching `docs/**` or `mkdocs.yml` | `mkdocs build --strict` still builds the site |
| `mutation` (`mutation` workflow, Mondays 04:23 UTC) | `schedule` + `workflow_dispatch` | the configured test subsets still kill the injected compiler faults, `--fail-under 50`; a ratchet, not a merge gate (see [mutation-testing.md](mutation-testing.md)) |

`build site` is **not** in the ruleset today for a concrete reason: `docs.yml` has no
`pull_request` trigger, so on a pull request the check never reports and requiring it would
block every merge. To promote it to a real PR gate, add `pull_request:` to the `on:` block of
[`docs.yml`](../.github/workflows/docs.yml) and then add `"build site"` to the contexts
below. The mutation job is deliberately scheduled: it takes ~15 minutes and is a weekly
ratchet against test-strength erosion, not a per-PR budget.

## The ruleset to apply

Settings → Rules → Rulesets → **New branch ruleset**; target the default branch `main`,
enforcement **Active**, no bypass actors. The same configuration can be applied with the
GitHub API, which is the exact payload an admin would POST to
`repos/dnhkng/open-rknpu/rulesets`:

```json
{
  "name": "protect-main",
  "target": "branch",
  "enforcement": "active",
  "conditions": {
    "ref_name": {
      "include": ["~DEFAULT_BRANCH"],
      "exclude": []
    }
  },
  "bypass_actors": [],
  "rules": [
    { "type": "deletion" },
    { "type": "non_fast_forward" },
    { "type": "required_linear_history" },
    {
      "type": "required_status_checks",
      "parameters": {
        "strict_required_status_checks_policy": true,
        "do_not_enforce_on_create": false,
        "required_status_checks": [
          { "context": "host (3.10)" },
          { "context": "host (3.12)" },
          { "context": "host (3.13)" },
          { "context": "runtime" },
          { "context": "wheel" }
        ]
      }
    },
    {
      "type": "pull_request",
      "parameters": {
        "required_approving_review_count": 0,
        "dismiss_stale_reviews_on_push": true,
        "require_code_owner_review": false,
        "require_last_push_approval": false,
        "required_review_thread_resolution": true
      }
    }
  ]
}
```

Notes on the choices:

* **`required_approving_review_count: 0`** because the repository has one maintainer and
  GitHub never counts an author's own approval; a required approval would deadlock every
  change. Raise it to 1 in the same pass that adds a second reviewer. The other review
  settings are on already, so an unresolved thread still blocks the merge.
* **`strict_required_status_checks_policy: true`** requires the PR branch to be up to date
  with `main`; this is deliberate, because the baseline and ledger are line-oriented files
  where a stale merge is how a wrong hash gets in.
* **`required_linear_history`** keeps the baseline/ledger diffs bisectable; this repository
  does not use merge commits.
* **`deletion` + `non_fast_forward`** make the protected history append-only.
* If a job is renamed, the context string changes and the old context silently stops gating
  while never reporting again: update the JSON in the same commit as the workflow edit.

## When a gate fails

A red `main` is the top priority; do not merge around it and do not re-run the job hoping for
a different answer. Every gate is deterministic (fixed seeds, no network, no board in CI), so
a red result means a real change.

| Failing check | What it means | First action |
| --- | --- | --- |
| `host (3.10/3.12/3.13)` | a parser, reference or rejection message changed | `make test`; a new `ERR:` or a new acceptance is a behaviour change, not a flake |
| coverage floor | a branch or guard lost its test | `make coverage`, then test the line or prove it unreachable and regenerate the table |
| coverage table | the docs table and the coverage data disagree | `python research/coverage_doc_table.py` and commit the regenerated table |
| C loader ASan/UBSan | a memory or undefined-behaviour bug in shipped C | reproduce with the exact environment line above; fix the C, never silence the sanitizer |
| `runtime` | the runtime or a harness no longer compiles | `make host-c`; `-Werror` failures are portability bugs |
| `wheel` | the installed distribution is broken | build the wheel, install it in a clean venv, and reproduce the import/CLI failure there |
| evidence integrity | a suite's manifest, references, board results and README disagree | `make evidence`; restore the missing artifact rather than the assertion |
| container baseline | a container changed, or a rejected model started compiling | `make baseline`; a real change needs fresh board evidence and an intentional `--update` in the same commit |
| campaign sweep | the published bytes and a fresh compile diverge | `make campaign`; a new drift is a serializer change, treat it as a baseline change |
| cost-model regression | the compiler emits more tasks/registers/arena | `make perf`; a cost regression is a compile-quality regression |
| reproducible sdist | the sdist carries something new or builds differently | `make reproducible`; the audit lists which file changed |
| primitives / docs links | an example broke, or a link or anchor rotted | `make primitives`, `make docs-check` |
| `build site` | `mkdocs build --strict` failed after a docs push | build the site locally and fix the page; `make docs-check` covers links, not the mkdocs build |
| `mutation` | the weekly ratchet dropped below 50 % | read the survivors in [mutation-testing.md](mutation-testing.md); a survivor is a test gap or a justified boundary |

## Reproducing every gate locally

All of these are host-only: no network, no board, no vendor SDK. They are the same commands
[CONTRIBUTING.md](../CONTRIBUTING.md) asks for before a pull request, and the exact recipes
for the evidence gates are in [verification.md](verification.md#every-gate-and-what-it-proves).

| Command | Gate |
| --- | --- |
| `make lint` | ruff on the maintained paths (`E9` + `F`, 120 columns) |
| `make test` | `host (3.10/3.12/3.13)` |
| `make coverage` | coverage floor, 99 % |
| `make host-c` | `runtime` |
| `make primitives` | low-level op examples |
| `make docs-check` | documentation links and anchors |
| `make evidence` | published evidence integrity |
| `make baseline` | container baseline (2,244 models) |
| `make campaign` | campaign sweep |
| `make perf` | cost-model regression |
| `make reproducible` | reproducible sdist and distribution audit |
| `make mutation-quick` | the bounded mutation smoke run (the weekly gate runs the full `--fail-under 50`) |

Hardware gates (`make board-io`, `make board-suite`) need the RV1103 board and the cross
toolchain and are not CI checks; the board evidence is inspected by `make evidence` instead.
The checklist row this page closes is A5 in
[publish-checklist.md](publish-checklist.md#ci-automation-community).
