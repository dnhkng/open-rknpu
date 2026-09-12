#!/usr/bin/env python3
"""SPDX-License-Identifier: MIT

Keep the uncovered-line table in `docs/verification.md` truthful.

Coverage floors rot in a specific way: the percentage in the docs stays roughly right while
the list of lines behind it becomes fiction. This script regenerates the table between the
`coverage-table:start` / `coverage-table:end` markers from the actual coverage data, so the
published claim ("98%, and here is every line that is not executed") cannot drift.

It reads `.coverage` when one exists, otherwise it runs the suite under coverage first. Each
line gets its source text plus the reason from `REASONS` (the module-level justification);
a reason that is wrong will be obvious next to the quoted source, which is the point.

    PYTHONPATH=src python3 research/coverage_doc_table.py            # rewrite the table
    PYTHONPATH=src python3 research/coverage_doc_table.py --check    # fail on drift
    PYTHONPATH=src python3 research/coverage_doc_table.py --run      # re-measure first
"""
from pathlib import Path
import argparse
import subprocess
import sys

import coverage

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "verification.md"
START = "<!-- coverage-table:start -->"
END = "<!-- coverage-table:end -->"
SOURCE = ROOT / "src"

# Why each module's remaining lines are not executed. Keep these honest: the table quotes the
# source next to the reason, so a claim that "the caller rejects it first" can be checked.
REASONS = {
    "accuracy.py": "shape/label guards; the public helpers are always called with matching arrays",
    "calibration.py": "reached only for a multi-input model, which the scheduler rejects before calibration",
    "chain_n.py": "graph-shape guards; dispatch only calls this profile on a matched chain",
    "cli.py": "argument validation; the CLI tests pass valid values on these paths",
    "compose.py": "external-overlaps-internal guard; inputs end before the arena and internals start at it",
    "depthwise.py": "shape guards superseded by the scheduler's own checks before this emitter runs",
    "elementwise.py": "per-channel broadcast guard, already excluded two lines earlier by a successful broadcast",
    "elementwise_chain.py": "rejection paths for DAG shapes the composer never routes here",
    "elementwise_multi.py": "rejection paths for DAG shapes the composer never routes here",
    "graph.py": "join-structure guards for node arrangements the matcher excludes",
    "join_dag.py": "band-commit guards that the branch selection excludes by construction",
    "liveness.py": "placement guard; the aligned end of the placed tensors is always a valid candidate",
    "lut.py": "LUT graph guards for stems the profile matcher rejects earlier",
    "native.py": "prequantized-weights guard; the importer always supplies plain float weights here",
    "normalize.py": "bias-folding tie path; the published models never hit the exact rounding tie",
    "padding.py": "rank guard; every caller passes HWC or NHWC",
    "pooling.py": "staged-pooling guards for forms the reduction matcher rejects first",
    "quantization.py": "overflow and multiplier-range guards for bands the profiles never emit",
    "reduction.py": "staged-pooling guards for forms the matcher rejects first",
    "scheduler.py": "quantization-boundary rejections (UINT8 scale 1 / zero point 0) that the accepted profiles satisfy by construction",
    "sequence.py": "defensive guards in the encoder; the decode round-trip after encoding rejects malformed input first",
    "strided.py": "geometry guard for a Conv form the matcher excludes",
    "tiled_chain.py": "geometry guards for kernels and channel counts the chain profile excludes",
    "transposed.py": "ConvTranspose attribute guards for forms the ONNX matcher excludes",
    "walk.py": "chain-start and channel-limit guards; the matcher rejects those graphs first",
}


# A few lines are provably unreachable in the current tree rather than merely untested.
# The reason is per line because the general module reason would hide the argument, and the
# argument is the interesting part: each one names the earlier guard or placement rule that
# makes the line impossible to reach, so a reviewer can check it and see when it stops being
# true (at which point the line becomes live and should get a test).
LINE_REASONS = {
    "graph.py:473": "diamond heads overlap until the join, so the allocator places them adjacent by construction",
    "graph.py:598": "the matcher only records Conv heads reading the stem output, and the emitter rechecks the same condition",
    "graph.py:846": "externals are placed outside the internal span by policy, so the overlap test cannot be true",
    "join_dag.py:347": "the free operand is by definition not committed, and every uncommitted produced tensor is a by_name key",
    "join_dag.py:350": "same branch as 347: the free operand is never in the committed set",
    "join_dag.py:471": "offsets are laid by the cursor loop after input0 and the output after the last end, so an overlap cannot occur",
    "join_dag.py:529": "_prepare recompiles every branch final and a depthwise entry can only be a single-layer branch final, so recompiled is never None",
    "liveness.py:165": "first-fit always finds a slot; an exhaustive and randomized search found no counterexample",
    "compose.py:250": "externals are placed outside the internal span by construction",
    "scheduler.py:98": "no in-tree emitter returns a legacy ORNPUBIN any more; the seam is pinned with a real container by test_scheduler_boundaries.py",
}


def measure(run):
    """Return (total_statements, [missing lines], {file: (missing, source lines)})."""
    data_file = ROOT / ".coverage"
    if run or not data_file.exists():
        result = subprocess.run(
            [sys.executable, "-m", "coverage", "run", "--source=src/open_rknpu",
             "-m", "unittest", "discover", "-s", "tests"],
            cwd=ROOT, capture_output=True, text=True, timeout=1800,
        )
        if result.returncode != 0:
            raise SystemExit(f"the test run failed:\n{result.stdout[-4000:]}\n{result.stderr[-2000:]}")
    cov = coverage.Coverage(data_file=str(data_file), source=[str(SOURCE / "open_rknpu")])
    cov.load()
    data = cov.get_data()
    total = 0
    missing = 0
    files = {}
    for path in sorted(data.measured_files()):
        analysis = cov.analysis2(path)
        statements, gaps = analysis[1], analysis[3]
        total += len(statements)
        if not gaps:
            continue
        source = Path(path).read_text().splitlines()
        files[Path(path).name] = (gaps, source)
        missing += len(gaps)
    return total, missing, files


def table(total, missing, files):
    covered = 100.0 * (total - missing) / total
    lines = [
        START,
        f"Coverage is {covered:.2f}% ({total:,} statements, {missing} uncovered lines in "
        f"{len(files)} of {len(list((SOURCE / 'open_rknpu').glob('*.py')))} modules). Every "
        "remaining line is listed here because a floor nobody can explain is useless; the "
        "reason column is the module-level justification and the quoted source is there to "
        "check it against.",
        "",
        "| Line | Source | Why it is not executed |",
        "| --- | --- | --- |",
    ]
    for name in sorted(files):
        gaps, source = files[name]
        for number in gaps:
            text = source[number - 1].strip().replace("|", "\\|")
            if len(text) > 96:
                text = text[:93] + "..."
            reason = LINE_REASONS.get(
                f"{name}:{number}",
                REASONS.get(name, "defensive guard for input the profiles do not emit"))
            lines.append(f"| `{name}:{number}` | `{text}` | {reason} |")
    lines.append(END)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail instead of rewriting")
    parser.add_argument("--run", action="store_true", help="re-run the suite under coverage first")
    args = parser.parse_args()

    total, missing, files = measure(args.run)
    block = table(total, missing, files)
    document = DOC.read_text()
    if START not in document or END not in document:
        raise SystemExit(f"{DOC} has no {START} / {END} markers")
    head, rest = document.split(START, 1)
    _old, tail = rest.split(END, 1)
    updated = head + block + tail

    if args.check:
        if updated != document:
            print("docs/verification.md coverage table is stale; "
                  "run: PYTHONPATH=src python3 research/coverage_doc_table.py", file=sys.stderr)
            return 1
        print(f"coverage table current: {total - missing}/{total} lines, {missing} uncovered")
        return 0

    DOC.write_text(updated)
    print(f"rewrote the table in {DOC.relative_to(ROOT)}: {total - missing}/{total} lines, "
          f"{missing} uncovered in {len(files)} modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
