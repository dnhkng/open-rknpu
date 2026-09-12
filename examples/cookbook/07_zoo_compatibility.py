"""SPDX-License-Identifier: MIT

Cookbook 7/7: a bounded compatibility scan over the ONNX files already in the tree.

No network: the script walks `research/pretrained/**/*.onnx`, one `model000.onnx`
per `research/*_suite/` and `research/generated/*.onnx`, and tries
`compile_sequence` on each. The walk is bounded (a fixed candidate cap, a size
cap and a wall-clock assertion) so it stays fast - the tree holds thousands of
regression models, and scanning them all is what `tests/` and `make` are for.

The compatibility table lists each model with its op set and either the profile
it compiled to or the exact rejection reason. The script asserts the table is
non-empty, that at least one model compiles and that at least one is rejected -
a scan that suddenly accepts everything (or nothing) is a red flag, not a result.

Run:
    PYTHONPATH=src python examples/cookbook/07_zoo_compatibility.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import onnx

from cookbook_common import build_dir, profile_of
from open_rknpu.scheduler import compile_sequence
from open_rknpu.sequence import decode_sequence

FOLDER = build_dir("07_zoo_compatibility")
REPO = Path(__file__).resolve().parents[2]
CAP = 16
MAX_BYTES = 4 * 1024 * 1024
TIME_BUDGET_S = 60.0


def candidates():
    """A deterministic, bounded list of `.onnx` files already in the tree."""
    research = REPO / "research"
    ordered = sorted((research / "pretrained").rglob("*.onnx"))
    ordered += sorted(research.glob("*_suite/model000.onnx"))
    ordered += sorted((research / "generated").glob("*.onnx"))
    selected, skipped = [], 0
    for path in ordered:
        if len(selected) >= CAP:
            break
        if not path.is_file() or path.stat().st_size > MAX_BYTES:
            skipped += 1
            continue
        selected.append(path)
    return selected, skipped


def scan():
    """Compile every candidate and record `(path, ops, outcome)`."""
    selected, skipped = candidates()
    assert selected, "no .onnx files found under research/ (run from the repository tree)"
    rows = []
    for path in selected:
        model = onnx.load(path)
        ops = sorted({node.op_type for node in model.graph.node})
        try:
            binary, meta = compile_sequence(model)
            outcome = "%s (%d B, %d task)" % (profile_of(meta), len(binary),
                                              decode_sequence(binary)["task_count"])
            rows.append(dict(path=path, ops=ops, compiled=True, outcome=outcome, profile=profile_of(meta)))
        except ValueError as error:
            rows.append(dict(path=path, ops=ops, compiled=False, outcome="rejected: %s" % error,
                             profile=None))
    return rows, skipped


def main():
    started = time.monotonic()
    rows, skipped = scan()
    elapsed = time.monotonic() - started
    compiled = [row for row in rows if row["compiled"]]
    rejected = [row for row in rows if not row["compiled"]]
    assert rows, "empty compatibility table"
    assert compiled, "no model in the bounded scan compiled"
    assert rejected, "no model in the bounded scan was rejected (the boundary may have moved)"
    assert elapsed < TIME_BUDGET_S, "the bounded scan took %.1fs" % elapsed

    lines = ["model\tops\tresult"]
    print("  bounded scan: %d candidates (cap %d, size cap %d B, %d skipped), %.2fs" %
          (len(rows), CAP, MAX_BYTES, skipped, elapsed))
    for row in rows:
        relative = row["path"].relative_to(REPO)
        ops = ",".join(row["ops"]) or "(no nodes)"
        if len(ops) > 40:
            ops = ops[:37] + "..."
        lines.append("%s\t%s\t%s" % (relative, ops, row["outcome"]))
        print("  %-46s %-30s %s" % (relative, ops, row["outcome"]))
    (FOLDER / "compatibility.txt").write_text("\n".join(lines) + "\n")
    print("07_zoo_compatibility: %d compiled, %d rejected, %d scanned in %.2fs -> %s" %
          (len(compiled), len(rejected), len(rows), elapsed, FOLDER / "compatibility.txt"))


if __name__ == "__main__":
    main()
