"""Board run for the 1x1 height-strip tiled chain (docs/plans/pipelining-plan.md S3/S7).

Regenerates `tiled_chain_probe/model_tiles{1,2,4}.bin` (serial) and their one-job
batched twins from `deep_chain_suite/model002.onnx`, then runs every variant with
`tests/board_bench.c` against the untiled chain's own expected bytes (16 cases per
variant). The container the directory held before the S7 pad/activation fix is kept as
`model_tiles2_pre_s7.bin` and run as the retained counter-example.

Evidence: `board_results.txt` (human) and `board_results.json` (machine, checked by
`tests/test_tiled_chain.py`).
"""
from pathlib import Path
import os
import argparse
import json
import re
import shutil
import subprocess
import sys
import time

import onnx

ROOT = Path(__file__).resolve().parent
SUITE = ROOT / "tiled_chain_probe"
DEEP = ROOT / "deep_chain_suite"
ADB = os.environ.get("ADB", "adb")
SERIAL = "498063e3262e55b7"
REMOTE = "/userdata/open-npu-research/tiled_chain"
LINE = re.compile(r"runs=(\d+) tasks=(\d+) (\S+) min=([\d.]+)us median=([\d.]+)us "
                  r"mean=([\d.]+)us max=([\d.]+)us unstable=(\d+) mismatches=(\d+)")


def adb(*args, timeout=180):
    return subprocess.run([ADB, "-s", SERIAL, *args], capture_output=True, text=True,
                          timeout=timeout)


def rebuild():
    sys.path.insert(0, str(ROOT.parent / "src"))
    from open_rknpu.sequence import decode_sequence
    from open_rknpu.tiled_chain import compile_tiled_chain
    model = onnx.load(DEEP / "model002.onnx")
    if not (SUITE / "model_tiles2_pre_s7.bin").exists() and (SUITE / "model_tiles2.bin").exists():
        shutil.copyfile(SUITE / "model_tiles2.bin", SUITE / "model_tiles2_pre_s7.bin")
    # The probe shares the untiled suite's input/expected bytes, so refresh them: a
    # rebuilt deep_chain_suite changes the reference (docs/plans/pipelining-plan.md S9).
    shutil.copyfile(DEEP / "input002.u8", SUITE / "input.u8")
    shutil.copyfile(DEEP / "expected002.i8", SUITE / "expected.i8")
    manifest = []
    for tiles in (1, 2, 4):
        for serial in (True, False):
            suffix = "" if serial else "_batched"
            data, meta = compile_tiled_chain(model, tiles=tiles, serial=serial)
            (SUITE / f"model_tiles{tiles}{suffix}.bin").write_bytes(data)
            manifest.append(dict(name=f"tiles{tiles}{suffix}", tiles=tiles,
                                 tasks=decode_sequence(data)["task_count"],
                                 mode="serial" if serial else "batched",
                                 layout=meta["layout"], arena_bytes=meta["arena_bytes"]))
    (SUITE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def bench(model, iterations):
    adb("push", str(model), "%s/t.bin" % REMOTE)
    text = ""
    for _ in range(3):
        run = adb("shell", "%s/board_bench %s/t.bin %s/t.u8 %d %s/t.i8"
                  % (REMOTE, REMOTE, REMOTE, iterations, REMOTE))
        text = (run.stdout + run.stderr).strip()
        if text:
            break
        time.sleep(6)
    match = LINE.search(text)
    if not match:
        return dict(passed=False, output=text[:160])
    return dict(passed=match.group(8) == "0" and match.group(9) == "0",
                tasks=int(match.group(2)), mode=match.group(3),
                min_us=float(match.group(4)), median_us=float(match.group(5)),
                unstable=int(match.group(8)), mismatches=int(match.group(9)))


def push_case(case, size):
    """One input/expected slice per call: `board_bench` reads a single case."""
    source = (SUITE / "input.u8").read_bytes()
    expected = (SUITE / "expected.i8").read_bytes()
    (Path("/tmp") / "tiled_case.u8").write_bytes(source[case * size:(case + 1) * size])
    (Path("/tmp") / "tiled_case.i8").write_bytes(expected[case * size:(case + 1) * size])
    adb("push", "/tmp/tiled_case.u8", "%s/t.u8" % REMOTE)
    adb("push", "/tmp/tiled_case.i8", "%s/t.i8" % REMOTE)


def run_all(model, cases, size, iterations):
    """Every input case, so the evidence is not one zero-input sample."""
    runs = []
    for case in range(cases):
        push_case(case, size)
        runs.append(bench(model, iterations))
    passed = all(run["passed"] for run in runs)
    return dict(passed=passed, cases=len(runs), iterations=iterations,
                detail=next((run.get("output", "") for run in runs if not run["passed"]), ""),
                tasks=runs[0].get("tasks"), mode=runs[0].get("mode"),
                min_us=min(r.get("min_us", 0) or 0 for r in runs),
                median_us=max(r.get("median_us", 0) or 0 for r in runs),
                unstable=sum(r.get("unstable", 0) or 0 for r in runs),
                mismatches=sum(r.get("mismatches", 0) or 0 for r in runs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument("--expect-bytes", type=int, default=192)
    parser.add_argument("--harness", default="/tmp/board_bench")
    args = parser.parse_args()
    manifest = rebuild()
    adb("shell", "mkdir -p %s" % REMOTE)
    adb("push", args.harness, REMOTE + "/board_bench")
    results, lines = [], []
    cases = [("pre-s7", SUITE / "model_tiles2_pre_s7.bin")] + \
            [(entry["name"], SUITE / f"model_{entry['name']}.bin") for entry in manifest]
    for name, model in cases:
        record = dict(variant=name, **run_all(model, args.cases, args.expect_bytes,
                                              args.iterations))
        results.append(record)
        line = ("%-16s tasks=%2s %-7s cases=%2d exact=%s min=%7.1fus median=%8.1fus "
                "mismatches=%d" % (name, record["tasks"], record["mode"], record["cases"],
                                   record["passed"], record["min_us"], record["median_us"],
                                   record["mismatches"]))
        lines.append(line)
        print(line, flush=True)
        (SUITE / "board_results.json").write_text(json.dumps(results, indent=2) + "\n")
    header = ("1x1 height-strip tiled chain (docs/plans/pipelining-plan.md S3, re-run for the S7 "
              "pad/activation fix), tests/board_bench.c, %d board cases per variant (every "
              "input in the file, not just the first), same input/expected bytes as the "
              "untiled deep_chain_suite/model002 container; `pre-s7` is the container the "
              "directory held before the fix:" % args.cases)
    (SUITE / "board_results.txt").write_text(header + "\n\n" + "\n".join(lines) + "\n")
    fixed = [r for r in results if r["variant"] != "pre-s7"]
    print("\n".join(lines))
    if not all(r["passed"] for r in fixed):
        sys.exit(1)


if __name__ == "__main__":
    main()
