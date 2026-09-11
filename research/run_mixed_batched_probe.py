"""S8 published-mode check: a mixed-engine DAG emitted as one linked job.

`open_rknpu.compose` now writes the measured engine hand-off control per transition
(`docs/plans/pipelining-plan.md` S8), so `compile_sequence(..., submission='batched')` emits a
mixed-engine container for graphs whose scheduled order is CNA runs followed by DPU
runs. This probe rebuilds those containers from the suites' own ONNX models, checks that
the serial compile still reproduces the retained (board-verified) container byte for
byte, and runs both modes on the board against the suite's expected bytes, case by case.

Evidence: `research/mixed_batched_probe/{board_results.json,summary.txt,README.md}`.
"""
from pathlib import Path
import os
import argparse
import json
import re
import subprocess
import sys
import time

import onnx

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "mixed_batched_probe"
OUT.mkdir(exist_ok=True)
ADB = os.environ.get("ADB", "adb")
SERIAL = "498063e3262e55b7"
REMOTE = "/userdata/open-npu-research/mixed_batched"
LINE = re.compile(r"runs=(\d+) tasks=(\d+) (\S+) min=([\d.]+)us median=([\d.]+)us "
                  r"mean=([\d.]+)us max=([\d.]+)us unstable=(\d+) mismatches=(\d+)")
MODELS = [("pool_join_suite", 0), ("pool_join_suite", 1),
          ("pooled_branches_suite", 0), ("pooled_branches_suite", 3)]


def adb(*args, timeout=180):
    return subprocess.run([ADB, "-s", SERIAL, *args], capture_output=True, text=True,
                          timeout=timeout)


def build():
    sys.path.insert(0, str(ROOT.parent / "src"))
    from open_rknpu.compose import batched_layout
    from open_rknpu.model import decode
    from open_rknpu.scheduler import compile_sequence
    from open_rknpu.sequence import decode_sequence
    manifest = []
    for suite, index in MODELS:
        source = ROOT / suite / f"model{index:03}.onnx"
        serial, _ = compile_sequence(source)
        retained = (ROOT / suite / f"model{index:03}.bin").read_bytes()
        if serial != retained:
            raise SystemExit("serial compile of %s no longer matches the retained container"
                            % source)
        batched, meta = compile_sequence(source, submission="batched")
        info = decode_sequence(batched)
        reason = batched_layout(batched, decode(batched))
        if reason is not None:
            raise SystemExit("%s batched container rejected: %s" % (source, reason))
        name = "%s_model%03d_batched.bin" % (suite, index)
        (OUT / name).write_bytes(batched)
        manifest.append(dict(suite=suite, index=index, file=name, tasks=info["task_count"],
                             mode="batched", submission=meta["submission"],
                             input_bytes=info["input_bytes"],
                             output_bytes=info["output_bytes"],
                             cases=min(4, (ROOT / suite / f"input{index:03}.u8").stat().st_size
                                       // info["input_bytes"])))
        print("built", name, info["task_count"], "tasks", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def push(src, dst):
    """Push a file, retrying the board's flaky USB link instead of using a stale one."""
    last = None
    for _ in range(6):
        last = adb("push", str(src), dst)
        if last.returncode == 0 and "not found" not in (last.stdout + last.stderr):
            return
        time.sleep(8)
    raise SystemExit("adb push failed: %s -> %s (%s)"
                     % (src, dst, (last.stdout + last.stderr).strip()[:120]))

def bench(model, input_bytes, expected_bytes, in_size, out_size, iterations, case):
    (Path("/tmp") / "mixed.u8").write_bytes(input_bytes[case * in_size:(case + 1) * in_size])
    (Path("/tmp") / "mixed.i8").write_bytes(
        expected_bytes[case * out_size:(case + 1) * out_size])
    push(model, REMOTE + "/t.bin")
    push("/tmp/mixed.u8", REMOTE + "/t.u8")
    push("/tmp/mixed.i8", REMOTE + "/t.i8")
    text = ""
    for _ in range(6):
        run = adb("shell", "%s/board_bench %s/t.bin %s/t.u8 %d %s/t.i8"
                  % (REMOTE, REMOTE, REMOTE, iterations, REMOTE))
        text = (run.stdout + run.stderr).strip()
        # An empty reply or an adb transport error is not a result: the board's USB
        # link re-enumerates under load, so wait and try again instead of recording it.
        if text and "not found" not in text and "device offline" not in text:
            break
        time.sleep(8)
    match = LINE.search(text)
    if not match:
        return dict(passed=False, case=case, output=text[:160])
    return dict(passed=match.group(8) == "0" and match.group(9) == "0", case=case,
                tasks=int(match.group(2)), mode=match.group(3),
                min_us=float(match.group(4)), median_us=float(match.group(5)),
                unstable=int(match.group(8)), mismatches=int(match.group(9)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--harness", default="/tmp/board_bench")
    args = parser.parse_args()
    manifest = build()
    adb("shell", "mkdir -p %s" % REMOTE)
    push(args.harness, REMOTE + "/board_bench")
    results, lines = [], []
    for entry in manifest:
        suite, index = entry["suite"], entry["index"]
        input_bytes = (ROOT / suite / f"input{index:03}.u8").read_bytes()
        expected_bytes = (ROOT / suite / f"expected{index:03}.i8").read_bytes()
        for mode, model in (("serial", ROOT / suite / f"model{index:03}.bin"),
                            ("batched", OUT / entry["file"])):
            runs = [bench(model, input_bytes, expected_bytes, entry["input_bytes"],
                          entry["output_bytes"], args.iterations, case)
                    for case in range(entry["cases"])]
            passed = all(run["passed"] for run in runs)
            record = dict(suite=suite, index=index, mode=mode, passed=passed,
                          cases=len(runs), iterations=args.iterations,
                          tasks=runs[0].get("tasks"),
                          min_us=min(r.get("min_us", 0) or 0 for r in runs),
                          median_us=max(r.get("median_us", 0) or 0 for r in runs),
                          mismatches=sum(r.get("mismatches", 0) or 0 for r in runs))
            results.append(record)
            line = ("%-22s %-7s tasks=%2s cases=%d exact=%s min=%7.1fus median=%8.1fus "
                    "mismatches=%d" % (suite, mode, record["tasks"], record["cases"],
                                       passed, record["min_us"], record["median_us"],
                                       record["mismatches"]))
            lines.append(line)
            print(line, flush=True)
            (OUT / "board_results.json").write_text(json.dumps(results, indent=2) + "\n")
    header = ("mixed-engine DAG in one linked job (docs/plans/pipelining-plan.md S8), "
              "tests/board_bench.c, %d cases per mode, expected bytes are the suite's own "
              "board-verified files:" % args.iterations)
    (OUT / "board_results.txt").write_text(header + "\n\n" + "\n".join(lines) + "\n")
    (OUT / "summary.txt").write_text(
        "%d containers, both modes exact=%s, %d board runs per case\n"
        % (len(manifest), all(r["passed"] for r in results), args.iterations))
    if not all(r["passed"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
