"""Board run for the K > 1 tiled suite (docs/plans/pipelining-plan.md S7).

For every chain in `tiled_k3_probe/manifest.json` it runs the untiled control, the
`tiles=2`/`tiles=4` strip containers and their one-job batched twins over every input
case with `tests/board_bench.c`, comparing each against the same integer reference.

A failing case reboots the board before the next one: a job that times out wedges the
NPU until reboot. Evidence: `board_results.txt` (human) and `board_results.json`
(machine, checked by `tests/test_tiled_chain.py`).
"""
from pathlib import Path
import os
import argparse
import json
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SUITE = ROOT / "tiled_k3_probe"
ADB = os.environ.get("ADB", "adb")
SERIAL = "498063e3262e55b7"
REMOTE = "/userdata/open-npu-research/tiled_k3"
LINE = re.compile(r"runs=(\d+) tasks=(\d+) (\S+) min=([\d.]+)us median=([\d.]+)us "
                  r"mean=([\d.]+)us max=([\d.]+)us unstable=(\d+) mismatches=(\d+)")


def adb(*args, timeout=180):
    return subprocess.run([ADB, "-s", SERIAL, *args], capture_output=True, text=True,
                          timeout=timeout)


def reboot():
    adb("shell", "reboot", timeout=30)
    ready = 0
    for _ in range(60):
        time.sleep(3)
        if "up" in adb("shell", "echo up", timeout=30).stdout:
            ready += 1
            if ready >= 2:
                time.sleep(4)
                return
        else:
            ready = 0
    raise SystemExit("board did not come back after reboot")


def bench(model, name, case, iterations):
    adb("push", str(model), "%s/t.bin" % REMOTE)
    adb("push", str(SUITE / f"{name}_input{case}.u8"), "%s/t.u8" % REMOTE)
    adb("push", str(SUITE / f"{name}_expected{case}.i8"), "%s/t.i8" % REMOTE)
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
        return dict(passed=False, output=text[:160], case=case)
    return dict(passed=match.group(8) == "0" and match.group(9) == "0", case=case,
                tasks=int(match.group(2)), mode=match.group(3),
                min_us=float(match.group(4)), median_us=float(match.group(5)),
                unstable=int(match.group(8)), mismatches=int(match.group(9)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--harness", default="/tmp/board_bench")
    parser.add_argument("--only", default=None)
    args = parser.parse_args()
    manifest = json.loads((SUITE / "manifest.json").read_text())
    adb("shell", "mkdir -p %s" % REMOTE)
    adb("push", args.harness, REMOTE + "/board_bench")
    results, lines = [], []
    for entry in manifest:
        name = entry["name"]
        if args.only and args.only != name:
            continue
        variants = [("untiled", SUITE / f"{name}_untiled.bin")]
        variants += [(key, SUITE / f"{name}_{key}.bin") for key in sorted(entry["variants"])]
        for key, model in variants:
            runs = [bench(model, name, case, args.iterations) for case in range(entry["cases"])]
            passed = all(run["passed"] for run in runs)
            record = dict(model=name, variant=key, passed=passed, cases=len(runs),
                          iterations=args.iterations, tasks=runs[0].get("tasks"),
                          mode=runs[0].get("mode"),
                          min_us=min(r.get("min_us", 0) for r in runs),
                          median_us=max(r.get("median_us", 0) for r in runs),
                          mismatches=sum(r.get("mismatches", 0) for r in runs),
                          unstable=sum(r.get("unstable", 0) for r in runs))
            results.append(record)
            line = ("%s %-14s tasks=%2s %-7s cases=%d exact=%s min=%7.1fus median=%8.1fus "
                    "mismatches=%d" % (name, key, record["tasks"], record["mode"], len(runs),
                                       passed, record["min_us"], record["median_us"],
                                       record["mismatches"]))
            lines.append(line)
            print(line, flush=True)
            (SUITE / "board_results.json").write_text(json.dumps(results, indent=2) + "\n")
            if not passed:
                reboot()
    header = ("K>1 height-strip tiling (docs/plans/pipelining-plan.md S7), tests/board_bench.c, "
              "%d cases each, identical input/expected bytes as each chain's untiled "
              "container:" % args.iterations)
    (SUITE / "board_results.txt").write_text(header + "\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
    if not all(r["passed"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
