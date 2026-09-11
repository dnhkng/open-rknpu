"""Measure the per-family task cost on the board (S5 residual).

Runs `tests/board_bench.c` on the three matched containers in
`research/family_cost_probe/` and derives the pool and elementwise task marginals
against the single-task CNA container. Writes `family_cost.json` / `summary.txt`.

usage: PYTHONPATH=src python3 research/run_family_cost_probe.py [--iterations N]
"""
from pathlib import Path
import os
import json
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "family_cost_probe"
ADB = os.environ.get("ADB", "adb")
SERIAL = "498063e3262e55b7"
REMOTE = "/userdata/open-npu-research/family_cost"
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
    raise SystemExit("board did not come back")


def bench(index, iterations):
    adb("push", str(OUT / f"model{index:03}.bin"), "%s/t.bin" % REMOTE)
    adb("push", str(OUT / f"input{index:03}.u8"), "%s/t.u8" % REMOTE)
    adb("push", str(OUT / f"expected{index:03}.i8"), "%s/t.i8" % REMOTE)
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
                mean_us=float(match.group(6)), max_us=float(match.group(7)),
                unstable=int(match.group(8)), mismatches=int(match.group(9)))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=128)
    args = parser.parse_args()
    manifest = json.loads((OUT / "manifest.json").read_text())
    adb("shell", "mkdir -p %s" % REMOTE)
    adb("push", "/tmp/board_bench", REMOTE + "/board_bench")
    results = {}
    for entry in manifest:
        reboot()
        index = entry["index"]
        results[entry["name"]] = dict(entry, **bench(index, args.iterations))
        record = results[entry["name"]]
        print("%-10s tasks=%d median=%7.1fus min=%7.1fus exact=%s" % (
            entry["name"], record.get("tasks", -1), record.get("median_us", -1),
            record.get("min_us", -1), record.get("passed")), flush=True)
        (OUT / "family_cost.json").write_text(json.dumps(results, indent=2) + "\n")
    conv = results.get("conv", {})
    lines = ["per-family task cost (8x8/C3, %d runs, serial submission):" % args.iterations]
    for name in ("conv", "conv_pool", "ew2"):
        record = results.get(name, {})
        if record.get("median_us"):
            lines.append("  %-10s %d task(s)  median %7.1fus  min %7.1fus" % (
                name, record["tasks"], record["median_us"], record["min_us"]))
    if conv.get("median_us"):
        for name, label in (("conv_pool", "pool task"), ("ew2", "elementwise task")):
            record = results.get(name, {})
            if record.get("median_us"):
                lines.append("  derived %-16s ~ %6.1fus median, %6.1fus min (vs one CNA task; "
                             "min is the board floor, medians are load-dominated)"
                             % (label, record["median_us"] - conv["median_us"],
                                record["min_us"] - conv["min_us"]))
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
