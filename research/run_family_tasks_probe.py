"""Fit per-family task cost from duplicated-task containers (S5).

For each family the runner benches every variant in `family_tasks_probe/` (one clean
boot per family, since serial submissions of idempotent duplicates cannot wedge the
NPU) and least-squares fits `cost(N) = C + N * task` over the family task counts.
Writes `family_tasks.json` and `summary.txt`.

usage: PYTHONPATH=src python3 research/run_family_tasks_probe.py [--iterations N]
"""
from pathlib import Path
import os
import json
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "family_tasks_probe"
ADB = os.environ.get("ADB", "adb")
SERIAL = "498063e3262e55b7"
REMOTE = "/userdata/open-npu-research/family_tasks"
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


def bench(name, iterations):
    adb("push", str(OUT / f"model_{name}.bin"), "%s/t.bin" % REMOTE)
    adb("push", str(OUT / f"input_{name}.u8"), "%s/t.u8" % REMOTE)
    adb("push", str(OUT / f"expected_{name}.i8"), "%s/t.i8" % REMOTE)
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
                mean_us=float(match.group(6)), max_us=float(match.group(7)))


def fit(xs, ys):
    """Least-squares slope/intercept and R^2 for cost = intercept + slope * tasks."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return slope, intercept, (1 - ss_res / ss_tot if ss_tot else 1.0)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=64)
    args = parser.parse_args()
    manifest = json.loads((OUT / "manifest.json").read_text())
    adb("shell", "mkdir -p %s" % REMOTE)
    adb("push", "/tmp/board_bench", REMOTE + "/board_bench")
    results = {}
    for family in ("conv", "pool", "elem"):
        entries = [entry for entry in manifest if entry["family"] == family]
        reboot()
        for entry in entries:
            record = dict(entry, **bench(entry["name"], args.iterations))
            results[entry["name"]] = record
            print("%-10s family_tasks=%2d container_tasks=%2d median=%7.1fus "
                  "min=%7.1fus exact=%s" % (
                entry["name"], entry["family_tasks"], record.get("tasks", -1),
                record.get("median_us", -1), record.get("min_us", -1),
                record.get("passed")), flush=True)
            (OUT / "family_tasks.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["per-family task cost from duplicated-task containers (%d runs each, serial):"
             % args.iterations]
    for family in ("conv", "pool", "elem"):
        entries = sorted((entry for entry in results.values() if entry["family"] == family),
                         key=lambda entry: entry["family_tasks"])
        xs = [entry["family_tasks"] for entry in entries]
        med = fit(xs, [entry["median_us"] for entry in entries])
        mn = fit(xs, [entry["min_us"] for entry in entries])
        exact = all(entry["passed"] for entry in entries)
        lines.append("  %-4s exact=%s  median fit: %6.1f us + %5.1f us/task (R2 %.2f) | "
                     "min fit: %6.1f us + %5.1f us/task (R2 %.2f)" % (
            family, exact, med[1], med[0], med[2], mn[1], mn[0], mn[2]))
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
