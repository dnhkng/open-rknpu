"""Cross-check the per-family cost table on held-out real containers (S5).

`build_family_cost_crosscheck.py` selected existing board-verified containers whose
(conv, pool, elementwise) task counts - and nothing else - differ from the duplicated
templates the table was fitted on. This runner benches each one with `board_bench`
(min/median/mean/max over N runs, exactness against the retained expected file), then

* compares each measurement with the table's slope-only prediction, and
* re-fits the three per-family costs plus an intercept on the held-out set alone, so
  two independent experiments can be compared.

Writes `crosscheck.json` and `summary.txt`.

usage: PYTHONPATH=src python3 research/run_family_cost_crosscheck.py [--iterations N]
"""
from pathlib import Path
import os
import argparse
import json
import re
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "family_cost_crosscheck"
ADB = os.environ.get("ADB", "adb")
SERIAL = "498063e3262e55b7"
REMOTE = "/userdata/open-npu-research/family_crosscheck"
LINE = re.compile(r"runs=(\d+) tasks=(\d+) (\S+) min=([\d.]+)us median=([\d.]+)us "
                  r"mean=([\d.]+)us max=([\d.]+)us unstable=(\d+) mismatches=(\d+)")


def adb(*args, timeout=180):
    return subprocess.run([ADB, "-s", SERIAL, *args], capture_output=True, text=True,
                          timeout=timeout)


def push(local, remote):
    """Push with a size check: a silently failed push once mislabelled a board row."""
    wanted = Path(local).stat().st_size
    for _ in range(3):
        adb("push", str(local), remote)
        listing = adb("shell", "wc -c < %s" % remote).stdout.strip()
        if listing.isdigit() and int(listing) == wanted:
            return
        time.sleep(2)
    raise SystemExit("push failed for %s" % local)


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


def bench(entry, iterations):
    push(ROOT / entry["model"], REMOTE + "/t.bin")
    push(ROOT / entry["input"], REMOTE + "/t.u8")
    push(ROOT / entry["expected"], REMOTE + "/t.i8")
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
        return dict(passed=False, output=text[:200])
    return dict(passed=match.group(8) == "0" and match.group(9) == "0",
                tasks=int(match.group(2)), mode=match.group(3),
                min_us=float(match.group(4)), median_us=float(match.group(5)),
                mean_us=float(match.group(6)), max_us=float(match.group(7)))


def least_squares(rows, ys):
    """Coefficients, standard errors and R^2 for y = c0 + c_conv*x1 + c_pool*x2 + c_elem*x3."""
    design = np.array(rows, dtype=float)
    values = np.array(ys, dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    residuals = values - design @ coefficients
    dof = max(1, len(values) - design.shape[1])
    variance = float(residuals @ residuals) / dof
    covariance = variance * np.linalg.pinv(design.T @ design)
    errors = np.sqrt(np.diag(covariance))
    total = float(((values - values.mean()) ** 2).sum())
    r2 = 1 - float(residuals @ residuals) / total if total else 1.0
    return coefficients, errors, r2, residuals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--output", default="crosscheck.json",
                        help="measurement file inside family_cost_crosscheck/")
    args = parser.parse_args()
    output = OUT / args.output
    summary_path = output.with_suffix(".txt")
    manifest = json.loads((OUT / "manifest.json").read_text())
    table = manifest["table_slopes_us"]
    entries = manifest["containers"]
    adb("shell", "mkdir -p %s" % REMOTE)
    push("/tmp/board_bench", REMOTE + "/board_bench")
    reboot()
    results = {}
    for entry in entries:
        record = dict(entry, **bench(entry, args.iterations))
        results[entry["name"]] = record
        error = (record["min_us"] - entry["table_slope_us"]) / entry["table_slope_us"] * 100
        print("%-28s tasks=%2d min=%7.1fus median=%7.1fus table=%6.1fus err=%+6.1f%% exact=%s"
              % (entry["name"], entry.get("tasks", -1), record.get("min_us", -1),
                 record.get("median_us", -1), entry["table_slope_us"], error,
                 record.get("passed")), flush=True)
        output.write_text(json.dumps(results, indent=2) + "\n")

    rows = [[1.0, record["conv"], record["pool"], record["elem"]] for record in results.values()]
    mins = [record["min_us"] for record in results.values()]
    medians = [record["median_us"] for record in results.values()]
    coefficients, errors, r2, residuals = least_squares(rows, mins)
    table_vector = np.array([table["conv"], table["pool"], table["elem"]])
    fitted_vector = coefficients[1:]
    noise = [record["median_us"] / record["min_us"] for record in results.values()]
    error_percent = [(record["min_us"] - record["table_slope_us"]) / record["table_slope_us"] * 100
                     for record in results.values()]

    lines = ["per-family cost cross-check on %d held-out containers (%d runs each, serial):"
             % (len(results), args.iterations),
             "  table (duplicated idempotent tasks): conv %.1f, pool %.1f, elem %.1f us/task"
             % tuple(table_vector),
             "  held-out fit (real dependency chains): intercept %.1f us, conv %.1f+-%.1f, "
             "pool %.1f+-%.1f, elem %.1f+-%.1f us/task (R2 %.3f)"
             % (coefficients[0], *np.column_stack([fitted_vector, errors[1:]]).ravel(), r2),
             "  table slope-only prediction error: median %+.1f%%, mean %+.1f%%, "
             "max |error| %.1f%%" % (float(np.median(error_percent)),
                                     float(np.mean(error_percent)),
                                     float(np.max(np.abs(error_percent))))]
    for name, record in sorted(results.items()):
        lines.append("  %-28s tasks=%2d conv=%d pool=%d elem=%d min=%7.1fus median=%7.1fus "
                     "table=%6.1fus noise=%.2fx exact=%s"
                     % (name, record["tasks"], record["conv"], record["pool"], record["elem"],
                        record["min_us"], record["median_us"], record["table_slope_us"],
                        record["median_us"] / record["min_us"], record["passed"]))
    lines.append("  median/min noise ratio: min %.2fx, median %.2fx, max %.2fx"
                 % (min(noise), float(np.median(noise)), max(noise)))
    summary_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[1:5]))


if __name__ == "__main__":
    main()
