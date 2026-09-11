"""S5: per-task cost model for the three task families.

Times single-task containers (one CNA conv, one DPU pool, one DPU elementwise) and
the deep chains to separate the per-submission cost from the marginal per-task cost.
Writes `research/group_probe/cost_model.json`.

usage: PYTHONPATH=src python3 research/run_cost_model.py
"""
from pathlib import Path
import os
import json
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "group_probe"

ADB = os.environ.get("ADB", "adb")
SERIAL = "498063e3262e55b7"
REMOTE = "/userdata/open-npu-research/cost_model"
LINE = re.compile(r"runs=(\d+) tasks=(\d+) (\S+) min=([\d.]+)us median=([\d.]+)us "
                  r"mean=([\d.]+)us max=([\d.]+)us unstable=(\d+) mismatches=(\d+)")

# One task per family, all with recorded board evidence in the ledger.
# Only one single-task container exists in the verified tree, so the per-family
# numbers are derived by differencing multi-task graphs of known shape.
CASES = [
    ("conv-1task", "accumulator_boundary_suite", 0, "one CNA Conv task"),
    ("conv+pool-2task", "scheduled_pool_suite", 0, "CNA Conv then DPU pool, serial"),
    ("conv+pool-2task-b", "mnist_pool_suite", 0, "CNA Conv then DPU pool, serial"),
    ("conv2+ew-3task", "add_suite", 0, "CNA Conv, CNA Conv, DPU elementwise, serial"),
    ("conv-chain-3", "native_chain_suite", 0, "3 CNA tasks, serial"),
]


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


def bench(suite, index, iterations, batched):
    folder = ROOT / suite
    model = folder / f"model{index:03}.bin"
    adb("push", str(model), "%s/t.bin" % REMOTE)
    adb("push", str(folder / f"input{index:03}.u8"), "%s/t.u8" % REMOTE)
    adb("push", str(folder / f"expected{index:03}.i8"), "%s/t.i8" % REMOTE)
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
                median_us=float(match.group(5)), min_us=float(match.group(4)))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--only", default=None)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    adb("shell", "mkdir -p %s" % REMOTE)
    adb("push", "/tmp/board_bench", REMOTE + "/board_bench")
    path = OUT / "cost_model.json"
    results = json.loads(path.read_text()) if args.merge and path.exists() else []
    done = {record["name"] for record in results}
    for name, suite, index, why in CASES:
        if (args.only and args.only not in name) or (name in done and not args.only):
            continue
        if not (ROOT / suite / f"model{index:03}.bin").exists():
            print("%-24s skipped (no %s)" % (name, suite), flush=True)
            continue
        reboot()
        record = dict(name=name, suite=suite, model=index, why=why,
                      iterations=args.iterations, **bench(suite, index, args.iterations, False))
        results = [entry for entry in results if entry["name"] != name] + [record]
        results.sort(key=lambda entry: entry["name"])
        print("%-24s tasks=%s %-8s median=%8.1fus exact=%s" % (
            name, record.get("tasks"), record.get("mode"), record.get("median_us", -1),
            record.get("passed")), flush=True)
        (OUT / "cost_model.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["per-task cost model (median of %d runs, board otherwise idle):" % args.iterations]
    by = {record["name"]: record for record in results}
    for record in results:
        if record.get("median_us") is not None and record.get("tasks"):
            lines.append("  %-24s %2d task(s)  %8.1f us  %.1f us/task" % (
                record["name"], record["tasks"], record["median_us"],
                record["median_us"] / record["tasks"]))
    lines.append("  note: only one single-task container exists and it uses a 30 KB "
                 "input, so per-family marginals are not derivable; the fixed/marginal "
                 "model comes from the paired deep-chain and loop-job runs in "
                 "research/group_probe/README.md")
    (OUT / "cost_model.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
