"""Deep-chain A/B: serial (N ioctls) vs one batched linked job (1 ioctl).

Runs the `deep_chain_suite` twins on the board with `tests/board_bench.c`, records
exactness and medians, and writes `batched_results.json` / `summary.txt`. A clean
boot precedes each model because a job that fails wedges the NPU.
"""
from pathlib import Path
import os
import json
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SUITE = ROOT / "deep_chain_suite"
ADB = os.environ.get("ADB", "adb")
SERIAL = "498063e3262e55b7"
REMOTE = "/userdata/open-npu-research/deep_chain"
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


def bench(model, index, iterations):
    adb("push", str(model), "%s/t.bin" % REMOTE)
    adb("push", str(SUITE / f"input{index:03}.u8"), "%s/t.u8" % REMOTE)
    adb("push", str(SUITE / f"expected{index:03}.i8"), "%s/t.i8" % REMOTE)
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
                median_us=float(match.group(5)), min_us=float(match.group(4)),
                unstable=int(match.group(8)), mismatches=int(match.group(9)))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--only", type=int, default=None)
    args = parser.parse_args()
    manifest = json.loads((SUITE / "manifest.json").read_text())
    adb("shell", "mkdir -p %s" % REMOTE)
    adb("push", "/tmp/board_bench", REMOTE + "/board_bench")
    results = []
    for entry in manifest:
        index = entry["index"]
        if args.only is not None and index != args.only:
            continue
        reboot()
        serial = bench(SUITE / f"model{index:03}.bin", index, args.iterations)
        batched = bench(SUITE / "batched" / f"model{index:03}.bin", index, args.iterations)
        reuse = bench(SUITE / "reuse" / f"model{index:03}.bin", index, args.iterations)
        reuse_batched = bench(SUITE / "reuse_batched" / f"model{index:03}.bin", index,
                              args.iterations)
        speedup = (serial.get("median_us", 0) / batched["median_us"]
                   if serial.get("median_us") and batched.get("median_us") else None)
        record = dict(index=index, layers=entry["layers"], iterations=args.iterations,
                      serial=serial, batched=batched, reuse=reuse,
                      reuse_batched=reuse_batched, speedup=speedup,
                      arena_bytes=entry["arena_bytes"],
                      reuse_arena_bytes=entry["reuse_arena_bytes"])
        results.append(record)
        print("%2d layers: serial %8.1fus  batched %8.1fus  reuse %8.1fus  reuse+batched %8.1fus"
              "  speedup %s  exact %s/%s/%s/%s" % (
            entry["layers"], serial.get("median_us", -1), batched.get("median_us", -1),
            reuse.get("median_us", -1), reuse_batched.get("median_us", -1),
            ("%.2fx" % speedup) if speedup else "-", serial.get("passed"),
            batched.get("passed"), reuse.get("passed"), reuse_batched.get("passed")), flush=True)
        (SUITE / "batched_results.json").write_text(json.dumps(results, indent=2) + "\n")
    ok = all(all(r[key].get("passed") for key in ("serial", "batched", "reuse", "reuse_batched"))
             for r in results)
    gains = [r["speedup"] for r in results if r["speedup"]]
    summary = ("deep chain: %d graphs, both modes exact=%s, batched speedup %.2fx..%.2fx "
               "over %d runs each" % (len(results), ok, min(gains), max(gains),
                                      args.iterations) if gains else "deep chain: no data")
    (SUITE / "summary.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
