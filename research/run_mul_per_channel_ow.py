"""Run the corrected per-channel-output-conversion variants on the board.

`check_mul_per_channel_ow.py` builds the baseline and four corrected variants of the
same 8x6x3 scalar-constant Mul. Each variant is run over the retained 32-input batch
with `board_run`, its output compared with the suite's expected file and with the
baseline output per channel. A variant that hangs gets its `dmesg` tail recorded and
the board rebooted before the next one, so a wedge cannot contaminate the sequence.

usage: PYTHONPATH=src python3 research/run_mul_per_channel_ow.py
"""
from pathlib import Path
import os
import json
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
SUITE = ROOT / "mul_per_channel_ow_suite"
EXPECTED = ROOT / "mul_broadcast_suite" / "expected007.i8"
ADB = os.environ.get("ADB", "adb")
SERIAL = "498063e3262e55b7"
REMOTE = "/userdata/open-npu-research/ow_check"
VARIANTS = ("model000", "model001", "model002", "model003", "model004", "model005")
RESULTS = SUITE / "ow_results.json"
SHAPE = (32, 8, 6, 3)  # 32 retained cases, 8x6, 3 channels


def adb(*args, timeout=120):
    return subprocess.run([ADB, "-s", SERIAL, *args], capture_output=True, text=True,
                          timeout=timeout)


def push(local, remote):
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


def run_variant(name):
    record = {}
    push(SUITE / f"{name}.bin", REMOTE + "/t.bin")
    push(SUITE / f"input{name[5:]}.u8", REMOTE + "/t.u8")
    text = ""
    for _ in range(3):
        result = adb("shell", "%s/board_run %s/t.bin %s/t.u8 %s/t.i8"
                     % (REMOTE, REMOTE, REMOTE, REMOTE))
        text = (result.stdout + result.stderr).strip()
        if text:
            break
        time.sleep(6)
    record["output"] = text[:200]
    record["passed"] = "rc=0" in text
    if record["passed"]:
        adb("pull", REMOTE + "/t.i8", str(SUITE / f"output{name[5:]}.i8"))
        record["bytes"] = (SUITE / f"output{name[5:]}.i8").stat().st_size
        return record
    record["dmesg"] = adb("shell", "dmesg | tail -6").stdout.strip()[-600:]
    reboot()
    return record


def channel_summary(candidate, baseline):
    """How the candidate output relates to the baseline, per output channel."""
    a = np.frombuffer(candidate, np.int8).reshape(SHAPE)
    b = np.frombuffer(baseline, np.int8).reshape(SHAPE)
    return [dict(channel=channel,
                 identical=int((a[..., channel] == b[..., channel]).sum()),
                 values=int(a[..., channel].size),
                 first=int(a[0, 0, 0, channel]), baseline_first=int(b[0, 0, 0, channel]))
            for channel in range(SHAPE[-1])]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = parser.parse_args()
    adb("shell", "mkdir -p %s" % REMOTE)
    push("/tmp/board_run", REMOTE + "/board_run")
    adb("shell", "chmod 777 %s/board_run" % REMOTE)
    expected = EXPECTED.read_bytes()
    results = json.loads(RESULTS.read_text()) if RESULTS.is_file() else {}
    for name in args.variants:
        record = run_variant(name)
        if record["passed"]:
            produced = (SUITE / f"output{name[5:]}.i8").read_bytes()
            record["exact"] = produced == expected
            record["bytes"] = len(produced)
        results[name] = record
        print("%-9s passed=%-5s exact=%s %s" % (name, record["passed"],
                                                record.get("exact"), record["output"]),
              flush=True)
        RESULTS.write_text(json.dumps(results, indent=2) + "\n")
    baseline = (SUITE / "output000.i8").read_bytes() if results["model000"]["passed"] else None
    lines = ["per-channel Mul output conversion via BS_OW_CFG.OW_SRC (8x6x3, 32 cases):"]
    for name in VARIANTS:
        if name not in results:
            continue
        record = results[name]
        lines.append("  %-9s passed=%s exact=%s bytes=%s"
                     % (name, record["passed"], record.get("exact"), record.get("bytes")))
        if baseline is not None and record.get("exact") is not None:
            lines.append("      vs baseline: " + ", ".join(
                "c%d %d/%d same (first %d vs %d)"
                % (entry["channel"], entry["identical"], entry["values"],
                   entry["first"], entry["baseline_first"])
                for entry in channel_summary(
                    (SUITE / f"output{name[5:]}.i8").read_bytes(), baseline)))
    (SUITE / "ow_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
