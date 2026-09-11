"""Run the ERDMA notch broadcast variants on the board.

Each variant is run over the retained 32-input batch with `board_run` and compared
with the suite's expected bytes (the verified materialized per-row output), so a
variant that broadcasts must match exactly. A variant that hangs has its `dmesg`
tail recorded and the board rebooted before the next one.

usage: PYTHONPATH=src python3 research/run_mul_broadcast_notch.py
"""
from pathlib import Path
import os
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parent
SUITE = ROOT / "mul_broadcast_notch_suite"
ADB = os.environ.get("ADB", "adb")
SERIAL = "498063e3262e55b7"
REMOTE = "/userdata/open-npu-research/notch_check"
RESULTS = SUITE / "notch_results.json"


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


def run_variant(entry):
    name = f"model{entry['index']:03d}"
    record = dict(entry)
    push(SUITE / f"{name}.bin", REMOTE + "/t.bin")
    push(SUITE / f"input{entry['index']:03d}.u8", REMOTE + "/t.u8")
    push(SUITE / f"expected{entry['index']:03d}.i8", REMOTE + "/t.i8")
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
        adb("pull", REMOTE + "/t.i8", str(SUITE / f"output{entry['index']:03d}.i8"))
        produced = (SUITE / f"output{entry['index']:03d}.i8").read_bytes()
        expected = (SUITE / f"expected{entry['index']:03d}.i8").read_bytes()
        record["exact"] = produced == expected
        record["bytes"] = len(produced)
        record["differing_bytes"] = sum(1 for a, b in zip(produced, expected) if a != b)
    else:
        record["dmesg"] = adb("shell", "dmesg | tail -6").stdout.strip()[-600:]
        reboot()
    return record


def main():
    adb("shell", "mkdir -p %s" % REMOTE)
    push("/tmp/board_run", REMOTE + "/board_run")
    adb("shell", "chmod 777 %s/board_run" % REMOTE)
    manifest = json.loads((SUITE / "manifest.json").read_text())
    results = {}
    for entry in manifest:
        record = run_variant(entry)
        results[entry["variant"]] = record
        print("%-38s passed=%-5s exact=%-5s %s" % (entry["variant"], record["passed"],
                                                   record.get("exact"), record["output"]),
              flush=True)
        RESULTS.write_text(json.dumps(results, indent=2) + "\n")
    lines = ["ERDMA notch broadcast variants (7x5x3 per-row constant, 32 cases):"]
    for entry in manifest:
        record = results[entry["variant"]]
        lines.append("  %-38s passed=%-5s exact=%-5s differing=%s"
                     % (entry["variant"], record["passed"], record.get("exact"),
                        record.get("differing_bytes")))
    (SUITE / "notch_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
