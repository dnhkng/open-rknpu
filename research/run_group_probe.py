"""S2 probe: what a batched job can actually contain, and what grouping can do.

Three questions, each answered with a verified container whose programs and arena are
untouched (container surgery only changes the descriptor table and the submission
flag, so any difference is caused by submission):

* **G1 terminal vs linked.** The only passing batched case (`chain_output_quantization`,
  2x CNA) carries next-command links. `depthwise_asymmetric_suite` is the same shape
  serially (2x CNA, terminal tails): does a batched job need the links?
* **G2 two-task control.** The known-good linked pair.
* **G3 job size.** Duplicate the verified linked pair into a four-descriptor job
  (the second pair re-reads the same surfaces, so a correct run is idempotent and
  still matches the expected bytes). Does a job complete beyond two tasks?

Cases run after a clean boot each, because a timed-out job leaves the NPU wedged.
Evidence: `research/group_probe/{results.json,summary.txt,README.md}`.
"""
from pathlib import Path
import os
import json
import re
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "group_probe"
OUT.mkdir(exist_ok=True)

ADB = os.environ.get("ADB", "adb")
SERIAL = "498063e3262e55b7"
REMOTE = "/userdata/open-npu-research/group_probe"
LINE = re.compile(r"runs=(\d+) tasks=(\d+) (\w+) min=([\d.]+)us median=([\d.]+)us "
                  r"mean=([\d.]+)us max=([\d.]+)us unstable=(\d+) mismatches=(\d+)")

sys.path.insert(0, str(ROOT))
from open_rknpu.model import checksum  # noqa: E402
from open_rknpu.sequence import decode_sequence  # noqa: E402

HEADER = 96
DESCRIPTOR = 16
ENGINE = {29: "CNA", 96: "DPU", 24: "DPU"}


def adb(*args, timeout=90):
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


def resign(data):
    data = bytearray(data)
    data[80:84] = b"\0" * 4
    struct.pack_into("<I", data, 80, checksum(bytes(data)))
    return bytes(data)


def set_mode(data, serial):
    data = bytearray(data)
    flags = struct.unpack_from("<I", data, 84)[0]
    flags = (flags | 1) if serial else (flags & ~1)
    struct.pack_into("<I", data, 84, flags)
    return resign(data)


def duplicate_descriptors(data):
    """Append a copy of the v3 descriptor table to make a four-task job."""
    info = decode_sequence(data)
    if info["format_version"] not in (3, 4):
        raise ValueError("descriptor duplication implemented for v3/v4")
    count = info["task_count"]
    table = data[HEADER:HEADER + count * DESCRIPTOR]
    payload = data[HEADER + count * DESCRIPTOR:]
    result = bytearray(data[:HEADER]) + bytearray(table) + bytearray(table) + bytearray(payload)
    struct.pack_into("<I", result, 60, count * 2)
    return resign(bytes(result))


def cases():
    out = []
    terminal = (ROOT / "depthwise_asymmetric_suite" / "model000.bin").read_bytes()
    info = decode_sequence(terminal)
    assert info["task_count"] == 2 and info["serial"]
    assert {ENGINE[t["enable"]] for t in info["tasks"]} == {"CNA"}
    out.append(("G1-2xCNA-terminal-committed", "depthwise_asymmetric_suite", 0,
                terminal, "committed serial control"))
    out.append(("G1-2xCNA-terminal-batched", "depthwise_asymmetric_suite", 0,
                set_mode(terminal, False), "batched, terminal tails (no links)"))

    linked = (ROOT / "chain_output_quantization_suite" / "model000.bin").read_bytes()
    out.append(("G2-2xCNA-linked-batched", "chain_output_quantization_suite", 0,
                linked, "committed batched control (linked tails)"))
    four = duplicate_descriptors(linked)
    info4 = decode_sequence(four)
    assert info4["task_count"] == 4 and info4["serial"] is False
    out.append(("G3-4xCNA-linked-batched", "chain_output_quantization_suite", 0,
                four, "batched four-descriptor job (second pair idempotent)"))
    out.append(("G3-4xCNA-linked-serial", "chain_output_quantization_suite", 0,
                set_mode(four, True), "four tasks, one ioctl each (control)"))
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--only", default=None)
    args = parser.parse_args()
    adb("shell", "mkdir -p %s" % REMOTE)
    adb("push", str(ROOT / "submission_probe" / "board_bench"), REMOTE + "/board_bench")
    results = []
    for name, suite, index, data, why in cases():
        if args.only and args.only not in name:
            continue
        reboot()
        Path("/tmp/group.bin").write_bytes(data)
        for src, dst in (("/tmp/group.bin", "t.bin"),
                         (str(ROOT / suite / f"input{index:03}.u8"), "t.u8"),
                         (str(ROOT / suite / f"expected{index:03}.i8"), "t.i8")):
            adb("push", src, "%s/%s" % (REMOTE, dst))
        text = ""
        for _ in range(3):
            run = adb("shell", "%s/board_bench %s/t.bin %s/t.u8 %d %s/t.i8"
                      % (REMOTE, REMOTE, REMOTE, args.iterations, REMOTE))
            text = (run.stdout + run.stderr).strip()
            if text:
                break
            time.sleep(6)
        match = LINE.search(text)
        record = dict(name=name, suite=suite, model=index, why=why,
                      serial=(data[84] & 1) == 1, passed=bool(match), output=text[:200])
        if match:
            record.update(tasks=int(match.group(2)), mode=match.group(3),
                          median_us=float(match.group(5)), unstable=int(match.group(8)),
                          mismatches=int(match.group(9)))
        results = [entry for entry in results if entry["name"] != name] + [record]
        results.sort(key=lambda entry: entry["name"])
        print("%-30s %-5s %s" % (name, "PASS" if record["passed"] else "FAIL",
                                 (text.splitlines() or ["(no output)"])[0][:78]), flush=True)
        (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    passed = [entry["name"] for entry in results if entry["passed"]]
    summary = "group probe: %d/%d cases exact and stable (%s)" % (
        len(passed), len(results), ", ".join(passed) or "none")
    (OUT / "summary.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
