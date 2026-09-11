"""Retained board experiment for the batched-submission rule (S1).

docs/plans/pipelining-plan.md S1 asks when one ioctl may carry several NPU tasks. This runner
takes committed containers whose programs, tensors and arena are already verified,
produces the batched variant of each (flag flip, or a link rewrite for the chain
case), and runs every variant on the board with a *clean boot per case*: a job that
times out leaves the NPU wedged until reboot, so cases cannot share a boot.

Result: a batched job completes only for a two-task single-engine list (the 67
verified `chain_output_quantization_suite` containers). Two-task mixed-engine lists
and three-or-more-task lists time out, with or without next-command links and with
per-task or job-union interrupt masks. Evidence lands in
`research/submission_probe/rule_results.json`.
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
PROBE = ROOT / "submission_probe"
sys.path.insert(0, str(ROOT))
from build_submission_probe import with_submission  # noqa: E402

ADB = os.environ.get("ADB", "adb")
SERIAL = "498063e3262e55b7"
REMOTE = "/userdata/open-npu-research/submission_probe"
LINE = re.compile(r"runs=(\d+) tasks=(\d+) (\w+) min=([\d.]+)us median=([\d.]+)us "
                  r"mean=([\d.]+)us max=([\d.]+)us unstable=(\d+) mismatches=(\d+)")
ENGINE = {29: "CNA", 96: "DPU", 24: "DPU"}


def adb(*args, timeout=90):
    return subprocess.run([ADB, "-s", SERIAL, *args], capture_output=True, text=True,
                          timeout=timeout)


def reboot():
    """Reboot and wait until adbd answers twice in a row."""
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


def link_tail(binary, expected_next):
    """Set the next-command link on every task but the last of a chain."""
    from build_submission_probe import payload_base
    base, info = payload_base(binary)
    data = bytearray(binary)
    links = []
    for position, task in enumerate(info["tasks"][:-1]):
        offset = base + task["command_offset"] + task["register_count"] * 8
        target = info["tasks"][position + 1]["command_offset"]
        struct.pack_into("<Q", data, offset, 0x101 << 48 | target << 16 | 0x10)
        struct.pack_into("<Q", data, offset + 8, 0x101 << 48 | 0x40 << 16 | 0x14)
        links.append(hex(target))
    from open_rknpu.model import checksum
    data[80:84] = b"\0" * 4
    struct.pack_into("<I", data, 80, checksum(bytes(data)))
    return bytes(data), links


def cases():
    """(name, suite, model index, bytes or None, why)"""
    from open_rknpu.sequence import decode_sequence
    out = []

    def committed(suite, index=0):
        return (ROOT / suite / f"model{index:03}.bin").read_bytes()

    for suite, index in (("chain_output_quantization_suite", 0),
                         ("mnist_pool_suite", 0),
                         ("mul_broadcast_mode_suite", 0),
                         ("runtime_scale_suite", 0),
                         ("pool_join_suite", 0)):
        base = committed(suite, index)
        info = decode_sequence(base)
        engines = "/".join(sorted({ENGINE.get(t["enable"], "?") for t in info["tasks"]}))
        out.append((f"{suite}-committed", suite, index, base,
                    "committed container (serial=%s): %d tasks, %s"
                    % (info["serial"], info["task_count"], engines)))
        if info["serial"]:
            out.append((f"{suite}-batched", suite, index, with_submission(base, False),
                        "batched flag twin: %d tasks, %s" % (info["task_count"], engines)))
    base = committed("native_chain_suite", 0)
    info = decode_sequence(base)
    linked, links = link_tail(with_submission(base, False), None)
    out.append(("native_chain_suite-committed", "native_chain_suite", 0, base,
                "committed serial 3x conv (control)"))
    out.append(("native_chain_suite-batched-linked", "native_chain_suite", 0, linked,
                "batched 3x conv with links %s" % links))
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="substring filter on the case name")
    parser.add_argument("--merge", action="store_true",
                        help="keep existing results and replace only re-run cases")
    args = parser.parse_args()
    adb("shell", "mkdir -p %s" % REMOTE)
    adb("push", str(PROBE / "board_bench"), REMOTE + "/board_bench")
    results = []
    path = PROBE / "rule_results.json"
    if args.merge and path.exists():
        results = json.loads(path.read_text())
    done = {entry["name"] for entry in results} if args.merge else set()
    for name, suite, index, data, why in cases():
        if args.only and args.only not in name:
            continue
        if name in done and not args.only:
            continue
        reboot()
        Path("/tmp/rule.bin").write_bytes(data)
        for src, dst in (("/tmp/rule.bin", "t.bin"),
                         (str(ROOT / suite / f"input{index:03}.u8"), "t.u8"),
                         (str(ROOT / suite / f"expected{index:03}.i8"), "t.i8")):
            adb("push", src, "%s/%s" % (REMOTE, dst))
        text = ""
        for _ in range(3):
            run = adb("shell", "%s/board_bench %s/t.bin %s/t.u8 16 %s/t.i8"
                      % (REMOTE, REMOTE, REMOTE, REMOTE))
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
        first = text.splitlines()[0][:80] if text else "(no output)"
        print("%-38s %-5s %s" % (name, "PASS" if record["passed"] else "FAIL", first),
              flush=True)
        path.write_text(json.dumps(results, indent=2) + "\n")
    # The rule under test: a serial job always runs; a batched job runs only for the
    # verified two-task single-engine container.
    def expected(entry):
        if entry.get("mode") == "serial":
            return True
        return entry["name"] == "chain_output_quantization_suite-committed"
    agree = [e for e in results if e["passed"] == expected(e)]
    batched_pass = [e["name"] for e in results if e["passed"] and e["name"].endswith("-batched")]
    summary = ("batched rule: %d/%d cases as expected over a clean boot each; "
               "batched cases that completed: %s"
               % (len(agree), len(results), ", ".join(batched_pass) or "none"))
    (PROBE / "board_summary.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
