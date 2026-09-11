"""S8 job-field probe: does the task-tail control word carry the engine transition?

Vendor capture (`research/vendor/`, `capture_pool_max`) decodes a Conv task's tail as
`0x10 = next payload-relative program offset`, `0x14 = control`, with

* `0x40` in front of another Conv (CNA) task (`capture_chain4`),
* `0x14` in front of a Pool (DPU) task (`capture_pool_max`, `capture_identity`),
* `0x28` on the terminal task of both shapes.

Our own emitters only ever write `0x40` (linked) or `0x28` (terminal) and refuse a
mixed-engine batched list, so the hypothesis is that a mixed list needs `0x14` - the
value the vendor compiler writes - at every CNA->DPU transition. This probe runs that
experiment on the board with verified serial containers whose programs and arena are
untouched: only the submission flag and the four tail words per task change.

Phases (each attempt after a clean boot; a failed job wedges the NPU until reboot):

1. `pair`      2 tasks  CNA->DPU   (`mnist_pool_suite/model000.bin`)   sweep the cross control
2. `join6`     6 tasks  3 CNA->3 DPU (`pool_join_suite/model000.bin`)   sweep the DPU->DPU control
3. `mixedhead7` 7 tasks 4 CNA->2 DPU->CNA (`mixed_head_suite/model001.bin`) sweep DPU->CNA

Every attempt is checked against the suite's own expected bytes for every input case, so
a pass means the whole mixed job ran in one ioctl and produced the verified output.

Evidence: `research/job_field_probe/{board_results.json,summary.txt,README.md}`.
"""
from pathlib import Path
import os
import argparse
import json
import re
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "job_field_probe"
OUT.mkdir(exist_ok=True)
TWINS = OUT / "twins"
TWINS.mkdir(exist_ok=True)
ADB = os.environ.get("ADB", "adb")
SERIAL = "498063e3262e55b7"
REMOTE = "/userdata/open-npu-research/job_field"
LINE = re.compile(r"runs=(\d+) tasks=(\d+) (\S+) min=([\d.]+)us median=([\d.]+)us "
                  r"mean=([\d.]+)us max=([\d.]+)us unstable=(\d+) mismatches=(\d+)")
CANDIDATES = [0x14, 0x40, 0x28, 0x18, 0x1C, 0x2C, 0x48, 0x30, 0x24, 0x44, 0x0C, 0x20]
ENGINE = {29: "CNA", 96: "DPU", 24: "DPU"}


def engines(info):
    return [ENGINE.get(task["enable"], task["enable"]) for task in info["tasks"]]


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


def resign(data):
    from open_rknpu.model import checksum
    data = bytearray(data)
    data[80:84] = b"\0" * 4
    struct.pack_into("<I", data, 80, checksum(bytes(data)))
    return bytes(data)


def payload_base(data):
    from open_rknpu.sequence import decode_sequence
    info = decode_sequence(data)
    if info["format_version"] == 5:
        return 112 + 16 * info["task_count"] + 64 * info["tensor_count"], info
    return 96 + 16 * info["task_count"] + 40 * info.get("constant_count", 0), info


def tail(data, base, info, position, control, link=0):
    """Rewrite the two link words of one task; the 0x41/0x81 words stay committed."""
    task = info["tasks"][position]
    start = base + task["command_offset"] + task["register_count"] * 8
    data = bytearray(data)
    struct.pack_into("<Q", data, start, 0x101 << 48 | link << 16 | 0x10)
    struct.pack_into("<Q", data, start + 8, 0x101 << 48 | control << 16 | 0x14)
    return bytes(data)


def batched(data, transitions):
    """One ioctl, every task linked in table order, last terminal."""
    base, info = payload_base(data)
    offsets = [task["command_offset"] for task in info["tasks"]]
    data = bytearray(data)
    data[84:88] = struct.pack("<I", struct.unpack_from("<I", data, 84)[0] & ~1)
    data = resign(bytes(data))
    for position in range(info["task_count"] - 1):
        data = tail(data, base, info, position, transitions[position],
                    link=offsets[position + 1])
    data = tail(data, base, info, info["task_count"] - 1, 0x28)
    return resign(bytes(data))


def run(model, suite, index, iterations):
    for src, dst in ((model, "t.bin"), (ROOT / suite / f"input{index:03}.u8", "t.u8"),
                     (ROOT / suite / f"expected{index:03}.i8", "t.i8")):
        adb("push", str(src), "%s/%s" % (REMOTE, dst))
    text = ""
    for _ in range(3):
        result = adb("shell", "%s/board_bench %s/t.bin %s/t.u8 %d %s/t.i8"
                     % (REMOTE, REMOTE, REMOTE, iterations, REMOTE))
        text = (result.stdout + result.stderr).strip()
        if text:
            break
        time.sleep(6)
    match = LINE.search(text)
    if not match:
        return dict(passed=False, output=text[:160])
    return dict(passed=match.group(8) == "0" and match.group(9) == "0",
                tasks=int(match.group(2)), mode=match.group(3),
                min_us=float(match.group(4)), median_us=float(match.group(5)),
                unstable=int(match.group(8)), mismatches=int(match.group(9)))


def attempt(name, data, suite, index, iterations, attempts):
    reboot()
    Path("/tmp/job_field.bin").write_bytes(data)
    result = run(Path("/tmp/job_field.bin"), suite, index, iterations)
    record = dict(case=name, suite=suite, index=index, iterations=iterations, **result)
    attempts.append(record)
    (OUT / "board_results.json").write_text(json.dumps(attempts, indent=2) + "\n")
    print("%-28s tasks=%-2s %-7s exact=%-5s min=%7.1fus mismatches=%d"
          % (name, result.get("tasks"), result.get("mode"), result.get("passed"),
             result.get("min_us") or 0, result.get("mismatches") or 0), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--harness", default="/tmp/board_bench")
    parser.add_argument("--max-candidates", type=int, default=len(CANDIDATES))
    args = parser.parse_args()
    adb("shell", "mkdir -p %s" % REMOTE)
    adb("push", args.harness, REMOTE + "/board_bench")
    attempts = []

    pair = (ROOT / "mnist_pool_suite" / "model000.bin").read_bytes()
    _, info = payload_base(pair)
    assert engines(info) == ["CNA", "DPU"], "pair shape changed"
    attempt("pair-serial-committed", pair, "mnist_pool_suite", 0, args.iterations, attempts)
    cross = None
    for control in CANDIDATES[:args.max_candidates]:
        data = batched(pair, [control])
        result = attempt("pair-batched-cna-dpu-0x%02x" % control, data, "mnist_pool_suite", 0,
                         args.iterations, attempts)
        if result["passed"]:
            cross = control
            (TWINS / ("pair_cna_dpu_0x%02x.bin" % control)).write_bytes(data)
            break
    if cross is None:
        print("no CNA->DPU control passed; mixed batched lists stay refused", flush=True)

    join6 = (ROOT / "pool_join_suite" / "model000.bin").read_bytes()
    _, info = payload_base(join6)
    assert engines(info) == ["CNA", "CNA", "CNA", "DPU", "DPU", "DPU"], "join6 shape changed"
    attempt("join6-serial-committed", join6, "pool_join_suite", 0, args.iterations, attempts)
    dpu = None
    if cross is not None:
        for control in CANDIDATES[:args.max_candidates]:
            data = batched(join6, [0x40, 0x40, cross, control, control])
            result = attempt("join6-batched-dpu-dpu-0x%02x" % control, data, "pool_join_suite",
                             0, args.iterations, attempts)
            if result["passed"]:
                dpu = control
                (TWINS / ("join6_dpu_dpu_0x%02x.bin" % control)).write_bytes(data)
                break
        if dpu is None:
            print("no DPU->DPU control passed; multi-DPU mixed jobs stay refused", flush=True)

    head = (ROOT / "mixed_head_suite" / "model001.bin").read_bytes()
    _, info = payload_base(head)
    assert engines(info) == ["CNA"] * 4 + ["DPU", "DPU", "CNA"], "head shape changed"
    attempt("mixedhead7-serial-committed", head, "mixed_head_suite", 1, args.iterations, attempts)
    back = None
    if cross is not None and dpu is not None:
        for control in CANDIDATES[:args.max_candidates]:
            data = batched(head, [0x40, 0x40, 0x40, cross, dpu, control])
            result = attempt("mixedhead7-batched-dpu-cna-0x%02x" % control, data,
                             "mixed_head_suite", 1, args.iterations, attempts)
            if result["passed"]:
                back = control
                (TWINS / ("mixedhead7_dpu_cna_0x%02x.bin" % control)).write_bytes(data)
                break
        if back is None:
            print("no DPU->CNA control passed", flush=True)

    summary = dict(cna_dpu=cross, dpu_dpu=dpu, dpu_cna=back,
                   candidate_order=["0x%02x" % c for c in CANDIDATES],
                   attempts=len(attempts), exact=all(a["passed"] for a in attempts
                                                     if "serial" in a["case"]))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["CNA->DPU=%s DPU->DPU=%s DPU->CNA=%s over %d board attempts"
             % (("0x%02x" % cross) if cross is not None else "none",
                ("0x%02x" % dpu) if dpu is not None else "none",
                ("0x%02x" % back) if back is not None else "none", len(attempts))]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print(lines[0])


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT.parent / "src"))
    main()
