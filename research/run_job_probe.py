"""S2 job-shape probe: links, job depth and engine mixing, with corrected links.

The next-command link is **payload-relative** (the front end fetches it from
`PC_DMA_BASE_ADDR` = the payload base), so a chain from program 0 to program 0x440
writes `0x10 = 0x440`. An earlier probe wrote `payload + 0x440` and mis-read every
linked case; this probe rebuilds them.

Cases (each after a clean boot; a failed job wedges the NPU until reboot):

* A2  two CNA tasks, terminal tails, batched      -> does a pair need links at all?
* A2L two CNA tasks, linked, batched              -> the known-good shape
* A3  three CNA tasks, linked, batched            -> job depth 3
* A4  four descriptors on a two-program loop      -> job depth 4 (idempotent)
* AX  two tasks, mixed CNA+DPU, linked, batched   -> does linking rescue mixing?

Evidence: `research/group_probe/{job_results.json,summary.txt,README.md}`.
"""
from pathlib import Path
import json
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "group_probe"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

from open_rknpu.model import checksum  # noqa: E402
from open_rknpu.sequence import decode_sequence  # noqa: E402
from run_group_probe import LINE, adb, reboot  # noqa: E402

REMOTE = "/userdata/open-npu-research/group_probe"
HEADER, DESCRIPTOR = 96, 16


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


def link(data, position, target, control=0x40):
    """Set task `position`'s next-command link to the payload-relative `target`."""
    info = decode_sequence(data)
    base = 96 + 16 * info["task_count"] + 40 * info.get("constant_count", 0)
    task = info["tasks"][position]
    offset = base + task["command_offset"] + task["register_count"] * 8
    data = bytearray(data)
    struct.pack_into("<Q", data, offset, 0x101 << 48 | target << 16 | 0x10)
    struct.pack_into("<Q", data, offset + 8, 0x101 << 48 | control << 16 | 0x14)
    return resign(bytes(data))


def set_masks(data, value):
    """Set every task descriptor's interrupt mask."""
    info = decode_sequence(data)
    data = bytearray(data)
    for position in range(info["task_count"]):
        struct.pack_into("<I", data, 112 + position * 16 + 12, value)
    return resign(bytes(data))


def link_all(data):
    """Link every task to the next program in execution (descriptor) order."""
    info = decode_sequence(data)
    targets = [task["command_offset"] for task in info["tasks"]]
    return chain(data, targets[1:] + [None])


def chain(data, targets):
    """Link task k to `targets[k]`; a None target leaves the task terminal."""
    for position, target in enumerate(targets):
        if target is None:
            continue
        data = link(data, position, target)
    return data


def duplicate_table(data, copies=2):
    info = decode_sequence(data)
    count = info["task_count"]
    table = data[HEADER:HEADER + count * DESCRIPTOR]
    payload = data[HEADER + count * DESCRIPTOR:]
    result = bytearray(data[:HEADER]) + bytearray(table) * copies + bytearray(payload)
    struct.pack_into("<I", result, 60, count * copies)
    return resign(bytes(result))


def cases():
    out = []

    def committed(suite, index=0):
        return (ROOT / suite / f"model{index:03}.bin").read_bytes()

    pair = committed("chain_output_quantization_suite")   # 2 CNA, serial=0, linked
    info = decode_sequence(pair)
    offsets = [task["command_offset"] for task in info["tasks"]]
    terminal = committed("depthwise_asymmetric_suite")    # 2 CNA, serial=1, terminal
    out.append(("C1-2xCNA-terminal", "depthwise_asymmetric_suite", 0,
                set_mode(terminal, False), "2 CNA, terminal tails, batched"))
    out.append(("C2-2xCNA-linked", "depthwise_asymmetric_suite", 0,
                link(set_mode(terminal, False), 0, offsets[1]),
                "2 CNA, link 0x%x, per-task mask" % offsets[1]))
    three = committed("native_chain_suite")               # 3 CNA, serial, terminal
    out.append(("C3-3xCNA-linked", "native_chain_suite", 0,
                link_all(set_mode(three, False)), "3 CNA, linked, per-task mask"))
    four = chain(duplicate_table(pair, 2), [offsets[1], offsets[0], offsets[1], None])
    out.append(("C4-4xCNA-loop-linked", "chain_output_quantization_suite", 0,
                four, "4 descriptors over a 2-program loop, linked"))
    join = committed("pool_join_suite")                   # v5, 6 mixed tasks
    out.append(("C5-6mixed-linked-union", "pool_join_suite", 0,
                set_masks(link_all(set_mode(join, False)), 3840),
                "v5 6 tasks CNA+DPU, linked, union mask 0xF00"))
    out.append(("C6-6mixed-linked", "pool_join_suite", 0,
                link_all(set_mode(join, False)),
                "v5 6 tasks CNA+DPU, linked, per-task masks"))
    mixed = committed("mnist_pool_suite")                 # DPU + CNA, serial, terminal
    out.append(("C7-2mixed-linked-union", "mnist_pool_suite", 0,
                set_masks(link_all(set_mode(mixed, False)), 3840),
                "2 tasks DPU+CNA, linked, union mask"))
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--only", default=None)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    adb("shell", "mkdir -p %s" % REMOTE)
    adb("push", "/tmp/board_bench", REMOTE + "/board_bench")
    path = OUT / "job_results.json"
    results = json.loads(path.read_text()) if args.merge and path.exists() else []
    done = {entry["name"] for entry in results}
    for name, suite, index, data, why in cases():
        if (args.only and args.only not in name) or (name in done and not args.only):
            continue
        reboot()
        Path("/tmp/job.bin").write_bytes(data)
        for src, dst in (("/tmp/job.bin", "t.bin"),
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
                                 (text.splitlines() or ["(no output)"])[0][:76]), flush=True)
        path.write_text(json.dumps(results, indent=2) + "\n")
    passed = [entry["name"] for entry in results if entry["passed"]]
    summary = "job-shape probe: %d/%d completed (%s)" % (
        len(passed), len(results), ", ".join(passed) or "none")
    (OUT / "summary_jobs.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
