"""Build the batched-submission probe containers (S1 of docs/plans/pipelining-plan.md).

The submission mode is header bit 0 of the ORNPUSEQ flags word (byte 84), so flipping
it and re-checksumming produces a container whose programs, tensor table and arena
are byte-identical to a verified one. That isolates the submission mode: any output
or timeout difference is caused by how the job is submitted, not by the compiled
commands.

The cases kept here are the ones `research/run_submission_rule.py` runs on the board:

* two-task single-engine (`chain_output_quantization_suite`) - the only batched
  ORNPUSEQ container that was ever emitted and verified;
* two-task mixed-engine flag twins (CNA + DPU);
* a three-task single-engine chain with explicit next-command links;
* mixed-engine v5 flag twins, including the six-task pool join.

Evidence: `rule_results.json`, `board_summary.txt`, `README.md`.
"""
from pathlib import Path
import json
import struct

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submission_probe"
OUT.mkdir(exist_ok=True)

from open_rknpu.model import checksum  # noqa: E402
from open_rknpu.sequence import decode_sequence  # noqa: E402

ENGINE = {29: "CNA", 96: "DPU", 24: "DPU"}

# (suite, model index) whose committed container is the serial control, plus whether
# the batched twin should carry next-command links.
CASES = [
    ("chain_output_quantization_suite", 0, False),   # 2x CNA, committed batched
    ("mnist_pool_suite", 0, False),                  # DPU + CNA
    ("mul_broadcast_mode_suite", 0, False),          # DPU + CNA
    ("native_chain_suite", 0, True),                 # 3x CNA, links
    ("runtime_scale_suite", 0, False),               # v5 CNA + DPU
    ("pool_join_suite", 0, False),                   # v5 6 tasks
]


def payload_base(binary):
    """Offset of the command payload for an ORNPUSEQ container."""
    info = decode_sequence(binary)
    if info["format_version"] == 5:
        return 112 + 16 * info["task_count"] + 64 * info["tensor_count"], info
    return 96 + 16 * info["task_count"] + 40 * info.get("constant_count", 0), info


def with_submission(binary, serial):
    """Same container with the submission flag set as requested."""
    result = bytearray(binary)
    flags = struct.unpack_from("<I", result, 84)[0]
    flags = (flags | 1) if serial else (flags & ~1)
    struct.pack_into("<I", result, 84, flags)
    result[80:84] = b"\0" * 4
    struct.pack_into("<I", result, 80, checksum(bytes(result)))
    return bytes(result)


def with_links(binary):
    """Link each task to the next program (0x10) with control 0x40."""
    base, info = payload_base(binary)
    data = bytearray(binary)
    for position, task in enumerate(info["tasks"][:-1]):
        offset = base + task["command_offset"] + task["register_count"] * 8
        # The link is payload-relative (PC_DMA_BASE_ADDR is the payload base).
        target = info["tasks"][position + 1]["command_offset"]
        struct.pack_into("<Q", data, offset, 0x101 << 48 | target << 16 | 0x10)
        struct.pack_into("<Q", data, offset + 8, 0x101 << 48 | 0x40 << 16 | 0x14)
    data[80:84] = b"\0" * 4
    struct.pack_into("<I", data, 80, checksum(bytes(data)))
    return bytes(data)


def main():
    manifest = []
    for suite, index, link in CASES:
        folder = ROOT / suite
        committed = (folder / f"model{index:03}.bin").read_bytes()
        base, info = payload_base(committed)
        engines = sorted({ENGINE.get(task["enable"], "?") for task in info["tasks"]})
        entry = dict(suite=suite, model=index, tasks=info["task_count"],
                     engines=engines, serial=bool(info["serial"]),
                     committed=str((folder / f"model{index:03}.bin").relative_to(ROOT.parent)))
        manifest.append(dict(entry, variant="committed", file=None,
                             note="verified container as committed" if not info["serial"]
                                  else "verified serial control"))
        if not info["serial"]:
            continue
        twin = with_submission(committed, False)
        if link:
            twin = with_links(twin)
        info2 = decode_sequence(twin)
        assert info2["serial"] is False and info2["task_count"] == info["task_count"]
        name = f"{suite}-{index:03}-batched" + ("-linked" if link else "")
        path = OUT / f"{name}.bin"
        path.write_bytes(twin)
        manifest.append(dict(entry, variant="batched" + ("-linked" if link else ""),
                             file=str(path.relative_to(ROOT.parent)),
                             note="flag twin of the verified container"
                                  + (" plus next-command links" if link else "")))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("built", len(manifest), "probe entries in", OUT)


if __name__ == "__main__":
    main()
