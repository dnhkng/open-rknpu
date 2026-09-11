"""Per-family cost table by task duplication (S5).

Differencing two different containers was below the board's noise floor
(`research/family_cost_probe/`). This probe instead repeats a *verified* task
descriptor N times inside one container: every copy reads and writes the same
addresses, so re-running it is idempotent and the expected bytes stay valid, while the
per-inference cost becomes `C + N * task`. Fitting that line over N gives a
per-family per-task cost that a slope is robust to load jitter.

Templates (all 8x8/C3, serial submission):

* `conv`   - one CNA Conv task (`family_cost_probe/model000.bin`);
* `pool`   - Conv + DPU pool, duplicating the pool descriptor (`model001.bin`);
* `elem`   - Conv + DPU elementwise Mul, duplicating the Mul descriptor (`model002.bin`).
"""
from pathlib import Path
import json
import struct

from open_rknpu.model import checksum
from open_rknpu.sequence import decode_sequence

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "family_cost_probe"
OUT = ROOT / "family_tasks_probe"
OUT.mkdir(exist_ok=True)
HEADER, DESCRIPTOR = 96, 16
COUNTS = (1, 2, 4, 8, 16, 32)
FAMILIES = (("conv", "model000", 0), ("pool", "model001", 1), ("elem", "model002", 1))


def resign(data):
    data = bytearray(data)
    data[80:84] = b"\0" * 4
    struct.pack_into("<I", data, 80, checksum(bytes(data)))
    return bytes(data)


def duplicate(base, index, copies):
    """Repeat descriptor `index` `copies` more times (the extra runs are idempotent).

    The task table sits at 96 for v3/v4 and after the 16-byte extension at 112 for v5;
    the tensor table (v5) follows it, so the payload start differs per version.
    """
    info = decode_sequence(base)
    count = info["task_count"]
    if info["format_version"] == 5:
        table_start = 112
        payload_start = 112 + 16 * count + 64 * info["tensor_count"]
    else:
        table_start = 96
        payload_start = 96 + 16 * count + 40 * info.get("constant_count", 0)
    table_end = table_start + count * DESCRIPTOR
    table = base[table_start:table_end]
    chosen = table[index * DESCRIPTOR:(index + 1) * DESCRIPTOR]
    # Keep everything between the task table and the payload (the v5 tensor table).
    result = (bytearray(base[:table_start]) + bytearray(table) + bytearray(chosen) * copies
              + bytearray(base[table_end:payload_start]) + bytearray(base[payload_start:]))
    struct.pack_into("<I", result, 60, count + copies)
    return resign(bytes(result))


def main():
    manifest = []
    for family, template, index in FAMILIES:
        base = (SOURCE / f"{template}.bin").read_bytes()
        base_info = decode_sequence(base)
        for copies in COUNTS:
            data = base if copies == 0 else duplicate(base, index, copies)
            info = decode_sequence(data)
            assert info["task_count"] == base_info["task_count"] + copies
            name = f"{family}_{copies + 1:02d}"
            (OUT / f"model_{name}.bin").write_bytes(data)
            for prefix, suffix in (("input", "u8"), ("expected", "i8")):
                source = SOURCE / f"{prefix}{int(template[5:]):03d}.{suffix}"
                (OUT / f"{prefix}_{name}.{suffix}").write_bytes(source.read_bytes())
            manifest.append(dict(name=name, family=family, copies=copies,
                                 tasks=info["task_count"],
                                 family_tasks=copies + 1,
                                 base_tasks=base_info["task_count"]))
            print(manifest[-1], flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("built", len(manifest), "family-task variants in", OUT)


if __name__ == "__main__":
    main()
