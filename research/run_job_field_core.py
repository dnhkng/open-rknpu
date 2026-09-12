"""S8 F1: are `core_mask` / `subcore_task[]` inert on RV1106?

the upstream `rknpu_job.c` ([provenance](../../vendor/README.md) from `research/`) forces `core_mask = RKNPU_CORE0_MASK` when the config has
one IRQ and reads `subcore_task[]` only when `num_irqs > 1`; RV1106 uses the single-entry
`rknpu_irqs`. The runtime now exposes `ornpu_set_submit_core(model, mask, windows, pairs)`
for the probe, and `tests/board_core.c` submits one verified container with several
mask/window sets, comparing the output bytes against the baseline and the expected file.

Evidence: `research/job_field_probe/core_fields.txt` and `core_fields.json`.
"""
from pathlib import Path
import os
import argparse
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "job_field_probe"
OUT.mkdir(exist_ok=True)
SUITE = ROOT / "mnist_pool_suite"
ADB = os.environ.get("ADB", "adb")
SERIAL = "498063e3262e55b7"
REMOTE = "/userdata/open-npu-research/job_field"
LINE = re.compile(r"^(\S+)\s+task=(\d+).*?runs=(\d+)\s+mismatches=(\d+)\s+"
                  r"changed=(\d+)\s+median=([\d.]+)us")


def adb(*args, timeout=180):
    return subprocess.run([ADB, "-s", SERIAL, *args], capture_output=True, text=True,
                          timeout=timeout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--harness", default="/tmp/board_core")
    args = parser.parse_args()
    adb("shell", "mkdir -p %s" % REMOTE)
    for src, dst in ((args.harness, "board_core"), (SUITE / "model000.bin", "t.bin"),
                     (SUITE / "input000.u8", "t.u8"),
                     (SUITE / "expected000.i8", "t.i8")):
        adb("push", str(src), "%s/%s" % (REMOTE, dst))
    run = adb("shell", "%s/board_core %s/t.bin %s/t.u8 %d %s/t.i8"
              % (REMOTE, REMOTE, REMOTE, args.iterations, REMOTE))
    text = (run.stdout + run.stderr).strip()
    print(text, flush=True)
    records = []
    for line in text.splitlines():
        match = LINE.match(line)
        if match:
            records.append(dict(config=match.group(1), tasks=int(match.group(2)),
                                runs=int(match.group(3)), mismatches=int(match.group(4)),
                                changed=match.group(5) == "1",
                                median_us=float(match.group(6))))
    (OUT / "core_fields.txt").write_text(
        "S8 F1: core_mask / subcore_task inertness on RV1103, tests/board_core.c,\n"
        "%d runs per configuration, `mnist_pool_suite/model000` (2 tasks CNA->DPU, "
        "serial, verified):\n\n%s\n" % (args.iterations, text))
    (OUT / "core_fields.json").write_text(json.dumps(records, indent=2) + "\n")
    if len(records) != 5 or any(r["mismatches"] or r["changed"] for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
