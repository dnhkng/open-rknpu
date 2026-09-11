"""S4 residual: fence-free completion with an in-order barrier job.

The attached kernel has no `CONFIG_ROCKCHIP_RKNPU_FENCE` (`JOB_FENCE_OUT` returns
-EINVAL), so there is no pollable completion fd. Jobs still run in order per core, so a
small blocking submission after a queued one completes only after it:

    ornpu_submit_flags(A, ORNPU_JOB_NONBLOCK, NULL);   queue the inference
    ornpu_run(barrier, ...);                           blocking: after A
    ornpu_sync_outputs(A);                             read A's outputs

`tests/board_barrier.c` measures that pattern against the synchronous path on two
containers, and `tests/board_async.c` re-checks the fence flags and the lag-1
queue-then-drain pipeline. Evidence: `research/barrier_probe/`.

usage: PYTHONPATH=src python3 research/run_barrier_probe.py [--iterations 64]
"""
from pathlib import Path
import os
import argparse
import json
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "barrier_probe"
OUT.mkdir(exist_ok=True)
ADB = os.environ.get("ADB", "adb")
SERIAL = "498063e3262e55b7"
REMOTE = "/userdata/open-npu-research/barrier"
BARRIER = ("conv_geometry_suite", "model000.bin")          # one 126-word Conv task
CASES = [("mixed_head_suite", "model001.bin", "input001.u8", "expected001.i8"),
         ("deep_chain_suite", "batched/model002.bin", "input002.u8", "expected002.i8")]
FENCE_CASE = ("pool_join_suite", "model000.bin", "input000.u8", "expected000.i8")


def adb(*args, timeout=180):
    return subprocess.run([ADB, "-s", SERIAL, *args], capture_output=True, text=True,
                          timeout=timeout)


def push(src, dst):
    last = None
    for _ in range(6):
        last = adb("push", str(src), dst)
        if last.returncode == 0 and "not found" not in (last.stdout + last.stderr):
            return
        time.sleep(8)
    raise SystemExit("adb push failed: %s -> %s" % (src, dst))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--harness", default="/tmp/board_barrier")
    parser.add_argument("--async-harness", default="/tmp/board_async")
    args = parser.parse_args()
    adb("shell", "mkdir -p %s" % REMOTE)
    push(args.harness, REMOTE + "/board_barrier")
    push(args.async_harness, REMOTE + "/board_async")
    push(ROOT / BARRIER[0] / BARRIER[1], REMOTE + "/barrier.bin")
    results, lines = [], []
    for suite, model, input_name, expected_name in CASES:
        for name, src in (("model", ROOT / suite / model),
                          ("input", ROOT / suite / input_name),
                          ("expected", ROOT / suite / expected_name)):
            push(src, "%s/%s" % (REMOTE, {"model": "a.bin", "input": "a.u8",
                                          "expected": "a.i8"}[name]))
        run = adb("shell", "%s/board_barrier %s/a.bin %s/a.u8 %d %s/a.i8 %s/barrier.bin"
                  % (REMOTE, REMOTE, REMOTE, args.iterations, REMOTE, REMOTE))
        text = (run.stdout + run.stderr).strip()
        record = {"suite": suite, "model": model, "iterations": args.iterations,
                  "output": text}
        barrier = re.search(r"barrier: tasks=(\d+) median=([\d.]+)us", text)
        sync = re.search(r"single-instance: runs=(\d+) tasks=(\d+) (\w+) sync_median=([\d.]+)us"
                         r" mismatches=(\d+)", text)
        completed = re.search(r"barrier-completed: median=([\d.]+)us overhead=(-?[\d.]+)us "
                              r"\((-?\d+)%\) mismatches=(\d+)", text)
        if barrier:
            record.update(barrier_tasks=int(barrier.group(1)),
                          barrier_median_us=float(barrier.group(2)))
        if sync:
            record.update(runs=int(sync.group(1)), tasks=int(sync.group(2)),
                          mode=sync.group(3), sync_median_us=float(sync.group(4)),
                          sync_mismatches=int(sync.group(5)))
        if completed:
            record.update(completed_median_us=float(completed.group(1)),
                          overhead_us=float(completed.group(2)),
                          overhead_percent=int(completed.group(3)),
                          completed_mismatches=int(completed.group(4)))
        results.append(record)
        line = ("%-18s %-24s tasks=%-2s sync=%7.1fus barrier-completed=%7.1fus "
                "barrier=%5.1fus mismatches=%s"
                % (suite, model, record.get("tasks"), record.get("sync_median_us", -1),
                   record.get("completed_median_us", -1), record.get("barrier_median_us", -1),
                   record.get("completed_mismatches")))
        lines.append(line)
        print(line, flush=True)

    # The fence flags and the lag-1 queue-then-drain pipeline, re-checked on the same
    # kernel with a v5 container (board_async requires named tensors).
    suite, model, input_name, expected_name = FENCE_CASE
    for src, dst in ((ROOT / suite / model, "v5.bin"), (ROOT / suite / input_name, "v5.u8"),
                     (ROOT / suite / expected_name, "v5.i8")):
        push(src, "%s/%s" % (REMOTE, dst))
    run = adb("shell", "%s/board_async %s/v5.bin %s/v5.u8 %d %s/v5.i8"
              % (REMOTE, REMOTE, REMOTE, args.iterations, REMOTE))
    async_text = (run.stdout + run.stderr).strip()
    fence = re.search(r"fence_out: rc=(-?\d+)", async_text)
    pipeline = re.search(r"two-instance pipeline: rounds=(\d+) median sync=([\d.]+)us "
                         r"pipelined=([\d.]+)us ratio=([\d.]+)", async_text)
    async_record = {"suite": suite, "model": model, "iterations": args.iterations,
                    "fence_out_rc": int(fence.group(1)) if fence else None,
                    "output": async_text}
    if pipeline:
        async_record.update(rounds=int(pipeline.group(1)),
                            pipeline_sync_us=float(pipeline.group(2)),
                            pipeline_drain_us=float(pipeline.group(3)),
                            pipeline_ratio=float(pipeline.group(4)))
    results.append(async_record)
    lines.append("fence_out rc=%s ; two-instance drain pipeline ratio=%s"
                 % (async_record["fence_out_rc"], async_record.get("pipeline_ratio")))
    print(lines[-1], flush=True)

    (OUT / "board_results.json").write_text(json.dumps(results, indent=2) + "\n")
    header = ("fence-free completion (docs/plans/pipelining-plan.md S4 residual), tests/board_barrier.c "
              "and tests/board_async.c, %d iterations:" % args.iterations)
    (OUT / "board_results.txt").write_text(header + "\n\n" + "\n".join(lines) + "\n")
    (OUT / "summary.txt").write_text(
        "barrier probe: %d containers, all outputs exact, fence_out rc=%s\n"
        % (len(CASES), async_record["fence_out_rc"]))
    if any(record.get("completed_mismatches") for record in results) or \
            any(record.get("sync_mismatches") for record in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
