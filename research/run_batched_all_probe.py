"""S10 residual: every container can be submitted as one linked job.

`sequence.relink_for_batched(data)` rewrites only the task tails (next-program link plus
the successor's fetch amount, `amount_control`) and the header submission flag; the
programs are untouched. `compile_sequence(..., submission='batched')` applies it to
whatever an emitter produced, so a profile whose emitter never learned about the
submission mode still runs as one job.

This probe takes retained, board-verified serial containers from suites whose emitters
build their containers by hand or one task at a time, checks the emitter path
(`compile_sequence`) agrees with the post-pass where the ONNX still compiles to the
retained bytes, and runs both modes on the board against the suite's own expected bytes.

Evidence: `research/batched_all_probe/{board_results.json,board_results.txt,manifest.json}`.
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
OUT = ROOT / "batched_all_probe"
OUT.mkdir(exist_ok=True)
ADB = os.environ.get("ADB", "adb")
SERIAL = "498063e3262e55b7"
REMOTE = "/userdata/open-npu-research/batched_all"
LINE = re.compile(r"runs=(\d+) tasks=(\d+) (\S+) min=([\d.]+)us median=([\d.]+)us "
                  r"mean=([\d.]+)us max=([\d.]+)us unstable=(\d+) mismatches=(\d+) "
                  r"engine_runs=(\d+)")
CASES = [("add_suite", 0), ("mul_batch_suite", 1), ("mul_batch_suite", 2),
         ("mul_batch_broadcast_suite", 3), ("elementwise_deep_suite", 2),
         ("add_geometry_suite", 0), ("conv_geometry_suite", 0)]


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
    raise SystemExit("adb push failed: %s -> %s (%s)"
                     % (src, dst, (last.stdout + last.stderr).strip()[:120]))


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


def build():
    sys.path.insert(0, str(ROOT.parent / "src"))
    from open_rknpu.compose import batched_layout, engine_runs
    from open_rknpu.scheduler import compile_sequence
    from open_rknpu.sequence import decode_sequence, relink_for_batched
    manifest = []
    for suite, index in CASES:
        retained = (ROOT / suite / f"model{index:03}.bin").read_bytes()
        info = decode_sequence(retained)
        batched = relink_for_batched(retained)
        reason = batched_layout(batched, decode_sequence(batched))
        if reason is not None:
            raise SystemExit("%s/%d post-pass rejected: %s" % (suite, index, reason))
        name = "%s_model%03d_onejob.bin" % (suite, index)
        (OUT / name).write_bytes(batched)
        # The emitter path must agree with the post-pass wherever the ONNX still
        # compiles to the retained serial container.
        onnx_path = ROOT / suite / f"model{index:03}.onnx"
        emitter = None
        if onnx_path.exists():
            try:
                serial, _ = compile_sequence(onnx_path)
                if serial == retained:
                    emitter, _ = compile_sequence(onnx_path, submission="batched")
            except ValueError:
                emitter = None
        runs = engine_runs(batched, decode_sequence(batched))
        manifest.append(dict(suite=suite, index=index, file=name,
                             tasks=info["task_count"],
                             enables=[task["enable"] for task in info["tasks"]],
                             input_bytes=info["input_bytes"],
                             output_bytes=info["output_bytes"],
                             cases=min(4, (ROOT / suite / f"input{index:03}.u8").stat().st_size
                                       // info["input_bytes"]),
                             runs=[list(run) for run in runs],
                             emitter_matches=(emitter == batched) if emitter else None))
        print("built %s: %d tasks %s -> %s" % (name, info["task_count"],
              [task["enable"] for task in info["tasks"]], manifest[-1]["emitter_matches"]),
              flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def bench(model, suite, index, entry, case, iterations):
    input_bytes = (ROOT / suite / f"input{index:03}.u8").read_bytes()
    expected_bytes = (ROOT / suite / f"expected{index:03}.i8").read_bytes()
    (Path("/tmp") / "batched_all.u8").write_bytes(
        input_bytes[case * entry["input_bytes"]:(case + 1) * entry["input_bytes"]])
    (Path("/tmp") / "batched_all.i8").write_bytes(
        expected_bytes[case * entry["output_bytes"]:(case + 1) * entry["output_bytes"]])
    push(model, REMOTE + "/t.bin")
    push("/tmp/batched_all.u8", REMOTE + "/t.u8")
    push("/tmp/batched_all.i8", REMOTE + "/t.i8")
    text = ""
    for _ in range(6):
        run = adb("shell", "%s/board_bench %s/t.bin %s/t.u8 %d %s/t.i8"
                  % (REMOTE, REMOTE, REMOTE, iterations, REMOTE))
        text = (run.stdout + run.stderr).strip()
        if text and "not found" not in text and "device offline" not in text:
            break
        time.sleep(8)
    match = LINE.search(text)
    if not match:
        return dict(passed=False, case=case, output=text[:160])
    return dict(passed=match.group(8) == "0" and match.group(9) == "0", case=case,
                tasks=int(match.group(2)), mode=match.group(3),
                min_us=float(match.group(4)), median_us=float(match.group(5)),
                unstable=int(match.group(8)), mismatches=int(match.group(9)),
                engine_runs=int(match.group(10)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--harness", default="/tmp/board_bench")
    args = parser.parse_args()
    manifest = build()
    adb("shell", "mkdir -p %s" % REMOTE)
    push(args.harness, REMOTE + "/board_bench")
    results, lines = [], []
    for entry in manifest:
        suite, index = entry["suite"], entry["index"]
        reboot()
        for mode, model in (("serial", ROOT / suite / f"model{index:03}.bin"),
                            ("one-job", OUT / entry["file"])):
            runs = [bench(model, suite, index, entry, case, args.iterations)
                    for case in range(entry["cases"])]
            passed = all(run["passed"] for run in runs)
            record = dict(suite=suite, index=index, mode=mode, passed=passed,
                          cases=len(runs), iterations=args.iterations,
                          tasks=runs[0].get("tasks"),
                          engine_runs=runs[0].get("engine_runs"),
                          min_us=min(r.get("min_us", 0) or 0 for r in runs),
                          median_us=max(r.get("median_us", 0) or 0 for r in runs),
                          mismatches=sum(r.get("mismatches", 0) or 0 for r in runs),
                          unstable=sum(r.get("unstable", 0) or 0 for r in runs),
                          detail=next((r.get("output", "") for r in runs
                                       if not r["passed"]), ""))
            results.append(record)
            line = ("%-26s model%03d %-7s tasks=%2s ioctls=%-2s cases=%d exact=%-5s "
                    "min=%7.1fus median=%8.1fus mismatches=%d"
                    % (suite, index, mode, record["tasks"], record["engine_runs"],
                       record["cases"], passed, record["min_us"], record["median_us"],
                       record["mismatches"]))
            lines.append(line)
            print(line, flush=True)
            (OUT / "board_results.json").write_text(json.dumps(results, indent=2) + "\n")
            if not passed:
                reboot()
    header = ("one linked job for containers whose emitters never knew about the mode "
              "(docs/plans/pipelining-plan.md S10 residual), tests/board_bench.c, %d cases per mode, "
              "clean boot per model; `ioctls` is `ornpu_info.engine_runs`:" % args.iterations)
    (OUT / "board_results.txt").write_text(header + "\n\n" + "\n".join(lines) + "\n")
    summary = ("batched-all: %d containers, both modes exact=%s\n"
               % (len(manifest), all(r["passed"] for r in results)))
    (OUT / "summary.txt").write_text(summary)
    print(summary)
    if not all(r["passed"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
