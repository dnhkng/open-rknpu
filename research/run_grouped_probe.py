"""S8/S10: the tail control is the successor's fetch amount, so every DAG is one job.

The tail's second word (register `0x14`) is the value the driver itself would program
into `PC_DATA_AMOUNT` for the task it links to:
`(regcfg_amount + 4 + 2 - 1)/2 - 1` (`rknpu_job.c`, RV1106 scale 2). The vendor captures
agree - `0x40` before a 126-word Conv, `0x14` before a 37-word pool, `0x28` before a
78-word elementwise task - and the earlier "engine hand-off" table was a misreading of
that coincidence. Its Conv->elementwise and DPU->CNA negatives were confounded as well:
the prefix of the run it probed carried `0x14` at a Conv->elementwise step, which is not
that transition's amount.

This runner has two parts.

* **Discovery** (`kind="discovery"`): container surgery on verified serial containers,
  one clean boot per case - a single Conv->elementwise transition with the successor's
  amount, the same transition with `0x14` as the retained counter-example, the
  elementwise->elementwise transition, and the vendor's Conv->Pool value.
* **Published** (`kind="published"`): every emitter that now honours
  `submission='batched'` - the four remaining DAG emitters (`diamond_tail`,
  `depthwise_join`, `pooled_dag`, `join_dag`) plus the composer/join-chain ones - built
  by `compile_sequence`, checked for serial byte-identity with the retained container,
  then run in both modes over the suite's own expected bytes. `tests/board_bench.c`
  reports `engine_runs=`, so the one-job claim is measured.

Evidence: `research/grouped_probe/{board_results.json,board_results.txt,manifest.json}`.
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
OUT = ROOT / "grouped_probe"
OUT.mkdir(exist_ok=True)
ADB = os.environ.get("ADB", "adb")
SERIAL = "498063e3262e55b7"
REMOTE = "/userdata/open-npu-research/grouped"
LINE = re.compile(r"runs=(\d+) tasks=(\d+) (\S+) min=([\d.]+)us median=([\d.]+)us "
                  r"mean=([\d.]+)us max=([\d.]+)us unstable=(\d+) mismatches=(\d+) "
                  r"engine_runs=(\d+)")
PUBLISHED = [("diamond_tail_suite", 0), ("depthwise_join_suite", 0), ("pooled_dag_suite", 0),
             ("join_dag_suite", 0), ("pool_join_suite", 0), ("pooled_branches_suite", 0),
             ("mixed_head_suite", 1), ("join_chain_suite", 5)]


def adb(*args, timeout=180):
    return subprocess.run([ADB, "-s", SERIAL, *args], capture_output=True, text=True,
                          timeout=timeout)


def push(src, dst):
    """Push a file, retrying the board's flaky USB link instead of using a stale one."""
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


def discovered():
    """(name, bytes, suite, index, expected_pass) surgery cases: one transition each."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT.parent / "src"))
    from open_rknpu.compose import amount_control
    from run_job_field_probe import payload_base, resign, tail
    cases = []

    def links(suite, index, link_map):
        data = (ROOT / suite / f"model{index:03}.bin").read_bytes()
        base, info = payload_base(data)
        offsets = [task["command_offset"] for task in info["tasks"]]
        out = bytearray(data)
        out[84:88] = (int.from_bytes(out[84:88], "little") & ~1).to_bytes(4, "little")
        out = bytearray(resign(bytes(out)))
        for position in range(info["task_count"]):
            control = link_map.get(position)
            out = bytearray(tail(bytes(out), base, info, position,
                                 0x28 if control is None else control,
                                 link=offsets[position + 1] if control is not None else 0))
        return resign(bytes(out))

    # mixed_head_suite/model000 = [Conv x4 (126 words), elementwise x2 (78 words)].
    amount78 = amount_control(78)
    cases.append(("conv->elementwise amount 0x%02x" % amount78,
                  links("mixed_head_suite", 0, {0: 0x40, 1: 0x40, 2: 0x40, 3: amount78}),
                  "mixed_head_suite", 0, True))
    cases.append(("conv->elementwise 0x14 (counter-example)",
                  links("mixed_head_suite", 0, {0: 0x40, 1: 0x40, 2: 0x40, 3: 0x14}),
                  "mixed_head_suite", 0, False))
    cases.append(("elementwise->elementwise amount 0x%02x" % amount78,
                  links("mixed_head_suite", 0, {0: 0x40, 1: 0x40, 2: 0x40, 3: amount78,
                                                4: amount78}),
                  "mixed_head_suite", 0, True))
    # mnist_pool_suite/model000 = [Conv (126), pool (37)]: the vendor's Conv->Pool value.
    cases.append(("conv->pool amount 0x%02x" % amount_control(37),
                  links("mnist_pool_suite", 0, {0: amount_control(37)}),
                  "mnist_pool_suite", 0, True))
    return cases


def build():
    sys.path.insert(0, str(ROOT.parent / "src"))
    from open_rknpu.compose import batched_layout, engine_runs
    from open_rknpu.scheduler import compile_sequence
    from open_rknpu.sequence import decode_sequence
    manifest = []
    for suite, index in PUBLISHED:
        source = ROOT / suite / f"model{index:03}.onnx"
        retained = (ROOT / suite / f"model{index:03}.bin").read_bytes()
        serial, _ = compile_sequence(source)
        if serial != retained:
            raise SystemExit("%s serial compile no longer matches the retained container"
                             % source)
        batched, meta = compile_sequence(source, submission="batched")
        info = decode_sequence(batched)
        reason = batched_layout(batched, info)
        if reason is not None:
            raise SystemExit("%s batched container rejected: %s" % (source, reason))
        name = "%s_model%03d_onejob.bin" % (suite, index)
        (OUT / name).write_bytes(batched)
        cases = (ROOT / suite / f"input{index:03}.u8").stat().st_size // info["input_bytes"]
        manifest.append(dict(suite=suite, index=index, file=name,
                             tasks=info["task_count"], input_bytes=info["input_bytes"],
                             output_bytes=info["output_bytes"], cases=min(4, cases),
                             meta_runs=meta.get("engine_runs"),
                             runs=[list(run) for run in engine_runs(batched, info)]))
        print("built %s: %d tasks, %d run(s) %s" % (name, info["task_count"],
              len(meta.get("engine_runs") or []), meta.get("engine_runs")), flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def bench(model, suite, index, entry, case, iterations):
    input_bytes = (ROOT / suite / f"input{index:03}.u8").read_bytes()
    expected_bytes = (ROOT / suite / f"expected{index:03}.i8").read_bytes()
    (Path("/tmp") / "grouped.u8").write_bytes(
        input_bytes[case * entry["input_bytes"]:(case + 1) * entry["input_bytes"]])
    (Path("/tmp") / "grouped.i8").write_bytes(
        expected_bytes[case * entry["output_bytes"]:(case + 1) * entry["output_bytes"]])
    push(model, REMOTE + "/t.bin")
    push("/tmp/grouped.u8", REMOTE + "/t.u8")
    push("/tmp/grouped.i8", REMOTE + "/t.i8")
    text = ""
    for _ in range(6):
        run = adb("shell", "%s/board_bench %s/t.bin %s/t.u8 %d %s/t.i8"
                  % (REMOTE, REMOTE, REMOTE, iterations, REMOTE))
        text = (run.stdout + run.stderr).strip()
        # An empty reply or an adb transport error is not a result: the board's USB
        # link re-enumerates under load, so wait and try again instead of recording it.
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

    # Discovery: container surgery, one clean boot per case.
    sys.path.insert(0, str(ROOT.parent / "src"))
    from open_rknpu.sequence import decode_sequence
    for name, data, suite, index, expected_pass in discovered():
        Path("/tmp/grouped.bin").write_bytes(data)
        info = decode_sequence(data)
        entry = dict(input_bytes=info["input_bytes"], output_bytes=info["output_bytes"])
        reboot()
        run = bench(Path("/tmp/grouped.bin"), suite, index, entry, 0, args.iterations)
        record = dict(kind="discovery", suite=suite, index=index, case=name,
                      passed=run["passed"], expected=expected_pass,
                      iterations=args.iterations, tasks=run.get("tasks"),
                      engine_runs=run.get("engine_runs"),
                      min_us=run.get("min_us") or 0, median_us=run.get("median_us") or 0,
                      mismatches=run.get("mismatches"), unstable=run.get("unstable"),
                      detail=run.get("output", ""))
        if not run["passed"]:
            reboot()
        results.append(record)
        line = ("%-42s tasks=%2s ioctls=%-2s exact=%-5s mismatches=%s"
                % (name, record["tasks"], record["engine_runs"], record["passed"],
                   record["mismatches"]))
        lines.append(line)
        print(line, flush=True)
        (OUT / "board_results.json").write_text(json.dumps(results, indent=2) + "\n")

    # Published: serial vs one linked job for every emitter that honours the request.
    for entry in manifest:
        suite, index = entry["suite"], entry["index"]
        reboot()
        for mode, model in (("serial", ROOT / suite / f"model{index:03}.bin"),
                            ("one-job", OUT / entry["file"])):
            runs = [bench(model, suite, index, entry, case, args.iterations)
                    for case in range(entry["cases"])]
            passed = all(run["passed"] for run in runs)
            record = dict(kind="published", suite=suite, index=index, mode=mode,
                          passed=passed, cases=len(runs), iterations=args.iterations,
                          tasks=runs[0].get("tasks"),
                          engine_runs=runs[0].get("engine_runs"),
                          min_us=min(r.get("min_us", 0) or 0 for r in runs),
                          median_us=max(r.get("median_us", 0) or 0 for r in runs),
                          mismatches=sum(r.get("mismatches", 0) or 0 for r in runs),
                          unstable=sum(r.get("unstable", 0) or 0 for r in runs),
                          detail=next((r.get("output", "") for r in runs
                                       if not r["passed"]), ""))
            results.append(record)
            line = ("%-20s model%03d %-7s tasks=%2s ioctls=%-2s cases=%d exact=%-5s "
                    "min=%7.1fus median=%8.1fus mismatches=%d"
                    % (suite, index, mode, record["tasks"], record["engine_runs"],
                       record["cases"], passed, record["min_us"], record["median_us"],
                       record["mismatches"]))
            lines.append(line)
            print(line, flush=True)
            (OUT / "board_results.json").write_text(json.dumps(results, indent=2) + "\n")
            if not passed:
                reboot()
    header = ("the tail control is the successor's fetch amount (docs/plans/pipelining-plan.md "
              "S8/S10), tests/board_bench.c, %d cases per mode, clean boot per model; "
              "`ioctls` is what the runtime reports (`ornpu_info.engine_runs`):"
              % args.iterations)
    (OUT / "board_results.txt").write_text(header + "\n\n" + "\n".join(lines) + "\n")
    published = [r for r in results if r["kind"] == "published"]
    discovery = [r for r in results if r["kind"] == "discovery"]
    summary = ("grouped: %d containers serial and one-job exact=%s, %d discovery cases "
               "as expected=%s\n" % (len(manifest),
                                     all(r["passed"] for r in published),
                                     len(discovery),
                                     all(r["passed"] == r["expected"]
                                         for r in discovery)))
    (OUT / "summary.txt").write_text(summary)
    print(summary)
    if not all(r["passed"] for r in published) or \
            not all(r["passed"] == r["expected"] for r in discovery):
        sys.exit(1)


if __name__ == "__main__":
    main()
