#!/usr/bin/env python3
"""SPDX-License-Identifier: MIT

Cost-model regression: pin the *shape of the emitted work*, not just the bytes.

`research/container_baseline.json` already pins the exact sha256 of every emitted
container, so a byte change cannot slip through. It says nothing about the cost
structure an unrelated refactor can inflate while the bytes look plausible: a
rewrite that splits one task into three, doubles the arena, or re-schedules the
same convolution into more engine blocks changes NPU time without changing what
the model computes. This harness recompiles a curated cross-family selection and
pins, per model:

* `tasks` - the number of engine tasks in the sequence;
* `blocks` - the total popcount of every task's enable mask, an estimate of the
  engine-block slots the scheduler reserved (the closest host-side proxy for
  engine time);
* `registers` - the total register writes across the task table;
* `arena_bytes`, `payload_bytes` - the workspace the runtime must reserve and the
  command payload it must submit;
* `bytes` - the container size (redundant with the baseline, kept for the report).

Compilation is deterministic, so the comparison is exact for every structural
field: a value may *shrink* (an improvement is reported, never a failure), but a
growth beyond the byte tolerance fails. A model that stops compiling (or starts
compiling after being pinned as a rejection) is a behaviour change and fails too.

The wall-clock time of the pass is printed for the reader but never enforced: it
is machine dependent and would only add flakiness.

    PYTHONPATH=src python3 research/perf_regression.py            # compare
    PYTHONPATH=src python3 research/perf_regression.py --update   # re-pin
"""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_rknpu.sequence import decode_sequence  # noqa: E402
from open_rknpu.scheduler import compile_sequence  # noqa: E402

DEFAULT_BASELINE = ROOT / "research" / "perf_baseline.json"

# One cross-section per emitter family, so a regression in any of them is visible.
SUITES = (
    "depthwise_suite", "depthwise_chain_suite", "depthwise_c16_suite",
    "chain_suite", "chain_k3_suite", "wide_k3_suite", "wide_k5_suite",
    "conv_asymmetric_suite", "stride2_suite", "group_dense_suite", "group_dilation_suite",
    "transpose_k5_dilation_suite", "native_large_kernels_combined_suite",
    "pooled_branches_suite", "pool_join_suite", "branch_join_suite", "join_dag_suite",
    "diamond_suite", "reduction_api_suite", "max_suite", "add_geometry_suite",
    "mul_suite", "mul_broadcast_suite", "standalone_mul_suite", "mul_reshape_suite",
    "per_channel_mul_suite", "mul_two_input_batch_suite", "leaky_suite", "prelu_public_suite",
    "lut_public_suite", "native_input_blocks_suite", "native_output_blocks_suite",
    "two_head_suite", "mixed_head_suite", "sequence_suite", "mnist_first_suite",
    "walk_chain_suite", "mel_kws_suite",
)
BYTE_TOLERANCE = 16          # a couple of header words may grow without being a cost regression
BYTE_TOLERANCE_FRACTION = 0.01


def selected_models():
    """Every pinned suite's first two containers, sorted, so the set cannot drift."""
    models = []
    for suite in SUITES:
        models.extend(sorted((ROOT / "research" / suite).glob("model*.onnx"))[:2])
    return models


def measure(model):
    """Compile one model and reduce the emitted sequence to its cost fields."""
    binary, _meta = compile_sequence(str(model))
    info = decode_sequence(bytes(binary))
    tasks = info["tasks"]
    return {
        "bytes": len(binary),
        "tasks": len(tasks),
        "blocks": sum(bin(int(task["mask"])).count("1") for task in tasks),
        "registers": sum(int(task["register_count"]) for task in tasks),
        "arena_bytes": int(info["arena_bytes"]),
        "payload_bytes": int(info["payload_bytes"]),
    }


def collect():
    record = {}
    for model in selected_models():
        key = str(model.relative_to(ROOT / "research"))
        try:
            record[key] = measure(model)
        except Exception as exc:  # the rejection is part of the pinned behaviour
            record[key] = {"error": type(exc).__name__}
    return record


def compare(baseline, current):
    """Return (window, failures, improvements): every structural field is exact."""
    failures = []
    improvements = []
    window = list(baseline)
    for key in current:
        if key not in window:
            window.append(key)
    for key in window:
        before, after = baseline.get(key), current.get(key)
        if before is None:
            failures.append(f"{key}: newly pinned model is absent from the baseline")
            continue
        if after is None:
            failures.append(f"{key}: model disappeared from the tree")
            continue
        if "error" in before or "error" in after:
            if before.get("error") != after.get("error"):
                failures.append(
                    f"{key}: compiled outcome changed {before.get('error', 'ok')} -> "
                    f"{after.get('error', 'ok')}")
            continue
        for field, value in before.items():
            got = after.get(field)
            if got is None:
                failures.append(f"{key}: {field} disappeared")
            elif field == "bytes":
                allowed = max(BYTE_TOLERANCE, int(value * BYTE_TOLERANCE_FRACTION))
                if got > value + allowed:
                    failures.append(f"{key}: bytes {value} -> {got} (over {allowed} tolerance)")
                elif got < value:
                    improvements.append(f"{key}: bytes {value} -> {got}")
            elif got > value:
                failures.append(f"{key}: {field} {value} -> {got}")
            elif got < value:
                improvements.append(f"{key}: {field} {value} -> {got}")
    return window, failures, improvements


def report(current, elapsed):
    rows = [(cost.get("blocks", 0), cost.get("registers", 0), key, cost)
            for key, cost in current.items() if "error" not in cost]
    rows.sort(reverse=True)
    print(f"models={len(current)} measured={len(rows)} seconds={elapsed:.2f}")
    print("  %-52s %6s %6s %8s %8s %8s" %
          ("model", "bytes", "tasks", "blocks", "regs", "arena"))
    for _blocks, _regs, key, cost in rows[:10]:
        print("  %-52s %6d %6d %8d %8d %8d" %
              (key, cost["bytes"], cost["tasks"], cost["blocks"],
               cost["registers"], cost["arena_bytes"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--update", action="store_true", help="re-pin the baseline")
    args = parser.parse_args()

    started = time.monotonic()
    current = collect()
    elapsed = time.monotonic() - started
    report(current, elapsed)

    baseline_path = Path(args.baseline)
    if args.update:
        baseline_path.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
        print(f"wrote {baseline_path.relative_to(ROOT)}: {len(current)} models")
        return 0

    if not baseline_path.is_file():
        print(f"no baseline at {baseline_path}; run with --update", file=sys.stderr)
        return 2
    baseline = json.loads(baseline_path.read_text())
    _window, failures, improvements = compare(baseline, current)
    for line in improvements:
        print(f"IMPROVED {line}")
    for line in failures:
        print(f"REGRESSION {line}", file=sys.stderr)
    print(f"baseline={len(baseline)} same={len(baseline) - len(failures)} "
          f"failures={len(failures)} improvements={len(improvements)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
