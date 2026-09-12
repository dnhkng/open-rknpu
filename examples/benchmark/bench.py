"""SPDX-License-Identifier: MIT

`examples/benchmark/bench.py` - a reusable sizing harness for `open-rknpu`
containers and the ONNX models they were compiled from.

The harness has two modes and one table shape:

* **host** (the default) never touches a board. For every container it decodes the
  header and reports the container size, `task_count`, the `engine_runs` ioctl
  structure and the wall-clock time of a fresh compile of the matching
  `model*.onnx` next to it, when one is present. This is where a model is sized:
  the task count, the ioctl count and the container bytes are fixed by the
  compiler and are reproducible anywhere; the compile time is machine dependent.
* **board** (`--board`) drives the repository's board runners over `adb` and
  reports one row per model:
  `model | inferences | exact bytes | ms/run | runs/s`. The default runner
  (`--runner io`) is `tests/board_io.c`'s staging protocol from
  `research/run_v5_suite.py`: each model's recorded `inputNNN.u8` / `expectedNNN.i8`
  is staged under its own directory and the whole submission is wall-clocked from
  the host and verified byte for byte, so its `ms/run` additionally includes the
  `adb` round trip, fixture I/O and `ornpu_open`. `--runner bench` runs
  `tests/board_bench.c` instead, which samples `clock_gettime(CLOCK_MONOTONIC)`
  immediately around each `ornpu_run`, so its `ms/run` is the same in-process
  measurement `docs/performance.md` tabulates.

Everything outside `--board` is host-only, deterministic and offline: no vendor
compiler is invoked, and the board binaries are built by the user exactly as
`docs/performance.md` documents. See `examples/benchmark/README.md` for the
measurement protocol, the definitions of warm-up/repetitions/percentiles, and the
recorded table.
"""
from pathlib import Path
import argparse
import os
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_rknpu.compose import engine_runs as container_engine_runs  # noqa: E402
from open_rknpu.model import decode  # noqa: E402
from open_rknpu.scheduler import compile_sequence  # noqa: E402

RESEARCH = ROOT / "research"
MODEL_INDEX = re.compile(r"model(\d+)")
BENCH_LINE = re.compile(
    r"runs=(?P<runs>\d+) tasks=\d+ (?P<mode>serial|batched) "
    r"min=(?P<min>[\d.]+)us median=(?P<median>[\d.]+)us mean=(?P<mean>[\d.]+)us "
    r"max=(?P<max>[\d.]+)us unstable=(?P<unstable>\d+) mismatches=(?P<mismatches>\d+) "
    r"engine_runs=(?P<engine_runs>\d+)")
IO_MODEL = re.compile(r"^model (\d+): (\d+) inputs passed \((\d+) outputs\)$", re.MULTILINE)
IO_PASS = re.compile(r"^PASS: \d+ v5 models, (\d+) inferences, (\d+) exact output bytes",
                     re.MULTILINE)
STATS = {"min": 0.0, "median": 0.5, "mean": None, "p90": 0.9}


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def display(path):
    """The repository-relative spelling of a selected path, when it is inside it."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def select(args):
    """The containers/ONNX models to measure, de-duplicated and sorted."""
    chosen, seen = [], set()

    def add(path):
        if path.suffix not in (".bin", ".onnx"):
            return
        key = str(path)
        if key not in seen:
            seen.add(key)
            chosen.append(path)

    for token in args.paths:
        add(Path(token))
    for name in args.suite:
        directory = RESEARCH / name
        for path in sorted(directory.glob("model*.bin")):
            add(path)
    for pattern in args.glob:
        for path in sorted(ROOT.glob(pattern)):
            add(path)
    return sorted(chosen, key=display)


# --------------------------------------------------------------------------- #
# Host mode
# --------------------------------------------------------------------------- #
def api_input_bytes(info):
    """The flat packed NHWC bytes one inference reads, across every input tensor."""
    tensors = info.get("input_tensors")
    if tensors:
        return sum(t["batch"] * t["height"] * t["width"] * t["channels"] for t in tensors)
    return info["input_bytes"]


def host_row(path, compile_enabled=True):
    """One decoded container (or compiled ONNX model) as a table row."""
    row = {"model": display(path), "bytes": "-", "task_count": "-",
           "engine_runs": "-", "compile_ms": "-"}
    if path.suffix == ".onnx":
        try:
            started = time.perf_counter()
            data, _meta = compile_sequence(str(path))
            data = bytes(data)
            row["compile_ms"] = "%.1f" % ((time.perf_counter() - started) * 1000.0)
        except (ValueError, KeyError, OSError) as exc:
            print("bench: %s: %s" % (display(path), exc), file=sys.stderr)
            return row
    else:
        data = path.read_bytes()
    try:
        info = decode(data)
    except (ValueError, KeyError) as exc:
        print("bench: %s: %s" % (display(path), exc), file=sys.stderr)
        row["bytes"] = len(data)
        return row
    row["bytes"] = len(data)
    row["task_count"] = info["task_count"]
    if "tasks" in info:  # ORNPUSEQ v3/v4/v5; a legacy ORNPUBIN has one program, one ioctl
        row["engine_runs"] = len(container_engine_runs(data, info))
    if path.suffix == ".bin" and compile_enabled:
        sibling = path.with_suffix(".onnx")
        if sibling.is_file():
            try:
                started = time.perf_counter()
                compile_sequence(str(sibling))
                row["compile_ms"] = "%.1f" % ((time.perf_counter() - started) * 1000.0)
            except (ValueError, KeyError, OSError) as exc:
                print("bench: %s: %s" % (display(sibling), exc), file=sys.stderr)
    return row


def host_rows(args, paths):
    return [host_row(path, compile_enabled=not args.no_compile) for path in paths]


# --------------------------------------------------------------------------- #
# Board mode
# --------------------------------------------------------------------------- #
def percentile(values, fraction):
    """The nearest-rank percentile of a small sample, clamped to the sample."""
    ordered = sorted(values)
    index = int(round(fraction * (len(ordered) - 1)))
    return ordered[min(max(index, 0), len(ordered) - 1)]


def selected_stat(values, stat):
    if STATS[stat] is None:
        return sum(values) / len(values)
    return percentile(values, STATS[stat])


def board_prefix(args):
    prefix = [args.tool]
    if args.serial:
        prefix += ["-s", args.serial]
    return prefix


def board_run(args, *command, timeout=900):
    """Run one adb command; the caller decides whether its output is a failure."""
    try:
        return subprocess.run(board_prefix(args) + list(command), capture_output=True,
                              text=True, errors="replace", timeout=timeout)
    except FileNotFoundError:
        raise SystemExit("bench: cannot run %r; install adb or pass --adb PATH" % args.tool) from None
    except subprocess.TimeoutExpired:
        raise SystemExit("bench: adb %s timed out after %s s"
                         % (" ".join(command), timeout)) from None


def board_check(args, *command, timeout=900):
    result = board_run(args, *command, timeout=timeout)
    if result.returncode != 0:
        raise SystemExit("bench: adb %s failed:\n%s%s"
                         % (" ".join(command), result.stdout, result.stderr))
    return result


def push(args, source, destination):
    board_check(args, "push", str(source), destination)


def fixture_paths(model):
    """The `inputNNN.u8` / `expectedNNN.i8` files that belong to `modelNNN.bin`."""
    index = MODEL_INDEX.search(model.stem)
    suffix = index.group(1) if index else ""
    return model.with_name("input%s.u8" % suffix), model.with_name("expected%s.i8" % suffix)


def stage_model(args, model, key):
    """Stage one model and its recorded case under its own remote directory."""
    remote = "%s/%s" % (args.remote.rstrip("/"), key)
    board_check(args, "shell", "rm -rf %s && mkdir -p %s" % (remote, remote))
    push(args, args.runner_binary, "%s/%s" % (remote, args.runner_name))
    push(args, model, "%s/model000.bin" % remote)
    input_path, expected_path = fixture_paths(model)
    if input_path.is_file():
        push(args, input_path, "%s/input000.u8" % remote)
    if expected_path.is_file():
        push(args, expected_path, "%s/expected000.i8" % remote)
    return remote, input_path, expected_path


def bench_iterations(args, model, input_path):
    """How many `board_bench` iterations one invocation runs."""
    if args.iterations:
        return args.iterations
    info = decode(model.read_bytes())
    case_bytes = api_input_bytes(info)
    if input_path.is_file() and case_bytes:
        return max(1, input_path.stat().st_size // case_bytes)
    return 1


def board_bench_model(args, model):
    """Time one model with `tests/board_bench.c`; return `(row, note)`."""
    row = {"model": display(model), "inferences": "-", "exact bytes": "-",
           "ms/run": "-", "runs/s": "-"}
    input_path, expected_path = fixture_paths(model)
    if not input_path.is_file():
        return row, "no input%s.u8 beside it" % (MODEL_INDEX.search(model.stem).group(1)
                                                 if MODEL_INDEX.search(model.stem) else "")
    iterations = bench_iterations(args, model, input_path)
    key = display(model).replace("/", "_").replace(".", "_")
    remote, _input, _expected = stage_model(args, model, key)
    command = "cd %s && ./%s model000.bin input000.u8 %d%s" % (
        remote, args.runner_name, iterations,
        " expected000.i8" if expected_path.is_file() else "")
    # One untimed invocation warms the adb/open path; board_bench additionally warms
    # up min(iterations/4, 8) times inside every invocation (docs/performance.md).
    board_run(args, "shell", command)
    samples = {"min": [], "median": [], "mean": []}
    runs, mismatches = 0, 0
    for _ in range(args.repeats):
        board_check(args, "shell", "sync")
        result = board_check(args, "shell", command)
        match = BENCH_LINE.search(result.stdout)
        if match is None:
            raise SystemExit("bench: %s: no measurement line in:\n%s"
                             % (display(model), result.stdout + result.stderr))
        runs = int(match.group("runs"))
        mismatches += int(match.group("mismatches"))
        for field in samples:
            samples[field].append(float(match.group(field)) / 1000.0)
    # `min` and `mean` use board_bench's own per-invocation statistic; `median`/`p90`
    # are taken over the per-invocation medians (one sample per repeat).
    source = "mean" if args.stat == "mean" else "min" if args.stat == "min" else "median"
    value = selected_stat(samples[source], args.stat)
    row["inferences"] = runs * args.repeats
    row["ms/run"] = "%.3f" % value
    row["runs/s"] = "%.1f" % (1000.0 / value) if value else "-"
    if expected_path.is_file() and mismatches == 0:
        per_case = expected_path.stat().st_size // max(1, iterations)
        row["exact bytes"] = runs * per_case * args.repeats
    return row, None


def board_io_model(args, model):
    """Time one model's whole recorded submission with `tests/board_io.c`."""
    row = {"model": display(model), "inferences": "-", "exact bytes": "-",
           "ms/run": "-", "runs/s": "-"}
    if decode(model.read_bytes())["format_version"] != 5:
        return row, "board_io needs a v5 named-tensor container (this one is legacy/v3/v4)"
    input_path, expected_path = fixture_paths(model)
    if not input_path.is_file() or not expected_path.is_file():
        return row, "needs inputNNN.u8 and expectedNNN.i8 beside it"
    key = display(model).replace("/", "_").replace(".", "_")
    remote, _input, _expected = stage_model(args, model, key)
    command = "cd %s && ./%s . 1" % (remote, args.runner_name)
    board_run(args, "shell", command)  # warm-up, discarded
    samples, inferences, exact_per_repeat = [], None, None
    for _ in range(args.repeats):
        board_check(args, "shell", "sync")  # flush writeback so the next run is not charged
        started = time.perf_counter()
        result = board_check(args, "shell", command)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        match = IO_MODEL.search(result.stdout)
        if match is None:
            raise SystemExit("bench: %s: no board_io result line in:\n%s"
                             % (display(model), result.stdout + result.stderr))
        inferences = int(match.group(2))
        samples.append(elapsed_ms / max(1, inferences))
        summary = IO_PASS.search(result.stdout)
        if summary:
            exact_per_repeat = int(summary.group(2))
    value = selected_stat(samples, args.stat)
    row["inferences"] = (inferences or 0) * args.repeats
    row["ms/run"] = "%.3f" % value
    row["runs/s"] = "%.1f" % (1000.0 / value) if value else "-"
    if exact_per_repeat is not None:
        row["exact bytes"] = exact_per_repeat * args.repeats
    return row, None


def board_rows(args, paths):
    rows = []
    for path in paths:
        if path.suffix == ".onnx":
            raise SystemExit("bench: %s is an ONNX model; --board needs a compiled .bin "
                             "container (compile it with open-rknpu first)" % display(path))
        try:
            row, note = (board_bench_model if args.runner == "bench" else board_io_model)(args, path)
        finally:
            # /userdata is a ~4 MB flash partition: never leave a staged model behind, not
            # even when the run raises. `--keep-remote` keeps it for debugging.
            if not args.keep_remote:
                key = display(path).replace("/", "_").replace(".", "_")
                board_run(args, "shell", "rm -rf %s/%s" % (args.remote.rstrip("/"), key))
        if note:
            print("# skipped %s: %s" % (display(path), note), file=sys.stderr)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_table(headers, rows, markdown):
    if markdown:
        lines = ["| " + " | ".join(headers) + " |",
                 "| " + " | ".join("---:" if i else "---" for i in range(len(headers))) + " |"]
        lines += ["| " + " | ".join(str(row.get(header, "-")) for header in headers) + " |"
                  for row in rows]
        return "\n".join(lines)
    widths = [len(header) for header in headers]
    for row in rows:
        for index, header in enumerate(headers):
            widths[index] = max(widths[index], len(str(row.get(header, "-"))))
    lines = ["  ".join(header.rjust(widths[index]) for index, header in enumerate(headers))]
    lines.append("  ".join("-" * width for width in widths))
    lines += ["  ".join(str(row.get(header, "-")).rjust(widths[index])
                        for index, header in enumerate(headers)) for row in rows]
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Latency/throughput and structure table for open-rknpu containers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="host: bytes, task_count, engine_runs and compile time; no board, no network.\n"
               "board: model | inferences | exact bytes | ms/run | runs/s over adb.")
    parser.add_argument("paths", nargs="*", help="container (.bin) or ONNX (.onnx) paths")
    parser.add_argument("--suite", action="append", default=[], metavar="NAME",
                        help="every research/<NAME>/model*.bin (repeatable)")
    parser.add_argument("--glob", action="append", default=[], metavar="PATTERN",
                        help="a repository-relative glob of containers (repeatable)")
    parser.add_argument("--markdown", action="store_true", help="emit a markdown table")
    parser.add_argument("--output", metavar="PATH", help="also write the table to PATH")
    parser.add_argument("--no-compile", action="store_true",
                        help="host mode: do not time the sibling model*.onnx")
    parser.add_argument("--board", action="store_true", help="drive the board over adb")
    parser.add_argument("--runner", choices=("io", "bench"), default="io",
                        help="board timing runner (default: io, tests/board_io.c's protocol)")
    parser.add_argument("--binary", metavar="PATH",
                        help="board binary to push (default: /tmp/board_io or /tmp/board_bench)")
    parser.add_argument("--repeats", type=int, default=5,
                        help="timed board submissions per model (default: 5)")
    parser.add_argument("--iterations", type=int,
                        help="bench runner: invocations rounds, default recorded cases")
    parser.add_argument("--stat", choices=sorted(STATS), default="median",
                        help="which per-repeat statistic fills ms/run (default: median)")
    parser.add_argument("--remote", default="/userdata/open-npu-research/bench",
                        help="board staging directory")
    parser.add_argument("--keep-remote", action="store_true",
                        help="leave the staged model on the board (default: remove it after "
                             "each model, because /userdata has only a few MB free)")
    parser.add_argument("--adb", dest="tool", default=os.environ.get("ADB", "adb"),
                        help="adb executable")
    parser.add_argument("--serial", default=os.environ.get("ADB_SERIAL"),
                        help="adb device serial (default: adb's only device)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    args.runner_binary = args.binary or ("/tmp/board_io" if args.runner == "io"
                                         else "/tmp/board_bench")
    args.runner_name = os.path.basename(args.runner_binary)
    paths = select(args)
    if not paths:
        parser.error("nothing selected: pass container paths and/or --suite/--glob")
    if args.board:
        headers = ("model", "inferences", "exact bytes", "ms/run", "runs/s")
        rows = board_rows(args, paths)
    else:
        headers = ("model", "bytes", "task_count", "engine_runs", "compile_ms")
        rows = host_rows(args, paths)
    text = render_table(headers, rows, args.markdown)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
