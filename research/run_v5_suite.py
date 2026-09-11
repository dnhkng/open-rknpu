"""MIT. Run a v5 suite directory through tests/board_io.c and record the evidence.

V3/v4 suites stream one model at a time through `board_api_test` and are covered
by `research/run_profile_suite.py`. v5 suites (named tensor tables) instead stage
the whole directory and invoke the file runner once, so this script:

* clears the remote staging directory of the previous suite,
* pushes every `model*.bin` / `input*.u8` / `expected*.i8` plus the local
  `board_io` build,
* runs `./board_io . <model count>` remotely (the runner derives the per-model
  case count from each input file),
* writes `board_results_<start>.json` and `board_summary.txt` next to the suite.

`manifest.json` supplies the per-model input/output byte counts.
"""
from pathlib import Path
import os
import argparse
import json
import subprocess

ADB = os.environ.get("ADB", "adb")
SERIAL = "498063e3262e55b7"

parser = argparse.ArgumentParser()
parser.add_argument("suite", help="suite directory under research/")
parser.add_argument("--remote", default=None,
                    help="board staging directory; defaults to /userdata/open-npu-research/<name>")
parser.add_argument("--binary", default="/tmp/board_io")
parser.add_argument("--cases", type=int, default=None,
                    help="recorded inputs per model; defaults to the manifest 'cases'")
args = parser.parse_args()

folder = Path(__file__).resolve().parent / args.suite
remote = args.remote or f"/userdata/open-npu-research/{args.suite}"
manifest = json.loads((folder / "manifest.json").read_text())


def adb(*command, **kwargs):
    return subprocess.run([ADB, "-s", SERIAL, *command], capture_output=True,
                          text=True, timeout=120, **kwargs)


adb("shell", f"rm -rf {remote} && mkdir -p {remote}")
adb("push", args.binary, f"{remote}/board_io")
for entry in manifest:
    index = entry["index"]
    for prefix, suffix in (("model", "bin"), ("input", "u8"), ("expected", "i8")):
        adb("push", str(folder / f"{prefix}{index:03}.{suffix}"),
            f"{remote}/{prefix}{index:03}.{suffix}")

run = adb("shell", f"cd {remote} && ./board_io . {len(manifest)}")
print(run.stdout, end="")
if run.stderr.strip():
    print(run.stderr, end="", flush=True)

lines = {int(line.split()[1].rstrip(":")): line.strip()
         for line in run.stdout.splitlines() if line.startswith("model ")}
results = []
for entry in manifest:
    index = entry["index"]
    line = lines.get(index, "missing")
    inferences = args.cases or entry.get("cases", 0)
    results.append(dict(model=f"{index:03}", passed="passed" in line,
                        inferences=inferences,
                        bytes=inferences * entry["output_bytes"], output=line))
(folder / "board_results_0.json").write_text(json.dumps(results, indent=2) + "\n")
lines_out = run.stdout.strip().splitlines()
summary = next((line for line in reversed(lines_out) if line.startswith(("PASS:", "FAIL:"))),
               lines_out[-1] if lines_out else "no output")
(folder / "board_summary.txt").write_text(summary + "\n")

passed = all(entry["passed"] for entry in results)
print(f"{'PASS' if passed else 'FAIL'}: {len(results)} models, "
      f"{sum(e['inferences'] for e in results)} inferences, "
      f"{sum(e['bytes'] for e in results)} exact output bytes")
raise SystemExit(0 if passed else 1)
