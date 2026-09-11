"""MIT. Recompile every published suite model and compare it with the checked-in map.

`research/container_baseline.json` records one entry per `research/*suite*/model*.onnx`:

* the sha256 of the emitted executable, when the scheduler compiles the model, or
* `ERR:<ExceptionType>` when the profile rejects it (the rejected models are pinned
  deliberately - a profile that silently starts compiling a model it used to refuse is a
  behaviour change too).

The map was captured before the 2026-09-11 cleanup and is the "no emitted container may
change" contract from `docs/plans/cleanup-plan.md`; the campaign sweep and the board ledger are the
byte-level counterparts. Run `--update` only when a container change is *intended*, and
only after the affected suites have fresh board evidence.

    PYTHONPATH=src python3 research/verify_suites.py
    PYTHONPATH=src python3 research/verify_suites.py --update
"""
from pathlib import Path
import argparse
import collections
import glob
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", default="research/container_baseline.json")
parser.add_argument("--update", action="store_true", help="rewrite the baseline from the current tree")
args = parser.parse_args()

from open_rknpu.scheduler import compile_sequence

models = sorted(glob.glob(str(ROOT / "research/*suite*/model*.onnx")))
current = {}
failures = {}
for path in models:
    key = str(Path(path).relative_to(ROOT / "research"))
    try:
        binary, _ = compile_sequence(path)
    except Exception as exc:  # the rejection itself is the recorded evidence
        current[key] = f"ERR:{type(exc).__name__}"
        failures[key] = str(exc)
        continue
    current[key] = hashlib.sha256(bytes(binary)).hexdigest()

baseline_path = ROOT / args.baseline
if args.update:
    baseline_path.write_text(json.dumps(current, indent=0, sort_keys=True) + "\n")
    print(f"wrote {baseline_path.relative_to(ROOT)}: {len(current)} models")
    raise SystemExit(0)

baseline = json.loads(baseline_path.read_text())
same = [k for k, v in baseline.items() if current.get(k) == v]
changed = [k for k, v in baseline.items() if current.get(k) != v]
added = [k for k in current if k not in baseline]
removed = [k for k in baseline if k not in current]
print(f"models={len(models)} baseline={len(baseline)} same={len(same)} "
      f"changed={len(changed)} added={len(added)} removed={len(removed)}")
print("rejections:", collections.Counter(v for v in current.values() if v.startswith("ERR:")))
for key in changed[:25]:
    print(f"  CHANGED {key}: {baseline[key][:16]} -> {current.get(key, 'ABSENT')[:16]}")
for key in added[:25]:
    print(f"  ADDED {key}")
for key in removed[:25]:
    print(f"  REMOVED {key}")
raise SystemExit(0 if not (changed or added or removed) else 1)
