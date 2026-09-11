"""Select held-out containers for the per-family cost cross-check (S5).

The per-family table in `research/family_tasks_probe/` was fitted from containers that
repeat one idempotent task descriptor N times. This probe asks the independent
question: does that table predict *real* dependency chains whose task mix it never
saw? It selects existing, already board-verified containers by their (conv, pool,
elementwise) task counts - read from the task enable masks - and records the table
prediction for each one. `run_family_cost_crosscheck.py` then benches them.

Development probe only; the compiler never reads this directory.
"""
from pathlib import Path
import json

from open_rknpu.sequence import decode_sequence

ROOT = Path(__file__).resolve().parent
TASKS = ROOT / "family_tasks_probe"
OUT = ROOT / "family_cost_crosscheck"
OUT.mkdir(exist_ok=True)
FAMILY = {29: "conv", 96: "pool", 24: "elem"}

# suite -> the (conv, pool, elem) mixes to take from it, first match wins.
SELECTION = {
    "deep_chain_suite": [(8, 0, 0), (12, 0, 0), (16, 0, 0)],
    "chain_multi_suite": [(3, 0, 0), (4, 0, 0)],
    "elementwise_deep_suite": [(2, 0, 2), (2, 0, 3), (2, 0, 4)],
    "diamond_suite": [(3, 0, 1)],
    "join_dag_suite": [(4, 0, 2), (4, 0, 3)],
    "pooled_dag_suite": [(5, 1, 2), (6, 1, 2), (6, 1, 3), (8, 1, 3)],
    # Single-DPU-task rows separate the two DPU families from the CNA one, and the
    # batch rows extend the elementwise end of the range.
    "runtime_scale_suite": [(1, 0, 1)],
    "scheduled_pool_suite": [(1, 1, 0), (1, 2, 0)],
    "mul_batch_broadcast_suite": [(3, 0, 3), (8, 0, 8), (16, 0, 16)],
    "pooled_branches_suite": [(5, 3, 2), (10, 3, 2)],
}


def fit(xs, ys):
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    return slope, mean_y - slope * mean_x


def family_counts(binary):
    return [sum(1 for task in decode_sequence(binary)["tasks"] if task["enable"] == code)
            for code in (29, 96, 24)]


def table():
    """Min-based per-family fits from the duplication probe, plus its mean intercept."""
    evidence = json.loads((TASKS / "family_tasks.json").read_text())
    slopes, intercepts = {}, []
    for family in ("conv", "pool", "elem"):
        entries = sorted((entry for entry in evidence.values() if entry["family"] == family),
                         key=lambda entry: entry["family_tasks"])
        slope, intercept = fit([entry["family_tasks"] for entry in entries],
                               [entry["min_us"] for entry in entries])
        slopes[family] = slope
        intercepts.append(intercept)
    return slopes, sum(intercepts) / len(intercepts)


def main():
    slopes, intercept = table()
    manifest = []
    for suite, wanted in SELECTION.items():
        folder = ROOT / suite
        remaining = list(wanted)
        for path in sorted(folder.glob("model*.bin")):
            counts = family_counts(path.read_bytes())
            if tuple(counts) not in remaining:
                continue
            remaining.remove(tuple(counts))
            index = path.stem[5:]
            input_path = folder / f"input{index}.u8"
            expected_path = folder / f"expected{index}.i8"
            assert input_path.is_file() and expected_path.is_file(), path
            tasks = sum(counts)
            prediction = sum(count * slopes[name] for count, name in zip(counts, ("conv", "pool", "elem")))
            manifest.append(dict(
                name=f"{suite}_{index}", suite=suite, index=index,
                model=str(path.relative_to(ROOT)), input=str(input_path.relative_to(ROOT)),
                expected=str(expected_path.relative_to(ROOT)), tasks=tasks,
                conv=counts[0], pool=counts[1], elem=counts[2],
                table_slope_us=round(prediction, 2),
                table_fixed_us=round(intercept + prediction, 2)))
        if remaining:
            raise SystemExit(f"{suite}: mixes not found {remaining}")
    (OUT / "manifest.json").write_text(json.dumps(dict(
        table_slopes_us={name: round(value, 2) for name, value in slopes.items()},
        table_mean_intercept_us=round(intercept, 2), containers=manifest), indent=2) + "\n")
    print(f"selected {len(manifest)} held-out containers with table "
          f"{ {name: round(value, 1) for name, value in slopes.items()} }")
    for entry in manifest:
        print(f"  {entry['name']:28s} conv={entry['conv']} pool={entry['pool']} "
              f"elem={entry['elem']} tasks={entry['tasks']} predicted={entry['table_slope_us']}us")


if __name__ == "__main__":
    main()
