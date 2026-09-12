"""SPDX-License-Identifier: MIT

Deterministic, bounded mutation testing driver for open-rknpu.

Coverage says which lines run; mutation testing says whether the suite would
notice if those lines did something different. This driver mutates a configured
list of compiler modules, runs a configured test subset per module, and reports
killed / survived / timeout / error mutants plus a mutation score.

Bounding and reproducibility
----------------------------
* Every mutant is applied inside a throwaway copy of ``src/open_rknpu`` under a
  temporary directory; ``PYTHONPATH`` points at that copy. The real working tree
  is never written to, and the driver asserts that a SHA-256 manifest of the
  original package is byte-identical before and after the run.
* Mutation candidates are collected by AST walk in source order, then selected
  with a fixed seed (``--seed``) so the mutant set and its order are identical
  on every run of the same command.
* Runs are serial. Each mutant gets ``--timeout`` seconds and the whole run gets
  a wall-clock ``--budget-seconds``; whatever is left when the budget expires is
  reported as skipped instead of silently dropped.
* ``--quick`` shrinks the per-module mutant cap, the per-mutant timeout and the
  budget so a smoke run finishes in under two minutes on this machine.

The test tree is read-only: pytest runs with ``-p no:cacheprovider`` and
``PYTHONDONTWRITEBYTECODE=1`` so no cache or bytecode is written back.

Usage::

    PYTHONPATH=src python research/run_mutation_tests.py            # full run
    PYTHONPATH=src python research/run_mutation_tests.py --quick    # smoke run
    PYTHONPATH=src python research/run_mutation_tests.py --modules quantization.py,padding.py
"""
from __future__ import annotations

import argparse
import ast
import bisect
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGE = REPO_ROOT / "src" / "open_rknpu"
DEFAULT_SCOPE = REPO_ROOT / "research" / "mutation_scope.json"

DEFAULT_SEED = 20240517
DEFAULT_MAX_MUTANTS = 10
DEFAULT_TIMEOUT = 30.0
DEFAULT_BUDGET = 240.0
QUICK_MAX_MUTANTS = 4
QUICK_TIMEOUT = 20.0
QUICK_BUDGET = 90.0

# Deterministic round-robin order for stratified selection. Keeping an explicit
# order means adding a new mutation kind never reshuffles the existing picks.
KIND_ORDER = ("binop", "compare", "boolop", "const", "return")
KIND_LABEL = {
    "binop": "operator",
    "compare": "comparison",
    "boolop": "boolean op",
    "const": "constant",
    "return": "return negation",
}

SKIP_TOKEN_TYPES = frozenset(
    {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
)


@dataclass(frozen=True)
class Mutant:
    """One textual edit of one module, with enough context to report it."""

    module: str
    kind: str
    line: int
    col: int
    start: int
    end: int
    original: str
    replacement: str
    snippet: str

    @property
    def change(self) -> str:
        return f"{self.original!r} -> {self.replacement!r}"

    @property
    def location(self) -> str:
        return f"{self.module}:{self.line}"


def _module_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_manifest() -> dict[str, str]:
    manifest = {}
    for path in sorted(SOURCE_PACKAGE.rglob("*.py")):
        manifest[str(path.relative_to(SOURCE_PACKAGE))] = _module_hash(path)
    return manifest


class _OperatorFinder:
    """Locate an operator token between two operand spans.

    CPython does not give ``ast.Add``/``ast.And``/``ast.Lt`` position
    attributes, so the driver tokenises the file once and looks for the expected
    operator text inside the gap between the operands. Real tokens only, so an
    operator-like character inside a comment is ignored.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        offsets = [0]
        for line in text.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        self.line_offsets = offsets
        self.tokens: list[tuple[int, int, str]] = []
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type in SKIP_TOKEN_TYPES:
                continue
            start = self.offset(token.start[0], token.start[1])
            end = self.offset(token.end[0], token.end[1])
            self.tokens.append((start, end, token.string))

    def offset(self, line: int, col: int) -> int:
        if line - 1 >= len(self.line_offsets):
            return len(self.text)
        return self.line_offsets[line - 1] + col

    def line_col(self, offset: int) -> tuple[int, int]:
        line = bisect.bisect_right(self.line_offsets, offset) - 1
        return line + 1, offset - self.line_offsets[line]

    def between(self, left: ast.AST, right: ast.AST, operator: str) -> tuple[int, int] | None:
        start = self.offset(left.end_lineno, left.end_col_offset)
        end = self.offset(right.lineno, right.col_offset)
        for tok_start, tok_end, text in self.tokens:
            if tok_start >= start and tok_end <= end and text == operator:
                return tok_start, tok_end
        return None


class _Collector(ast.NodeVisitor):
    """Collect every single-edit mutant in a module, in source order."""

    def __init__(self, module: str, text: str) -> None:
        self.module = module
        self.text = text
        self.lines = text.splitlines()
        self.finder = _OperatorFinder(text)
        self.mutants: list[Mutant] = []

    def _snippet(self, line: int) -> str:
        raw = self.lines[line - 1].strip() if 0 < line <= len(self.lines) else ""
        return raw if len(raw) <= 100 else raw[:97] + "..."

    def _add(self, kind: str, start: int, end: int, original: str, replacement: str) -> None:
        line, col = self.finder.line_col(start)
        self.mutants.append(
            Mutant(
                module=self.module,
                kind=kind,
                line=line,
                col=col,
                start=start,
                end=end,
                original=original,
                replacement=replacement,
                snippet=self._snippet(line),
            )
        )

    def _operator_mutant(self, kind: str, left: ast.AST, right: ast.AST, old: str, new: str) -> None:
        span = self.finder.between(left, right, old)
        if span is not None:
            self._add(kind, span[0], span[1], old, new)

    def visit_BinOp(self, node: ast.BinOp) -> None:  # noqa: N802 (ast visitor API)
        if isinstance(node.op, ast.Add):
            self._operator_mutant("binop", node.left, node.right, "+", "-")
        elif isinstance(node.op, ast.Sub):
            self._operator_mutant("binop", node.left, node.right, "-", "+")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:  # noqa: N802
        operands = [node.left, *node.comparators]
        for index, op in enumerate(node.ops):
            if isinstance(op, ast.Lt):
                self._operator_mutant("compare", operands[index], operands[index + 1], "<", "<=")
            elif isinstance(op, ast.LtE):
                self._operator_mutant("compare", operands[index], operands[index + 1], "<=", "<")
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:  # noqa: N802
        if isinstance(node.op, ast.And):
            old, new = "and", "or"
        elif isinstance(node.op, ast.Or):
            old, new = "or", "and"
        else:
            old = new = ""
        if old:
            for left, right in zip(node.values, node.values[1:]):
                self._operator_mutant("boolop", left, right, old, new)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802
        if isinstance(node.value, bool):
            self._add(
                "const",
                self.finder.offset(node.lineno, node.col_offset),
                self.finder.offset(node.end_lineno, node.end_col_offset),
                "True" if node.value else "False",
                "False" if node.value else "True",
            )
        elif isinstance(node.value, int):
            self._add(
                "const",
                self.finder.offset(node.lineno, node.col_offset),
                self.finder.offset(node.end_lineno, node.end_col_offset),
                repr(node.value),
                repr(node.value + 1),
            )
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:  # noqa: N802
        if node.value is not None:
            start = self.finder.offset(node.value.lineno, node.value.col_offset)
            end = self.finder.offset(node.value.end_lineno, node.value.end_col_offset)
            original = self.text[start:end]
            self._add("return", start, end, original, f"not ({original})")
        self.generic_visit(node)


def collect_mutants(module: str) -> list[Mutant]:
    text = (SOURCE_PACKAGE / module).read_text(encoding="utf-8")
    collector = _Collector(module, text)
    collector.visit(ast.parse(text))
    return collector.mutants


def select_mutants(candidates: list[Mutant], limit: int, seed: int) -> list[Mutant]:
    """Pick at most ``limit`` candidates, stratified across mutation kinds.

    Buckets are shuffled with the fixed seed, then drawn round-robin so a module
    with hundreds of integer constants still gets operator, comparison, boolean
    and return mutants. The final list is sorted by source position so the
    reported order matches the file.
    """
    if limit <= 0:
        return []
    if limit >= len(candidates):
        return sorted(candidates, key=lambda m: (m.line, m.col, m.kind))
    rng = random.Random(seed)
    buckets: dict[str, list[Mutant]] = {}
    for kind in KIND_ORDER:
        bucket = [m for m in candidates if m.kind == kind]
        rng.shuffle(bucket)
        buckets[kind] = bucket
    chosen: list[Mutant] = []
    depth = 0
    while len(chosen) < limit:
        progressed = False
        for kind in KIND_ORDER:
            bucket = buckets[kind]
            if depth < len(bucket):
                chosen.append(bucket[depth])
                progressed = True
                if len(chosen) >= limit:
                    break
        if not progressed:
            break
        depth += 1
    return sorted(chosen, key=lambda m: (m.line, m.col, m.kind))


def apply_mutant(text: str, mutant: Mutant) -> str:
    return text[: mutant.start] + mutant.replacement + text[mutant.end :]


def run_tests(tests: list[str], pythonpath: Path, timeout: float) -> tuple[str, float, str]:
    """Run one test subset against the copied tree.

    Returns ``(verdict, seconds, detail)`` where verdict is one of ``survived``,
    ``killed``, ``timeout`` or ``error``.
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(pythonpath) if not existing else f"{pythonpath}{os.pathsep}{existing}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-x",
        "-q",
        "-p",
        "no:cacheprovider",
        "-p",
        "no:randomly",
        "--no-header",
        "--tb=no",
        *tests,
    ]
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "timeout", time.monotonic() - started, f"exceeded {timeout:.0f}s"
    except OSError as exc:  # pragma: no cover - environment failure
        return "error", time.monotonic() - started, f"launch failed: {exc}"
    elapsed = time.monotonic() - started
    if proc.returncode == 0:
        return "survived", elapsed, "tests passed on the mutant"
    if proc.returncode == 1:
        return "killed", elapsed, "test failure"
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    return "error", elapsed, f"pytest exit {proc.returncode}: {tail[-1] if tail else 'no output'}"


@dataclass
class ModuleResult:
    module: str
    tests: list[str]
    counts: dict[str, int] = field(default_factory=lambda: {"killed": 0, "survived": 0, "timeout": 0, "error": 0})
    survivors: list[Mutant] = field(default_factory=list)
    errors: list[tuple[Mutant | None, str]] = field(default_factory=list)
    selected_mutants: list[Mutant] = field(default_factory=list)
    skipped: int = 0
    selected: int = 0
    baseline: str = "not run"
    seconds: float = 0.0

    @property
    def score(self) -> float:
        detected = self.counts["killed"] + self.counts["timeout"]
        total = detected + self.counts["survived"]
        return 100.0 * detected / total if total else 0.0


def load_scope(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "modules" not in data or not isinstance(data["modules"], dict):
        raise SystemExit(f"{path}: expected a 'modules' object")
    return data


def resolve_selection(requested: str | None, scope: dict) -> list[str]:
    available = list(scope["modules"])
    if not requested or requested.strip().lower() == "all":
        return available
    wanted = [item.strip() for item in requested.replace(",", " ").split() if item.strip()]
    resolved = []
    for item in wanted:
        matches = [m for m in available if m == item or Path(m).stem == Path(item).stem]
        if not matches:
            raise SystemExit(f"module {item!r} is not in the mutation scope")
        resolved.extend(matches)
    return resolved


def mutate_module(
    module: str,
    tests: list[str],
    copy_src: Path,
    text: str,
    limit: int,
    seed: int,
    timeout: float,
    deadline: float,
    dry_run: bool,
) -> ModuleResult:
    result = ModuleResult(module=module, tests=tests)
    candidates = collect_mutants(module)
    mutants = select_mutants(candidates, limit, seed)
    result.selected = len(mutants)
    result.selected_mutants = mutants
    if dry_run:
        result.baseline = "dry run"
        return result

    baseline_verdict, _, detail = run_tests(tests, copy_src, timeout)
    result.baseline = baseline_verdict
    if baseline_verdict != "survived":
        # The suite must pass on the unmutated copy; otherwise every mutant
        # would be scored "killed" by a pre-existing failure.
        result.counts["error"] += 1
        result.errors.append((None, f"baseline subset failed on clean copy ({baseline_verdict}): {detail}"))
        result.skipped = len(mutants)
        return result

    started = time.monotonic()
    target = copy_src / "open_rknpu" / module
    try:
        for index, mutant in enumerate(mutants):
            if time.monotonic() >= deadline:
                result.skipped = len(mutants) - index
                break
            mutated = apply_mutant(text, mutant)
            try:
                ast.parse(mutated)
            except SyntaxError as exc:  # pragma: no cover - generator bug guard
                result.counts["error"] += 1
                result.errors.append((mutant, f"generated source does not parse: {exc}"))
                continue
            target.write_text(mutated, encoding="utf-8")
            verdict, _, detail = run_tests(tests, copy_src, timeout)
            result.counts[verdict] += 1
            if verdict == "survived":
                result.survivors.append(mutant)
            elif verdict == "error":
                result.errors.append((mutant, detail))
    finally:
        target.write_text(text, encoding="utf-8")
        result.seconds = time.monotonic() - started
    return result


def total_score(results: list[ModuleResult]) -> float:
    detected = sum(r.counts["killed"] + r.counts["timeout"] for r in results)
    survived = sum(r.counts["survived"] for r in results)
    return 100.0 * detected / (detected + survived) if detected + survived else 0.0


def print_summary(results: list[ModuleResult], total_seconds: float, source_ok: bool, budget: float) -> None:
    print("\n== Mutation summary ==")
    header = f"{'module':<20} {'tests':>5} {'killed':>7} {'survived':>8} {'timeout':>7} {'error':>5} {'skipped':>7} {'score':>7}"
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.module:<20} {len(result.tests):>5} {result.counts['killed']:>7} "
            f"{result.counts['survived']:>8} {result.counts['timeout']:>7} {result.counts['error']:>5} "
            f"{result.skipped:>7} {result.score:>6.1f}%"
        )
    killed = sum(r.counts["killed"] for r in results)
    survived = sum(r.counts["survived"] for r in results)
    timed_out = sum(r.counts["timeout"] for r in results)
    errors = sum(r.counts["error"] for r in results)
    skipped = sum(r.skipped for r in results)
    detected = killed + timed_out
    score = 100.0 * detected / (detected + survived) if detected + survived else 0.0
    print("-" * len(header))
    print(
        f"{'TOTAL':<20} {'':>5} {killed:>7} {survived:>8} {timed_out:>7} {errors:>5} {skipped:>7} {score:>6.1f}%"
    )
    print("\nmutation score = (killed + timeout) / (killed + timeout + survived); errors excluded from the score")
    print(f"total wall time: {total_seconds:.1f}s (budget {budget:.0f}s)")

    survivors = [(r.module, m) for r in results for m in r.survivors]
    print(f"\n== Surviving mutants ({len(survivors)}) ==")
    if not survivors:
        print("none: every selected mutant was detected")
    for module, mutant in survivors:
        print(f"{module}:{mutant.line}:{mutant.col}  [{KIND_LABEL[mutant.kind]}]  {mutant.change}")
        print(f"    {mutant.snippet}")

    if errors:
        print(f"\n== Errored mutants ({errors}) ==")
        for result in results:
            for mutant, detail in result.errors:
                if mutant is None:
                    print(f"{result.module}: {detail}")
                    continue
                print(f"{mutant.location}  [{KIND_LABEL[mutant.kind]}]  {mutant.change}")
                print(f"    {detail}")

    skipped_total = sum(r.skipped for r in results)
    if skipped_total:
        print(f"\n{skipped_total} mutant(s) were not run because the wall-clock budget expired")
    print(f"\nsource tree unchanged after run: {source_ok}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE, help="mutation scope JSON (default: %(default)s)")
    parser.add_argument("--modules", default="all", help="comma/space separated module names, e.g. quantization.py,walk.py")
    parser.add_argument("--max-mutants", type=int, default=DEFAULT_MAX_MUTANTS, help="mutants per module (default: %(default)s)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="seconds per mutant (default: %(default)s)")
    parser.add_argument("--budget-seconds", type=float, default=DEFAULT_BUDGET, help="wall-clock budget (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="selection seed (default: %(default)s)")
    parser.add_argument("--quick", action="store_true", help="small bounded smoke run (under two minutes)")
    parser.add_argument("--dry-run", action="store_true", help="list the selected mutants without running pytest")
    parser.add_argument("--fail-under", type=float, default=None, metavar="PERCENT",
                        help="exit non-zero when the total mutation score is below this percentage")
    parser.add_argument("--json", type=Path, default=None, help="also write machine-readable results to this path")
    args = parser.parse_args(argv)

    if args.quick:
        args.max_mutants = min(args.max_mutants, QUICK_MAX_MUTANTS)
        args.timeout = min(args.timeout, QUICK_TIMEOUT)
        args.budget_seconds = min(args.budget_seconds, QUICK_BUDGET)

    scope = load_scope(args.scope)
    seed = scope.get("seed", args.seed)
    modules = resolve_selection(args.modules, scope)

    manifest_before = _package_manifest()
    started = time.monotonic()
    deadline = started + args.budget_seconds
    results: list[ModuleResult] = []
    temp_root = Path(tempfile.mkdtemp(prefix="open-rknpu-mutants-"))
    copy_src = temp_root / "src"
    print(f"scope: {args.scope}")
    print(f"modules: {', '.join(modules)}")
    print(f"seed={seed} max-mutants={args.max_mutants} timeout={args.timeout:.0f}s budget={args.budget_seconds:.0f}s")
    print(f"workdir: {temp_root} (removed on exit)\n")
    try:
        shutil.copytree(
            SOURCE_PACKAGE,
            copy_src / "open_rknpu",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        for module in modules:
            spec = scope["modules"][module]
            tests = spec["tests"] if isinstance(spec, dict) else list(spec)
            limit = spec.get("max_mutants", args.max_mutants) if isinstance(spec, dict) else args.max_mutants
            missing = [t for t in tests if not (REPO_ROOT / t).is_file()]
            if missing:
                raise SystemExit(f"{module}: missing test files {missing}")
            if args.quick:
                limit = min(limit, QUICK_MAX_MUTANTS)
            text = (SOURCE_PACKAGE / module).read_text(encoding="utf-8")
            result = mutate_module(
                module, tests, copy_src, text, limit, seed, args.timeout, deadline, args.dry_run
            )
            results.append(result)
            print(
                f"[{module}] selected={result.selected} killed={result.counts['killed']} "
                f"survived={result.counts['survived']} timeout={result.counts['timeout']} "
                f"error={result.counts['error']} skipped={result.skipped} in {result.seconds:.1f}s"
            )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    total_seconds = time.monotonic() - started

    manifest_after = _package_manifest()
    source_ok = manifest_before == manifest_after
    if not source_ok:
        raise SystemExit("working tree changed during the run: aborting (this is a driver bug)")

    if args.dry_run:
        print("\n== Selected mutants (dry run) ==")
        for result in results:
            print(f"\n[{result.module}] {result.selected} of the collected candidates")
            for mutant in result.selected_mutants:
                print(f"{mutant.location}:{mutant.col}  [{KIND_LABEL[mutant.kind]}]  {mutant.change}")
                print(f"    {mutant.snippet}")
    else:
        print_summary(results, total_seconds, source_ok, args.budget_seconds)

    if args.json:
        payload = {
            "scope": str(args.scope),
            "seed": seed,
            "max_mutants": args.max_mutants,
            "timeout": args.timeout,
            "budget_seconds": args.budget_seconds,
            "quick": args.quick,
            "total_score": total_score(results),
            "fail_under": args.fail_under,
            "dry_run": args.dry_run,
            "total_seconds": total_seconds,
            "source_unchanged": source_ok,
            "manifest_before": manifest_before,
            "modules": [
                {
                    "module": r.module,
                    "tests": r.tests,
                    "selected": r.selected,
                    "counts": r.counts,
                    "skipped": r.skipped,
                    "score": r.score,
                    "seconds": r.seconds,
                    "survivors": [
                        {
                            "file": r.module,
                            "line": m.line,
                            "col": m.col,
                            "kind": m.kind,
                            "original": m.original,
                            "replacement": m.replacement,
                            "snippet": m.snippet,
                        }
                        for m in r.survivors
                    ],
                    "errors": [
                        {"line": m.line if m else None, "detail": d} for m, d in r.errors
                    ],
                }
                for r in results
            ],
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    if args.fail_under is not None and not args.dry_run:
        score = total_score(results)
        if score < args.fail_under:
            print(f"\nFAIL: mutation score {score:.1f}% is below --fail-under {args.fail_under:.1f}%")
            return 1
        print(f"\nPASS: mutation score {score:.1f}% meets --fail-under {args.fail_under:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
