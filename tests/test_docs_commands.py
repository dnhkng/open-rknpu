"""SPDX-License-Identifier: MIT

Keep the host-safe commands the documentation tells a user to run actually runnable.

The test walks every fenced code block in ``docs/*.md``, ``docs/plans/*.md``,
``examples/*/README.md`` and ``examples/*/*.md`` (the two example globs are de-duplicated),
joins backslash continuations, and collects the shell command lines that start with
``python ``, ``python3 ``, ``PYTHONPATH=src python [3] ``, ``make ``, ``open-rknpu ``,
``ruff `` or ``pytest ``. Every collected command is either **executed** or recorded as
**skipped** with one reason from this documented rule, checked in order:

1. ``# ... board ...`` on the same line: the doc itself marks the line as hardware work.
2. A forbidden token: ``adb``, ``gcc``, ``arm-rockchip830``, ``torch``, ``pip install``,
   ``git ``, ``curl``, ``uv ``, ``--dataset``, ``train.py``, ``fetch_``, ``mkdocs``,
   ``twine``, ``mutmut`` or ``coverage``.
3. A placeholder or shell variable (``<model.onnx>``, ``$script``, a glob ``*``): the line
   is a usage pattern, not a concrete command.
4. The whole ``tests/`` suite (``-m unittest discover -s tests`` / ``-m pytest tests``):
   this module is part of that suite, so executing it here would recurse.
5. A ``make <target>`` whose recipe in the ``Makefile`` contains a forbidden token: the
   documented line hides the vendor toolchain behind the target (``make board-io``).
6. A ``research/build_*.py`` publisher: it regenerates a suite inside ``research/`` and the
   isolated copy cannot host the ~340 MB corpus, so the test skips it rather than let it
   write into the checkout. The two read-only sweeps (``verify_suites.py``,
   ``campaign_sweep.py``) still run.
7. The ``open-rknpu`` console script is not installed in this environment.
8. A referenced input artifact does not exist (``model.onnx``, ``calibration/``,
   ``/tmp/board_io``): the documented workflow builds it in an earlier step.
9. The command's script drives the board itself - a subprocess call through ``adb``/a
   vendor compiler, a model from ``train.py`` (PyTorch), or a recorded ``actual.f32``.
10. The selected build directory does not ship the board-pulled ``actual.f32`` that the
    script reads.

Host-safe commands run in an isolated temporary copy of the repository root (no git
worktree: a plain ``copytree`` that skips ``.git``, ``research`` and caches) with ``cwd`` set
to that copy, ``PYTHONPATH=<copy>/src``, a temporary ``HOME`` and a temporary ruff cache, so
a command can never write into the checkout. ``research/`` is symlinked in because the
read-only sweeps read it; the commands that publish into it are skipped by rule 6, so it is
only ever read. The files a failed isolation would touch are snapshotted by size and mtime
before and after; they must be unchanged.

The inventory is printed on every run and the module fails unless at least eight commands
executed successfully with zero failures, every documented command is executed or carries a
skip reason, and the whole run stayed under its (generous) time budget. Documented commands that are broken
are reported through those reasons instead of being weakened away.
"""
import ast
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples" / "notebooks" / "open_rknpu_walkthrough.ipynb"

TARGET_GLOBS = ("docs/*.md", "docs/plans/*.md", "examples/*/README.md", "examples/*/*.md")
COMMAND_PREFIXES = ("python ", "python3 ", "PYTHONPATH=src python ", "PYTHONPATH=src python3 ",
                    "make ", "open-rknpu ", "ruff ", "pytest ")
SKIP_TOKENS = ("adb", "gcc", "arm-rockchip830", "torch", "pip install", "git ", "curl", "uv ",
               "--dataset", "train.py", "fetch_", "mkdocs", "twine", "mutmut", "coverage")
BOARD_COMMENT = re.compile(r"#.*\bboard\b", re.IGNORECASE)
TRAILING_COMMENT = re.compile(r"\s+#.*$")
INTERPRETER_TOKEN = re.compile(r"\bpython3?\b")
INTERPRETER_PREFIX = re.compile(r"^((?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*)python3?(?=\s)")
FULL_SUITE = re.compile(r"-m\s+(?:unittest\s+discover\s+-s\s+tests|pytest\s+tests)")
PLACEHOLDER = re.compile(r"<[^>\s]+>")
DATA_SUFFIXES = (".onnx", ".bin", ".u8", ".i8", ".f32", ".npy", ".json", ".py")
SUBPROCESS_CALLS = ("run", "Popen", "call", "check_call", "check_output")
PUBLISHER_SCRIPT = re.compile(r"research/build_[^/]*\.py$")
IGNORE = shutil.ignore_patterns(".git", "research", "__pycache__", "*.pyc", ".ruff_cache",
                                ".pytest_cache", "dist", "*.egg-info")
WATCH_GLOBS = ("src/open_rknpu/**/*.py", "examples/*/build/**", "examples/*/build-*/**",
               "examples/*/sanity-results/**")
PER_COMMAND_TIMEOUT = 120
MIN_EXECUTED = 8
# A generous ceiling, not a benchmark: the point is to catch a documented command that
# hangs or that turned into a whole-suite run. The measured wall time is ~15 s idle and
# exceeds a minute when the machine is busy (several subagents, a coverage run), so a tight
# number here would fail for reasons that have nothing to do with the documentation.
MODULE_BUDGET_SECONDS = 300


# --------------------------------------------------------------------------- #
# Extraction: fenced blocks -> logical shell command lines
# --------------------------------------------------------------------------- #
def target_files():
    """Every markdown file the rule walks, de-duplicated across the example globs."""
    files = set()
    for pattern in TARGET_GLOBS:
        files.update(ROOT.glob(pattern))
    return sorted(files)


def fenced_blocks(text):
    """Yield ``(fence_line_number, [(line_number, line), ...])`` for each fenced block."""
    lines = text.splitlines()
    inside = False
    start = 0
    block = []
    for number, line in enumerate(lines, 1):
        if line.strip().startswith("```"):
            if inside:
                yield start, block
                block = []
            else:
                start = number
            inside = not inside
            continue
        if inside:
            block.append((number, line))
    if inside:
        yield start, block


def logical_commands(block):
    """Join backslash continuations and yield ``(command, first_line_number)``."""
    commands = []
    pending = None
    pending_line = 0
    for number, line in block:
        stripped = line.strip()
        if pending is not None:
            pending += " " + stripped
            if not stripped.endswith("\\"):
                commands.append((pending.replace("\\ ", " ").strip(), pending_line))
                pending = None
            continue
        if stripped.startswith(COMMAND_PREFIXES):
            if stripped.endswith("\\"):
                pending = stripped
                pending_line = number
            else:
                commands.append((stripped, number))
    if pending is not None:
        commands.append((pending.replace("\\ ", " ").strip(), pending_line))
    return commands


def code_of(command):
    """The command with its trailing ``#`` comment removed."""
    return TRAILING_COMMENT.sub("", command).strip()


def normalize_command(command):
    """A dedupe key: comments stripped, whitespace collapsed, interpreter anonymised."""
    return INTERPRETER_TOKEN.sub("<python>", " ".join(code_of(command).split()))


def extract_commands():
    """Every documented command once, with the ``path:line`` locations that document it."""
    entries = {}
    for path in target_files():
        for _start, block in fenced_blocks(path.read_text()):
            for command, line in logical_commands(block):
                entry = entries.get(normalize_command(command))
                if entry is None:
                    entry = dict(text=command, reason=None, returncode=None, output="",
                                 seconds=0.0, locations=[])
                    entries[normalize_command(command)] = entry
                entry["locations"].append("%s:%d" % (path.relative_to(ROOT), line))
    return list(entries.values())


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def split(command):
    """``shlex.split`` that tolerates a line shlex cannot balance."""
    try:
        return shlex.split(code_of(command))
    except ValueError:
        return []


def path_tokens(command):
    """Path-like argument tokens, ignoring option flags and ``-c``/``-m`` payloads."""
    parts = split(command)
    if "-c" in parts or "-m" in parts:
        return []
    tokens = []
    for token in parts:
        if token.startswith("--") and "=" in token:
            token = token.split("=", 1)[1]
        if token.startswith("-") or token in ("python", "python3", "open-rknpu", "ruff", "make",
                                              "pytest", "check"):
            continue
        if "/" in token or token.endswith(DATA_SUFFIXES):
            tokens.append(token)
    return tokens


def resolve_token(token):
    """A documented token as a repository-relative path."""
    token = token.rstrip("/")
    path = Path(token)
    return path if path.is_absolute() else ROOT / path


def missing_input(command):
    """The first referenced input artifact that does not exist, or ``None``."""
    for token in path_tokens(command):
        if not resolve_token(token).exists():
            return token
    return None


def referenced_script(command):
    """The ``.py`` file a ``python <script>`` command runs, if it names one."""
    parts = split(command)
    index = 0
    while index < len(parts) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", parts[index]):
        index += 1
    if index >= len(parts) or parts[index] not in ("python", "python3"):
        return None
    index += 1
    if index >= len(parts) or parts[index].startswith("-"):
        return None
    script = ROOT / parts[index]
    return script if script.suffix == ".py" and script.is_file() else None


def board_script_marker(command):
    """Why the command's own script needs the board or PyTorch, or ``None``.

    The script is parsed, so a *printed* board recipe (``examples/cookbook/01`` prints adb
    commands as text) is not mistaken for a script that runs one: only an actual
    ``subprocess`` call through adb/a vendor compiler counts.
    """
    script = referenced_script(command)
    if script is None:
        return None
    source = script.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess"
                and node.func.attr in SUBPROCESS_CALLS):
            continue
        call = ast.unparse(node).lower()
        if any(marker.lower() in call for marker in ("adb", "arm-rockchip830", "gcc")):
            return "the board or vendor toolchain (its script spawns adb/gcc)"
    if "train.py" in source:
        return "the model produced by train.py (PyTorch)"
    return None


def board_artifact_reason(command):
    """Why the script's selected build directory lacks its board-pulled artifact."""
    script = referenced_script(command)
    if script is None or "actual.f32" not in script.read_text():
        return None
    if "--calibrated" in command:
        directory = script.parent / "build-native-calibrated"
    elif "--native" in command:
        directory = script.parent / "build-native"
    else:
        directory = script.parent / "build"
    if (directory / "actual.f32").is_file():
        return None
    return "the board-pulled %s/actual.f32 recorded by the adb step" % directory.name


def make_recipe_reason(command):
    """A ``make`` target whose Makefile recipe contains a forbidden token.

    ``make board-io`` reads as a plain host command, but its recipe is the vendor
    cross-compiler, so running it needs the fetched (git-ignored) toolchain and is not
    deterministic. The target's recipe is parsed from the ``Makefile`` the command names.
    """
    parts = split(command)
    if not parts or parts[0] != "make":
        return None
    index = 1
    directory = ROOT
    target = None
    while index < len(parts):
        token = parts[index]
        if token in ("-C", "--directory"):
            index += 1
            directory = ROOT / parts[index] if index < len(parts) else directory
        elif token in ("-f", "--file"):
            index += 1
            if index < len(parts):
                directory = (ROOT / parts[index]).parent
        elif token.startswith("-") or "=" in token:
            pass
        else:
            target = token
            break
        index += 1
    makefile = directory / "Makefile"
    if target is None or not makefile.is_file():
        return None
    recipe = []
    collecting = False
    for line in makefile.read_text().splitlines():
        if line.startswith(target + ":"):
            collecting = True
            continue
        if collecting:
            if line.startswith(("\t", " ")):
                recipe.append(line)
            else:
                break
    lowered = " ".join(recipe).lower()
    for token in SKIP_TOKENS:
        if token in lowered:
            return "the %r make target runs %r itself" % (target, token)
    return None


def publisher_reason(command):
    """A ``research/build_*.py`` command that writes a generated suite into ``research/``."""
    script = referenced_script(command)
    if script is None or not script.is_relative_to(ROOT):
        return None
    if PUBLISHER_SCRIPT.search(script.relative_to(ROOT).as_posix()):
        return "it regenerates a published suite inside research/ (skipped: no repo writes)"
    return None


def skip_reason(command):
    """The documented reason this command is not run on the host, or ``None``."""
    if BOARD_COMMENT.search(command):
        return "the doc marks it as a board command (# board)"
    lowered = code_of(command).lower()
    for token in SKIP_TOKENS:
        if token in lowered:
            return "needs %r" % token
    if PLACEHOLDER.search(command) or "$" in command or "*" in command:
        return "parameterized usage line (placeholder or shell variable)"
    if FULL_SUITE.search(command):
        return "runs the whole tests/ suite, which contains this module"
    recipe = make_recipe_reason(command)
    if recipe:
        return recipe
    publisher = publisher_reason(command)
    if publisher:
        return publisher
    parts = split(command)
    if parts and parts[0] == "open-rknpu" and shutil.which("open-rknpu") is None:
        return "the open-rknpu console script is not installed here (pip install -e .)"
    token = missing_input(command)
    if token:
        return "missing input artifact %s" % token
    marker = board_script_marker(command)
    if marker:
        return "needs %s" % marker
    return board_artifact_reason(command)


# --------------------------------------------------------------------------- #
# Isolated execution
# --------------------------------------------------------------------------- #
def interpreter_command(command):
    """Replace the documented ``python``/``python3`` with the running interpreter."""
    return INTERPRETER_PREFIX.sub(
        lambda match: match.group(1) + shlex.quote(sys.executable) + " ", command, count=1)


def build_workspace(folder):
    """Copy the repository root without VCS, the suite corpus or caches; symlink research."""
    shutil.copytree(ROOT, folder, ignore=IGNORE, dirs_exist_ok=True)
    research = ROOT / "research"
    if research.is_dir():
        os.symlink(research, folder / "research")
    (folder / ".home").mkdir(exist_ok=True)


def watched_state():
    """Size and mtime of the files a broken isolation would rewrite."""
    state = {}
    for pattern in WATCH_GLOBS:
        for path in ROOT.glob(pattern):
            if path.is_file():
                stat = path.stat()
                state[str(path.relative_to(ROOT))] = (stat.st_size, stat.st_mtime_ns)
    return state


def run_documented_command(command, folder):
    """Run one command in the isolated copy; return ``(returncode, output, seconds)``."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(folder / "src")
    env["HOME"] = str(folder / ".home")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["RUFF_CACHE_DIR"] = str(folder / ".ruff_cache")
    env["PYTHONHASHSEED"] = "0"
    started = time.monotonic()
    try:
        process = subprocess.run(interpreter_command(command), shell=True, cwd=str(folder),
                                 env=env, capture_output=True, text=True, errors="replace",
                                 timeout=PER_COMMAND_TIMEOUT)
        return process.returncode, process.stdout + process.stderr, time.monotonic() - started
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        if isinstance(output, bytes):
            output = output.decode("utf-8", "replace")
        return None, output + "\n[timed out after %d s]" % PER_COMMAND_TIMEOUT, \
            time.monotonic() - started


def print_inventory(commands, elapsed):
    """Print the executed/skipped coverage a reader needs to trust the test."""
    executed = [entry for entry in commands if entry["reason"] is None]
    skipped = [entry for entry in commands if entry["reason"] is not None]
    print("=" * 78)
    print("docs-commands inventory: %d documented commands, %d executed, %d skipped (%.1f s)"
          % (len(commands), len(executed), len(skipped), elapsed))
    for entry in executed:
        print("  EXEC rc=%-3s %6.2fs  %s   [%s]"
              % (entry["returncode"], entry["seconds"], entry["text"], ", ".join(entry["locations"])))
    for entry in skipped:
        print("  SKIP %s   %s   [%s]"
              % (entry["reason"], entry["text"], ", ".join(entry["locations"])))
    print("=" * 78)


class DocumentedCommandsTest(unittest.TestCase):
    """Execute every host-safe documented command in an isolated copy of the repository."""

    @classmethod
    def setUpClass(cls):
        cls.commands = extract_commands()
        cls.folder = Path(tempfile.mkdtemp(prefix="rknpu-docs-commands-"))
        build_workspace(cls.folder)
        for entry in cls.commands:
            entry["reason"] = skip_reason(entry["text"])
        cls.before = watched_state()
        started = time.monotonic()
        for entry in cls.commands:
            if entry["reason"] is None:
                entry["returncode"], entry["output"], entry["seconds"] = \
                    run_documented_command(entry["text"], cls.folder)
        cls.elapsed = time.monotonic() - started
        cls.after = watched_state()
        print_inventory(cls.commands, cls.elapsed)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.folder, ignore_errors=True)

    def executed(self):
        return [entry for entry in self.commands if entry["reason"] is None]

    def test_every_documented_command_is_executed_or_skipped_with_a_reason(self):
        self.assertGreater(len(self.commands), 0, "no documented commands were found")
        for entry in self.commands:
            if entry["reason"] is not None:
                self.assertTrue(str(entry["reason"]).strip(), entry["text"])

    def test_documented_script_paths_exist(self):
        missing = []
        for entry in self.commands:
            for token in path_tokens(entry["text"]):
                if token.endswith(".py") and not resolve_token(token).exists():
                    missing.append("%s -> %s" % (entry["locations"][0], token))
        self.assertEqual(missing, [], "documented scripts that do not exist: %s" % missing)

    def test_host_safe_commands_succeed(self):
        failures = []
        for entry in self.executed():
            if entry["returncode"] != 0:
                failures.append("rc=%s %s\n%s"
                                % (entry["returncode"], entry["locations"][0],
                                   entry["output"][-800:]))
        self.assertEqual(failures, [], "documented host-safe commands failed:\n\n"
                         + "\n\n".join(failures))

    def test_at_least_eight_commands_executed(self):
        succeeded = sum(1 for entry in self.executed() if entry["returncode"] == 0)
        self.assertGreaterEqual(succeeded, MIN_EXECUTED,
                                "only %d documented commands executed" % succeeded)

    def test_isolated_workspace_leaves_the_repository_unchanged(self):
        changed = sorted(set(self.before) ^ set(self.after))
        changed += sorted(key for key in self.before
                          if key in self.after and self.before[key] != self.after[key])
        self.assertEqual(changed, [], "documented commands wrote into the repository: %s"
                         % changed)

    def test_module_stays_within_its_time_budget(self):
        self.assertLess(self.elapsed, MODULE_BUDGET_SECONDS,
                        "the documented commands took %.1f s" % self.elapsed)


class NotebookWalkthroughTest(unittest.TestCase):
    """E11: the walkthrough notebook is valid nbformat 4 JSON with no committed outputs."""

    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text())

    def test_valid_nbformat_4_without_outputs(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertIsInstance(self.notebook["cells"], list)
        self.assertTrue(self.notebook["cells"])
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] == "code":
                self.assertEqual(cell.get("outputs", []), [],
                                 "cell %d has committed outputs" % index)
                self.assertIsNone(cell.get("execution_count"),
                                  "cell %d has an execution count" % index)

    def test_code_cells_are_python(self):
        code_cells = 0
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                code_cells += 1
                ast.parse("".join(cell["source"]))
        self.assertGreaterEqual(code_cells, 5, "the walkthrough lost its code cells")


if __name__ == "__main__":
    unittest.main()
