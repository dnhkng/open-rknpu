"""SPDX-License-Identifier: MIT

Keep ``docs/registers.md`` and ``docs/errors.md`` complete and unregenerable-different.

The tests independently walk ``src/open_rknpu/*.py`` so a message the generator failed
to see is still caught, and they regenerate both documents into a temporary directory
and require the bytes to match the committed files.
"""
import ast
import importlib.util
import pathlib
import re
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "open_rknpu"
DOCS = ROOT / "docs"
TYPED_EXCEPTIONS = ("ValueError", "RuntimeError", "KeyError")
GENERATOR = ROOT / "research" / "build_reference_docs.py"


def _load_generator():
    """Load the generator by path so the test does not depend on a package layout."""
    spec = importlib.util.spec_from_file_location("build_reference_docs", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _leading_literal(node):
    """Mirror of the generator's message extraction: return ``(text, dynamic)``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, False
    if isinstance(node, ast.JoinedStr):
        prefix = ""
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                prefix += value.value
            else:
                break
        return (prefix or None), True
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mod, ast.Add)):
        text, _ = _leading_literal(node.left)
        return text, True
    return None, True


def walk_raised_messages():
    """Walk ``src/open_rknpu/*.py`` for every typed raise.

    Returns ``(literal_messages, dynamic_expressions)``: the list of literal strings
    (prefixes included) and the set of source expressions that carry no literal text.
    """
    literals = []
    dynamic = set()
    for path in sorted(SRC.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
                continue
            func = node.exc.func
            if not isinstance(func, ast.Name) or func.id not in TYPED_EXCEPTIONS:
                continue
            if not node.exc.args:
                dynamic.add((path.name, ""))
                continue
            text, _dynamic = _leading_literal(node.exc.args[0])
            if text is None:
                dynamic.add((path.name, ast.unparse(node.exc.args[0])))
            else:
                literals.append(text)
    return literals, dynamic


def normalize(text):
    """Collapse whitespace and undo the markdown escaping a table cell needs."""
    text = text.replace("\\|", "|").replace("\\n", "\n").replace("\\\\", "\\")
    text = (text.replace("\u2018", "'").replace("\u2019", "'")
                .replace("\u201c", '"').replace("\u201d", '"'))
    return " ".join(text.split())


class ReferenceDocsTest(unittest.TestCase):
    def setUp(self):
        self.generator = _load_generator()

    def test_registers_doc_lists_every_register(self):
        from open_rknpu.register_profile import REGISTERS

        document = (DOCS / "registers.md").read_text()
        for register, default, tag in REGISTERS:
            self.assertIn("`0x%04x`" % register, document,
                          "register 0x%04x is missing from docs/registers.md" % register)
            self.assertIn("`0x%08x`" % default, document)
            self.assertIn("`0x%04x`" % tag, document)
        rows = [line for line in document.splitlines() if re.match(r"^\| `0x[0-9a-f]{4}` \|", line)]
        self.assertEqual(len(rows), len(REGISTERS),
                         "docs/registers.md has %d rows for %d profile registers"
                         % (len(rows), len(REGISTERS)))
        parsed = []
        for row in rows:
            match = re.match(r"^\| `0x([0-9a-f]{4})` \| `0x([0-9a-f]{4})` \| `0x([0-9a-f]{8})` \|",
                             row)
            self.assertIsNotNone(match, "malformed register row: %r" % row)
            parsed.append((int(match.group(1), 16), int(match.group(3), 16), int(match.group(2), 16)))
        self.assertEqual(sorted(parsed), sorted(REGISTERS),
                         "docs/registers.md rows do not match the REGISTERS table")

    def test_registers_doc_marks_undecoded_rows(self):
        document = (DOCS / "registers.md").read_text()
        rows = [line for line in document.splitlines() if re.match(r"^\| `0x[0-9a-f]{4}` \|", line)]
        undecoded = sum(1 for register, _default, _tag in self.generator.REGISTERS
                        if self.generator.register_meaning(register).startswith(
                            self.generator.UNDECODED))
        marked = sum(1 for row in rows if self.generator.UNDECODED in row)
        self.assertEqual(marked, undecoded)
        for row in rows:
            self.assertTrue(row.split("|")[4].strip(), "a register row has an empty meaning cell")
        addresses = {register for register, _default, _tag in self.generator.REGISTERS}
        for register, meaning in self.generator.REGISTER_MEANINGS.items():
            self.assertIn(register, addresses, "meaning table names an unknown register")
            self.assertTrue(meaning, "meaning table has an empty meaning")

    def test_errors_doc_contains_every_literal_message(self):
        document = normalize((DOCS / "errors.md").read_text())
        literals, dynamic = walk_raised_messages()
        for message in literals:
            self.assertIn(normalize(message), document,
                          "message missing from docs/errors.md: %r" % message)
        markers = sum(1 for line in (DOCS / "errors.md").read_text().splitlines()
                      if line.startswith("| *(dynamic message)*"))
        self.assertEqual(markers, len(dynamic),
                         "docs/errors.md marks %d dynamic messages for %d found in src"
                         % (markers, len(dynamic)))

    def test_every_message_has_a_curated_meaning(self):
        for entry in self.generator.extract_compiler_messages():
            lookup = (entry["text"] if entry["text"] is not None
                      else "(dynamic) " + entry["expression"])
            self.generator.error_meaning(entry["module"], lookup)

    def test_message_keys_are_interpreter_portable(self):
        """`ast.unparse` quoting differs across Python versions; the key must not.

        The drift guard keys dynamic messages by their unparsed expression, and Python 3.10
        and 3.13 render f-strings with double quotes where 3.12 uses single quotes. The
        canonical form (whitespace collapsed, quotes normalised) must map both to the same
        curated entry, otherwise the reference-doc tests pass on one interpreter only.
        """
        canonical = self.generator._canonical_expression
        single = "f'{path}: expected uint8/float32 [N,{','.join(map(str, shape[1:]))}]'"
        double = 'f"{path}: expected uint8/float32 [N,{\',\'.join(map(str, shape[1:]))}]"'
        self.assertEqual(canonical(single), canonical(double))
        self.assertIn("(dynamic) " + canonical(double), self.generator.ERROR_MEANINGS)

    def test_regenerating_matches_the_committed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.generator.write_documents(tmp)
            for name in ("registers.md", "errors.md"):
                generated = (pathlib.Path(tmp) / name).read_text()
                committed = (DOCS / name).read_text()
                self.assertEqual(generated, committed,
                                 "docs/%s differs from a fresh generator run" % name)


if __name__ == "__main__":
    unittest.main()
