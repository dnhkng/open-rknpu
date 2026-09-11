"""MIT. Check every markdown link in the tree: relative targets exist, anchors resolve.

Walks the whole repository (there is no VCS metadata to enumerate from), skipping the
regenerable trees and the vendored `research/pretrained/` copies we do not own. Anchors
are resolved with the GitHub heading-slug rule plus explicit `<a name=...>` / `{#id}`
targets, so a table of contents cannot silently rot.

    python3 research/check_docs_links.py
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "build", "dist", ".venv-open", "__pycache__", "ghidra_project",
             ".pytest_cache", ".ruff_cache", "node_modules"}
VENDORED = (ROOT / "research/pretrained",)
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
HEADING = re.compile(r"^#{1,6} +(.+?)\s*$", re.M)
LINK_TARGETS = (
    re.compile(r'<a +(?:name|id)="([^"]+)"'),
    re.compile(r"\{#([\w-]+)\}"),
)


def markdown_files():
    for path in sorted(ROOT.rglob("*.md")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if any(path.is_relative_to(v) for v in VENDORED):
            continue
        yield path


def slugs(text):
    found = set()
    for match in HEADING.finditer(text):
        found.add(re.sub(r"[^a-z0-9 \-]", "", match.group(1).lower()).replace(" ", "-"))
    for pattern in LINK_TARGETS:
        found.update(pattern.findall(text))
    return found


cache = {}


def read(path):
    if path not in cache:
        try:
            cache[path] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            cache[path] = None
    return cache[path]


files = list(markdown_files())
broken = unresolved = 0
for path in files:
    text = read(path)
    if text is None:
        print(f"UNREADABLE {path.relative_to(ROOT)}")
        broken += 1
        continue
    for match in LINK.finditer(text):
        target = match.group(1).strip()
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        anchor = ""
        if "#" in target:
            target, anchor = target.split("#", 1)
        if target:
            destination = (path.parent / target).resolve()
            if not destination.exists():
                print(f"BROKEN {path.relative_to(ROOT)} -> {match.group(1)}")
                broken += 1
                continue
        else:
            destination = path
        if anchor and destination.suffix == ".md":
            target_text = read(destination)
            if target_text is not None and anchor not in slugs(target_text):
                print(f"ANCHOR {path.relative_to(ROOT)} -> {match.group(1)}")
                unresolved += 1

print(f"checked {len(files)} markdown files; broken links: {broken}; unresolved anchors: {unresolved}")
raise SystemExit(0 if not (broken or unresolved) else 1)
