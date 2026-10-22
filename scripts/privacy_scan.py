"""Repo-wide privacy scan for precise-locality identifiers.

Scans every text file under the repo root (excluding .git and other
generated caches) for forbidden identifier categories and reports any
violation that is not explicitly allowlisted. Exit 0 when clean, 1
otherwise. Also importable: scan_root() and scan_text() return a list of
violation dicts.

Forbidden categories
--------------------
- town: the household town name, any case
- zip: the household ZIP code as a standalone token
- coords: the weather-request coordinates as standalone tokens

The allowlist (scripts/privacy_allowlist.txt) records exact
"path :: literal" pairs that are deliberately retained, each with a
reason. This script, the allowlist and the labeled injection fixtures
skip themselves so the scan stays self-consistent.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

CATEGORIES = {
    "town": [re.compile(r"port jefferson", re.IGNORECASE)],
    "zip": [re.compile(r"\b11777\b")],
    "coords": [re.compile(r"\b40\.9465\b"), re.compile(r"\b-?73\.0693\b")],
}

# Files that legitimately name the forbidden tokens and are therefore
# skipped by the repo-wide scan.
SELF_SKIP = frozenset({
    "scripts/privacy_scan.py",
    "scripts/privacy_allowlist.txt",
    "tests/fixtures/privacy_injection_fixtures.txt",
})

SKIP_DIRS = frozenset({".git", "__pycache__", ".pytest_cache", ".venv", "node_modules"})


def load_allowlist(root: Path) -> dict[str, list[str]]:
    """Parse scripts/privacy_allowlist.txt into {relpath: [literal, ...]}."""
    allow: dict[str, list[str]] = {}
    path = Path(root) / "scripts" / "privacy_allowlist.txt"
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rel, _, literal = line.partition("::")
        allow.setdefault(rel.strip(), []).append(literal.strip())
    return allow


def scan_text(text: str, rel: str, allow: dict[str, list[str]] | None = None) -> list[dict]:
    """Return violations found in one text blob.

    A match is a violation unless an allowlist literal for this file
    contains the matched text.
    """
    allow = allow or {}
    violations: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for category, patterns in CATEGORIES.items():
            for rx in patterns:
                for match in rx.finditer(line):
                    token = match.group(0)
                    if any(token in lit for lit in allow.get(rel, [])):
                        continue
                    violations.append({
                        "file": rel,
                        "line": lineno,
                        "category": category,
                        "match": token,
                    })
    return violations


def scan_root(root: str | Path) -> list[dict]:
    """Scan the whole repo tree; return all unallowlisted violations."""
    root = Path(root)
    allow = load_allowlist(root)
    violations: list[dict] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            full = Path(dirpath) / name
            rel = str(full.relative_to(root)).replace(os.sep, "/")
            if rel in SELF_SKIP:
                continue
            try:
                text = full.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            violations.extend(scan_text(text, rel, allow))
    return violations


def main(argv: list[str] | None = None) -> int:
    root = Path(argv[1]) if argv and len(argv) > 1 else Path.cwd()
    violations = scan_root(root)
    if violations:
        for v in violations:
            print(f"{v['file']}:{v['line']}: [{v['category']}] {v['match']!r}")
        print(f"{len(violations)} unallowlisted identifier(s) found")
        return 1
    print("privacy scan clean: no unallowlisted identifiers")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
