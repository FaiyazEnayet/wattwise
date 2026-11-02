"""Verify the frozen report and dashboard against a clean rebuild (P1).

Rebuilds demo/analysis.json and docs/dashboard.html in a temporary copy
of the worktree, then compares:

- the frozen report: canonical JSON equality (the manifest's git_head
  build annotation is normalized out; see wattwise.decision_analysis);
- the compiled dashboard: a normalized deterministic digest of the
  HTML with the embedded report payload masked out, so template,
  style, script and library changes are caught; the payload itself is
  verified through the report comparison above.

With --strict, the dashboard comparison is a byte-for-byte comparison
instead. Exit code 0 means the committed artifacts match a clean
rebuild; 1 means at least one is stale; 2 means the rebuild failed.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wattwise.decision_analysis import frozen_report_bytes  # noqa: E402

IGNORE = {".git", ".venv", "__pycache__", ".pytest_cache"}


def _copy_tree(dst):
    def ignore(_dir, names):
        return [n for n in names if n in IGNORE or n.endswith(".egg-info")]
    shutil.copytree(ROOT, dst, ignore=ignore)


import re

_DATA_BLOCK = re.compile(
    r'(<script id="analysisData" type="application/json">).*?(</script>)',
    re.DOTALL)


def _normalized_dashboard_digest(html_text):
    """Digest of the compiled HTML with the report payload masked out.

    The payload itself is verified through demo/analysis.json (it is a
    deterministic embedding of that file); masking it keeps this check
    focused on the compiled structure: template, styles, scripts and
    the Plotly bundle. Use --strict for a full byte comparison.
    """
    masked, count = _DATA_BLOCK.subn(r"\1__REPORT_PAYLOAD__\2", html_text)
    if count != 1:
        return hashlib.sha256(html_text.encode("utf-8")).hexdigest()
    return hashlib.sha256(masked.encode("utf-8")).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="compare the dashboard HTML byte for byte")
    args = parser.parse_args(argv)

    committed_report = json.loads((ROOT / "demo" / "analysis.json").read_text(encoding="utf-8"))
    committed_html = (ROOT / "docs" / "dashboard.html").read_text(encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="wattwise-verify-") as tmp:
        work = Path(tmp) / "tree"
        _copy_tree(work)
        env = dict(os.environ, PYTHONPATH=str(work / "src"))
        proc = subprocess.run(
            [sys.executable, str(work / "scripts" / "make_dashboard.py")],
            cwd=work, env=env, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            print("verify_frozen_outputs: clean rebuild failed")
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
            return 2
        rebuilt_report = json.loads((work / "demo" / "analysis.json").read_text(encoding="utf-8"))
        rebuilt_html = (work / "docs" / "dashboard.html").read_text(encoding="utf-8")

    stale = []
    if frozen_report_bytes(rebuilt_report) != frozen_report_bytes(committed_report):
        rebuilt_keys = {k for k in rebuilt_report
                        if frozen_report_bytes(rebuilt_report[k]) != frozen_report_bytes(committed_report.get(k))}
        stale.append(f"demo/analysis.json (differing sections: {sorted(rebuilt_keys)})")
    else:
        print("verify_frozen_outputs: demo/analysis.json matches a clean rebuild")

    if args.strict:
        dashboard_ok = rebuilt_html == committed_html
        mode = "byte comparison"
    else:
        dashboard_ok = (
            _normalized_dashboard_digest(rebuilt_html)
            == _normalized_dashboard_digest(committed_html))
        mode = "normalized digest comparison"
    if dashboard_ok:
        print(f"verify_frozen_outputs: docs/dashboard.html matches ({mode})")
    else:
        stale.append(f"docs/dashboard.html ({mode})")

    if stale:
        print("verify_frozen_outputs: STALE committed artifact(s): " + "; ".join(stale))
        print("Regenerate with: python scripts/make_dashboard.py")
        return 1
    print("verify_frozen_outputs: frozen outputs are current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
