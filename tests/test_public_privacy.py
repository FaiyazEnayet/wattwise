"""Privacy regression protection for the public repo (Stage 3 wattwise P0).

test_stage3_wattwise_0 is the executable regression case for the P0
contract: the repo-wide scan must return zero unallowlisted
precise-locality identifiers, the committed analysis.json must follow
the privacy schema, docs/dashboard.html must carry no locality or
coordinate query, and controlled injection of each forbidden category
must fail the scan.

This file contains no forbidden literals itself; injection fixtures
live in tests/fixtures/privacy_injection_fixtures.txt, which the
scanner deliberately skips.
"""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "privacy_injection_fixtures.txt"


def _load_scanner():
    spec = importlib.util.spec_from_file_location(
        "privacy_scan", ROOT / "scripts" / "privacy_scan.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


privacy_scan = _load_scanner()


def _fixture_section(name):
    """Return the non-comment lines under [name] in the fixture file."""
    lines, in_section = [], False
    for line in FIXTURES.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == f"[{name}]":
            in_section = True
            continue
        if stripped.startswith("["):
            in_section = False
        elif in_section and stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return "\n".join(lines)


def test_stage3_wattwise_0():
    """P0 regression case: public artifacts carry no precise locality."""
    # 1. Repo-wide scan: zero unallowlisted identifiers.
    violations = privacy_scan.scan_root(ROOT)
    assert violations == [], f"unallowlisted identifiers: {violations}"

    # 2. build_report privacy schema on the committed artifact: the
    # household descriptor is region-level only, never town or ZIP.
    report = json.loads((ROOT / "demo" / "analysis.json")
                        .read_text(encoding="utf-8"))
    household = report["household"]
    assert "town" not in household
    assert household["region"] == "Long Island, NY"
    assert privacy_scan.scan_text(json.dumps(report),
                                  "demo/analysis.json") == []

    # 3. Generated dashboard: no precise locality, no coordinate query.
    dashboard = (ROOT / "docs" / "dashboard.html").read_text(encoding="utf-8")
    assert privacy_scan.scan_text(dashboard, "docs/dashboard.html") == []
    assert "archive-api.open-meteo.com" not in dashboard

    # 4. Committed weather metadata: no coordinates, no request URL.
    metadata = json.loads((ROOT / "demo" / "household_weather.metadata.json")
                          .read_text(encoding="utf-8"))
    assert "request" not in metadata
    assert "url" not in metadata
    assert "location" not in metadata
    assert metadata["location_policy"].startswith("Exact request parameters")
    assert privacy_scan.scan_text(json.dumps(metadata),
                                  "demo/household_weather.metadata.json") == []


@pytest.mark.parametrize("section,category",
                         [("town", "town"),
                          ("zip", "zip"),
                          ("coords", "coords")])
def test_privacy_scan_detects_injected_category(section, category):
    """Controlled injection: the scanner must flag each forbidden category."""
    injected = _fixture_section(section)
    assert injected, f"fixture section [{section}] is empty"
    violations = privacy_scan.scan_text(injected, rel="injected")
    assert any(v["category"] == category for v in violations), (
        f"scanner missed injected {category}: {injected!r}")
