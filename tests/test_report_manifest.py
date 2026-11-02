"""Stage 3 wattwise P1: reproducible evidence manifests + build immutability.

The frozen report must carry a manifest that fingerprints every input,
the pricing assumptions and the analysis code, so that reusing a stale
report is refused and a clean rebuild can be compared byte for byte
(after canonicalization) against the committed report.
"""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from wattwise.decision_analysis import (
    DEFAULT_SUPPLY,
    TARIFF_SCHEDULES,
    ManifestMismatch,
    build_manifest,
    build_report,
    canonical_json_bytes,
    code_fingerprint,
    frozen_report_bytes,
    manifest_digest,
    validate_report_contract,
    verify_reuse,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILES = [
    "pseg_interval_15min.csv",
    "pseg_rates_detailed.csv",
    "pseg_bills.csv",
    "pseg_appliance_estimates.csv",
    "household_weather.csv",
]


def _mini_root(tmp_path):
    """A root whose demo/ holds copies of the real report inputs."""
    demo = tmp_path / "demo"
    demo.mkdir()
    for name in INPUT_FILES:
        shutil.copy(ROOT / "demo" / name, demo / name)
    return tmp_path


def _mutate(path):
    data = bytearray(path.read_bytes())
    # Flip one digit byte well inside the file to another digit.
    for i in range(len(data) // 2, len(data)):
        if 48 <= data[i] <= 56:  # '0'..'8'
            data[i] = data[i] + 1
            break
    path.write_bytes(bytes(data))


def test_stage3_wattwise_1_manifest_records_every_input(tmp_path):
    root = _mini_root(tmp_path)
    manifest = build_manifest(root)
    roles = [e["role"] for e in manifest["inputs"]]
    assert roles == ["intervals", "bill_details", "billing_history",
                     "appliance_estimates", "weather"]
    for entry in manifest["inputs"]:
        raw = (root / entry["path"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(raw).hexdigest()
        assert entry["bytes"] == len(raw)
        assert entry["rows"] > 0
        assert entry["path"].startswith("demo/")
    assert manifest["pricing"]["supply_usd_per_kwh"] == DEFAULT_SUPPLY
    assert manifest["pricing"]["tariff_schedule_version"] == TARIFF_SCHEDULES[0].version
    assert manifest["code"]["sha256"] == code_fingerprint()
    env = manifest["environment"]
    assert env["python"].startswith("3.")
    assert env["pandas"] and env["numpy"] and env["scipy"]
    assert manifest["manifest_digest"] == manifest_digest(manifest)


@pytest.mark.parametrize("name", ["pseg_interval_15min.csv",
                                  "pseg_rates_detailed.csv",
                                  "pseg_bills.csv",
                                  "household_weather.csv"])
def test_stage3_wattwise_1_single_input_change_changes_digest(tmp_path, name):
    root = _mini_root(tmp_path)
    before = build_manifest(root)["manifest_digest"]
    _mutate(root / "demo" / name)
    after = build_manifest(root)["manifest_digest"]
    assert after != before


def test_stage3_wattwise_1_code_change_changes_fingerprint(tmp_path):
    target = tmp_path / "rules.py"
    target.write_text("RULE = 1\n", encoding="utf-8")
    first = code_fingerprint([target])
    target.write_text("RULE = 2\n", encoding="utf-8")
    assert code_fingerprint([target]) != first


def test_stage3_wattwise_1_reuse_refused_after_input_change(tmp_path):
    root = _mini_root(tmp_path)
    stored = {"manifest": build_manifest(root)}
    verify_reuse(root, stored)  # must not raise
    _mutate(root / "demo" / "household_weather.csv")
    with pytest.raises(ManifestMismatch):
        verify_reuse(root, stored)


def test_stage3_wattwise_1_reuse_refused_without_manifest(tmp_path):
    root = _mini_root(tmp_path)
    with pytest.raises(ManifestMismatch):
        verify_reuse(root, {"schema_version": 1})


def test_stage3_wattwise_1_manifest_has_no_private_values(tmp_path):
    root = _mini_root(tmp_path)
    manifest = build_manifest(root)
    blob = json.dumps(manifest)
    assert str(root) not in blob
    assert "latitude" not in blob and "longitude" not in blob
    for entry in manifest["inputs"]:
        assert not Path(entry["path"]).is_absolute()


def test_stage3_wattwise_1_canonical_json_rejects_nonfinite():
    with pytest.raises(ValueError):
        canonical_json_bytes({"x": float("nan")})
    with pytest.raises(ValueError):
        canonical_json_bytes({"x": float("inf")})
    # Canonical form is order-insensitive and stable.
    assert canonical_json_bytes({"b": 1, "a": [1.5, -0.0]}) == \
        canonical_json_bytes({"a": [1.5, 0.0], "b": 1})


def test_stage3_wattwise_1_clean_build_matches_frozen_report():
    report = build_report(ROOT)
    validate_report_contract(report)
    committed = json.loads((ROOT / "demo" / "analysis.json").read_text(encoding="utf-8"))
    assert frozen_report_bytes(report) == frozen_report_bytes(committed)
    assert report["manifest"]["manifest_digest"] == manifest_digest(report["manifest"])


def test_stage3_wattwise_1_stale_frozen_report_fails_verifier(tmp_path):
    # Copy the pieces the verifier needs into a clean tree.
    work = tmp_path / "tree"
    for name in ["src", "scripts", "dashboard_ui", "portfolio_ui", "demo"]:
        shutil.copytree(ROOT / name, work / name)
    shutil.copy(ROOT / "pyproject.toml", work / "pyproject.toml")
    (work / "docs").mkdir()
    shutil.copy(ROOT / "docs" / "dashboard.html", work / "docs" / "dashboard.html")
    # Deliberately stale: change one published value in the frozen report.
    frozen = work / "demo" / "analysis.json"
    report = json.loads(frozen.read_text(encoding="utf-8"))
    report["summary"]["total_billed_usd"] += 1.0
    frozen.write_text(json.dumps(report, separators=(",", ":"), allow_nan=False),
                      encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(work / "scripts" / "verify_frozen_outputs.py")],
        capture_output=True, text=True, cwd=work, timeout=1200)
    assert proc.returncode == 1
    assert "analysis.json" in proc.stdout + proc.stderr
