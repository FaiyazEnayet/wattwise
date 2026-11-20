"""Stage 3 wattwise P11: metrics and artifact registry.

Every published numeric metric must trace to a hash-addressed input, a
unit and a reproduction command, with zero dangling references: each
registry metric resolves to a value inside its hash-addressed artifact,
and a deliberately removed artifact makes validation fail.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.reporting import (  # noqa: E402
    RegistryError,
    check_registry,
    load_registry,
    render_metric_table,
    render_registry_section,
    render_run_table,
    validate_registry,
)

REGISTRY = ROOT / "experiments" / "registry.jsonl"


def _mini_registry(tmp_path):
    """A tiny registry pointing at a small synthetic artifact."""
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({"result": {"mae": 1.564}}), encoding="utf-8")
    source = tmp_path / "input.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    import hashlib
    registry = tmp_path / "registry.jsonl"
    registry.write_text(json.dumps({
        "run_id": "toy-run",
        "title": "Toy run",
        "artifact": "artifact.json",
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "inputs": [{"path": "input.csv",
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}],
        "temporal_split": {"description": "Selection 2 days, holdout 1 day",
                           "selection_days": 2, "holdout_days": 1},
        "reproduction_command": "python scripts/run_toy.py",
        "metrics": [{
            "name": "toy MAE",
            "value": 1.564,
            "unit": "kWh/day",
            "estimator": "toy estimator",
            "comparator": "toy baseline",
            "n_origins": 1,
            "uncertainty": "single origin; no interval reported",
            "caveats": [],
            "artifact_key": "result.mae",
        }],
    }) + "\n", encoding="utf-8")
    return tmp_path, registry


def test_stage3_wattwise_11_registry_loads_and_validates():
    entries = load_registry(ROOT)
    assert len(entries) == 3
    assert {e["run_id"] for e in entries} == {
        "frozen-report-v1-2-0", "forecast-tournament-e1", "neural-benchmark-fixed"}
    assert validate_registry(ROOT) == []


def test_stage3_wattwise_11_every_metric_resolves_to_its_artifact():
    for entry in load_registry(ROOT):
        artifact = json.loads((ROOT / entry["artifact"]).read_text(encoding="utf-8"))
        assert entry["metrics"], entry["run_id"]
        for metric in entry["metrics"]:
            assert metric["unit"] and metric["estimator"] and metric["comparator"]
            assert metric["uncertainty"] and isinstance(metric["caveats"], list)
            assert metric["n_origins"] is None or metric["n_origins"] >= 1
            assert entry["reproduction_command"].startswith("python ")
            assert (ROOT / entry["artifact"]).exists()
    assert validate_registry(ROOT) == []


def test_stage3_wattwise_11_tampered_artifact_hash_fails(tmp_path):
    root, registry = _mini_registry(tmp_path)
    assert validate_registry(root, registry) == []
    (root / "artifact.json").write_text(json.dumps({"result": {"mae": 9.999}}),
                                        encoding="utf-8")
    errors = validate_registry(root, registry)
    assert errors and any("artifact" in e and "sha256" in e for e in errors)


def test_stage3_wattwise_11_removed_artifact_fails_validation(tmp_path):
    root, registry = _mini_registry(tmp_path)
    (root / "artifact.json").unlink()
    errors = validate_registry(root, registry)
    assert errors and any("missing" in e for e in errors)
    with pytest.raises(RegistryError):
        check_registry(root, registry)


def test_stage3_wattwise_11_removed_input_fails_validation(tmp_path):
    root, registry = _mini_registry(tmp_path)
    (root / "input.csv").unlink()
    errors = validate_registry(root, registry)
    assert errors and any("input" in e and "missing" in e for e in errors)


def test_stage3_wattwise_11_metric_value_must_match_artifact(tmp_path):
    root, registry = _mini_registry(tmp_path)
    lines = registry.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[0])
    entry["metrics"][0]["value"] = 2.5
    registry.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    errors = validate_registry(root, registry)
    assert errors and any("does not match" in e for e in errors)


def test_stage3_wattwise_11_duplicate_run_id_rejected(tmp_path):
    root, registry = _mini_registry(tmp_path)
    entry = json.loads(registry.read_text(encoding="utf-8"))
    with registry.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    errors = validate_registry(root, registry)
    assert errors and any("duplicate run_id" in e for e in errors)


def test_stage3_wattwise_11_metric_without_unit_rejected(tmp_path):
    root, registry = _mini_registry(tmp_path)
    entry = json.loads(registry.read_text(encoding="utf-8"))
    del entry["metrics"][0]["unit"]
    registry.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    errors = validate_registry(root, registry)
    assert errors and any("unit" in e for e in errors)


def test_stage3_wattwise_11_null_metric_needs_a_reason(tmp_path):
    root, registry = _mini_registry(tmp_path)
    entry = json.loads(registry.read_text(encoding="utf-8"))
    entry["metrics"][0]["value"] = None
    entry["metrics"][0]["artifact_key"] = "result.missing"
    registry.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    errors = validate_registry(root, registry)
    assert errors  # a null value must explain itself in caveats or uncertainty


def test_stage3_wattwise_11_rendered_tables_carry_run_ids_and_units():
    entries = load_registry(ROOT)
    run_table = render_run_table(entries)
    for entry in entries:
        assert entry["run_id"] in run_table
        assert entry["artifact"] in run_table
        assert entry["reproduction_command"] in run_table
    metric_table = render_metric_table(entries[0])
    assert "kWh/day" in metric_table or "USD" in metric_table
    assert "frozen-report-v1-2-0" not in metric_table  # metric rows carry no run id


def test_stage3_wattwise_11_readme_block_matches_generated_section():
    entries = load_registry(ROOT)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"<!-- BEGIN REGISTERED RUNS -->(.*?)<!-- END REGISTERED RUNS -->",
                      readme, re.DOTALL)
    assert match, "README must carry the generated registered-runs block"
    assert match.group(0).strip() == render_registry_section(entries).strip()


def test_stage3_wattwise_11_rebuilt_report_still_validates(tmp_path):
    """A clean rebuild rewrites the report's build annotations only.

    The registry hashes the frozen report over its canonical frozen
    bytes, so the rebuilt file (new git revision recorded) must still
    validate. This is the CI order: make_dashboard.py rebuilds the
    report before verify_frozen_outputs.py runs.
    """
    root = tmp_path / "tree"
    (root / "demo").mkdir(parents=True)
    (root / "experiments").mkdir()
    shutil.copy(ROOT / "experiments" / "registry.jsonl",
                root / "experiments" / "registry.jsonl")
    for name in ["analysis.json", "forecast_tournament.json",
                 "neural_forecast.json", "pseg_interval_15min.csv",
                 "pseg_rates_detailed.csv", "pseg_bills.csv",
                 "pseg_appliance_estimates.csv", "household_weather.csv"]:
        shutil.copy(ROOT / "demo" / name, root / "demo" / name)
    assert validate_registry(root) == []
    report_path = root / "demo" / "analysis.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["manifest"]["git_head"] = "deadbeef-rebuilt"
    report["manifest"]["environment"] = {"python": "3.12.99"}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert validate_registry(root) == []


def test_stage3_wattwise_11_unknown_hash_mode_rejected(tmp_path):
    root, registry = _mini_registry(tmp_path)
    entry = json.loads(registry.read_text(encoding="utf-8"))
    entry["artifact_hash_mode"] = "magic"
    registry.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    errors = validate_registry(root, registry)
    assert errors and any("artifact_hash_mode" in e for e in errors)


def test_stage3_wattwise_11_removed_artifact_fails_verifier(tmp_path):
    # Copy the pieces the verifier needs into a clean tree.
    work = tmp_path / "tree"
    for name in ["src", "scripts", "dashboard_ui", "portfolio_ui", "demo",
                 "evaluation", "experiments"]:
        shutil.copytree(ROOT / name, work / name)
    shutil.copy(ROOT / "pyproject.toml", work / "pyproject.toml")
    (work / "docs").mkdir()
    shutil.copy(ROOT / "docs" / "dashboard.html", work / "docs" / "dashboard.html")
    # Deliberately remove a registered artifact.
    (work / "demo" / "neural_forecast.json").unlink()
    proc = subprocess.run(
        [sys.executable, str(work / "scripts" / "verify_frozen_outputs.py")],
        capture_output=True, text=True, cwd=work, timeout=600)
    assert proc.returncode == 1
    assert "neural_forecast.json" in proc.stdout + proc.stderr
