"""Stage 3 WW12: generalized household pipeline.

Pins:
- Documented interval/bill schemas are validated on load: missing or
  extra columns, unparseable or duplicate timestamps, non-finite or
  negative energy, and inverted bill periods all fail loudly.
- The uniform tariff interface prices hand-computed fixtures exactly,
  and the Rate 194 adapter agrees with the existing analyze path.
- Outputs are deterministic: repeated summaries give identical
  canonical bytes, and the committed synthetic fixture regenerates
  byte for byte from its declared generation process.
- Within-home results stay per household and cross-home results are
  reported separately as labeled aggregates over the per-home values.
- The synthetic fixture and the example report carry no private
  identifiers and no measured household data.

The synthetic fixture is generated from a declared synthetic process
(invented parameters, seeded generator; see wattwise.synthetic), never
derived from the measured household series.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from wattwise import household as hh
from wattwise import synthetic

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "synthetic"


def _write_intervals(tmp_path, rows, header="timestamp,kwh"):
    path = tmp_path / "intervals.csv"
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _flat_dataset(tmp_path):
    path = _write_intervals(tmp_path, [
        "2026-01-05T00:00:00,1.0",
        "2026-01-05T01:00:00,2.0",
        "2026-01-05T02:00:00,3.0",
        "2026-01-05T03:00:00,4.0",
    ])
    tariff = hh.FlatTariff(name="flat-fixture", delivery_per_kwh=0.10,
                           supply_per_kwh=0.15, daily_service_charge=0.50)
    return hh.HouseholdDataset(household_id="fixture-home",
                               intervals=hh.load_household_intervals(path),
                               tariff=tariff)


# --- schema validation -----------------------------------------------------

def test_stage3_wattwise_12_schema_accepts_canonical_fixture(tmp_path):
    ds = _flat_dataset(tmp_path)
    assert list(ds.intervals.columns) == ["kwh"]
    assert len(ds.intervals) == 4
    assert float(ds.intervals["kwh"].sum()) == pytest.approx(10.0)


@pytest.mark.parametrize("rows,header", [
    (["2026-01-05T00:00:00,1.0"], "timestamp,energy"),          # wrong column
    (["2026-01-05T00:00:00,1.0,9"], "timestamp,kwh,extra"),    # extra column
    (["not-a-time,1.0"], "timestamp,kwh"),                      # bad timestamp
    (["2026-01-05T00:00:00,-1.0"], "timestamp,kwh"),            # negative kwh
    (["2026-01-05T00:00:00,nan"], "timestamp,kwh"),             # non-finite kwh
    (["2026-01-05T00:00:00,1.0", "2026-01-05T00:00:00,2.0"],
     "timestamp,kwh"),                                          # duplicate ts
    (["2026-01-05T01:00:00,1.0", "2026-01-05T00:00:00,2.0"],
     "timestamp,kwh"),                                          # out of order
])
def test_stage3_wattwise_12_schema_rejects_invalid_intervals(tmp_path, rows, header):
    path = _write_intervals(tmp_path, rows, header=header)
    with pytest.raises(hh.SchemaError):
        hh.load_household_intervals(path)


def test_stage3_wattwise_12_bills_schema_validation(tmp_path):
    good = tmp_path / "bills.csv"
    good.write_text(
        "period_start,period_end,kwh,bill_usd\n"
        "2026-01-01,2026-02-01,100.0,25.50\n", encoding="utf-8")
    bills = hh.load_household_bills(good)
    assert len(bills) == 1

    bad = tmp_path / "bills_bad.csv"
    bad.write_text(
        "period_start,period_end,kwh,bill_usd\n"
        "2026-02-01,2026-01-01,100.0,25.50\n", encoding="utf-8")
    with pytest.raises(hh.SchemaError):
        hh.load_household_bills(bad)

    negative = tmp_path / "bills_neg.csv"
    negative.write_text(
        "period_start,period_end,kwh,bill_usd\n"
        "2026-01-01,2026-02-01,-3.0,25.50\n", encoding="utf-8")
    with pytest.raises(hh.SchemaError):
        hh.load_household_bills(negative)


def test_stage3_wattwise_12_household_id_must_be_anonymous_label():
    tariff = hh.FlatTariff(name="f", delivery_per_kwh=0.1,
                           supply_per_kwh=0.1, daily_service_charge=0.5)
    frame = pd.DataFrame({"kwh": [1.0]},
                         index=pd.to_datetime(["2026-01-05T00:00:00"]))
    with pytest.raises(hh.SchemaError):
        hh.HouseholdDataset(household_id="My Home",
                            intervals=frame, tariff=tariff)


# --- tariff interface ------------------------------------------------------

def test_stage3_wattwise_12_flat_tariff_known_answer(tmp_path):
    summary = hh.summarize_household(_flat_dataset(tmp_path))
    # 10 kWh at $0.25 all-in = $2.50 energy; 1 day at $0.50 service.
    assert summary["total_kwh"] == pytest.approx(10.0)
    assert summary["bill_estimate"]["energy_usd"] == pytest.approx(2.50)
    assert summary["bill_estimate"]["fixed_usd"] == pytest.approx(0.50)
    assert summary["bill_estimate"]["total_usd"] == pytest.approx(3.00)
    assert summary["peak"] is None  # a flat tariff has no peak window


def test_stage3_wattwise_12_tou_tariff_known_answer():
    import datetime as dt
    tariff = hh.TimeOfUseTariff(
        name="tou-fixture", peak_delivery_per_kwh=0.20,
        offpeak_delivery_per_kwh=0.10, peak_supply_per_kwh=0.30,
        offpeak_supply_per_kwh=0.10, daily_service_charge=0.50)
    monday_4pm = dt.datetime(2026, 1, 5, 16)   # Monday: peak window
    monday_10am = dt.datetime(2026, 1, 5, 10)  # Monday: off-peak
    saturday_4pm = dt.datetime(2026, 1, 10, 16)  # Saturday: off-peak
    assert tariff.is_peak(monday_4pm)
    assert not tariff.is_peak(monday_10am)
    assert not tariff.is_peak(saturday_4pm)
    assert tariff.energy_charge(2.0, monday_4pm) == pytest.approx(2.0 * 0.50)
    assert tariff.energy_charge(2.0, monday_10am) == pytest.approx(2.0 * 0.20)
    assert tariff.fixed_charge(30) == pytest.approx(15.0)


def test_stage3_wattwise_12_tariff_from_spec_validation():
    tariff = hh.tariff_from_spec({"kind": "flat", "name": "x",
                                  "delivery_per_kwh": 0.1,
                                  "supply_per_kwh": 0.2,
                                  "daily_service_charge": 0.5})
    assert isinstance(tariff, hh.FlatTariff)
    with pytest.raises(hh.SchemaError):
        hh.tariff_from_spec({"kind": "mystery"})
    with pytest.raises(hh.SchemaError):
        hh.tariff_from_spec({"kind": "flat", "name": "x",
                             "delivery_per_kwh": -0.1,
                             "supply_per_kwh": 0.2,
                             "daily_service_charge": 0.5})


def test_stage3_wattwise_12_rate194_adapter_matches_analyze(tmp_path):
    from wattwise.analyze import monthly_bill
    from wattwise.rates import Rate194

    path = _write_intervals(tmp_path, [
        f"2026-01-{day:02d}T{hour:02d}:00:00,{(day + hour) / 10.0}"
        for day in range(5, 12) for hour in range(24)
    ])
    intervals = hh.load_household_intervals(path)
    hourly = intervals["kwh"]
    expected = monthly_bill(hourly, Rate194())
    ds = hh.HouseholdDataset(household_id="adapter-home", intervals=intervals,
                             tariff=hh.rate194_tariff())
    summary = hh.summarize_household(ds)
    assert summary["bill_estimate"]["total_usd"] == pytest.approx(
        expected["total"], abs=0.011)
    assert summary["peak"] is not None
    assert 0.0 < summary["peak"]["peak_kwh"] < summary["total_kwh"]


# --- determinism -----------------------------------------------------------

def test_stage3_wattwise_12_deterministic_outputs(tmp_path):
    ds = _flat_dataset(tmp_path)
    first = hh.report_bytes(hh.build_household_report([ds]))
    second = hh.report_bytes(hh.build_household_report([ds]))
    assert first == second

    a = synthetic.make_synthetic_intervals("synthetic-a")
    b = synthetic.make_synthetic_intervals("synthetic-a")
    pd.testing.assert_frame_equal(a, b)
    c = synthetic.make_synthetic_intervals("synthetic-b")
    assert not a["kwh"].equals(c["kwh"])


def test_stage3_wattwise_12_committed_fixture_regenerates_byte_exact():
    from scripts.make_synthetic_fixture import render_fixture_files
    for rel, expected_bytes in render_fixture_files().items():
        committed = (FIXTURE_DIR / rel).read_bytes()
        assert committed == expected_bytes, f"fixture drift in {rel}"


# --- fixture, example report, within/cross-home separation -----------------

def _load_fixture_datasets():
    manifest = json.loads((FIXTURE_DIR / "households.json")
                          .read_text(encoding="utf-8"))
    datasets = []
    for home in manifest["households"]:
        hid = home["household_id"]
        datasets.append(hh.HouseholdDataset(
            household_id=hid,
            intervals=hh.load_household_intervals(
                FIXTURE_DIR / f"{hid}_intervals.csv"),
            bills=hh.load_household_bills(FIXTURE_DIR / f"{hid}_bills.csv"),
            tariff=hh.tariff_from_spec(home["tariff"]),
        ))
    return manifest, datasets


def test_stage3_wattwise_12_fixture_loads_and_reconciles():
    manifest, datasets = _load_fixture_datasets()
    assert manifest["generation"]["method"]
    assert "synthetic" in manifest["generation"]["method"].lower()
    assert len(datasets) == 2
    for ds in datasets:
        assert 0 < len(ds.intervals) <= 2000  # small public fixture
        summary = hh.summarize_household(ds)
        # The generator priced the fixture bills from the same synthetic
        # intervals, so reconciliation is exact up to cent rounding.
        assert summary["bill_reconciliation"] is not None
        for period in summary["bill_reconciliation"]:
            assert period["difference_pct"] == pytest.approx(0.0, abs=1e-9)
            assert period["reconstruction_difference_usd"] == pytest.approx(
                0.0, abs=0.011)


def test_stage3_wattwise_12_within_and_cross_home_reported_separately():
    _, datasets = _load_fixture_datasets()
    report = hh.build_household_report(datasets)

    within = report["within_home"]
    assert sorted(within) == ["synthetic-a", "synthetic-b"]
    for ds in datasets:
        standalone = hh.summarize_household(ds)
        assert within[ds.household_id] == standalone

    cross = report["cross_home"]
    assert cross["household_count"] == 2
    assert cross["household_ids"] == ["synthetic-a", "synthetic-b"]
    # Cross-home carries labeled aggregates over the per-home values and
    # keeps each home's total attributed; it never pools the homes into
    # one series or one household total.
    assert cross["total_kwh_by_home"] == {
        hid: within[hid]["total_kwh"] for hid in within}
    stats = cross["mean_daily_kwh"]
    values = sorted(within[h]["mean_daily_kwh"] for h in within)
    assert stats["min"] == pytest.approx(values[0])
    assert stats["max"] == pytest.approx(values[-1])
    assert values[0] <= stats["median"] <= values[-1]
    assert "intervals" not in json.dumps(cross)
    assert report["scope"].startswith("Each home")


def test_stage3_wattwise_12_example_report_matches_fixture():
    _, datasets = _load_fixture_datasets()
    committed = (FIXTURE_DIR / "example_report.json").read_bytes()
    expected = hh.report_bytes(hh.build_household_report(datasets)) + b"\n"
    assert committed == expected
    parsed = json.loads(committed)
    assert parsed["schema"] == "wattwise-household-report/1"
    assert sorted(parsed["within_home"]) == ["synthetic-a", "synthetic-b"]


def test_stage3_wattwise_12_fixture_has_no_private_identifiers():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "privacy_scan", ROOT / "scripts" / "privacy_scan.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for path in sorted(FIXTURE_DIR.iterdir()):
        text = path.read_text(encoding="utf-8")
        assert module.scan_text(text, path.name) == [], path.name
        lowered = text.lower()
        assert "@" not in text  # no email-shaped content anywhere


def test_stage3_wattwise_12_cli_household_report(tmp_path):
    from wattwise.cli import main
    intervals = FIXTURE_DIR / "synthetic-a_intervals.csv"
    manifest = json.loads((FIXTURE_DIR / "households.json")
                          .read_text(encoding="utf-8"))
    tariff_path = tmp_path / "tariff.json"
    tariff_path.write_text(json.dumps(manifest["households"][0]["tariff"]),
                           encoding="utf-8")
    out = tmp_path / "report.json"
    rc = main(["household", "--id", "synthetic-a",
               "--intervals", str(intervals),
               "--bills", str(FIXTURE_DIR / "synthetic-a_bills.csv"),
               "--tariff-json", str(tariff_path),
               "--out", str(out)])
    assert rc == 0
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert list(parsed["within_home"]) == ["synthetic-a"]
    assert parsed["cross_home"]["household_count"] == 1
