"""Run the forecast model tournament and write demo/forecast_tournament.json.

Rebuilds the daily series from the committed interval export and the
cooling degree-days from the committed weather file, then evaluates the
control (weekly persistence), ridge, constrained tree, neural, and
ensemble candidates on identical frozen origins. The decision rule is
applied mechanically; the result is reported win, lose, or retain, and
nothing here promotes a model automatically.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wattwise.decision_analysis import interval_quality, read_intervals
from wattwise.tournament import run_tournament


def main():
    root = Path(__file__).resolve().parents[1]
    intervals = read_intervals(root / "demo" / "pseg_interval_15min.csv")
    daily, _quality = interval_quality(intervals)
    result = run_tournament(daily, root / "demo" / "household_weather.csv")
    if not result.get("available"):
        raise SystemExit(f"tournament unavailable: {result.get('reason')}")
    out = root / "demo" / "forecast_tournament.json"
    out.write_text(json.dumps(result, indent=2))
    for row in result["one_step"]:
        print(f"{row['model']:16s} selection MAE {row['selection_mae']:.3f} "
              f"holdout MAE {row['holdout_mae']:.3f} "
              f"WAPE {row['holdout_wape']:.3f}")
    for row in result["horizons"]:
        mae = row["mae_kwh"]
        print(f"{row['model']:16s} horizon {row['horizon']:2d}d "
              f"MAE {mae if mae is None else round(mae, 3)} "
              f"origins {row['holdout_origins']}")
    decision = result["decision"]
    print(f"decision: {decision['decision']}"
          + (f" winner: {decision['winner']}" if decision["winner"] else ""))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
