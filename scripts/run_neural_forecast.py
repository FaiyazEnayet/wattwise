"""Run the neural-forecaster benchmark and write demo/neural_forecast.json.

Rebuilds the daily series from the committed interval export and the
cooling degree-days from the committed weather file, then evaluates the
neural candidate against the existing baselines on identical frozen
origins. The candidate is reported win or lose; nothing here promotes it.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wattwise.decision_analysis import interval_quality, read_intervals
from wattwise.neural_forecast import evaluate_neural


def main():
    root = Path(__file__).resolve().parents[1]
    intervals = read_intervals(root / "demo" / "pseg_interval_15min.csv")
    daily, _quality = interval_quality(intervals)
    result = evaluate_neural(daily, root / "demo" / "household_weather.csv")
    if not result.get("available"):
        raise SystemExit(f"neural benchmark unavailable: {result.get('reason')}")
    out = root / "demo" / "neural_forecast.json"
    out.write_text(json.dumps(result, indent=2))
    for row in result["one_step"]:
        print(f"{row['model']:15s} selection MAE {row['selection_mae']:.3f} "
              f"holdout MAE {row['holdout_mae']:.3f} "
              f"WAPE {row['holdout_wape']:.3f}")
    print(f"chosen config: {result['chosen_config']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
