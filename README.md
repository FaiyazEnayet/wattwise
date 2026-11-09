<p align="center"><img src="docs/readme-banner.svg?v=type-20261017b" alt="WattWise by Faiyaz Enayet" width="100%"></p>

<h1 align="center">WattWise</h1>

<p align="center">
  <a href="https://github.com/FaiyazEnayet/wattwise/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/FaiyazEnayet/wattwise/ci.yml?branch=main&amp;label=CI&amp;style=flat-square&amp;labelColor=24292f" alt="CI status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-996515?style=flat-square&amp;labelColor=24292f" alt="MIT license"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&amp;labelColor=24292f" alt="Python 3.10 or newer">
  <img src="https://img.shields.io/badge/built%20with-NumPy%20%2B%20SciPy%20%2B%20Plotly-355246?style=flat-square&amp;labelColor=24292f" alt="Built with NumPy, SciPy and Plotly">
</p>

<p align="center">
  <a href="https://faiyazenayet.github.io/wattwise/docs/dashboard.html"><img src="https://img.shields.io/badge/Open-Dashboard-f4edda?style=for-the-badge&amp;labelColor=0b1915" alt="Open the interactive dashboard"></a>
  <a href="docs/methodology.md"><img src="https://img.shields.io/badge/Explore-Methodology-9ac7aa?style=for-the-badge&amp;labelColor=0b1915" alt="Explore the methodology"></a>
  <a href="#run-locally"><img src="https://img.shields.io/badge/Run-Locally-193128?style=for-the-badge&amp;labelColor=0b1915" alt="Run it locally"></a>
</p>

WattWise studies my household's electricity use and asks which change is worth making, and what evidence would show that it saved money. It uses 39,800 real fifteen-minute smart-meter readings, 13 detailed utility bills, an additional earlier bill and matched town-level weather. Original measurements, reconstructed accounting, forecasts and counterfactual scenarios remain distinct.

<p align="center"><a href="https://faiyazenayet.github.io/wattwise/docs/dashboard.html"><img src="docs/analysis-demo.gif" alt="Chronological measured electricity"></a></p>

## Explore

- **Overview and meter history:** inspect the open-bill planning forecast, filter recorded days, switch between energy and demand, and inspect original quarter-hour readings and weekday load patterns.
- **Bill laboratory:** reconcile end-exclusive periods, separate service from usage-linked charges, and explain bill changes with exact Shapley attribution.
- **Tariff regret:** replay the same year under Rates 180, 194 and 195 with an editable supply assumption; inspect monthly gains and losses against Rate 194.
- **Savings frontier:** combine declared removable watts with bounded peak shifting, conserve shifted energy and avoid counting the same reduction twice.
- **Task scheduler:** find a contiguous quarter-hour cycle between an earliest start and a deadline from user-entered power and duration.
- **Measurement value:** compare acting with doing nothing under declared savings scenarios, and calculate the perfect-information ceiling before paying to resolve uncertainty.
- **Cooling and patterns:** compare daily energy with ERA5 weather, inspect uncertainty and held-out errors, test a declared AC-use scenario, and inspect daily load-shape groups, residual outliers and utility appliance cost estimates.
- **Forecast evaluation:** compare weekly persistence, a weekday median and regularized autoregression with chronological selection and a separate final holdout, including 7-, 14- and 28-day cumulative errors with explicit sample counts.
- **Evidence and intervention notebook:** inspect the source audit, log changes and confounders, then compare complete days against earlier weather/weekday-matched controls where the records overlap. Import and export local records.

## Evidence

![Measured energy, tariff scenarios, held-out forecasts and daily patterns](docs/analysis-evidence.png)

| Check | Evidence |
|---|---|
| Original export | 39,800 intervals; 4,264.91 kWh; September 19, 2025 through November 7, 2026 at 14:45 |
| Detailed bills and interval coverage | 13 detailed bills; 12 fully covered by the interval export, all within 0.5% of billed energy. The September 3 to October 2, 2025 bill starts before the export and has partial interval coverage. |
| Billing history | 15 issued bills: the 13 detailed bills, one earlier history-only bill, and the October 2026 bill documented by its bill email and portal breakdown. |
| Bill reconstruction | Maximum unrounded difference about $0.005136 over 13 detailed bills, using their stored inputs |
| Standardized annual Rate 194 subtotal | $1,148.17 energy plus service under declared replay assumptions |
| Cheapest replay alternative | Rate 195 at $1,135.45, $12.71 below Rate 194 under the declared replay assumptions; adjustments sit outside the replay scope |
| Final 28-day forecast MAE | Weekly persistence 1.749; selected autoregression 1.753 kWh/day |
| Separate neural benchmark, final 28-day daily MAE | 2.091 kWh/day, conditional on realized weather; lowest reported selection MAE, with weekly persistence (1.749) and autoregression (1.753) setting the final-holdout mark to beat |
| Separate forecast tournament decision | Weekly persistence selected under the recorded adoption rule. Ridge, constrained tree, neural and ensemble record higher cumulative MAE at both the 7- and 14-day holdout horizons under the no-future-weather protocol, which keeps the adoption bar explicit for future contenders. |
| Cooling conditional holdout MAE | 1.990 kWh/day versus 1.544 for the weekday baseline; the comparison sets a quantified target for the cooling specification |

Money subtotals are shown to cents and MAEs to three decimal places. The $12.71 tariff difference uses the unrounded report subtotals, rather than subtracting the displayed rounded amounts. The bill reconstruction difference is rounded to six decimal places.

The billing audit resolves the apparent discrepancy in the reconciliation. An earlier reconciliation included the next billing period's first day and compared some periods against chart-read estimates. Correcting the boundary and using the later detailed extraction removes most of the discrepancy, and the original readings stand as recorded, with the accounting aligned to them. The household uses individual room AC units and has no EV. Assumptions stay editable, every model result stays visible in the report, and savings of every size receive the same scrutiny.

The forecast results draw a clear line between selection and holdout performance. Autoregression leads the earlier selection window, and weekly persistence leads the final holdout by a small margin. The cooling fit establishes a temperature association and sets a quantified holdout target for the specification to meet.

The separate neural benchmark in `demo/neural_forecast.json` uses realized target-day weather, which makes it a conditional benchmark with a distinct role from a forecast using only information available in advance. It records the lowest reported one-step selection MAE, and the final holdout sets the mark for the next iteration. `demo/forecast_tournament.json` evaluates a separate no-future-weather protocol and records the adoption of weekly persistence under that protocol. Both benchmarks sit in `demo/` as standalone evidence files, separate from the shipped dashboard. The frozen report selects `ridge_ar` on the earlier selection window, and the tournament records its own protocol-specific verdict.

## Architecture

`src/wattwise/decision_analysis.py` builds the full-history report. `dashboard_ui/decisions.js` contains pure, independently tested browser arithmetic. `scripts/make_dashboard.py` embeds Plotly and the bundled Sora/IBM Plex Mono fonts once and no report data; the dashboard loads the current report from `demo/analysis.json` on each page load. The interface uses native controls, visible focus styles, bounded table scrolling, responsive CSS and reduced-motion rules without a frontend build framework.

The `wattwise insights` command and its companion modules remain available. Their heuristic bands and seeded experiments stay separate from the chronological evaluation in the dashboard. `demo/analysis.json` is the source for the interface.

## Run locally

```sh
python -m pip install -e ".[dev]"
python -m wattwise.cli analyze --root . --out demo/analysis.json
python scripts/make_dashboard.py
python -m pytest tests -q
node --test tests/test_dashboard_math.cjs
python scripts/verify_frozen_outputs.py
```

Open `docs/dashboard.html` directly. Its core analysis works offline. Optional explicit refreshes:

```sh
python scripts/fetch_household_weather.py --latitude <lat> --longitude <lon>  # coordinates are private
python scripts/make_dashboard.py
python scripts/make_research_assets.py
```

Weather metadata records its provider and date range; exact request parameters stay private. The pipeline runs on local files and on-demand steps, with live meter connection, daily scraping, cloud synchronization and appliance automation outside its operating scope. For presentation-only changes, `python scripts/make_dashboard.py --reuse-analysis` uses the stored report only when its manifest digest matches the current inputs, pricing assumptions and analysis code. It refuses reuse when the manifest is missing or the digest differs. Rebuild without that flag after changing those inputs or rules.

## Use it on another household

The report above is one household's. The `wattwise.household` module generalizes the pipeline: documented, validated schemas for interval and bill CSVs that stay in private local files, a uniform tariff interface (declared flat and time-of-use tariffs, plus Rate 194), and a `wattwise household` command that summarizes a home from its own files. Reports keep within-home results per household and cross-home aggregates separate, and never pool intervals across homes. Public CI exercises this path on a small synthetic fixture (two invented households from a declared seeded process in `wattwise.synthetic`, with a generated example report), so the path is fully testable from the fixture alone. See [the generalized pipeline guide](docs/generalized-pipeline.md) for the schemas and a worked example.

## Data and reproducibility

`demo/pseg_interval_15min.csv` stores interval kWh and average kW. `demo/pseg_rates_detailed.csv` supplies actual rates and accounting components. `demo/pseg_bills.csv` retains the longer history and earlier source estimates. `demo/pseg_appliance_estimates.csv` contains utility dollar estimates. `demo/household_weather.csv` contains Open-Meteo ERA5 reanalysis rather than home-sensor measurements; its provider, date range and units are in `demo/household_weather.metadata.json`. `demo/analysis.json` contains the results and a full evidence manifest: input paths, hashes, row counts and parser/timestamp conventions; pricing assumptions and tariff version; an analysis-code fingerprint; a manifest digest; and build annotations for the environment and Git revision. Build annotations are excluded from the reuse digest. The [methodology](docs/methodology.md) defines source precedence, calendar boundaries, counterfactual assumptions and validation protocols.

The normal `python scripts/verify_frozen_outputs.py` check compares the report at nine significant digits after normalizing its build annotations, and checks the dashboard structure and application code with the Plotly bundle masked out. The artifacts pass that check. The dashboard loads the current report from `demo/analysis.json` on each page load. The optional `--strict` dashboard byte comparison varies across environments with different Plotly builds, and the check suite treats the normalized comparison above as the reproducibility standard.

## Limitations

- Scope: one household and roughly one seasonal cycle; results describe this household, and the household module extends the same pipeline to additional homes.
- CSV timestamps carry no UTC offsets and no repeated-hour flag. Both DST transition days sit outside the daily models pending utility confirmation; original billing sums remain inspectable in full.
- Alternative tariffs use the 2026 delivery schedule, a fixed editable supply assumption and calendar-month blocks. Taxes, adjustments, eligibility and actual billing-cycle block proration sit outside the replay scope.
- Flexible energy and removable watts enter as declared assumptions. Aggregate scenarios work at household level; individual appliance jobs and comfort feasibility sit outside their scope. The separate task scheduler schedules cycles from its declared inputs; circuit capacity and device operation sit outside its scope.
- Utility appliance costs enter as estimates. Weather associations and observational intervention comparisons quantify associations and observed differences, with causal attribution outside their scope.
- Forecast bands use empirical residuals. One-step daily accuracy is not multiday accuracy; the open-bill envelope is not a calibrated monthly interval. Forecast origins overlap. In the current data every 28-day holdout window crosses the excluded November DST day, so the 28-day cumulative holdout metrics are reported as null, with estimation reserved for complete windows.
- Solar, fuel-switching and EV economics sit outside the decision results; each rests on additional data, which defines the extension path for the analysis.

## License

[MIT](LICENSE). Utility exports, weather data, bundled fonts and dependencies retain their respective terms. Weather provenance and attribution are recorded in the metadata.
