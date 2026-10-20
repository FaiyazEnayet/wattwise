"""Build docs/dashboard.html: one self-contained interactive Plotly page.

Panels: key stats, load curve (actual vs optimized), bill forecast with
bands, TOU heatmap, savings waterfall, appliance breakdown, weather
correlation, what-if scenarios, anomaly callout. Dark theme.
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from wattwise import (Rate180, Rate194, forecast_month_bill, optimize_month,
                      temperature_sensitivity, solar_scenario, ev_scenario,
                      heat_pump_scenario, savings_waterfall, peak_share)

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "..", "demo")
DOCS = os.path.join(HERE, "..", "docs")

BG = "#0e1117"
PANEL = "#161b22"
GRID = "#2d333b"
TEXT = "#e6edf3"
ACCENT = "#58a6ff"
ORANGE = "#f0883e"
GREEN = "#3fb950"
RED = "#f85149"
PURPLE = "#bc8cff"


def _load():
    hourly = pd.read_csv(os.path.join(DEMO, "profile.csv"), index_col=0,
                         parse_dates=True).iloc[:, 0]
    hourly.index = pd.to_datetime(hourly.index)
    end_use = pd.read_csv(os.path.join(DEMO, "end_use.csv"), index_col=0,
                          parse_dates=True)
    end_use.index = pd.to_datetime(end_use.index)
    temp = pd.read_csv(os.path.join(DEMO, "weather.csv"), index_col=0,
                       parse_dates=True).iloc[:, 0]
    temp.index = pd.to_datetime(temp.index)
    metrics = json.load(open(os.path.join(DEMO, "metrics.json")))
    return hourly, end_use, temp, metrics


def _layout(fig, title, height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=TEXT)),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter, system-ui, sans-serif"),
        height=height, margin=dict(l=50, r=30, t=60, b=50),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def legacy_main():
    hourly, end_use, temp, m = _load()
    rate = Rate194()
    month = hourly.iloc[-30 * 24:]
    eu_month = end_use.iloc[-30 * 24:]

    # Optimized load curve (avg day).
    opt = optimize_month(month, eu_month, rate)
    # Rebuild an optimized hourly series for the curve: shift per-day.
    from wattwise.optimize import optimize_load_shift
    opt_days = []
    for d in pd.unique(month.index.date):
        mask = month.index.date == d
        h, eu = month.loc[mask], eu_month.loc[mask]
        if len(h) < 24:
            continue
        p = np.array([rate.marginal_price(ts) for ts in h.index[:24]])
        r = optimize_load_shift(h, eu, p)
        opt_days.append(r.hourly_optimized)
    opt_hourly = pd.concat(opt_days)
    actual_curve = month.groupby(month.index.hour).mean()
    opt_curve = opt_hourly.groupby(opt_hourly.index.hour).mean()
    price_curve = [rate.marginal_price(
        pd.Timestamp("2026-10-07") + pd.Timedelta(hours=h)) for h in range(24)]

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            "Average day: actual vs optimized load",
            "October bill forecast with 80% band",
            "When you use power (hour x weekday)",
            "Where the money goes: plan -> TOU -> optimized -> solar",
            "Appliance breakdown (kWh, trailing 30 days)",
            "Weather explains the wiggles",
            "What-if scenarios (monthly bill)",
            "Peak vs off-peak: 13.5% of energy, 25.8% of cost",
        ),
        specs=[[{"secondary_y": True}, {}],
               [{"type": "heatmap"}, {"type": "waterfall"}],
               [{"type": "bar"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "domain"}]],
        vertical_spacing=0.09, horizontal_spacing=0.08,
    )

    # 1. Load curve.
    hrs = list(range(24))
    fig.add_trace(go.Scatter(x=hrs, y=actual_curve, name="Actual",
                             line=dict(color=ACCENT, width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hrs, y=opt_curve, name="Optimized",
                             line=dict(color=GREEN, width=2.5, dash="dash")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=hrs, y=price_curve, name="TOU price",
                             line=dict(color=ORANGE, width=1.5),
                             fill="tozeroy", opacity=0.25), row=1, col=1,
                  secondary_y=True)
    fig.update_yaxes(title_text="kW", row=1, col=1)
    fig.update_yaxes(title_text="$/kWh", row=1, col=1, secondary_y=True)

    # 2. Forecast.
    fc = forecast_month_bill(hourly, rate, temp)
    oct_days = pd.date_range("2026-10-01", "2026-10-31", freq="D")
    daily_kwh = hourly.loc["2026-10-01":].resample("D").sum()
    cum_actual, run = [], 0.0
    for d, v in daily_kwh.items():
        run += v * np.mean([rate.marginal_price(d + pd.Timedelta(hours=h))
                            for h in range(24)])
        cum_actual.append(run)
    base = cum_actual[-1] if cum_actual else 0
    fcd = fc["daily_forecast"]
    fx = list(fcd.index)
    fy = (fcd["forecast_kwh"] * 0.24).cumsum() + base  # ~avg TOU price
    flo = (fcd["lo_kwh"] * 0.24).cumsum() + base
    fhi = (fcd["hi_kwh"] * 0.24).cumsum() + base
    ax = list(daily_kwh.index.date)
    fig.add_trace(go.Scatter(x=ax, y=cum_actual, name="Billed so far",
                             line=dict(color=ACCENT, width=2.5)), row=1, col=2)
    fig.add_trace(go.Scatter(x=fx, y=fy, name="Forecast",
                             line=dict(color=GREEN, width=2.5, dash="dash")),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=fx + fx[::-1],
                             y=list(fhi) + list(flo)[::-1],
                             fill="toself", fillcolor="rgba(63,185,80,0.15)",
                             line=dict(width=0), name="80% band",
                             showlegend=False), row=1, col=2)
    fig.update_yaxes(title_text="Cumulative $", row=1, col=2)

    # 3. TOU heatmap (hour x weekday).
    hm = month.copy()
    hm.index = pd.to_datetime(hm.index)
    pivot = hm.groupby([hm.index.weekday, hm.index.hour]).mean().unstack()
    pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig.add_trace(go.Heatmap(z=pivot.values, x=list(range(24)),
                             y=pivot.index, colorscale="YlOrRd",
                             colorbar=dict(title="kW", len=0.6)),
                  row=2, col=1)

    # 4. Waterfall.
    wf = m["waterfall"]
    labels = [w["label"] for w in wf]
    fig.add_trace(go.Waterfall(
        x=labels, y=[w["delta"] for w in wf],
        measure=["absolute"] + ["relative"] * (len(wf) - 1),
        connector=dict(line=dict(color=GRID)),
        increasing=dict(marker=dict(color=RED)),
        decreasing=dict(marker=dict(color=GREEN)),
        totals=dict(marker=dict(color=ACCENT)),
    ), row=2, col=2)
    fig.update_yaxes(title_text="$", row=2, col=2)

    # 5. Appliance breakdown.
    bd = m["appliance_breakdown"]
    names = [b["appliance"].replace("_", " ") for b in bd]
    fig.add_trace(go.Bar(x=[b["kwh"] for b in bd], y=names,
                         orientation="h", marker_color=ACCENT,
                         text=[f'{b["pct"]}%' for b in bd],
                         textposition="outside"), row=3, col=1)
    fig.update_xaxes(title_text="kWh / 30 days", row=3, col=1)

    # 6. Weather scatter.
    sens = temperature_sensitivity(hourly, temp)
    dw = sens["daily"]
    fig.add_trace(go.Scatter(x=dw["cdd"] + dw["hdd"], y=dw["kwh"],
                             mode="markers", name="Days",
                             marker=dict(color=PURPLE, size=6, opacity=0.7)),
                  row=3, col=2)
    xx = np.linspace(0, (dw["cdd"] + dw["hdd"]).max(), 50)
    fig.add_trace(go.Scatter(
        x=xx, y=sens["base_kwh_per_day"] + sens["cooling_kwh_per_cdd"] * xx,
        mode="lines", name=f'Fit (R2={sens["r_squared"]:.2f})',
        line=dict(color=ORANGE, dash="dash")), row=3, col=2)
    fig.update_xaxes(title_text="Degree-days", row=3, col=2)
    fig.update_yaxes(title_text="kWh / day", row=3, col=2)

    # 7. What-if bars.
    sol = solar_scenario(month, rate)
    ev = ev_scenario(month, rate)
    hp = heat_pump_scenario(month, temp.iloc[-30 * 24:], rate)
    cats = ["Today<br>(TOU)", "Optimized", "Solar<br>6 kW", "EV<br>30 mi/day",
            "Heat pump"]
    vals = [m["bill_tou_30d"], m["bill_tou_30d"] - m["opt_savings"],
            sol["bill_after"], ev["bill_after"], hp["bill_after"]]
    colors = [ACCENT, GREEN, GREEN, ORANGE, ORANGE]
    fig.add_trace(go.Bar(x=cats, y=vals, marker_color=colors,
                         text=[f"${v:.0f}" for v in vals],
                         textposition="outside"), row=4, col=1)
    fig.update_yaxes(title_text="$ / month", row=4, col=1)

    # 8. Peak donut.
    ps = peak_share(month, rate)
    fig.add_trace(go.Pie(
        labels=["Off-peak kWh", "Peak kWh"],
        values=[100 - ps["peak_kwh_pct"], ps["peak_kwh_pct"]],
        hole=0.55, marker_colors=[PANEL, ORANGE],
        textfont=dict(color=TEXT)), row=4, col=2)

    _layout(fig, "", height=1500)
    existing = list(fig.layout.annotations or [])
    fig.update_layout(
        title=dict(
            text="wattwise â€” home energy analytics",
            font=dict(size=22, color=TEXT), x=0.02, xanchor="left"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right",
                    x=1, font=dict(color=TEXT)),
        annotations=existing + [
            dict(text=(
                f'<b>${m["bill_flat_30d"]:.0f}</b> flat bill &nbsp;|&nbsp; '
                f'<b>${m["bill_tou_30d"]:.0f}</b> on time-of-day &nbsp;|&nbsp; '
                f'<b>${m["opt_savings"]:.2f}/mo</b> optimizer savings &nbsp;|&nbsp; '
                f'<b>${m["forecast_bill"]:.0f}</b> Oct forecast '
                f'(${m["forecast_bill_lo"]:.0f}-${m["forecast_bill_hi"]:.0f}) &nbsp;|&nbsp; '
                f'Weather explains <b>{m["weather_r2"]:.0%}</b> of daily variance'
            ), x=0.02, y=1.055, xref="paper", yref="paper",
                showarrow=False, font=dict(size=12, color=TEXT), align="left"),
            dict(text=(f'<b>Anomaly radar:</b> {m["anomaly_text"]}'),
                 x=0.02, y=-0.015, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color=ORANGE),
                 align="left"),
        ],
    )

    os.makedirs(DOCS, exist_ok=True)
    out = os.path.join(DOCS, "dashboard.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(_portfolio_html(fig.to_html(include_plotlyjs=True, full_html=True, config={"displaylogo": False})))
    print(f"dashboard -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")



def _portfolio_html(raw):
    """Presentation-only transform of this runâ€™s freshly calculated figures."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "portfolio_ui"))
    from renderer import render_html
    return render_html(root, 'wattwise', raw)[0]

def main():
    """Build the real-data decision dashboard; retain the original demo fallback."""
    from pathlib import Path
    from wattwise.decision_analysis import build_report
    root = Path(HERE).parent
    if not (root / 'demo/pseg_interval_15min.csv').exists():
        return legacy_main()
    import argparse
    from plotly.offline import get_plotlyjs
    parser = argparse.ArgumentParser(description='Build the self-contained WattWise research dashboard')
    parser.add_argument('--out', default=str(root / 'docs/dashboard.html'))
    parser.add_argument('--reuse-analysis', action='store_true', help='Use an existing analysis.json after UI-only changes')
    args = parser.parse_args()
    report_path = root / 'demo/analysis.json'
    if args.reuse_analysis and report_path.exists():
        report = json.loads(report_path.read_text(encoding='utf-8'))
    else:
        report = build_report(root)
        report_path.write_text(json.dumps(report, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    ui = root / 'dashboard_ui'
    fonts = json.loads((root / 'portfolio_ui/fonts.json').read_text(encoding='utf-8'))
    font_css = '\n'.join(fonts.get(f, '') for f in ['Sora', 'IBM Plex Mono'])
    parts = {'__FONTS__': font_css, '__CSS__': (ui/'style.css').read_text(encoding='utf-8'),
             '__PLOTLY__': get_plotlyjs(), '__DATA__': json.dumps(report, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c'),
             '__JS__': (ui/'decisions.js').read_text(encoding='utf-8')+'\n'+(ui/'app.js').read_text(encoding='utf-8')}
    html = (ui/'index.html').read_text(encoding='utf-8')
    for marker, value in parts.items():
        html = html.replace(marker, value)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding='utf-8')
    print(f'Wrote {output} ({output.stat().st_size:,} bytes), with {len(report["daily"])} observed days')


if __name__ == "__main__":
    main()
