"""Render docs/demo.gif: an animated tour of the wattwise dashboard.

Four scenes, matplotlib, dark theme to match the dashboard:
  1. Average-day load curve with the TOU price band rising underneath.
  2. Bill forecast: cumulative October line extending with its band.
  3. Savings waterfall building step by step.
  4. TOU heatmap fading in.
Every frame is computed from the demo data by this script; nothing is
hand-drawn.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wattwise import Rate194, forecast_month_bill

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "..", "demo")
DOCS = os.path.join(HERE, "..", "docs")

BG, FG, GRID = "#0e1117", "#e6edf3", "#2d333b"
ACCENT, GREEN, ORANGE = "#58a6ff", "#3fb950", "#f0883e"
FPS = 8


def _load():
    hourly = pd.read_csv(os.path.join(DEMO, "profile.csv"), index_col=0,
                         parse_dates=True).iloc[:, 0]
    hourly.index = pd.to_datetime(hourly.index)
    temp = pd.read_csv(os.path.join(DEMO, "weather.csv"), index_col=0,
                       parse_dates=True).iloc[:, 0]
    temp.index = pd.to_datetime(temp.index)
    m = json.load(open(os.path.join(DEMO, "metrics.json")))
    return hourly, temp, m


def main():
    hourly, temp, m = _load()
    rate = Rate194()
    month = hourly.iloc[-30 * 24:]
    curve = month.groupby(month.index.hour).mean().to_numpy()
    prices = np.array([rate.marginal_price(
        pd.Timestamp("2026-10-07") + pd.Timedelta(hours=h)) for h in range(24)])

    fc = forecast_month_bill(hourly, rate, temp)
    daily = hourly.loc["2026-10-01":].resample("D").sum()
    cum, run = [], 0.0
    for d, v in daily.items():
        run += v * np.mean([rate.marginal_price(d + pd.Timedelta(hours=h))
                            for h in range(24)])
        cum.append(run)
    fcd = fc["daily_forecast"]
    fx = np.arange(len(cum), len(cum) + len(fcd))
    fy = (fcd["forecast_kwh"] * 0.24).cumsum().to_numpy() + cum[-1]
    fband = 1.2816 * 0.24 * 2.0 * np.sqrt(np.arange(1, len(fcd) + 1))

    wf = m["waterfall"]
    wlabels = [w["label"].split(" (")[0] for w in wf]
    wvals = [w["delta"] for w in wf]

    hm = month.copy()
    piv = hm.groupby([hm.index.weekday, hm.index.hour]).mean().unstack().values

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), facecolor=BG)
    for ax in axes.flat:
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=8)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.grid(color=GRID, alpha=0.4, linewidth=0.5)
        ax.title.set_color(FG)
        ax.title.set_fontsize(10)
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
    fig.tight_layout(pad=2.5, rect=[0, 0.04, 1, 0.94])
    fig.suptitle("wattwise — home energy analytics   |   "
                 f"flat ${m['bill_flat_30d']:.0f}  →  "
                 f"optimized ${m['bill_tou_30d'] - m['opt_savings']:.0f}/mo",
                 color=FG, fontsize=13, y=0.98)

    ax1, ax2, ax3, ax4 = axes.flat
    ax1.set_title("Average day: load vs TOU price")
    ax1.set_xlim(0, 23); ax1.set_ylim(0, curve.max() * 1.35)
    ax1.set_xlabel("hour"); ax1.set_ylabel("kW")
    (ln1,) = ax1.plot([], [], color=ACCENT, lw=2.5)
    ax1b = ax1.twinx()
    ax1b.set_ylim(0, prices.max() * 2.2)
    ax1b.tick_params(colors=FG, labelsize=8)
    (ln1p,) = ax1b.plot([], [], color=ORANGE, lw=1.5, alpha=0.8)

    ax2.set_title("October bill forecast")
    ax2.set_xlim(0, 30); ax2.set_ylim(0, max(cum[-1], fy[-1]) * 1.15)
    ax2.set_xlabel("day of October"); ax2.set_ylabel("cumulative $")
    (ln2,) = ax2.plot([], [], color=ACCENT, lw=2.5)
    (ln2f,) = ax2.plot([], [], color=GREEN, lw=2.5, ls="--")
    band = ax2.fill_between([], [], [], color=GREEN, alpha=0.15)

    ax3.set_title("Savings waterfall")
    ax3.set_ylim(min(0, sum(wvals)) * 1.1 - 5, wvals[0] + 5)
    ax3.set_ylabel("$ / month")
    ax3.set_xticks(range(len(wlabels)))
    ax3.set_xticklabels(wlabels, rotation=18, ha="right", fontsize=8,
                        color=FG)

    ax4.set_title("When power gets used (hour x weekday)")
    ax4.set_xlabel("hour"); ax4.set_ylabel("weekday")
    ax4.set_yticks(range(7))
    ax4.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                        fontsize=8, color=FG)
    im = ax4.imshow(np.zeros_like(piv), aspect="auto", cmap="YlOrRd",
                    vmin=0, vmax=piv.max(), origin="lower",
                    extent=[-0.5, 23.5, -0.5, 6.5])

    frames = []
    N1, N2, N3, N4 = 24, 21, len(wvals), 12

    def draw(i):
        nonlocal band
        if i < N1:  # scene 1: load curve draws left to right
            n = i + 1
            ln1.set_data(range(n), curve[:n])
            ln1p.set_data(range(n), prices[:n])
        elif i < N1 + N2:  # scene 2: forecast extends
            j = i - N1 + 1
            ln2.set_data(range(len(cum)), cum)
            ln2f.set_data(fx[:j], fy[:j])
            band.remove()
            band = ax2.fill_between(fx[:j], fy[:j] - fband[:j],
                                    fy[:j] + fband[:j], color=GREEN, alpha=0.15)
        elif i < N1 + N2 + N3:  # scene 3: waterfall builds
            j = i - N1 - N2
            run_ = wvals[0]
            xs = [0]
            heights = [wvals[0]]
            bottoms = [0]
            for k in range(1, j + 1):
                xs.append(k)
                heights.append(abs(wvals[k]))
                bottoms.append(run_ + min(wvals[k], 0))
                run_ += wvals[k]
            ax3.clear()
            ax3.set_facecolor(BG)
            ax3.tick_params(colors=FG, labelsize=8)
            for s in ax3.spines.values():
                s.set_color(GRID)
            ax3.grid(color=GRID, alpha=0.4, linewidth=0.5)
            ax3.set_title("Savings waterfall", color=FG, fontsize=10)
            ax3.set_ylabel("$ / month", color=FG)
            ax3.set_ylim(min(0, sum(wvals)) * 1.1 - 5, wvals[0] + 5)
            ax3.set_xticks(range(len(wlabels)))
            ax3.set_xticklabels(wlabels, rotation=18, ha="right", fontsize=8,
                                color=FG)
            cols = [ACCENT if k == 0 else (GREEN if wvals[k] < 0 else ORANGE)
                    for k in range(len(xs))]
            ax3.bar(xs, heights, bottom=bottoms, color=cols, width=0.6)
        else:  # scene 4: heatmap fades in
            a = (i - N1 - N2 - N3 + 1) / N4
            im.set_data(piv * a + piv.max() * 0.02 * (1 - a))
            im.set_alpha(min(a * 1.5, 1.0))
        return []

    total = N1 + N2 + N3 + N4
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, draw, frames=total, blit=False)
    os.makedirs(DOCS, exist_ok=True)
    out = os.path.join(DOCS, "demo.gif")
    anim.save(out, writer="pillow", fps=FPS)
    plt.close(fig)
    print(f"gif -> {out} ({os.path.getsize(out)/1e6:.2f} MB, {total} frames)")


if __name__ == "__main__":
    main()
