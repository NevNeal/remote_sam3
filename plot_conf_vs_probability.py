#!/usr/bin/env python3
"""
Scatter plot: SAM3 segment confidence against the phenovision flower
prediction probability, one point per photo, with a least-squares line.

    python plot_conf_vs_probability.py export/all_genera/all_genera_segment_metadata.csv

    x  prediction_probability   the annotation model's P(flower present)
    y  conf_top5_mean           mean of the photo's five highest SAM3 mask scores

Asks whether the two models agree: does a photo the classifier was surer had a
flower also get more confident segments? Photos with no mask have no y value
and are left out (the count is in the caption). Detections below the run's
MIN_SCORE were never kept, so y has a floor there by construction.

Writes segment_conf_vs_prediction_probability.png next to the CSV (or --out).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metadata", help="all_genera_segment_metadata.csv")
    parser.add_argument("--out", default=None, help="Output PNG")
    args = parser.parse_args()

    path = Path(args.metadata)
    out = Path(args.out) if args.out else path.with_name(
        "segment_conf_vs_prediction_probability.png")
    df = pd.read_csv(path, usecols=["prediction_probability", "conf_top5_mean", "genus"])
    total = len(df)
    df = df.dropna(subset=["prediction_probability", "conf_top5_mean"])
    x = df["prediction_probability"].to_numpy(float)
    y = df["conf_top5_mean"].to_numpy(float)
    if len(x) < 2:
        raise SystemExit("fewer than two photos have both values - nothing to plot")

    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1] if x.std() and y.std() else float("nan")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    # Small, translucent points: with hundreds of thousands of photos the density
    # is the picture. Rasterised so the file stays small.
    alpha = float(np.clip(4000 / len(x), 0.04, 0.6))
    ax.scatter(x, y, s=5, color=SERIES, alpha=alpha, linewidths=0, rasterized=True)

    xs = np.array([x.min(), x.max()])
    ax.plot(xs, slope * xs + intercept, color=INK, linewidth=2)
    ax.annotate(f"y = {slope:.3f}x + {intercept:.3f}\nr = {r:.3f},  r² = {r * r:.3f}",
                xy=(xs[1], slope * xs[1] + intercept), xytext=(-8, 10),
                textcoords="offset points", ha="right", va="bottom",
                color=INK, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=GRID))

    ax.set_xlabel("Prediction probability  (phenovision, flower present)", color=INK_SECONDARY)
    ax.set_ylabel("Segment confidence  (mean of top 5 SAM3 mask scores)", color=INK_SECONDARY)
    ax.set_title("Segment confidence vs. prediction probability", loc="left",
                 color=INK, fontsize=13, fontweight="bold", pad=24)
    ax.text(0, 1.02, f"{len(x):,} photos in {df['genus'].nunique():,} genera, "
            f"one point each; {total - len(x):,} photos without a mask not shown",
            transform=ax.transAxes, color=INK_SECONDARY, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, length=0)
    ax.margins(0.03)

    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print(f"plot       : n={len(x):,}  slope={slope:.4f}  intercept={intercept:.4f}  "
          f"r={r:.4f} -> {out}")


if __name__ == "__main__":
    main()
