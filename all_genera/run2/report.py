#!/usr/bin/env python3
"""
After run 2's segmentation: run 1's metadata CSV, genus summary and review
figures, plus the per-tier comparison with phenovision.

    python all_genera/run2/report.py results/all_genera_run2_flower \\
        --index data/all_genera_run2_index.parquet --export export/all_genera_run2

Each photo was picked as one of its genus's 10 lowest, 10 median or 10 highest
phenovision prediction probabilities (pv_tier, see build_index.py). SAM3 ran at
--min-score 0.5 and kept each photo's 5 most confident masks, so
conf_top5_mean is the mean of every mask the photo has.

Writes into --export, next to what all_genera_report.py writes (the metadata
CSV, now with pv_tier and pv_rank, the genus summary, figures/<Family>/...):

    tier_genus_summary.csv     one row per genus x tier: photos, photos with a
                               mask, mean prediction probability and mean
                               conf_top5_mean, each with its 95% CI
    tier_family_summary.csv    the same per family x tier
    scatter_tier_<low|median|high>.png
                               one point per genus: mean prediction
                               probability of its 10 photos in that tier (x)
                               against the mean conf_top5_mean of those with a
                               mask (y). Bars are each genus's 95% CIs; the
                               line is least squares with its 95% CI band.
    family_histograms/<Family>.png
                               every kept mask score (up to 5 per photo) of
                               the family's photos, stacked by tier
"""

import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import all_genera_report as run1  # noqa: E402

TIERS = ("low", "median", "high")
TIER_COLOR = {"low": "#d1495b", "median": "#edae49", "high": "#2a78d6"}
TIER_LABEL = {"low": "10 lowest", "median": "10 median", "high": "10 highest"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
COLUMNS = run1.COLUMNS[:7] + ["pv_tier", "pv_rank"] + run1.COLUMNS[7:]
MIN_SCORE = 0.5

# Two-sided 95% t quantiles for small df; larger df use the series in t975().
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="Run 2's output directory")
    parser.add_argument("--index", required=True, help="all_genera_run2_index.parquet")
    parser.add_argument("--export", required=True, help="Folder to fill for download")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--quality", type=int, default=88, help="JPEG quality")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip the per-photo review figures")
    args = parser.parse_args()

    run = Path(args.run_dir).expanduser().resolve()
    export = Path(args.export).expanduser().resolve()
    export.mkdir(parents=True, exist_ok=True)

    meta = run1.build_metadata(run, args.index)
    tiers = pd.read_parquet(args.index, columns=["photo_id", "pv_tier", "pv_rank"])
    meta = meta.merge(tiers, on="photo_id", how="left")
    meta = run1.select(meta)
    print(f"metadata   : {len(meta):,} photos, {meta['genus'].nunique():,} genera; "
          f"{int((meta['num_masks'] > 0).sum()):,} with a mask")
    meta[COLUMNS].to_csv(export / run1.METADATA, index=False)
    shutil.copy2(export / run1.METADATA, run / run1.METADATA)
    run1.genus_summary(meta).to_csv(export / run1.SUMMARY, index=False)

    genus = tier_summary(meta, "genus")
    genus.to_csv(export / "tier_genus_summary.csv", index=False)
    tier_summary(meta, "family").to_csv(export / "tier_family_summary.csv", index=False)
    for tier in TIERS:
        scatter(genus[genus["pv_tier"] == tier], tier, export / f"scatter_tier_{tier}.png")

    hist_dir = export / "family_histograms"
    hist_dir.mkdir(exist_ok=True)
    scores = mask_scores(meta)
    jobs = [(family, group, hist_dir) for family, group in scores.groupby("family")]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(_histogram_star, jobs, chunksize=4))
    print(f"histograms : {len(jobs):,} families -> {hist_dir}")

    if args.no_figures:
        return
    picked = meta[meta["selected"] != ""]
    jobs = [(run, export, r, args.quality) for r in picked.to_dict("records")
            if not (export / r["review_figure"]).exists()]
    print(f"figures    : {len(picked) - len(jobs):,} already drawn, {len(jobs):,} to go")
    failed = 0
    if jobs:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for error in pool.map(run1._draw_star, jobs, chunksize=8):
                if error:
                    failed += 1
                    print(f"  failed: {error}")
    print(f"figures    : {len(jobs) - failed:,} drawn" + (f", {failed:,} failed" if failed else ""))


def t975(df):
    """Two-sided 95% Student t quantile."""
    if df < 1:
        return np.nan
    if df in _T975:
        return _T975[df]
    z = 1.959964
    return (z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
            + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3))


def mean_ci(values):
    """(mean, half-width of the 95% t interval, n). The half-width is NaN below n=2."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0:
        return np.nan, np.nan, 0
    if n == 1:
        return float(v[0]), np.nan, 1
    return float(v.mean()), float(t975(n - 1) * v.std(ddof=1) / np.sqrt(n)), n


def tier_summary(meta, level):
    rows = []
    for (name, tier), group in meta.groupby([level, "pv_tier"], sort=True):
        pp, pp_ci, _ = mean_ci(group["prediction_probability"])
        conf, conf_ci, with_mask = mean_ci(group["conf_top5_mean"])
        row = {level: name, "pv_tier": tier}
        if level == "genus":
            row["family"] = group["family"].iloc[0]
        row.update(photos=len(group), photos_with_mask=with_mask,
                   prediction_probability_mean=pp, prediction_probability_ci95=pp_ci,
                   conf_top5_mean_mean=conf, conf_top5_mean_ci95=conf_ci,
                   conf_max_mean=group["conf_max"].mean(),
                   masks=int(group["num_masks"].sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def ols_band(x, y, grid):
    """Least-squares line through (x, y) and the 95% CI of its mean on grid."""
    n = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    s = np.sqrt((resid ** 2).sum() / (n - 2))
    sxx = ((x - x.mean()) ** 2).sum()
    fit = slope * grid + intercept
    half = t975(n - 2) * s * np.sqrt(1 / n + (grid - x.mean()) ** 2 / sxx)
    r = np.corrcoef(x, y)[0, 1] if x.std() and y.std() else np.nan
    return slope, intercept, r, fit, half


def scatter(genus, tier, out):
    total = len(genus)
    genus = genus.dropna(subset=["prediction_probability_mean", "conf_top5_mean_mean"])
    x = genus["prediction_probability_mean"].to_numpy(float)
    y = genus["conf_top5_mean_mean"].to_numpy(float)
    color = TIER_COLOR[tier]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    alpha = float(np.clip(600 / max(len(x), 1), 0.05, 0.5))
    # Both quantities live in [0, 1]; a genus with two photos can have a t
    # interval far wider than that, so the bars are cut at the bounds.
    xerr = _clipped(x, genus["prediction_probability_ci95"], 0.0, 1.0)
    yerr = _clipped(y, genus["conf_top5_mean_ci95"], MIN_SCORE, 1.0)
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none",
                ecolor=color, elinewidth=0.4, alpha=alpha * 0.6, rasterized=True)
    ax.scatter(x, y, s=7, color=color, alpha=min(1.0, alpha * 1.6), linewidths=0,
               rasterized=True)
    if len(x) >= 3:
        grid = np.linspace(x.min(), x.max(), 200)
        slope, intercept, r, fit, half = ols_band(x, y, grid)
        ax.fill_between(grid, fit - half, fit + half, color=INK, alpha=0.15, linewidth=0)
        ax.plot(grid, fit, color=INK, linewidth=1.8)
        ax.annotate(f"y = {slope:.3f}x + {intercept:.3f}\nr = {r:.3f},  r² = {r * r:.3f}",
                    xy=(0.98, 0.04), xycoords="axes fraction", ha="right", va="bottom",
                    color=INK, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=GRID))
    if len(x):
        ax.set_xlim(max(0.0, x.min() - 0.03), 1.005)
        ax.set_ylim(max(MIN_SCORE, y.min() - 0.03), 1.005)
    ax.set_xlabel(f"Mean prediction probability of the genus's {TIER_LABEL[tier]} "
                  "phenovision photos", color=INK_SECONDARY)
    ax.set_ylabel("Mean segment confidence  (per photo: mean of its top 5 SAM3 masks)",
                  color=INK_SECONDARY)
    ax.set_title(f"Phenovision vs. SAM3 per genus - {TIER_LABEL[tier]} photos",
                 loc="left", color=INK, fontsize=13, fontweight="bold", pad=34)
    ax.text(0, 1.02, f"{len(x):,} genera, one point each; {total - len(x):,} with no "
            f"mask >= {MIN_SCORE} in this tier not shown.\nBars: each genus's 95% CI. "
            "Line: least squares, shaded 95% CI.",
            transform=ax.transAxes, color=INK_SECONDARY, fontsize=8.5)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"scatter    : {tier:6s} n={len(x):,} -> {out}")


def _clipped(centre, half, low, high):
    """Symmetric half-widths -> a 2 x N asymmetric xerr/yerr kept inside [low, high]."""
    half = half.fillna(0).to_numpy(float)
    return np.vstack([np.minimum(half, centre - low), np.minimum(half, high - centre)]) \
        .clip(min=0)


def mask_scores(meta):
    """One row per kept mask: family, pv_tier, score."""
    m = meta[meta["num_masks"] > 0][["family", "pv_tier", "mask_scores"]].copy()
    m["family"] = m["family"].fillna("Unknown_family")
    m["score"] = m["mask_scores"].astype(str).str.split(",")
    m = m.explode("score")
    m["score"] = pd.to_numeric(m["score"], errors="coerce")
    return m.dropna(subset=["score"])[["family", "pv_tier", "score"]]


def _histogram_star(job):
    return histogram(*job)


def histogram(family, scores, out_dir):
    # Scores come out of a bf16 forward pass in steps of 1/256 here; bins two
    # steps wide with edges between the steps, so no bin gets an extra one.
    bins = np.arange(MIN_SCORE, 1.0 + 1e-9, 2 / 256) - 1 / 512
    stacks = [scores.loc[scores["pv_tier"] == t, "score"].to_numpy() for t in TIERS]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.hist(stacks, bins=bins, stacked=True, color=[TIER_COLOR[t] for t in TIERS],
            label=[f"{TIER_LABEL[t]} phenovision ({len(s):,})" for t, s in zip(TIERS, stacks)],
            edgecolor=SURFACE, linewidth=0.3)
    ax.set_xlim(MIN_SCORE, 1.0)
    ax.set_xlabel("SAM3 mask confidence (top 5 masks per photo)", color=INK_SECONDARY)
    ax.set_ylabel("Masks", color=INK_SECONDARY)
    ax.set_title(family, loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, title="photo picked from genus's",
              title_fontsize=8, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_dir / f"{run1._safe(family) or 'Unknown_family'}.png", facecolor=SURFACE)
    plt.close(fig)


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, length=0)


if __name__ == "__main__":
    main()
