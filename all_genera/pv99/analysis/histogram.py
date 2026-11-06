#!/usr/bin/env python3
"""
Histogram of each photo's top mask confidence (conf_max) over every pv99 photo.

    python all_genera/pv99/local/histogram.py 

SAM3 ran at a 0.5 detection threshold, so photos with no mask have no top
confidence; they are counted in the subtitle and in the CSV, not plotted.
Writes export/pv99/analysis/top_confidence_histogram.png and ..._bins.csv.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_args(parser)
    args = parser.parse_args()

    project = common.project_dir(args.project)
    out = common.analysis_dir(project)
    meta = common.load_metadata(project, ["photo_id", "num_masks", "conf_max", "status"])
    total = len(meta)
    top = meta.loc[meta["num_masks"].fillna(0) > 0, "conf_max"].astype(float).to_numpy()
    none = total - len(top)

    edges = np.round(np.arange(0.50, 1.0001, 0.01), 2)
    counts, _ = np.histogram(top, bins=edges)
    pd.DataFrame({"bin_low": edges[:-1], "bin_high": edges[1:], "photos": counts}).to_csv(
        out / "top_confidence_histogram_bins.csv", index=False)

    q10, q50, q90 = np.quantile(top, [0.1, 0.5, 0.9])
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
    fig.patch.set_facecolor(common.SURFACE)
    common.style(ax)
    ax.bar(edges[:-1], counts, width=0.01, align="edge", color=common.BLUE,
           edgecolor=common.SURFACE, linewidth=1.0)
    ax.axvline(q50, color=common.INK, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(q50, counts.max() * 1.02, f"median {q50:.3f}", ha="right", va="bottom",
            fontsize=9, color=common.INK, rotation=0)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0, counts.max() * 1.10)
    ax.set_xlabel("Top mask confidence per photo (SAM3 score of its best 'flower' mask)",
                  color=common.INK, fontsize=10)
    ax.set_ylabel("Photos per 0.01 bin", color=common.INK, fontsize=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.suptitle(f"Top mask confidence, all {total:,} pv99 photos", x=0.06, ha="left",
                 fontsize=14, color=common.INK, fontweight="bold")
    ax.set_title(f"{len(top):,} photos with a mask ({100 * len(top) / total:.1f}%) plotted; "
                 f"{none:,} ({100 * none / total:.1f}%) had no mask at or above 0.5.   "
                 f"10th / 50th / 90th percentile {q10:.3f} / {q50:.3f} / {q90:.3f}",
                 loc="left", fontsize=9, color=common.INK_2)
    fig.tight_layout()
    fig.savefig(out / "top_confidence_histogram.png", facecolor=common.SURFACE)
    print(f"photos     : {total:,} ({len(top):,} with a mask, {none:,} without)")
    print(f"percentiles: p10 {q10:.4f}  p50 {q50:.4f}  p90 {q90:.4f}")
    print(f"           : {out / 'top_confidence_histogram.png'}")


if __name__ == "__main__":
    main()
