#!/usr/bin/env python3
"""
Review figures (run 2's layout: photo with outlined masks, top-5 cut-outs on
the right) for 30 photos per family, where a family has that many.

    python all_genera/pv99/local/figures.py  [--per-family 30]

The photos are pack_select.figure_pool(): shuffled once, spread across
the family's genera (one per genus before any genus gets a second), photos
with at least one mask only. The pool holds 10 spares per family, used in
order if a photo cannot be drawn. Writes

    export/pv99/analysis/figures/<Family>/<Genus>_<photo_id>.jpg
    export/pv99/analysis/figures/figures_index.csv
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import review_figure  # noqa: E402  (repo root, via common)


def draw(job):
    project, out_dir, row, k, n = job
    out = Path(out_dir) / common.safe(row["family"]) / \
        f"{common.safe(row['genus'])}_{int(row['photo_id'])}.jpg"
    try:
        if not out.exists():
            image = Image.open(common.local_image(project, row["image_path"])).convert("RGB")
            masks, scores = common.photo_masks(project, row["mask_path"])
            if masks.shape[1:] != (image.height, image.width):
                masks = [m for m in (Image.fromarray(x).resize(image.size) for x in masks)]
                import numpy as np
                masks = np.stack([np.asarray(m, dtype=bool) for m in masks])
            top = list(masks[:review_figure.TOP_K])
            title = (f"{row.get('scientific_name') or row['genus']}   photo "
                     f"{int(row['photo_id'])}   top-5 conf {row['conf_top5_mean']:.3f}"
                     f"   {k} of {n} in {row['family']}")
            fig = review_figure.make_figure(image, top, list(scores[:len(top)]), title,
                                            len(masks))
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".part")
            fig.save(tmp, format="JPEG", quality=88)
            tmp.replace(out)
        return row["photo_id"], str(out), ""
    except Exception as exc:
        return row["photo_id"], "", f"{type(exc).__name__}: {str(exc)[:200]}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_args(parser)
    parser.add_argument("--per-family", type=int, default=30)
    parser.add_argument("--workers", type=int, default=common.n_workers())
    args = parser.parse_args()

    project = common.project_dir(args.project)
    out_dir = common.analysis_dir(project, "figures")
    meta = common.load_metadata(project)
    rows = common.figure_selection(meta).sort_values(["family", "rank"])

    done, failed = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for family, g in rows.groupby("family", sort=True):
            records = g.to_dict("records")
            n = min(args.per_family, len(records))
            kept = []
            # First n, then spares one at a time until n are drawn or none remain.
            queue = records[:]
            while len(kept) < n and queue:
                batch, queue = queue[:n - len(kept)], queue[n - len(kept):]
                jobs = [(project, out_dir, r, len(kept) + i + 1, n) for i, r in enumerate(batch)]
                for r, (pid, path, err) in zip(batch, pool.map(draw, jobs)):
                    if err:
                        failed.append({"family": family, "photo_id": pid, "error": err})
                    else:
                        kept.append({"family": family, "genus": r["genus"],
                                     "photo_id": pid, "figure": path})
            done += kept
            print(f"{family:<24} {len(kept):>3}/{n}", flush=True)

    index = pd.DataFrame(done)
    index.to_csv(out_dir / "figures_index.csv", index=False)
    if failed:
        pd.DataFrame(failed).to_csv(out_dir / "figures_failed.csv", index=False)
    print(f"figures    : {len(index):,} in {index['family'].nunique()} families -> {out_dir}")
    print(f"failed     : {len(failed)}")


if __name__ == "__main__":
    main()
