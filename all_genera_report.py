#!/usr/bin/env python3
"""
After the all-genera run: one metadata CSV for every photo, and review figures
for three photos per genus - the lowest, the median and the highest confidence.

    python all_genera_report.py results/all_genera_flower \\
        --index data/all_genera_index.parquet --export export/all_genera

A photo's confidence here is the mean of its five highest SAM3 mask scores (all
of them, if it has fewer than five): `conf_top5_mean`. The annotation's
prediction_probability is carried along as a column and never used to rank.

Writes into --export (the folder to download):

    all_genera_segment_metadata.csv   one row per photo in the run: genus,
                                      family, prediction_probability, num_masks,
                                      mask_scores, conf_top5_mean, which of
                                      low/median/high it was picked as, and the
                                      paths of its image, masks and figure
    all_genera_genus_summary.csv      one row per genus
    figures/<Family>/<Genus>_<low|median|high>_<photo_id>.jpg

A genus with two detected photos gets low and high; with one, just high. A
figure that already exists is not redrawn, so the job can be resubmitted.
"""

import argparse
import glob
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import review_figure
from maskio import load_photo_masks

TOP = 5
METADATA = "all_genera_segment_metadata.csv"
SUMMARY = "all_genera_genus_summary.csv"
COLUMNS = ["photo_id", "genus", "family", "scientific_name", "observation_uuid",
           "prediction_probability", "certainty", "status", "num_masks", "mask_scores",
           "conf_top5_mean", "conf_max", "selected", "genus_rank", "genus_detected",
           "review_figure", "image_path", "mask_path", "mask_bytes", "width", "height",
           "seconds", "infer_s", "save_s", "row_index", "batch", "error"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="The all-genera run's output directory")
    parser.add_argument("--index", required=True, help="all_genera_index.parquet")
    parser.add_argument("--export", required=True, help="Folder to fill for download")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--quality", type=int, default=88, help="JPEG quality")
    parser.add_argument("--no-figures", action="store_true", help="Only write the CSVs")
    args = parser.parse_args()

    run = Path(args.run_dir).expanduser().resolve()
    export = Path(args.export).expanduser().resolve()
    export.mkdir(parents=True, exist_ok=True)

    meta = build_metadata(run, args.index)
    meta = select(meta)
    picked = meta[meta["selected"] != ""]
    print(f"metadata   : {len(meta):,} photos, {meta['genus'].nunique():,} genera; "
          f"{int((meta['num_masks'] > 0).sum()):,} with a mask; "
          f"{len(picked):,} picked for figures")

    meta[COLUMNS].to_csv(export / METADATA, index=False)
    shutil.copy2(export / METADATA, run / METADATA)
    genus_summary(meta).to_csv(export / SUMMARY, index=False)
    print(f"           : {export / METADATA}")

    if args.no_figures:
        return
    jobs = [(run, export, r, args.quality) for r in picked.to_dict("records")
            if not (export / r["review_figure"]).exists()]
    print(f"figures    : {len(picked) - len(jobs):,} already drawn, {len(jobs):,} to go")
    failed = 0
    if jobs:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for i, error in enumerate(pool.map(_draw_star, jobs, chunksize=8), 1):
                if error:
                    failed += 1
                    print(f"  failed: {error}")
                if i % 2000 == 0:
                    print(f"           : {i:,}/{len(jobs):,}", flush=True)
    print(f"figures    : {len(jobs) - failed:,} drawn -> {export / 'figures'}"
          + (f", {failed:,} failed" if failed else ""))


def build_metadata(run, index_path):
    frames = []
    for f in sorted(glob.glob(str(run / "results_shard_*.csv"))):
        try:
            frames.append(pd.read_csv(f, low_memory=False))
        except pd.errors.EmptyDataError:
            continue
    if not frames:
        raise SystemExit(f"no results_shard_*.csv in {run}")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["photo_id", "row_index", "status"])
    df = df.drop_duplicates(subset=["photo_id"], keep="last")
    df["photo_id"] = df["photo_id"].astype("int64")
    df["num_masks"] = pd.to_numeric(df["num_masks"], errors="coerce").fillna(0).astype(int)

    index = pd.read_parquet(index_path, columns=[
        "photo_id", "genus", "family", "scientific_name", "observation_uuid",
        "prediction_probability", "certainty"])
    df = df.merge(index, on="photo_id", how="left")
    df["genus"] = df["genus"].fillna(df["taxon_name"].astype(str).str.split().str[0])

    tops = [_top_scores(s) for s in df["mask_scores"]]
    df["conf_top5_mean"] = [np.mean(t) if t else np.nan for t in tops]
    df["conf_max"] = [t[0] if t else np.nan for t in tops]
    df["mask_path"] = df["mask_paths"]
    if "mask_bytes" not in df:
        df["mask_bytes"] = np.nan
    return df.sort_values(["genus", "photo_id"]).reset_index(drop=True)


def _top_scores(cell):
    """The (up to) five highest mask scores in a mask_scores cell, descending."""
    if not isinstance(cell, str) or not cell:
        return []
    scores = sorted((float(s) for s in cell.split(",") if s), reverse=True)
    return scores[:TOP]


def select(meta):
    """Mark the lowest / median / highest conf_top5_mean photo of each genus."""
    meta["selected"] = ""
    meta["genus_rank"] = np.nan
    meta["genus_detected"] = 0
    meta["review_figure"] = ""
    detected = meta[meta["conf_top5_mean"].notna()]
    for genus, group in detected.groupby("genus", sort=False):
        order = group.sort_values(["conf_top5_mean", "photo_id"]).index
        n = len(order)
        meta.loc[order, "genus_rank"] = np.arange(1, n + 1)
        meta.loc[order, "genus_detected"] = n
        picks = {"high": order[-1]}
        if n >= 2:
            picks["low"] = order[0]
        if n >= 3:
            picks["median"] = order[(n - 1) // 2]
        for label, idx in picks.items():
            meta.loc[idx, "selected"] = label
            family = _safe(meta.loc[idx, "family"]) or "Unknown_family"
            meta.loc[idx, "review_figure"] = (
                f"figures/{family}/{_safe(genus)}_{label}_{int(meta.loc[idx, 'photo_id'])}.jpg")
    return meta


def _safe(value):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(value or "")
                   ) if isinstance(value, str) else ""


def genus_summary(meta):
    detected = meta[meta["conf_top5_mean"].notna()]
    by = meta.groupby("genus")
    conf = detected.groupby("genus")["conf_top5_mean"]
    out = pd.DataFrame({
        "family": by["family"].first(),
        "photos": by.size(),
        "photos_with_mask": by["num_masks"].apply(lambda s: int((s > 0).sum())),
        "masks": by["num_masks"].sum(),
        "conf_top5_mean_low": conf.min(),
        "conf_top5_mean_median": conf.median(),
        "conf_top5_mean_high": conf.max(),
        "prediction_probability_mean": by["prediction_probability"].mean(),
    })
    out["detection_rate_pct"] = 100 * out["photos_with_mask"] / out["photos"]
    return out.rename_axis("genus").reset_index()


def _draw_star(job):
    return draw(*job)


def draw(run, export, row, quality):
    try:
        image_path = Path(str(row["image_path"]))
        image = Image.open(image_path if image_path.is_absolute() else run / image_path)
        image = image.convert("RGB")
        all_masks = load_photo_masks(run, row["mask_paths"])
        scores = [float(s) for s in str(row["mask_scores"]).split(",")]
        order = sorted(range(len(all_masks)), key=lambda i: scores[i],
                       reverse=True)[:review_figure.TOP_K]
        where = {"low": "lowest", "median": "median", "high": "highest"}[row["selected"]]
        title = (f"{row.get('scientific_name') or row['genus']}   photo {int(row['photo_id'])}"
                 f"   top-5 conf {row['conf_top5_mean']:.3f}"
                 f"   {where} of {int(row['genus_detected'])} in genus")
        figure = review_figure.make_figure(image, [all_masks[i] for i in order],
                                           [scores[i] for i in order], title, len(all_masks))
        out = export / row["review_figure"]
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".part")
        figure.save(tmp, format="JPEG", quality=quality)
        tmp.replace(out)
        return None
    except Exception as exc:
        return f"photo_id={row.get('photo_id')}: {str(exc)[:300]}"


if __name__ == "__main__":
    main()
