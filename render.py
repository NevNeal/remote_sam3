#!/usr/bin/env python3
"""
Make the overlays and cut-outs for a run that was segmented with --masks-only.

Everything the GPU was needed for is already on disk: the .npy masks, the
downloaded images, and each photo's scores in the results CSVs. What is left is
PNG encoding, which is CPU work, so it runs here on a CPU partition rather than
on a B200's clock.

    python render.py results/62741_flower --shard 0 --num-shards 20 --workers 16

Writes into the same run directory, with the same names segment.py would have
used, so the finished tree is indistinguishable from a full run:

    <run>/overlays/batch_00001/Genus_species_<photo_id>_overlay.png
    <run>/segments/batch_00001/Genus_species_<photo_id>_segment_0.png
    <run>/render_shard_000.csv      one row per photo rendered

Resumable: a photo whose overlay already exists is skipped. Sharding is by
row_index, as in segment.py, so any number of CPU tasks can split the run.
"""

import argparse
import csv
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from segment import save_cutout, save_overlay

RENDER_COLUMNS = ["photo_id", "status", "overlay_path", "segment_paths",
                  "render_s", "render_bytes", "error"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="A --masks-only run's output directory")
    parser.add_argument("--prompt", default="flower", help="Label drawn on the overlay")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="Processes encoding PNGs")
    args = parser.parse_args()

    run = Path(args.run_dir).expanduser().resolve()
    photos = load_detected(run)
    photos = photos[photos["row_index"] % max(1, args.num_shards) == args.shard]
    todo = [r for r in photos.to_dict("records")
            if not (run / _overlay_rel(r)).exists()]
    print(f"render     : shard {args.shard} of {args.num_shards}: {len(photos):,} photos "
          f"with masks, {len(photos) - len(todo):,} already rendered, {len(todo):,} to go")
    if not todo:
        return

    log = run / f"render_shard_{args.shard:03d}.csv"
    new = not log.exists() or log.stat().st_size == 0
    started = time.perf_counter()
    done = failed = 0
    with open(log, "a", newline="", encoding="utf-8") as handle, \
            ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        writer = csv.DictWriter(handle, fieldnames=RENDER_COLUMNS)
        if new:
            writer.writeheader()
        jobs = ((run, r, args.prompt) for r in todo)
        for result in pool.map(_render_star, jobs, chunksize=8):
            writer.writerow(result)
            done += 1
            failed += result["status"] != "ok"
            if done % 1000 == 0:
                handle.flush()
                rate = done / (time.perf_counter() - started)
                print(f"           : {done:,}/{len(todo):,}  {rate:.1f} photos/s", flush=True)
    elapsed = time.perf_counter() - started
    print(f"render     : {done:,} photos in {elapsed:.0f}s "
          f"({done / elapsed:.1f} photos/s), {failed:,} failed -> {log}")


def load_detected(run):
    """Every photo with at least one mask, last record per photo, as collect.py."""
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
    df["num_masks"] = pd.to_numeric(df["num_masks"], errors="coerce").fillna(0)
    df = df[(df["status"] == "ok") & (df["num_masks"] > 0)].copy()
    df["row_index"] = df["row_index"].astype(int)
    return df


def _overlay_rel(row):
    """overlays/<batch>/<stem>_overlay.png, from the first mask's path."""
    first = Path(str(row["mask_paths"]).split(";")[0])
    stem = first.name.rsplit("_instance_", 1)[0]
    return Path("overlays") / first.parent.name / f"{stem}_overlay.png"


def _render_star(job):
    return render_one(*job)


def render_one(run, row, prompt):
    started = time.perf_counter()
    result = {"photo_id": int(row["photo_id"]), "status": "ok", "overlay_path": "",
              "segment_paths": "", "render_s": "", "render_bytes": 0, "error": ""}
    try:
        image_path = Path(str(row["image_path"]))
        image = Image.open(image_path if image_path.is_absolute() else run / image_path)
        image = image.convert("RGB")
        mask_rels = str(row["mask_paths"]).split(";")
        scores = [float(s) for s in str(row["mask_scores"]).split(",")]
        masks = [np.load(run / m) for m in mask_rels]

        written, segments = [], []
        for i, (rel, mask) in enumerate(zip(mask_rels, masks)):
            stem = Path(rel).name.rsplit("_instance_", 1)[0]
            cutout = run / "segments" / Path(rel).parent.name / f"{stem}_segment_{i}.png"
            if save_cutout(image, mask, cutout):
                segments.append(str(cutout.relative_to(run)))
                written.append(cutout)
        overlay_rel = _overlay_rel(row)
        save_overlay(image, masks, scores, prompt, run / overlay_rel)
        written.append(run / overlay_rel)

        result.update(overlay_path=str(overlay_rel), segment_paths=";".join(segments),
                      render_bytes=sum(p.stat().st_size for p in written))
    except Exception as exc:
        result.update(status="failed", error=str(exc)[:500])
    result["render_s"] = f"{time.perf_counter() - started:.4f}"
    return result


if __name__ == "__main__":
    main()
