#!/usr/bin/env python3
"""
After the pv99 GPU array: one metadata table for every photo, and nothing else.
The masks stay where segment.py wrote them (<run>/masks/batch_NNNNN/*.npz); this
does not copy, pack or render them.

    python all_genera/pv99/export.py results/pv99_flower \\
        --index data/pv99_index.parquet --export export/pv99

Writes into --export:

    pv99_segment_metadata.parquet   one row per photo: photo_id, genus, family,
                                    scientific_name, observation_uuid,
                                    prediction_probability, certainty, status,
                                    num_masks, mask_scores, conf_top5_mean,
                                    conf_max, mask_path (relative to the run
                                    folder), mask_bytes, width, height,
                                    image_path, batch, error
    pv99_genus_summary.csv          one row per genus
    run_metrics.csv, pv99_index_genera.csv   copied in, if present

--csv also writes the metadata as CSV (several GB at this size).
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import all_genera_report as run1  # noqa: E402

METADATA = "pv99_segment_metadata"
SUMMARY = "pv99_genus_summary.csv"
COLUMNS = ["photo_id", "genus", "family", "scientific_name", "observation_uuid",
           "prediction_probability", "certainty", "status", "num_masks", "mask_scores",
           "conf_top5_mean", "conf_max", "mask_path", "mask_bytes", "width", "height",
           "image_path", "batch", "error"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="The pv99 run's output directory")
    parser.add_argument("--index", required=True, help="pv99_index.parquet")
    parser.add_argument("--export", required=True, help="Folder to write into")
    parser.add_argument("--csv", action="store_true", help="Also write the metadata as CSV")
    args = parser.parse_args()

    run = Path(args.run_dir).expanduser().resolve()
    export = Path(args.export).expanduser().resolve()
    export.mkdir(parents=True, exist_ok=True)

    meta = run1.build_metadata(run, args.index)
    for column in COLUMNS:
        if column not in meta:
            meta[column] = None
    meta = meta[COLUMNS]
    meta.to_parquet(export / f"{METADATA}.parquet", index=False)
    if args.csv:
        meta.to_csv(export / f"{METADATA}.csv", index=False)
    run1.genus_summary(meta).to_csv(export / SUMMARY, index=False)

    for f in [run / "run_metrics.csv", Path(args.index).with_name(
            Path(args.index).stem + "_genera.csv")]:
        if f.is_file():
            shutil.copy2(f, export / f.name)

    masked = meta["num_masks"] > 0
    print(f"metadata   : {len(meta):,} photos, {meta['genus'].nunique():,} genera; "
          f"{int(masked.sum()):,} with a mask, {int(meta['num_masks'].sum()):,} masks")
    print(f"masks      : {meta['mask_bytes'].fillna(0).sum() / 1e9:.2f} GB of .npz "
          f"under {run / 'masks'}")
    print(f"           : {export / (METADATA + '.parquet')}")


if __name__ == "__main__":
    main()
