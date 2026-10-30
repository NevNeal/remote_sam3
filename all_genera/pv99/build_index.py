#!/usr/bin/env python3
"""
The pv99 run's index: every flower-detected angiosperm photo that phenovision
scored at prediction_probability >= --min-probability (0.99), with no per-genus
cap.

    python all_genera/pv99/build_index.py \\
        --annotations $SHARE/r.dinnage/.../annotations_internal_all_..._2026-03-15.csv \\
        --image-root $SHARE \\
        --out data/pv99_index.parquet

Steps 1-3 are run 1's (build_genus_index.py): angiosperm genera from the iNat
taxa dump, the annotation rows with trait 'flower present' + predictionClass
'Detected' (one per photo, the most probable), and the photo_id -> file listing,
cached and reused. Step 4 keeps the photos at or above the probability cut that
have a file on disk.

Writes the index (segment.py's --local-index columns, plus genus, family,
observation_uuid, prediction_probability, certainty) and <out stem>_genera.csv:
one row per genus with how many photos it had at each step. The last lines of
the log are the photo count the GPU job will see.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from build_genus_index import angiosperm_names, flower_rows, image_paths  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", required=True, help="phenovision annotations CSV")
    parser.add_argument("--image-root", required=True,
                        help="Directory that file_name's leading 'data/' maps to")
    parser.add_argument("--images-subdir", default="phenobase_inat_data/images/medium")
    parser.add_argument("--taxa", default="data/taxa.csv.gz",
                        help="iNat taxa dump; downloaded here if missing")
    parser.add_argument("--path-cache", default="data/image_paths_all.parquet",
                        help="photo_id -> file listing; built if missing")
    parser.add_argument("--min-probability", type=float, default=0.99)
    parser.add_argument("--out", required=True, help="Output parquet")
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    genera, families = angiosperm_names(Path(args.taxa))
    print(f"taxonomy   : {len(genera):,} angiosperm genus names, {len(families):,} families")

    rows, seen = flower_rows(args.annotations, args.chunksize)
    print(f"annotations: {seen:,} rows read; {len(rows):,} flower-detected photos "
          f"in {rows['genus'].nunique():,} genera")
    rows["angiosperm"] = rows["genus"].isin(genera) | rows["family"].isin(families)
    kept = rows[rows["angiosperm"]].copy()
    print(f"angiosperms: {len(kept):,} photos in {kept['genus'].nunique():,} genera")

    sure = kept[kept["prediction_probability"] >= args.min_probability]
    print(f"probability: {len(sure):,} at >= {args.min_probability} "
          f"({100 * len(sure) / max(1, len(kept)):.1f}%) in {sure['genus'].nunique():,} genera")

    paths = image_paths(Path(args.path_cache), Path(args.image_root).expanduser(),
                        args.images_subdir)
    chosen = sure.merge(paths, on="photo_id", how="inner")
    chosen = chosen.sort_values("photo_id").reset_index(drop=True)
    chosen["taxon_id"] = chosen["genus"].map(genera).fillna(0).astype("int64")
    index = chosen[["photo_id", "taxon_id", "scientific_name", "reproductive_condition",
                    "file_name", "genus", "family", "observation_uuid",
                    "prediction_probability", "certainty"]]
    index.to_parquet(out, index=False)

    by = kept.groupby("genus")
    summary = pd.DataFrame({
        "family": by["family"].first(),
        "flower_detected": by.size(),
        "at_min_probability": sure.groupby("genus").size(),
        "selected": chosen.groupby("genus").size(),
    }).fillna(0).astype({"at_min_probability": int, "selected": int})
    summary.insert(1, "taxon_id", summary.index.map(genera).fillna(0).astype("int64"))
    summary_path = out.with_name(out.stem + "_genera.csv")
    summary.rename_axis("genus").reset_index().to_csv(summary_path, index=False)

    print(f"images     : {len(chosen):,} of those have a file on disk")
    print(f"index      : {len(index):,} photos in {index['genus'].nunique():,} genera -> {out}")
    print(f"           : per-genus counts -> {summary_path}")


if __name__ == "__main__":
    main()
