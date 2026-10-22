#!/usr/bin/env python3
"""
Run 2's index: 30 flower-detected photos per angiosperm genus, chosen by the
phenovision prediction probability rather than at random.

    python all_genera/run2/build_index.py \\
        --annotations $SHARE/r.dinnage/.../annotations_internal_all_..._2026-03-15.csv \\
        --image-root $SHARE \\
        --out data/all_genera_run2_index.parquet

Steps 1-3 are run 1's (build_genus_index.py): angiosperm genera from the iNat
taxa dump, the annotation rows with trait 'flower present' + predictionClass
'Detected' (one per photo, the most probable), and the photo_id -> file
listing, cached and reused. Step 4 is new. Within each genus, over the photos
that have a file on disk, sorted by prediction_probability:

    low     the --per-tier (10) lowest
    median  the --per-tier photos centred on the median
    high    the --per-tier highest

Ties in prediction_probability (common: it is stored at half precision) are
broken by a seeded shuffle, so the pick is reproducible. A genus with fewer
than 3 x --per-tier photos keeps all of them, split into thirds by rank
(one photo is 'median'; two are 'low' and 'high').

Writes the index (segment.py's --local-index columns, plus genus, family,
observation_uuid, prediction_probability, certainty, pv_tier, pv_rank) and
<out stem>_genera.csv: one row per genus with its iNat taxon_id and how many
photos it had at each step. flower_detected there is the genus's phenovision
hit count.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from build_genus_index import angiosperm_names, flower_rows, image_paths  # noqa: E402

TIERS = ("low", "median", "high")


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
    parser.add_argument("--per-tier", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
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

    paths = image_paths(Path(args.path_cache), Path(args.image_root).expanduser(),
                        args.images_subdir)
    on_disk = kept.merge(paths, on="photo_id", how="left").dropna(subset=["file_name"])
    print(f"images     : {len(on_disk):,} with a file on disk, "
          f"{on_disk['genus'].nunique():,} genera")

    chosen = pick_tiers(on_disk, args.per_tier, args.seed)
    chosen["taxon_id"] = chosen["genus"].map(genera).fillna(0).astype("int64")
    index = chosen[["photo_id", "taxon_id", "scientific_name", "reproductive_condition",
                    "file_name", "genus", "family", "observation_uuid",
                    "prediction_probability", "certainty", "pv_tier", "pv_rank"]]
    index.to_parquet(out, index=False)

    by = rows.groupby("genus")
    summary = pd.DataFrame({
        "family": by["family"].first(),
        "angiosperm": by["angiosperm"].first(),
        "flower_detected": by.size(),
        "with_image": on_disk.groupby("genus").size(),
        "selected": chosen.groupby("genus").size(),
    }).fillna({"with_image": 0, "selected": 0}).astype({"with_image": int, "selected": int})
    summary.insert(1, "taxon_id", summary.index.map(genera).fillna(0).astype("int64"))
    summary_path = out.with_name(out.stem + "_genera.csv")
    summary.rename_axis("genus").reset_index().to_csv(summary_path, index=False)

    full = 3 * args.per_tier
    per = chosen.groupby("genus").size()
    print(f"index      : {len(index):,} photos in {len(per):,} genera -> {out}")
    print(f"           : {(per >= full).sum():,} genera have all {full}; "
          f"{(per < full).sum():,} have fewer")
    print(f"           : {index['pv_tier'].value_counts().reindex(TIERS).to_dict()}")
    print(f"           : per-genus counts -> {summary_path}")


def pick_tiers(photos, per_tier, seed):
    """The low / median / high per_tier photos of each genus by
    prediction_probability, with pv_tier and pv_rank (0 = lowest in the genus)."""
    rng = np.random.default_rng(seed)
    photos = photos.assign(_tie=rng.random(len(photos)))
    photos = photos.sort_values(["genus", "prediction_probability", "_tie"])
    picked = []
    for _, group in photos.groupby("genus", sort=False):
        n = len(group)
        tier = np.full(n, "", dtype=object)
        if n >= 3 * per_tier:
            mid = (n - per_tier) // 2
            tier[:per_tier] = "low"
            tier[mid:mid + per_tier] = "median"
            tier[n - per_tier:] = "high"
        elif n == 1:
            tier[0] = "median"
        elif n == 2:
            tier[:] = ["low", "high"]
        else:
            for name, part in zip(TIERS, np.array_split(np.arange(n), 3)):
                tier[part] = name
        group = group.assign(pv_tier=tier, pv_rank=np.arange(n))
        picked.append(group[group["pv_tier"] != ""])
    out = pd.concat(picked, ignore_index=True).drop(columns="_tie")
    return out.sort_values(["genus", "photo_id"]).reset_index(drop=True)


if __name__ == "__main__":
    main()
