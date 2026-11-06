#!/usr/bin/env python3
"""
Which source images the pv99 download bundle needs, so the review figures and
top-segment cut-outs can be drawn off HPG without copying all ~270k photos.

    python all_genera/pv99/pack_select.py export/pv99/pv99_segment_metadata.parquet \\
        --image-root $IMAGE_ROOT --out export/pv99/pack_images.txt

Two pools, both restricted to photos with at least one mask:

  figures   per family, FIG_N (+ FIG_SPARE) photos spread across its genera:
            the photos are shuffled once (--seed), ranked within their genus,
            and taken genus-rank first, so a family of 40 genera gives one
            photo from each before any genus gives a second. A family with
            fewer masked photos keeps all of them.
  segments  for the TOP_FAMILIES families with the most photos, the TOP_POOL
            photos with the highest single-mask score (conf_max). The final
            100 per family are chosen locally from this pool.

Writes one path per line, relative to --image-root (what tar -C wants), plus
<out stem>_selection.csv: photo_id, family, genus, pool, rank.
"""

import argparse
from pathlib import Path

import pandas as pd

FIG_N = 30
FIG_SPARE = 10
TOP_FAMILIES = 30
TOP_POOL = 200


def figure_pool(meta, per_family=FIG_N + FIG_SPARE, seed=0):
    masked = meta[meta["num_masks"].fillna(0) > 0]
    shuffled = masked.sample(frac=1.0, random_state=seed)
    shuffled = shuffled.assign(genus_rank=shuffled.groupby("genus").cumcount())
    shuffled = shuffled.assign(order=range(len(shuffled)))
    ordered = shuffled.sort_values(["family", "genus_rank", "order"])
    picked = ordered.groupby("family", sort=False).head(per_family).copy()
    picked["rank"] = picked.groupby("family").cumcount()
    picked["pool"] = "figures"
    return picked


def top_families(meta, n=TOP_FAMILIES):
    return meta["family"].value_counts().head(n).index.tolist()


def segment_pool(meta, families, per_family=TOP_POOL):
    masked = meta[(meta["num_masks"].fillna(0) > 0) & meta["family"].isin(families)]
    ordered = masked.sort_values(["family", "conf_max", "photo_id"],
                                 ascending=[True, False, True])
    picked = ordered.groupby("family", sort=False).head(per_family).copy()
    picked["rank"] = picked.groupby("family").cumcount()
    picked["pool"] = "segments"
    return picked


def relative(path, root):
    path, root = str(path), str(root).rstrip("/") + "/"
    if path.startswith(root):
        return path[len(root):]
    raise ValueError(f"{path} is not under --image-root {root}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metadata", help="pv99_segment_metadata.parquet")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    meta = pd.read_parquet(args.metadata)
    figures = figure_pool(meta, seed=args.seed)
    segments = segment_pool(meta, top_families(meta))
    both = pd.concat([figures, segments], ignore_index=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    both[["photo_id", "family", "genus", "pool", "rank"]].to_csv(
        out.with_name(out.stem + "_selection.csv"), index=False)
    paths = sorted({relative(p, args.image_root) for p in both["image_path"]})
    with open(out, "w", newline="\n") as handle:   # tar -T reads \r as part of the name
        handle.write("\n".join(paths) + "\n")
    print(f"figures    : {len(figures):,} photos from {figures['family'].nunique()} families")
    print(f"segments   : {len(segments):,} photos from {segments['family'].nunique()} families")
    print(f"images     : {len(paths):,} distinct files -> {out}")


if __name__ == "__main__":
    main()
