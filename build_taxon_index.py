#!/usr/bin/env python3
"""
Build a local index for one taxon: every photo of it that the phenovision
annotations say has a flower, with the path of its image already on /blue.

    python build_taxon_index.py --taxon-id 62741 \\
        --parquet data/inat_photos.parquet \\
        --annotations $SHARE/r.dinnage/.../annotations_internal_all_..._2026-03-15.csv \\
        --image-root $SHARE \\
        --path-index data/local_image_paths.parquet \\
        --out data/62741_flower_index.parquet

The annotations CSV has no taxon_id and no file path, so it takes three joins:

  1. inat_photos.parquet -> every photo_id of the taxon (research grade, exact
     taxon_id, so varieties and subspecies are not included). Also written out
     as data/<taxon>_photos.parquet.
  2. annotations CSV -> rows whose observedImageGuid (the photo_id) is one of
     those, with trait 'flower present' and predictionClass 'Detected'. Rows
     are per observation: observedImageGuid is the one photo the model saw, so
     an observation's other photos are not in the CSV and are not used.
  3. photo_id -> image file. The images are <images-subdir>/batch_N/<photo_id>.webp
     and N does not follow from the photo_id, so the path is looked up in
     --path-index first and anything it does not cover is found by listing the
     batch folders.

The output has the columns segment.py's --local-index expects (photo_id,
taxon_id, scientific_name, reproductive_condition, file_name), with file_name
as 'data/<path under image-root>', the same convention as local_image_paths.
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd

TRAIT = "flower present"
DETECTED = "Detected"
CSV_COLUMNS = ["scientificName", "trait", "predictionClass", "observedImageGuid",
               "occurrenceID", "predictionProbability", "certainty", "countImages"]
EXTS = (".webp", ".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".gif")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--taxon-id", type=int, required=True)
    parser.add_argument("--parquet", required=True, help="inat_photos.parquet")
    parser.add_argument("--annotations", required=True, help="phenovision annotations CSV")
    parser.add_argument("--image-root", required=True,
                        help="Directory that file_name's leading 'data/' maps to")
    parser.add_argument("--images-subdir", default="phenobase_inat_data/images/medium",
                        help="Where the batch_N folders are, under --image-root")
    parser.add_argument("--path-index", default=None,
                        help="Parquet with photo_id and file_name (local_image_paths.parquet)")
    parser.add_argument("--out", required=True, help="Output parquet")
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    root = Path(args.image_root).expanduser()

    # 1. The taxon's photos.
    t0 = time.time()
    taxon = pd.read_parquet(args.parquet, filters=[("taxon_id", "==", args.taxon_id)])
    taxon = taxon.drop_duplicates(subset=["photo_id"])
    taxon_path = out.with_name(f"{args.taxon_id}_photos.parquet")
    taxon.to_parquet(taxon_path, index=False)
    ids = set(taxon["photo_id"].astype("int64"))
    print(f"parquet    : {len(ids):,} photos of taxon {args.taxon_id} "
          f"in {time.time() - t0:.0f}s -> {taxon_path}")
    if not ids:
        raise SystemExit(f"taxon {args.taxon_id} is not in {args.parquet}")

    # 2. Flower-detected annotation rows for those photos.
    rows, seen, matched = select_annotations(args.annotations, ids, args.taxon_id,
                                              args.chunksize)
    print(f"annotations: {seen:,} rows read; {matched:,} for this taxon's photos; "
          f"{len(rows):,} photos with '{TRAIT}' + {DETECTED}")
    if rows.empty:
        raise SystemExit("no flower-detected annotation rows for this taxon")

    # 3. Image paths.
    paths = lookup_paths(args.path_index, set(rows["photo_id"]), root)
    print(f"paths      : {len(paths):,} found in {args.path_index or '(no path index)'}")
    missing = set(rows["photo_id"]) - set(paths)
    if missing:
        t0 = time.time()
        found = scan_batches(root / args.images_subdir, missing, root)
        paths.update(found)
        print(f"           : {len(found):,} more by listing {args.images_subdir}/batch_* "
              f"in {time.time() - t0:.0f}s")

    rows["file_name"] = rows["photo_id"].map(paths)
    lost = rows[rows["file_name"].isna()]
    index = rows.dropna(subset=["file_name"]).sort_values("photo_id")
    index.to_parquet(out, index=False)
    if len(lost):
        lost_path = out.with_name(out.stem + "_missing.csv")
        lost.drop(columns=["file_name"]).to_csv(lost_path, index=False)
        print(f"missing    : {len(lost):,} flower-detected photos have no image on disk"
              f" -> {lost_path}")
    print(f"index      : {len(index):,} photos -> {out}")


def select_annotations(csv_path, ids, taxon_id, chunksize):
    """Stream the CSV; keep this taxon's flower-detected rows, one per photo."""
    kept, seen, matched = [], 0, 0
    reader = pd.read_csv(csv_path, usecols=lambda c: c in CSV_COLUMNS, dtype=str,
                         chunksize=chunksize)
    for chunk in reader:
        seen += len(chunk)
        chunk["photo_id"] = pd.to_numeric(chunk["observedImageGuid"], errors="coerce")
        chunk = chunk[chunk["photo_id"].isin(ids)]
        matched += len(chunk)
        chunk = chunk[(chunk["trait"] == TRAIT) & (chunk["predictionClass"] == DETECTED)]
        if len(chunk):
            kept.append(chunk)
    if not kept:
        return pd.DataFrame(), seen, matched
    df = pd.concat(kept, ignore_index=True)
    df["photo_id"] = df["photo_id"].astype("int64")
    df["predictionProbability"] = pd.to_numeric(df["predictionProbability"], errors="coerce")
    df = (df.sort_values("predictionProbability", ascending=False)
            .drop_duplicates(subset=["photo_id"]))
    return pd.DataFrame({
        "photo_id": df["photo_id"],
        "taxon_id": taxon_id,
        "scientific_name": df["scientificName"],
        "reproductive_condition": df["trait"],
        "observation_uuid": df["occurrenceID"],
        "prediction_probability": df["predictionProbability"],
        "certainty": df["certainty"],
        "count_images": pd.to_numeric(df.get("countImages"), errors="coerce"),
    }).reset_index(drop=True), seen, matched


def lookup_paths(path_index, wanted, root):
    """photo_id -> file_name from an existing index, kept only if the file
    (or the same stem with another extension) is really there."""
    if not path_index or not Path(path_index).exists():
        return {}
    df = pd.read_parquet(path_index, columns=["photo_id", "file_name"])
    df = df[df["photo_id"].isin(wanted)].drop_duplicates(subset=["photo_id"])
    out = {}
    for pid, name in zip(df["photo_id"], df["file_name"]):
        rel = _exists(root, str(name).removeprefix("data/"))
        if rel:
            out[int(pid)] = "data/" + rel
    return out


def scan_batches(images_dir, wanted, root):
    """List every batch_* folder once, keeping only the photo_ids asked for."""
    found = {}
    for batch in sorted(os.scandir(images_dir), key=lambda e: e.name):
        if not (batch.is_dir() and batch.name.startswith("batch_")):
            continue
        for entry in os.scandir(batch.path):
            stem, ext = os.path.splitext(entry.name)
            if ext in EXTS and stem.isdigit() and int(stem) in wanted:
                found.setdefault(int(stem), "data/" + Path(entry.path).relative_to(root).as_posix())
        if len(found) == len(wanted):
            break
    return found


def _exists(root, rel):
    path = root / rel
    if path.exists():
        return rel
    for ext in EXTS:
        alt = path.with_suffix(ext)
        if alt.exists():
            return alt.relative_to(root).as_posix()
    return None


if __name__ == "__main__":
    main()
