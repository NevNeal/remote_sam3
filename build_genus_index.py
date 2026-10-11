#!/usr/bin/env python3
"""
Build a local index of up to N flower-detected photos for every angiosperm
genus in the phenovision annotations, each with its image path on /blue.

    python build_genus_index.py \\
        --annotations $SHARE/r.dinnage/.../annotations_internal_all_..._2026-03-15.csv \\
        --image-root $SHARE \\
        --out data/all_genera_index.parquet

Four steps:

  1. Which genera are angiosperms. The annotations have genus and family but
     nothing above them, so the taxonomy comes from iNaturalist's taxa dump
     (taxa.csv.gz, ~40 MB, downloaded once): a genus counts if iNat has a genus
     of that name under Angiospermae (taxon 47125), or its family is one.
  2. The annotations -> rows with trait 'flower present' and predictionClass
     'Detected', one per photo (observedImageGuid is the photo_id).
  3. photo_id -> image file. Images are <images-subdir>/batch_N/<photo_id>.webp
     and N does not follow from the photo_id, so every batch folder is listed
     once and the listing is cached in --path-cache for the next run.
  4. Up to --per-genus photos per genus, chosen at random (--seed) from the
     photos that really have an image on disk. A genus with fewer keeps all it
     has.

Writes the index (the columns segment.py's --local-index expects, plus genus,
family, observation_uuid and the annotation's prediction_probability), and
<out stem>_genera.csv: one row per genus with how many photos it had at each
step, including the genera dropped as non-angiosperm.
"""

import argparse
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

TRAIT = "flower present"
DETECTED = "Detected"
ANGIOSPERMAE = 47125
TAXA_URL = "https://inaturalist-open-data.s3.amazonaws.com/taxa.csv.gz"
CSV_COLUMNS = ["scientificName", "trait", "family", "genus", "occurrenceID",
               "predictionClass", "certainty", "predictionProbability", "observedImageGuid"]
EXTS = (".webp", ".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".gif")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", required=True, help="phenovision annotations CSV")
    parser.add_argument("--image-root", required=True,
                        help="Directory that file_name's leading 'data/' maps to")
    parser.add_argument("--images-subdir", default="phenobase_inat_data/images/medium",
                        help="Where the batch_N folders are, under --image-root")
    parser.add_argument("--taxa", default="data/taxa.csv.gz",
                        help="iNat taxa dump; downloaded here if missing")
    parser.add_argument("--path-cache", default="data/image_paths_all.parquet",
                        help="photo_id -> file listing of the batch folders; built if missing")
    parser.add_argument("--per-genus", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="Output parquet")
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    root = Path(args.image_root).expanduser()

    genera, families = angiosperm_names(Path(args.taxa))
    print(f"taxonomy   : {len(genera):,} angiosperm genus names, "
          f"{len(families):,} family names from {args.taxa}")

    rows, seen = flower_rows(args.annotations, args.chunksize)
    print(f"annotations: {seen:,} rows read; {len(rows):,} photos with "
          f"'{TRAIT}' + {DETECTED} in {rows['genus'].nunique():,} genera")

    rows["angiosperm"] = rows["genus"].isin(genera) | rows["family"].isin(families)
    kept = rows[rows["angiosperm"]].copy()
    print(f"angiosperms: {len(kept):,} photos in {kept['genus'].nunique():,} genera "
          f"({rows.loc[~rows['angiosperm'], 'genus'].nunique():,} genera dropped)")

    paths = image_paths(Path(args.path_cache), root, args.images_subdir)
    kept = kept.merge(paths, on="photo_id", how="left")
    on_disk = kept.dropna(subset=["file_name"])
    print(f"images     : {len(on_disk):,} of those photos have a file on disk "
          f"in {on_disk['genus'].nunique():,} genera")

    # Shuffle once, then the first N of each genus: a random N, reproducibly.
    chosen = (on_disk.sample(frac=1, random_state=args.seed)
              .groupby("genus", sort=False).head(args.per_genus)
              .sort_values(["genus", "photo_id"]).reset_index(drop=True))
    chosen["taxon_id"] = chosen["genus"].map(genera).fillna(0).astype("int64")
    index = chosen[["photo_id", "taxon_id", "scientific_name", "reproductive_condition",
                    "file_name", "genus", "family", "observation_uuid",
                    "prediction_probability", "certainty"]]
    index.to_parquet(out, index=False)

    summary = pd.DataFrame({
        "family": rows.groupby("genus")["family"].first(),
        "angiosperm": rows.groupby("genus")["angiosperm"].first(),
        "flower_detected": rows.groupby("genus").size(),
        "with_image": on_disk.groupby("genus").size(),
        "selected": chosen.groupby("genus").size(),
    }).fillna({"with_image": 0, "selected": 0}).astype({"with_image": int, "selected": int})
    summary_path = out.with_name(out.stem + "_genera.csv")
    summary.rename_axis("genus").reset_index().to_csv(summary_path, index=False)

    per = chosen.groupby("genus").size()
    print(f"index      : {len(index):,} photos in {len(per):,} genera -> {out}")
    print(f"           : {(per >= args.per_genus).sum():,} genera have the full "
          f"{args.per_genus}; {(per < args.per_genus).sum():,} have fewer "
          f"(median {int(per[per < args.per_genus].median()) if (per < args.per_genus).any() else 0})")
    print(f"           : per-genus counts -> {summary_path}")


def angiosperm_names(taxa_path):
    """({genus name: taxon_id}, {family names}) for everything under Angiospermae."""
    if not taxa_path.exists():
        taxa_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"taxonomy   : downloading {TAXA_URL}")
        tmp = taxa_path.with_name(taxa_path.name + ".part")
        urllib.request.urlretrieve(TAXA_URL, tmp)
        tmp.replace(taxa_path)
    taxa = pd.read_csv(taxa_path, sep="\t", dtype=str,
                       usecols=["taxon_id", "ancestry", "rank", "name", "active"])
    under = ("/" + taxa["ancestry"].fillna("") + "/").str.contains(f"/{ANGIOSPERMAE}/",
                                                                  regex=False)
    taxa = taxa[under]
    # Active names first, so an inactive homonym does not take the taxon_id.
    genus = taxa[taxa["rank"].isin(["genus", "genushybrid"])].sort_values(
        "active", ascending=False).drop_duplicates(subset=["name"])
    genera = dict(zip(genus["name"], genus["taxon_id"].astype("int64")))
    families = set(taxa.loc[taxa["rank"] == "family", "name"])
    return genera, families


def flower_rows(csv_path, chunksize):
    """Stream the CSV; keep flower-detected rows, the most probable per photo."""
    kept, seen = [], 0
    for chunk in pd.read_csv(csv_path, usecols=lambda c: c in CSV_COLUMNS, dtype=str,
                             chunksize=chunksize):
        seen += len(chunk)
        chunk = chunk[(chunk["trait"] == TRAIT) & (chunk["predictionClass"] == DETECTED)]
        if len(chunk):
            kept.append(chunk)
    if not kept:
        raise SystemExit(f"no '{TRAIT}' + {DETECTED} rows in {csv_path}")
    df = pd.concat(kept, ignore_index=True)
    df["photo_id"] = pd.to_numeric(df["observedImageGuid"], errors="coerce")
    df["prediction_probability"] = pd.to_numeric(df["predictionProbability"], errors="coerce")
    df["genus"] = df["genus"].str.strip()
    df = df.dropna(subset=["photo_id", "genus"])
    df = df[df["genus"] != ""]
    df["photo_id"] = df["photo_id"].astype("int64")
    df = (df.sort_values("prediction_probability", ascending=False)
            .drop_duplicates(subset=["photo_id"]))
    return pd.DataFrame({
        "photo_id": df["photo_id"],
        "scientific_name": df["scientificName"],
        "reproductive_condition": df["trait"],
        "genus": df["genus"],
        "family": df["family"].str.strip(),
        "observation_uuid": df["occurrenceID"],
        "prediction_probability": df["prediction_probability"],
        "certainty": df["certainty"],
    }).reset_index(drop=True), seen


def image_paths(cache, root, images_subdir):
    """photo_id -> file_name ('data/<path under root>') for every image on disk."""
    if cache.exists():
        paths = pd.read_parquet(cache)
        print(f"paths      : {len(paths):,} image files from cache {cache}")
        return paths
    t0 = time.time()
    images_dir = root / images_subdir
    batches = sorted(e.path for e in os.scandir(images_dir)
                     if e.is_dir() and e.name.startswith("batch_"))

    def listing(batch):
        ids, names = [], []
        prefix = "data/" + Path(batch).relative_to(root).as_posix() + "/"
        for entry in os.scandir(batch):
            stem, ext = os.path.splitext(entry.name)
            if ext in EXTS and stem.isdigit():
                ids.append(int(stem))
                names.append(prefix + entry.name)
        return ids, names

    ids, names = [], []
    # Listing is metadata I/O, not CPU: threads overlap the waits on Lustre.
    with ThreadPoolExecutor(max_workers=16) as pool:
        for batch_ids, batch_names in pool.map(listing, batches):
            ids.extend(batch_ids)
            names.extend(batch_names)
    paths = (pd.DataFrame({"photo_id": pd.array(ids, dtype="int64"), "file_name": names})
             .drop_duplicates(subset=["photo_id"]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    paths.to_parquet(cache, index=False)
    print(f"paths      : {len(paths):,} image files in {len(batches):,} batch folders, "
          f"listed in {time.time() - t0:.0f}s -> {cache}")
    return paths


if __name__ == "__main__":
    main()
