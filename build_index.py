#!/usr/bin/env python3
"""
Build the parquet photo index that segment.py reads.

The index is pure metadata - no images. iNaturalist publishes its whole database
as three tab-separated dumps on a public S3 bucket; this joins them, keeps only
research-grade observations, and writes seven columns:

    taxon_id  photo_id  extension  taxon_name  quality_grade  latitude  longitude

The image URL is not stored, because it is derivable:

    https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/original.{extension}

Result: ~289M rows, ~4.6 GB (2026-08 dumps). A filtered read for one taxon takes
well under a second, versus ~20 minutes of paging the iNaturalist API against its
rate limits. That speed is the entire reason the index exists.

Note on duplicates: iNat's dumps contain ~0.1% byte-identical duplicate photo
rows, so the index inherits them. They are NOT de-duplicated here -- a DISTINCT
over 289M rows is an expensive extra pass, and the index is meant to mirror the
source. segment.py drops them per-taxon at read time instead, which is where it
matters, because the two copies would otherwise be assigned to different shards.

    python build_index.py --data-dir data

Steps are independently skippable and each one skips itself if its output is
already on disk, so an interrupted run can simply be rerun:

    python build_index.py --data-dir DIR --download-only   # ~28 GB of CSVs
    python build_index.py --data-dir DIR --build-only      # just the join

Needs ~35 GB of free space and ~16 GB of RAM. On HiPerGator, submit it as a CPU
job rather than running it on a login node: sbatch slurm/build_index.sbatch
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import duckdb

S3_BUCKET = "s3://inaturalist-open-data"

# (filename, approximate download size) - order matters only for the progress
# story: smallest first, so a credentials or network problem surfaces in seconds.
SOURCES = [
    ("taxa.csv.gz", "~40 MB"),
    ("observations.csv.gz", "~11 GB"),
    ("photos.csv.gz", "~17 GB"),
]

# The dumps have no embedded schema, so DuckDB needs the column lists. Everything
# is read as VARCHAR and cast afterwards: a single malformed row in 259M would
# otherwise abort the whole join.
PHOTOS_COLUMNS = {
    "photo_uuid": "VARCHAR", "photo_id": "VARCHAR", "observation_uuid": "VARCHAR",
    "observer_id": "VARCHAR", "extension": "VARCHAR", "license": "VARCHAR",
    "width": "VARCHAR", "height": "VARCHAR", "position": "VARCHAR",
}
OBSERVATIONS_COLUMNS = {
    "observation_uuid": "VARCHAR", "observer_id": "VARCHAR", "latitude": "VARCHAR",
    "longitude": "VARCHAR", "positional_accuracy": "VARCHAR", "taxon_id": "VARCHAR",
    "quality_grade": "VARCHAR", "observed_on": "VARCHAR", "anomaly_score": "VARCHAR",
}
TAXA_COLUMNS = {
    "taxon_id": "VARCHAR", "ancestry": "VARCHAR", "rank_level": "VARCHAR",
    "rank": "VARCHAR", "name": "VARCHAR", "active": "VARCHAR",
}


def download(data_dir):
    """Pull the three dumps from S3. Anonymous - the bucket is public."""
    # Resolve the executable explicitly: on Windows the installed entry point is
    # aws.CMD, which CreateProcess will not find from the bare name "aws".
    aws = shutil.which("aws")
    if not aws:
        sys.exit("ERROR: the `aws` CLI is not on PATH. It is in requirements.txt; "
                 "activate the environment first.")

    data_dir.mkdir(parents=True, exist_ok=True)
    for name, size in SOURCES:
        target = data_dir / name
        if target.exists():
            print(f"  [have] {name} ({target.stat().st_size / 1e9:.2f} GB)")
            continue
        print(f"  [get ] {name} (expect {size})")
        started = time.time()
        result = subprocess.run(
            [aws, "s3", "cp", "--no-sign-request", f"{S3_BUCKET}/{name}", str(target)]
        )
        if result.returncode != 0:
            sys.exit(f"ERROR: aws s3 cp failed for {name}")
        gb = target.stat().st_size / 1e9
        elapsed = time.time() - started
        print(f"         {gb:.2f} GB in {elapsed / 60:.1f} min ({gb * 1000 / elapsed:.0f} MB/s)")


def build(data_dir, parquet, memory_limit, threads):
    """Join photos -> observations -> taxa and write the parquet."""
    missing = [name for name, _ in SOURCES if not (data_dir / name).exists()]
    if missing:
        sys.exit(f"ERROR: missing {', '.join(missing)} - run without --build-only first.")

    if parquet.exists():
        print(f"  [have] {parquet.name} ({parquet.stat().st_size / 1e9:.2f} GB)")
        print("         delete it to force a rebuild.")
        return

    def path(name):
        return (data_dir / name).as_posix()

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    # Spill to the data directory rather than /tmp, which is small on compute nodes.
    con.execute(f"SET temp_directory='{data_dir.as_posix()}/duckdb_tmp'")
    # Without this DuckDB tries to preserve row order across the join and blows
    # its memory limit sorting 400M rows. We sort per-taxon at read time anyway.
    con.execute("SET preserve_insertion_order=false")

    # A partial file from a killed job would otherwise look like a finished index.
    staging = parquet.with_suffix(".parquet.partial")
    staging.unlink(missing_ok=True)

    print("  joining photos + observations + taxa, research-grade only")
    print("  (20-60 min depending on disk speed)")
    started = time.time()
    con.execute(f"""
        COPY (
            SELECT
                TRY_CAST(o.taxon_id AS INTEGER)  AS taxon_id,
                TRY_CAST(p.photo_id AS BIGINT)   AS photo_id,
                p.extension                      AS extension,
                t.name                           AS taxon_name,
                o.quality_grade                  AS quality_grade,
                TRY_CAST(o.latitude  AS FLOAT)   AS latitude,
                TRY_CAST(o.longitude AS FLOAT)   AS longitude
            FROM read_csv('{path("photos.csv.gz")}',
                          sep='\t', header=true, columns={PHOTOS_COLUMNS}) p
            JOIN read_csv('{path("observations.csv.gz")}',
                          sep='\t', header=true, columns={OBSERVATIONS_COLUMNS}) o
              ON p.observation_uuid = o.observation_uuid
            LEFT JOIN read_csv('{path("taxa.csv.gz")}',
                          sep='\t', header=true, columns={TAXA_COLUMNS}) t
              ON o.taxon_id = t.taxon_id
            WHERE o.quality_grade = 'research'
        )
        TO '{staging.as_posix()}'
        (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 500000)
    """)

    rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{staging.as_posix()}')"
    ).fetchone()[0]
    con.close()

    staging.replace(parquet)
    print(f"\n  {rows:,} rows, {parquet.stat().st_size / 1e9:.2f} GB, "
          f"{(time.time() - started) / 60:.1f} min")
    print(f"  -> {parquet}")


def verify(parquet, taxon_id):
    """Prove the index answers the only question segment.py asks of it."""
    print(f"\n-- verify: taxon {taxon_id} ------------------------------------------")
    con = duckdb.connect()
    started = time.time()
    rows = con.execute(f"""
        SELECT photo_id, extension, taxon_name
        FROM read_parquet('{parquet.as_posix()}')
        WHERE taxon_id = {taxon_id}
        LIMIT 3
    """).fetchdf()
    elapsed = time.time() - started
    total = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{parquet.as_posix()}')
        WHERE taxon_id = {taxon_id}
    """).fetchone()[0]
    con.close()

    print(f"  {total:,} photos found in {elapsed:.1f}s")
    for _, row in rows.iterrows():
        print(f"  {row['taxon_name']}: https://inaturalist-open-data.s3.amazonaws.com"
              f"/photos/{int(row['photo_id'])}/original.{row['extension']}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True,
                        help="Where the CSVs and the parquet live (use /blue, not $HOME)")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--memory-limit", default="16GB", help="DuckDB memory ceiling")
    parser.add_argument("--threads", type=int, default=8, help="DuckDB worker threads")
    parser.add_argument("--verify-taxon", type=int, default=53324,
                        help="Taxon to spot-check after building (0 to skip)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    parquet = data_dir / "inat_photos.parquet"

    if not args.build_only:
        print("-- download ---------------------------------------------------------")
        download(data_dir)

    if not args.download_only:
        print("-- build ------------------------------------------------------------")
        build(data_dir, parquet, args.memory_limit, args.threads)
        if args.verify_taxon:
            verify(parquet, args.verify_taxon)

        print("\nThe three .csv.gz files are only needed for rebuilds (iNat refreshes")
        print("monthly). Delete them to reclaim ~28 GB, or keep them if you plan to")
        print("rebuild soon:")
        print(f"  rm {data_dir}/*.csv.gz")


if __name__ == "__main__":
    main()
