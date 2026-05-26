#!/usr/bin/env python3
"""
Download iNaturalist open-data CSVs from S3 and build a local parquet index.

Downloads (one-time, ~28 GB):
    s3://inaturalist-open-data/photos.csv.gz        (~17 GB)
    s3://inaturalist-open-data/observations.csv.gz  (~11 GB)
    s3://inaturalist-open-data/taxa.csv.gz           (<10 MB)

Outputs:
    inat_db/data/inat_photos.parquet  -- joined, research-grade only, sorted by taxon_id

Usage:
    python inat_db/build_db.py                 # download + build
    python inat_db/build_db.py --build-only    # skip download, just build parquet
    python inat_db/build_db.py --download-only # just download, don't build
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import duckdb

# -- Paths --------------------------------------------------------------------
DATA_DIR  = Path(__file__).parent / "data"
PHOTOS_GZ = DATA_DIR / "photos.csv.gz"
OBS_GZ    = DATA_DIR / "observations.csv.gz"
TAXA_GZ   = DATA_DIR / "taxa.csv.gz"
PARQUET   = DATA_DIR / "inat_photos.parquet"

S3_BASE = "s3://inaturalist-open-data"

FILES = [
    (f"{S3_BASE}/taxa.csv.gz",         TAXA_GZ,   "<10 MB"),
    (f"{S3_BASE}/observations.csv.gz", OBS_GZ,    "~11 GB"),
    (f"{S3_BASE}/photos.csv.gz",       PHOTOS_GZ, "~17 GB"),
]

# -- Download -----------------------------------------------------------------
def download_file(src, dst, label):
    if dst.exists():
        gb = dst.stat().st_size / 1e9
        print(f"  [skip]     {dst.name}  ({gb:.2f} GB on disk)")
        return
    print(f"  [download] {dst.name}  (expected {label})")
    t0 = time.time()
    result = subprocess.run(["aws", "s3", "cp", "--no-sign-request", src, str(dst)])
    if result.returncode != 0:
        print(f"  ERROR: aws s3 cp failed for {src}", file=sys.stderr)
        sys.exit(1)
    elapsed = time.time() - t0
    gb = dst.stat().st_size / 1e9
    print(f"  -> {gb:.2f} GB in {elapsed:.0f}s  ({gb / elapsed * 1000:.1f} MB/s)")

def run_downloads():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Downloading iNaturalist open-data CSVs ===")
    for src, dst, label in FILES:
        download_file(src, dst, label)
    print()

# -- Build parquet ------------------------------------------------------------
def build_parquet():
    for f in [PHOTOS_GZ, OBS_GZ, TAXA_GZ]:
        if not f.exists():
            print(f"ERROR: missing {f} -- run without --build-only first.", file=sys.stderr)
            sys.exit(1)

    if PARQUET.exists():
        gb = PARQUET.stat().st_size / 1e9
        print(f"[skip] parquet already exists: {PARQUET.name} ({gb:.2f} GB)")
        print("  Delete it manually to rebuild.")
        return

    print("=== Building parquet index ===")
    print("  Joining photos + observations + taxa (research-grade only) ...")
    print("  (This may take 20-60 min depending on disk speed.)")

    t0 = time.time()
    con = duckdb.connect()
    con.execute("SET threads = 4")
    con.execute("SET memory_limit = '12GB'")

    p_photos = str(PHOTOS_GZ).replace("\\", "/")
    p_obs    = str(OBS_GZ).replace("\\", "/")
    p_taxa   = str(TAXA_GZ).replace("\\", "/")
    p_out    = str(PARQUET).replace("\\", "/")

    sql = f"""
        COPY (
            SELECT
                TRY_CAST(o.taxon_id AS INTEGER) AS taxon_id,
                TRY_CAST(p.photo_id AS BIGINT)  AS photo_id,
                p.extension                      AS extension,
                t.name                           AS taxon_name,
                o.quality_grade                  AS quality_grade,
                TRY_CAST(o.latitude  AS FLOAT)  AS latitude,
                TRY_CAST(o.longitude AS FLOAT)  AS longitude
            FROM read_csv('{p_photos}', sep='\\t', header=true,
                          columns={{'photo_uuid':'VARCHAR','photo_id':'VARCHAR',
                                    'observation_uuid':'VARCHAR','observer_id':'VARCHAR',
                                    'extension':'VARCHAR','license':'VARCHAR',
                                    'width':'VARCHAR','height':'VARCHAR','position':'VARCHAR'}}) p
            JOIN read_csv('{p_obs}', sep='\\t', header=true,
                          columns={{'observation_uuid':'VARCHAR','observer_id':'VARCHAR',
                                    'latitude':'VARCHAR','longitude':'VARCHAR',
                                    'positional_accuracy':'VARCHAR','taxon_id':'VARCHAR',
                                    'quality_grade':'VARCHAR','observed_on':'VARCHAR',
                                    'anomaly_score':'VARCHAR'}}) o
              ON p.observation_uuid = o.observation_uuid
            LEFT JOIN read_csv('{p_taxa}', sep='\\t', header=true,
                          columns={{'taxon_id':'VARCHAR','ancestry':'VARCHAR',
                                    'rank_level':'VARCHAR','rank':'VARCHAR',
                                    'name':'VARCHAR','active':'VARCHAR'}}) t
              ON o.taxon_id = t.taxon_id
            WHERE o.quality_grade = 'research'
            ORDER BY TRY_CAST(o.taxon_id AS INTEGER)
        )
        TO '{p_out}' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 500000)
    """
    con.execute(sql)
    con.close()

    elapsed = time.time() - t0
    gb = PARQUET.stat().st_size / 1e9
    con2 = duckdb.connect(read_only=True)
    row_count = con2.execute(f"SELECT COUNT(*) FROM read_parquet('{p_out}')").fetchone()[0]
    con2.close()

    print(f"\n=== Parquet built ===")
    print(f"  Path      : {PARQUET}")
    print(f"  Size      : {gb:.2f} GB")
    print(f"  Rows      : {row_count:,}")
    print(f"  Build time: {elapsed/60:.1f} min")

# -- Query benchmark ----------------------------------------------------------
def benchmark_query(taxon_id=53324):
    if not PARQUET.exists():
        print("No parquet to benchmark yet.")
        return
    p_out = str(PARQUET).replace("\\", "/")
    print(f"\n=== Query benchmark (taxon_id={taxon_id}) ===")
    con = duckdb.connect(read_only=True)
    t0 = time.time()
    df = con.execute(f"""
        SELECT photo_id, extension, taxon_name
        FROM read_parquet('{p_out}')
        WHERE taxon_id = {taxon_id}
    """).df()
    elapsed = time.time() - t0
    con.close()
    print(f"  {len(df):,} photos in {elapsed:.3f}s")
    if not df.empty:
        row = df.iloc[0]
        print(f"  Sample URL : https://inaturalist-open-data.s3.amazonaws.com"
              f"/photos/{int(row['photo_id'])}/original.{row['extension']}")
        print(f"  Taxon name : {row['taxon_name']}")

# -- Main ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-only",    action="store_true",
                    help="Skip download, only build parquet from existing CSVs")
    ap.add_argument("--download-only", action="store_true",
                    help="Only download CSVs, don't build parquet")
    ap.add_argument("--benchmark",     type=int, default=53324, metavar="TAXON_ID",
                    help="Run a query benchmark after building (default: 53324)")
    args = ap.parse_args()

    if not args.build_only:
        run_downloads()
    if not args.download_only:
        build_parquet()
        benchmark_query(args.benchmark)

if __name__ == "__main__":
    main()
