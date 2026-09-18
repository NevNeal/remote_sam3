#!/usr/bin/env python3
"""
Merge the per-shard CSVs a SLURM array leaves behind into one results.csv,
and report what the run actually produced.

Every array task appends only to its own results_shard_NNN.csv, so "merging" is
a concatenate + sanity check rather than a conflict resolution.  This also
answers the two questions you always want after a big array: did every shard
finish, and which photo_ids still need a rerun.

Usage:
    python merge_shards.py /blue/<group>/<user>/results/160559_flower
    python merge_shards.py <output_dir> --num-shards 60      # check completeness
    python merge_shards.py <output_dir> --requeue-list retry_ids.txt
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

_p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
_p.add_argument("output_dir", help="The run's output tree (contains results_shard_*.csv)")
_p.add_argument("--num-shards", type=int, default=None,
                help="Expected shard count; warns about shards that produced no CSV")
_p.add_argument("--out", default="results.csv", help="Merged filename (inside output_dir)")
_p.add_argument("--requeue-list", default=None,
                help="Write the photo_ids that failed (dl_failed/seg_failed) to this file")
_args = _p.parse_args()

BASE = Path(_args.output_dir).expanduser().resolve()
if not BASE.is_dir():
    sys.exit(f"Not a directory: {BASE}")

shard_csvs = sorted(BASE.glob("results_shard_*.csv"))
if not shard_csvs:
    sys.exit(f"No results_shard_*.csv under {BASE}")

frames = []
for csv_path in shard_csvs:
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"  {csv_path.name}: empty, skipped")
        continue
    frames.append(df)
    print(f"  {csv_path.name}: {len(df):,} rows")

merged = pd.concat(frames, ignore_index=True)

# A task killed mid-write (SLURM time limit) can leave a truncated final line,
# which pandas may read as a duplicate or a NaN-heavy row.  Drop both.
before = len(merged)
merged = merged.dropna(subset=["photo_id"])
merged["photo_id"] = merged["photo_id"].astype("int64")
merged = merged.drop_duplicates(subset=["photo_id"], keep="last")
if len(merged) != before:
    print(f"\nDropped {before - len(merged):,} duplicate/incomplete rows")

merged = merged.sort_values("global_index").reset_index(drop=True)
out_path = BASE / _args.out
merged.to_csv(out_path, index=False)

# -- Report ------------------------------------------------------------------
counts = merged["status"].value_counts()
print(f"\nMerged -> {out_path}")
print(f"  photos         : {len(merged):,}")
for status in ("ok", "dl_failed", "seg_failed"):
    print(f"  {status:<15}: {int(counts.get(status, 0)):,}")

ok = merged[merged["status"] == "ok"]
if not ok.empty:
    total_masks = int(pd.to_numeric(ok["num_masks"], errors="coerce").fillna(0).sum())
    with_masks = int((pd.to_numeric(ok["num_masks"], errors="coerce").fillna(0) > 0).sum())
    print(f"  masks kept     : {total_masks:,}")
    print(f"  photos w/ mask : {with_masks:,} ({with_masks / len(ok):.1%} of ok)")

# -- Completeness ------------------------------------------------------------
shards_seen = sorted(merged["shard"].dropna().astype(int).unique())
print(f"\n  shards present : {len(shards_seen)}")
if _args.num_shards:
    missing = sorted(set(range(_args.num_shards)) - set(shards_seen))
    if missing:
        print(f"  MISSING shards : {missing}")
        print(f"  -> rerun with:  sbatch --array={','.join(str(m) for m in missing)} ...")
    else:
        print(f"  all {_args.num_shards} shards reported")

# -- Failures worth retrying -------------------------------------------------
failed = merged[merged["status"].isin(["dl_failed", "seg_failed"])]
if not failed.empty and _args.requeue_list:
    Path(_args.requeue_list).write_text(
        "\n".join(str(v) for v in failed["photo_id"]) + "\n", encoding="utf-8"
    )
    print(f"\n  {len(failed):,} failed photo_ids -> {_args.requeue_list}")
elif not failed.empty:
    print(f"\n  {len(failed):,} failures (pass --requeue-list to dump their photo_ids)")
    top = failed["error"].astype(str).str.slice(0, 70).value_counts().head(5)
    for msg, n in top.items():
        print(f"    {n:>7,}  {msg}")
