#!/usr/bin/env python3
"""
Gather the per-shard CSVs a run leaves behind into one results.csv, and say what
actually happened.

Each shard appends only to its own results_shard_NNN.csv while it runs, so this is
a concatenate plus a set of sanity checks, not a conflict resolution. It answers
the two questions you always have after a big run: did every shard finish, and
what needs doing again.

    python collect.py results/160559_flower --num-shards 60

--num-shards is optional but worth passing: without it there is no way to tell a
shard that finished from one that never started.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="A run's output directory")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="How many shards the run was submitted with")
    parser.add_argument("--retry-ids", default=None,
                        help="Write failed photo_ids to this file")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        sys.exit(f"Not a directory: {run_dir}")

    shard_files = sorted(run_dir.glob("results_shard_*.csv"))
    if not shard_files:
        sys.exit(f"No results_shard_*.csv in {run_dir}")

    merged = read_shards(shard_files)
    merged = clean(merged)

    out_path = run_dir / "results.csv"
    merged.to_csv(out_path, index=False)

    report(merged, out_path)
    check_shards(merged, args.num_shards)
    show_failures(merged, args.retry_ids)


def read_shards(shard_files):
    frames = []
    print("-- shards -----------------------------------------------------------")
    for path in shard_files:
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            print(f"  {path.name:<28} empty")
            continue
        print(f"  {path.name:<28} {len(frame):>9,} rows")
        frames.append(frame)
    if not frames:
        sys.exit("Every shard CSV was empty.")
    return pd.concat(frames, ignore_index=True)


# A complete row always has all three. A row missing any of them was still being
# written when the process died, whatever else happened to parse out of it.
REQUIRED = ["photo_id", "row_index", "status"]


def clean(merged):
    """Drop the damage a job killed at its walltime can leave behind.

    A task terminated mid-write leaves a truncated final line. Filtering on
    photo_id alone does not catch it: photo_id is the FIRST column, so it is
    precisely the field a truncated line still has. Such a row parses with a
    valid id and NaN for everything past the cut. Requiring the later columns
    too is what actually catches it.

    Order matters. That truncated row is a second mention of a photo_id the shard
    had already recorded properly, so incomplete rows must go BEFORE
    de-duplication -- dedup first, with keep="last", would keep the fragment and
    throw away the good row.

    After that, keep="last" is correct: a requeued task re-recording a photo means
    the later row is the newer attempt.
    """
    before = len(merged)
    incomplete = int(merged[REQUIRED].isna().any(axis=1).sum())
    merged = merged.dropna(subset=REQUIRED).copy()
    for column in ("photo_id", "row_index"):
        merged[column] = merged[column].astype("int64")

    merged = merged.drop_duplicates(subset=["photo_id"], keep="last")
    duplicates = before - incomplete - len(merged)

    if incomplete or duplicates:
        print(f"\n  dropped {incomplete:,} incomplete (truncated mid-write) "
              f"and {duplicates:,} duplicate rows")
    return merged.sort_values("row_index").reset_index(drop=True)


def report(merged, out_path):
    counts = merged["status"].value_counts()
    print(f"\n-- results -> {out_path.name} ---------------------------------------")
    print(f"  photos processed  {len(merged):>9,}")
    for status in ("ok", "download_failed", "segment_failed"):
        print(f"  {status:<17} {int(counts.get(status, 0)):>9,}")

    ok = merged[merged["status"] == "ok"]
    if ok.empty:
        return

    masks = pd.to_numeric(ok["num_masks"], errors="coerce").fillna(0)
    detected = int((masks > 0).sum())
    print(f"  masks kept        {int(masks.sum()):>9,}")
    print(f"  photos w/ a mask  {detected:>9,}  ({detected / len(ok):.0%} of ok)")

    seconds = pd.to_numeric(ok["seconds"], errors="coerce").dropna()
    if not seconds.empty:
        print(f"  median s/photo    {seconds.median():>9.2f}")
        # A median well above what the GPU needs means the run was waiting on
        # downloads, not on SAM3 - raise --workers before adding GPUs.
        if seconds.median() > 2.5:
            print("    ^ looks download-bound; try a higher WORKERS")


def check_shards(merged, num_shards):
    present = sorted(merged["shard"].dropna().astype(int).unique())
    print(f"\n-- shards present: {len(present)} ---------------------------------------")
    if not num_shards:
        print("  pass --num-shards to check for shards that never reported")
        return

    missing = sorted(set(range(num_shards)) - set(present))
    if not missing:
        print(f"  all {num_shards} reported")
        return

    spec = ",".join(str(shard) for shard in missing)
    print(f"  MISSING: {spec}")
    print("  rerun exactly those - NUM_SHARDS must match the original run:")
    print(f"    NUM_SHARDS={num_shards} sbatch --array={spec} slurm/array.sbatch")


def show_failures(merged, retry_path):
    failed = merged[merged["status"].isin(["download_failed", "segment_failed"])]
    if failed.empty:
        return

    print(f"\n-- {len(failed):,} failures ----------------------------------------------")
    top = failed["error"].astype(str).str.slice(0, 66).value_counts().head(5)
    for message, count in top.items():
        print(f"  {count:>7,}  {message}")
    print("\n  Dead photo URLs are normal - iNat users delete photos, and the")
    print("  open-data dump lags. A steady few percent is expected; a sudden jump")
    print("  usually means you were rate-limited and should lower WORKERS.")

    if retry_path:
        Path(retry_path).write_text(
            "\n".join(str(v) for v in failed["photo_id"]) + "\n", encoding="utf-8")
        print(f"\n  photo_ids -> {retry_path}")


if __name__ == "__main__":
    main()
