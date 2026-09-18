#!/usr/bin/env python3
"""
Detailed metrics for a finished run: where the time went, how busy the GPUs
were, what SAM3 found, and what the whole index would cost at this rate.

    python collect.py     results/local_flower --num-shards 5    # first
    python run_metrics.py results/local_flower

Reads from the run directory (everything segment.py and run_shard.sh leave):
    results_shard_*.csv         one row per photo, with per-stage timing
    timing_shard_*.json         per-shard phases: model load, loop, host, GPU
    telemetry/gpu_shard_*.csv   nvidia-smi samples (utilisation, memory, power)
and, when sacct is on the PATH, the SLURM accounting for the job(s) named in
the timing JSONs (allocated time, peak RSS, CPU time, final state).

Prints a report and writes metrics.json and per_shard.csv next to the results.
Works for local runs (images read in place on /blue) and download runs alike.

Definitions:
    loop       segment.py's main loop, first photo to last (excludes model load)
    busy       sum of infer_s: the forward pass, bracketed by cuda.synchronize
    wait       sum of wait_s: main loop idle, waiting on the image reader
    allocated  sacct ElapsedRaw: how long SLURM held the GPU
"""

import argparse
import glob
import json
import shutil
import subprocess
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

STAGES = ["wait_s", "prep_s", "infer_s", "post_s", "save_s"]
NUMERIC = STAGES + ["seconds", "download_s", "download_bytes", "download_mbps",
                    "reused_image", "output_bytes", "width", "height",
                    "num_masks", "shard", "done_at"]
# data/local_image_paths.parquet: every flowering row of the dated annotations.
FULL_INDEX = 2_849_542


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="A run's output directory")
    parser.add_argument("--project-to", type=int, default=FULL_INDEX,
                        help=f"Photos to project cost for (default {FULL_INDEX:,}, "
                             "the full local index)")
    parser.add_argument("--gpus", type=int, nargs="+", default=[5, 20, 60],
                        help="GPU counts to project wall clock for")
    parser.add_argument("--no-sacct", action="store_true",
                        help="Skip querying SLURM accounting")
    args = parser.parse_args()

    run = Path(args.run_dir).expanduser().resolve()
    photos = load_photos(run)
    shards = load_shards(run)
    telemetry = load_telemetry(run, shards)
    sacct = pd.DataFrame() if args.no_sacct else load_sacct(shards)

    per_shard = shard_table(photos, shards, telemetry, sacct)
    summary = {
        "run": run.name,
        "setup": setup(photos, shards),
        "throughput": throughput(photos, shards, per_shard, sacct),
        "stages": stage_table(photos),
        "reads": reads(photos),
        "detections": detections(photos),
        "storage": storage(photos),
    }
    summary["projection"] = projection(summary, args.project_to, args.gpus)

    per_shard.to_csv(run / "per_shard.csv", index=False)
    (run / "metrics.json").write_text(json.dumps(summary, indent=2, default=_json))
    print(render(summary, per_shard, photos))
    print(f"\nwrote {run / 'metrics.json'} and {run / 'per_shard.csv'}")


# -- loading -------------------------------------------------------------------

def load_photos(run):
    """Same cleaning as collect.py: drop truncated rows, then keep the last
    record of each photo."""
    files = sorted(glob.glob(str(run / "results_shard_*.csv")))
    if not files:
        raise SystemExit(f"no results_shard_*.csv in {run}")
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, low_memory=False))
        except pd.errors.EmptyDataError:
            continue
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["photo_id", "row_index", "status"])
    df = df.drop_duplicates(subset=["photo_id"], keep="last")
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
    return df.reset_index(drop=True)


def load_shards(run):
    rows = []
    for f in sorted(glob.glob(str(run / "timing_shard_*.json"))):
        t = json.loads(Path(f).read_text())
        t.update({f"n_{k}": v for k, v in t.pop("tally", {}).items()})
        rows.append(t)
    return pd.DataFrame(rows)


def load_telemetry(run, shards):
    """Per-shard GPU utilisation / memory / power from the nvidia-smi samples,
    restricted to this shard's GPU when the sampler could see several."""
    out = {}
    uuids = dict(zip(shards.get("shard", []), shards.get("gpu_uuid", [])))
    for f in glob.glob(str(run / "telemetry" / "gpu_shard_*.csv")):
        shard = int(Path(f).stem.split("_")[-1])
        try:
            t = pd.read_csv(f, skipinitialspace=True)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        t.columns = [c.split(" [")[0].strip() for c in t.columns]
        uuid = uuids.get(shard)
        if uuid and "uuid" in t and (t["uuid"] == uuid).any():
            t = t[t["uuid"] == uuid]
        cols = ("utilization.gpu", "utilization.memory", "memory.used",
                "memory.total", "power.draw", "temperature.gpu")
        for c in cols:
            t[c] = pd.to_numeric(t.get(c), errors="coerce")
        util = t["utilization.gpu"]
        out[shard] = {
            "samples": len(t),
            "gpu_util_mean": util.mean(),
            "gpu_util_median": util.median(),
            "gpu_idle_samples_pct": 100 * (util == 0).mean() if len(t) else np.nan,
            "gpu_mem_peak_gb": t["memory.used"].max() / 1024,
            "gpu_mem_total_gb": t["memory.total"].max() / 1024,
            "power_mean_w": t["power.draw"].mean(),
            "power_peak_w": t["power.draw"].max(),
            "temp_peak_c": t["temperature.gpu"].max(),
        }
    return out


def load_sacct(shards):
    """Accounting for every job the shards ran under. Empty if sacct is absent
    (e.g. off-cluster) or the jobs have aged out."""
    if shards.empty or not shutil.which("sacct"):
        return pd.DataFrame()
    ids = set()
    for col in ("slurm_array_job_id", "slurm_job_id"):
        ids |= {str(v) for v in shards.get(col, pd.Series(dtype=str)).dropna()}
    if not ids:
        return pd.DataFrame()
    fmt = "JobID,NodeList,State,Start,End,ElapsedRaw,TotalCPU,MaxRSS,AllocCPUS,ReqMem"
    try:
        text = subprocess.run(
            ["sacct", "-j", ",".join(sorted(ids)), "-P", "--units=M", f"--format={fmt}"],
            capture_output=True, text=True, timeout=60, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return pd.DataFrame()
    df = pd.read_csv(StringIO(text), sep="|", dtype=str)
    if df.empty:
        return df
    base = df["JobID"].str.split(".").str[0]
    main = df[df["JobID"] == base].copy()
    # MaxRSS and TotalCPU live on the .batch step, not the parent row.
    batch = df[df["JobID"].str.endswith(".batch")].copy()
    batch["JobID"] = batch["JobID"].str.replace(".batch", "", regex=False)
    main = main.drop(columns=["MaxRSS", "TotalCPU"]).merge(
        batch[["JobID", "MaxRSS", "TotalCPU"]], on="JobID", how="left")
    main["elapsed_s"] = pd.to_numeric(main["ElapsedRaw"], errors="coerce")
    main["max_rss_gb"] = pd.to_numeric(
        main["MaxRSS"].str.rstrip("M"), errors="coerce") / 1024
    main["cpu_s"] = main["TotalCPU"].map(_clock_to_s)
    for c in ("Start", "End"):
        main[c] = pd.to_datetime(main[c], errors="coerce")
    return main


def _clock_to_s(value):
    """sacct TotalCPU ('[D-]HH:MM:SS[.fff]' or 'MM:SS.fff') -> seconds."""
    if not isinstance(value, str) or not value:
        return np.nan
    days = 0
    if "-" in value:
        d, value = value.split("-", 1)
        days = int(d)
    parts = [float(p) for p in value.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return days * 86400 + h * 3600 + m * 60 + s


def _sacct_row(sacct, shard_row):
    """Match a shard to its sacct row: array tasks are <array>_<task>, a
    non-array job is just its job id."""
    if sacct.empty or shard_row is None:
        return {}
    array, task = shard_row.get("slurm_array_job_id"), shard_row.get("slurm_array_task_id")
    key = f"{array}_{task}" if _present(array) and _present(task) else str(shard_row.get("slurm_job_id"))
    hit = sacct[sacct["JobID"] == key]
    return hit.iloc[0].to_dict() if len(hit) else {}


def _present(value):
    return value is not None and value == value and str(value) not in ("", "None")


# -- tables --------------------------------------------------------------------

def shard_table(photos, shards, telemetry, sacct):
    ids = sorted(set(photos["shard"].dropna().astype(int))
                 | set(shards.get("shard", pd.Series(dtype=int))))
    rows = []
    for s in ids:
        p = photos[photos["shard"] == s]
        match = shards[shards["shard"] == s] if len(shards) else shards
        t = match.iloc[0].to_dict() if len(match) else None
        a = _sacct_row(sacct, t)
        tel = telemetry.get(s, {})
        loop = (t or {}).get("loop_s", np.nan)
        done = p["done_at"].dropna()
        rows.append({
            "shard": s,
            "host": a.get("NodeList") or (t or {}).get("host", ""),
            "gpu": (t or {}).get("gpu", ""),
            "state": a.get("State", ""),
            "photos": len(p),
            "with_mask": int((p["num_masks"] > 0).sum()),
            "failed": int(p["status"].isin(["download_failed", "segment_failed"]).sum()),
            "allocated_s": a.get("elapsed_s", np.nan),
            "model_load_s": (t or {}).get("model_load_s", np.nan),
            "loop_s": loop,
            "photos_per_s": len(p) / loop if _positive(loop) else np.nan,
            "busy_pct_of_loop": 100 * p["infer_s"].sum() / loop if _positive(loop) else np.nan,
            "wait_pct_of_loop": 100 * p["wait_s"].sum() / loop if _positive(loop) else np.nan,
            "save_pct_of_loop": 100 * p["save_s"].sum() / loop if _positive(loop) else np.nan,
            "infer_s_median": p["infer_s"].median(),
            "seconds_p95": p["seconds"].quantile(0.95),
            "first_done": done.min(), "last_done": done.max(),
            "started_at": (t or {}).get("started_at", np.nan),
            "finished_at": (t or {}).get("finished_at", np.nan),
            "cpu_util_pct": (100 * a["cpu_s"] / (a["elapsed_s"] * float(a["AllocCPUS"]))
                             if a and _positive(a.get("elapsed_s")) else np.nan),
            "max_rss_gb": a.get("max_rss_gb", np.nan),
            **{k: tel.get(k, np.nan) for k in (
                "gpu_util_mean", "gpu_idle_samples_pct", "gpu_mem_peak_gb",
                "power_mean_w", "temp_peak_c")},
        })
    return pd.DataFrame(rows)


def _positive(x):
    return x is not None and x == x and x > 0


def setup(photos, shards):
    first = shards.iloc[0] if len(shards) else {}
    return {
        "shards": int(photos["shard"].nunique()),
        "hosts": sorted(shards["host"].dropna().unique().tolist()) if "host" in shards else [],
        "gpus": shards["gpu"].value_counts().to_dict() if "gpu" in shards else {},
        "distinct_gpu_uuids": int(shards["gpu_uuid"].nunique()) if "gpu_uuid" in shards else None,
        "workers": first.get("workers"),
        "chunk": first.get("chunk"),
        "discard_outputs": first.get("discard_outputs"),
        "slurm_jobs": sorted({str(v) for v in shards.get("slurm_array_job_id",
                              pd.Series(dtype=str)).dropna()}),
        "resumed_photos": int(shards["resumed_from"].sum()) if "resumed_from" in shards else 0,
        "local_images": bool((photos["reused_image"] == 1).all()),
    }


def throughput(photos, shards, per_shard, sacct):
    n = len(photos)
    started = shards["started_at"].min() if "started_at" in shards else np.nan
    finished = shards["finished_at"].max() if "finished_at" in shards else np.nan
    wall = finished - started if _positive(finished - started) else np.nan
    # Overlap: if the shards genuinely ran side by side, the sum of their
    # lifetimes is ~N x the wall clock. Near 1x means they ran one after another.
    lifetimes = (shards["finished_at"] - shards["started_at"]).sum() if len(shards) else np.nan
    loop_total = per_shard["loop_s"].sum()
    busy = photos["infer_s"].sum()
    allocated = per_shard["allocated_s"].sum()
    return {
        "photos": n,
        "wall_clock_s": wall,
        "concurrency": lifetimes / wall if _positive(wall) else np.nan,
        "aggregate_photos_per_s": n / wall if _positive(wall) else np.nan,
        "per_gpu_photos_per_s_loop": n / loop_total if _positive(loop_total) else np.nan,
        "per_gpu_photos_per_s_median": per_shard["photos_per_s"].median(),
        "per_gpu_photos_per_s_min": per_shard["photos_per_s"].min(),
        "per_gpu_photos_per_s_max": per_shard["photos_per_s"].max(),
        "model_load_s_mean": per_shard["model_load_s"].mean(),
        "gpu_hours_loop": loop_total / 3600,
        "gpu_hours_busy": busy / 3600,
        "gpu_hours_allocated": allocated / 3600 if _positive(allocated) else np.nan,
        "busy_pct_of_loop": 100 * busy / loop_total if _positive(loop_total) else np.nan,
        "busy_pct_of_allocated": 100 * busy / allocated if _positive(allocated) else np.nan,
        "startup_overhead_pct": (100 * (allocated - loop_total) / allocated
                                 if _positive(allocated) else np.nan),
        "gpu_util_nvidia_smi_mean": per_shard["gpu_util_mean"].mean(),
        "gpu_mem_peak_gb": per_shard["gpu_mem_peak_gb"].max(),
        "power_mean_w": per_shard["power_mean_w"].mean(),
        "max_rss_gb": per_shard["max_rss_gb"].max(),
        "cpu_util_pct_mean": per_shard["cpu_util_pct"].mean(),
        "sacct_available": not sacct.empty,
    }


def stage_table(photos):
    """Per-photo seconds by stage, and each stage's share of main-loop time."""
    ok = photos[photos["status"] == "ok"]
    total = ok[STAGES].sum().sum()
    out = {}
    for col in STAGES + ["seconds"]:
        s = ok[col].dropna()
        out[col] = {
            "mean": s.mean(), "median": s.median(), "p90": s.quantile(0.9),
            "p99": s.quantile(0.99), "max": s.max(), "total_h": s.sum() / 3600,
            "share_pct": 100 * s.sum() / total if col in STAGES and total else np.nan,
        }
    return out


def reads(photos):
    """Image fetch: disk reads for local runs, HTTP downloads otherwise."""
    p = photos.dropna(subset=["download_s"])
    mp = photos["width"] * photos["height"] / 1e6
    mbps = p["download_bytes"] / 1e6 / p["download_s"].replace(0, np.nan)
    return {
        "source": "disk (/blue, in place)" if (photos["reused_image"] == 1).all() else "download",
        "file_mb_median": p["download_bytes"].median() / 1e6,
        "file_mb_p95": p["download_bytes"].quantile(0.95) / 1e6,
        "total_gb": p["download_bytes"].sum() / 1e9,
        "read_s_median": p["download_s"].median(),
        "read_s_p95": p["download_s"].quantile(0.95),
        "read_s_max": p["download_s"].max(),
        "read_mb_s_median": mbps.median(),
        "megapixels_median": mp.median(),
        "megapixels_p95": mp.quantile(0.95),
        "long_side_px_median": photos[["width", "height"]].max(axis=1).median(),
    }


def detections(photos):
    ok = photos[photos["status"] == "ok"]
    masks = ok["num_masks"].fillna(0).astype(int)
    scores = (ok["mask_scores"].dropna().astype(str).str.split(",")
              .explode().pipe(pd.to_numeric, errors="coerce").dropna())
    top = (ok["mask_scores"].dropna().astype(str).str.split(",").str[0]
           .pipe(pd.to_numeric, errors="coerce").dropna())
    bins = masks.clip(upper=6).value_counts().sort_index()
    by_taxon = (ok.assign(hit=masks > 0)
                .groupby("taxon_name", dropna=False)["hit"].agg(["size", "mean"])
                .sort_values("size", ascending=False))
    by_grade = (ok.assign(hit=masks > 0)
                .groupby("quality_grade", dropna=False)["hit"].agg(["size", "mean"]))
    return {
        "photos_ok": len(ok),
        "photos_with_mask": int((masks > 0).sum()),
        "detection_rate_pct": 100 * (masks > 0).mean() if len(ok) else np.nan,
        "masks_total": int(masks.sum()),
        "masks_per_photo_mean": masks.mean(),
        "masks_per_detected_photo_mean": masks[masks > 0].mean(),
        "masks_per_photo_max": int(masks.max()) if len(masks) else 0,
        "masks_histogram": {("6+" if k == 6 else str(k)): int(v) for k, v in bins.items()},
        "score_all_median": scores.median(), "score_all_p10": scores.quantile(0.1),
        "score_all_p90": scores.quantile(0.9),
        "score_top_median": top.median(), "score_top_p10": top.quantile(0.1),
        "score_top_below_0_5_pct": 100 * (top < 0.5).mean() if len(top) else np.nan,
        "taxa": int(ok["taxon_name"].nunique()),
        "top_taxa": [{"taxon": str(k), "photos": int(r["size"]), "rate_pct": 100 * r["mean"]}
                     for k, r in by_taxon.head(10).iterrows()],
        "by_quality_grade": {str(k): {"photos": int(r["size"]), "rate_pct": 100 * r["mean"]}
                             for k, r in by_grade.iterrows()},
    }


def storage(photos):
    """What the outputs cost on disk. output_bytes includes the source image,
    which in a local run is the share's file, not ours - subtract it there."""
    out = photos["output_bytes"].fillna(0)
    if (photos["reused_image"] == 1).all():
        out = (out - photos["download_bytes"].fillna(0)).clip(lower=0)
    ok = photos[photos["status"] == "ok"]
    return {
        "written_gb": out.sum() / 1e9,
        "written_mb_per_photo": out.sum() / 1e6 / max(len(ok), 1),
        "written_mb_per_detected_photo": out[photos["num_masks"] > 0].mean() / 1e6
        if (photos["num_masks"] > 0).any() else np.nan,
        "includes_source_image": not (photos["reused_image"] == 1).all(),
    }


def projection(summary, target, gpu_counts):
    t = summary["throughput"]
    rate = t["per_gpu_photos_per_s_loop"]
    if not _positive(rate):
        return {}
    loop_h = target / rate / 3600
    # Allocated overhead (queue excluded): model load + anything outside the loop.
    overhead = t["gpu_hours_allocated"] / t["gpu_hours_loop"] if _positive(
        t["gpu_hours_allocated"]) else np.nan
    fail = 1 - summary["detections"]["photos_ok"] / max(t["photos"], 1)
    return {
        "photos": target,
        "per_gpu_photos_per_s": rate,
        "gpu_hours_loop": loop_h,
        "gpu_hours_allocated_est": loop_h * overhead if overhead == overhead else np.nan,
        "wall_clock_h": {str(n): loop_h / n for n in gpu_counts},
        "storage_tb": summary["storage"]["written_mb_per_photo"] * target * (1 - fail) / 1e6,
        "expected_masks": summary["detections"]["masks_per_photo_mean"] * target * (1 - fail),
    }


# -- report --------------------------------------------------------------------

def render(s, per_shard, photos):
    su, tp, st, rd, de, sg, pr = (s[k] for k in (
        "setup", "throughput", "stages", "reads", "detections", "storage", "projection"))
    L = [f"== run metrics: {s['run']} " + "=" * max(0, 56 - len(s["run"]))]

    L += ["", "-- setup " + "-" * 62,
          f"  shards {su['shards']} on {len(su['hosts'])} host(s): {', '.join(su['hosts']) or '?'}",
          f"  gpus   " + ", ".join(f"{n}x {g}" for g, n in su["gpus"].items())
          + (f"  ({su['distinct_gpu_uuids']} distinct GPU UUIDs)" if su["distinct_gpu_uuids"] else ""),
          f"  images {rd['source']};  workers {su['workers']}, chunk {su['chunk']}"
          + (f";  resumed {su['resumed_photos']:,}" if su["resumed_photos"] else ""),
          f"  slurm  {', '.join(su['slurm_jobs']) or '-'}"
          + ("" if tp["sacct_available"] else "  (sacct not available - allocated/RSS/CPU blank)")]

    L += ["", "-- throughput " + "-" * 57,
          f"  wall clock          {hms(tp['wall_clock_s'])}   "
          f"(shards overlapped {f(tp['concurrency'], '.1f')}x)",
          f"  aggregate           {f(tp['aggregate_photos_per_s'], '.2f')} photos/s  "
          f"= {f(_per_h(tp['aggregate_photos_per_s']), ',.0f')} photos/h",
          f"  per GPU (loop)      {f(tp['per_gpu_photos_per_s_loop'], '.2f')} photos/s  "
          f"(shards {f(tp['per_gpu_photos_per_s_min'], '.2f')}-{f(tp['per_gpu_photos_per_s_max'], '.2f')})",
          f"  model load          {f(tp['model_load_s_mean'], '.1f')} s per shard",
          f"  GPU-hours           loop {f(tp['gpu_hours_loop'], '.2f')}  busy {f(tp['gpu_hours_busy'], '.2f')}"
          f"  allocated {f(tp['gpu_hours_allocated'], '.2f')}",
          f"  GPU busy            {f(tp['busy_pct_of_loop'], '.0f')}% of loop, "
          f"{f(tp['busy_pct_of_allocated'], '.0f')}% of allocated"
          f"   (nvidia-smi mean {f(tp['gpu_util_nvidia_smi_mean'], '.0f')}%)",
          f"  GPU memory peak     {f(tp['gpu_mem_peak_gb'], '.1f')} GB;  power mean "
          f"{f(tp['power_mean_w'], '.0f')} W",
          f"  host RAM peak       {f(tp['max_rss_gb'], '.1f')} GB;  CPU util "
          f"{f(tp['cpu_util_pct_mean'], '.0f')}% of allocated cores"]

    L += ["", "-- seconds per photo (status ok) " + "-" * 38,
          f"  {'stage':<9}{'mean':>8}{'median':>8}{'p90':>8}{'p99':>8}{'max':>8}{'share':>8}"]
    for col in STAGES + ["seconds"]:
        r = st[col]
        share = "" if r["share_pct"] != r["share_pct"] else f"{r['share_pct']:.0f}%"
        L.append(f"  {col.removesuffix('_s') if col != 'seconds' else 'total*':<9}"
                 + "".join(f"{f(r[k], '.3f'):>8}" for k in ("mean", "median", "p90", "p99", "max"))
                 + f"{share:>8}")
    L.append("  *total = prep+infer+post+save; wait (GPU idle on image reads) is not in it")
    L += _bottleneck(st, tp)

    L += ["", "-- image reads " + "-" * 56,
          f"  file size           median {f(rd['file_mb_median'], '.2f')} MB, p95 "
          f"{f(rd['file_mb_p95'], '.2f')} MB  ({f(rd['total_gb'], '.1f')} GB read)",
          f"  read+decode time    median {f(rd['read_s_median'], '.3f')} s, p95 "
          f"{f(rd['read_s_p95'], '.3f')} s, max {f(rd['read_s_max'], '.2f')} s",
          f"  per-thread speed    median {f(rd['read_mb_s_median'], '.1f')} MB/s",
          f"  image size          median {f(rd['megapixels_median'], '.2f')} MP "
          f"(long side {f(rd['long_side_px_median'], '.0f')} px), p95 {f(rd['megapixels_p95'], '.2f')} MP"]

    hist = "  ".join(f"{k}:{v:,}" for k, v in de["masks_histogram"].items())
    L += ["", "-- detections " + "-" * 57,
          f"  photos with a mask  {de['photos_with_mask']:,} of {de['photos_ok']:,} "
          f"({f(de['detection_rate_pct'], '.1f')}%)",
          f"  masks               {de['masks_total']:,} total; {f(de['masks_per_photo_mean'], '.2f')}/photo, "
          f"{f(de['masks_per_detected_photo_mean'], '.2f')}/detected photo, max {de['masks_per_photo_max']}",
          f"  masks per photo     {hist}",
          f"  score, every mask   median {f(de['score_all_median'], '.3f')}  "
          f"p10 {f(de['score_all_p10'], '.3f')}  p90 {f(de['score_all_p90'], '.3f')}",
          f"  score, best mask    median {f(de['score_top_median'], '.3f')}  "
          f"p10 {f(de['score_top_p10'], '.3f')}  "
          f"({f(de['score_top_below_0_5_pct'], '.0f')}% of detected photos below 0.5)"]
    if de["by_quality_grade"]:
        L.append("  by quality grade    " + "  ".join(
            f"{k} {v['rate_pct']:.0f}% of {v['photos']:,}" for k, v in de["by_quality_grade"].items()))
    L.append(f"  {de['taxa']:,} taxa; most photographed (detection rate):")
    for t in de["top_taxa"]:
        L.append(f"    {t['photos']:>6,}  {t['rate_pct']:>4.0f}%  {t['taxon'][:48]}")

    L += ["", "-- storage " + "-" * 60,
          f"  written             {f(sg['written_gb'], '.2f')} GB  "
          f"({f(sg['written_mb_per_photo'], '.2f')} MB/photo, "
          f"{f(sg['written_mb_per_detected_photo'], '.2f')} MB/detected photo)"
          + ("" if sg["includes_source_image"] else "; source images excluded")]

    if pr:
        L += ["", f"-- projection: {pr['photos']:,} photos at this rate " + "-" * 25,
              f"  GPU-hours           {f(pr['gpu_hours_loop'], ',.0f')} in the loop, ~"
              f"{f(pr['gpu_hours_allocated_est'], ',.0f')} allocated",
              "  wall clock          " + "   ".join(
                  f"{n} GPUs {hms(h * 3600)}" for n, h in pr["wall_clock_h"].items()),
              f"  storage             ~{f(pr['storage_tb'], '.2f')} TB,  ~"
              f"{f(pr['expected_masks'], ',.0f')} masks",
              "  (queue time and per-shard model loads for many small shards not included)"]

    L += ["", "-- per shard " + "-" * 58]
    cols = [("shard", "shard", "d"), ("host", "host", "s"), ("photos", "photos", ",d"),
            ("photos_per_s", "ph/s", ".2f"), ("busy_pct_of_loop", "busy%", ".0f"),
            ("wait_pct_of_loop", "wait%", ".0f"), ("save_pct_of_loop", "save%", ".0f"),
            ("model_load_s", "load_s", ".0f"), ("loop_s", "loop_s", ".0f"),
            ("allocated_s", "alloc_s", ".0f"), ("gpu_util_mean", "smi%", ".0f"),
            ("gpu_mem_peak_gb", "memGB", ".1f"), ("max_rss_gb", "rssGB", ".1f"),
            ("state", "state", "s")]
    L.append("  " + " ".join(f"{h:>8}" if h != "host" else f"{h:<14}" for _, h, _ in cols))
    for _, r in per_shard.iterrows():
        cells = []
        for c, h, fmt in cols:
            v = r[c]
            if h == "host":
                cells.append(f"{str(v)[:14]:<14}")
            elif fmt == "s":
                cells.append(f"{str(v)[:8]:>8}")
            else:
                cells.append(f"{f(v, fmt):>8}")
        L.append("  " + " ".join(cells))
    slow = per_shard["photos_per_s"]
    if len(slow) > 1 and slow.min() < 0.8 * slow.median():
        worst = per_shard.loc[slow.idxmin()]
        L.append(f"  shard {int(worst['shard'])} ran {100 * (1 - worst['photos_per_s'] / slow.median()):.0f}% "
                 f"below the median - check its node ({worst['host']}) for a shared GPU or slow disk")

    slowest = photos.nlargest(5, "seconds")[["photo_id", "shard", "seconds", "width", "height", "num_masks"]]
    if len(slowest):
        L += ["", "-- slowest photos " + "-" * 53]
        for _, r in slowest.iterrows():
            L.append(f"  photo {int(r['photo_id']):>12}  shard {int(r['shard'])}  {r['seconds']:.2f} s  "
                     f"{f(r['width'], '.0f')}x{f(r['height'], '.0f')}  {f(r['num_masks'], '.0f')} masks")
    return "\n".join(L)


def _bottleneck(st, tp):
    """One line saying what limits this run, from the stage shares."""
    shares = {k: st[k]["share_pct"] for k in STAGES}
    top = max(shares, key=lambda k: shares[k] if shares[k] == shares[k] else -1)
    hint = {
        "wait_s": "image reads - raise WORKERS, or check /blue load",
        "infer_s": "the GPU forward pass - more GPUs is the only lever (or --bf16)",
        "save_s": "writing masks/overlays/cutouts - disk bound; drop outputs you do not need",
        "prep_s": "CPU preprocessing - more CPUs per task",
        "post_s": "mask post-processing on CPU - more CPUs per task",
    }[top]
    return [f"  bottleneck: {top.removesuffix('_s')} ({shares[top]:.0f}% of loop) -> {hint}"]


def f(x, fmt):
    try:
        return "—" if x is None or x != x else format(x, fmt)
    except (TypeError, ValueError):
        return str(x)


def _per_h(rate):
    return rate * 3600 if rate == rate else np.nan


def hms(seconds):
    if seconds is None or seconds != seconds:
        return "—"
    s = int(round(seconds))
    if s >= 86400:
        return f"{s // 86400}d {s % 86400 // 3600:02d}h {s % 3600 // 60:02d}m"
    return f"{s // 3600}h {s % 3600 // 60:02d}m {s % 60:02d}s"


def _json(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if value != value else float(value)
    return str(value)


if __name__ == "__main__":
    main()
