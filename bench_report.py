#!/usr/bin/env python3
"""
Summarise a benchmark run saved by slurm/bench_report.sbatch.

    python bench_report.py benchmarks/62741_flower_20260918_120000

Reads, from that directory:
    raw/results_shard_*.csv     one row per photo with download/GPU/save timing
    raw/timing_shard_*.json     per-shard phases (model load, loop, host, GPU)
    raw/telemetry/gpu_*.csv     nvidia-smi samples (utilisation, memory, power)
    sacct.txt                   SLURM accounting for the array

Writes report.md, per_shard.csv, per_node.csv and summary.json next to them.

Definitions used throughout:
    allocated GPU time   how long SLURM held the GPU (sacct Elapsed per task)
    busy GPU time        sum of per-photo infer_s (forward pass, cuda-synced)
    download wait        sum of wait_s: main loop idle, waiting for a download
    download speed       bytes / download_s for each photo (one connection)
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TIMING = ["wait_s", "prep_s", "infer_s", "post_s", "save_s", "seconds",
          "download_s", "download_bytes", "download_mbps", "output_bytes"]


def main():
    bench = Path(sys.argv[1]).resolve()
    raw = bench / "raw"

    photos = load_photos(raw)
    shards = load_shards(raw)
    sacct = load_sacct(bench / "sacct.txt")
    telemetry = load_telemetry(raw, shards)

    per_shard = shard_table(photos, shards, sacct, telemetry)
    per_node = node_table(per_shard)
    summary = overall(photos, per_shard, sacct)

    per_shard.to_csv(bench / "per_shard.csv", index=False)
    per_node.to_csv(bench / "per_node.csv", index=False)
    (bench / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    report = render(bench.name, summary, per_shard, per_node, photos)
    (bench / "report.md").write_text(report, encoding="utf-8")
    print(report)


# -- loading -------------------------------------------------------------------

def load_photos(raw):
    files = sorted(glob.glob(str(raw / "results_shard_*.csv")))
    if not files:
        raise SystemExit(f"no results_shard_*.csv in {raw}")
    df = pd.concat((pd.read_csv(f, low_memory=False) for f in files), ignore_index=True)
    # A shard killed mid-write leaves a truncated last line; keep complete rows.
    df = df.dropna(subset=["photo_id", "status"])
    df = df.drop_duplicates(subset=["photo_id"], keep="last")
    for col in TIMING + ["shard", "num_masks"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    return df


def load_shards(raw):
    rows = []
    for f in sorted(glob.glob(str(raw / "timing_shard_*.json"))):
        t = json.loads(Path(f).read_text())
        t.update({f"n_{k}": v for k, v in t.pop("tally", {}).items()})
        rows.append(t)
    return pd.DataFrame(rows)


def load_sacct(path):
    """Main task rows (JobID like 123_4) with MaxRSS/TotalCPU from the .batch step."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path, sep="|", dtype=str)
    main = df[df["JobID"].str.fullmatch(r"\d+_\d+")].copy()
    batch = df[df["JobID"].str.fullmatch(r"\d+_\d+\.batch")].copy()
    batch["JobID"] = batch["JobID"].str.replace(".batch", "", regex=False)
    main = main.drop(columns=["MaxRSS", "TotalCPU"]).merge(
        batch[["JobID", "MaxRSS", "TotalCPU"]], on="JobID", how="left")
    main["shard"] = main["JobID"].str.split("_").str[1].astype(int)
    main["elapsed_s"] = pd.to_numeric(main["ElapsedRaw"], errors="coerce")
    main["max_rss_gb"] = pd.to_numeric(
        main["MaxRSS"].str.rstrip("M"), errors="coerce") / 1024
    for c in ("Start", "End"):
        main[c] = pd.to_datetime(main[c], errors="coerce")
    return main


def load_telemetry(raw, shards):
    """Mean GPU utilisation / memory / power per shard, from nvidia-smi samples."""
    out = {}
    uuids = dict(zip(shards.get("shard", []), shards.get("gpu_uuid", [])))
    for f in glob.glob(str(raw / "telemetry" / "gpu_shard_*.csv")):
        shard = int(Path(f).stem.split("_")[-1])
        try:
            t = pd.read_csv(f, skipinitialspace=True)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        t.columns = [c.split(" [")[0].strip() for c in t.columns]
        uuid = uuids.get(shard)
        if uuid and "uuid" in t and (t["uuid"] == uuid).any():
            t = t[t["uuid"] == uuid]
        for c in ("utilization.gpu", "memory.used", "power.draw"):
            t[c] = pd.to_numeric(t.get(c), errors="coerce")
        out[shard] = {
            "samples": len(t),
            "gpu_util_mean_pct": t["utilization.gpu"].mean(),
            "gpu_util_busy_samples_pct": (t["utilization.gpu"] > 0).mean() * 100,
            "gpu_mem_peak_gb": t["memory.used"].max() / 1024,
            "power_mean_w": t["power.draw"].mean(),
        }
    return out


# -- tables --------------------------------------------------------------------

def shard_table(photos, shards, sacct, telemetry):
    rows = []
    shard_ids = sorted(set(photos["shard"].dropna().astype(int))
                       | set(shards.get("shard", pd.Series(dtype=int)))
                       | set(sacct.get("shard", pd.Series(dtype=int))))
    for s in shard_ids:
        p = photos[photos["shard"] == s]
        t = shards[shards["shard"] == s].iloc[0] if len(shards) and (shards["shard"] == s).any() else {}
        a = sacct[sacct["shard"] == s].iloc[0] if len(sacct) and (sacct["shard"] == s).any() else {}
        tel = telemetry.get(s, {})
        dl = p[p["reused_image"] != 1] if "reused_image" in p else p
        loop = t.get("loop_s", np.nan)
        rows.append({
            "shard": s,
            "node": a.get("NodeList") if len(a) else t.get("host"),
            "state": a.get("State", "") if len(a) else "",
            "gpu": t.get("gpu", ""),
            "photos": len(p),
            "ok": int((p["status"] == "ok").sum()),
            "no_detections": int((p["status"] == "no_detections").sum()),
            "download_failed": int((p["status"] == "download_failed").sum()),
            "segment_failed": int((p["status"] == "segment_failed").sum()),
            "allocated_s": a.get("elapsed_s", np.nan) if len(a) else np.nan,
            "model_load_s": t.get("model_load_s", np.nan),
            "loop_s": loop,
            "photos_per_s": len(p) / loop if loop and loop == loop else np.nan,
            "gpu_busy_s": p["infer_s"].sum(),
            "gpu_busy_pct_of_loop": 100 * p["infer_s"].sum() / loop if loop else np.nan,
            "download_wait_s": p["wait_s"].sum(),
            "download_wait_pct_of_loop": 100 * p["wait_s"].sum() / loop if loop else np.nan,
            "prep_s": p["prep_s"].sum(), "post_s": p["post_s"].sum(),
            "save_s": p["save_s"].sum(),
            "download_gb": dl["download_bytes"].sum() / 1e9,
            "download_s_median": dl["download_s"].median(),
            "download_s_p95": dl["download_s"].quantile(0.95),
            "download_mb_s_median": dl["download_mbps"].median(),
            "download_gb_per_loop_hour": dl["download_bytes"].sum() / 1e9 / (loop / 3600) if loop else np.nan,
            "output_gb_if_kept": p["output_bytes"].sum() / 1e9,
            "max_rss_gb": a.get("max_rss_gb", np.nan) if len(a) else np.nan,
            **{k: tel.get(k, np.nan) for k in ("gpu_util_mean_pct", "gpu_mem_peak_gb", "power_mean_w")},
        })
    return pd.DataFrame(rows)


def node_table(per_shard):
    g = per_shard.groupby("node", dropna=False)
    out = g.agg(
        shards=("shard", "count"),
        photos=("photos", "sum"),
        gpu_hours_allocated=("allocated_s", lambda x: x.sum() / 3600),
        gpu_hours_busy=("gpu_busy_s", lambda x: x.sum() / 3600),
        download_wait_hours=("download_wait_s", lambda x: x.sum() / 3600),
        download_gb=("download_gb", "sum"),
        gpu_util_mean_pct=("gpu_util_mean_pct", "mean"),
    ).reset_index()
    out["busy_pct_of_allocated"] = 100 * out["gpu_hours_busy"] / out["gpu_hours_allocated"]
    return out


def overall(photos, per_shard, sacct):
    wall = np.nan
    if len(sacct) and sacct["Start"].notna().any():
        wall = (sacct["End"].max() - sacct["Start"].min()).total_seconds()
    dl = photos[photos.get("reused_image") != 1]
    ok = photos[photos["status"].isin(["ok", "no_detections"])]
    errs = photos.loc[photos["status"] == "download_failed", "error"].astype(str)
    return {
        "taxon_id": int(photos["taxon_id"].iloc[0]),
        "taxon_name": photos["taxon_name"].iloc[0],
        "shards": len(per_shard),
        "nodes": per_shard["node"].nunique(),
        "photos": len(photos),
        "status": photos["status"].value_counts().to_dict(),
        "masks": int(photos["num_masks"].sum()),
        "wall_clock_s": wall,
        "first_start": str(sacct["Start"].min()) if len(sacct) else None,
        "last_end": str(sacct["End"].max()) if len(sacct) else None,
        "aggregate_photos_per_s": len(photos) / wall if wall == wall and wall else np.nan,
        "gpu_hours_allocated": per_shard["allocated_s"].sum() / 3600,
        "gpu_hours_busy": per_shard["gpu_busy_s"].sum() / 3600,
        "download_wait_hours": per_shard["download_wait_s"].sum() / 3600,
        "download_gb": dl["download_bytes"].sum() / 1e9,
        "download_s_median": dl["download_s"].median(),
        "download_s_p95": dl["download_s"].quantile(0.95),
        "download_s_max": dl["download_s"].max(),
        "download_mb_s_median": dl["download_mbps"].median(),
        "download_mb_s_p5": dl["download_mbps"].quantile(0.05),
        "download_mb_s_p95": dl["download_mbps"].quantile(0.95),
        "download_mb_median": dl["download_bytes"].median() / 1e6,
        "aggregate_download_mb_s": dl["download_bytes"].sum() / 1e6 / wall if wall == wall and wall else np.nan,
        "per_photo_median_s": {c: ok[c].median() for c in ("wait_s", "prep_s", "infer_s", "post_s", "save_s")},
        "output_gb_if_kept": photos["output_bytes"].sum() / 1e9,
        "output_mb_per_photo": photos["output_bytes"].sum() / 1e6 / max(len(ok), 1),
        "top_download_errors": errs.str.slice(0, 120).value_counts().head(5).to_dict(),
    }


# -- report --------------------------------------------------------------------

def md_table(df, cols, fmt):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("—" if v != v else format(v, fmt.get(c, ".1f")))
            else:
                cells.append(str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def hms(s):
    if s != s:
        return "—"
    s = int(s)
    return f"{s // 3600}h {s % 3600 // 60:02d}m {s % 60:02d}s"


def render(name, s, per_shard, per_node, photos):
    busy_pct = 100 * s["gpu_hours_busy"] / s["gpu_hours_allocated"] if s["gpu_hours_allocated"] else float("nan")
    med = s["per_photo_median_s"]
    lines = [
        f"# Benchmark {name}",
        "",
        f"**{s['taxon_name']}** (taxon {s['taxon_id']}): {s['photos']:,} photos, "
        f"{s['shards']} GPUs on {s['nodes']} node(s).",
        "",
        "## Overall",
        "",
        f"- Wall clock, first shard start to last shard end: **{hms(s['wall_clock_s'])}** "
        f"({s['first_start']} to {s['last_end']})",
        f"- Throughput: **{s['aggregate_photos_per_s']:.2f} photos/s** across all GPUs",
        f"- Status: " + ", ".join(f"{k} {v:,}" for k, v in s["status"].items())
        + f"; {s['masks']:,} masks",
        f"- GPU time: **{s['gpu_hours_allocated']:.2f} GPU-h allocated**, "
        f"{s['gpu_hours_busy']:.2f} GPU-h actually computing (**{busy_pct:.0f}% busy**)",
        f"- Waiting on downloads: {s['download_wait_hours']:.2f} GPU-h",
        f"- Median seconds per photo: wait {med['wait_s']:.3f} · prep {med['prep_s']:.3f} · "
        f"**GPU {med['infer_s']:.3f}** · post {med['post_s']:.3f} · save {med['save_s']:.3f}",
        "",
        "## Downloads",
        "",
        f"- {s['download_gb']:.1f} GB downloaded; median photo {s['download_mb_median']:.2f} MB",
        f"- Time per photo: median {s['download_s_median']:.2f} s, p95 {s['download_s_p95']:.2f} s, "
        f"max {s['download_s_max']:.1f} s",
        f"- Speed per connection: median **{s['download_mb_s_median']:.2f} MB/s** "
        f"(p5 {s['download_mb_s_p5']:.2f}, p95 {s['download_mb_s_p95']:.2f})",
        f"- Aggregate over the run: {s['aggregate_download_mb_s']:.1f} MB/s",
    ]
    if s["top_download_errors"]:
        lines += ["- Most common download errors:"]
        lines += [f"  - {n:,}× `{e}`" for e, n in s["top_download_errors"].items()]
    lines += [
        "",
        "## Storage (outputs were deleted)",
        "",
        f"- Keeping everything would have used **{s['output_gb_if_kept']:.1f} GB** "
        f"({s['output_mb_per_photo']:.1f} MB per processed photo, including the photo).",
        "",
        "## Per node",
        "",
        md_table(per_node, ["node", "shards", "photos", "gpu_hours_allocated", "gpu_hours_busy",
                            "busy_pct_of_allocated", "download_wait_hours", "download_gb",
                            "gpu_util_mean_pct"],
                 {"gpu_hours_allocated": ".2f", "gpu_hours_busy": ".2f",
                  "download_wait_hours": ".2f", "download_gb": ".1f"}),
        "## Per shard",
        "",
        md_table(per_shard, ["shard", "node", "state", "photos", "ok", "download_failed",
                             "allocated_s", "model_load_s", "loop_s", "photos_per_s",
                             "gpu_busy_pct_of_loop", "download_wait_pct_of_loop",
                             "download_mb_s_median", "download_s_p95", "gpu_util_mean_pct",
                             "gpu_mem_peak_gb", "max_rss_gb"],
                 {"photos_per_s": ".2f", "download_mb_s_median": ".2f", "download_s_p95": ".2f",
                  "gpu_mem_peak_gb": ".1f", "max_rss_gb": ".1f", "allocated_s": ".0f",
                  "loop_s": ".0f"}),
        "Columns: `allocated_s` = SLURM elapsed; `gpu_busy_pct_of_loop` = sum of forward-pass "
        "time / loop time; `download_wait_pct_of_loop` = time the GPU sat idle waiting for "
        "downloads; `gpu_util_mean_pct` = nvidia-smi average.",
        "",
        "Raw data: `raw/` (per-photo CSVs, shard JSONs, telemetry), `logs/`, `sacct.txt`.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
