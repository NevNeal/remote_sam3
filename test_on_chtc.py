"""
test_on_chtc.py — runs inside the Apptainer container on a CHTC GPU compute node.

Performs four timed, independently-failing tests and writes a JSON results
file so analyze_results.py can produce a single summary later.

Usage (inside the container):
    python test_on_chtc.py --taxon-id 591507 --output-dir test_output [--limit 1]

Tests:
    1. environment   : Python, torch, CUDA, GPU info, disk, memory
    2. network       : iNaturalist API reachable; one small metadata fetch
    3. huggingface   : SAM3 model config is reachable / token works
    4. pipeline      : run segmentation_pipeline.py end-to-end with --limit N

Each test records duration, status (ok | failed | skipped), error string,
and arbitrary detail fields. Steps after a hard-dependency failure are
skipped (e.g. no network -> no HF check -> no pipeline run).
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path


def now() -> float:
    return time.time()


def run_test(name, results, fn, *, depends_on=None):
    if depends_on:
        prior = next((r for r in results if r["name"] == depends_on), None)
        if prior is None or prior["status"] != "ok":
            results.append({
                "name": name,
                "status": "skipped",
                "duration_sec": 0.0,
                "error": f"dependency {depends_on!r} did not succeed",
                "detail": {},
            })
            return

    start = now()
    detail = {}
    try:
        fn(detail)
        results.append({
            "name": name,
            "status": "ok",
            "duration_sec": round(now() - start, 3),
            "error": "",
            "detail": detail,
        })
    except Exception as error:
        results.append({
            "name": name,
            "status": "failed",
            "duration_sec": round(now() - start, 3),
            "error": f"{type(error).__name__}: {error}",
            "detail": detail,
            "traceback": traceback.format_exc(),
        })


def test_environment(detail):
    import torch
    detail["python_version"] = sys.version.split()[0]
    detail["hostname"] = socket.gethostname()
    detail["torch_version"] = torch.__version__
    detail["cuda_available"] = torch.cuda.is_available()
    detail["cuda_version"] = torch.version.cuda
    detail["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    detail["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    if torch.cuda.is_available():
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i,
                "name": props.name,
                "memory_gb": round(props.total_memory / 1e9, 2),
                "capability": f"{props.major}.{props.minor}",
            })
        detail["gpus"] = gpus
    else:
        detail["gpus"] = []
        raise RuntimeError("CUDA is not available — job landed on a non-GPU node or driver mismatch")

    disk = shutil.disk_usage(".")
    detail["cwd_free_gb"] = round(disk.free / 1e9, 2)
    detail["cwd_total_gb"] = round(disk.total / 1e9, 2)


def test_network(detail):
    import requests
    url = "https://api.inaturalist.org/v1/observations"
    t0 = now()
    response = requests.get(
        url,
        params={"taxon_id": 591507, "quality_grade": "research", "per_page": 1},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    detail["inat_status_code"] = response.status_code
    detail["inat_round_trip_sec"] = round(now() - t0, 3)
    detail["inat_total_results"] = data.get("total_results")


def test_huggingface(detail):
    from huggingface_hub import HfApi
    api = HfApi()
    detail["hf_token_present"] = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    detail["hf_home"] = os.environ.get("HF_HOME", "")
    info = api.model_info("facebook/sam3", timeout=30)
    detail["sam3_gated"] = getattr(info, "gated", "unknown")
    detail["sam3_sha"] = info.sha


def test_pipeline(detail, taxon_id, output_dir, limit, script_path):
    cmd = [
        sys.executable,
        str(script_path),
        str(taxon_id),
        output_dir,
        "--limit",
        str(limit),
    ]
    detail["command"] = " ".join(cmd)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    detail["returncode"] = proc.returncode
    detail["stdout_tail"] = proc.stdout[-2000:]
    detail["stderr_tail"] = proc.stderr[-2000:]

    if proc.returncode != 0:
        raise RuntimeError(f"segmentation_pipeline.py exited with code {proc.returncode}")

    base = Path(script_path).parent / output_dir
    detail["output_base"] = str(base)
    detail["output_exists"] = base.is_dir()
    if base.is_dir():
        detail["files"] = {
            "images": sum(1 for p in base.glob("images/**/*") if p.is_file()),
            "masks": sum(1 for _ in base.glob("masks/**/*.npy")),
            "overlays": sum(1 for _ in base.glob("overlays/**/*.png")),
        }
        detail["mask_summary_exists"] = (base / "mask_summary.csv").exists()


def main():
    parser = argparse.ArgumentParser(description="CHTC GPU node test harness")
    parser.add_argument("--taxon-id", type=int, default=591507)
    parser.add_argument("--output-dir", default="test_output")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--results-file", default="test_results.json")
    parser.add_argument("--pipeline-script", default=None,
                        help="Path to segmentation_pipeline.py (default: next to this script)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    pipeline_script = Path(args.pipeline_script) if args.pipeline_script else script_dir / "segmentation_pipeline.py"

    results = []
    started = now()

    run_test("environment", results, test_environment)
    run_test("network", results, test_network, depends_on="environment")
    run_test("huggingface", results, test_huggingface, depends_on="network")
    run_test(
        "pipeline",
        results,
        lambda d: test_pipeline(d, args.taxon_id, args.output_dir, args.limit, pipeline_script),
        depends_on="huggingface",
    )

    summary = {
        "started_at": started,
        "finished_at": now(),
        "total_duration_sec": round(now() - started, 3),
        "args": vars(args),
        "results": results,
        "all_ok": all(r["status"] == "ok" for r in results),
    }

    with open(args.results_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== TEST SUMMARY ===")
    for r in results:
        marker = {"ok": "PASS", "failed": "FAIL", "skipped": "SKIP"}[r["status"]]
        print(f"  [{marker}] {r['name']:14s} {r['duration_sec']:>7.2f}s  {r['error']}")
    print(f"\nResults written to: {args.results_file}")
    sys.exit(0 if summary["all_ok"] else 1)


if __name__ == "__main__":
    main()
