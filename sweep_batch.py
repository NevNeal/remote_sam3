#!/usr/bin/env python3
"""
Find the batch size that saturates one GPU, before spending an hour at the wrong one.

`segment.py --batch-size N` is a guess until something measures it. This runs the
same forward pass at a range of batch sizes over the same pool of images and
reports images per second and peak VRAM for each, so the hour-long run can be
launched at the size that actually wins rather than at a round number.

Only the GPU is measured. The images are decoded once, up front, and reused for
every batch size, and nothing is written: the number here is the ceiling a
perfect reader and a perfect writer would let `segment.py` approach, not what it
will reach. Compare it with the images/s that `slurm/b200.sbatch` reports to see
how much of the GPU the rest of the pipeline is costing you.

    python sweep_batch.py --local-index data/local_image_paths.parquet \\
        --image-root /home/neal.nevyn/blue_guralnick/share \\
        --prompt flower --batch-sizes 1,2,4,8,16,32,64 --pool 128

An out-of-memory batch is reported as OOM and the sweep carries on: that is the
edge of the GPU, and finding it is the point.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import Sam3Model, Sam3Processor

from segment import _forward_batch, fetch_local, load_local_photos

DTYPES = {"fp32": False, "bf16": True}


def load_pool(args):
    """Decode `--pool` images once, so every batch size is timed on the same work."""
    photos = load_local_photos(args.local_index, args.image_root,
                               args.pool * args.oversample, 0, 1)
    rows = photos.to_dict("records")
    t0 = time.perf_counter()
    images = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _, image, error, _ in pool.map(fetch_local, rows):
            if image is not None:
                images.append(image)
            if len(images) >= args.pool:
                break
    if not images:
        raise SystemExit(f"could not read any images under {args.image_root}")
    megapixels = sum(w * h for w, h in (i.size for i in images)) / 1e6 / len(images)
    print(f"pool       : {len(images)} images decoded in {time.perf_counter() - t0:.1f}s"
          f", {megapixels:.1f} MP average")
    return images


def time_one(model, processor, images, batch_size, args, bf16):
    """images/s and peak VRAM for one (batch size, dtype), or an OOM marker."""
    batches = [images[i:i + batch_size]
               for i in range(0, len(images) - batch_size + 1, batch_size)]
    if not batches:
        return None

    try:
        # One warm-up batch: the first call at a new shape allocates workspace and
        # picks kernels, and timing that instead of the steady state would make
        # every batch size look worse than the one before it.
        _forward_batch(model, processor, batches[0], args.prompt,
                       args.min_score, "cuda", bf16)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        started = time.perf_counter()
        for batch in batches:
            _forward_batch(model, processor, batch, args.prompt,
                           args.min_score, "cuda", bf16)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return {"batch_size": batch_size, "dtype": "bf16" if bf16 else "fp32",
                "error": "OOM", "detail": str(exc).split("\n")[0]}
    except Exception as exc:
        torch.cuda.empty_cache()
        return {"batch_size": batch_size, "dtype": "bf16" if bf16 else "fp32",
                "error": type(exc).__name__, "detail": str(exc).split("\n")[0]}

    done = len(batches) * batch_size
    return {
        "batch_size": batch_size,
        "dtype": "bf16" if bf16 else "fp32",
        "batches": len(batches),
        "images": done,
        "seconds": round(elapsed, 3),
        "images_per_s": round(done / elapsed, 2),
        "s_per_batch": round(elapsed / len(batches), 4),
        "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 1e9, 1),
    }


def render(results, total_vram_gb):
    rows = [f"{'batch':>6} {'dtype':>6} {'images/s':>9} {'s/batch':>8} "
            f"{'peak VRAM':>10} {'of GPU':>7}",
            "-" * 52]
    for r in results:
        if r.get("error"):
            rows.append(f"{r['batch_size']:>6} {r['dtype']:>6} {r['error']:>9}"
                        f"   {r.get('detail', '')[:28]}")
            continue
        share = (f"{100 * r['peak_vram_gb'] / total_vram_gb:.0f}%"
                 if total_vram_gb else "-")
        rows.append(f"{r['batch_size']:>6} {r['dtype']:>6} {r['images_per_s']:>9.2f}"
                    f" {r['s_per_batch']:>8.3f} {r['peak_vram_gb']:>9.1f}G {share:>7}")
    return "\n".join(rows)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible — this measures a GPU, so there is nothing to do")

    props = torch.cuda.get_device_properties(0)
    total_vram_gb = props.total_memory / 1e9
    print(f"gpu        : {props.name} ({total_vram_gb:.0f} GB)")

    images = load_pool(args)

    t0 = time.perf_counter()
    model = Sam3Model.from_pretrained("facebook/sam3").to("cuda").eval()
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    print(f"model load : {time.perf_counter() - t0:.1f}s")

    results = []
    for dtype in args.dtypes:
        for batch_size in args.batch_sizes:
            if batch_size > len(images):
                continue
            result = time_one(model, processor, images, batch_size, args, DTYPES[dtype])
            if result is None:
                continue
            results.append(result)
            note = result.get("error") or f"{result['images_per_s']:.2f} images/s"
            print(f"  {dtype} batch {batch_size:>3}: {note}")

    print()
    print(render(results, total_vram_gb))

    ok = [r for r in results if not r.get("error")]
    if ok:
        best = max(ok, key=lambda r: r["images_per_s"])
        baseline = next((r for r in ok if r["batch_size"] == 1
                         and r["dtype"] == best["dtype"]), None)
        speedup = (f", {best['images_per_s'] / baseline['images_per_s']:.1f}x"
                   f" batch 1 at the same dtype" if baseline else "")
        print(f"\nbest       : batch {best['batch_size']} {best['dtype']},"
              f" {best['images_per_s']:.2f} images/s{speedup}")
        print(f"             BATCH_SIZE={best['batch_size']}"
              + (" BF16=1" if best["dtype"] == "bf16" else " BF16=")
              + " sbatch slurm/b200.sbatch")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "gpu": props.name, "total_vram_gb": round(total_vram_gb, 1),
            "pool": len(images), "prompt": args.prompt, "min_score": args.min_score,
            "results": results,
        }, indent=2))
        print(f"wrote      : {args.out}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local-index", required=True,
                        help="Parquet of images already on disk (local_image_paths.parquet)")
    parser.add_argument("--image-root", required=True,
                        help="Directory the index's leading 'data/' maps to")
    parser.add_argument("--prompt", default="flower", help="SAM3 text prompt")
    parser.add_argument("--min-score", type=float, default=0.9)
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32,64",
                        help="Comma-separated batch sizes to time")
    parser.add_argument("--dtypes", default="fp32,bf16",
                        help="Comma-separated: fp32, bf16, or both")
    parser.add_argument("--pool", type=int, default=128,
                        help="Images to decode once and reuse for every batch size. "
                             "Must be at least the largest batch size.")
    parser.add_argument("--oversample", type=int, default=2,
                        help="Index rows to read per pooled image, so missing files "
                             "on disk still leave a full pool")
    parser.add_argument("--workers", type=int, default=16, help="Decode threads")
    parser.add_argument("--out", default=None, help="Write the raw numbers here as JSON")
    args = parser.parse_args()
    args.batch_sizes = sorted({int(b) for b in args.batch_sizes.split(",") if b.strip()})
    args.dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    unknown = set(args.dtypes) - set(DTYPES)
    if unknown:
        parser.error(f"unknown dtype(s): {', '.join(sorted(unknown))}")
    return args


if __name__ == "__main__":
    main()
