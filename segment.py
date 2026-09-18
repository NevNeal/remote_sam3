#!/usr/bin/env python3
"""
Run SAM3 text-prompted instance segmentation over every research-grade iNaturalist
photo for one taxon.

Reads photo metadata from the parquet index (see build_index.py), downloads each
image from iNat's open-data S3 bucket, runs one SAM3 forward pass per image, and
writes masks, overlays, transparent cut-outs and a results CSV.

Runs on one GPU. To use many GPUs, run many copies with different --shard values;
see README "What a shard is". Each shard writes its own CSV, so shards never
collide and each one resumes independently.

    python segment.py --taxon-id 160559 --prompt flower \\
        --parquet /blue/GROUP/USER/data/inat_photos.parquet \\
        --out /blue/GROUP/USER/results/160559_flower \\
        --shard 0 --num-shards 60

Output layout (shared by all shards of a run):

    <out>/images/batch_00001/Genus_species_<photo_id>.jpg
    <out>/masks/batch_00001/Genus_species_<photo_id>_instance_0.npy
    <out>/overlays/batch_00001/Genus_species_<photo_id>_overlay.png
    <out>/segments/batch_00001/Genus_species_<photo_id>_segment_0.png
    <out>/results_shard_000.csv
    <out>/errors_shard_000.txt
"""

import argparse
import csv
import os
import platform
import re
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from transformers import Sam3Model, Sam3Processor
from urllib3.util.retry import Retry

PHOTO_URL = "https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/original.{ext}"

# Images per batch_NNNNN subfolder. Keeps directory listings survivable - a big
# taxon is hundreds of thousands of files, and Lustre does not enjoy one flat dir.
BATCH_SIZE = 1000

# How many downloaded images to hold in memory before segmenting them. Bigger
# means the GPU waits less at chunk boundaries and more RAM is used.
CHUNK = 64

MASK_THRESHOLD = 0.80        # binarisation cutoff inside SAM3's post-processing
HTTP_TIMEOUT = 60

OVERLAY_ALPHA = 95
OVERLAY_COLOR = (255, 0, 0)
BOX_COLOR = (255, 255, 0)
BOX_WIDTH = 4

CSV_COLUMNS = [
    "photo_id", "taxon_id", "taxon_name", "quality_grade", "latitude", "longitude",
    "photo_url", "row_index", "shard", "batch", "status", "num_masks",
    "image_path", "overlay_path", "mask_paths", "segment_paths",
    "mask_scores", "mask_avg_rgb", "seconds", "error",
]


# -- Stage 1: which photos are mine? ------------------------------------------

def load_photos(parquet, taxon_id, limit, shard, num_shards):
    """Read this shard's photo rows for one taxon out of the parquet index.

    row_index is assigned over the FULL taxon before sharding, so filenames and
    batch folders are identical no matter how many shards you run.
    """
    t0 = time.time()
    df = pd.read_parquet(
        parquet,
        columns=["photo_id", "extension", "taxon_id", "taxon_name",
                 "quality_grade", "latitude", "longitude"],
        filters=[("taxon_id", "==", taxon_id)],
    ).sort_values("photo_id").reset_index(drop=True)
    print(f"parquet    : {len(df):,} photos for taxon {taxon_id} in {time.time() - t0:.1f}s")

    if df.empty:
        raise SystemExit(f"No research-grade photos for taxon_id={taxon_id}.")
    if limit:
        df = df.head(limit).copy()

    df["row_index"] = np.arange(len(df))
    df["photo_url"] = [
        PHOTO_URL.format(photo_id=p, ext=e)
        for p, e in zip(df["photo_id"], df["extension"])
    ]
    df["stem"] = [
        f"{_genus_species(n)}_{int(p)}" for n, p in zip(df["taxon_name"], df["photo_id"])
    ]
    df["batch"] = [f"batch_{i // BATCH_SIZE + 1:05d}" for i in df["row_index"]]

    if num_shards > 1:
        total = len(df)
        df = df[df["row_index"] % num_shards == shard].copy()
        print(f"shard      : {shard} of {num_shards} -> {len(df):,} of {total:,} photos")
        if df.empty:
            raise SystemExit(f"Shard {shard} got no rows (taxon smaller than {num_shards}).")

    return df


def _genus_species(name):
    """'Symphyotrichum novae-angliae' -> 'Symphyotrichum_novaeangliae'."""
    parts = str(name or "").strip().split()
    if len(parts) >= 2:
        return f"{_clean(parts[0].capitalize())}_{_clean(parts[1].lower())}"
    return _clean(name) or "Unknown_unknown"


def _clean(value):
    return re.sub(r"[^A-Za-z0-9_]+", "", re.sub(r"\s+", "_", str(value or "").strip()))


# -- Stage 2: downloading -----------------------------------------------------

def make_session(workers):
    session = requests.Session()
    retry = Retry(total=8, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",), raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry, pool_connections=workers, pool_maxsize=workers)
    session.mount("https://", adapter)
    return session


def fetch(session, row, image_path):
    """Return (row, PIL image or None, error string or None).

    An already-downloaded image is reused, which is what makes a rerun cheap.
    """
    if image_path.exists() and image_path.stat().st_size > 0:
        try:
            return row, Image.open(image_path).convert("RGB"), None
        except Exception:
            pass  # truncated from a killed job - fall through and re-download

    try:
        response = session.get(row["photo_url"], timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        data = response.content
        image_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = image_path.with_suffix(image_path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(image_path)
        return row, Image.open(BytesIO(data)).convert("RGB"), None
    except Exception as exc:
        return row, None, str(exc)


# -- Stage 3: mask bookkeeping ------------------------------------------------

def to_binary(mask, size):
    """SAM3 mask (tensor or array, maybe not image-sized) -> uint8 0/1 at `size`."""
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = (np.squeeze(np.asarray(mask)) > 0).astype(np.uint8)
    if (mask.shape[1], mask.shape[0]) != size:
        resized = Image.fromarray(mask * 255).resize(size, Image.NEAREST)
        mask = (np.asarray(resized) > 0).astype(np.uint8)
    return mask


def bbox(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def avg_rgb(image, mask):
    """Mean colour inside one mask - the whole point for flower-colour work."""
    pixels = np.asarray(image)[mask > 0]
    if not len(pixels):
        return ""
    r, g, b = pixels.mean(axis=0).round().astype(int)
    return f"{r}|{g}|{b}"


def save_cutout(image, mask, path):
    """Transparent PNG of one instance, cropped to its bounding box."""
    box = bbox(mask)
    if box is None:
        return False
    x1, y1, x2, y2 = box
    rgba = np.asarray(image.convert("RGBA")).copy()
    rgba[..., 3] = np.where(mask > 0, 255, 0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba[y1:y2 + 1, x1:x2 + 1], mode="RGBA").save(path)
    return True


def save_overlay(image, masks, scores, prompt, path):
    """One annotated PNG: red masks, yellow boxes, confidence labels."""
    base = image.convert("RGBA")
    mask_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    box_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(box_layer)
    font = _font(max(16, int(min(base.size) * 0.025)))

    for mask, score in zip(masks, scores):
        alpha = Image.fromarray(mask * OVERLAY_ALPHA, mode="L")
        tint = Image.new("RGBA", base.size, OVERLAY_COLOR + (0,))
        tint.putalpha(alpha)
        mask_layer = Image.alpha_composite(mask_layer, tint)

        box = bbox(mask)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        for offset in range(BOX_WIDTH):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                           outline=BOX_COLOR + (255,))
        _label(draw, x1, max(0, y1 - font.size - 12), f"{prompt} {score:.3f}", font)

    path.parent.mkdir(parents=True, exist_ok=True)
    flat = Image.alpha_composite(Image.alpha_composite(base, mask_layer), box_layer)
    flat.convert("RGB").save(path, quality=95)


def _font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                 "C:/Windows/Fonts/arialbd.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _label(draw, x, y, text, font):
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    draw.rectangle([left - 4, top - 4, right + 4, bottom + 4], fill=(0, 0, 0, 180))
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)


# -- Stage 4: the run ---------------------------------------------------------

class Results:
    """Append-only per-shard CSV. One row per photo, written as we go, so a job
    killed at its walltime loses nothing but the image in flight."""

    def __init__(self, path, err_path):
        self.path = path
        self.err_path = err_path
        if not path.exists() or path.stat().st_size == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(CSV_COLUMNS)

    def already_done(self):
        """photo_ids this shard has already recorded, for resume."""
        try:
            done = pd.read_csv(self.path, usecols=["photo_id"])
            return set(done["photo_id"].dropna().astype("int64"))
        except (pd.errors.EmptyDataError, ValueError, KeyError):
            return set()

    def write(self, **fields):
        with open(self.path, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([fields.get(c, "") for c in CSV_COLUMNS])

    def note_error(self, message):
        with open(self.err_path, "a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")


def main():
    args = parse_args()
    out = Path(args.out).expanduser().resolve()
    results = Results(out / f"results_shard_{args.shard:03d}.csv",
                      out / f"errors_shard_{args.shard:03d}.txt")

    print(f"host       : {platform.node()}")
    print(f"slurm job  : {os.environ.get('SLURM_JOB_ID', '-')}"
          f" task {os.environ.get('SLURM_ARRAY_TASK_ID', '-')}")

    photos = load_photos(args.parquet, args.taxon_id, args.limit,
                         args.shard, args.num_shards)

    done = results.already_done()
    todo = photos[~photos["photo_id"].isin(done)]
    print(f"prompt     : '{args.prompt}'  (keep detections >= {args.min_score})")
    print(f"output     : {out}")
    print(f"resume     : {len(done):,} already done, {len(todo):,} to go")
    if todo.empty:
        print("nothing to do.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"gpu        : {props.name} ({props.total_memory / 1e9:.0f} GB)")
    else:
        print("gpu        : NONE - running on CPU, this will be very slow")

    model = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    session = make_session(args.workers)

    tally = {"ok": 0, "no_detections": 0, "download_failed": 0, "segment_failed": 0}
    rows = todo.to_dict("records")
    progress = tqdm(total=len(rows), desc=f"shard {args.shard}", dynamic_ncols=True)

    # Download a chunk in parallel, then segment that chunk one image at a time.
    # The GPU idles briefly at each chunk boundary; in exchange the control flow
    # is obvious and memory is bounded. With workers=16 and CHUNK=64 the stall is
    # a fraction of a second against ~60s of GPU work per chunk.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(rows), CHUNK):
            chunk = rows[start:start + CHUNK]
            fetched = pool.map(
                lambda row: fetch(session, row, _image_path(out, row)), chunk
            )
            for row, image, error in fetched:
                _handle(row, image, error, out, model, processor,
                        device, args, results, tally, progress)

    progress.close()
    print(f"\nshard {args.shard} done: " +
          "  ".join(f"{k}={v:,}" for k, v in tally.items()))
    print(f"results: {results.path}")


def _image_path(out, row):
    ext = str(row["extension"]).lower().lstrip(".")
    ext = ext if ext in ("jpg", "jpeg", "png") else "jpg"
    return out / "images" / row["batch"] / f"{row['stem']}.{ext}"


def _handle(row, image, error, out, model, processor, device,
            args, results, tally, progress):
    """Segment one image and record exactly one CSV row for it."""
    common = {
        "photo_id": row["photo_id"], "taxon_id": row["taxon_id"],
        "taxon_name": row["taxon_name"], "quality_grade": row["quality_grade"],
        "latitude": row["latitude"], "longitude": row["longitude"],
        "photo_url": row["photo_url"], "row_index": row["row_index"],
        "shard": args.shard, "batch": row["batch"],
        "image_path": str(_image_path(out, row).relative_to(out)),
    }

    if image is None:
        tally["download_failed"] += 1
        results.write(status="download_failed", num_masks=0, error=error, **common)
        results.note_error(f"[download] photo_id={row['photo_id']} {error}")
        progress.update(1)
        return

    started = time.time()
    try:
        masks, scores = segment(model, processor, image, args.prompt,
                                args.min_score, device, args.bf16)
    except Exception as exc:
        tally["segment_failed"] += 1
        results.write(status="segment_failed", num_masks=0, error=str(exc),
                      seconds=f"{time.time() - started:.2f}", **common)
        results.note_error(f"[segment] photo_id={row['photo_id']} {exc}")
        progress.update(1)
        return

    stem = row["stem"]
    mask_paths, segment_paths, rgbs = [], [], []
    for i, mask in enumerate(masks):
        mask_path = out / "masks" / row["batch"] / f"{stem}_instance_{i}.npy"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(mask_path, mask)
        mask_paths.append(str(mask_path.relative_to(out)))

        cutout = out / "segments" / row["batch"] / f"{stem}_segment_{i}.png"
        if save_cutout(image, mask, cutout):
            segment_paths.append(str(cutout.relative_to(out)))

        rgbs.append(avg_rgb(image, mask))

    overlay_rel = ""
    if masks:
        overlay = out / "overlays" / row["batch"] / f"{stem}_overlay.png"
        save_overlay(image, masks, scores, args.prompt, overlay)
        overlay_rel = str(overlay.relative_to(out))
        tally["ok"] += 1
    else:
        tally["no_detections"] += 1

    results.write(
        status="ok", num_masks=len(masks),
        overlay_path=overlay_rel,
        mask_paths=";".join(mask_paths),
        segment_paths=";".join(segment_paths),
        mask_scores=",".join(f"{s:.4f}" for s in scores),
        mask_avg_rgb=";".join(rgbs),
        seconds=f"{time.time() - started:.2f}",
        **common,
    )
    progress.update(1)


def segment(model, processor, image, prompt, min_score, device, bf16):
    """One SAM3 forward pass. Returns (masks, scores) sorted most-confident first."""
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        if bf16 and device == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

    detections = processor.post_process_instance_segmentation(
        outputs,
        threshold=min_score,
        mask_threshold=MASK_THRESHOLD,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    raw_masks = detections.get("masks")
    raw_scores = detections.get("scores")
    if raw_masks is None or raw_scores is None or len(raw_scores) == 0:
        return [], []

    scores = [float(s) for s in raw_scores]
    pairs = sorted(zip(raw_masks, scores), key=lambda pair: pair[1], reverse=True)

    masks, kept = [], []
    for mask, score in pairs:
        if score < min_score:
            continue
        binary = to_binary(mask, image.size)
        if binary.any():
            masks.append(binary)
            kept.append(score)
    return masks, kept


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--taxon-id", type=int, required=True, help="iNaturalist taxon ID")
    parser.add_argument("--parquet", required=True, help="Path to inat_photos.parquet")
    parser.add_argument("--out", required=True, help="Output directory (on /blue)")
    parser.add_argument("--prompt", default="flower", help="SAM3 text prompt")
    parser.add_argument("--min-score", type=float, default=0.9,
                        help="Discard detections below this confidence")
    parser.add_argument("--shard", type=int, default=0, help="This worker's shard index")
    parser.add_argument("--num-shards", type=int, default=1, help="Total shards in the run")
    parser.add_argument("--workers", type=int, default=16, help="Parallel image downloads")
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only the first N photos of the taxon (for testing)")
    parser.add_argument("--bf16", action="store_true",
                        help="Run inference in bfloat16 (faster; slightly different masks)")
    args = parser.parse_args()
    args.num_shards = max(1, args.num_shards)
    return args


if __name__ == "__main__":
    main()
