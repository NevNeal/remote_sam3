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
        --parquet data/inat_photos.parquet \\
        --out results/160559_flower \\
        --shard 0 --num-shards 60

Images that are already on /blue are segmented in place instead, with no taxon
and no S3 (see load_local_photos):

    python segment.py --local-index data/local_image_paths.parquet \\
        --image-root /home/neal.nevyn/blue_guralnick/share \\
        --out results/local_flower --prompt flower \\
        --shard 0 --num-shards 5

One forward pass per image leaves a big GPU mostly idle. --batch-size pushes N
images through together, which is what makes a B200 worth asking for, and
--max-seconds stops the shard on a wall-clock budget instead of at the end of
the list, for timed throughput tests:

    python segment.py --local-index data/local_image_paths.parquet \\
        --image-root /home/neal.nevyn/blue_guralnick/share \\
        --out results/b200_flower --prompt flower --bf16 \\
        --batch-size 32 --save-workers 16 --max-seconds 3600

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
import json
import os
import platform
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from itertools import islice
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

# How deep the pipeline runs, as multiples of the batch size: how many photos may
# sit decoded in memory ahead of the GPU, and how many may be waiting to be
# written behind it. Both are bounded because every photo in flight holds a
# full-resolution image, and a B200 batch of 64 is a lot of megapixels.
PREFETCH_BATCHES = 3
SAVE_BATCHES = 4

# The local index records every path as .webp, but what is actually on disk may
# have kept its original extension. Try the recorded name, then these.
LOCAL_EXTS = ("webp", "jpg", "jpeg", "png", "JPG", "JPEG", "PNG", "gif")

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
    # Timing, per photo. The download runs in a worker thread and overlaps the
    # GPU work; wait_s is how long the main loop actually sat waiting for it,
    # i.e. GPU idle time caused by downloads. prep/infer/post/save split the
    # main-loop work; infer_s is bracketed by cuda.synchronize so it is true
    # GPU time, not kernel-launch time. seconds = prep + infer + post + save.
    "host", "gpu", "done_at", "width", "height",
    "download_s", "download_bytes", "download_mbps", "reused_image",
    "wait_s", "prep_s", "infer_s", "post_s", "save_s", "output_bytes",
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

    # iNat's own dumps contain ~0.1% byte-identical duplicate photo rows, so the
    # index inherits them. Left in place they are worse than merely wasteful: the
    # two copies get different row_index values, land in DIFFERENT shards, and
    # those two shards then segment the same photo and write the same mask and
    # overlay filenames concurrently. Dedupe before row_index is assigned.
    duplicates = int(df["photo_id"].duplicated().sum())
    if duplicates:
        df = df.drop_duplicates(subset=["photo_id"]).reset_index(drop=True)
        print(f"           : dropped {duplicates:,} duplicate photo_id rows")

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


def load_local_photos(index, image_root, limit, shard, num_shards):
    """Read this shard's rows out of an index of images already on disk.

    Same contract as load_photos and the same interleaved sharding, minus the
    network: the index's `file_name` is a path whose leading `data/` is replaced
    by `image_root`, and the image is read where it already sits on /blue. There
    is no taxon filter — the index IS the selection (local_image_paths.parquet
    is the flowering rows of the annotation CSV).
    """
    t0 = time.time()
    df = pd.read_parquet(index, columns=["photo_id", "taxon_id", "scientific_name",
                                         "reproductive_condition", "file_name"])
    df = df.sort_values("photo_id").reset_index(drop=True)
    print(f"index      : {len(df):,} local photos from {index} in {time.time() - t0:.1f}s")

    if df.empty:
        raise SystemExit(f"No rows in {index}.")

    duplicates = int(df["photo_id"].duplicated().sum())
    if duplicates:
        df = df.drop_duplicates(subset=["photo_id"]).reset_index(drop=True)
        print(f"           : dropped {duplicates:,} duplicate photo_id rows")

    if limit:
        df = df.head(limit).copy()

    df["row_index"] = np.arange(len(df))
    df["taxon_name"] = df["scientific_name"]
    # Columns the CSV and the report expect but a local run has no source for.
    df["quality_grade"] = ""
    df["latitude"] = ""
    df["longitude"] = ""

    root = Path(image_root).expanduser()
    paths = [str(root / re.sub(r"^data/", "", str(f))) for f in df["file_name"]]
    df["local_path"] = paths
    df["photo_url"] = paths          # recorded in the CSV as the image's origin
    df["extension"] = [Path(f).suffix.lstrip(".").lower() for f in df["file_name"]]
    df["stem"] = [
        f"{_genus_species(n)}_{int(p)}" for n, p in zip(df["taxon_name"], df["photo_id"])
    ]
    df["batch"] = [f"batch_{i // BATCH_SIZE + 1:05d}" for i in df["row_index"]]

    if num_shards > 1:
        total = len(df)
        df = df[df["row_index"] % num_shards == shard].copy()
        print(f"shard      : {shard} of {num_shards} -> {len(df):,} of {total:,} photos")
        if df.empty:
            raise SystemExit(f"Shard {shard} got no rows (index smaller than {num_shards}).")

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
    """Return (row, PIL image or None, error string or None, timing dict).

    An already-downloaded image is reused, which is what makes a rerun cheap.
    download_s covers the whole HTTP transfer including any retries.
    """
    if image_path.exists() and image_path.stat().st_size > 0:
        try:
            info = {"reused_image": 1, "download_bytes": image_path.stat().st_size}
            return row, Image.open(image_path).convert("RGB"), None, info
        except Exception:
            pass  # truncated from a killed job - fall through and re-download

    started = time.perf_counter()
    info = {"reused_image": 0}
    try:
        response = session.get(row["photo_url"], timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        data = response.content
        elapsed = time.perf_counter() - started
        info.update(download_s=round(elapsed, 4), download_bytes=len(data),
                    download_mbps=round(len(data) / 1e6 / elapsed, 3) if elapsed else "")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = image_path.with_suffix(image_path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(image_path)
        return row, Image.open(BytesIO(data)).convert("RGB"), None, info
    except Exception as exc:
        info["download_s"] = round(time.perf_counter() - started, 4)
        return row, None, str(exc), info


def resolve_local(path):
    """The index says .webp; the file on disk may have kept its original
    extension. Returns the path that exists, or None."""
    path = Path(path)
    if path.exists():
        return path
    for ext in LOCAL_EXTS:
        alternative = path.with_suffix(f".{ext}")
        if alternative.exists():
            return alternative
    return None


def fetch_local(row):
    """fetch()'s signature, for images already on /blue. Nothing is downloaded
    or copied — the file is opened where it lies, and never deleted."""
    started = time.perf_counter()
    info = {"reused_image": 1, "download_mbps": ""}
    path = resolve_local(row["local_path"])
    if path is None:
        info["download_s"] = round(time.perf_counter() - started, 4)
        return row, None, f"missing: {row['local_path']} (no known extension)", info
    try:
        image = Image.open(path).convert("RGB")
    except Exception as exc:
        info["download_s"] = round(time.perf_counter() - started, 4)
        return row, None, f"unreadable: {path}: {exc}", info
    row["local_path"] = str(path)          # the extension that actually existed
    info["download_bytes"] = path.stat().st_size
    info["download_s"] = round(time.perf_counter() - started, 4)
    return row, image, None, info


def _prefetch(rows, fetch_one, workers, depth):
    """Yield (row, image, error, info) in order, keeping `depth` reads in flight.

    Submitting every row at once would decode the whole shard into RAM, and
    fetching a fixed chunk and then segmenting it leaves the GPU idle at every
    chunk boundary. This holds the readers a constant distance ahead instead, so
    the next batch is already decoded when the last one leaves the GPU.

    info["wait_s"] is how long the consumer actually blocked on this photo — GPU
    idle time caused by reading, which is the number that says whether `workers`
    is high enough.
    """
    rows = iter(rows)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque(pool.submit(fetch_one, row) for row in islice(rows, depth))
        while pending:
            future = pending.popleft()
            nxt = next(rows, None)
            if nxt is not None:
                pending.append(pool.submit(fetch_one, nxt))
            waited = time.perf_counter()
            row, image, error, info = future.result()
            info["wait_s"] = round(time.perf_counter() - waited, 4)
            yield row, image, error, info


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
    killed at its walltime loses nothing but the image in flight.

    The writes are locked because the saver threads call them: two threads
    appending to one file would interleave half-written lines, which is the same
    corruption that gives every shard its own CSV in the first place."""

    def __init__(self, path, err_path):
        self.path = path
        self.err_path = err_path
        self.lock = threading.Lock()
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
        row = [fields.get(c, "") for c in CSV_COLUMNS]
        with self.lock, open(self.path, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)

    def note_error(self, message):
        with self.lock, open(self.err_path, "a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")


class Saver:
    """Writes each photo's outputs on a worker thread, and owns every piece of
    bookkeeping the threads share.

    Saving a photo is PNG encoding plus a dozen small writes: hundreds of
    milliseconds of CPU. One image per forward pass hides that behind the next
    read, but a B200 finishes a batch of 32 faster than one thread can write it,
    and the GPU ends up waiting on Pillow. `depth` bounds how many photos may be
    in flight, because each one holds its decoded image and its masks in memory
    until it has been written.

    workers=1 runs everything on the calling thread — exactly what the
    single-image path did before there was a Saver.
    """

    def __init__(self, workers, depth, results, tally, progress):
        self.results, self.tally, self.progress = results, tally, progress
        self.pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        self.slots = threading.Semaphore(depth)
        self.lock = threading.Lock()
        self.gpu_s = 0.0

    def submit(self, fn, *args):
        if self.pool is None:
            fn(*args)
            return
        self.slots.acquire()         # blocks the GPU loop once `depth` are queued
        self.pool.submit(self._run, fn, *args)

    def _run(self, fn, *args):
        try:
            fn(*args)
        except Exception as exc:
            # Deliberately no CSV row: a photo whose outputs did not get written
            # is not done, and the next run should pick it up again.
            self.results.note_error(f"[save] {exc}")
            self.bump("save_failed")
        finally:
            self.slots.release()

    def record(self, key, fields):
        """One CSV row, one tally bump, one tick of the bar, for one photo."""
        self.results.write(**fields)
        with self.lock:
            self.gpu_s += float(fields.get("infer_s") or 0)
        self.bump(key)

    def bump(self, key):
        with self.lock:
            self.tally[key] = self.tally.get(key, 0) + 1
            self.progress.update(1)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)


def main():
    args = parse_args()
    out = Path(args.out).expanduser().resolve()
    results = Results(out / f"results_shard_{args.shard:03d}.csv",
                      out / f"errors_shard_{args.shard:03d}.txt")

    shard_started = time.time()
    args.host = platform.node()
    print(f"host       : {args.host}")
    print(f"slurm job  : {os.environ.get('SLURM_JOB_ID', '-')}"
          f" task {os.environ.get('SLURM_ARRAY_TASK_ID', '-')}")

    t0 = time.perf_counter()
    if args.local_index:
        photos = load_local_photos(args.local_index, args.image_root, args.limit,
                                   args.shard, args.num_shards)
    else:
        photos = load_photos(args.parquet, args.taxon_id, args.limit,
                             args.shard, args.num_shards)
    parquet_s = time.perf_counter() - t0

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
        args.gpu = props.name
        args.gpu_uuid = f"GPU-{props.uuid}"
        print(f"gpu        : {props.name} ({props.total_memory / 1e9:.0f} GB)")
    else:
        args.gpu, args.gpu_uuid = "cpu", ""
        print("gpu        : NONE - running on CPU, this will be very slow")

    t0 = time.perf_counter()
    model = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model_load_s = time.perf_counter() - t0
    print(f"model load : {model_load_s:.1f}s")
    if args.local_index:
        fetch_one = fetch_local
        print(f"images     : read in place under {args.image_root}, never deleted")
    else:
        session = make_session(args.workers)
        fetch_one = lambda row: fetch(session, row, _image_path(out, row))

    tally = {"ok": 0, "no_detections": 0, "download_failed": 0,
             "segment_failed": 0, "save_failed": 0}
    rows = todo.to_dict("records")
    progress = tqdm(total=len(rows), desc=f"shard {args.shard}", dynamic_ncols=True)
    saver = Saver(args.save_workers, max(8, args.batch_size * SAVE_BATCHES),
                  results, tally, progress)
    depth = args.workers + args.batch_size * PREFETCH_BATCHES
    print(f"pipeline   : batch {args.batch_size}, {args.workers} readers {depth} deep,"
          f" {args.save_workers} savers"
          + (f", stop after {args.max_seconds:.0f}s" if args.max_seconds else ""))

    # Read ahead, segment a batch at a time, write behind. The three stages
    # overlap, so the GPU is only idle when the readers fall behind (wait_s) or
    # the savers back up (the Saver's semaphore).
    loop_started = time.perf_counter()
    deadline = loop_started + args.max_seconds if args.max_seconds else None
    stopped_early = False
    batch = []
    reader = _prefetch(rows, fetch_one, args.workers, depth)
    try:
        for row, image, error, info in reader:
            if image is None:
                _handle(row, None, error, info, out, model, processor, device,
                        args, results, tally, progress, saver)
            elif args.batch_size > 1:
                batch.append((row, image, info))
                if len(batch) >= args.batch_size:
                    _run_batch(batch, out, model, processor, device, args,
                               results, tally, progress, saver)
                    batch = []
            else:
                _handle(row, image, None, info, out, model, processor, device,
                        args, results, tally, progress, saver)
            if deadline is not None and time.perf_counter() >= deadline:
                stopped_early = True
                break
        # Whatever is already decoded gets segmented, deadline or not: it is one
        # more forward pass, and dropping it would mean re-reading those images.
        if batch:
            _run_batch(batch, out, model, processor, device, args,
                       results, tally, progress, saver)
    finally:
        # Stopping on the clock leaves reads in flight; close the generator so
        # its pool shuts down here rather than whenever it is collected.
        reader.close()
        # The GPU is done but photos are still being written, and their CSV rows
        # belong to this run, so loop_s has to include the drain.
        saver.close()
    loop_s = time.perf_counter() - loop_started

    progress.close()
    photos_done = sum(tally.values())
    print(f"\nshard {args.shard} done: " +
          "  ".join(f"{k}={v:,}" for k, v in tally.items()))
    if stopped_early:
        print(f"stopped    : --max-seconds {args.max_seconds:.0f} reached,"
              f" {len(rows) - photos_done:,} of this shard's photos untouched")
    if loop_s:
        print(f"loop       : {loop_s:.1f}s for {photos_done:,} photos"
              f" ({photos_done / loop_s:.2f} photos/s)")
        print(f"gpu        : {saver.gpu_s:.1f}s in forward passes"
              f" ({100 * saver.gpu_s / loop_s:.0f}% of the loop)")
    print(f"results: {results.path}")

    # Shard-level phases, for the benchmark report. Per-photo detail is in the CSV.
    timing = {
        "shard": args.shard, "num_shards": args.num_shards, "host": args.host,
        "gpu": args.gpu, "gpu_uuid": args.gpu_uuid, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "workers": args.workers, "batch_size": args.batch_size,
        "prefetch": depth, "save_workers": args.save_workers, "bf16": args.bf16,
        "discard_outputs": args.discard_outputs,
        "max_seconds": args.max_seconds, "stopped_early": stopped_early,
        "started_at": shard_started, "finished_at": time.time(),
        "parquet_s": round(parquet_s, 2), "model_load_s": round(model_load_s, 2),
        "loop_s": round(loop_s, 2), "gpu_infer_s": round(saver.gpu_s, 2),
        "photos": photos_done, "photos_todo": len(rows), "resumed_from": len(done),
        "tally": tally,
    }
    (out / f"timing_shard_{args.shard:03d}.json").write_text(json.dumps(timing, indent=2))


def _image_path(out, row):
    """Where this photo's image is. Local runs read it in place on /blue;
    otherwise it is the download target under <out>."""
    if row.get("local_path"):
        return Path(row["local_path"])
    ext = str(row["extension"]).lower().lstrip(".")
    ext = ext if ext in ("jpg", "jpeg", "png") else "jpg"
    return out / "images" / row["batch"] / f"{row['stem']}.{ext}"


def _run_batch(batch, out, model, processor, device, args, results, tally, progress, saver):
    """One forward pass over a whole batch, then one CSV row per photo in it."""
    outcomes = segment_batch(model, processor, [image for _, image, _ in batch],
                             args.prompt, args.min_score, device, args.bf16)
    for (row, image, info), outcome in zip(batch, outcomes):
        _handle(row, image, None, info, out, model, processor, device,
                args, results, tally, progress, saver, outcome=outcome)


def _handle(row, image, error, info, out, model, processor, device,
            args, results, tally, progress, saver, outcome=None):
    """Account for exactly one photo: segment it if nobody else has, then hand
    it to the saver.

    `outcome` is the (masks, scores, stages) a batched forward pass already
    produced for this photo, or the exception that batch raised for it. Left
    None, the photo is segmented here, one image per forward pass.
    """
    image_path = _image_path(out, row)
    common = {
        "photo_id": row["photo_id"], "taxon_id": row["taxon_id"],
        "taxon_name": row["taxon_name"], "quality_grade": row["quality_grade"],
        "latitude": row["latitude"], "longitude": row["longitude"],
        "photo_url": row["photo_url"], "row_index": row["row_index"],
        "shard": args.shard, "batch": row["batch"],
        "image_path": _relative(image_path, out),
        "host": args.host, "gpu": args.gpu,
        **info,
    }

    if image is None:
        saver.record("download_failed",
                     dict(status="download_failed", num_masks=0, error=error,
                          done_at=f"{time.time():.3f}", **common))
        results.note_error(f"[download] photo_id={row['photo_id']} {error}")
        return

    common.update(width=image.size[0], height=image.size[1])
    started = time.perf_counter()
    if outcome is None:
        try:
            outcome = segment(model, processor, image, args.prompt,
                              args.min_score, device, args.bf16)
        except Exception as exc:
            outcome = exc

    if isinstance(outcome, Exception):
        saver.record("segment_failed",
                     dict(status="segment_failed", num_masks=0, error=str(outcome),
                          seconds=f"{time.perf_counter() - started:.3f}",
                          done_at=f"{time.time():.3f}", **common))
        results.note_error(f"[segment] photo_id={row['photo_id']} {outcome}")
        if args.discard_outputs and not row.get("local_path"):
            _discard([image_path])
        return

    masks, scores, stages = outcome
    saver.submit(_save_photo, row, image, image_path, masks, scores, stages,
                 common, out, args, saver)


def _save_photo(row, image, image_path, masks, scores, stages, common, out, args, saver):
    """Write one photo's masks, cut-outs and overlay, then its CSV row.

    Runs on a Saver thread. Everything the GPU had to be present for is already
    done — this is PNG encoding and file writes, and the next batch should not
    be waiting behind it.
    """
    saving = time.perf_counter()
    stem = row["stem"]
    written = []
    mask_paths, segment_paths, rgbs = [], [], []
    for i, mask in enumerate(masks):
        mask_path = out / "masks" / row["batch"] / f"{stem}_instance_{i}.npy"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(mask_path, mask)
        mask_paths.append(str(mask_path.relative_to(out)))
        written.append(mask_path)

        cutout = out / "segments" / row["batch"] / f"{stem}_segment_{i}.png"
        if save_cutout(image, mask, cutout):
            segment_paths.append(str(cutout.relative_to(out)))
            written.append(cutout)

        rgbs.append(avg_rgb(image, mask))

    overlay_rel = ""
    status_key = "no_detections"
    if masks:
        overlay = out / "overlays" / row["batch"] / f"{stem}_overlay.png"
        save_overlay(image, masks, scores, args.prompt, overlay)
        overlay_rel = str(overlay.relative_to(out))
        written.append(overlay)
        status_key = "ok"
    save_s = time.perf_counter() - saving

    # What this photo would cost on disk if kept: the downloaded image plus
    # everything written for it. Measured before any discard.
    output_bytes = sum(p.stat().st_size for p in written + [image_path] if p.exists())
    if args.discard_outputs:
        # In local mode the image is the source data on /blue, not ours to delete.
        _discard(written if row.get("local_path") else written + [image_path])

    # prep + infer + post + save, as the CSV's header comment promises. Summed
    # from the stages rather than wall-clocked, so a batched photo gets its own
    # share and not the whole batch's.
    seconds = save_s + sum(float(stages[k]) for k in ("prep_s", "infer_s", "post_s"))

    saver.record(status_key, dict(
        status="ok", num_masks=len(masks),
        overlay_path=overlay_rel,
        mask_paths=";".join(mask_paths),
        segment_paths=";".join(segment_paths),
        mask_scores=",".join(f"{s:.4f}" for s in scores),
        mask_avg_rgb=";".join(rgbs),
        seconds=f"{seconds:.3f}",
        save_s=f"{save_s:.4f}", output_bytes=output_bytes,
        done_at=f"{time.time():.3f}",
        **stages, **common))


def _relative(path, out):
    """Path relative to the run directory, or absolute if it lives elsewhere
    (a local run's images do)."""
    try:
        return str(path.relative_to(out))
    except ValueError:
        return str(path)


def _discard(paths):
    """Benchmark mode: delete a photo's files as soon as they are measured."""
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def segment(model, processor, image, prompt, min_score, device, bf16):
    """One SAM3 forward pass over one image.

    Returns (masks, scores, stages): masks/scores sorted most-confident first,
    stages = {prep_s, infer_s, post_s} for the timing columns.
    """
    return _forward_batch(model, processor, [image], prompt, min_score, device, bf16)[0]


def segment_batch(model, processor, images, prompt, min_score, device, bf16):
    """SAM3 over a list of images in one forward pass.

    Returns one (masks, scores, stages) tuple per image, in the order given, or
    that image's exception in its slot.

    A batch that will not run is halved and retried rather than lost. How much
    VRAM a batch needs depends on the resolutions in it, so the largest batch
    that fits is not a constant, and one unlucky pair of 50-megapixel photos
    should cost two forward passes rather than a whole shard. Splitting also
    isolates a single corrupt image instead of failing the 31 beside it.
    """
    try:
        return _forward_batch(model, processor, images, prompt, min_score, device, bf16)
    except Exception as exc:
        if len(images) == 1:
            return [exc]
        if not getattr(segment_batch, "warned", False):
            segment_batch.warned = True
            print(f"\nbatch of {len(images)} failed, halving and retrying: {exc}")
            print("           if this keeps happening, --batch-size is too big for this"
                  " GPU, or the processor will not collate these image sizes")
        if device == "cuda":
            torch.cuda.empty_cache()
        half = len(images) // 2
        return (segment_batch(model, processor, images[:half],
                              prompt, min_score, device, bf16)
                + segment_batch(model, processor, images[half:],
                                prompt, min_score, device, bf16))


def _forward_batch(model, processor, images, prompt, min_score, device, bf16):
    """The forward pass itself, over a list of images. Raises on failure.

    prep_s and infer_s are each photo's share of the batch's preprocessing and
    forward pass: the batch is indivisible, so there is no per-photo number to
    report and the share is what keeps the CSV's columns summable. post_s is the
    batch's share of post-processing plus this photo's own mask thresholding,
    which genuinely is per-photo.
    """
    cuda = device == "cuda"
    n = len(images)

    t0 = time.perf_counter()
    inputs = _processor_call(processor, images, prompt).to(device)
    if cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    with torch.inference_mode():
        if bf16 and cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)
    if cuda:
        torch.cuda.synchronize()     # kernels are async; wait so this is real GPU time
    t2 = time.perf_counter()

    detections = processor.post_process_instance_segmentation(
        outputs,
        threshold=min_score,
        mask_threshold=MASK_THRESHOLD,
        target_sizes=inputs.get("original_sizes").tolist(),
    )
    if len(detections) != n:
        raise RuntimeError(f"post-processing returned {len(detections)} results "
                           f"for a batch of {n}")
    t3 = time.perf_counter()

    prep_s, infer_s, post_s = (t1 - t0) / n, (t2 - t1) / n, (t3 - t2) / n
    results = []
    for image, detection in zip(images, detections):
        started = time.perf_counter()
        masks, scores = _keep_masks(detection, image.size, min_score)
        results.append((masks, scores, {
            "prep_s": f"{prep_s:.4f}",
            "infer_s": f"{infer_s:.4f}",
            "post_s": f"{post_s + time.perf_counter() - started:.4f}",
        }))
    return results


def _processor_call(processor, images, prompt):
    """The processor call for a batch.

    SAM3's processor wants one text per image; a build that instead broadcasts a
    single string rejects the list, so try the list first and fall back.
    """
    try:
        return processor(images=images, text=[prompt] * len(images), return_tensors="pt")
    except (TypeError, ValueError):
        return processor(images=images, text=prompt, return_tensors="pt")


def _keep_masks(detection, size, min_score):
    """SAM3's raw output for one image -> (masks, scores), most confident first."""
    masks, kept = [], []
    raw_masks = detection.get("masks")
    raw_scores = detection.get("scores")
    if raw_masks is not None and raw_scores is not None and len(raw_scores):
        scores = [float(s) for s in raw_scores]
        pairs = sorted(zip(raw_masks, scores), key=lambda pair: pair[1], reverse=True)
        for mask, score in pairs:
            if score < min_score:
                continue
            binary = to_binary(mask, size)
            if binary.any():
                masks.append(binary)
                kept.append(score)
    return masks, kept


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--taxon-id", type=int, help="iNaturalist taxon ID")
    parser.add_argument("--parquet", help="Path to inat_photos.parquet")
    parser.add_argument("--local-index",
                        help="Parquet of images already on disk (local_image_paths.parquet). "
                             "Segments those files in place; no taxon, no S3.")
    parser.add_argument("--image-root",
                        help="Directory the local index's leading 'data/' maps to")
    parser.add_argument("--out", required=True, help="Output directory (on /blue)")
    parser.add_argument("--prompt", default="flower", help="SAM3 text prompt")
    parser.add_argument("--min-score", type=float, default=0.9,
                        help="Discard detections below this confidence")
    parser.add_argument("--shard", type=int, default=0, help="This worker's shard index")
    parser.add_argument("--num-shards", type=int, default=1, help="Total shards in the run")
    parser.add_argument("--workers", type=int, default=16,
                        help="Threads reading images ahead of the GPU")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Images per forward pass. 1 is one at a time, which is "
                             "all an L4 has room for; 8-64 is what makes a B200 worth "
                             "asking for. A batch that will not fit is halved and retried.")
    parser.add_argument("--save-workers", type=int, default=1,
                        help="Threads writing masks, cut-outs and overlays. 1 writes them "
                             "on the main thread and the GPU waits; raise it whenever "
                             "--batch-size is raised.")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Stop the segmentation loop after this many seconds, for "
                             "timed throughput tests. Everything finished is kept, and "
                             "the rest is picked up by the next run's resume.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only the first N photos of the taxon (for testing)")
    parser.add_argument("--bf16", action="store_true",
                        help="Run inference in bfloat16 (faster; slightly different masks)")
    parser.add_argument("--discard-outputs", action="store_true",
                        help="Benchmark mode: write each photo's image, masks, overlay and "
                             "cut-outs (so timing is real), measure them, then delete them. "
                             "The results CSV and timing JSON are kept.")
    args = parser.parse_args()
    if args.local_index:
        if not args.image_root:
            parser.error("--image-root is required with --local-index")
    elif not (args.taxon_id and args.parquet):
        parser.error("--taxon-id and --parquet are required "
                     "(or use --local-index with --image-root)")
    args.num_shards = max(1, args.num_shards)
    args.batch_size = max(1, args.batch_size)
    args.workers = max(1, args.workers)
    args.save_workers = max(1, args.save_workers)
    return args


if __name__ == "__main__":
    main()
