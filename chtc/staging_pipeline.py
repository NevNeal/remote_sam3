#!/usr/bin/env python3
"""
Staging segmentation pipeline -- reads photo URLs for one taxon directly from the
iNat parquet index (queried with DuckDB) and runs SAM3 instance segmentation.

Designed to run inside the CHTC container, with the parquet supplied from /staging
via transfer_input_files. Outputs are written under --output-dir and are intended to
be shipped to Research Drive via the submit file's output_destination (pelican://).

What it saves (everything bucketed into batches of 1000 images):
    images/batch_NNNNN/      every raw image that was downloaded
    masks/batch_NNNNN/       .npy mask for each detection scoring > --min-score (0.9)
    overlays/batch_NNNNN/    one annotated overlay per image that had >0.9 detections
    segments/batch_NNNNN/    transparent PNG cut-out of each individual >0.9 segment
    results.csv              one row per photo in the taxon (parquet metadata +
                             image / overlay / mask / segment filepaths +
                             per-mask confidence list, ordered high->low)

Usage:
    python staging_pipeline.py \
        --taxon-id 160559 \
        --output-dir a100_160559_flower \
        --parquet /staging/u/<netid>/inat_photos.parquet \
        --prompt "flower"
"""

import argparse, os, re, csv, time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import torch
from transformers import Sam3Model, Sam3Processor

# -- Args ---------------------------------------------------------------------
_p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
_p.add_argument("--taxon-id",   type=int, required=True,  help="iNaturalist taxon ID")
_p.add_argument("--output-dir", required=True,            help="Output folder (under cwd)")
_p.add_argument("--parquet",    type=str, required=True,
                help="Path to inat_photos.parquet (e.g. the staged copy from /staging)")
_p.add_argument("--prompt",     type=str, default="flower", help="SAM3 text prompt")
_p.add_argument("--limit",      type=int, default=None,   help="Process at most N images")
_p.add_argument("--min-score",  type=float, default=0.9,
                help="Only save masks/overlays/segments at or above this confidence")
_args = _p.parse_args()

# -- Settings -----------------------------------------------------------------
TAXON_ID    = _args.taxon_id
LIMIT       = _args.limit
TEXT_PROMPT = _args.prompt
MIN_SCORE   = _args.min_score          # save threshold: keep detections >= this

CWD        = Path.cwd()
BASE_DIR   = (CWD / _args.output_dir).resolve()
PARQUET    = Path(_args.parquet)

IMAGE_BASE   = BASE_DIR / "images"
MASK_BASE    = BASE_DIR / "masks"
OVERLAY_BASE = BASE_DIR / "overlays"
SEGMENT_BASE = BASE_DIR / "segments"
OUT_CSV      = BASE_DIR / "results.csv"
ERR_LOG      = BASE_DIR / "errors.txt"

BATCH_SIZE   = 1_000                    # images per batch folder
MAX_WORKERS  = 4
TIMEOUT_SEC  = 60
CHUNK_BYTES  = 1024 * 1024
SKIP_IF_EXISTS = True

# Detection threshold passed to the model.  Kept equal to MIN_SCORE so we never
# spend disk on detections we are going to discard anyway.
SCORE_THRESHOLD = MIN_SCORE
MASK_THRESHOLD  = 0.80
LOG_FAILURES    = True

OVERLAY_ALPHA = 95
OVERLAY_COLOR = (255, 0, 0)
BOX_COLOR     = (255, 255, 0)
BOX_WIDTH     = 4
TEXT_MARGIN   = 4

for d in [BASE_DIR, IMAGE_BASE, MASK_BASE, OVERLAY_BASE, SEGMENT_BASE]:
    os.makedirs(d, exist_ok=True)

# -- Stage 1: query parquet for this taxon ------------------------------------
def load_photos_from_parquet():
    if not PARQUET.exists():
        raise FileNotFoundError(
            f"Parquet not found: {PARQUET}. "
            "Stage it from /staging via transfer_input_files, or pass --parquet."
        )
    print(f"Reading parquet: taxon_id={TAXON_ID}  ({PARQUET})")
    t0 = time.time()
    df = pd.read_parquet(
        PARQUET,
        columns=["photo_id", "extension", "taxon_id", "taxon_name",
                 "quality_grade", "latitude", "longitude"],
        filters=[("taxon_id", "==", TAXON_ID)],
    ).sort_values("photo_id").reset_index(drop=True)
    print(f"  {len(df):,} photos found in {time.time() - t0:.3f}s")
    if df.empty:
        raise SystemExit(f"No research-grade photos found for taxon_id={TAXON_ID}.")
    return df

# -- Path helpers -------------------------------------------------------------
def sanitize(v):
    v = re.sub(r"\s+", "_", str(v or "").strip())
    return re.sub(r"[^A-Za-z0-9_]+", "", v)

def genus_species(name):
    if not isinstance(name, str) or not name.strip():
        return "Unknown_unknown"
    parts = name.strip().split()
    if len(parts) >= 2:
        return f"{sanitize(parts[0].capitalize())}_{sanitize(parts[1].lower())}"
    return sanitize(name)

def img_ext(ext):
    ext = str(ext or "").lower().lstrip(".")
    return ext if ext in {"jpg", "jpeg", "png"} else "jpg"

def batch_name(gidx):
    return f"batch_{(int(gidx) - 1) // BATCH_SIZE + 1:05d}"

def build_paths(row):
    gidx  = int(row["global_index"])
    bn    = batch_name(gidx)
    ext   = img_ext(row["extension"])
    stem  = f"{row['genus_species']}_{int(row['photo_id'])}"
    iname = f"{stem}.{ext}"
    idir  = IMAGE_BASE   / bn
    mdir  = MASK_BASE    / bn
    odir  = OVERLAY_BASE / bn
    sdir  = SEGMENT_BASE / bn
    ipath = idir / iname
    opath = odir / f"{stem}_overlay.png"
    return pd.Series({
        "batch_name": bn, "stem": stem, "image_name": iname,
        "image_dir": str(idir), "mask_dir": str(mdir),
        "overlay_dir": str(odir), "segment_dir": str(sdir),
        "image_path": str(ipath), "overlay_path": str(opath),
        "image_relpath": os.path.relpath(ipath, BASE_DIR),
    })

def build_df():
    raw = load_photos_from_parquet()
    if LIMIT is not None:
        raw = raw.head(LIMIT).copy()
    raw = raw.reset_index(drop=True)
    raw["photo_url"] = (
        "https://inaturalist-open-data.s3.amazonaws.com/photos/"
        + raw["photo_id"].astype(str) + "/original." + raw["extension"].astype(str)
    )
    raw["genus_species"] = raw["taxon_name"].astype(str).apply(genus_species)
    raw["global_index"]  = np.arange(1, len(raw) + 1, dtype=int)
    return pd.concat([raw, raw.apply(build_paths, axis=1)], axis=1)

# -- HTTP helpers -------------------------------------------------------------
def make_session():
    s = requests.Session()
    retry = Retry(total=8, backoff_factor=1.5,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",), raise_on_status=False)
    a = HTTPAdapter(max_retries=retry,
                    pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
    s.mount("https://", a); s.mount("http://", a)
    return s

def download_bytes(session, url):
    r = session.get(url, stream=True, timeout=TIMEOUT_SEC)
    if r.status_code == 429:
        time.sleep(10)
        r = session.get(url, stream=True, timeout=TIMEOUT_SEC)
    r.raise_for_status()
    return b"".join(c for c in r.iter_content(chunk_size=CHUNK_BYTES) if c)

# -- Mask/overlay/segment helpers ---------------------------------------------
def to_uint8(mask):
    if isinstance(mask, torch.Tensor): mask = mask.detach().cpu().numpy()
    return (np.squeeze(np.array(mask)) > 0).astype(np.uint8)

def bbox(mask):
    ys, xs = np.where(np.squeeze(mask) > 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else None

def resize_mask(mask, sz):
    mask = np.squeeze(mask)
    tw, th = sz
    if (mask.shape[1], mask.shape[0]) == (tw, th): return mask
    img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    return (np.array(img.resize((tw, th), resample=Image.NEAREST)) > 0).astype(np.uint8)

def sort_by_conf(masks, scores):
    p = sorted(zip(masks, scores), key=lambda x: float(x[1]), reverse=True)
    return ([x[0] for x in p], [float(x[1]) for x in p]) if p else ([], [])

def save_segment_png(image_pil, mask_np, out_path):
    """Write a transparent PNG of one segment, cropped to its bounding box."""
    b = bbox(mask_np)
    if b is None:
        return False
    x1, y1, x2, y2 = b
    rgba = np.array(image_pil.convert("RGBA"))
    rgba[..., 3] = np.where(np.squeeze(mask_np) > 0, 255, 0).astype(np.uint8)
    crop = rgba[y1:y2 + 1, x1:x2 + 1]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(crop, mode="RGBA").save(out_path)
    return True

def get_font(size=22):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
              "C:/Windows/Fonts/arialbd.ttf"]:
        if os.path.exists(p): return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()

def draw_label(draw, xy, text, font):
    x, y = xy
    try:
        b = draw.textbbox((x, y), text, font=font)
        tw, th = b[2]-b[0], b[3]-b[1]
    except Exception:
        tw, th = draw.textsize(text, font=font)
    draw.rectangle([x, y, x+tw+2*TEXT_MARGIN, y+th+2*TEXT_MARGIN], fill=(0, 0, 0, 180))
    draw.text((x+TEXT_MARGIN, y+TEXT_MARGIN), text, fill=(255, 255, 255, 255), font=font)

def make_overlay(image_pil, masks_np, scores, output_path):
    img  = image_pil.convert("RGBA")
    w, h = img.size
    ml   = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dl   = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(dl)
    font = get_font(max(16, int(min(w, h) * 0.025)))
    for i, mask in enumerate(masks_np):
        if not (mask > 0).any(): continue
        al = Image.fromarray((mask > 0).astype(np.uint8) * OVERLAY_ALPHA, mode="L")
        ci = Image.new("RGBA", img.size, OVERLAY_COLOR + (0,)); ci.putalpha(al)
        ml = Image.alpha_composite(ml, ci)
        b  = bbox(mask)
        if b is None: continue
        x1, y1, x2, y2 = b
        for off in range(BOX_WIDTH):
            draw.rectangle([x1-off, y1-off, x2+off, y2+off], outline=BOX_COLOR+(255,))
        lbl = f"{TEXT_PROMPT} {scores[i]:.3f}" if i < len(scores) else TEXT_PROMPT
        draw_label(draw, (x1, max(0, y1 - font.size - 12)), lbl, font)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.alpha_composite(Image.alpha_composite(img, ml), dl).convert("RGB").save(
        output_path, quality=95)

# -- CSV logging --------------------------------------------------------------
# One row per photo in the taxon (parquet metadata duplicated per photo, as asked),
# plus the produced filepaths and the ordered confidence list.
CSV_HEADER = [
    "photo_id", "taxon_id", "taxon_name", "extension", "quality_grade",
    "latitude", "longitude", "photo_url",
    "global_index", "batch_name", "status", "num_masks",
    "image_relpath", "overlay_relpath", "mask_relpaths", "segment_relpaths",
    "mask_scores", "error",
]

def ensure_header():
    if not OUT_CSV.exists() or OUT_CSV.stat().st_size == 0:
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER)

def append_row(row):
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

def load_done():
    if not OUT_CSV.exists() or OUT_CSV.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(OUT_CSV, usecols=["photo_id", "status"])
        df = df[df["status"].astype(str).isin(["ok", "dl_failed", "seg_failed"])]
        return {int(v) for v in df["photo_id"].dropna() if str(v).lstrip("-").isdigit()}
    except Exception:
        return set()

def log_err(line):
    with open(ERR_LOG, "a", encoding="utf-8") as f:
        f.write(line.strip() + "\n")

# -- Main segmentation loop ---------------------------------------------------
def run_segmentation():
    df = build_df()
    print(f"\nTaxon ID    : {TAXON_ID}")
    print(f"Prompt      : '{TEXT_PROMPT}'")
    print(f"Save >=      : {MIN_SCORE}")
    print(f"Batch size  : {BATCH_SIZE}")
    print(f"Total photos: {len(df):,}")
    print(f"Output      : {BASE_DIR}\n")

    ensure_header()
    done    = load_done()
    df_todo = df[~df["photo_id"].isin(done)].copy()
    print(f"Done: {len(done):,}  Remaining: {len(df_todo):,}")
    if df_todo.empty:
        print("Nothing to do.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            pr = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {pr.name} ({pr.total_memory/1e9:.1f} GB)")

    model     = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model.eval()
    session   = make_session()

    def dl_job(rd):
        pid   = int(rd["photo_id"])
        url   = str(rd["photo_url"])
        ipath = str(rd["image_path"])
        if SKIP_IF_EXISTS and os.path.exists(ipath) and os.path.getsize(ipath) > 0:
            return (pid, "exists", None, None)
        try:
            return (pid, "downloaded", download_bytes(session, url), None)
        except Exception as e:
            return (pid, "dl_failed", None, str(e))

    prefetch = min(MAX_WORKERS * 2, 64)
    records  = df_todo.to_dict(orient="records")
    n        = len(records)
    by_pid   = {int(r["photo_id"]): r for r in records}
    progress = tqdm(total=n, desc="Segmenting", dynamic_ncols=True)
    ok = dlfail = segfail = skip = nodet = 0

    def write_atomic(path, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        with open(tmp, "wb") as f: f.write(data)
        os.replace(tmp, path)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        it       = iter(records)
        inflight = {}
        for _ in range(min(prefetch, n)):
            row = next(it, None)
            if row is None: break
            fut = ex.submit(dl_job, row)
            inflight[fut] = int(row["photo_id"])

        while inflight:
            for future in as_completed(list(inflight.keys()), timeout=None):
                break
            pid = inflight.pop(future)
            row = by_pid[pid]
            bn    = row["batch_name"]; stem = row["stem"]
            ipath = row["image_path"]; idir = row["image_dir"]
            mdir  = row["mask_dir"];   odir = row["overlay_dir"]; sdir = row["segment_dir"]
            irel  = row["image_relpath"]
            gidx  = int(row["global_index"])
            tid   = row.get("taxon_id", "")
            tname = str(row["taxon_name"])
            ext   = str(row["extension"])
            qg    = row.get("quality_grade", "")
            lat   = row.get("latitude", "")
            lon   = row.get("longitude", "")
            url   = str(row["photo_url"])
            opath = row["overlay_path"]

            def meta_prefix(status, num, orel, mrels, srels, scores_str, err):
                return [pid, tid, tname, ext, qg, lat, lon, url,
                        gidx, bn, status, num, irel, orel,
                        mrels, srels, scores_str, err]

            try:
                ret_pid, status, data, err = future.result()
                assert ret_pid == pid

                if status == "dl_failed":
                    dlfail += 1
                    if LOG_FAILURES:
                        append_row(meta_prefix("dl_failed", 0, "", "", "", "", err or ""))
                        log_err(f"[DL_FAIL] photo_id={pid} url={url} err={err}")
                    progress.update(1)
                else:
                    for d in [idir, mdir, odir, sdir]:
                        os.makedirs(d, exist_ok=True)
                    if status == "downloaded" and data is not None:
                        write_atomic(ipath, data)
                        image_pil = Image.open(BytesIO(data)).convert("RGB")
                    else:
                        skip += 1
                        image_pil = Image.open(ipath).convert("RGB")

                    inputs = processor(images=image_pil, text=TEXT_PROMPT,
                                       return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    post = processor.post_process_instance_segmentation(
                        outputs,
                        threshold=SCORE_THRESHOLD,
                        mask_threshold=MASK_THRESHOLD,
                        target_sizes=inputs.get("original_sizes").tolist(),
                    )[0]

                    masks, scores = post.get("masks"), post.get("scores")
                    if masks is None or scores is None:
                        ml, sl = [], []
                    else:
                        ml = [masks[i] for i in range(masks.shape[0])] \
                             if isinstance(masks, torch.Tensor) else list(masks)
                        sl = scores.detach().cpu().tolist() \
                             if isinstance(scores, torch.Tensor) else list(scores)
                    ml, sl = sort_by_conf(ml, sl)

                    # Keep only detections >= MIN_SCORE (model threshold already
                    # filters, but guard in case of float edge cases).
                    keep = [(m, s) for m, s in zip(ml, sl) if float(s) >= MIN_SCORE]

                    mask_rels = []; seg_rels = []; kept_masks = []; kept_scores = []
                    for ii, (mask, sc) in enumerate(keep):
                        mnp = resize_mask(to_uint8(mask), image_pil.size)
                        if not (mnp > 0).any():
                            continue
                        kept_masks.append(mnp); kept_scores.append(float(sc))
                        # mask .npy
                        mname = f"{stem}_instance_{ii}.npy"
                        mpath = os.path.join(mdir, mname)
                        np.save(mpath, mnp)
                        mask_rels.append(os.path.relpath(mpath, BASE_DIR))
                        # transparent segment cut-out
                        spath = os.path.join(sdir, f"{stem}_segment_{ii}.png")
                        if save_segment_png(image_pil, mnp, spath):
                            seg_rels.append(os.path.relpath(spath, BASE_DIR))

                    orel = ""
                    if kept_masks:
                        make_overlay(image_pil, kept_masks, kept_scores, opath)
                        orel = os.path.relpath(opath, BASE_DIR)
                    else:
                        nodet += 1

                    scores_str = ",".join(f"{s:.4f}" for s in kept_scores)
                    append_row(meta_prefix(
                        "ok", len(kept_masks), orel,
                        ";".join(mask_rels), ";".join(seg_rels), scores_str, ""))
                    ok += 1
                    progress.update(1)

            except Exception as e:
                segfail += 1
                if LOG_FAILURES:
                    append_row(meta_prefix("seg_failed", 0, "", "", "", "", str(e)))
                    log_err(f"[SEG_FAIL] photo_id={pid} err={e}")
                progress.update(1)

            nxt = next(it, None)
            if nxt is not None:
                nf = ex.submit(dl_job, nxt)
                inflight[nf] = int(nxt["photo_id"])

    progress.close()
    print(f"\nDone.  OK:{ok:,}  DL_fail:{dlfail:,}  Seg_fail:{segfail:,}  "
          f"NoDetect(>={MIN_SCORE}):{nodet:,}")
    print(f"Output: {BASE_DIR}")

if __name__ == "__main__":
    run_segmentation()
