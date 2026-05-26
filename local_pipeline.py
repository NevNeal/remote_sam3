#!/usr/bin/env python3
"""
Local segmentation pipeline -- reads photo URLs from the iNat parquet index
instead of calling the iNaturalist API.

Requires inat_db/data/inat_photos.parquet to exist.
Build it first with:  python inat_db/build_db.py

Usage:
    python local_pipeline.py \
        --taxon-id  53324 \
        --output-dir 53324_flower_segmentations \
        --limit 100 \
        --prompt "flower"
"""

import argparse, os, re, csv, time
from io import BytesIO
from pathlib import Path

import duckdb
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
_p.add_argument("--output-dir", required=True,            help="Output folder (under repo root)")
_p.add_argument("--limit",      type=int, default=None,   help="Process at most N images")
_p.add_argument("--prompt",     type=str, default="flower", help="SAM3 text prompt")
_p.add_argument("--parquet",    type=str, default=None,
                help="Path to inat_photos.parquet (default: inat_db/data/inat_photos.parquet)")
_args = _p.parse_args()

# -- Settings -----------------------------------------------------------------
TAXON_ID   = _args.taxon_id
LIMIT      = _args.limit
TEXT_PROMPT = _args.prompt

REPO_ROOT  = Path(__file__).parent
BASE_DIR   = REPO_ROOT / _args.output_dir
PARQUET    = Path(_args.parquet) if _args.parquet else REPO_ROOT / "inat_db" / "data" / "inat_photos.parquet"

IMAGE_BASE   = BASE_DIR / "images"
MASK_BASE    = BASE_DIR / "masks"
OVERLAY_BASE = BASE_DIR / "overlays"
OUT_CSV      = BASE_DIR / "mask_summary.csv"
ERR_LOG      = BASE_DIR / "errors.txt"

BATCH_SIZE   = 10_000
MAX_WORKERS  = 2
TIMEOUT_SEC  = 60
CHUNK_BYTES  = 1024 * 1024
SKIP_IF_EXISTS = True

SCORE_THRESHOLD = 0.85
MASK_THRESHOLD  = 0.80
LOG_FAILURES    = True

MAKE_OVERLAY             = True
OVERWRITE_OVERLAYS       = True
OVERLAY_ALPHA            = 95
OVERLAY_COLOR            = (255, 0, 0)
BOX_COLOR                = (255, 255, 0)
BOX_WIDTH                = 4
TEXT_MARGIN              = 4

for d in [BASE_DIR, IMAGE_BASE, MASK_BASE, OVERLAY_BASE]:
    os.makedirs(d, exist_ok=True)

# -- Stage 1: query parquet for photo URLs ------------------------------------
def load_photos_from_parquet():
    if not PARQUET.exists():
        raise FileNotFoundError(
            f"Parquet not found: {PARQUET}\n"
            "Run:  python inat_db/build_db.py"
        )
    p = str(PARQUET).replace("\\", "/")
    print(f"Querying parquet: taxon_id={TAXON_ID} ...")
    t0 = time.time()
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT
            photo_id,
            extension,
            taxon_name,
            taxon_id,
            latitude,
            longitude
        FROM read_parquet('{p}')
        WHERE taxon_id = {TAXON_ID}
        ORDER BY photo_id
    """).df()
    con.close()
    elapsed = time.time() - t0
    print(f"  {len(df):,} photos found in {elapsed:.3f}s")
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

def img_ext(url):
    if not isinstance(url, str): return "jpg"
    _, ext = os.path.splitext(url.split("?", 1)[0])
    ext = ext.lower().lstrip(".")
    return ext if ext in {"jpg", "jpeg", "png"} else "jpg"

def batch_name(gidx):
    return f"batch_{(int(gidx) - 1) // BATCH_SIZE + 1:05d}"

def build_paths(row):
    gidx  = int(row["global_index"])
    bn    = batch_name(gidx)
    ext   = img_ext(str(row["photo_url"]))
    iname = (f"{row['genus_species']}_{int(row['photo_id'])}"
             f"_image{int(row['image_index'])}.{ext}")
    stem  = os.path.splitext(iname)[0]
    idir  = str(IMAGE_BASE   / bn)
    mdir  = str(MASK_BASE    / bn)
    odir  = str(OVERLAY_BASE / bn)
    ipath = os.path.join(idir, iname)
    opath = os.path.join(odir, f"{stem}_overlay.png")
    return pd.Series({
        "batch_name": bn, "image_name": iname, "stem": stem,
        "image_dir": idir, "mask_dir": mdir, "overlay_dir": odir,
        "image_path": ipath, "overlay_path": opath,
        "image_relpath":          os.path.relpath(ipath, BASE_DIR),
        "mask_batch_relpath":     os.path.relpath(mdir,  BASE_DIR),
        "overlay_relpath_expected": os.path.relpath(opath, BASE_DIR),
    })

def build_df():
    raw = load_photos_from_parquet()
    if LIMIT is not None:
        raw = raw.head(LIMIT).copy()

    raw["photo_url"] = (
        "https://inaturalist-open-data.s3.amazonaws.com/photos/"
        + raw["photo_id"].astype(str) + "/original." + raw["extension"]
    )
    raw["genus_species"] = raw["taxon_name"].astype(str).apply(genus_species)
    # Each row is one photo; image_index within same taxon+photo_id is always 1
    raw["image_index"]   = 1
    raw = raw.reset_index(drop=True)
    raw["global_index"]  = np.arange(1, len(raw) + 1, dtype=int)

    paths = raw.apply(build_paths, axis=1)
    return pd.concat([raw, paths], axis=1)

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

# -- Mask/overlay helpers -----------------------------------------------------
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

def avg_rgb(img_np, mask_np):
    mb = np.squeeze(mask_np) > 0
    if not mb.any(): return None
    px = img_np[mb]
    if not px.size: return None
    a = px.mean(axis=0)
    return (int(round(float(a[0]))), int(round(float(a[1]))), int(round(float(a[2]))))

def rgb_str(t):
    return f"({t[0]},{t[1]},{t[2]})" if t else ""

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
    except:
        tw, th = draw.textsize(text, font=font)
    draw.rectangle([x, y, x+tw+2*TEXT_MARGIN, y+th+2*TEXT_MARGIN], fill=(0, 0, 0, 180))
    draw.text((x+TEXT_MARGIN, y+TEXT_MARGIN), text, fill=(255, 255, 255, 255), font=font)

def write_atomic(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as f: f.write(data)
    os.replace(tmp, path)

def make_overlay(image_path, mask_files, scores, output_path):
    img  = Image.open(image_path).convert("RGBA")
    w, h = img.size
    ml   = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dl   = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(dl)
    font = get_font(max(16, int(min(w, h) * 0.025)))
    valid = 0
    for i, mp in enumerate(mask_files):
        try:
            mask = resize_mask(np.load(mp), img.size)
            if not (mask > 0).any(): continue
            valid += 1
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
        except Exception as e:
            print(f"Warning overlay: {mp}: {e}")
    if valid == 0:
        draw_label(draw, (10, 10), "No detections", font)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.alpha_composite(Image.alpha_composite(img, ml), dl).convert("RGB").save(output_path, quality=95)
    return valid

# -- CSV logging --------------------------------------------------------------
CSV_HEADER = [
    "global_index", "batch_name", "image_relpath", "mask_batch_relpath",
    "overlay_relpath", "image_name", "photo_id", "image_index",
    "taxon_id", "taxon_name", "photo_url", "num_instances",
    "mask_files", "mask_scores", "avg_RGB", "status", "error",
    "segmentation_duration_sec",
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
        df = pd.read_csv(OUT_CSV, usecols=["global_index", "status"])
        df = df[df["status"].astype(str).isin(["ok", "dl_failed", "seg_failed"])]
        return {int(v) for v in df["global_index"].dropna() if str(v).lstrip("-").isdigit()}
    except:
        return set()

def log_err(line):
    with open(ERR_LOG, "a", encoding="utf-8") as f:
        f.write(line.strip() + "\n")

# -- Main segmentation loop ---------------------------------------------------
def run_segmentation():
    df = build_df()
    print(f"\nTaxon ID   : {TAXON_ID}")
    print(f"Prompt     : '{TEXT_PROMPT}'")
    print(f"Total rows : {len(df):,}")
    print(f"Output     : {BASE_DIR}\n")

    ensure_header()
    done    = load_done()
    df_todo = df[~df["global_index"].isin(done)].copy()
    print(f"Done: {len(done):,}  Remaining: {len(df_todo):,}")
    if df_todo.empty:
        print("Nothing to do.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {p.name} ({p.total_memory/1e9:.1f} GB)")

    model     = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model.eval()
    session   = make_session()

    def dl_job(rd):
        gidx  = int(rd["global_index"])
        url   = str(rd["photo_url"])
        ipath = str(rd["image_path"])
        if SKIP_IF_EXISTS and os.path.exists(ipath) and os.path.getsize(ipath) > 0:
            return (gidx, "exists", None, None)
        try:
            return (gidx, "downloaded", download_bytes(session, url), None)
        except Exception as e:
            return (gidx, "dl_failed", None, str(e))

    prefetch = min(MAX_WORKERS * 2, 64)
    records  = df_todo.to_dict(orient="records")
    n        = len(records)
    by_gidx  = {int(r["global_index"]): r for r in records}
    progress = tqdm(total=n, desc="Segmenting", dynamic_ncols=True)
    ok = dlfail = segfail = skip = ocount = nodet = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        it       = iter(records)
        inflight = {}
        for _ in range(min(prefetch, n)):
            row = next(it, None)
            if row is None: break
            f = ex.submit(dl_job, row)
            inflight[f] = int(row["global_index"])

        while inflight:
            for future in as_completed(list(inflight.keys()), timeout=None): break

            gidx = inflight.pop(future)
            row  = by_gidx[gidx]
            bn   = row["batch_name"];  iname = row["image_name"]; stem = row["stem"]
            ipath = row["image_path"]; idir  = row["image_dir"];  mdir = row["mask_dir"]
            odir  = row["overlay_dir"]; opath = row["overlay_path"]
            irel  = row["image_relpath"]; mrel = row["mask_batch_relpath"]
            orel  = ""
            photo_id = int(row["photo_id"])
            img_idx  = int(row["image_index"])
            tid      = row.get("taxon_id", "")
            tname    = str(row["taxon_name"])
            url      = str(row["photo_url"])
            seg_dur  = ""

            try:
                ret_gidx, status, data, err = future.result()
                assert ret_gidx == gidx

                if status == "dl_failed":
                    dlfail += 1
                    if LOG_FAILURES:
                        append_row([gidx, bn, irel, mrel, orel, iname, photo_id, img_idx,
                                    tid, tname, url, 0, "", "", "", "dl_failed", err or "", ""])
                        log_err(f"[DL_FAIL] gidx={gidx} url={url} err={err}")
                    progress.update(1)

                else:
                    for d in [idir, mdir, odir]: os.makedirs(d, exist_ok=True)
                    if status == "downloaded" and data is not None:
                        write_atomic(ipath, data)
                        image_pil = Image.open(BytesIO(data)).convert("RGB")
                    else:
                        skip += 1
                        image_pil = Image.open(ipath).convert("RGB")

                    seg_t  = time.perf_counter()
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

                    img_np = np.array(image_pil)
                    saved_files = []; rgb_vals = []; saved_rels = []
                    for ii, mask in enumerate(ml):
                        mnp  = resize_mask(to_uint8(mask), image_pil.size)
                        rgb_vals.append(avg_rgb(img_np, mnp))
                        mname = f"{stem}_instance_{ii}.npy"
                        mpath = os.path.join(mdir, mname)
                        np.save(mpath, mnp)
                        saved_files.append(mpath)
                        saved_rels.append(os.path.relpath(mpath, BASE_DIR))

                    seg_dur    = round(time.perf_counter() - seg_t, 3)
                    scores_str = ",".join(f"{s:.4f}" for s in sl)
                    files_str  = ";".join(saved_rels)
                    rgb_str_v  = ",".join(rgb_str(v) for v in rgb_vals if v)

                    if MAKE_OVERLAY:
                        if os.path.exists(opath) and not OVERWRITE_OVERLAYS:
                            orel = os.path.relpath(opath, BASE_DIR)
                        else:
                            v = make_overlay(ipath, saved_files, sl, opath)
                            orel = os.path.relpath(opath, BASE_DIR)
                            ocount += 1
                            if v == 0: nodet += 1

                    append_row([gidx, bn, irel, mrel, orel, iname, photo_id, img_idx,
                                tid, tname, url, len(saved_files), files_str,
                                scores_str, rgb_str_v, "ok", "", seg_dur])
                    ok += 1
                    progress.update(1)

            except Exception as e:
                segfail += 1
                if LOG_FAILURES:
                    append_row([gidx, bn, irel, mrel, orel, iname, photo_id, img_idx,
                                tid, tname, url, 0, "", "", "", "seg_failed", str(e), ""])
                    log_err(f"[SEG_FAIL] gidx={gidx} err={e}")
                progress.update(1)

            nxt = next(it, None)
            if nxt is not None:
                nf = ex.submit(dl_job, nxt)
                inflight[nf] = int(nxt["global_index"])

    progress.close()
    print(f"\nDone.  OK:{ok:,}  DL_fail:{dlfail:,}  Seg_fail:{segfail:,}  "
          f"Overlays:{ocount:,}  NoDetect:{nodet:,}")
    print(f"Output: {BASE_DIR}")

if __name__ == "__main__":
    run_segmentation()

