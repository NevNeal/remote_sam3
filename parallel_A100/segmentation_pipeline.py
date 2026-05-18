# ============================================================
# Parallel A100 segmentation pipeline
#
# Designed for 5 simultaneous A100 jobs on CHTC GPU Lab.
# Each HTCondor process owns one shard (1/N) of the dataset.
#
# Taxon scope : Aizoaceae family (iNat taxon 49320) by default
# Text prompt : "flower"
#
# Sharding:
#   All jobs fetch metadata independently (fast, idempotent).
#   Each job takes rows where global_index % num_shards == shard.
#   Each job writes to its own output_dir (output_shard_N/).
#   Merge shards after all jobs complete.
#
# Usage:
#   python segmentation_pipeline.py \
#       --taxon-ids 49320 \
#       --output-dir output_shard_0 \
#       --shard 0 --num-shards 5
# ============================================================

import argparse, os, re, csv, time
from io import BytesIO
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import torch
from transformers import Sam3Model, Sam3Processor

# ── Args ──────────────────────────────────────────────────────────────────────
_p = argparse.ArgumentParser()
_p.add_argument("--taxon-ids",   type=str, default="49320")
_p.add_argument("--output-dir",  required=True)
_p.add_argument("--shard",       type=int, default=0)
_p.add_argument("--num-shards",  type=int, default=1)
_p.add_argument("--limit",       type=int, default=None)
_p.add_argument("--prompt",      type=str, default=None)
_args = _p.parse_args()

# ── Settings ──────────────────────────────────────────────────────────────────
TAXON_IDS : List[int] = [int(t.strip()) for t in _args.taxon_ids.split(",") if t.strip()]
SHARD      = _args.shard
NUM_SHARDS = _args.num_shards
LIMIT      = _args.limit
QUALITY    = "research"

PER_PAGE   = 200
DELAY_SEC  = 1.1
API_BASE   = "https://api.inaturalist.org/v1/observations"

LOCAL_ROOT   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.join(LOCAL_ROOT, _args.output_dir)
IMAGE_BASE   = os.path.join(BASE_DIR, "images")
MASK_BASE    = os.path.join(BASE_DIR, "masks")
OVERLAY_BASE = os.path.join(BASE_DIR, "overlays")
OUT_CSV      = os.path.join(BASE_DIR, "mask_summary.csv")
ERR_LOG      = os.path.join(BASE_DIR, "errors.txt")
BATCH_SIZE   = 10_000

# 4 workers keeps the GPU fed across a large family-level dataset
MAX_WORKERS = 4
TIMEOUT_SEC = 60
CHUNK_BYTES = 1024 * 1024
SLEEP_BETWEEN_REQUESTS_SEC = 0.0
SKIP_IF_EXISTS = True

TEXT_PROMPT     = _args.prompt if _args.prompt is not None else "flower"
SCORE_THRESHOLD = 0.85
MASK_THRESHOLD  = 0.8
LOG_FAILURES    = True

MAKE_OVERLAY_FOR_EVERY_SEGMENTED_IMAGE = True
OVERWRITE_EXISTING_OVERLAYS = True
OVERLAY_ALPHA = 95
OVERLAY_COLOR = (255, 0, 0)
BOX_COLOR     = (255, 255, 0)
BOX_WIDTH     = 4
TEXT_MARGIN   = 4
UPDATE_METADATA_WITH_AVG_RGB = True

for d in [BASE_DIR, IMAGE_BASE, MASK_BASE, OVERLAY_BASE]:
    os.makedirs(d, exist_ok=True)

# ── HTTP helpers ───────────────────────────────────────────────────────────────
def make_session(max_workers=MAX_WORKERS):
    s = requests.Session()
    retry = Retry(total=8, backoff_factor=1.5,
                  status_forcelist=(429,500,502,503,504),
                  allowed_methods=("GET",), raise_on_status=False)
    a = HTTPAdapter(max_retries=retry, pool_connections=max_workers, pool_maxsize=max_workers)
    s.mount("https://", a); s.mount("http://", a)
    s.headers.update({"User-Agent": "inat-parallel-a100/1.0", "Accept": "application/json"})
    return s

def fetch_json(session, url, params, timeout=60):
    r = session.get(url, params=params, timeout=timeout)
    if r.status_code == 429:
        time.sleep(10)
        r = session.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ── Metadata URL helpers ───────────────────────────────────────────────────────
def infer_ext(url):
    if not url: return None
    try:
        parts = os.path.basename(urlparse(url).path).rsplit(".", 1)
        return parts[1].lower() if len(parts)==2 and parts[1].lower() in {"jpg","jpeg","png","gif"} else None
    except: return None

def s3_url(photo_id, ext):
    return f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/original.{ext}" if photo_id and ext else None

def best_photo_url(photo):
    pid, api_url, api_orig = photo.get("id"), photo.get("url"), photo.get("original_url")
    ext = infer_ext(api_orig) or infer_ext(api_url)
    u = s3_url(pid, ext)
    if u: return u
    if api_orig: return api_orig
    if api_url:
        for t in ["square","small","medium","large","original"]:
            if f"/{t}." in api_url: return api_url.replace(f"/{t}.","/original.")
        return api_url
    return None

# ── Metadata CSV schema ────────────────────────────────────────────────────────
METADATA_COLUMNS = [
    "observation_id","observation_uuid","quality_grade","observed_on",
    "time_observed_at","created_at","updated_at","license_code",
    "geoprivacy","taxon_geoprivacy","location","latitude","longitude",
    "place_guess","captive","identifications_count","comments_count",
    "faves_count","user_id","user_login","taxon_id","taxon_name",
    "taxon_preferred_common_name","taxon_rank","taxon_ancestry",
    "photo_id","photo_uuid","photo_license_code","photo_attribution",
    "photo_width","photo_height","photo_url_original",
]

def rows_from_obs(obs):
    taxon = obs.get("taxon") or {}
    user  = obs.get("user")  or {}
    base  = {
        "observation_id": obs.get("id"), "observation_uuid": obs.get("uuid"),
        "quality_grade": obs.get("quality_grade"), "observed_on": obs.get("observed_on"),
        "time_observed_at": obs.get("time_observed_at"), "created_at": obs.get("created_at"),
        "updated_at": obs.get("updated_at"), "license_code": obs.get("license_code"),
        "geoprivacy": obs.get("geoprivacy"), "taxon_geoprivacy": obs.get("taxon_geoprivacy"),
        "location": obs.get("location"), "latitude": obs.get("latitude"),
        "longitude": obs.get("longitude"), "place_guess": obs.get("place_guess"),
        "captive": obs.get("captive"), "identifications_count": obs.get("identifications_count"),
        "comments_count": obs.get("comments_count"), "faves_count": obs.get("faves_count"),
        "user_id": user.get("id"), "user_login": user.get("login"),
        "taxon_id": taxon.get("id"), "taxon_name": taxon.get("name"),
        "taxon_preferred_common_name": taxon.get("preferred_common_name"),
        "taxon_rank": taxon.get("rank"), "taxon_ancestry": taxon.get("ancestry"),
    }
    photos = obs.get("photos") or []
    if not photos:
        row = dict(base); [row.setdefault(c,"") for c in METADATA_COLUMNS]; yield row; return
    for photo in photos:
        row = dict(base)
        row.update({"photo_id": photo.get("id"), "photo_uuid": photo.get("uuid"),
                    "photo_license_code": photo.get("license_code"),
                    "photo_attribution": photo.get("attribution"),
                    "photo_width": photo.get("width"), "photo_height": photo.get("height"),
                    "photo_url_original": best_photo_url(photo)})
        [row.setdefault(c,"") for c in METADATA_COLUMNS]; yield row

def resume_id(path):
    if not os.path.exists(path): return 0
    with open(path,"rb") as f:
        try: f.seek(-65536, os.SEEK_END)
        except: f.seek(0)
        lines = f.read().splitlines()
    for line in reversed(lines):
        if line.strip():
            first = line.decode("utf-8","ignore").split(",",1)[0].strip()
            if first == "observation_id": return 0
            try: return int(first)
            except: return 0
    return 0

# ── Download metadata ──────────────────────────────────────────────────────────
def meta_path(tid): return os.path.join(BASE_DIR, f"inat_taxon_{tid}_{QUALITY}_obs_photo_metadata.csv")

def download_meta_for_taxon(session, tid):
    path = meta_path(tid)
    id_above = resume_id(path)
    print(f"[taxon {tid}] resume id_above={id_above}")
    exists = os.path.exists(path)
    f = open(path,"a",newline="",encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
    if not exists: w.writeheader()
    params_base = {"taxon_id":tid,"quality_grade":QUALITY,"per_page":PER_PAGE,
                   "order":"asc","order_by":"id","photos":"true"}
    obs_total = rows_total = 0
    try:
        while True:
            params = dict(params_base)
            if id_above > 0: params["id_above"] = id_above
            data = fetch_json(session, API_BASE, params)
            results = data.get("results",[])
            if not results: print(f"[taxon {tid}] done."); break
            last_id = None
            for obs in results:
                last_id = obs.get("id", last_id)
                obs_total += 1
                for row in rows_from_obs(obs):
                    w.writerow(row); rows_total += 1
            f.flush()
            if last_id is None: break
            id_above = int(last_id)
            print(f"[taxon {tid}] id={id_above}  obs={obs_total:,}  rows={rows_total:,}")
            time.sleep(DELAY_SEC)
    finally: f.close()
    print(f"[taxon {tid}] -> {path}")

def download_metadata_csv():
    session = make_session(max_workers=8)
    for tid in TAXON_IDS: download_meta_for_taxon(session, tid)

# ── Path helpers ───────────────────────────────────────────────────────────────
def sanitize(v):
    v = re.sub(r"\s+","_",str(v or "").strip())
    return re.sub(r"[^A-Za-z0-9_]+","",v)

def genus_species(name):
    if not isinstance(name,str) or not name.strip(): return "Unknown_unknown"
    parts = name.strip().split()
    return f"{sanitize(parts[0].capitalize())}_{sanitize(parts[1].lower())}" if len(parts)>=2 else sanitize(name)

def img_ext(url):
    if not isinstance(url,str): return "jpg"
    _,ext = os.path.splitext(url.split("?",1)[0])
    ext = ext.lower().lstrip(".")
    return ext if ext in {"jpg","jpeg","png"} else "jpg"

def batch_name(gidx): return f"batch_{(int(gidx)-1)//BATCH_SIZE+1:05d}"

def build_paths(row):
    gidx  = int(row["global_index"])
    bn    = batch_name(gidx)
    ext   = img_ext(str(row["photo_url_original"]))
    iname = f"{row['genus_species']}_{int(row['observation_id'])}_image{int(row['image_index'])}.{ext}"
    stem  = os.path.splitext(iname)[0]
    idir  = os.path.join(IMAGE_BASE,   bn)
    mdir  = os.path.join(MASK_BASE,    bn)
    odir  = os.path.join(OVERLAY_BASE, bn)
    ipath = os.path.join(idir, iname)
    opath = os.path.join(odir, f"{stem}_overlay.png")
    return pd.Series({"batch_name":bn,"image_name":iname,"stem":stem,
                       "image_dir":idir,"mask_dir":mdir,"overlay_dir":odir,
                       "image_path":ipath,"overlay_path":opath,
                       "image_relpath":os.path.relpath(ipath,BASE_DIR),
                       "mask_batch_relpath":os.path.relpath(mdir,BASE_DIR),
                       "overlay_relpath_expected":os.path.relpath(opath,BASE_DIR)})

# ── Image/mask/overlay helpers ─────────────────────────────────────────────────
def write_atomic(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path+".part"
    with open(tmp,"wb") as f: f.write(data)
    os.replace(tmp, path)

def download_bytes(session, url):
    r = session.get(url, stream=True, timeout=TIMEOUT_SEC)
    if r.status_code==429: time.sleep(10); r = session.get(url, stream=True, timeout=TIMEOUT_SEC)
    r.raise_for_status()
    chunks = [c for c in r.iter_content(chunk_size=CHUNK_BYTES) if c]
    if SLEEP_BETWEEN_REQUESTS_SEC > 0: time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)
    return b"".join(chunks)

def to_uint8(mask):
    if isinstance(mask, torch.Tensor): mask = mask.detach().cpu().numpy()
    return (np.squeeze(np.array(mask)) > 0).astype(np.uint8)

def bbox(mask):
    ys,xs = np.where(np.squeeze(mask)>0)
    return (int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())) if len(xs) else None

def resize_mask(mask, sz):
    mask = np.squeeze(mask)
    tw,th = sz
    if (mask.shape[1],mask.shape[0])==(tw,th): return mask
    img = Image.fromarray((mask>0).astype(np.uint8)*255)
    return (np.array(img.resize((tw,th),resample=Image.NEAREST))>0).astype(np.uint8)

def sort_by_conf(masks, scores):
    p = sorted(zip(masks,scores), key=lambda x: float(x[1]), reverse=True)
    return ([x[0] for x in p],[float(x[1]) for x in p]) if p else ([],[])

def avg_rgb(img_np, mask_np):
    mb = np.squeeze(mask_np)>0
    if not mb.any(): return None
    px = img_np[mb]
    if not px.size: return None
    a = px.mean(axis=0)
    return (int(round(float(a[0]))),int(round(float(a[1]))),int(round(float(a[2]))))

def rgb_str(t): return f"({t[0]},{t[1]},{t[2]})" if t else ""

def get_font(size=22):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
              "C:/Windows/Fonts/arialbd.ttf"]:
        if os.path.exists(p): return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()

def draw_label(draw, xy, text, font):
    x,y = xy
    try: b=draw.textbbox((x,y),text,font=font); tw,th=b[2]-b[0],b[3]-b[1]
    except: tw,th=draw.textsize(text,font=font)
    draw.rectangle([x,y,x+tw+2*TEXT_MARGIN,y+th+2*TEXT_MARGIN],fill=(0,0,0,180))
    draw.text((x+TEXT_MARGIN,y+TEXT_MARGIN),text,fill=(255,255,255,255),font=font)

def make_overlay(image_path, mask_files, scores, output_path):
    img = Image.open(image_path).convert("RGBA")
    w,h = img.size
    ml = Image.new("RGBA",img.size,(0,0,0,0))
    dl = Image.new("RGBA",img.size,(0,0,0,0))
    draw = ImageDraw.Draw(dl)
    font = get_font(max(16,int(min(w,h)*0.025)))
    valid = 0
    for i,mp in enumerate(mask_files):
        try:
            mask = resize_mask(np.load(mp), img.size)
            if not (mask>0).any(): continue
            valid += 1
            al = Image.fromarray((mask>0).astype(np.uint8)*OVERLAY_ALPHA,mode="L")
            ci = Image.new("RGBA",img.size,OVERLAY_COLOR+(0,)); ci.putalpha(al)
            ml = Image.alpha_composite(ml,ci)
            b = bbox(mask)
            if b is None: continue
            x1,y1,x2,y2 = b
            for off in range(BOX_WIDTH): draw.rectangle([x1-off,y1-off,x2+off,y2+off],outline=BOX_COLOR+(255,))
            lbl = f"{TEXT_PROMPT} {scores[i]:.3f}" if i<len(scores) else TEXT_PROMPT
            draw_label(draw,(x1,max(0,y1-font.size-12)),lbl,font)
        except Exception as e: print(f"Warning: {mp}: {e}")
    if valid==0: draw_label(draw,(10,10),"No detections",font)
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    Image.alpha_composite(Image.alpha_composite(img,ml),dl).convert("RGB").save(output_path,quality=95)
    return valid

# ── Logging helpers ────────────────────────────────────────────────────────────
CSV_HEADER = [
    "global_index","shard","batch_name","image_relpath","mask_batch_relpath",
    "overlay_relpath","image_name","observation_id","image_index","taxon_id",
    "taxon_name","photo_url_original","num_instances","mask_files","mask_scores",
    "avg_RGB","status","error","segmentation_duration_sec",
]

def ensure_header(path):
    if not os.path.exists(path) or os.path.getsize(path)==0:
        os.makedirs(os.path.dirname(path),exist_ok=True)
        with open(path,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER); f.flush(); os.fsync(f.fileno())

def append_row(path, row):
    with open(path,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow(row); f.flush(); os.fsync(f.fileno())

def load_done(path):
    if not os.path.exists(path) or os.path.getsize(path)==0: return set()
    try:
        df = pd.read_csv(path, usecols=["global_index","status"])
        df = df[df["status"].astype(str).isin(["ok","dl_failed","seg_failed"])]
        return {int(v) for v in df["global_index"].dropna() if str(v).lstrip("-").isdigit()}
    except: return set()

def log_err(line):
    os.makedirs(os.path.dirname(ERR_LOG),exist_ok=True)
    with open(ERR_LOG,"a",encoding="utf-8") as f: f.write(line.strip()+"\n")

# ── Build processing dataframe ─────────────────────────────────────────────────
def build_df():
    frames = []
    for tid in TAXON_IDS:
        path = meta_path(tid)
        print(f"Loading: {path}")
        df = pd.read_csv(path)
        needed = {"observation_id","taxon_name","photo_url_original"}
        miss = needed - set(df.columns)
        if miss: raise ValueError(f"taxon {tid} missing cols: {miss}")
        df = df[df["photo_url_original"].notna()
                & (df["photo_url_original"].astype(str).str.len()>0)].copy()
        df["observation_id"] = pd.to_numeric(df["observation_id"],errors="coerce").astype("Int64")
        df = df[df["observation_id"].notna()].copy()
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["image_index"]   = df.groupby("observation_id").cumcount()+1
    df["genus_species"] = df["taxon_name"].astype(str).apply(genus_species)
    df = df.reset_index(drop=True)
    df["global_index"]  = np.arange(1, len(df)+1, dtype=int)
    if NUM_SHARDS > 1:
        df = df[df["global_index"] % NUM_SHARDS == SHARD % NUM_SHARDS].copy()
        print(f"Shard {SHARD}/{NUM_SHARDS}: {len(df):,} rows")
    return pd.concat([df, df.apply(build_paths, axis=1)], axis=1)

# ── Update metadata with avg_RGB ───────────────────────────────────────────────
def update_avg_rgb():
    if not UPDATE_METADATA_WITH_AVG_RGB or not os.path.exists(OUT_CSV): return
    summary = pd.read_csv(OUT_CSV)
    if "avg_RGB" not in summary.columns or "status" not in summary.columns: return
    summary = summary[summary["status"].astype(str).eq("ok")].copy()
    required = {"observation_id","image_index","photo_url_original","avg_RGB"}
    if required - set(summary.columns): return
    for tid in TAXON_IDS:
        path = meta_path(tid)
        if not os.path.exists(path): continue
        meta = pd.read_csv(path)
        meta["observation_id"] = pd.to_numeric(meta["observation_id"],errors="coerce").astype("Int64")
        meta["image_index"]    = meta.groupby("observation_id").cumcount()+1
        sub = summary[["observation_id","image_index","photo_url_original","avg_RGB"]].copy()
        sub["observation_id"] = pd.to_numeric(sub["observation_id"],errors="coerce").astype("Int64")
        sub["image_index"]    = pd.to_numeric(sub["image_index"],   errors="coerce").astype("Int64")
        if "avg_RGB" in meta.columns: meta = meta.drop(columns=["avg_RGB"])
        meta = meta.merge(sub, on=["observation_id","image_index","photo_url_original"], how="left")
        meta.to_csv(path, index=False)
        print(f"[taxon {tid}] avg_RGB filled: {meta['avg_RGB'].notna().sum():,}")

# ── Main segmentation loop ─────────────────────────────────────────────────────
def run_segmentation():
    df = build_df()
    print(f"\nTaxon IDs  : {TAXON_IDS}")
    print(f"Shard      : {SHARD}/{NUM_SHARDS}")
    print(f"Rows       : {len(df):,}")
    print(f"Prompt     : '{TEXT_PROMPT}'")
    print(f"DL workers : {MAX_WORKERS}")
    print(f"Output     : {BASE_DIR}\n")

    ensure_header(OUT_CSV)
    done    = load_done(OUT_CSV)
    df_todo = df[~df["global_index"].isin(done)].copy()
    if LIMIT is not None: df_todo = df_todo.head(LIMIT)
    print(f"Done: {len(done):,}  Remaining: {len(df_todo):,}")
    if len(df_todo)==0: print("Nothing to do."); return

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
        gidx,url,ipath = int(rd["global_index"]),str(rd["photo_url_original"]),str(rd["image_path"])
        if SKIP_IF_EXISTS and os.path.exists(ipath) and os.path.getsize(ipath)>0:
            return (gidx,"exists",None,None)
        try: return (gidx,"downloaded",download_bytes(session,url),None)
        except Exception as e: return (gidx,"dl_failed",None,str(e))

    prefetch    = min(MAX_WORKERS*2, 128)
    records     = df_todo.to_dict(orient="records")
    n           = len(records)
    by_gidx     = {int(r["global_index"]):r for r in records}
    progress    = tqdm(total=n, desc=f"Shard {SHARD}", dynamic_ncols=True)
    ok=dlfail=segfail=skip=ocount=nodet = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        it       = iter(records)
        inflight = {}
        for _ in range(min(prefetch,n)):
            row = next(it,None)
            if row is None: break
            f = ex.submit(dl_job,row); inflight[f]=int(row["global_index"])

        while inflight:
            for future in as_completed(list(inflight.keys()),timeout=None): break

            gidx = inflight.pop(future)
            row  = by_gidx[gidx]
            bn,iname,stem = row["batch_name"],row["image_name"],row["stem"]
            ipath,idir,mdir = row["image_path"],row["image_dir"],row["mask_dir"]
            odir,opath      = row["overlay_dir"],row["overlay_path"]
            irel,mrel       = row["image_relpath"],row["mask_batch_relpath"]
            orel            = ""
            obs_id          = int(row["observation_id"])
            img_idx         = int(row["image_index"])
            tid             = row.get("taxon_id","")
            tname           = str(row["taxon_name"])
            url             = str(row["photo_url_original"])
            seg_dur         = ""

            try:
                ret_gidx,status,data,err = future.result()
                assert ret_gidx==gidx

                if status=="dl_failed":
                    dlfail+=1
                    if LOG_FAILURES:
                        append_row(OUT_CSV,[gidx,SHARD,bn,irel,mrel,orel,iname,obs_id,img_idx,tid,tname,url,0,"","","","dl_failed",err or "",seg_dur])
                        log_err(f"[DL_FAIL] gidx={gidx} url={url} err={err}")
                    progress.update(1)

                else:
                    for d in [idir,mdir,odir]: os.makedirs(d,exist_ok=True)
                    if status=="downloaded" and data is not None:
                        write_atomic(ipath,data); image_pil=Image.open(BytesIO(data)).convert("RGB")
                    else:
                        skip+=1; image_pil=Image.open(ipath).convert("RGB")

                    seg_t = time.perf_counter()
                    inputs = processor(images=image_pil,text=TEXT_PROMPT,return_tensors="pt").to(device)
                    with torch.no_grad(): outputs = model(**inputs)
                    post = processor.post_process_instance_segmentation(
                        outputs, threshold=SCORE_THRESHOLD, mask_threshold=MASK_THRESHOLD,
                        target_sizes=inputs.get("original_sizes").tolist())[0]

                    masks,scores = post.get("masks"),post.get("scores")
                    if masks is None or scores is None:
                        ml,sl = [],[]
                    else:
                        ml = [masks[i] for i in range(masks.shape[0])] if isinstance(masks,torch.Tensor) else list(masks)
                        sl = scores.detach().cpu().tolist() if isinstance(scores,torch.Tensor) else list(scores)
                    ml,sl = sort_by_conf(ml,sl)

                    img_np = np.array(image_pil)
                    saved_files=[];saved_rels=[];rgb_vals=[]
                    for ii,mask in enumerate(ml):
                        mnp = resize_mask(to_uint8(mask),image_pil.size)
                        rgb_vals.append(avg_rgb(img_np,mnp))
                        mname=f"{stem}_instance_{ii}.npy"; mpath=os.path.join(mdir,mname)
                        np.save(mpath,mnp); saved_files.append(mpath)
                        saved_rels.append(os.path.relpath(mpath,BASE_DIR))

                    seg_dur = round(time.perf_counter()-seg_t,3)
                    scores_str = ",".join(f"{s:.4f}" for s in sl)
                    files_str  = ";".join(saved_rels)
                    rgb_str_val= ",".join(rgb_str(v) for v in rgb_vals if v)

                    if MAKE_OVERLAY_FOR_EVERY_SEGMENTED_IMAGE:
                        if os.path.exists(opath) and not OVERWRITE_EXISTING_OVERLAYS:
                            orel = os.path.relpath(opath,BASE_DIR)
                        else:
                            v = make_overlay(ipath,saved_files,sl,opath)
                            orel=os.path.relpath(opath,BASE_DIR); ocount+=1
                            if v==0: nodet+=1

                    append_row(OUT_CSV,[gidx,SHARD,bn,irel,mrel,orel,iname,obs_id,img_idx,tid,tname,url,len(saved_files),files_str,scores_str,rgb_str_val,"ok","",seg_dur])
                    ok+=1; progress.update(1)

            except Exception as e:
                segfail+=1
                if LOG_FAILURES:
                    append_row(OUT_CSV,[gidx,SHARD,bn,irel,mrel,orel,iname,obs_id,img_idx,tid,tname,url,0,"","","","seg_failed",str(e),seg_dur])
                    log_err(f"[SEG_FAIL] gidx={gidx} err={e}")
                progress.update(1)

            nxt = next(it,None)
            if nxt is not None:
                nf=ex.submit(dl_job,nxt); inflight[nf]=int(nxt["global_index"])

    progress.close()
    print(f"\nShard {SHARD} done.  OK:{ok:,}  DL_fail:{dlfail:,}  Seg_fail:{segfail:,}")
    print(f"Overlays:{ocount:,}  NoDetect:{nodet:,}  Output:{BASE_DIR}")

if __name__ == "__main__":
    download_metadata_csv()
    run_segmentation()
    update_avg_rgb()