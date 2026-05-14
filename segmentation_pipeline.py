# ============================================================
# iNaturalist TAXON_ID -> metadata CSV -> download -> SAM3 segmentation
#
# Output structure:
#
# {output_folder}/
#   images/
#     batch_00001/      # images 1 through 10,000
#     batch_00002/      # images 10,001 through 20,000
#   masks/
#     batch_00001/      # masks only for images 1 through 10,000
#     batch_00002/      # masks only for images 10,001 through 20,000
#   overlays/
#     batch_00001/      # overlays only for images 1 through 10,000
#     batch_00002/      # overlays only for images 10,001 through 20,000
#   segments_pngs/
#
# Batch rule:
# - Batch is determined ONLY by global image index.
# - This keeps images, masks, and overlays aligned.
#
# Segmentation output:
# - Saves every original image.
# - Saves every mask as .npy.
# - Saves an overlay for every segmented image.
# - Extracts average RGB for each mask.
# - Sorts masks from highest to lowest confidence.
#
# avg_RGB format:
# - Stored in mask_summary.csv and metadata CSV.
# - Example:
#     "(182,91,104),(210,145,122),(156,73,88)"
# - Tuple order matches mask confidence order from highest to lowest.
#
# Notes:
# - facebook/sam3 may require accepting model terms and HF login.
# ============================================================


# ---------------------------
# 1) Imports
# ---------------------------
import argparse
import os
import re
import csv
import time
from io import BytesIO
from typing import Dict, Any, Optional

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


# ============================================================
# 2) ARGS
# ============================================================

_parser = argparse.ArgumentParser(description="iNaturalist SAM3 segmentation pipeline")
_parser.add_argument("taxon_id", type=int, help="iNaturalist taxon ID to process")
_parser.add_argument("output_folder", help="Name of the output folder (created inside the script directory)")
_parser.add_argument("--limit", type=int, default=None, help="Stop after processing this many images")
_parser.add_argument("--prompt", type=str, default=None, help="SAM3 text prompt (default: 'open flower')")
_args = _parser.parse_args()


# ============================================================
# 3) USER SETTINGS
# ============================================================

TAXON_ID = _args.taxon_id
LIMIT = _args.limit
QUALITY = "research"

# iNaturalist API settings
PER_PAGE = 200
DELAY_SEC = 1.1
API_BASE = "https://api.inaturalist.org/v1/observations"

# Local output root (same directory as this script)
LOCAL_ROOT = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.join(LOCAL_ROOT, _args.output_folder)

# Metadata output
METADATA_CSV = os.path.join(
    BASE_DIR,
    f"inat_taxon_{TAXON_ID}_{QUALITY}_obs_photo_metadata.csv"
)

# Output folders
IMAGE_BASE = os.path.join(BASE_DIR, "images")
MASK_BASE = os.path.join(BASE_DIR, "masks")
OVERLAY_BASE = os.path.join(BASE_DIR, "overlays")
SEGMENTS_PNGS = os.path.join(BASE_DIR, "segments_pngs")

# Logs
OUT_CSV = os.path.join(BASE_DIR, "mask_summary.csv")
ERR_LOG = os.path.join(BASE_DIR, "errors.txt")

# Batch size
BATCH_SIZE = 10_000

# Download settings
MAX_WORKERS = 1
TIMEOUT_SEC = 60
CHUNK_BYTES = 1024 * 1024
SLEEP_BETWEEN_REQUESTS_SEC = 0.0
SKIP_IF_EXISTS = True

# SAM3 settings
TEXT_PROMPT = _args.prompt if _args.prompt is not None else "open flower"
SCORE_THRESHOLD = 0.85
MASK_THRESHOLD = 0.8
LOG_FAILURES = True

# Overlay settings
MAKE_OVERLAY_FOR_EVERY_SEGMENTED_IMAGE = True
OVERWRITE_EXISTING_OVERLAYS = True

OVERLAY_ALPHA = 95
OVERLAY_COLOR = (255, 0, 0)
BOX_COLOR = (255, 255, 0)
BOX_WIDTH = 4
TEXT_MARGIN = 4

# If True, updates the metadata CSV with avg_RGB values after segmentation.
UPDATE_METADATA_WITH_AVG_RGB = True


# ============================================================
# 3) MAKE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(IMAGE_BASE, exist_ok=True)
os.makedirs(MASK_BASE, exist_ok=True)
os.makedirs(OVERLAY_BASE, exist_ok=True)
os.makedirs(SEGMENTS_PNGS, exist_ok=True)


# ============================================================
# 4) HTTP HELPERS
# ============================================================

def make_session(max_workers: int = 8) -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=8,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=max_workers,
        pool_maxsize=max_workers,
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "User-Agent": "inat-taxon-segmentation-pipeline/4.0",
        "Accept": "application/json",
    })

    return session


def fetch_json(
    session: requests.Session,
    url: str,
    params: Dict[str, Any],
    timeout: int = 60,
) -> Dict[str, Any]:

    response = session.get(url, params=params, timeout=timeout)

    if response.status_code == 429:
        time.sleep(10)
        response = session.get(url, params=params, timeout=timeout)

    response.raise_for_status()
    return response.json()


# ============================================================
# 5) METADATA URL HELPERS
# ============================================================

def infer_ext_from_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None

    try:
        path = urlparse(url).path
        base = os.path.basename(path)

        if "." not in base:
            return None

        ext = base.rsplit(".", 1)[1].lower()

        if ext in {"jpg", "jpeg", "png", "gif"}:
            return ext

        return None

    except Exception:
        return None


def build_s3_original_url(photo_id: Optional[int], ext: Optional[str]) -> Optional[str]:
    if not photo_id or not ext:
        return None

    return f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/original.{ext}"


def best_original_photo_url(photo: Dict[str, Any]) -> Optional[str]:
    photo_id = photo.get("id")
    api_url = photo.get("url")
    api_original = photo.get("original_url")

    ext = infer_ext_from_url(api_original) or infer_ext_from_url(api_url)

    s3_url = build_s3_original_url(photo_id, ext)
    if s3_url:
        return s3_url

    if api_original:
        return api_original

    if api_url:
        for token in ["square", "small", "medium", "large", "original"]:
            if f"/{token}." in api_url:
                return api_url.replace(f"/{token}.", "/original.")

        return api_url

    return None


# ============================================================
# 6) METADATA CSV SCHEMA
# ============================================================

METADATA_COLUMNS = [
    "observation_id",
    "observation_uuid",
    "quality_grade",
    "observed_on",
    "time_observed_at",
    "created_at",
    "updated_at",
    "license_code",
    "geoprivacy",
    "taxon_geoprivacy",
    "location",
    "latitude",
    "longitude",
    "place_guess",
    "captive",
    "identifications_count",
    "comments_count",
    "faves_count",
    "user_id",
    "user_login",
    "taxon_id",
    "taxon_name",
    "taxon_preferred_common_name",
    "taxon_rank",
    "taxon_ancestry",
    "photo_id",
    "photo_uuid",
    "photo_license_code",
    "photo_attribution",
    "photo_width",
    "photo_height",
    "photo_url_original",
]


def rows_from_obs(obs: Dict[str, Any]):
    taxon = obs.get("taxon") or {}
    user = obs.get("user") or {}

    base = {
        "observation_id": obs.get("id"),
        "observation_uuid": obs.get("uuid"),
        "quality_grade": obs.get("quality_grade"),
        "observed_on": obs.get("observed_on"),
        "time_observed_at": obs.get("time_observed_at"),
        "created_at": obs.get("created_at"),
        "updated_at": obs.get("updated_at"),
        "license_code": obs.get("license_code"),
        "geoprivacy": obs.get("geoprivacy"),
        "taxon_geoprivacy": obs.get("taxon_geoprivacy"),
        "location": obs.get("location"),
        "latitude": obs.get("latitude"),
        "longitude": obs.get("longitude"),
        "place_guess": obs.get("place_guess"),
        "captive": obs.get("captive"),
        "identifications_count": obs.get("identifications_count"),
        "comments_count": obs.get("comments_count"),
        "faves_count": obs.get("faves_count"),
        "user_id": user.get("id"),
        "user_login": user.get("login"),
        "taxon_id": taxon.get("id"),
        "taxon_name": taxon.get("name"),
        "taxon_preferred_common_name": taxon.get("preferred_common_name"),
        "taxon_rank": taxon.get("rank"),
        "taxon_ancestry": taxon.get("ancestry"),
    }

    photos = obs.get("photos") or []

    if not photos:
        row = dict(base)

        for col in METADATA_COLUMNS:
            row.setdefault(col, "")

        yield row
        return

    for photo in photos:
        row = dict(base)

        row.update({
            "photo_id": photo.get("id"),
            "photo_uuid": photo.get("uuid"),
            "photo_license_code": photo.get("license_code"),
            "photo_attribution": photo.get("attribution"),
            "photo_width": photo.get("width"),
            "photo_height": photo.get("height"),
            "photo_url_original": best_original_photo_url(photo),
        })

        for col in METADATA_COLUMNS:
            row.setdefault(col, "")

        yield row


def resume_last_obs_id_from_csv(path: str) -> int:
    if not os.path.exists(path):
        return 0

    last_line = None

    with open(path, "rb") as file:
        try:
            file.seek(-65536, os.SEEK_END)
        except OSError:
            file.seek(0)

        lines = file.read().splitlines()

        for line in reversed(lines):
            if line.strip():
                last_line = line.decode("utf-8", errors="ignore")
                break

    if not last_line or last_line.startswith("observation_id,"):
        return 0

    first_cell = last_line.split(",", 1)[0].strip()

    try:
        return int(first_cell)
    except ValueError:
        return 0


# ============================================================
# 7) DOWNLOAD METADATA
# ============================================================

def download_metadata_csv():
    session = make_session(max_workers=8)
    id_above = resume_last_obs_id_from_csv(METADATA_CSV)

    print(f"Metadata resume: id_above={id_above}")

    file_exists = os.path.exists(METADATA_CSV)

    out_file = open(METADATA_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=METADATA_COLUMNS)

    if not file_exists:
        writer.writeheader()

    params_base = {
        "taxon_id": TAXON_ID,
        "quality_grade": QUALITY,
        "per_page": PER_PAGE,
        "order": "asc",
        "order_by": "id",
        "photos": "true",
    }

    obs_count = 0
    row_count = 0

    try:
        while True:
            params = dict(params_base)

            if id_above > 0:
                params["id_above"] = id_above

            data = fetch_json(session, API_BASE, params)
            results = data.get("results", [])

            if not results:
                print("No more metadata results returned. Done.")
                break

            last_id = None
            batch_obs = 0
            batch_rows = 0

            for obs in results:
                last_id = obs.get("id", last_id)
                batch_obs += 1

                for row in rows_from_obs(obs):
                    writer.writerow(row)
                    batch_rows += 1

            out_file.flush()

            obs_count += batch_obs
            row_count += batch_rows

            if last_id is None:
                print("Warning: metadata batch had no observation IDs. Stopping.")
                break

            id_above = int(last_id)

            print(
                f"metadata last_obs_id={id_above}  "
                f"batch_obs={batch_obs:,}  "
                f"batch_rows={batch_rows:,}  "
                f"total_obs={obs_count:,}  "
                f"total_rows={row_count:,}"
            )

            time.sleep(DELAY_SEC)

    finally:
        out_file.close()

    print("Metadata written to:", METADATA_CSV)


# ============================================================
# 8) PATH HELPERS
# ============================================================

def sanitize_token(value: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9_]+", "", value)
    return value


def genus_species_from_taxon_name(taxon_name: str) -> str:
    if not isinstance(taxon_name, str) or not taxon_name.strip():
        return "Unknown_unknown"

    parts = taxon_name.strip().split()

    if len(parts) >= 2:
        genus = parts[0].capitalize()
        species = parts[1].lower()
        return f"{sanitize_token(genus)}_{sanitize_token(species)}"

    return sanitize_token(taxon_name)


def infer_image_ext_from_url(url: str) -> str:
    if not isinstance(url, str) or not url:
        return "jpg"

    base = url.split("?", 1)[0]
    _, ext = os.path.splitext(base)
    ext = ext.lower().lstrip(".")

    if ext in {"jpg", "jpeg", "png"}:
        return ext

    return "jpg"


def build_filename(
    genus_species: str,
    observation_id: int,
    image_index: int,
    ext: str,
) -> str:

    return f"{genus_species}_{observation_id}_image{image_index}.{ext}"


def batch_name_from_global_index(global_index: int) -> str:
    batch_num = (int(global_index) - 1) // BATCH_SIZE + 1
    return f"batch_{batch_num:05d}"


def build_paths_for_row(row):
    global_index = int(row["global_index"])
    batch_name = batch_name_from_global_index(global_index)

    url = str(row["photo_url_original"])
    ext = infer_image_ext_from_url(url)

    image_name = build_filename(
        genus_species=row["genus_species"],
        observation_id=int(row["observation_id"]),
        image_index=int(row["image_index"]),
        ext=ext,
    )

    stem = os.path.splitext(image_name)[0]

    image_dir = os.path.join(IMAGE_BASE, batch_name)
    mask_dir = os.path.join(MASK_BASE, batch_name)
    overlay_dir = os.path.join(OVERLAY_BASE, batch_name)

    image_path = os.path.join(image_dir, image_name)
    overlay_name = f"{stem}_overlay.png"
    overlay_path = os.path.join(overlay_dir, overlay_name)

    image_relpath = os.path.relpath(image_path, BASE_DIR)
    mask_batch_relpath = os.path.relpath(mask_dir, BASE_DIR)
    overlay_relpath = os.path.relpath(overlay_path, BASE_DIR)

    return pd.Series({
        "batch_name": batch_name,
        "image_name": image_name,
        "stem": stem,
        "image_dir": image_dir,
        "mask_dir": mask_dir,
        "overlay_dir": overlay_dir,
        "image_path": image_path,
        "overlay_path": overlay_path,
        "image_relpath": image_relpath,
        "mask_batch_relpath": mask_batch_relpath,
        "overlay_relpath_expected": overlay_relpath,
    })


# ============================================================
# 9) IMAGE, MASK, RGB, AND OVERLAY HELPERS
# ============================================================

def write_bytes_atomic(out_path: str, data: bytes):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    temp_path = out_path + ".part"

    with open(temp_path, "wb") as file:
        file.write(data)

    os.replace(temp_path, out_path)


def download_image_bytes(session: requests.Session, url: str) -> bytes:
    response = session.get(url, stream=True, timeout=TIMEOUT_SEC)

    if response.status_code == 429:
        time.sleep(10)
        response = session.get(url, stream=True, timeout=TIMEOUT_SEC)

    response.raise_for_status()

    chunks = []

    for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
        if chunk:
            chunks.append(chunk)

    if SLEEP_BETWEEN_REQUESTS_SEC > 0:
        time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)

    return b"".join(chunks)


def tensor_to_uint8_mask(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    else:
        mask = np.array(mask)

    mask = np.squeeze(mask)

    return (mask > 0).astype(np.uint8)


def mask_to_bbox(mask):
    mask = np.squeeze(mask)
    mask_bool = mask > 0

    ys, xs = np.where(mask_bool)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())

    return x1, y1, x2, y2


def resize_mask_if_needed(mask, target_size):
    mask = np.squeeze(mask)

    target_w, target_h = target_size
    mask_h, mask_w = mask.shape

    if (mask_w, mask_h) == (target_w, target_h):
        return mask

    mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    mask_img = mask_img.resize((target_w, target_h), resample=Image.NEAREST)

    return (np.array(mask_img) > 0).astype(np.uint8)


def sort_instances_by_confidence(masks_list, scores_list):
    paired = list(zip(masks_list, scores_list))
    paired = sorted(paired, key=lambda item: float(item[1]), reverse=True)

    if len(paired) == 0:
        return [], []

    sorted_masks = [item[0] for item in paired]
    sorted_scores = [float(item[1]) for item in paired]

    return sorted_masks, sorted_scores


def average_rgb_for_mask(image_rgb_np, mask_np):
    mask_np = np.squeeze(mask_np)
    mask_bool = mask_np > 0

    if not mask_bool.any():
        return None

    pixels = image_rgb_np[mask_bool]

    if pixels.size == 0:
        return None

    avg = pixels.mean(axis=0)

    return (
        int(round(float(avg[0]))),
        int(round(float(avg[1]))),
        int(round(float(avg[2]))),
    )


def rgb_tuple_to_string(rgb_tuple):
    if rgb_tuple is None:
        return ""

    r, g, b = rgb_tuple
    return f"({r},{g},{b})"


def get_font(size=22):
    possible_fonts = [
        # Windows
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/DejaVuSans-Bold.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]

    for font_path in possible_fonts:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size=size)

    return ImageFont.load_default()


def draw_label(draw, xy, text, font):
    x, y = xy

    try:
        bbox = draw.textbbox((x, y), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        text_w, text_h = draw.textsize(text, font=font)

    bg_x1 = x
    bg_y1 = y
    bg_x2 = x + text_w + 2 * TEXT_MARGIN
    bg_y2 = y + text_h + 2 * TEXT_MARGIN

    draw.rectangle(
        [bg_x1, bg_y1, bg_x2, bg_y2],
        fill=(0, 0, 0, 180),
    )

    draw.text(
        (x + TEXT_MARGIN, y + TEXT_MARGIN),
        text,
        fill=(255, 255, 255, 255),
        font=font,
    )


def create_overlay_from_saved_masks(
    image_path,
    mask_files,
    scores,
    output_path,
):
    image = Image.open(image_path).convert("RGBA")
    width, height = image.size

    mask_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(draw_layer)

    font_size = max(16, int(min(width, height) * 0.025))
    font = get_font(size=font_size)

    valid_instances = 0

    for instance_index, mask_path in enumerate(mask_files):
        try:
            mask = np.load(mask_path)
            mask = resize_mask_if_needed(mask, image.size)

            mask_bool = mask > 0

            if not mask_bool.any():
                continue

            valid_instances += 1

            alpha_mask = mask_bool.astype(np.uint8) * OVERLAY_ALPHA
            alpha_img = Image.fromarray(alpha_mask, mode="L")

            color_img = Image.new("RGBA", image.size, OVERLAY_COLOR + (0,))
            color_img.putalpha(alpha_img)

            mask_layer = Image.alpha_composite(mask_layer, color_img)

            bbox = mask_to_bbox(mask)

            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox

            for offset in range(BOX_WIDTH):
                draw.rectangle(
                    [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                    outline=BOX_COLOR + (255,),
                )

            if instance_index < len(scores):
                label = f"{TEXT_PROMPT} {scores[instance_index]:.3f}"
            else:
                label = TEXT_PROMPT

            label_y = max(0, y1 - font_size - 12)
            draw_label(draw, (x1, label_y), label, font)

        except Exception as error:
            print(f"Warning: could not process mask {mask_path}: {error}")

    if valid_instances == 0:
        draw_label(draw, (10, 10), "No detections", font)

    composite = Image.alpha_composite(image, mask_layer)
    composite = Image.alpha_composite(composite, draw_layer)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    composite.convert("RGB").save(output_path, quality=95)

    return valid_instances


# ============================================================
# 10) LOGGING HELPERS
# ============================================================

CSV_HEADER = [
    "global_index",
    "batch_name",
    "image_relpath",
    "mask_batch_relpath",
    "overlay_relpath",
    "image_name",
    "observation_id",
    "image_index",
    "taxon_name",
    "photo_url_original",
    "num_instances",
    "mask_files",
    "mask_scores",
    "avg_RGB",
    "status",
    "error",
]


def ensure_csv_header(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(CSV_HEADER)
            file.flush()
            os.fsync(file.fileno())


def append_csv_row(path, row_list):
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(row_list)
        file.flush()
        os.fsync(file.fileno())


def load_done_global_indices(path) -> set:
    done = set()

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return done

    try:
        df_done = pd.read_csv(path, usecols=["global_index", "status"])
        df_done = df_done[df_done["status"].astype(str).isin(["ok", "dl_failed", "seg_failed"])]

        for value in df_done["global_index"].dropna().tolist():
            try:
                done.add(int(value))
            except Exception:
                pass

    except Exception:
        return set()

    return done


def log_error(line: str):
    os.makedirs(os.path.dirname(ERR_LOG), exist_ok=True)

    with open(ERR_LOG, "a", encoding="utf-8") as file:
        file.write(line.strip() + "\n")


# ============================================================
# 11) BUILD PROCESSING DATAFRAME
# ============================================================

def build_processing_dataframe():
    print("Loading metadata CSV:", METADATA_CSV)

    df = pd.read_csv(METADATA_CSV)

    needed = {"observation_id", "taxon_name", "photo_url_original"}
    missing = needed - set(df.columns)

    if missing:
        raise ValueError(f"Metadata CSV missing required columns: {missing}")

    df = df[
        df["photo_url_original"].notna()
        & (df["photo_url_original"].astype(str).str.len() > 0)
    ].copy()

    df["observation_id"] = pd.to_numeric(
        df["observation_id"],
        errors="coerce"
    ).astype("Int64")

    df = df[df["observation_id"].notna()].copy()

    df["image_index"] = df.groupby("observation_id").cumcount() + 1
    df["genus_species"] = df["taxon_name"].astype(str).apply(genus_species_from_taxon_name)

    df = df.reset_index(drop=True)
    df["global_index"] = np.arange(1, len(df) + 1, dtype=int)

    path_df = df.apply(build_paths_for_row, axis=1)
    df = pd.concat([df, path_df], axis=1)

    return df


# ============================================================
# 12) UPDATE METADATA CSV WITH AVG_RGB
# ============================================================

def update_metadata_csv_with_avg_rgb():
    if not UPDATE_METADATA_WITH_AVG_RGB:
        return

    if not os.path.exists(METADATA_CSV):
        print("Metadata CSV not found. Cannot add avg_RGB.")
        return

    if not os.path.exists(OUT_CSV):
        print("mask_summary.csv not found. Cannot add avg_RGB.")
        return

    metadata_df = pd.read_csv(METADATA_CSV)
    summary_df = pd.read_csv(OUT_CSV)

    if "avg_RGB" not in summary_df.columns:
        print("mask_summary.csv does not contain avg_RGB yet.")
        return

    if "status" not in summary_df.columns:
        print("mask_summary.csv does not contain status column.")
        return

    metadata_df["observation_id"] = pd.to_numeric(
        metadata_df["observation_id"],
        errors="coerce"
    ).astype("Int64")

    metadata_df["image_index"] = metadata_df.groupby("observation_id").cumcount() + 1

    summary_df = summary_df[summary_df["status"].astype(str).eq("ok")].copy()

    required = {
        "observation_id",
        "image_index",
        "photo_url_original",
        "avg_RGB",
    }

    missing = required - set(summary_df.columns)

    if missing:
        print(f"Cannot update metadata CSV. Missing columns in mask_summary.csv: {missing}")
        return

    summary_df = summary_df[
        [
            "observation_id",
            "image_index",
            "photo_url_original",
            "avg_RGB",
        ]
    ].copy()

    summary_df["observation_id"] = pd.to_numeric(
        summary_df["observation_id"],
        errors="coerce"
    ).astype("Int64")

    summary_df["image_index"] = pd.to_numeric(
        summary_df["image_index"],
        errors="coerce"
    ).astype("Int64")

    if "avg_RGB" in metadata_df.columns:
        metadata_df = metadata_df.drop(columns=["avg_RGB"])

    metadata_df = metadata_df.merge(
        summary_df,
        on=["observation_id", "image_index", "photo_url_original"],
        how="left",
    )

    metadata_df.to_csv(METADATA_CSV, index=False)

    filled = metadata_df["avg_RGB"].notna().sum()

    print(f"Updated metadata CSV with avg_RGB values for {filled:,} rows.")
    print("Updated metadata CSV:", METADATA_CSV)


# ============================================================
# 13) SEGMENTATION PIPELINE
# ============================================================

def run_segmentation():
    df = build_processing_dataframe()

    print("Total metadata image rows:", f"{len(df):,}")
    print("Batch size:", f"{BATCH_SIZE:,}")
    print("Base output:", BASE_DIR)
    print("Images:", IMAGE_BASE)
    print("Masks:", MASK_BASE)
    print("Overlays:", OVERLAY_BASE)

    ensure_csv_header(OUT_CSV)

    done = load_done_global_indices(OUT_CSV)
    print("Already logged:", f"{len(done):,}")

    df_todo = df[~df["global_index"].isin(done)].copy()
    if LIMIT is not None:
        df_todo = df_todo.head(LIMIT)
    print("Remaining to segment:", f"{len(df_todo):,}")

    if len(df_todo) == 0:
        print("Nothing left to segment.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model.eval()

    session = make_session(max_workers=MAX_WORKERS)

    def download_job(row_dict):
        global_index = int(row_dict["global_index"])
        url = str(row_dict["photo_url_original"])
        image_path = str(row_dict["image_path"])

        if SKIP_IF_EXISTS and os.path.exists(image_path) and os.path.getsize(image_path) > 0:
            return (global_index, "exists", None, None)

        try:
            data = download_image_bytes(session, url)
            return (global_index, "downloaded", data, None)

        except Exception as error:
            return (global_index, "dl_failed", None, str(error))

    prefetch = min(MAX_WORKERS * 2, 128)

    todo_records = df_todo.to_dict(orient="records")
    total_todo = len(todo_records)
    row_by_global_index = {int(row["global_index"]): row for row in todo_records}

    progress = tqdm(total=total_todo, desc="Download + SAM3 segment", dynamic_ncols=True)

    ok_count = 0
    dl_failed = 0
    seg_failed = 0
    skipped_existing_download = 0
    overlay_count = 0
    no_detection_overlay_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        iterator = iter(todo_records)
        in_flight = {}

        for _ in range(min(prefetch, total_todo)):
            row = next(iterator, None)

            if row is None:
                break

            future = executor.submit(download_job, row)
            in_flight[future] = int(row["global_index"])

        while in_flight:
            completed_futures = []

            for future in as_completed(list(in_flight.keys()), timeout=None):
                completed_futures.append(future)
                break

            for future in completed_futures:
                global_index = in_flight.pop(future)
                row = row_by_global_index[global_index]

                batch_name = row["batch_name"]
                image_name = row["image_name"]
                stem = row["stem"]

                image_path = row["image_path"]
                image_dir = row["image_dir"]
                mask_dir = row["mask_dir"]
                overlay_dir = row["overlay_dir"]
                overlay_path = row["overlay_path"]

                image_relpath = row["image_relpath"]
                mask_batch_relpath = row["mask_batch_relpath"]
                overlay_relpath = ""

                obs_id = int(row["observation_id"])
                image_index = int(row["image_index"])
                taxon_name = str(row["taxon_name"])
                url = str(row["photo_url_original"])

                try:
                    returned_global_index, status, data, error = future.result()
                    assert returned_global_index == global_index

                    if status == "dl_failed":
                        dl_failed += 1

                        if LOG_FAILURES:
                            append_csv_row(OUT_CSV, [
                                global_index,
                                batch_name,
                                image_relpath,
                                mask_batch_relpath,
                                overlay_relpath,
                                image_name,
                                obs_id,
                                image_index,
                                taxon_name,
                                url,
                                0,
                                "",
                                "",
                                "",
                                "dl_failed",
                                error or "",
                            ])

                            log_error(
                                f"[DL_FAIL] global_index={global_index} url={url} error={error}"
                            )

                        progress.update(1)

                    else:
                        os.makedirs(image_dir, exist_ok=True)
                        os.makedirs(mask_dir, exist_ok=True)
                        os.makedirs(overlay_dir, exist_ok=True)

                        if status == "downloaded" and data is not None:
                            write_bytes_atomic(image_path, data)

                        elif status == "exists":
                            skipped_existing_download += 1

                        if status == "downloaded" and data is not None:
                            image_pil = Image.open(BytesIO(data)).convert("RGB")
                        else:
                            image_pil = Image.open(image_path).convert("RGB")

                        inputs = processor(
                            images=image_pil,
                            text=TEXT_PROMPT,
                            return_tensors="pt",
                        ).to(device)

                        with torch.no_grad():
                            outputs = model(**inputs)

                        post = processor.post_process_instance_segmentation(
                            outputs,
                            threshold=SCORE_THRESHOLD,
                            mask_threshold=MASK_THRESHOLD,
                            target_sizes=inputs.get("original_sizes").tolist(),
                        )[0]

                        masks = post.get("masks", None)
                        scores = post.get("scores", None)

                        if masks is None or scores is None:
                            masks_list = []
                            scores_list = []
                        else:
                            if isinstance(masks, torch.Tensor):
                                masks_list = [masks[i] for i in range(masks.shape[0])]
                            else:
                                masks_list = list(masks)

                            if isinstance(scores, torch.Tensor):
                                scores_list = scores.detach().cpu().tolist()
                            else:
                                scores_list = list(scores)

                        masks_list, scores_list = sort_instances_by_confidence(
                            masks_list,
                            scores_list,
                        )

                        image_rgb_np = np.array(image_pil.convert("RGB"))

                        saved_mask_files = []
                        saved_mask_relpaths = []
                        avg_rgb_values = []

                        for instance_index, mask in enumerate(masks_list):
                            mask_np = tensor_to_uint8_mask(mask)
                            mask_np = resize_mask_if_needed(mask_np, image_pil.size)

                            avg_rgb = average_rgb_for_mask(image_rgb_np, mask_np)
                            avg_rgb_values.append(avg_rgb)

                            mask_name = f"{stem}_instance_{instance_index}.npy"
                            mask_path = os.path.join(mask_dir, mask_name)

                            np.save(mask_path, mask_np)

                            saved_mask_files.append(mask_path)
                            saved_mask_relpaths.append(os.path.relpath(mask_path, BASE_DIR))

                        num_instances = len(saved_mask_files)

                        mask_scores_str = ",".join(
                            f"{float(score):.4f}" for score in scores_list
                        )

                        mask_files_str = ";".join(saved_mask_relpaths)

                        avg_rgb_str = ",".join(
                            rgb_tuple_to_string(rgb_value)
                            for rgb_value in avg_rgb_values
                            if rgb_value is not None
                        )

                        if MAKE_OVERLAY_FOR_EVERY_SEGMENTED_IMAGE:
                            if os.path.exists(overlay_path) and not OVERWRITE_EXISTING_OVERLAYS:
                                overlay_relpath = os.path.relpath(overlay_path, BASE_DIR)
                            else:
                                valid_instances = create_overlay_from_saved_masks(
                                    image_path=image_path,
                                    mask_files=saved_mask_files,
                                    scores=scores_list,
                                    output_path=overlay_path,
                                )

                                overlay_relpath = os.path.relpath(overlay_path, BASE_DIR)
                                overlay_count += 1

                                if valid_instances == 0:
                                    no_detection_overlay_count += 1

                        append_csv_row(OUT_CSV, [
                            global_index,
                            batch_name,
                            image_relpath,
                            mask_batch_relpath,
                            overlay_relpath,
                            image_name,
                            obs_id,
                            image_index,
                            taxon_name,
                            url,
                            int(num_instances),
                            mask_files_str,
                            mask_scores_str,
                            avg_rgb_str,
                            "ok",
                            "",
                        ])

                        ok_count += 1
                        progress.update(1)

                except Exception as error:
                    seg_failed += 1

                    if LOG_FAILURES:
                        append_csv_row(OUT_CSV, [
                            global_index,
                            batch_name,
                            image_relpath,
                            mask_batch_relpath,
                            overlay_relpath,
                            image_name,
                            obs_id,
                            image_index,
                            taxon_name,
                            url,
                            0,
                            "",
                            "",
                            "",
                            "seg_failed",
                            str(error),
                        ])

                        log_error(
                            f"[SEG_FAIL] global_index={global_index} image={image_path} error={error}"
                        )

                    progress.update(1)

                next_row = next(iterator, None)

                if next_row is not None:
                    next_future = executor.submit(download_job, next_row)
                    in_flight[next_future] = int(next_row["global_index"])

    progress.close()

    print("\nDone.")
    print("Logged OK:", f"{ok_count:,}")
    print("Download failed:", f"{dl_failed:,}")
    print("Segmentation failed:", f"{seg_failed:,}")
    print("Skipped download because image already existed:", f"{skipped_existing_download:,}")
    print("Overlays created:", f"{overlay_count:,}")
    print("Overlays with no detections:", f"{no_detection_overlay_count:,}")
    print("Metadata CSV:", METADATA_CSV)
    print("Segmentation CSV:", OUT_CSV)
    print("Images:", IMAGE_BASE)
    print("Masks:", MASK_BASE)
    print("Overlays:", OVERLAY_BASE)
    print("Error log:", ERR_LOG)


# ============================================================
# 14) RUN PIPELINE
# ============================================================

if __name__ == "__main__":
    download_metadata_csv()
    run_segmentation()
    update_metadata_csv_with_avg_rgb()
