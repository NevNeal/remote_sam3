#!/usr/bin/env python3
"""
Download the highest-resolution (original) iNaturalist photos for a taxon,
plus the observation metadata CSV. No segmentation, no SAM3, no torch.

The download path mirrors segmentation_pipeline.py: the same retrying
requests.Session, the same S3 "original.<ext>" URL builder, and the same
streaming byte download with 429 back-off.

Images are written as:
    <output>/obs_{observation_id}_img_{n}.{ext}      (n starts at 1)

Metadata is written to the same folder as:
    <output>/inat_taxon_{taxon_id}_obs_photo_metadata.csv

Usage:
    python download_images_only.py --taxon-id 631312 --output "~/Desktop/blood-drop emlets"
    python download_images_only.py --taxon-id 631312 --research-grade-only
    python download_images_only.py --taxon-id 631312 --limit 50 --metadata-only
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional here
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else _NullBar()

    class _NullBar:
        def update(self, n=1):
            pass

        def set_postfix(self, *args, **kwargs):
            pass

        def close(self):
            pass


# ============================================================
# 1) SETTINGS
# ============================================================

API_BASE = "https://api.inaturalist.org/v1/observations"
PER_PAGE = 200
DELAY_SEC = 1.1

TIMEOUT_SEC = 60
CHUNK_BYTES = 1024 * 1024
SLEEP_BETWEEN_REQUESTS_SEC = 0.0
SKIP_IF_EXISTS = True


# ============================================================
# 2) HTTP HELPERS  (same shape as segmentation_pipeline.make_session)
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
        "User-Agent": "inat-taxon-image-downloader/1.0",
        "Accept": "application/json",
    })

    return session


def fetch_json(session: requests.Session, url: str, params: Dict[str, Any],
               timeout: int = 60) -> Dict[str, Any]:
    response = session.get(url, params=params, timeout=timeout)

    if response.status_code == 429:
        time.sleep(10)
        response = session.get(url, params=params, timeout=timeout)

    response.raise_for_status()
    return response.json()


# ============================================================
# 3) URL HELPERS -- always resolve to the original (largest) photo
# ============================================================

def infer_ext_from_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None

    try:
        base = os.path.basename(urlparse(url).path)

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


def infer_image_ext_from_url(url: str) -> str:
    if not isinstance(url, str) or not url:
        return "jpg"

    _, ext = os.path.splitext(url.split("?", 1)[0])
    ext = ext.lower().lstrip(".")

    if ext in {"jpg", "jpeg", "png"}:
        return ext

    return "jpg"


# ============================================================
# 4) METADATA
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
    "photo_index",
    "image_filename",
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

    for index, photo in enumerate(photos, start=1):
        url = best_original_photo_url(photo)

        row = dict(base)
        row.update({
            "photo_id": photo.get("id"),
            "photo_uuid": photo.get("uuid"),
            "photo_license_code": photo.get("license_code"),
            "photo_attribution": photo.get("attribution"),
            "photo_width": photo.get("width"),
            "photo_height": photo.get("height"),
            "photo_url_original": url,
            "photo_index": index,
            "image_filename": build_filename(obs.get("id"), index, url),
        })

        for col in METADATA_COLUMNS:
            row.setdefault(col, "")

        yield row


def build_filename(observation_id, image_index: int, url: Optional[str]) -> str:
    ext = infer_image_ext_from_url(url or "")
    return f"obs_{observation_id}_img_{image_index}.{ext}"


def download_metadata_csv(session: requests.Session, taxon_id: int,
                          metadata_csv: str, quality_grade: Optional[str]):
    """Page the observations API and write one CSV row per photo."""
    out_file = open(metadata_csv, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=METADATA_COLUMNS)
    writer.writeheader()

    params_base = {
        "taxon_id": taxon_id,
        "per_page": PER_PAGE,
        "order": "asc",
        "order_by": "id",
        "photos": "true",
    }

    if quality_grade:
        params_base["quality_grade"] = quality_grade

    id_above = 0
    obs_count = 0
    row_count = 0
    rows = []

    try:
        while True:
            params = dict(params_base)

            if id_above > 0:
                params["id_above"] = id_above

            data = fetch_json(session, API_BASE, params)
            results = data.get("results", [])

            if not results:
                break

            last_id = None

            for obs in results:
                last_id = obs.get("id", last_id)
                obs_count += 1

                for row in rows_from_obs(obs):
                    writer.writerow(row)
                    rows.append(row)
                    row_count += 1

            out_file.flush()

            if last_id is None:
                print("Warning: metadata batch had no observation IDs. Stopping.")
                break

            id_above = int(last_id)

            print(f"metadata last_obs_id={id_above}  "
                  f"total_obs={obs_count:,}  total_photo_rows={row_count:,}")

            time.sleep(DELAY_SEC)

    finally:
        out_file.close()

    print(f"Metadata written to: {metadata_csv}  "
          f"({obs_count:,} observations, {row_count:,} photo rows)")

    return rows


# ============================================================
# 5) IMAGE DOWNLOAD  (same method as segmentation_pipeline)
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


def download_all_images(session: requests.Session, rows, out_dir: str,
                        max_workers: int, limit: Optional[int],
                        error_log: str):
    todo = [row for row in rows if row.get("photo_url_original")]

    if limit is not None:
        todo = todo[:limit]

    print(f"Images to fetch: {len(todo):,}")

    ok = 0
    skipped = 0
    failed = 0

    def job(row):
        image_path = os.path.join(out_dir, row["image_filename"])

        if SKIP_IF_EXISTS and os.path.exists(image_path) and os.path.getsize(image_path) > 0:
            return ("exists", row, None)

        try:
            data = download_image_bytes(session, str(row["photo_url_original"]))
            write_bytes_atomic(image_path, data)
            return ("downloaded", row, None)

        except Exception as error:
            return ("failed", row, str(error))

    progress = tqdm(total=len(todo), desc="Download originals", dynamic_ncols=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for status, row, error in executor.map(job, todo):
            if status == "downloaded":
                ok += 1
            elif status == "exists":
                skipped += 1
            else:
                failed += 1
                with open(error_log, "a", encoding="utf-8") as log:
                    log.write(f"{row['image_filename']}\t{row['photo_url_original']}\t{error}\n")

            progress.update(1)
            progress.set_postfix(ok=ok, skipped=skipped, failed=failed)

    progress.close()

    print(f"Downloaded: {ok:,}   Already present: {skipped:,}   Failed: {failed:,}")

    if failed:
        print(f"Failures logged to: {error_log}")


# ============================================================
# 6) MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--taxon-id", type=int, required=True,
                        help="iNaturalist taxon_id to download")
    parser.add_argument("--output", required=True,
                        help="Output folder for images + metadata CSV")
    parser.add_argument("--research-grade-only", action="store_true",
                        help="Only include research-grade observations")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many images (metadata is still complete)")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Parallel image downloads (default: 8)")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Write the metadata CSV, skip image download")
    args = parser.parse_args()

    out_dir = os.path.abspath(os.path.expanduser(args.output))
    os.makedirs(out_dir, exist_ok=True)

    quality_grade = "research" if args.research_grade_only else None
    suffix = "research" if args.research_grade_only else "all"

    metadata_csv = os.path.join(
        out_dir, f"inat_taxon_{args.taxon_id}_{suffix}_obs_photo_metadata.csv"
    )
    error_log = os.path.join(out_dir, "download_errors.txt")

    print(f"Taxon ID      : {args.taxon_id}")
    print(f"Quality grade : {quality_grade or 'any'}")
    print(f"Output folder : {out_dir}")

    session = make_session(max_workers=args.max_workers)

    rows = download_metadata_csv(session, args.taxon_id, metadata_csv, quality_grade)

    if not rows:
        print("No observations found. Nothing to download.")
        return

    if args.metadata_only:
        print("--metadata-only set; skipping image download.")
        return

    download_all_images(session, rows, out_dir, args.max_workers, args.limit, error_log)


if __name__ == "__main__":
    sys.exit(main())
