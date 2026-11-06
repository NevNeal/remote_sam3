#!/usr/bin/env python3
"""
The 100 top segments of each of the 30 families with the most pv99 photos.

    python all_genera/pv99/local/top_segments.py  [--per-family 100]

A photo's segment is its single most confident mask. A family's candidates are
pack_select.segment_pool() (its 200 photos with the highest conf_max),
taken in score order with at most --genus-cap (10) from any one genus, so a
family of many genera is not represented by one showy plant; if that leaves
fewer than --per-family, the rest are filled in score order without the cap.
Writes, per family,

    export/pv99/analysis/top_segments/<Family>/<rank>_<Genus>_<photo_id>.png   cut-out, transparent
    export/pv99/analysis/top_segments/<Family>/context/<photo_id>.jpg          box + 25% margin, 512 px
    export/pv99/analysis/top_segments/sheets/<Family>.jpg                       10 x 10 contact sheet
    export/pv99/analysis/top_segments/top_segments.csv                          one row per segment
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

MARGIN = 0.25         # context around the mask box, share of its long side
MIN_FRAC = 0.35       # context crop is never smaller than this share of the short side
SIZE = 512            # context crops are letterboxed to SIZE x SIZE
PAD_PX = 6            # transparent border around a cut-out
TILE = 180
SHEET_COLS = 10
SHEET_BG = (24, 24, 27)
TILE_BG = (96, 96, 102)


def choose(pool, per_family, cap):
    """Score order, at most `cap` per genus, then fill without the cap."""
    pool = pool.sort_values(["conf_max", "photo_id"], ascending=[False, True])
    picked, counts = [], {}
    for idx, row in pool.iterrows():
        if counts.get(row["genus"], 0) < cap:
            picked.append(idx)
            counts[row["genus"]] = counts.get(row["genus"], 0) + 1
        if len(picked) == per_family:
            break
    if len(picked) < per_family:
        rest = [i for i in pool.index if i not in set(picked)]
        picked += rest[:per_family - len(picked)]
    out = pool.loc[picked].copy()
    out = out.sort_values(["conf_max", "photo_id"], ascending=[False, True])
    out["seg_rank"] = range(1, len(out) + 1)
    return out


def mask_box(mask):
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def context_box(box, width, height, margin=MARGIN, min_frac=MIN_FRAC):
    x0, y0, x1, y1 = box
    pad = margin * max(x1 - x0, y1 - y0)
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    least = min_frac * min(width, height)
    out = []
    for a, b, limit in ((x0, x1, width), (y0, y1, height)):
        if b - a < least:
            mid = (a + b) / 2
            a, b = mid - least / 2, mid + least / 2
        if a < 0:
            a, b = 0, b - a
        if b > limit:
            a, b = a - (b - limit), limit
        out.append((max(0.0, a), min(float(limit), b)))
    (x0, x1), (y0, y1) = out
    return int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1))


def letterbox(image, size=SIZE, fill=(128, 128, 128)):
    w, h = image.size
    scale = size / max(w, h)
    image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def extract(job):
    project, out_dir, row = job
    fam = common.safe(row["family"])
    cut_path = Path(out_dir) / fam / \
        f"{row['seg_rank']:03d}_{common.safe(row['genus'])}_{int(row['photo_id'])}.png"
    ctx_path = Path(out_dir) / fam / "context" / f"{int(row['photo_id'])}.jpg"
    try:
        image = Image.open(common.local_image(project, row["image_path"])).convert("RGB")
        masks, scores = common.photo_masks(project, row["mask_path"])
        mask = masks[0]
        if mask.shape != (image.height, image.width):
            mask = np.asarray(Image.fromarray(mask).resize(image.size, Image.NEAREST), bool)
        box = mask_box(mask)
        x0, y0, x1, y1 = box
        x0, y0 = max(0, x0 - PAD_PX), max(0, y0 - PAD_PX)
        x1, y1 = min(image.width, x1 + PAD_PX), min(image.height, y1 + PAD_PX)
        rgba = image.convert("RGBA")
        rgba.putalpha(Image.fromarray((mask * 255).astype(np.uint8)))
        cut_path.parent.mkdir(parents=True, exist_ok=True)
        rgba.crop((x0, y0, x1, y1)).save(cut_path)
        ctx_path.parent.mkdir(parents=True, exist_ok=True)
        letterbox(image.crop(context_box(box, image.width, image.height))).save(
            ctx_path, quality=92)
        return dict(photo_id=int(row["photo_id"]), cutout=str(cut_path),
                    context=str(ctx_path), seg_score=float(scores[0]),
                    mask_area_frac=float(mask.mean()),
                    box_w_frac=(box[2] - box[0]) / image.width,
                    box_h_frac=(box[3] - box[1]) / image.height, error="")
    except Exception as exc:
        return dict(photo_id=int(row["photo_id"]), error=f"{type(exc).__name__}: {exc}"[:200])


def _font(size):
    for name in ("DejaVuSans.ttf", "arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def contact_sheet(family, rows, out_path):
    rows = rows.sort_values("seg_rank")
    n_rows = int(np.ceil(len(rows) / SHEET_COLS))
    head = 48
    sheet = Image.new("RGB", (SHEET_COLS * TILE, head + n_rows * TILE), SHEET_BG)
    draw = ImageDraw.Draw(sheet)
    draw.text((10, head // 2), f"{family}: top {len(rows)} segments "
              f"({rows['genus'].nunique()} genera, score {rows['seg_score'].min():.3f}"
              f"-{rows['seg_score'].max():.3f})", fill=(235, 235, 235), font=_font(22),
              anchor="lm")
    small = _font(13)
    for i, r in enumerate(rows.itertuples()):
        x, y = (i % SHEET_COLS) * TILE, head + (i // SHEET_COLS) * TILE
        tile = Image.new("RGBA", (TILE - 4, TILE - 4), TILE_BG + (255,))
        cut = Image.open(r.cutout)
        cut.thumbnail((TILE - 10, TILE - 26))
        tile.alpha_composite(cut, ((TILE - 4 - cut.width) // 2, 3 + (TILE - 30 - cut.height) // 2))
        sheet.paste(tile.convert("RGB"), (x + 2, y + 2))
        draw.text((x + 6, y + TILE - 6), f"{r.seg_rank} {r.genus}"[:22], fill=(240, 240, 240),
                  font=small, anchor="ls")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=90)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_args(parser)
    parser.add_argument("--per-family", type=int, default=100)
    parser.add_argument("--genus-cap", type=int, default=10)
    parser.add_argument("--workers", type=int, default=common.n_workers())
    args = parser.parse_args()

    project = common.project_dir(args.project)
    out_dir = common.analysis_dir(project, "top_segments")
    meta = common.load_metadata(project)
    pool = common.segment_selection(meta)
    family_photos = meta["family"].value_counts()

    chosen = pd.concat([choose(g, args.per_family, args.genus_cap)
                        for _, g in pool.groupby("family")], ignore_index=True)
    chosen["family_photos"] = chosen["family"].map(family_photos)
    chosen["family_rank"] = chosen["family_photos"].rank(method="dense", ascending=False)
    jobs = [(project, out_dir, r) for r in chosen.to_dict("records")]
    with ProcessPoolExecutor(args.workers) as ex:
        got = pd.DataFrame(list(ex.map(extract, jobs, chunksize=16)))
    # The metadata has its own `error` column (segment.py's); keep extract()'s.
    table = chosen.drop(columns=["error"], errors="ignore").merge(got, on="photo_id")
    table["error"] = table["error"].fillna("")
    bad = table[table["error"] != ""]
    table = table[table["error"] == ""]
    cols = ["family", "family_rank", "family_photos", "seg_rank", "photo_id", "genus",
            "scientific_name", "seg_score", "conf_max", "num_masks", "mask_area_frac",
            "box_w_frac", "box_h_frac", "image_path", "mask_path", "cutout", "context"]
    table = table[cols].sort_values(["family_rank", "seg_rank"])
    table.to_csv(out_dir / "top_segments.csv", index=False)
    for family, g in table.groupby("family"):
        contact_sheet(family, g, out_dir / "sheets" / f"{common.safe(family)}.jpg")
    print(f"segments   : {len(table):,} from {table['family'].nunique()} families -> {out_dir}")
    if len(bad):
        bad.to_csv(out_dir / "top_segments_failed.csv", index=False)
        print(f"failed     : {len(bad)} (top_segments_failed.csv)")


if __name__ == "__main__":
    main()
