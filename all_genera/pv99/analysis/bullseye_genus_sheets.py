#!/usr/bin/env python3
"""
One summary sheet per genus, for checking the bullseye calls genus by genus in
ImageAnt.

    python all_genera/pv99/analysis/bullseye_genus_sheets.py               # >= 10 bullseye photos
    python all_genera/pv99/analysis/bullseye_genus_sheets.py --min-bullseye 5 --per-sheet 30

A genus gets a sheet when the model called at least --min-bullseye of its
photos bullseye (360 genera at 10 in the 2026-11-10 run). The sheet shows up to
--per-sheet (20) of those bullseye-called photos, highest P(bullseye) first,
each as the crop around its top SAM3 segment that the model saw, with P on the
tile. The header names the genus and family and gives the count.

Writes export/pv99/analysis/bullseye/imageant_genera/:

    images/<Family>__<Genus>.jpg   one sheet per genus (sorted by family in ImageAnt)
    genus_sheets.csv               file, genus, family, rank, bullseye, told, adjusted share
    bullseye_genus.ias             the ImageAnt task script
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from bullseye_classify import top_segment_crop  # noqa: E402
from PIL import ImageFont  # noqa: E402


def _font(size):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)      # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()

TILE, COLS, HEAD, GAP = 260, 5, 92, 4
BG = (24, 24, 27)

SCRIPT = """\
#
# pv99 bullseye check, one genus per image: up to 20 of the genus's photos that
# the model called bullseye, highest P(bullseye) first.
#
main_tasks:
  - text: 'Do these flowers have a bullseye pattern?'
    variable: 'bullseye'
    responses:
      - text: '"y": yes, bullseye'
        action: keypress{y ~ Y}
      - text: '"n": no bullseye'
        action: keypress{n ~ N}
        jump-to: no_reason

#
# Only asked when the answer is "no bullseye".
#
no_reason:
  - text: 'Why not a bullseye?'
    variable: 'no_reason'
    default: ''
    responses:
      - text: '"c": composite head'
        action: keypress{c ~ composite_head}
      - text: '"o": other reason (note it in other_notes.csv)'
        action: keypress{o ~ other}
"""


def tile(project, row):
    try:
        crop, _, _ = top_segment_crop(project, row["image_path"], row["mask_path"], TILE)
    except Exception:
        return Image.new("RGB", (TILE, TILE), (96, 96, 102))
    draw = ImageDraw.Draw(crop)
    draw.rectangle((0, TILE - 22, 64, TILE), fill=(20, 20, 22))
    draw.text((6, TILE - 11), f"{row['bullseye__bullseye']:.2f}", fill=(240, 240, 240),
              font=_font(15), anchor="lm")
    return crop


def make_sheet(job):
    project, g, rows, path = job
    tiles = [tile(project, r) for r in rows]
    n_rows = int(np.ceil(len(tiles) / COLS))
    width = COLS * TILE + (COLS + 1) * GAP
    sheet = Image.new("RGB", (width, HEAD + n_rows * (TILE + GAP) + GAP), BG)
    draw = ImageDraw.Draw(sheet)
    draw.text((14, 30), g["genus"], fill=(245, 245, 245), font=_font(38), anchor="lm")
    name_w = draw.textlength(g["genus"], font=_font(38))
    draw.text((26 + name_w, 33), g["family"], fill=(170, 170, 175), font=_font(26), anchor="lm")
    draw.text((14, 72), f"{g['bullseye']} of {g['told']} photos called bullseye; "
              f"showing the top {len(tiles)}", fill=(170, 170, 175), font=_font(19), anchor="lm")
    for i, t in enumerate(tiles):
        sheet.paste(t, (GAP + (i % COLS) * (TILE + GAP), HEAD + (i // COLS) * (TILE + GAP)))
    sheet.save(path, quality=88)
    return str(path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_args(parser)
    parser.add_argument("--min-bullseye", type=int, default=10)
    parser.add_argument("--per-sheet", type=int, default=20)
    parser.add_argument("--workers", type=int, default=common.n_workers())
    args = parser.parse_args()

    project = common.project_dir(args.project)
    folder = common.analysis_dir(project, "bullseye")
    out = common.analysis_dir(project, "bullseye/imageant_genera")
    images = out / "images"
    images.mkdir(exist_ok=True)

    genus = pd.read_csv(folder / "genus_bullseye.csv")
    genera = genus[genus["bullseye"] >= args.min_bullseye].copy()
    photos = pd.read_csv(folder / "photos.csv")
    photos = photos[photos["call"] == "bullseye"].merge(
        common.load_metadata(project, ["photo_id", "image_path", "mask_path"]), on="photo_id")
    photos = photos.sort_values("bullseye__bullseye", ascending=False)
    groups = {k: g.head(args.per_sheet).to_dict("records")
              for k, g in photos.groupby(["genus", "family"])}

    genera["file"] = [f"{common.safe(r.family)}__{common.safe(r.genus)}.jpg" for r in genera.itertuples()]
    jobs = [(project, g, groups[(g["genus"], g["family"])], images / g["file"])
            for g in genera.to_dict("records")]
    print(f"[sheets] {len(jobs)} genera with >= {args.min_bullseye} bullseye photos -> {images}", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        list(ex.map(make_sheet, jobs, chunksize=2))

    genera = genera.sort_values("file")
    genera[["file", "genus", "family", "rank", "bullseye", "told", "posterior"]].rename(
        columns={"posterior": "adjusted_share"}).to_csv(out / "genus_sheets.csv", index=False)
    (out / "bullseye_genus.ias").write_text(SCRIPT)
    notes = out / "other_notes.csv"
    if not notes.exists():
        notes.write_text("file,note\n")
    print(f"[sheets] done: {len(genera)} sheets, {genera['family'].nunique()} families", flush=True)


if __name__ == "__main__":
    main()
