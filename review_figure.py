#!/usr/bin/env python3
"""
One figure per photo for judging segment quality by eye, from a finished run.

    python review_figure.py results/62741_flower --sample 200
    python review_figure.py results/62741_flower --photo-id 13098033
    python review_figure.py results/62741_flower --shard 0 --num-shards 4   # all

Left: the original image with its most confident masks highlighted, each
outlined in its own colour and labelled with its number and confidence.
Right: those same segments as cut-outs, one per row, most confident first,
with the number faintly in the corner and the confidence beneath it. At most
TOP_K (5) masks per photo; a photo with more says so in the header.

Reads only what the run left behind - the results CSVs, the .npy masks and the
image each row points at - so it needs no GPU and no render.py output. Writes

    <run>/review/batch_00001/Genus_species_<photo_id>_review.png

and skips a photo whose figure already exists.
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

TOP_K = 5
PANEL_H = 1000                # height of the figure body, px
HEADER_H = 44
GAP = 12
TILE_W = 300
BG = (24, 24, 27)
TILE_BG = (96, 96, 102)
CHECK_BG = (116, 116, 122)
TEXT = (235, 235, 235)
MUTED = (150, 150, 158)
# Okabe-Ito, minus the yellow: distinguishable with any colour vision, and none
# of them is the colour of a Rudbeckia.
COLORS = [(86, 180, 233), (213, 94, 0), (0, 158, 115), (204, 121, 167), (0, 114, 178)]
FILL_ALPHA = 70
OUTLINE_PX = 3


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="A run's output directory")
    parser.add_argument("--sample", type=int, default=None,
                        help="Draw only N randomly chosen photos (default: all)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for --sample")
    parser.add_argument("--photo-id", type=int, nargs="+", default=None,
                        help="Draw only these photos")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    run = Path(args.run_dir).expanduser().resolve()
    photos = load_detected(run)
    if args.photo_id:
        photos = photos[photos["photo_id"].isin(args.photo_id)]
    if args.sample and args.sample < len(photos):
        photos = photos.sample(n=args.sample, random_state=args.seed)
    photos = photos[photos["row_index"] % max(1, args.num_shards) == args.shard]

    todo = [r for r in photos.to_dict("records") if not (run / _figure_rel(r)).exists()]
    print(f"review     : {len(photos):,} photos, {len(photos) - len(todo):,} already drawn, "
          f"{len(todo):,} to go")
    failed = 0
    if len(todo) <= 1 or args.workers <= 1:
        results = [draw_one(run, r) for r in todo]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_draw_star, ((run, r) for r in todo), chunksize=8))
    for photo_id, error in results:
        if error:
            failed += 1
            print(f"  failed photo_id={photo_id}: {error}")
    print(f"review     : {len(todo) - failed:,} figures -> {run / 'review'}"
          + (f", {failed:,} failed" if failed else ""))


def load_detected(run):
    """Every photo with at least one mask, last record per photo."""
    frames = []
    for f in sorted(glob.glob(str(run / "results_shard_*.csv"))):
        try:
            frames.append(pd.read_csv(f, low_memory=False))
        except pd.errors.EmptyDataError:
            continue
    if not frames:
        raise SystemExit(f"no results_shard_*.csv in {run}")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["photo_id", "row_index", "status"])
    df = df.drop_duplicates(subset=["photo_id"], keep="last")
    df["num_masks"] = pd.to_numeric(df["num_masks"], errors="coerce").fillna(0)
    df = df[(df["status"] == "ok") & (df["num_masks"] > 0)].copy()
    df["row_index"] = df["row_index"].astype(int)
    df["photo_id"] = df["photo_id"].astype("int64")
    return df


def _figure_rel(row):
    first = Path(str(row["mask_paths"]).split(";")[0])
    stem = first.name.rsplit("_instance_", 1)[0]
    return Path("review") / first.parent.name / f"{stem}_review.png"


def _draw_star(job):
    return draw_one(*job)


def draw_one(run, row):
    try:
        image_path = Path(str(row["image_path"]))
        image = Image.open(image_path if image_path.is_absolute() else run / image_path)
        image = image.convert("RGB")
        mask_rels = str(row["mask_paths"]).split(";")
        scores = [float(s) for s in str(row["mask_scores"]).split(",")]
        # segment.py writes masks most confident first; sort anyway, so the
        # figure is right even for a CSV that was not.
        order = sorted(range(len(mask_rels)), key=lambda i: scores[i], reverse=True)[:TOP_K]
        masks = [np.load(run / mask_rels[i]) > 0 for i in order]
        title = (f"{row.get('taxon_name', '')}   photo {int(row['photo_id'])}   "
                 f"{image.width}x{image.height}")
        figure = make_figure(image, masks, [scores[i] for i in order], title, len(mask_rels))
        out = run / _figure_rel(row)
        out.parent.mkdir(parents=True, exist_ok=True)
        figure.save(out)
        return int(row["photo_id"]), None
    except Exception as exc:
        return int(row["photo_id"]), str(exc)[:300]


def make_figure(image, masks, scores, title, total_masks):
    """image: PIL RGB. masks: bool arrays at image size, most confident first."""
    scale = PANEL_H / image.height
    left_w = max(1, round(image.width * scale))
    width = GAP + left_w + GAP + TILE_W + GAP
    fig = Image.new("RGB", (width, HEADER_H + PANEL_H + GAP), BG)
    draw = ImageDraw.Draw(fig)

    shown = (f"top {len(masks)} of {total_masks} masks" if total_masks > len(masks)
             else f"{len(masks)} mask{'s' if len(masks) != 1 else ''}")
    draw.text((GAP, HEADER_H // 2), title, fill=TEXT, font=_font(20), anchor="lm")
    draw.text((width - GAP, HEADER_H // 2), shown, fill=MUTED, font=_font(18), anchor="rm")

    fig.paste(_overlay(image, masks, scores, (left_w, PANEL_H)), (GAP, HEADER_H))

    tile_h = (PANEL_H - GAP * (TOP_K - 1)) // TOP_K
    x = GAP + left_w + GAP
    for i, (mask, score) in enumerate(zip(masks, scores)):
        y = HEADER_H + i * (tile_h + GAP)
        fig.paste(_tile(image, mask, score, i, (TILE_W, tile_h)), (x, y))
    return fig


def _overlay(image, masks, scores, size):
    """The photo at display size, each mask tinted and outlined in its colour."""
    base = image.resize(size, Image.LANCZOS).convert("RGBA")
    labels = []
    # Least confident first, so mask 1's outline is never drawn over.
    for i in reversed(range(len(masks))):
        color = COLORS[i % len(COLORS)]
        big = Image.fromarray(masks[i].astype(np.uint8) * 255).resize(size, Image.NEAREST)
        fill = Image.new("RGBA", size, color + (0,))
        fill.putalpha(big.point(lambda v: FILL_ALPHA if v else 0))
        base = Image.alpha_composite(base, fill)
        inner = big.filter(ImageFilter.MinFilter(2 * OUTLINE_PX + 1))
        edge = Image.fromarray(((np.asarray(big) > 0) & (np.asarray(inner) == 0)
                                ).astype(np.uint8) * 255)
        line = Image.new("RGBA", size, color + (0,))
        line.putalpha(edge)
        base = Image.alpha_composite(base, line)
        box = big.getbbox()
        if box:
            labels.append((i, box, color))

    draw = ImageDraw.Draw(base)
    font = _font(20)
    for i, (x1, y1, _, _), color in sorted(labels):
        text = f"{i + 1}  {scores[i]:.3f}"
        w = draw.textlength(text, font=font) + 14
        x = min(max(0, x1), size[0] - w)
        y = y1 - 30 if y1 >= 30 else y1 + 4
        draw.rounded_rectangle([x, y, x + w, y + 26], radius=5, fill=color + (255,))
        draw.text((x + 7, y + 13), text, fill=(255, 255, 255), font=font, anchor="lm")
    return base.convert("RGB")


def _tile(image, mask, score, i, size):
    """One segment, cut out and fitted into a tile, with its number and score."""
    color = COLORS[i % len(COLORS)]
    # A checkerboard, so what is transparent reads as transparent and a dark
    # flower centre is not lost against a dark tile.
    yy, xx = np.indices((size[1], size[0])) // 12
    checker = np.where(((yy + xx) % 2)[..., None], TILE_BG, CHECK_BG).astype(np.uint8)
    tile = Image.fromarray(checker, "RGB").convert("RGBA")
    draw = ImageDraw.Draw(tile)

    ys, xs = np.nonzero(mask)
    if len(xs):
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        rgba = np.dstack([np.asarray(image)[y1:y2, x1:x2],
                          mask[y1:y2, x1:x2].astype(np.uint8) * 255])
        cut = Image.fromarray(rgba, "RGBA")
        room = (size[0] - 16, size[1] - 16 - 26)
        k = min(room[0] / cut.width, room[1] / cut.height)
        cut = cut.resize((max(1, round(cut.width * k)), max(1, round(cut.height * k))),
                         Image.LANCZOS)
        tile.alpha_composite(cut, ((size[0] - cut.width) // 2,
                                   8 + (room[1] - cut.height) // 2))

    # The number, faint, in the corner - drawn on its own layer so it can be
    # translucent over the cut-out.
    ghost = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(ghost).text((10, 2), str(i + 1), fill=color + (110,), font=_font(56))
    tile.alpha_composite(ghost)

    draw.text((size[0] - 10, size[1] - 14), f"conf {score:.3f}", fill=TEXT,
              font=_font(18), anchor="rm")
    area = 100 * mask.mean()
    draw.text((10, size[1] - 14), f"{area:.1f}% of image", fill=(215, 215, 220),
              font=_font(15), anchor="lm")
    draw.rectangle([0, 0, size[0] - 1, size[1] - 1], outline=color + (255,), width=3)
    return tile.convert("RGB")


_FONTS = {}


def _font(size):
    if size not in _FONTS:
        for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                     "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
                     "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
                     "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                     "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
                     "C:/Windows/Fonts/arialbd.ttf"):
            if os.path.exists(path):
                _FONTS[size] = ImageFont.truetype(path, size=size)
                break
        else:
            try:
                _FONTS[size] = ImageFont.load_default(size=size)
            except TypeError:
                _FONTS[size] = ImageFont.load_default()
    return _FONTS[size]


if __name__ == "__main__":
    main()
