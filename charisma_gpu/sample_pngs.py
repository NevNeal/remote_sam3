#!/usr/bin/env python3
"""
One review PNG per photo for a random sample of photos, from cut-out segments.

    python charisma_gpu/sample_pngs.py --png-dir path/to/segments \\
        --csv charisma_rhirta.csv --out-dir samples --n 5000 [--clut clut_rhirta.csv]

Each PNG (<photo_id>.png) has one row per SAM3 segment of that photo:

    cut-out | per-pixel CLUT map | petal map (cyan = disk found, red = petal bullseye)

with the segment's petal_bullseye, disk_red, ray and disk shares. index.csv
lists every sample with its largest petal_bullseye, so the folder can be
sorted or filtered. The sample is random with a fixed --seed, so it can be
remade. Photos already drawn are skipped.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from charisma import Clut  # noqa: E402
from petal import petal_bullseye  # noqa: E402
from run import png_rows  # noqa: E402

T = 200          # thumbnail size
CAP = 34         # caption height under each row


def _font(size, bold=False):
    for path in (("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_photo(job):
    photo_id, segment_files, clut_path, out_dir = job
    out = Path(out_dir) / f"{photo_id}.png"
    clut = Clut.from_csv(clut_path)
    pal = np.array([tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in clut.hexes], np.uint8)
    bull, disk_c = clut.names.index("bullseye"), pal[clut.names.index("disk")]
    rows, best = [], 0.0
    for idx, path in segment_files:
        a = np.asarray(Image.open(path).convert("RGBA"))
        rgb, fg = a[..., :3], a[..., 3] >= 128
        if not fg.any():
            continue
        res = petal_bullseye(rgb, fg, clut)
        lab = res["labels"]
        colour = pal[np.clip(lab, 0, None)]
        colour[~fg] = 128
        petal = colour.copy()
        petal[(lab == bull) & ~res["petal"]] = disk_c
        edge = res["disk"] & ~ndi.binary_erosion(res["disk"], iterations=max(1, fg.shape[0] // 100))
        petal[edge] = (0, 255, 255)
        cut = rgb.copy()
        cut[~fg] = 128
        n = fg.sum()
        s = res["shares"]
        ray = (lab == clut.names.index("ray")).sum() / n
        dsk = (lab == clut.names.index("disk")).sum() / n
        best = max(best, float(s["petal_bullseye"]))
        rows.append((idx, (cut, colour, petal),
                     f"mask {idx}   petal bullseye {s['petal_bullseye']:.1%}   disk red "
                     f"{s['disk_red']:.1%}   ray {ray:.0%}   disk {dsk:.0%}   ({n} px)"))
    if not rows:
        return photo_id, len(segment_files), 0.0, ""
    font, bold = _font(13), _font(15, bold=True)
    sheet = Image.new("RGB", (3 * T, 26 + len(rows) * (T + CAP)), (90, 90, 90))
    dr = ImageDraw.Draw(sheet)
    dr.text((6, 5), f"photo {photo_id}   cut-out | CLUT | petal (cyan = disk)", fill="white", font=bold)
    for k, (_, panels, caption) in enumerate(rows):
        y = 26 + k * (T + CAP)
        for j, arr in enumerate(panels):
            im = Image.fromarray(arr)
            scale = (T - 6) / max(im.size)       # small segments scaled up too
            im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                           Image.NEAREST)
            sheet.paste(im, (j * T + (T - im.width) // 2, y + (T - im.height) // 2))
        dr.text((6, y + T + 8), caption, fill="white", font=font)
    sheet.save(out, optimize=True)
    return photo_id, len(rows), best, out.name


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--png-dir", required=True, help="the cut-out segments folder")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n", type=int, default=5000, help="photos to sample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clut", default=str(HERE / "clut_rhirta.csv"))
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = p.parse_args()

    photos = png_rows(args.png_dir)
    sample = photos.sample(min(args.n, len(photos)), random_state=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in sample.itertuples() if not (out_dir / f"{r.photo_id}.png").exists()]
    print(f"[samples] {len(photos)} photos, sampling {len(sample)} (seed {args.seed}), "
          f"{len(sample) - len(todo)} already drawn, {len(todo)} to draw", flush=True)
    jobs = [(r.photo_id, r.segment_files, args.clut, str(out_dir)) for r in todo]
    done = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, res in enumerate(pool.map(draw_photo, jobs, chunksize=8), 1):
            done.append(res)
            if i % 500 == 0 or i == len(jobs):
                print(f"[samples] {i}/{len(jobs)}", flush=True)
    index_path = out_dir / "index.csv"
    new = pd.DataFrame(done, columns=["photo_id", "segments", "max_petal_bullseye", "file"])
    if index_path.exists():
        new = pd.concat([pd.read_csv(index_path), new]).drop_duplicates("photo_id", keep="last")
    new.sort_values("max_petal_bullseye", ascending=False).to_csv(index_path, index=False)
    print(f"[samples] {out_dir}  (index.csv sorted by max_petal_bullseye)")


if __name__ == "__main__":
    main()
