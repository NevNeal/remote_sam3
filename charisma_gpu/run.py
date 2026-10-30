#!/usr/bin/env python3
"""
Colour classes for every SAM3 segment of a run, on the GPU.

Reads a segment.py run directory (results_shard_*.csv + the .npz masks), takes
each mask's pixels straight from the photo (no cut-out PNGs needed), and writes
one row per segment with two answers per CLUT colour:

    px_<colour>       share of the segment's pixels the CLUT puts in that colour
                      (every pixel classified on its own)
    px_<colour>_rgb   mean colour of those pixels, hex
    ch_<colour>       charisma's answer: share after its histogram + Lab merge,
                      with each merged colour named by the CLUT (as R's
                      charisma(threshold = 0, pavo = FALSE), checked against it)
    ch_k              charisma's k (number of colour classes)

plus photo_id, mask_index, mask_score and n_pixels. Rows are appended as photos
finish, so a run that is cut off resumes where it stopped.

    python charisma_gpu/run.py --run-dir results/62741_flower \\
        --clut charisma_gpu/clut_rhirta.csv --out results/62741_flower/charisma_rhirta.csv
"""

import argparse
import glob
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from charisma import DEFAULT_CLUT, Clut, charisma_batch, classify_pixels  # noqa: E402
from maskio import load_photo_masks  # noqa: E402


def resolve(run_dir, path):
    path = Path(str(path).replace("\\", "/"))
    return path if path.is_absolute() else Path(run_dir) / path


def photo_segments(row, run_dir, min_score):
    """One results row -> [(mask_index, score, [P, 3] uint8 pixels)]. The photo
    is opened exactly as segment.py opened it (no EXIF rotation), so the masks
    line up."""
    image = np.asarray(Image.open(resolve(run_dir, row["image_path"])).convert("RGB"))
    masks = load_photo_masks(run_dir, str(row["mask_paths"]).replace("\\", "/"))
    scores = [float(s) for s in str(row.get("mask_scores", "")).split(",") if s.strip()]
    out = []
    for i, mask in enumerate(masks):
        score = scores[i] if i < len(scores) else float("nan")
        if score < min_score:
            continue
        if mask.shape != image.shape[:2]:
            mask = np.asarray(Image.fromarray(mask).resize(
                (image.shape[1], image.shape[0]), Image.NEAREST))
        out.append((i, score, image[mask]))
    return out


SEGMENT_NAME = re.compile(r"_(\d+)_segment_(\d+)\.png$", re.IGNORECASE)


def png_rows(png_dir):
    """Cut-out PNGs (render.py's segments/) -> one row per photo, its segment
    files in mask order. photo_id and mask index come from the file name
    <Genus>_<species>_<photo_id>_segment_<i>.png."""
    files = sorted(glob.glob(os.path.join(png_dir, "**", "*_segment_*.png"), recursive=True))
    by_photo = {}
    for f in files:
        m = SEGMENT_NAME.search(f)
        if m:
            by_photo.setdefault(int(m.group(1)), []).append((int(m.group(2)), f))
    return pd.DataFrame([{"photo_id": pid, "segment_files": sorted(segs)}
                         for pid, segs in by_photo.items()])


def png_segments(row):
    """Foreground pixels of each cut-out: R charisma's round(alpha) >= 1."""
    out = []
    for idx, path in row["segment_files"]:
        a = np.asarray(Image.open(path).convert("RGBA"))
        out.append((idx, float("nan"), a[a[..., 3] >= 128][:, :3]))
    return out


class Photos(torch.utils.data.Dataset):
    """Photo decoding and mask reading in DataLoader workers."""

    def __init__(self, rows, run_dir, min_score):
        self.rows, self.run_dir, self.min_score = rows, run_dir, min_score

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        try:
            if "segment_files" in row:
                return row, png_segments(row), ""
            return row, photo_segments(row, self.run_dir, self.min_score), ""
        except Exception as exc:              # a bad file is recorded, not fatal
            return row, [], f"{type(exc).__name__}: {exc}"


def _items(batch):
    return batch


def to_hex(rgb):
    if not np.all(np.isfinite(rgb)):
        return ""
    r, g, b = (int(round(min(max(v, 0), 255))) for v in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def load_results(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "results_shard_*.csv")))
    if not files:
        raise SystemExit(f"no results_shard_*.csv in {run_dir}")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    df = df[(df["status"] == "ok") & (df["num_masks"].fillna(0) > 0)]
    return df.drop_duplicates("photo_id", keep="last").reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", help="segment.py output directory (photos + .npz masks)")
    src.add_argument("--png-dir", help="a folder of cut-out segment PNGs (searched recursively); "
                                       "mask scores are then left blank")
    p.add_argument("--out", required=True, help="CSV to write (appended; resumes)")
    p.add_argument("--clut", default=str(DEFAULT_CLUT))
    p.add_argument("--min-score", type=float, default=0.0,
                   help="skip masks below this SAM3 score (all are kept by default)")
    p.add_argument("--mode", choices=("both", "pixel", "charisma"), default="both")
    p.add_argument("--bins", type=int, default=4, help="charisma bins per channel")
    p.add_argument("--cutoff", type=float, default=20.0, help="charisma Lab merge cutoff")
    p.add_argument("--photos-per-batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="only this many photos (testing)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    clut = Clut.from_csv(args.clut)
    results = png_rows(args.png_dir) if args.png_dir else load_results(args.run_dir)
    out = Path(args.out)
    done = set()
    if out.exists():
        done = set(pd.read_csv(out, usecols=["photo_id"])["photo_id"])
    todo = results[~results["photo_id"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"[charisma] {len(results)} photos with segments, {len(done)} done, {len(todo)} to go "
          f"| clut {Path(args.clut).name}: {', '.join(clut.names)} | {args.device}", flush=True)
    if todo.empty:
        return

    loader = torch.utils.data.DataLoader(
        Photos(todo.to_dict("records"), args.run_dir, args.min_score),
        batch_size=args.photos_per_batch, num_workers=args.workers,
        collate_fn=_items, prefetch_factor=4 if args.workers else None)
    cols = ["photo_id", "mask_index", "mask_score", "n_pixels", "error"]
    if args.mode in ("both", "pixel"):
        cols += [f"px_{n}" for n in clut.names] + [f"px_{n}_rgb" for n in clut.names]
    if args.mode in ("both", "charisma"):
        cols += ["ch_k"] + [f"ch_{n}" for n in clut.names]
    header = not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    start, n_photos, n_segs = time.time(), 0, 0
    for batch in loader:
        recs, segs = [], []
        for row, items, err in batch:
            if err or not items:
                recs.append({"photo_id": row["photo_id"], "mask_index": -1,
                             "error": err or "no masks at this score"})
            for idx, score, px in items:
                recs.append({"photo_id": row["photo_id"], "mask_index": idx,
                             "mask_score": score, "n_pixels": len(px), "error": ""})
                segs.append(px)
        seg_recs = [r for r in recs if r["mask_index"] >= 0]
        nonempty = [i for i, s in enumerate(segs) if len(s)]
        if nonempty:
            batch_segs = [segs[i] for i in nonempty]
            if args.mode in ("both", "pixel"):
                shares, mean_rgb = classify_pixels(batch_segs, clut, device=args.device)
                for j, i in enumerate(nonempty):
                    for c, name in enumerate(clut.names):
                        seg_recs[i][f"px_{name}"] = round(float(shares[j, c]), 5)
                        seg_recs[i][f"px_{name}_rgb"] = to_hex(mean_rgb[j, c]) if shares[j, c] > 0 else ""
            if args.mode in ("both", "charisma"):
                ch = charisma_batch(batch_segs, clut, bins=args.bins, cutoff=args.cutoff,
                                    device=args.device)
                for j, i in enumerate(nonempty):
                    seg_recs[i]["ch_k"] = ch[j]["k"]
                    for name in clut.names:
                        seg_recs[i][f"ch_{name}"] = round(ch[j]["props"].get(name, 0.0), 5)
        pd.DataFrame(recs, columns=cols).to_csv(out, mode="a", header=header, index=False)
        header = False
        n_photos += len(batch)
        n_segs += len(segs)
        el = time.time() - start
        if n_photos == len(batch) or (n_photos // args.photos_per_batch) % 20 == 0 or n_photos == len(todo):
            eta = (len(todo) - n_photos) / (n_photos / el)
            print(f"[charisma] {n_photos}/{len(todo)} photos, {n_segs} segments | "
                  f"{n_photos / el:.1f} photos/s, {n_segs / el:.1f} segments/s | "
                  f"eta {eta / 60:.1f} min", flush=True)
    print(f"[charisma] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
