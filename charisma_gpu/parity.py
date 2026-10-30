#!/usr/bin/env python3
"""
charisma_gpu against R charisma on the same cut-out PNGs.

    Rscript charisma_gpu/parity.R paths.txt r.csv [clut.csv]
    python  charisma_gpu/parity.py paths.txt r.csv [--clut clut.csv]

Prints, per image, whether the colour classes match and the largest
difference in any class's proportion, then a summary and the timing of both.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from charisma import DEFAULT_CLUT, Clut, charisma_batch  # noqa: E402


def png_pixels(path):
    """Foreground pixels of a cut-out PNG: R's round(alpha) >= 1, i.e. alpha >= 128."""
    a = np.asarray(Image.open(path).convert("RGBA"))
    return a[a[..., 3] >= 128][:, :3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths")
    p.add_argument("r_csv")
    p.add_argument("--clut", default=str(DEFAULT_CLUT))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    paths = [l.strip() for l in open(args.paths) if l.strip()]
    clut = Clut.from_csv(args.clut)
    t0 = time.time()
    segs = [png_pixels(x) for x in paths]
    t1 = time.time()
    res = charisma_batch(segs, clut, device=args.device)
    if args.device == "cuda":
        torch.cuda.synchronize()
    t2 = time.time()

    r = pd.read_csv(args.r_csv)
    same_k = same_cols = 0
    worst = []
    for path, py in zip(paths, res):
        rr = r[r["path"] == path]
        if rr.empty:
            continue
        r_props = dict(zip(rr["colour"], rr["prop"]))
        cols_match = set(r_props) == set(py["props"])
        same_cols += cols_match
        same_k += int(rr["k"].iloc[0]) == py["k"]
        diff = max(abs(r_props.get(c, 0) - py["props"].get(c, 0))
                   for c in set(r_props) | set(py["props"]))
        worst.append(diff)
        if not cols_match or diff > 0.01:
            print(f"  {Path(path).name}: R {sorted(r_props.items())}\n"
                  f"  {' ' * len(Path(path).name)}  py {sorted(py['props'].items())}")
    n = len(worst)
    print(f"\n{n} images | same colour classes {same_cols}/{n} | same k {same_k}/{n} | "
          f"max |prop diff| median {np.median(worst):.4f}, max {max(worst):.4f}")
    print(f"R  : {r.drop_duplicates('path')['seconds'].sum():.1f} s total, "
          f"{r.drop_duplicates('path')['seconds'].mean():.2f} s per image")
    print(f"py : {t2 - t1:.2f} s for the batch on {args.device} "
          f"(+{t1 - t0:.2f} s reading PNGs)")


if __name__ == "__main__":
    main()
