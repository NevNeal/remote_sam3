#!/usr/bin/env python3
"""
Every mask of a run in two files, as small as lossless gets:

    python all_genera/run2/pack_masks.py results/all_genera_run2_flower \\
        --metadata export/all_genera_run2/all_genera_segment_metadata.csv \\
        --out export/all_genera_run2/masks

    masks.rle.xz          the masks
    masks_index.parquet   one row per mask: photo_id, mask (0 = most
                          confident), score, the photo's height and width, the
                          mask's bounding box y0 x0 h w, and nbytes (its length
                          in the decompressed stream, in order)

Each mask is cropped to its bounding box, flattened row by row and run-length
coded: alternating runs of 0s and 1s, starting with 0s (that first run may be
0 long), each run a LEB128 varint, trailing 0s dropped. All masks are
concatenated and the stream is xz-compressed as one piece, so the compressor
sees the whole run rather than one small file at a time. On a 2,000-photo
sample of run 1 this was 5x smaller than the per-photo .npz files and 20%
smaller than bit-packing the same crops.

Reading it back needs numpy, pandas and the standard library:

    from pack_masks import read_pack
    for row, mask in read_pack("export/all_genera_run2/masks"):
        ...   # row: the index row; mask: H x W bool array, full frame
"""

import argparse
import lzma
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from maskio import load_photo_masks  # noqa: E402

BLOB = "masks.rle.xz"
INDEX = "masks_index.parquet"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="The run's output directory (mask_path is relative to it)")
    parser.add_argument("--metadata", required=True, help="all_genera_segment_metadata.csv")
    parser.add_argument("--out", required=True, help="Folder for the two files")
    parser.add_argument("--max-masks", type=int, default=5,
                        help="Keep each photo's N most confident masks (0 = all)")
    args = parser.parse_args()

    run = Path(args.run_dir).expanduser().resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.metadata, usecols=["photo_id", "num_masks", "mask_path",
                                               "mask_scores", "mask_bytes"])
    meta = meta[meta["num_masks"] > 0].sort_values("photo_id")

    rows, before = [], 0
    compressor = lzma.LZMACompressor(preset=9 | lzma.PRESET_EXTREME)
    tmp = out / (BLOB + ".part")
    with open(tmp, "wb") as blob:
        for i, r in enumerate(meta.itertuples(index=False), 1):
            masks = load_photo_masks(run, r.mask_path)
            scores = [float(s) for s in str(r.mask_scores).split(",")]
            before += int(r.mask_bytes) if pd.notna(r.mask_bytes) else 0
            order = sorted(range(len(masks)), key=lambda k: scores[k], reverse=True)
            if args.max_masks:
                order = order[:args.max_masks]
            for rank, k in enumerate(order):
                data, box = encode(masks[k])
                blob.write(compressor.compress(data))
                rows.append((r.photo_id, rank, scores[k], *masks[k].shape, *box, len(data)))
            if i % 20000 == 0:
                print(f"packed     : {i:,}/{len(meta):,} photos", flush=True)
        blob.write(compressor.flush())
    tmp.replace(out / BLOB)

    index = pd.DataFrame(rows, columns=["photo_id", "mask", "score", "height", "width",
                                        "y0", "x0", "h", "w", "nbytes"])
    index = index.astype({"photo_id": "int64", "mask": "uint8", "score": "float32",
                          "height": "uint16", "width": "uint16", "y0": "uint16",
                          "x0": "uint16", "h": "uint16", "w": "uint16", "nbytes": "uint32"})
    index.to_parquet(out / INDEX, index=False, compression="zstd", compression_level=19)
    after = (out / BLOB).stat().st_size + (out / INDEX).stat().st_size
    print(f"packed     : {len(index):,} masks of {len(meta):,} photos -> {out}")
    print(f"           : {after / 1e6:,.1f} MB  (the .npz files: {before / 1e6:,.1f} MB)")


def encode(mask):
    """A full-frame bool mask -> (varint run bytes of its bbox crop, (y0, x0, h, w))."""
    ys = np.flatnonzero(mask.any(axis=1))
    xs = np.flatnonzero(mask.any(axis=0))
    if not len(ys):
        return b"", (0, 0, 0, 0)
    y0, x0 = int(ys[0]), int(xs[0])
    h, w = int(ys[-1]) - y0 + 1, int(xs[-1]) - x0 + 1
    flat = mask[y0:y0 + h, x0:x0 + w].ravel().astype(np.int8)
    changes = np.flatnonzero(np.diff(np.concatenate(([0], flat, [0]))))
    runs = np.diff(np.concatenate(([0], changes)))
    return varints(runs), (y0, x0, h, w)


def varints(values):
    """Non-negative ints -> LEB128 bytes."""
    values = np.asarray(values, dtype=np.uint64)
    size = np.ones(len(values), dtype=np.int64)
    for k in range(1, 10):
        size += values >= (1 << (7 * k))
    out = np.empty(int(size.sum()), dtype=np.uint8)
    start = np.concatenate(([0], np.cumsum(size)[:-1]))
    for k in range(int(size.max(initial=0))):
        has = size > k
        byte = (values[has] >> np.uint64(7 * k)) & np.uint64(0x7F)
        more = (size[has] > k + 1).astype(np.uint64) << np.uint64(7)
        out[start[has] + k] = (byte | more).astype(np.uint8)
    return out.tobytes()


def unvarints(data):
    """LEB128 bytes -> int64 array."""
    b = np.frombuffer(data, dtype=np.uint8)
    if not len(b):
        return np.zeros(0, dtype=np.int64)
    last = (b & 0x80) == 0
    group = np.concatenate(([0], np.cumsum(last)[:-1]))
    first = np.concatenate(([0], np.flatnonzero(last)[:-1] + 1))
    pos = np.arange(len(b)) - first[group]
    parts = (b & 0x7F).astype(np.int64) << (7 * pos)
    return np.bincount(group, weights=parts).astype(np.int64)


def decode(data, row):
    """One mask's bytes + its index row -> full-frame H x W bool array."""
    full = np.zeros((int(row["height"]), int(row["width"])), dtype=bool)
    h, w = int(row["h"]), int(row["w"])
    if h and w:
        runs = unvarints(data)
        values = np.arange(len(runs)) % 2 == 1
        crop = np.zeros(h * w, dtype=bool)
        filled = np.repeat(values, runs)
        crop[:len(filled)] = filled
        y0, x0 = int(row["y0"]), int(row["x0"])
        full[y0:y0 + h, x0:x0 + w] = crop.reshape(h, w)
    return full


def read_pack(folder):
    """Yield (index row as a dict, full-frame bool mask) for every mask, in order."""
    folder = Path(folder)
    index = pd.read_parquet(folder / INDEX)
    with lzma.open(folder / BLOB, "rb") as blob:
        for row in index.to_dict("records"):
            yield row, decode(blob.read(int(row["nbytes"])), row)


if __name__ == "__main__":
    main()
