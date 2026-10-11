#!/usr/bin/env python3
"""
Reading and writing a photo's masks. Numpy only, so the CPU tools (render.py,
review_figure.py, all_genera_report.py) can use it without importing torch.

Two formats, told apart by the file extension in the results CSV's mask_paths:

  npy   the original: one <stem>_instance_<i>.npy per mask, each a full-frame
        uint8 array of 0/1. One byte per pixel, uncompressed: 187,628 bytes for
        a 500x375 photo, however small the flower.

  npz   one <stem>_masks.npz per photo, holding every mask of that photo:
            packed   uint8   the N x H x W stack of 0/1, 8 pixels to the byte
                             (np.packbits over the flattened stack)
            shape    int32   [N, H, W], to undo the packing
            scores   float32 [N], each mask's confidence, most confident first
        and the archive is deflate-compressed (np.savez_compressed). Lossless:
        load_masks() returns exactly the arrays save_masks() was given.

    masks, scores = load_masks("masks/batch_00001/Genus_species_123_masks.npz")
    masks.shape   # (N, H, W), dtype bool
"""

from pathlib import Path

import numpy as np

NPZ_SUFFIX = "_masks.npz"


def save_masks(path, masks, scores):
    """Write one photo's masks (a list of H x W 0/1 arrays) as a packed .npz.
    Returns the bytes written."""
    path = Path(path)
    stack = np.stack([np.asarray(m) > 0 for m in masks])
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as handle:
        np.savez_compressed(
            handle,
            packed=np.packbits(stack, axis=None),
            shape=np.asarray(stack.shape, dtype=np.int32),
            scores=np.asarray(scores, dtype=np.float32))
    # Renamed into place, so a job killed mid-write never leaves a half file
    # under the real name.
    tmp.replace(path)
    return path.stat().st_size


def load_masks(path):
    """A packed .npz -> (masks as an N x H x W bool array, scores)."""
    with np.load(path) as data:
        shape = tuple(int(v) for v in data["shape"])
        count = int(np.prod(shape))
        masks = np.unpackbits(data["packed"], count=count).reshape(shape).astype(bool)
        return masks, data["scores"].astype(float)


def load_photo_masks(run, mask_paths):
    """Every mask of one photo, from the results CSV's mask_paths cell, in
    either format. Returns a list of H x W bool arrays, in the order recorded."""
    run = Path(run)
    paths = [p for p in str(mask_paths).split(";") if p]
    if len(paths) == 1 and paths[0].endswith(".npz"):
        return list(load_masks(run / paths[0])[0])
    return [np.load(run / p) > 0 for p in paths]


def stem_and_batch(mask_paths):
    """('Genus_species_123', 'batch_00001') from a mask_paths cell, either format."""
    first = Path(str(mask_paths).split(";")[0])
    name = first.name
    if name.endswith(NPZ_SUFFIX):
        stem = name[:-len(NPZ_SUFFIX)]
    else:
        stem = name.rsplit("_instance_", 1)[0]
    return stem, first.parent.name
