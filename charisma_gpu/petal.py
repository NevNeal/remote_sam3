"""
Red on the petals, not on the disk: a spatial step after the per-pixel CLUT.

Colour alone can't separate a dark maroon bullseye from a reddish-brown disk
(their hue, saturation and value overlap), so a CLUT that catches maroon ray
bases also paints many disks red. This step finds the disk by its shape and
counts red as bullseye only outside it.

Per image (a cut-out: RGB plus a foreground mask):

  1. name every pixel with the CLUT (it needs names "bullseye", "disk", "ray")
  2. core = foreground pixels that are dark (v < dark_v) or named disk or other,
     and not ray, green or pale; holes filled (pink anther centres, highlights)
  3. take the largest connected piece of the core; r0 = its largest inscribed
     radius (maximum of the distance transform)
  4. morphological opening of that piece with a disc of radius open_frac * r0:
     petal-base patches, narrower than the disk, are erased; the disk survives
  5. disk = the largest opened piece, grown by grow_frac * r0, provided it is
     compact (area <= max_area_ratio * pi * r0^2) - a ring of maroon petal bases
     around a cut-out with no disk is not compact, so it is not taken as a disk
  6. steps 2-5 again with bullseye-named pixels allowed to seed the core too;
     that disk replaces the first if its radius is at most max_disk_frac of the
     flower's (95th-percentile distance of the cut-out from its centre). This
     catches disks whose outer florets are bright red-brown: the red rim stops
     at the disk edge (0.15-0.21 of the flower in testing), while a bright
     bullseye ring touching the disk runs out along the petals (0.34-0.50)
  7. bullseye-named pixels inside the disk -> "disk_red"; elsewhere -> "petal"

What it can't fix: a maroon ring with no gap between it and a reddish disk is
one solid blob, and is counted as disk; and a disk that is never found (too
fragmented, or not compact) leaves its red counted as petal.
"""

import numpy as np
import torch
from scipy import ndimage as ndi

from charisma import srgb_to_hsv


def _largest(mask):
    lab, n = ndi.label(mask)
    if n == 0:
        return np.zeros_like(mask)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def _find_disk(seed, fg, soft, open_frac, grow_frac, max_area_ratio, min_r0):
    """Steps 2-5: the compact disk inside the seeded core, or an empty mask."""
    disk = np.zeros_like(fg)
    piece = _largest(ndi.binary_fill_holes(seed) & fg)
    if not piece.any():
        return disk
    dist = ndi.distance_transform_edt(piece)
    r0 = float(dist.max())
    if r0 < min_r0:
        return disk
    rho = open_frac * r0
    eroded = dist > rho
    if not eroded.any():
        return disk
    opened = (ndi.distance_transform_edt(~eroded) <= rho) & piece
    body = _largest(opened)
    if body.sum() > max_area_ratio * np.pi * r0 ** 2:
        return disk
    grown = ndi.distance_transform_edt(~body) <= grow_frac * r0
    return grown & fg & ~soft


def pixel_labels(rgb, clut):
    """H x W x 3 uint8 -> (H x W CLUT index, H x W value %)."""
    h, w = rgb.shape[:2]
    hsv = srgb_to_hsv(torch.from_numpy(rgb.reshape(-1, 3).astype(np.float64) / 255))
    return clut.label(hsv).numpy().reshape(h, w), hsv[:, 2].numpy().reshape(h, w)


def petal_bullseye(rgb, fg, clut, dark_v=45.0, open_frac=0.6, grow_frac=0.15,
                   max_area_ratio=4.0, min_r0=2.0, max_disk_frac=0.28):
    """Split a cut-out's bullseye-coloured pixels into petal and disk.

    rgb: H x W x 3 uint8; fg: H x W bool. Returns a dict with
      labels   H x W CLUT index (-1 outside fg)
      petal    H x W bool, bullseye-coloured pixels on the petals
      disk     H x W bool, the disk found (empty if none)
      shares   {"petal_bullseye", "disk_red", "disk_area"} as shares of fg
    """
    idx = {n: i for i, n in enumerate(clut.names)}
    for need in ("bullseye", "disk", "ray"):
        if need not in idx:
            raise ValueError(f"petal_bullseye needs a CLUT with a '{need}' colour")
    fg = np.asarray(fg, dtype=bool)
    labels, v = pixel_labels(rgb, clut)
    red = fg & (labels == idx["bullseye"])
    soft = np.isin(labels, [idx[n] for n in ("ray", "green", "pale") if n in idx])
    seed = (v < dark_v) | (labels == idx["disk"])
    if "other" in idx:
        seed |= labels == idx["other"]
    dark_disk = _find_disk(fg & ~soft & seed, fg, soft, open_frac, grow_frac,
                           max_area_ratio, min_r0)
    # A disk whose outer florets are bright red-brown has its red rim left out of
    # dark_disk. Find the disk again with red allowed to seed it, and keep that
    # version when it is still disk-sized: a red rim stops at the disk edge
    # (radius 0.15-0.21 of the flower in testing), a bullseye runs out along the
    # petals (0.34-0.50).
    red_disk = _find_disk(fg & ~soft & (seed | red), fg, soft, open_frac, grow_frac,
                          max_area_ratio, min_r0)
    disk = dark_disk
    if red_disk.any():
        cy, cx = ndi.center_of_mass(red_disk)
        yy, xx = np.nonzero(fg)
        r_flower = np.percentile(np.hypot(yy - cy, xx - cx), 95)
        if np.sqrt(red_disk.sum() / np.pi) <= max_disk_frac * r_flower:
            disk = red_disk | dark_disk

    petal = red & ~disk
    n = max(int(fg.sum()), 1)
    labels = np.where(fg, labels, -1)
    return dict(labels=labels, petal=petal, disk=disk,
                shares=dict(petal_bullseye=petal.sum() / n,
                            disk_red=(red & disk).sum() / n,
                            disk_area=disk.sum() / n))
