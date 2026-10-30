"""
charisma (Schwartz et al., R package v1.0.0) in PyTorch, batched over many
segments at once so it runs on a GPU.

What R's charisma(img, threshold = 0, pavo = FALSE) does, and what this
reproduces step for step:

  1. background = pixels whose rounded alpha is < 1 (alpha < 128 of 255)
  2. bin every other pixel into a bins^3 histogram in sRGB [0, 1] (right-closed
     intervals); a bin's colour is the mean of its pixels
  3. drop empty bins; complete-linkage hierarchical clustering of the bin colours
     in CIE Lab (D65), cut at height `cutoff`; each group's colour is the
     pixel-count weighted mean of its bins in sRGB. Quirk kept from R: if that
     leaves 2 or fewer groups the merge is skipped and every bin stays a colour
  4. reassign every pixel to the nearest of those colours in Lab, then move each
     colour to the Lab mean of its pixels (recolorize::imposeColors)
  5. name each colour with the CLUT (first matching row, "black" if none, as R
     does) and add up the pixel shares per name

classify_pixels() is the alternative: every pixel is named by the CLUT
directly, with no binning or clustering, so a segment's share of a colour is
exactly the share of its pixels that fall in that CLUT region.

pavo's adjacency statistics (charisma's pavo = TRUE) are not reproduced.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── CLUT ─────────────────────────────────────────────────────────────────────

EPS_HI = 0.9999     # charisma's ranges are lo::hi meaning lo <= x < hi.9999


@dataclass
class Clut:
    """A charisma colour look-up table: per colour name, a list of rows, each
    row a (h ranges, s ranges, v ranges) triple; a colour matches when any row
    has its h, s and v each inside one of that row's ranges. h is degrees
    [0, 360), s and v percent."""
    names: list
    hexes: list
    rules: list          # per name: list of (h, s, v), each a list of (lo, hi)

    @classmethod
    def from_csv(cls, path):
        df = pd.read_csv(path, dtype=str)
        names, hexes, rules = [], [], []
        for row in df.itertuples(index=False):
            hs, ss, vs = (str(getattr(row, c)).split(",") for c in ("h", "s", "v"))
            if not len(hs) == len(ss) == len(vs):
                raise ValueError(f"{row[0]}: h, s and v need the same number of ranges")
            parsed = []
            for h, s, v in zip(hs, ss, vs):
                parsed.append(tuple(
                    [tuple(float(x) for x in part.strip().split("::"))
                     for part in field.split("|")]
                    for field in (h, s, v)))
            names.append(row[0])
            hexes.append(row[1])
            rules.append(parsed)
        return cls(names, hexes, rules)

    def label(self, hsv):
        """hsv: [N, 3] tensor (h degrees, s %, v %) -> [N] index into names.
        The first matching name wins; no match gives index 0, as in R."""
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        out = torch.full((hsv.shape[0],), -1, dtype=torch.long, device=hsv.device)
        for i, rows in enumerate(self.rules):
            hit = torch.zeros_like(h, dtype=torch.bool)
            for hr, sr, vr in rows:
                ok = None
                for x, ranges in ((h, hr), (s, sr), (v, vr)):
                    inside = torch.zeros_like(hit)
                    for lo, hi in ranges:
                        inside |= (x >= lo) & (x < hi + EPS_HI)
                    ok = inside if ok is None else ok & inside
                hit |= ok
            out = torch.where((out < 0) & hit, torch.full_like(out, i), out)
        return out.clamp(min=0)


DEFAULT_CLUT = Path(__file__).resolve().parent / "clut_default.csv"

# ── colour spaces, as R's grDevices::convertColor does them ──────────────────


def _rgb_to_xyz_matrix():
    """grDevices::make.rgb for sRGB: primaries and D65 white -> M with
    XYZ = linear_rgb @ M."""
    def c2to3(x, y):
        return np.array([x / y, 1.0, (1 - x - y) / y])
    white = c2to3(0.3127, 0.3290)
    prim = np.stack([c2to3(0.64, 0.33), c2to3(0.30, 0.60), c2to3(0.15, 0.06)])
    s = white @ np.linalg.inv(prim)
    return s[:, None] * prim, white


_M, _WHITE = _rgb_to_xyz_matrix()
_EPS = 216 / 24389
_KAPPA = 24389 / 27


def srgb_to_lab(rgb):
    """[N, 3] sRGB in [0, 1] -> CIE Lab (D65)."""
    m = torch.as_tensor(_M, dtype=rgb.dtype, device=rgb.device)
    w = torch.as_tensor(_WHITE, dtype=rgb.dtype, device=rgb.device)
    lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = (lin @ m) / w
    f = torch.where(xyz > _EPS, xyz.clamp(min=0) ** (1 / 3), (_KAPPA * xyz + 16) / 116)
    return torch.stack([116 * f[:, 1] - 16,
                        500 * (f[:, 0] - f[:, 1]),
                        200 * (f[:, 1] - f[:, 2])], dim=1)


def lab_to_srgb(lab):
    """CIE Lab (D65) -> sRGB in [0, 1], clipped (convertColor's clip = TRUE)."""
    m = torch.as_tensor(np.linalg.inv(_M), dtype=lab.dtype, device=lab.device)
    w = torch.as_tensor(_WHITE, dtype=lab.dtype, device=lab.device)
    fy = (lab[:, 0] + 16) / 116
    fx = fy + lab[:, 1] / 500
    fz = fy - lab[:, 2] / 200
    f = torch.stack([fx, fy, fz], dim=1)
    xyz = torch.where(f ** 3 > _EPS, f ** 3, (116 * f - 16) / _KAPPA)
    # Y uses L directly below the cube-root knee, exactly as the standard does.
    y_lin = torch.where(lab[:, 0] > _KAPPA * _EPS, fy ** 3, lab[:, 0] / _KAPPA)
    xyz = torch.stack([xyz[:, 0], y_lin, xyz[:, 2]], dim=1) * w
    lin = xyz @ m
    rgb = torch.where(lin <= 0.0031308, 12.92 * lin,
                      1.055 * lin.clamp(min=0) ** (1 / 2.4) - 0.055)
    return rgb.clamp(0, 1)


def srgb_to_hsv(rgb):
    """[N, 3] sRGB in [0, 1] -> (h degrees [0, 360), s %, v %), as R's rgb2hsv."""
    mx, _ = rgb.max(dim=1)
    mn, _ = rgb.min(dim=1)
    delta = mx - mn
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    safe = torch.where(delta > 0, delta, torch.ones_like(delta))
    h = torch.where(mx == r, ((g - b) / safe) % 6,
        torch.where(mx == g, (b - r) / safe + 2, (r - g) / safe + 4))
    h = torch.where(delta > 0, h * 60, torch.zeros_like(h))
    s = torch.where(mx > 0, delta / torch.where(mx > 0, mx, torch.ones_like(mx)),
                    torch.zeros_like(mx))
    return torch.stack([h, s * 100, mx * 100], dim=1)


# ── the batch ────────────────────────────────────────────────────────────────

def pack(segments, device):
    """A list of [P_i, 3] uint8 arrays (foreground pixels only) -> one float
    [P, 3] tensor in [0, 1] and the segment index of every pixel."""
    sizes = [len(s) for s in segments]
    pix = torch.from_numpy(np.concatenate(segments)).to(device, non_blocking=True)
    seg = torch.repeat_interleave(torch.arange(len(segments), device=device),
                                  torch.as_tensor(sizes, device=device))
    return pix.float() / 255, seg


def complete_linkage_groups(points, cutoff):
    """Group labels from complete-linkage clustering cut at height `cutoff`
    (hclust(method = "complete") + cutree(h = cutoff)). Complete-linkage heights
    only rise, so merging closest pairs until the next merge would exceed the
    cutoff gives the same partition. Labels are numbered by first appearance,
    as cutree numbers them."""
    n = len(points)
    d = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    members = {i: [i] for i in range(n)}
    alive = np.ones(n, dtype=bool)
    while alive.sum() > 1:
        sub = np.where(alive[:, None] & alive[None, :], d, np.inf)
        i, j = np.unravel_index(np.argmin(sub), sub.shape)
        if sub[i, j] > cutoff:
            break
        i, j = min(i, j), max(i, j)
        merged = np.maximum(d[i], d[j])          # complete linkage: farthest pair
        d[i, :] = merged
        d[:, i] = merged
        d[i, i] = np.inf
        d[j, :] = d[:, j] = np.inf
        alive[j] = False
        members[i] += members.pop(j)
    label = np.empty(n, dtype=int)
    for g, idx in enumerate(members.values()):
        label[idx] = g
    order = {g: k for k, g in enumerate(dict.fromkeys(label))}
    return np.array([order[g] for g in label])


def _merge_groups(centers, sizes, cutoff):
    """One segment's non-empty bins -> its merged colours (sRGB), as
    charisma_recluster with refit_method = 'imposeColors'."""
    if len(centers) < 2:
        return centers
    lab = srgb_to_lab(torch.from_numpy(centers)).numpy()
    groups = complete_linkage_groups(lab, cutoff)
    order = list(dict.fromkeys(groups))
    if len(order) <= 2:
        return centers
    merged = []
    for g in order:
        idx = groups == g
        merged.append((centers[idx] * sizes[idx, None]).sum(0) / sizes[idx].sum())
    return np.stack(merged)


def charisma_batch(segments, clut, bins=4, cutoff=20.0, device="cuda", chunk=2_000_000):
    """charisma on many segments. segments: list of [P_i, 3] uint8 foreground
    pixel arrays. Returns, per segment, a dict: k, props {name: share},
    centers [K, 3] sRGB 0-255, sizes [K], labels [K].

    Runs in float64, as R does: a colour centre that lands on a CLUT boundary
    (e.g. v = 75.99 vs 76.01) changes the name of a whole cluster, so float32
    rounding in the pixel assignment is enough to flip one."""
    pix, seg = pack(segments, device)
    pix = pix.double()
    n_seg = len(segments)
    nb = bins ** 3

    # 2. histogram bins, c1 fastest (expand.grid order)
    b = (torch.ceil(pix * bins).clamp(1, bins) - 1).long()
    lin = b[:, 0] + bins * b[:, 1] + bins * bins * b[:, 2]
    key = seg * nb + lin
    counts = torch.bincount(key, minlength=n_seg * nb).view(n_seg, nb)
    sums = torch.zeros(n_seg * nb, 3, device=device, dtype=pix.dtype).index_add_(0, key, pix).view(n_seg, nb, 3)
    centers = (sums / counts.clamp(min=1).unsqueeze(-1)).double().cpu().numpy()
    counts_np = counts.cpu().numpy()

    # 3. per-segment hierarchical merge (<= bins^3 colours each; CPU)
    merged = []
    for i in range(n_seg):
        keep = counts_np[i] > 0
        merged.append(_merge_groups(centers[i][keep], counts_np[i][keep].astype(float), cutoff))
    k_max = max(1, max(len(m) for m in merged))
    ctr = np.full((n_seg, k_max, 3), np.nan)
    for i, m in enumerate(merged):
        ctr[i, :len(m)] = m
    ctr_t = torch.from_numpy(ctr).to(device)
    ctr_lab = srgb_to_lab(ctr_t.view(-1, 3).nan_to_num(0)).view(n_seg, k_max, 3)
    valid = ~torch.isnan(ctr_t[..., 0])

    # 4. nearest colour in Lab, then each colour -> Lab mean of its pixels
    assign = torch.empty(len(pix), dtype=torch.long, device=device)
    for start in range(0, len(pix), chunk):
        sl = slice(start, start + chunk)
        lab = srgb_to_lab(pix[sl])
        d = ((lab[:, None, :] - ctr_lab[seg[sl]]) ** 2).sum(-1)
        d = d.masked_fill(~valid[seg[sl]], float("inf"))
        assign[sl] = d.argmin(dim=1)
    key = seg * k_max + assign
    size = torch.bincount(key, minlength=n_seg * k_max).view(n_seg, k_max)
    lab_sum = torch.zeros(n_seg * k_max, 3, device=device, dtype=pix.dtype)
    for start in range(0, len(pix), chunk):
        sl = slice(start, start + chunk)
        lab_sum.index_add_(0, key[sl], srgb_to_lab(pix[sl]))
    lab_mean = lab_sum.view(n_seg, k_max, 3) / size.clamp(min=1).unsqueeze(-1)
    final_lab = torch.where(size.unsqueeze(-1) > 0, lab_mean, ctr_lab).double()
    final_rgb = lab_to_srgb(final_lab.view(-1, 3)).view(n_seg, k_max, 3)

    # 5. CLUT names on 0-255 colours, as color2label does
    rgb255 = final_rgb * 255
    labels = clut.label(srgb_to_hsv(rgb255.view(-1, 3) / 255)).view(n_seg, k_max)

    out = []
    size_np, labels_np, rgb_np, valid_np = (x.cpu().numpy() for x in (size, labels, rgb255, valid))
    for i in range(n_seg):
        v = valid_np[i]
        sz, lb = size_np[i][v], labels_np[i][v]
        total = sz.sum()
        props = {}
        for name_idx in np.unique(lb):
            props[clut.names[name_idx]] = float(sz[lb == name_idx].sum() / total) if total else 0.0
        out.append(dict(k=len(props), props=props, centers=rgb_np[i][v], sizes=sz,
                        labels=[clut.names[j] for j in lb]))
    return out


def classify_pixels(segments, clut, device="cuda"):
    """Every pixel named by the CLUT directly. Returns [n_segments, n_names]
    shares (rows sum to 1) and the mean sRGB 0-255 colour of each name."""
    pix, seg = pack(segments, device)
    n_seg, n_names = len(segments), len(clut.names)
    lab = clut.label(srgb_to_hsv(pix))
    key = seg * n_names + lab
    counts = torch.bincount(key, minlength=n_seg * n_names).view(n_seg, n_names).double()
    rgb_sum = torch.zeros(n_seg * n_names, 3, device=device, dtype=torch.float64)
    rgb_sum.index_add_(0, key, pix.double())
    mean_rgb = rgb_sum.view(n_seg, n_names, 3) / counts.clamp(min=1).unsqueeze(-1) * 255
    shares = counts / counts.sum(1, keepdim=True).clamp(min=1)
    return shares.cpu().numpy(), mean_rgb.cpu().numpy()
