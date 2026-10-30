#!/usr/bin/env python3
"""
Geography of relative bullseye size in Rudbeckia hirta.

    python charisma_gpu/bullseye_maps.py --root C:/Users/nevyn/Documents/rhirta_bullseye \\
        --inat-parquet inat_db/data/inat_photos.parquet

Relative bullseye size (RBS) of a photo = red on the petals / (red on the
petals + yellow ray), over its usable segments, pixel-weighted. "Red on the
petals" is petal_bullseye: bullseye-coloured pixels outside the detected disk
(petal.py). The CLUT's other classes (disk, green, pale, the pink/purple
"other") are not in it. A segment is usable if it has >= --min-px pixels,
>= --min-ray yellow ray, and a disk was found, which drops tiny background
blobs and non-flower masks.

Reads   root/charisma/charisma_rhirta.csv, root/metadata/results.csv (mask
        scores), and photo coordinates from the iNat photo index parquet.
Writes  root/maps/: one PNG per figure, rhirta_bullseye_maps.pdf (all of them
        with captions), rhirta_bullseye_interactive.html, rbs_by_photo.csv,
        rbs_by_state_province.csv, correlogram.csv.
"""

import argparse
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402
import cartopy.crs as ccrs                                 # noqa: E402
import cartopy.feature as cfeature                         # noqa: E402
import cartopy.io.shapereader as shpreader                 # noqa: E402
import shapely                                             # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages       # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

# Reference palette (dataviz skill): one-hue blue sequential ramp, grey neutrals.
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE)
GREYS = LinearSegmentedColormap.from_list("seq_grey", ["#e4e3df", "#52514e"])
SURFACE, INK, INK2, NONE = "#fcfcfb", "#0b0b0b", "#52514e", "#c9c8c3"
LAND, WATER, LINES = "#f3f2ee", "#ffffff", "#a8a7a2"
TAXA = (62741, 81354, 242698, 242699, 242697)          # R. hirta and its subspecies
NA_EXTENT = (-128, -60, 18, 58)
TX_EXTENT = (-104.5, -92.5, 25.5, 36.5)
PROJ = ccrs.LambertConformal(central_longitude=-96, standard_parallels=(33, 45))

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "#d6d5d0", "axes.titleweight": "bold", "axes.titlesize": 12,
    "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
})
PCT = plt.FuncFormatter(lambda v, _: f"{v:.0%}")


# ── data ─────────────────────────────────────────────────────────────────────

def build_table(root, inat_parquet, min_px, min_ray):
    ch = pd.read_csv(root / "charisma" / "charisma_rhirta.csv")
    ch = ch[ch.mask_index >= 0]
    res = pd.read_csv(root / "metadata" / "results.csv", usecols=["photo_id", "mask_scores"])
    sc = res.assign(s=res.mask_scores.astype(str).str.split(",")).explode("s")
    sc["mask_index"] = sc.groupby("photo_id").cumcount()
    sc["mask_score"] = pd.to_numeric(sc.s, errors="coerce")
    ch = ch.drop(columns="mask_score", errors="ignore").merge(
        sc[["photo_id", "mask_index", "mask_score"]], on=["photo_id", "mask_index"], how="left")
    ok = (ch.n_pixels >= min_px) & (ch.px_ray >= min_ray) & (ch.disk_area > 0)
    seg = ch[ok].copy()
    seg["red_px"] = seg.petal_bullseye * seg.n_pixels
    seg["yellow_px"] = seg.px_ray * seg.n_pixels
    photo = seg.groupby("photo_id").agg(red_px=("red_px", "sum"), yellow_px=("yellow_px", "sum"),
                                        segments=("mask_index", "size"),
                                        mask_score=("mask_score", "max")).reset_index()
    photo["rbs"] = photo.red_px / (photo.red_px + photo.yellow_px)
    taxa = ", ".join(str(t) for t in TAXA)
    geo = duckdb.connect().sql(
        f"select photo_id, latitude, longitude from '{Path(inat_parquet).as_posix()}' "
        f"where taxon_id in ({taxa})").df().drop_duplicates("photo_id")
    photo = photo.merge(geo, on="photo_id", how="left")
    counts = dict(segments_all=len(ch), segments_used=int(ok.sum()),
                  photos_all=int(ch.photo_id.nunique()), photos_used=len(photo),
                  photos_geo=int(photo.latitude.notna().sum()))
    photo = photo.dropna(subset=["latitude", "longitude"])
    return photo, counts


# ── map helpers ──────────────────────────────────────────────────────────────

def base_map(ax, extent=NA_EXTENT, states=True):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=LAND, edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=WATER, edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor=WATER, edgecolor=LINES,
                   linewidth=0.3, zorder=1)
    if states:
        ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor=LINES, linewidth=0.3,
                       facecolor="none", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor=INK2, linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=INK2, linewidth=0.5, zorder=3)
    ax.spines["geo"].set_edgecolor("#d6d5d0")


def projected(df):
    xyz = PROJ.transform_points(ccrs.PlateCarree(), df.longitude.values, df.latitude.values)
    return xyz[:, 0], xyz[:, 1]


def in_extent(df, extent=NA_EXTENT, pad=(5, 3)):
    return df[(df.longitude.between(extent[0] - pad[0], extent[1] + pad[0])) &
              (df.latitude.between(extent[2] - pad[1], extent[3] + pad[1]))]


def new_map(title, subtitle, extent=NA_EXTENT):
    fig = plt.figure(figsize=(11, 7.2))
    ax = fig.add_axes([0.02, 0.06, 0.84, 0.82], projection=PROJ)
    base_map(ax, extent=extent)
    fig.text(0.02, 0.955, title, fontsize=15, fontweight="bold", color=INK)
    fig.text(0.02, 0.918, subtitle, fontsize=10, color=INK2)
    cax = fig.add_axes([0.875, 0.2, 0.018, 0.55])
    return fig, ax, cax


def colourbar(fig, mappable, cax, label, fmt=PCT, extend="neither"):
    cb = fig.colorbar(mappable, cax=cax, extend=extend)
    cb.set_label(label, color=INK2)
    cb.outline.set_visible(False)
    if fmt is not None:
        cb.ax.yaxis.set_major_formatter(fmt)
    return cb


def hex_map(df, values, title, subtitle, label, gridsize, mincnt, cmap=SEQ, vmin=0, vmax=None,
            fmt=PCT, reduce=np.mean, extent=NA_EXTENT, norm=None):
    fig, ax, cax = new_map(title, subtitle, extent)
    x, y = projected(df)
    kw = dict(norm=norm) if norm is not None else dict(vmin=vmin, vmax=vmax)
    hb = ax.hexbin(x, y, C=values, reduce_C_function=reduce, gridsize=gridsize, mincnt=mincnt,
                   cmap=cmap, extent=ax.get_extent(), linewidths=0.4, edgecolors=SURFACE,
                   zorder=2, **kw)
    colourbar(fig, hb, cax, label, fmt)
    return fig


def point_map(fig, ax, cax, df, present, vmax=0.6, transform=None, s=(2, 7)):
    x, y = projected(df) if transform is None else (df.longitude.values, df.latitude.values)
    kw = {} if transform is None else dict(transform=transform)
    none = df.rbs.to_numpy() < present
    ax.scatter(x[none], y[none], s=s[0], c=NONE, linewidths=0, zorder=2, **kw)
    r = df.rbs.to_numpy()[~none]
    order = np.argsort(r)
    sc = ax.scatter(x[~none][order], y[~none][order], s=s[1], c=r[order], cmap=SEQ,
                    vmin=present, vmax=vmax, linewidths=0.25, edgecolors=SURFACE, zorder=3, **kw)
    colourbar(fig, sc, cax, "relative bullseye size", extend="max")


# ── statistics ───────────────────────────────────────────────────────────────

def binned(df, col, value, edges, n_boot=400, min_n=30, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        v = df.loc[(df[col] >= lo) & (df[col] < hi), value].to_numpy()
        if len(v) < min_n:
            continue
        boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)
        out.append(((lo + hi) / 2, v.mean(), *np.percentile(boots, [2.5, 97.5]), len(v)))
    return pd.DataFrame(out, columns=["x", "mean", "lo", "hi", "n"])


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2 +
         np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(a))


def morans_correlogram(df, value, edges_km, n=4000, n_perm=199, seed=0):
    """Moran's I in distance bands, with a permutation 95% envelope."""
    rng = np.random.default_rng(seed)
    s = df.sample(min(n, len(df)), random_state=seed)
    lat, lon, z = s.latitude.to_numpy(), s.longitude.to_numpy(), s[value].to_numpy(float)
    i, j = np.triu_indices(len(s), k=1)
    d = haversine_km(lat[i], lon[i], lat[j], lon[j])
    band = np.digitize(d, edges_km) - 1
    keep = (band >= 0) & (band < len(edges_km) - 1)
    i, j, band = i[keep], j[keep], band[keep]
    nb = len(edges_km) - 1
    w = np.bincount(band, minlength=nb).astype(float)

    def moran(zz):
        zc = zz - zz.mean()
        num = np.bincount(band, weights=zc[i] * zc[j], minlength=nb)
        return (len(zz) / np.maximum(w, 1)) * num / (zc ** 2).sum()

    obs = moran(z)
    perm = np.array([moran(rng.permutation(z)) for _ in range(n_perm)])
    mids = (np.asarray(edges_km[:-1]) + np.asarray(edges_km[1:])) / 2
    return pd.DataFrame({"km": mids, "I": obs, "lo": np.percentile(perm, 2.5, 0),
                         "hi": np.percentile(perm, 97.5, 0), "pairs": w})


def admin1_summary(df):
    shp = shpreader.natural_earth(resolution="50m", category="cultural",
                                  name="admin_1_states_provinces_lakes")
    recs = [r for r in shpreader.Reader(shp).records()
            if r.attributes.get("adm0_a3") in ("USA", "CAN", "MEX")]
    px, py = df.longitude.to_numpy(), df.latitude.to_numpy()
    region = np.full(len(df), "", dtype=object)
    for r in recs:
        inside = shapely.contains_xy(r.geometry, px, py) & (region == "")
        region[inside] = r.attributes["name"]
    df = df.assign(region=region)
    g = df[df.region != ""].groupby("region").agg(
        n=("rbs", "size"), mean_rbs=("rbs", "mean"), prevalence=("has_bullseye", "mean"))
    return g, {r.attributes["name"]: r.geometry for r in recs}


def choropleth(g, geoms, col, title, subtitle, label, min_n=50):
    fig, ax, cax = new_map(title, subtitle)
    shown = g[g.n >= min_n]
    norm = plt.Normalize(0, float(shown[col].max()))
    for name, geom in geoms.items():
        if name in shown.index:
            colour = SEQ(norm(shown.loc[name, col]))
        elif name in g.index:
            colour = "#e9e8e4"      # too few photos to colour
        else:
            continue
        ax.add_geometries([geom], ccrs.PlateCarree(), facecolor=colour, edgecolor=SURFACE,
                          linewidth=0.6, zorder=2)
    colourbar(fig, plt.cm.ScalarMappable(norm=norm, cmap=SEQ), cax, label)
    fig.text(0.02, 0.03, f"Pale grey: fewer than {min_n} photos.", fontsize=8.5, color=INK2)
    return fig


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--inat-parquet", required=True)
    p.add_argument("--min-px", type=int, default=1000)
    p.add_argument("--min-ray", type=float, default=0.25)
    p.add_argument("--present", type=float, default=0.05,
                   help="RBS at or above which a flower counts as having a bullseye")
    args = p.parse_args()
    root = Path(args.root)
    out = root / "maps"
    out.mkdir(parents=True, exist_ok=True)

    photo, counts = build_table(root, args.inat_parquet, args.min_px, args.min_ray)
    photo["has_bullseye"] = (photo.rbs >= args.present).astype(float)
    photo["inat_url"] = "https://www.inaturalist.org/photos/" + photo.photo_id.astype(str)
    photo.to_csv(out / "rbs_by_photo.csv", index=False)
    na = in_extent(photo)
    present = na[na.has_bullseye == 1]
    pres_lbl = f"RBS ≥ {args.present:.0%}"
    print(counts, "| in North America:", len(na), "| with bullseye:", len(present))

    figs = []          # (name, figure, caption)

    # 1. every photo
    fig, ax, cax = new_map("Relative bullseye size, every photo",
                           f"{len(na):,} R. hirta photos; grey = no bullseye (< {args.present:.0%}); "
                           "blue = red share of the petal colour")
    point_map(fig, ax, cax, na, args.present)
    figs.append(("01_points", fig, "Every photo with coordinates. Photos without a bullseye are "
                 "grey and drawn underneath; the rest are coloured by relative bullseye size, "
                 "largest on top."))

    # 2-5. hexbins
    grid, mincnt = (55, 32), 15
    figs.append(("02_hex_prevalence", hex_map(
        na, na.has_bullseye, "How often a bullseye is present",
        f"Share of photos per hexagon with {pres_lbl} (hexagons with ≥ {mincnt} photos)",
        "share of photos with a bullseye", grid, mincnt, vmax=0.25),
        f"Prevalence: the share of photos in each hexagon with relative bullseye size at or above "
        f"{args.present:.0%}. Colour scale capped at 25%."))
    figs.append(("03_hex_mean", hex_map(
        na, na.rbs, "Mean relative bullseye size",
        f"Mean over all photos per hexagon, zeros included (hexagons with ≥ {mincnt} photos)",
        "mean relative bullseye size", grid, mincnt, vmax=0.06),
        "Mean relative bullseye size per hexagon, counting flowers without a bullseye as 0. "
        "This mixes how often and how big; maps 2 and 4 separate them."))
    figs.append(("04_hex_size_when_present", hex_map(
        present, present.rbs, "Bullseye size when there is one",
        f"Median relative bullseye size of photos with {pres_lbl} (hexagons with ≥ 5 such photos)",
        "median relative bullseye size", (40, 23), 5, vmin=args.present, vmax=0.35,
        reduce=np.median),
        "Among flowers that have a bullseye, how big it is. Separates intensity from prevalence."))
    figs.append(("05_hex_sample_size", hex_map(
        na, np.ones(len(na)), "Where the photos are",
        "Photos per hexagon (log scale): read the other maps against this", "photos per hexagon",
        grid, 1, cmap=GREYS, fmt=plt.FuncFormatter(lambda v, _: f"{v:,.0f}"), reduce=np.sum,
        norm=LogNorm(vmin=1, vmax=1000)),
        "Sampling effort: iNaturalist photos per hexagon. Dense hexagons give stable estimates; "
        "sparse ones are noisy."))

    # 5b. the Texas-Oklahoma hotspot, finer hexagons
    tx = in_extent(photo, TX_EXTENT, pad=(0, 0))
    figs.append(("05b_texas_prevalence", hex_map(
        tx, tx.has_bullseye, "Texas and Oklahoma close-up: how often a bullseye is present",
        f"Share of photos per hexagon with {pres_lbl} (hexagons with ≥ 8 photos); "
        f"{len(tx):,} photos", "share of photos with a bullseye", (34, 30), 8, vmax=0.4,
        extent=TX_EXTENT),
        "The prevalence hotspot at finer resolution."))

    # 6. classes
    fig, ax, cax = new_map("Bullseye size classes",
                           "Each photo in one class; larger classes drawn on top")
    x, y = projected(na)
    bounds = [args.present, 0.15, 0.30, 0.50]
    names = ["none", f"small ({args.present:.0%}–15%)", "medium (15–30%)",
             "large (30–50%)", "dominant (≥ 50%)"]
    colours = [NONE, BLUE[3], BLUE[6], BLUE[9], BLUE[12]]
    cls = np.digitize(na.rbs, bounds)
    for k in range(len(names)):
        m = cls == k
        ax.scatter(x[m], y[m], s=2 if k == 0 else 5 + 3 * k, c=colours[k], linewidths=0.25,
                   edgecolors=SURFACE if k else "none", zorder=2 + k,
                   label=f"{names[k]}  (n = {m.sum():,})")
    cax.remove()
    ax.legend(loc="lower left", frameon=True, facecolor=SURFACE, edgecolor="#d6d5d0",
              fontsize=9, markerscale=1.8, title="relative bullseye size", title_fontsize=9)
    figs.append(("06_classes", fig, "The same photos in five classes. 'Dominant' flowers are "
                 "mostly red-rayed forms, where red covers most of the ray."))

    # 7-8. states and provinces
    g, geoms = admin1_summary(na)
    g.sort_values("n", ascending=False).to_csv(out / "rbs_by_state_province.csv")
    figs.append(("07_states_prevalence", choropleth(
        g, geoms, "prevalence", "Bullseye prevalence by state and province",
        f"Share of photos with {pres_lbl}", "share of photos with a bullseye"),
        "Prevalence per US state, Canadian province and Mexican state with at least 50 photos."))
    figs.append(("08_states_mean", choropleth(
        g, geoms, "mean_rbs", "Mean relative bullseye size by state and province",
        "Zeros included", "mean relative bullseye size"),
        "Mean relative bullseye size per state or province, zeros included, at least 50 photos."))

    # 9. gradients: small multiples, one y-axis each
    lat_edges = np.arange(24, 56, 1.0)
    lon_edges = np.arange(-125, -62, 2.0)
    panels = [("latitude", lat_edges, "has_bullseye", na, "Prevalence vs latitude", "latitude (°N)",
               "share with a bullseye"),
              ("latitude", lat_edges, "rbs", present, "Size when present vs latitude",
               "latitude (°N)", "mean RBS (bullseye flowers)"),
              ("longitude", lon_edges, "has_bullseye", na, "Prevalence vs longitude",
               "longitude (°E)", "share with a bullseye"),
              ("longitude", lon_edges, "rbs", present, "Size when present vs longitude",
               "longitude (°E)", "mean RBS (bullseye flowers)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    fig.subplots_adjust(top=0.86, hspace=0.42, wspace=0.25)
    fig.text(0.06, 0.955, "Bullseye across latitude and longitude", fontsize=15, fontweight="bold")
    fig.text(0.06, 0.918, "Bin means with bootstrap 95% intervals (≥ 30 photos per bin for "
             "prevalence, ≥ 15 bullseye photos for size); gaps = too few photos", fontsize=10,
             color=INK2)
    for ax, (col, edges, val, data, title, xl, yl) in zip(axes.ravel(), panels):
        b = binned(data, col, val, edges, min_n=30 if val == "has_bullseye" else 15)
        # break the line and band where bins are missing, so gaps aren't drawn as data
        full = pd.DataFrame({"x": (edges[:-1] + edges[1:]) / 2}).merge(b, on="x", how="left")
        ax.fill_between(full.x, full.lo, full.hi, color=BLUE[2], linewidth=0)
        ax.plot(full.x, full["mean"], color=BLUE[8], linewidth=2)
        ax.scatter(b.x, b["mean"], s=14, color=BLUE[8], zorder=3, edgecolors=SURFACE,
                   linewidths=1)
        ax.set_title(title, loc="left")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.yaxis.set_major_formatter(PCT)
        ax.grid(axis="y", color="#ecebe7", linewidth=0.8)
        ax.set_ylim(bottom=0)
    figs.append(("09_gradients", fig, "Prevalence (all flowers) and size when present (bullseye "
                 "flowers only), binned by 1° of latitude and 2° of longitude."))

    # 10. spatial autocorrelation
    edges_km = [0, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    fig.subplots_adjust(top=0.78, wspace=0.08)
    fig.text(0.06, 0.94, "How far does the pattern reach? Spatial correlogram", fontsize=15,
             fontweight="bold")
    fig.text(0.06, 0.885, "Moran's I by distance band (4,000 random photos); grey band = 95% of "
             "199 random relabellings", fontsize=10, color=INK2)
    cor_rows = []
    for ax, (val, title) in zip(axes, [("has_bullseye", "Bullseye present"),
                                       ("rbs", "Relative bullseye size")]):
        c = morans_correlogram(na, val, edges_km)
        c["variable"] = val
        cor_rows.append(c)
        ax.fill_between(c.km, c.lo, c.hi, color="#e4e3df", linewidth=0)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.plot(c.km, c.I, color=BLUE[8], linewidth=2)
        ax.scatter(c.km, c.I, s=16, color=BLUE[8], zorder=3, edgecolors=SURFACE, linewidths=1)
        ax.set_xscale("log")
        ax.set_title(title, loc="left")
        ax.set_xlabel("distance between photos (km, log scale)")
        ax.grid(axis="y", color="#ecebe7", linewidth=0.8)
    axes[0].set_ylabel("Moran's I")
    pd.concat(cor_rows).to_csv(out / "correlogram.csv", index=False)
    figs.append(("10_correlogram", fig, "Moran's I above the grey band means photos that "
                 "distance apart are more alike than chance. Where it falls into the band is the "
                 "scale of the spatial pattern."))

    # 11. world
    fig = plt.figure(figsize=(11, 5.6))
    ax = fig.add_axes([0.02, 0.06, 0.84, 0.8], projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor=LAND, edgecolor="none")
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor=INK2, linewidth=0.4)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), edgecolor=LINES, linewidth=0.3)
    cax = fig.add_axes([0.875, 0.2, 0.018, 0.55])
    point_map(fig, ax, cax, photo, args.present, transform=ccrs.PlateCarree(), s=(2, 6))
    fig.text(0.02, 0.93, "Every photo, worldwide", fontsize=15, fontweight="bold")
    fig.text(0.02, 0.895, f"{len(photo):,} photos; {len(photo) - len(na):,} outside North America "
             "(introduced or cultivated)", fontsize=10, color=INK2)
    figs.append(("11_world", fig, "All photos worldwide. Outside North America R. hirta is "
                 "introduced or cultivated."))

    # save PNGs + one PDF
    with PdfPages(out / "rhirta_bullseye_maps.pdf") as pdf:
        cover = plt.figure(figsize=(11, 7.2))
        cover.text(0.06, 0.82, "Rudbeckia hirta: where are the bullseyes?", fontsize=22,
                   fontweight="bold")
        lines = [
            "Relative bullseye size (RBS) = red on the petals ÷ (red on the petals + yellow ray), "
            "per photo.",
            "Red on the petals = bullseye-coloured pixels outside the detected disk; disk, green, "
            "pale and pink/purple",
            "'other' pixels are not counted. Pooled over a photo's usable SAM3 segments, weighted "
            "by pixels.",
            "",
            f"Usable segment: ≥ {args.min_px:,} pixels, ≥ {args.min_ray:.0%} yellow ray, and a "
            "disk found.",
            f"Segments: {counts['segments_used']:,} of {counts['segments_all']:,} usable.   "
            f"Photos: {counts['photos_used']:,} of {counts['photos_all']:,}, "
            f"{counts['photos_geo']:,} with coordinates, {len(na):,} in North America.",
            f"A photo 'has a bullseye' at RBS ≥ {args.present:.0%}: {int(na.has_bullseye.sum()):,} "
            f"of {len(na):,} North American photos ({na.has_bullseye.mean():.1%}).",
            "",
            "Read with care: iNaturalist photos follow people, not plants (see the sampling map); "
            "red-rayed garden",
            "cultivars count as bullseyes; dense places weigh more; photo date and flower age "
            "(bullseyes can fade)",
            "are not controlled for.",
        ]
        for k, line in enumerate(lines):
            cover.text(0.06, 0.70 - k * 0.045, line, fontsize=11, color=INK if k < 3 else INK2)
        pdf.savefig(cover)
        plt.close(cover)
        for name, fig, caption in figs:
            fig.savefig(out / f"{name}.png", dpi=170)
            fig.text(0.02, 0.008, caption, fontsize=8.5, color=INK2)
            pdf.savefig(fig)
            plt.close(fig)

    # interactive map
    import plotly.graph_objects as go
    # North America only: points far outside the clipped conic view leave plotly's geo blank
    pts = na.sort_values("rbs")
    hover = ("photo %{customdata[0]}<br>RBS %{customdata[1]:.1%}<br>"
             "%{customdata[2]} segment(s)<extra></extra>")
    figi = go.Figure()
    for sel, name, colour, size in ((pts.rbs < args.present, "no bullseye", NONE, 3),
                                    (pts.rbs >= args.present, "bullseye", None, 5)):
        d = pts[sel]
        marker = dict(size=size, color=colour, line=dict(width=0)) if colour else dict(
            size=size, color=d.rbs,
            colorscale=[[k / (len(BLUE) - 1), c] for k, c in enumerate(BLUE)],
            cmin=args.present, cmax=0.6, colorbar=dict(title="RBS", tickformat=".0%"),
            line=dict(width=0.3, color=SURFACE))
        figi.add_trace(go.Scattergeo(lon=d.longitude, lat=d.latitude, mode="markers", name=name,
                                     marker=marker, customdata=d[["photo_id", "rbs", "segments"]],
                                     hovertemplate=hover))
    figi.update_geos(scope="north america", projection_type="conic conformal",
                     projection_parallels=[33, 45], projection_rotation=dict(lon=-96),
                     lataxis_range=[18, 58], lonaxis_range=[-128, -60],
                     showland=True, landcolor=LAND, showlakes=True, lakecolor=WATER,
                     showocean=True, oceancolor=WATER, showsubunits=True, subunitcolor=LINES,
                     countrycolor=INK2, coastlinecolor=INK2)
    figi.update_layout(title="R. hirta relative bullseye size (hover a point; photos open at "
                             "inaturalist.org/photos/&lt;id&gt;)", paper_bgcolor=SURFACE,
                       height=800, legend=dict(orientation="h", y=-0.02))
    figi.write_html(out / "rhirta_bullseye_interactive.html", include_plotlyjs=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
