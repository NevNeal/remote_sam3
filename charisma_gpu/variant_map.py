#!/usr/bin/env python3
"""
Where to expect regional variants, cultivars and hybrids of Rudbeckia hirta,
worldwide: literature ranges plus records.

    python charisma_gpu/variant_map.py --root C:/Users/nevyn/Documents/rhirta_bullseye \\
        --inat-parquet inat_db/data/inat_photos.parquet

Layers
  varieties   approximate ranges of var. hirta, angustifolia and floridana from
              Perdue (1957) as summarised in Flora of North America; var.
              pulcherrima occurs nearly throughout the native range
  records     iNaturalist photos identified to variety (local photo index);
              GBIF records outside the US and Canada (root/metadata/
              gbif_rhirta_outside_us_ca.csv, from the GBIF occurrence API)
  this study  0.5-degree cells where >= 20% of photos have a bullseye (wild
              red-based forms), and red-rayed flowers with RBS >= 50%, which
              are mostly Gloriosa-type garden cultivars (root/maps/rbs_by_photo.csv)
  breeding    where the main cultivar lines were bred or introduced

Writes root/maps/variants/: PNGs, rhirta_variants_map.pdf (with a sources
page) and rhirta_variants_interactive.html.
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
from shapely.geometry import Polygon, box                  # noqa: E402
from shapely.ops import unary_union                        # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages       # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D                        # noqa: E402
from matplotlib.patches import Patch                       # noqa: E402

# Reference palette (dataviz skill). Three categorical slots only (all-pairs safe on maps).
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"           # blue, orange, aqua
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE)
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
LAND, WATER, LINES, NATIVE = "#f3f2ee", "#ffffff", "#a8a7a2", "#e2ded2"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "font.size": 9.5,
})

VARIETY_TAXA = {81354: "pulcherrima", 242699: "hirta", 242697: "angustifolia",
                242698: "floridana"}
HIRTA_STATES = ["Pennsylvania", "New Jersey", "Delaware", "Maryland", "District of Columbia",
                "Virginia", "West Virginia", "North Carolina", "South Carolina", "Georgia",
                "Tennessee", "Kentucky", "Ohio", "Indiana", "Illinois"]
GULF_STATES = ["South Carolina", "Georgia", "Florida", "Alabama", "Mississippi", "Louisiana",
               "Texas"]
# Inland edge of the Gulf and south Atlantic coastal plain, roughly (lon, lat), then the coast.
COASTAL_PLAIN = Polygon([(-78.5, 33.8), (-80.0, 34.6), (-81.5, 33.9), (-82.6, 33.4),
                         (-84.0, 32.6), (-85.2, 32.4), (-86.6, 32.5), (-88.3, 33.3),
                         (-90.4, 32.6), (-92.5, 32.1), (-94.6, 31.8), (-96.5, 30.8),
                         (-97.8, 29.3), (-98.4, 27.8), (-97.9, 25.8), (-96.0, 25.0),
                         (-80.0, 24.0), (-76.0, 33.0)])
BREEDING = [  # name, lon, lat, what
    ("Smith College, MA", -72.64, 42.32, "Blakeslee: colchicine tetraploids, 1930s–50s"),
    ("Burpee, PA", -75.10, 40.21, "'Gloriosa Daisy' launched 1957"),
    ("Benary, Germany", 9.65, 51.42, "'Prairie Sun' (AAS 2003), 'Denver Daisy'"),
    ("Thompson & Morgan, UK", 1.16, 52.06, "'Cherokee Sunset' (AAS 2002)"),
    ("Denver Botanic Gardens, CO", -104.96, 39.73, "'Denver Daisy' introduced (Plant Select)"),
    ("Pacific Plug & Liner, CA", -121.76, 36.91,
     "Echibeckia 'Summerina' (Echinacea × R. hirta)"),
]


def base(ax, extent, scale="50m", states=True):
    if extent is None:
        ax.set_global()
    else:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale(scale), facecolor=WATER, edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale(scale), facecolor=LAND, edgecolor="none", zorder=0)
    if states:
        ax.add_feature(cfeature.STATES.with_scale(scale), edgecolor=LINES, linewidth=0.3,
                       facecolor="none", zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale(scale), edgecolor=INK2, linewidth=0.45, zorder=4)
    ax.add_feature(cfeature.COASTLINE.with_scale(scale), edgecolor=INK2, linewidth=0.45, zorder=4)
    ax.spines["geo"].set_edgecolor("#d6d5d0")


def titles(fig, title, subtitle):
    fig.text(0.02, 0.955, title, fontsize=15, fontweight="bold")
    fig.text(0.02, 0.918, subtitle, fontsize=10, color=INK2)


def admin1():
    shp = shpreader.natural_earth("50m", "cultural", "admin_1_states_provinces_lakes")
    return {(r.attributes["adm0_a3"], r.attributes["name"]): r.geometry
            for r in shpreader.Reader(shp).records()
            if r.attributes.get("adm0_a3") in ("USA", "CAN")}


def native_range(a1):
    us = [g for (c, n), g in a1.items() if c == "USA" and n not in ("Alaska", "Hawaii")]
    ca = [g for (c, n), g in a1.items() if c == "CAN" and n not in
          ("Yukon", "Northwest Territories", "Nunavut")]
    rng = unary_union(us + ca)
    return rng.intersection(box(-170, 24, -50, 56))   # southern Canada only


def variety_shapes(a1):
    hirta = unary_union([a1[("USA", n)] for n in HIRTA_STATES])
    gulf = unary_union([a1[("USA", n)] for n in GULF_STATES])
    angust = COASTAL_PLAIN.intersection(gulf)
    flor = box(-83.6, 24.3, -79.5, 29.6).intersection(a1[("USA", "Florida")])
    return hirta, angust, flor


def hotspot_cells(photo, present=0.05, cell=0.5, min_n=15, min_prev=0.2):
    p = photo.assign(cx=np.floor(photo.longitude / cell) * cell,
                     cy=np.floor(photo.latitude / cell) * cell,
                     b=(photo.rbs >= present).astype(float))
    g = p.groupby(["cx", "cy"]).agg(n=("b", "size"), prev=("b", "mean")).reset_index()
    g = g[(g.n >= min_n) & (g.prev >= min_prev)]
    return unary_union([box(x, y, x + cell, y + cell) for x, y in zip(g.cx, g.cy)]), len(g)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--inat-parquet", required=True)
    args = p.parse_args()
    root = Path(args.root)
    out = root / "maps" / "variants"
    out.mkdir(parents=True, exist_ok=True)

    gbif = pd.read_csv(root / "metadata" / "gbif_rhirta_outside_us_ca.csv").dropna(
        subset=["lat", "lon"])
    photo = pd.read_csv(root / "maps" / "rbs_by_photo.csv")
    na_photo = photo[photo.longitude.between(-130, -55) & photo.latitude.between(15, 60)]
    red_rayed = na_photo[na_photo.rbs >= 0.5]
    taxa = ", ".join(str(t) for t in VARIETY_TAXA)
    var_pts = duckdb.connect().sql(
        f"select taxon_id, latitude, longitude from '{Path(args.inat_parquet).as_posix()}' "
        f"where taxon_id in ({taxa}) and latitude is not null").df()
    var_pts["variety"] = var_pts.taxon_id.map(VARIETY_TAXA)

    a1 = admin1()
    native = native_range(a1)
    hirta, angust, flor = variety_shapes(a1)
    hot, n_hot = hotspot_cells(na_photo)
    pc = ccrs.PlateCarree()
    figs = []

    # ── 1. world ────────────────────────────────────────────────────────────
    shp = shpreader.natural_earth("110m", "cultural", "admin_0_countries")
    recs = list(shpreader.Reader(shp).records())
    counts = gbif.country.value_counts()
    fig = plt.figure(figsize=(12, 6.6))
    ax = fig.add_axes([0.01, 0.07, 0.86, 0.8], projection=ccrs.Robinson())
    base(ax, None, "110m", states=False)
    norm = LogNorm(vmin=1, vmax=max(2, counts.max()))
    for r in recs:
        iso = r.attributes.get("ISO_A2_EH") or r.attributes.get("ISO_A2")
        if iso in ("US", "CA"):
            continue
        geom = r.geometry
        if iso == "FR":     # Natural Earth's France includes French Guiana; colour Europe only
            geom = geom.intersection(box(-6, 41, 10, 52))
        if iso in counts.index:
            ax.add_geometries([geom], pc, facecolor=SEQ(norm(counts[iso])),
                              edgecolor=SURFACE, linewidth=0.3, zorder=1)
    ax.add_geometries([native], pc, facecolor=NATIVE, edgecolor="none", zorder=1)
    ax.add_geometries([native], pc, facecolor="none", edgecolor=INK2, hatch="////",
                      linewidth=0, zorder=2)
    ax.scatter(gbif.lon, gbif.lat, s=1.2, c=INK, alpha=0.35, linewidths=0, transform=pc, zorder=3)
    for name, lon, lat, _ in BREEDING:
        ax.scatter(lon, lat, s=90, marker="*", c=S2, edgecolors=INK, linewidths=0.6,
                   transform=pc, zorder=6)
    cax = fig.add_axes([0.885, 0.22, 0.016, 0.5])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ), cax=cax)
    cb.set_label("GBIF records per country (outside US/Canada)", color=INK2)
    cb.outline.set_visible(False)
    ax.legend(handles=[
        Patch(facecolor=NATIVE, edgecolor=INK2, hatch="////",
              label="native range: wild regional varieties + garden cultivars"),
        Line2D([], [], marker="o", ls="", markersize=3, color=INK, alpha=0.6,
               label=f"GBIF record outside US/Canada (n = {len(gbif):,})"),
        Line2D([], [], marker="*", ls="", markersize=11, markerfacecolor=S2,
               markeredgecolor=INK, label="where cultivar lines were bred / introduced")],
        loc="lower left", fontsize=8.5, frameon=True, facecolor=SURFACE, edgecolor="#d6d5d0")
    titles(fig, "Where to expect R. hirta and its garden forms, worldwide",
           "Outside North America every population is introduced, so expect Gloriosa-type "
           "cultivars and their escapes, not wild regional varieties")
    figs.append(("01_world", fig,
                 "Countries shaded by GBIF records (log scale); dots are individual records. "
                 "Europe (Germany, Benelux, Sweden, Russia, Switzerland, UK, France, Austria) and "
                 "East Asia (South Korea, Japan, China) hold most introduced records."))

    # ── 2. North America ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(13.5, 6.9))
    ax = fig.add_axes([0.01, 0.07, 0.66, 0.82],
                      projection=ccrs.LambertConformal(-96, standard_parallels=(33, 45)))
    base(ax, (-126, -62, 22, 54))
    ax.add_geometries([native], pc, facecolor=NATIVE, edgecolor="none", zorder=1)
    for geom, col in ((hirta, S1), (angust, S2), (flor, S3)):
        ax.add_geometries([geom], pc, facecolor=col, alpha=0.22, edgecolor="none", zorder=2)
        ax.add_geometries([geom], pc, facecolor="none", edgecolor=col, linewidth=1.6, zorder=3)
    ax.add_geometries([hot], pc, facecolor="none", edgecolor=INK, hatch="xxxx", linewidth=0.8,
                      zorder=5)
    v = var_pts
    for name, col, size in (("pulcherrima", INK2, 3), ("hirta", S1, 9), ("angustifolia", S2, 9),
                            ("floridana", S3, 9)):
        d = v[v.variety == name]
        ax.scatter(d.longitude, d.latitude, s=size, c=col, edgecolors=SURFACE, linewidths=0.4,
                   transform=pc, zorder=6, alpha=0.9 if name != "pulcherrima" else 0.55)
    ax.scatter(red_rayed.longitude, red_rayed.latitude, s=26, marker="X", c=INK,
               edgecolors=SURFACE, linewidths=0.6, transform=pc, zorder=7)
    for name, lon, lat, _ in BREEDING[:2] + BREEDING[4:]:
        ax.scatter(lon, lat, s=120, marker="*", c=S2, edgecolors=INK, linewidths=0.6,
                   transform=pc, zorder=8)
    nv = v.variety.value_counts()
    handles = [
        Patch(facecolor=NATIVE, label="native range (var. pulcherrima nearly throughout)"),
        Patch(facecolor=S1, alpha=0.4, edgecolor=S1, label="var. hirta (Appalachians–Atlantic)"),
        Patch(facecolor=S2, alpha=0.4, edgecolor=S2,
              label="var. angustifolia (Gulf coastal plain, SC–TX)"),
        Patch(facecolor=S3, alpha=0.4, edgecolor=S3, label="var. floridana (central & S Florida)"),
        Line2D([], [], marker="o", ls="", color=INK2, markersize=4,
               label=f"iNat photo IDed var. pulcherrima (n = {nv.get('pulcherrima', 0):,})"),
        Line2D([], [], marker="o", ls="", color=S1, markersize=5,
               label=f"iNat photo IDed var. hirta (n = {nv.get('hirta', 0):,})"),
        Line2D([], [], marker="o", ls="", color=S2, markersize=5,
               label=f"iNat photo IDed var. angustifolia (n = {nv.get('angustifolia', 0):,})"),
        Line2D([], [], marker="o", ls="", color=S3, markersize=5,
               label=f"iNat photo IDed var. floridana (n = {nv.get('floridana', 0):,})"),
        Patch(facecolor="none", edgecolor=INK, hatch="xxxx",
              label="wild red-based bullseye common (≥ 20% of photos, this study)"),
        Line2D([], [], marker="X", ls="", color=INK, markersize=7,
               label=f"red-rayed flower, RBS ≥ 50%: likely cultivar (n = {len(red_rayed)})"),
        Line2D([], [], marker="*", ls="", markersize=12, markerfacecolor=S2, markeredgecolor=INK,
               label="cultivar breeding / introduction")]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.68, 0.48), fontsize=8.5,
               frameon=False, title="North America", title_fontsize=10)
    titles(fig, "Regional varieties and garden forms in the native range",
           "Variety ranges are approximate (Perdue 1957 / Flora of North America); points are "
           "records")
    figs.append(("02_north_america", fig,
                 "Expect var. hirta in the Appalachian–Atlantic region, var. angustifolia on the "
                 "Gulf coastal plain, var. floridana in peninsular Florida, and var. pulcherrima "
                 f"nearly everywhere. Hatched: {n_hot} half-degree cells where wild red-based "
                 "bullseyes are common. ✕: red-rayed flowers, mostly garden cultivars."))

    # ── 3. Europe and East Asia ────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 6.6))
    regions = [("Europe", (-11, 45, 35, 66),
                ccrs.LambertConformal(15, 52, standard_parallels=(40, 60))),
               ("East Asia", (100, 146, 22, 50),
                ccrs.LambertConformal(125, 38, standard_parallels=(30, 45)))]
    for k, (name, ext, proj) in enumerate(regions):
        ax = fig.add_axes([0.01 + k * 0.5, 0.16, 0.47, 0.7], projection=proj)
        base(ax, ext, states=False)
        g = gbif[gbif.lon.between(ext[0], ext[1]) & gbif.lat.between(ext[2], ext[3])]
        xyz = proj.transform_points(pc, g.lon.values, g.lat.values)
        hb = ax.hexbin(xyz[:, 0], xyz[:, 1], gridsize=28, mincnt=1, cmap=SEQ,
                       norm=LogNorm(vmin=1, vmax=200), linewidths=0.3, edgecolors=SURFACE,
                       extent=ax.get_extent(), zorder=2)
        offsets = {"Thompson & Morgan, UK": (1.0, 2.2, "left"),
                   "Benary, Germany": (1.0, -1.6, "left")}
        for bname, lon, lat, _ in BREEDING:
            if ext[0] <= lon <= ext[1] and ext[2] <= lat <= ext[3]:
                ax.scatter(lon, lat, s=120, marker="*", c=S2, edgecolors=INK, linewidths=0.6,
                           transform=pc, zorder=6)
                dx, dy, ha = offsets.get(bname, (0.6, 0.4, "left"))
                ax.text(lon + dx, lat + dy, bname, transform=pc, fontsize=9, zorder=7,
                        color=INK, ha=ha, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none",
                                  alpha=0.85))
        ax.set_title(f"{name}: {len(g):,} GBIF records", loc="left", fontsize=11,
                     fontweight="bold")
    cax = fig.add_axes([0.25, 0.085, 0.5, 0.018])
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal")
    cb.set_label("records per hexagon", color=INK2)
    cb.outline.set_visible(False)
    titles(fig, "Introduced R. hirta: Europe and East Asia",
           "Garden escapes and seed-mix plantings; expect cultivar traits (doubled or bicolored "
           "rays, large heads)")
    figs.append(("03_europe_east_asia", fig,
                 "GBIF records outside the native range. In Europe it is listed as naturalised "
                 "in many countries (Euro+Med); in East Asia it is recorded from Korea, Japan, "
                 "China and Taiwan (POWO)."))

    # ── save ────────────────────────────────────────────────────────────────
    with PdfPages(out / "rhirta_variants_map.pdf") as pdf:
        for name, f, caption in figs:
            f.savefig(out / f"{name}.png", dpi=170)
            f.text(0.02, 0.012, caption, fontsize=8.5, color=INK2, wrap=True)
            pdf.savefig(f)
            plt.close(f)
        page = plt.figure(figsize=(12, 7.6))
        page.text(0.05, 0.92, "What to expect where, and why", fontsize=17, fontweight="bold")
        rows = [
            ("Wild regional varieties", "Native range only (US, southern Canada).",
             "var. hirta: Appalachians to Atlantic coast; var. angustifolia: Gulf coastal plain "
             "SC–TX; var. floridana: central/southern Florida; var. pulcherrima: nearly "
             "throughout, weedy, often purplish near the ray base."),
            ("Wild red-based bullseyes", "Texas Hill Country / Edwards Plateau above all.",
             "This study: up to 30–40% of photos per area. A wild Texas accession (USDA GRIN "
             "TX-06) is described with a red-brown basal spot on the rays."),
            ("Garden cultivars", "Gardens and roadsides wherever temperate gardening is common.",
             "Gloriosa daisies are colchicine-doubled tetraploids (Blakeslee, Smith College; "
             "Burpee 1957). Modern lines include 'Indian Summer', 'Toto', 'Prairie Sun' and "
             "'Denver Daisy' (Benary), and 'Cherokee Sunset' (Thompson & Morgan). In this "
             "study the red-rayed flowers cluster in the Northeast and Great Lakes."),
            ("Introduced populations", "Europe, East Asia, scattered elsewhere (53 countries).",
             "GBIF: ~9,500 records outside the US/Canada, mostly Germany, Belgium, Sweden, the "
             "Netherlands, Russia, Switzerland, the UK, South Korea and Japan. Expect garden "
             "forms and their escapes, not native varieties."),
            ("Hybrids", "Garden centres and plantings only.",
             "R. hirta rarely crosses with other Rudbeckia: 1 confirmed hybrid in 844 seedlings "
             "(Palmer et al. 2009). The sold 'hybrids' are Echinacea × R. hirta (Echibeckia "
             "'Summerina', patented 'ET RDB' lines), USDA zone 6 and warmer."),
        ]
        import textwrap
        y = 0.86
        for head, where, why in rows:
            page.text(0.05, y, head, fontsize=12, fontweight="bold", va="top")
            page.text(0.27, y, where, fontsize=10.5, va="top")
            lines = textwrap.wrap(why, 125)
            for k, line in enumerate(lines):
                page.text(0.27, y - 0.034 - k * 0.027, line, fontsize=9.5, color=INK2, va="top")
            y -= 0.05 + 0.027 * len(lines) + 0.012
        page.text(0.05, y + 0.005, "Sources", fontsize=12, fontweight="bold", va="top")
        src_top = y - 0.04
        sources = [
            "Flora of North America vol. 21, Rudbeckia hirta (varieties after Perdue 1957), via "
            "EOL; Compositae database; Wikipedia 'Rudbeckia hirta'.",
            "Kew POWO and Euro+Med PlantBase (introduced/naturalised areas); GBIF occurrence "
            "API, taxon 3114225 (records outside the US/Canada, fetched 2026-10-30).",
            "Palmer et al. 2009, HortScience (crossability in Rudbeckia); USPTO plant patents "
            "'ET RDB' (Echinacea × R. hirta); USDA GRIN accession TX-06.",
            "Burpee / Blakeslee history (Plant Care Today; Wikipedia); Benary, Thompson & Morgan "
            "and AAS cultivar notes (UW Extension, Mississippi State Extension).",
            "This study: 33,615 iNaturalist photos; relative bullseye size from the GPU charisma "
            "run (root/maps/rbs_by_photo.csv).",
        ]
        for k, s in enumerate(sources):
            page.text(0.05, src_top - k * 0.03, "• " + s, fontsize=9, color=INK2, va="top")
        pdf.savefig(page)
        page.savefig(out / "04_what_to_expect.png", dpi=170)
        plt.close(page)

    # ── interactive ────────────────────────────────────────────────────────
    import plotly.graph_objects as go
    figi = go.Figure()
    figi.add_trace(go.Scattergeo(lon=gbif.lon, lat=gbif.lat, mode="markers",
                                 name=f"GBIF record outside US/Canada ({len(gbif):,})",
                                 marker=dict(size=3, color=INK, opacity=0.45),
                                 text=gbif.country + " · " + gbif.year.fillna(0).astype(int).astype(str)
                                 .replace("0", "year ?") + " · " + gbif.basis.fillna(""),
                                 hovertemplate="%{text}<extra></extra>"))
    for name, col in (("pulcherrima", INK2), ("hirta", S1), ("angustifolia", S2),
                      ("floridana", S3)):
        d = var_pts[var_pts.variety == name]
        figi.add_trace(go.Scattergeo(lon=d.longitude, lat=d.latitude, mode="markers",
                                     name=f"iNat var. {name} ({len(d):,})",
                                     marker=dict(size=5, color=col, line=dict(width=0.3,
                                                                              color=SURFACE)),
                                     hovertemplate=f"var. {name}<extra></extra>"))
    figi.add_trace(go.Scattergeo(lon=red_rayed.longitude, lat=red_rayed.latitude, mode="markers",
                                 name=f"red-rayed, RBS ≥ 50%: likely cultivar ({len(red_rayed)})",
                                 marker=dict(size=8, symbol="x", color=INK),
                                 customdata=red_rayed[["photo_id", "rbs"]],
                                 hovertemplate="photo %{customdata[0]}<br>RBS %{customdata[1]:.0%}"
                                               "<extra></extra>"))
    figi.add_trace(go.Scattergeo(lon=[b[1] for b in BREEDING], lat=[b[2] for b in BREEDING],
                                 mode="markers+text", name="cultivar breeding / introduction",
                                 text=[b[0] for b in BREEDING], textposition="top right",
                                 customdata=[b[3] for b in BREEDING],
                                 marker=dict(size=14, symbol="star", color=S2,
                                             line=dict(width=1, color=INK)),
                                 hovertemplate="%{text}<br>%{customdata}<extra></extra>"))
    figi.update_geos(projection_type="natural earth", showland=True, landcolor=LAND,
                     showocean=True, oceancolor=WATER, showcountries=True, countrycolor=LINES,
                     showsubunits=True, subunitcolor="#cfcdc7", coastlinecolor=INK2)
    figi.update_layout(title="Where to expect R. hirta variants, cultivars and hybrids "
                             "(click legend items to toggle layers)",
                       paper_bgcolor=SURFACE, height=820,
                       legend=dict(orientation="h", y=-0.03))
    figi.write_html(out / "rhirta_variants_interactive.html", include_plotlyjs=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
