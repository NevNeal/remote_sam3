#!/usr/bin/env python3
"""
Relative bullseye size (RBS) of Rudbeckia hirta against environment and
urbanisation, contiguous US only.

    python charisma_gpu/env_urban_maps.py --root C:/Users/nevyn/Documents/rhirta_bullseye

Covariates (root/environment_data/, sampled at each photo's coordinates)
  WorldClim 2.1, 5 arc-min, 1970-2000 (Fick & Hijmans 2017): bio1, bio5, bio12,
      bio15; mean solar radiation June-August; elevation
  glUV, 15 arc-min, 2004-2013 (Beckmann et al. 2014): annual mean UV-B (UVB1),
      UV-B of the highest month (UVB3)
  GHS-SMOD R2023A, epoch 2020, 1 km (Schiavina et al. 2023): degree of
      urbanisation, level 2, grouped into city / suburban-town / rural

Reads root/maps/rbs_by_photo.csv. Writes root/maps/environment/:
  rhirta_bullseye_environment.pdf, rhirta_bullseye_urban_rural.pdf, PNGs,
  rbs_us_covariates.csv (every photo with its covariates), model tables.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402
import rasterio                                            # noqa: E402
from rasterio.warp import transform as warp_transform      # noqa: E402
import cartopy.crs as ccrs                                 # noqa: E402
import cartopy.feature as cfeature                         # noqa: E402
import cartopy.io.shapereader as shpreader                 # noqa: E402
import shapely                                             # noqa: E402
import statsmodels.api as sm                               # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages       # noqa: E402
from matplotlib.colors import LinearSegmentedColormap      # noqa: E402

BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE)
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, NONE = "#fcfcfb", "#0b0b0b", "#52514e", "#c9c8c3"
LAND, WATER, LINES = "#f3f2ee", "#ffffff", "#a8a7a2"
US_EXTENT = (-124.8, -66.8, 24.3, 49.5)
PROJ = ccrs.AlbersEqualArea(central_longitude=-96, standard_parallels=(29.5, 45.5))
PRESENT = 0.05

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "#d6d5d0", "axes.titleweight": "bold", "axes.titlesize": 10.5,
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
})
PCT = plt.FuncFormatter(lambda v, _: f"{v:.0%}")

# name, label, unit, path relative to environment_data, scale
COVARIATES = [
    ("temp", "Annual mean temperature", "°C", "wc2.1_5m_bio/wc2.1_5m_bio_1.tif", 1),
    ("tmax", "Max temperature, warmest month", "°C", "wc2.1_5m_bio/wc2.1_5m_bio_5.tif", 1),
    ("precip", "Annual precipitation", "mm", "wc2.1_5m_bio/wc2.1_5m_bio_12.tif", 1),
    ("pseas", "Precipitation seasonality", "CV, %", "wc2.1_5m_bio/wc2.1_5m_bio_15.tif", 1),
    ("srad", "Solar radiation, June–August", "MJ m⁻² day⁻¹", "srad_jja", 0.001),
    ("elev", "Elevation", "m", "wc2.1_5m_elev/wc2.1_5m_elev.tif", 1),
    ("uvb", "UV-B, annual mean", "J m⁻² day⁻¹", "gluv/56459_UVB1_Annual_Mean_UV-B.asc", 1),
    ("uvb_max", "UV-B, highest month", "J m⁻² day⁻¹", "gluv/56461_UVB3_Mean_UV-B_of_Highest_Month.asc", 1),
]
SMOD = {30: ("Urban centre", "City"), 23: ("Dense urban cluster", "Suburban / town"),
        22: ("Semi-dense urban cluster", "Suburban / town"),
        21: ("Suburban or peri-urban", "Suburban / town"),
        13: ("Rural cluster", "Rural"), 12: ("Low density rural", "Rural"),
        11: ("Very low density rural", "Rural"), 10: ("Water", None)}
CLASS_ORDER = ["City", "Suburban / town", "Rural"]
CLASS_COLOURS = {"City": S2, "Suburban / town": S1, "Rural": S3}


# ── data ─────────────────────────────────────────────────────────────────────

def lower48():
    shp = shpreader.natural_earth("50m", "cultural", "admin_1_states_provinces_lakes")
    geoms = [r.geometry for r in shpreader.Reader(shp).records()
             if r.attributes.get("adm0_a3") == "USA"
             and r.attributes["name"] not in ("Alaska", "Hawaii")]
    return shapely.union_all(geoms), {r.attributes["name"]: r.geometry
                                      for r in shpreader.Reader(shp).records()
                                      if r.attributes.get("adm0_a3") == "USA"}


def sample(path, lon, lat, crs_from="EPSG:4326"):
    with rasterio.open(path) as src:
        xs, ys = (lon, lat) if src.crs is None or src.crs.to_string() == crs_from else \
            warp_transform(crs_from, src.crs, list(lon), list(lat))
        vals = np.array([v[0] for v in src.sample(zip(xs, ys))], dtype=float)
        nod = src.nodata
    if nod is not None:
        vals[vals == nod] = np.nan
    vals[vals < -1e30] = np.nan
    return vals


def srad_jja(env, lon, lat):
    months = [env / "wc2.1_5m_srad" / f"wc2.1_5m_srad_{m:02d}.tif" for m in (6, 7, 8)]
    return np.nanmean([sample(m, lon, lat) for m in months], axis=0)


def build(root):
    env = root / "environment_data"
    photo = pd.read_csv(root / "maps" / "rbs_by_photo.csv")
    us, states = lower48()
    inside = shapely.contains_xy(us, photo.longitude.values, photo.latitude.values)
    d = photo[inside].copy().reset_index(drop=True)
    lon, lat = d.longitude.values, d.latitude.values
    for name, _, _, rel, scale in COVARIATES:
        v = srad_jja(env, lon, lat) if rel == "srad_jja" else sample(env / rel, lon, lat)
        d[name] = v * scale
    smod_tif = next((env / "GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V2_0").glob("*.tif"))
    d["smod"] = sample(smod_tif, lon, lat).astype("float")
    d["smod_class"] = d.smod.map(lambda c: SMOD.get(int(c), (None, None))[0] if c == c else None)
    d["settlement"] = d.smod.map(lambda c: SMOD.get(int(c), (None, None))[1] if c == c else None)
    tx = states["Texas"]
    d["texas"] = shapely.contains_xy(tx, lon, lat)
    d["has_bullseye"] = (d.rbs >= PRESENT).astype(float)
    d["red_rayed"] = (d.rbs >= 0.5).astype(float)
    return d, states


# ── helpers ──────────────────────────────────────────────────────────────────

def us_axes(fig, rect):
    ax = fig.add_axes(rect, projection=PROJ)
    ax.set_extent(US_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=WATER, edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=LAND, edgecolor="none", zorder=0)
    ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor=LINES, linewidth=0.3,
                   facecolor="none", zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor=INK2, linewidth=0.4, zorder=4)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=INK2, linewidth=0.4, zorder=4)
    ax.spines["geo"].set_edgecolor("#d6d5d0")
    return ax


def raster_window(path, scale=1.0, bbox=(-126, -65, 23, 51)):
    with rasterio.open(path) as src:
        win = rasterio.windows.from_bounds(bbox[0], bbox[2], bbox[1], bbox[3], src.transform)
        a = src.read(1, window=win, masked=True).astype(float).filled(np.nan)
        b = rasterio.windows.bounds(win, src.transform)
    a[a < -1e30] = np.nan
    return a * scale, (b[0], b[2], b[1], b[3])


def titles(fig, t, s, y=0.955):
    fig.text(0.03, y, t, fontsize=15, fontweight="bold")
    fig.text(0.03, y - 0.035, s, fontsize=9.5, color=INK2)


def binned(x, y, q=10, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    edges = np.unique(np.quantile(x, np.linspace(0, 1, q + 1)))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x <= hi)
        if m.sum() < 15:
            continue
        v = y[m]
        boots = rng.choice(v, size=(n_boot, len(v))).mean(1)
        out.append((np.median(x[m]), v.mean(), *np.percentile(boots, [2.5, 97.5]), m.sum()))
    return pd.DataFrame(out, columns=["x", "mean", "lo", "hi", "n"])


def prop_ci(v, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    b = rng.choice(v, size=(n_boot, len(v))).mean(1)
    return v.mean(), *np.percentile(b, [2.5, 97.5])


def fit_glm(d, y, xs, family):
    z = d[xs].apply(lambda c: (c - c.mean()) / c.std())
    X = sm.add_constant(z)
    m = sm.GLM(d[y], X, family=family).fit()
    ci = m.conf_int()
    return pd.DataFrame({"coef": m.params, "lo": ci[0], "hi": ci[1], "p": m.pvalues}).drop(
        "const"), m


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root)
    out = root / "maps" / "environment"
    out.mkdir(parents=True, exist_ok=True)
    d, states = build(root)
    names = [c[0] for c in COVARIATES]
    labels = {c[0]: f"{c[1]} ({c[2]})" for c in COVARIATES}
    d.to_csv(out / "rbs_us_covariates.csv", index=False)
    full = d.dropna(subset=names + ["settlement"])
    pres = full[full.has_bullseye == 1]
    print(f"US photos {len(d)}, with all covariates {len(full)}, with bullseye {len(pres)}")
    print(full.settlement.value_counts().to_dict())
    env = root / "environment_data"
    efigs, ufigs = [], []

    # E1 covariate maps with bullseye photos
    fig = plt.figure(figsize=(12, 6.6))
    titles(fig, "The environmental variables, with bullseye flowers on top",
           f"Contiguous US. Black dots: the {len(pres):,} photos with relative bullseye size ≥ "
           f"{PRESENT:.0%}")
    for k, (name, lab, unit, rel, scale) in enumerate(COVARIATES):
        r, c = divmod(k, 4)
        ax = us_axes(fig, [0.01 + c * 0.247, 0.53 - r * 0.43, 0.235, 0.31])
        if rel == "srad_jja":
            arrs = [raster_window(env / "wc2.1_5m_srad" / f"wc2.1_5m_srad_{m:02d}.tif", scale)
                    for m in (6, 7, 8)]
            a, ext = np.nanmean([x[0] for x in arrs], axis=0), arrs[0][1]
        else:
            a, ext = raster_window(env / rel, scale)
        vmin, vmax = np.nanpercentile(full[name], [1, 99])
        im = ax.imshow(a, extent=ext, origin="upper", transform=ccrs.PlateCarree(), cmap=SEQ,
                       vmin=vmin, vmax=vmax, zorder=1, interpolation="nearest")
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=WATER, edgecolor="none",
                       zorder=2)
        ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor=WATER, edgecolor="none",
                       zorder=2)
        ax.scatter(pres.longitude, pres.latitude, s=1.2, c=INK, linewidths=0, alpha=0.6,
                   transform=ccrs.PlateCarree(), zorder=5)
        ax.set_title(lab, loc="left", fontsize=9.5)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03)
        cb.set_label(unit, color=INK2, fontsize=8)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7)
    efigs.append(("E1_covariate_maps", fig, "WorldClim 2.1 (5 arc-min, 1970–2000), glUV "
                  "(15 arc-min, 2004–2013). Colour scales span the 1st–99th percentile of values "
                  "at the photo locations."))

    # E2 / E3 relationships
    for tag, yname, data, ylabel, title in (
            ("E2_prevalence", "has_bullseye", full, "share of photos with a bullseye",
             "How often a bullseye is present, along each gradient"),
            ("E3_size", "rbs", pres, "mean relative bullseye size",
             "How big the bullseye is (bullseye flowers only), along each gradient")):
        fig, axes = plt.subplots(2, 4, figsize=(12, 6.8), sharey=True)
        fig.subplots_adjust(top=0.84, hspace=0.45, wspace=0.12, left=0.07, right=0.98, bottom=0.1)
        titles(fig, title, "Deciles of each variable; dots are decile means, bands bootstrap 95% "
               "intervals. Blue: whole US. Orange: excluding Texas.")
        for ax, name in zip(axes.ravel(), names):
            for sub, col in ((data, S1), (data[~data.texas], S2)):
                b = binned(sub[name].values, sub[yname].values)
                ax.fill_between(b.x, b.lo, b.hi, color=col, alpha=0.18, linewidth=0)
                ax.plot(b.x, b["mean"], color=col, linewidth=1.8)
                ax.scatter(b.x, b["mean"], s=10, color=col, zorder=3)
            ax.set_title(dict((c[0], c[1]) for c in COVARIATES)[name], loc="left", fontsize=9)
            ax.set_xlabel(dict((c[0], c[2]) for c in COVARIATES)[name])
            ax.yaxis.set_major_formatter(PCT)
            ax.grid(axis="y", color="#ecebe7", linewidth=0.8)
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)
        efigs.append((tag, fig, "Binned means, no model. Texas holds a large share of all "
                      "photos and bullseyes, so the orange lines show whether a pattern survives "
                      "without it."))

    # E4 correlation matrix
    cm = full[names].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8.5, 7))
    fig.subplots_adjust(left=0.32, bottom=0.3, top=0.88, right=0.95)
    im = ax.imshow(cm.values, cmap=DIV, vmin=-1, vmax=1)
    short = [dict((c[0], c[1]) for c in COVARIATES)[n] for n in names]
    ax.set_xticks(range(len(names)), short, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), short)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color=INK if abs(cm.values[i, j]) < 0.7 else "white")
    fig.colorbar(im, ax=ax, fraction=0.04).set_label("Spearman correlation", color=INK2)
    fig.text(0.03, 0.95, "The variables move together", fontsize=15, fontweight="bold")
    fig.text(0.03, 0.915, "Correlations at the photo locations; the model below uses a "
             "reduced set to limit collinearity", fontsize=9.5, color=INK2)
    efigs.append(("E4_correlations", fig, "Temperature, solar radiation and UV-B are strongly "
                  "correlated across the US, so their separate effects cannot be cleanly "
                  "distinguished with these data."))

    # E5 models
    xs = ["temp", "precip", "pseas", "srad", "elev", "uvb_max"]
    rows = []
    for subset_name, sub in (("Whole US", full), ("Excluding Texas", full[~full.texas])):
        c1, m1 = fit_glm(sub, "has_bullseye", xs, sm.families.Binomial())
        c1["model"], c1["subset"], c1["n"] = "presence", subset_name, len(sub)
        sp = sub[sub.has_bullseye == 1].copy()
        sp["logit_rbs"] = np.log(sp.rbs / (1 - sp.rbs).clip(lower=1e-4))
        c2, m2 = fit_glm(sp, "logit_rbs", xs, sm.families.Gaussian())
        c2["model"], c2["subset"], c2["n"] = "size", subset_name, len(sp)
        rows += [c1, c2]
    coefs = pd.concat(rows).reset_index().rename(columns={"index": "variable"})
    coefs.to_csv(out / "model_coefficients.csv", index=False)
    vif = []
    z = full[xs].apply(lambda c: (c - c.mean()) / c.std())
    for i, x in enumerate(xs):
        others = sm.add_constant(z.drop(columns=x))
        r2 = sm.OLS(z[x], others).fit().rsquared
        vif.append((x, 1 / (1 - r2)))
    pd.DataFrame(vif, columns=["variable", "VIF"]).to_csv(out / "vif.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    fig.subplots_adjust(top=0.74, left=0.22, wspace=0.08, bottom=0.14)
    titles(fig, "All variables together: what still matters?",
           "Standardised coefficients (per 1 SD) with 95% CIs. Left: logistic regression of "
           "bullseye presence. Right: linear regression of log-odds RBS, bullseye flowers only")
    ylab = [dict((c[0], c[1]) for c in COVARIATES)[x] for x in xs]
    for ax, model, xl in ((axes[0], "presence", "log-odds of having a bullseye, per SD"),
                          (axes[1], "size", "log-odds of RBS, per SD")):
        for k, (subset_name, col) in enumerate((("Whole US", S1), ("Excluding Texas", S2))):
            c = coefs[(coefs.model == model) & (coefs.subset == subset_name)].set_index(
                "variable").loc[xs]
            yy = np.arange(len(xs)) + (k - 0.5) * 0.28
            ax.errorbar(c.coef, yy, xerr=[c.coef - c.lo, c.hi - c.coef], fmt="o", color=col,
                        markersize=5, capsize=0, linewidth=1.6,
                        label=f"{subset_name} (n = {int(c.n.iloc[0]):,})")
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_yticks(range(len(xs)), ylab)
        ax.invert_yaxis()
        ax.set_xlabel(xl)
        ax.grid(axis="x", color="#ecebe7", linewidth=0.8)
    for ax in axes:
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.06), fontsize=8, frameon=False)
    axes[0].set_title("Bullseye present?", loc="left")
    axes[1].set_title("How big, when present?", loc="left")
    vif_txt = ", ".join(f"{a} {b:.1f}" for a, b in vif)
    efigs.append(("E5_models", fig, f"Reduced variable set (VIF: {vif_txt}). Observations are "
                  "spatially autocorrelated, so the intervals are too narrow; treat this as "
                  "description, not inference."))

    # E6 residual map
    _, mfull = fit_glm(full, "has_bullseye", xs, sm.families.Binomial())
    z = full[xs].apply(lambda c: (c - c.mean()) / c.std())
    full = full.assign(resid=full.has_bullseye - mfull.predict(sm.add_constant(z)))
    fig = plt.figure(figsize=(11, 7))
    titles(fig, "Where the climate model misses",
           "Observed minus predicted bullseye prevalence per hexagon (≥ 15 photos); red = more "
           "bullseyes than climate predicts")
    ax = us_axes(fig, [0.02, 0.05, 0.84, 0.82])
    xyz = PROJ.transform_points(ccrs.PlateCarree(), full.longitude.values, full.latitude.values)
    hb = ax.hexbin(xyz[:, 0], xyz[:, 1], C=full.resid, reduce_C_function=np.mean, gridsize=48,
                   mincnt=15, cmap=DIV, vmin=-0.2, vmax=0.2, extent=ax.get_extent(),
                   linewidths=0.3, edgecolors=SURFACE, zorder=2)
    cax = fig.add_axes([0.875, 0.2, 0.018, 0.55])
    cb = fig.colorbar(hb, cax=cax)
    cb.set_label("observed − predicted prevalence", color=INK2)
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_major_formatter(PCT)
    efigs.append(("E6_residuals", fig, "Residuals from the whole-US presence model. Clusters "
                  "of red or blue mark places where something other than these climate "
                  "variables drives the bullseye."))

    # U1 maps by settlement class
    fig = plt.figure(figsize=(13, 4.4))
    titles(fig, "Relative bullseye size in cities, suburbs and the countryside",
           f"Each photo classed by GHS-SMOD 2020 (1 km). Grey: no bullseye; blue: relative "
           f"bullseye size")
    for k, cls in enumerate(CLASS_ORDER):
        ax = us_axes(fig, [0.01 + k * 0.33, 0.17, 0.315, 0.6])
        sub = full[full.settlement == cls]
        none = sub[sub.rbs < PRESENT]
        some = sub[sub.rbs >= PRESENT].sort_values("rbs")
        ax.scatter(none.longitude, none.latitude, s=1.2, c=NONE, linewidths=0,
                   transform=ccrs.PlateCarree(), zorder=2)
        sc = ax.scatter(some.longitude, some.latitude, s=4, c=some.rbs, cmap=SEQ, vmin=PRESENT,
                        vmax=0.6, linewidths=0, transform=ccrs.PlateCarree(), zorder=3)
        prev = sub.has_bullseye.mean()
        ax.set_title(f"{cls}\n{len(sub):,} photos, {prev:.1%} with a bullseye", loc="left",
                     fontsize=10)
    cax = fig.add_axes([0.35, 0.09, 0.3, 0.03])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", extend="max")
    cb.set_label("relative bullseye size", color=INK2)
    cb.outline.set_visible(False)
    cb.ax.xaxis.set_major_formatter(PCT)
    ufigs.append(("U1_maps", fig, "City = urban centre; suburban / town = dense, semi-dense and "
                  "suburban/peri-urban clusters; rural = rural clusters and low / very low "
                  "density rural (GHS-SMOD level 2)."))

    # U2 summary by class, with Texas split
    rows = []
    for region, sub in (("Whole US", full), ("Texas", full[full.texas]),
                        ("Rest of US", full[~full.texas])):
        for cls in CLASS_ORDER:
            s = sub[sub.settlement == cls]
            pr = prop_ci(s.has_bullseye)
            rr = prop_ci(s.red_rayed)
            sz = prop_ci(s[s.has_bullseye == 1].rbs)
            rows.append(dict(region=region, settlement=cls, n=len(s),
                             prevalence=pr[0], prev_lo=pr[1], prev_hi=pr[2],
                             red_rayed=rr[0], rr_lo=rr[1], rr_hi=rr[2],
                             size=sz[0], size_lo=sz[1], size_hi=sz[2]))
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "urban_rural_summary.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2))
    fig.subplots_adjust(top=0.78, wspace=0.28, bottom=0.17, left=0.06, right=0.98)
    titles(fig, "City, suburb or countryside: does it change the bullseye?",
           "Bootstrap 95% intervals. Texas is split out because it dominates the bullseye "
           "photos")
    regions = ["Whole US", "Texas", "Rest of US"]
    for ax, (col, lo, hi, title, xl) in zip(axes, (
            ("prevalence", "prev_lo", "prev_hi", "Bullseye present (RBS ≥ 5%)", "share of photos"),
            ("size", "size_lo", "size_hi", "Size when present", "mean relative bullseye size"),
            ("red_rayed", "rr_lo", "rr_hi", "Red-rayed (RBS ≥ 50%): likely cultivars",
             "share of photos"))):
        for k, cls in enumerate(CLASS_ORDER):
            s = summ[summ.settlement == cls].set_index("region").loc[regions]
            yy = np.arange(len(regions)) + (k - 1) * 0.25
            ax.errorbar(s[col], yy, xerr=[s[col] - s[lo], s[hi] - s[col]], fmt="o",
                        color=CLASS_COLOURS[cls], markersize=6, capsize=0, linewidth=1.8,
                        label=cls)
        ax.set_yticks(range(len(regions)), regions)
        ax.invert_yaxis()
        ax.set_title(title, loc="left")
        ax.set_xlabel(xl)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}")
                                     if col == "red_rayed" else PCT)
        ax.grid(axis="x", color="#ecebe7", linewidth=0.8)
        ax.set_xlim(left=0)
    axes[0].legend(loc="lower right", fontsize=8.5, frameon=False)
    ufigs.append(("U2_by_class", fig, "Left: how often a bullseye occurs. Middle: its size "
                  "where present. Right: the red-rayed flowers that are mostly garden "
                  "cultivars."))

    # U3 detailed GHSL classes
    det = full.groupby("smod_class").agg(n=("rbs", "size"), prev=("has_bullseye", "mean"),
                                         rr=("red_rayed", "mean")).reset_index()
    order = [SMOD[k][0] for k in (30, 23, 22, 21, 13, 12, 11) if SMOD[k][0] in set(det.smod_class)]
    det = det.set_index("smod_class").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.subplots_adjust(top=0.8, left=0.24, right=0.97, bottom=0.14)
    titles(fig, "All seven settlement classes", "Share of photos with a bullseye (blue) and "
           "red-rayed flowers (orange); n photos at right")
    yy = np.arange(len(det))
    ax.barh(yy - 0.18, det.prev, height=0.34, color=S1, label="bullseye present")
    ax.barh(yy + 0.18, det.rr, height=0.34, color=S2, label="red-rayed (RBS ≥ 50%)")
    for k, r in det.iterrows():
        ax.text(max(r.prev, r.rr) + 0.004, k, f"n = {int(r.n):,}", va="center", fontsize=8.5,
                color=INK2)
    ax.set_yticks(yy, det.smod_class)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PCT)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", color="#ecebe7", linewidth=0.8)
    ufigs.append(("U3_detail", fig, "GHS-SMOD level-2 classes from most to least urban."))

    for pdf_name, figs, cover_title, cover_lines in (
            ("rhirta_bullseye_environment.pdf", efigs,
             "Relative bullseye size and the environment",
             ["Contiguous US. Relative bullseye size (RBS) = red on the petals ÷ (red on the "
              "petals + yellow ray).",
              f"{len(full):,} photos with every covariate; {len(pres):,} ({len(pres) / len(full):.1%}) "
              f"with a bullseye (RBS ≥ {PRESENT:.0%}).",
              "Covariates: WorldClim 2.1 (temperature, precipitation, seasonality, summer solar "
              "radiation, elevation),",
              "glUV (UV-B, annual and highest month). Sources and caveats: "
              "rhirta_environment_sources.pdf.",
              "",
              "Read with care: these are correlations across space. Texas holds many of the "
              "bullseyes,",
              "temperature, sunlight and UV-B rise together toward the south, and the photos "
              "are spatially",
              "clustered, so model intervals are too narrow."]),
            ("rhirta_bullseye_urban_rural.pdf", ufigs,
             "Relative bullseye size: city, suburb, countryside",
             ["Contiguous US. Each photo classed by the Global Human Settlement Layer degree of "
              "urbanisation",
              "(GHS-SMOD R2023A, epoch 2020, 1 km): city, suburban / town, rural.",
              "Counts: " + ", ".join(f"{c} {int((full.settlement == c).sum()):,}"
                                     for c in CLASS_ORDER) + ".",
              "",
              "Why it matters: if red-rayed flowers are garden cultivars, they should be "
              "commoner in towns.",
              "Caveat: iNaturalist can obscure locations (up to ~0.2°), which blurs a 1 km "
              "classification."])):
        with PdfPages(out / pdf_name) as pdf:
            cover = plt.figure(figsize=(11, 7))
            cover.text(0.06, 0.8, cover_title, fontsize=21, fontweight="bold")
            for k, line in enumerate(cover_lines):
                cover.text(0.06, 0.68 - k * 0.05, line, fontsize=11,
                           color=INK if k < 4 else INK2)
            pdf.savefig(cover)
            plt.close(cover)
            for name, fig, caption in figs:
                fig.savefig(out / f"{name}.png", dpi=160)
                fig.text(0.03, 0.01, caption, fontsize=8.3, color=INK2)
                pdf.savefig(fig)
                plt.close(fig)
    print(summ.round(4).to_string())
    print(coefs.round(3).to_string())
    print("wrote", out)


if __name__ == "__main__":
    main()
