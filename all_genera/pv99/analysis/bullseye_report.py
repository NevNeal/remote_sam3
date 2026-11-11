#!/usr/bin/env python3
"""
From per-photo bullseye answers to the genera and families most likely to have
a bullseye pattern.

    python all_genera/pv99/analysis/bullseye_report.py

Input: export/pv99/analysis/bullseye/photos_shard*.csv (bullseye_classify.py).

Per photo: "can't tell" when P(unclear) >= ABSTAIN_CUT; otherwise the call is
the most probable of bullseye / partial / none, and the photo is "told".

Per genus and per family: k = told photos called bullseye, n = told photos.
pv99 has at most 50 photos per genus, and many genera far fewer, so a raw share
k/n over-ranks tiny genera that went 3 for 3. The ranking uses an empirical-
Bayes beta-binomial instead: a Beta(a, b) prior is fitted by maximum likelihood
to every unit with n >= MIN_FIT, and each unit is scored by its posterior mean
(k + a) / (n + a + b), with a 90% credible interval. A unit with few photos is
pulled toward the overall rate; one with many keeps close to its own share.
`mean_p` (mean P(bullseye | told) without thresholding) is reported alongside.

Writes into export/pv99/analysis/bullseye/:

  photos.csv            every photo: metadata, option probabilities, call
  genus_bullseye.csv    one row per genus, sorted by posterior mean
  family_bullseye.csv   one row per family, sorted by posterior mean, with its
                        number of genera, how many of them rank as bullseye
                        genera (posterior >= 0.5), and its top genera
  controls.csv          where genera with a textbook visible bullseye landed
  top_genera.png, top_families.png
  sheets/top_photos.jpg            the 100 photos with the highest P(bullseye)
  sheets/random_bullseye.jpg       100 random photos called bullseye (precision check)
  sheets/random_not.jpg            100 random told photos called partial/none
  sheets/genus_<rank>_<Genus>.jpg  for the top 30 genera, their 20 best photos
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from scipy import optimize, special, stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from bullseye_classify import top_segment_crop  # noqa: E402
from top_segments import _font  # noqa: E402

ABSTAIN_CUT = 0.5
MIN_FIT = 10          # units with at least this many told photos fit the prior
MIN_RANK_GENUS = 10   # ... and are eligible for the ranked genus figure/sheets
MIN_RANK_FAMILY = 50
TOP_GENERA = 40
TOP_FAMILIES = 30
SHEET_GENERA = 30
TILE = 200
COLS = 10

# Genera where a bullseye is plainly visible in many species, and a few where
# it is plainly absent; a sanity check on the ranking, not used to fit it.
CONTROLS_YES = ["Rudbeckia", "Gaillardia", "Coreopsis", "Ratibida", "Gazania", "Arctotis",
                "Dimorphotheca", "Papaver", "Hibiscus", "Tithonia", "Helianthus",
                "Echinacea", "Ursinia", "Cosmos", "Gorteria", "Calochortus"]
CONTROLS_NO = ["Lupinus", "Salvia", "Solidago", "Carex", "Quercus", "Trifolium",
               "Penstemon", "Asclepias", "Daucus", "Euphorbia"]


def load(project):
    folder = common.analysis_dir(project, "bullseye")
    parts = sorted(folder.glob("photos_shard*.csv"))
    if not parts:
        raise SystemExit(f"no photos_shard*.csv in {folder}: run bullseye_classify.py first")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df = df.drop_duplicates("photo_id", keep="last")
    df["error"] = df["error"].fillna("")
    meta = common.load_metadata(project, ["photo_id", "image_path", "mask_path"])
    return df.merge(meta, on="photo_id", how="left"), folder, len(parts)


def calls(df):
    ok = df[df["error"] == ""].copy()
    ok["told"] = ok["bullseye__unclear"] < ABSTAIN_CUT
    real = ok[["bullseye__bullseye", "bullseye__partial", "bullseye__none"]].to_numpy()
    ok["call"] = np.array(["bullseye", "partial", "none"])[real.argmax(1)]
    ok.loc[~ok["told"], "call"] = "cant_tell"
    ok["p_given_told"] = ok["bullseye__bullseye"] / (1 - ok["bullseye__unclear"]).clip(lower=1e-6)
    return ok


def fit_beta(k, n):
    """Maximum-likelihood Beta(a, b) for beta-binomial counts."""
    k, n = np.asarray(k, float), np.asarray(n, float)

    def nll(log_ab):
        a, b = np.exp(log_ab)
        return -np.sum(special.betaln(k + a, n - k + b) - special.betaln(a, b))

    m = max(1e-3, min(0.999, k.sum() / n.sum()))
    res = optimize.minimize(nll, np.log([m * 5, (1 - m) * 5]), method="Nelder-Mead")
    return tuple(np.clip(np.exp(res.x), 0.05, 1e4))


def summarise(ok, by, prior=None):
    told = ok[ok["told"]]
    g = ok.groupby(by)
    t = told.groupby(by)
    table = pd.DataFrame({
        "photos": g.size(),
        "told": t.size(),
        "bullseye": t["call"].apply(lambda c: int((c == "bullseye").sum())),
        "partial": t["call"].apply(lambda c: int((c == "partial").sum())),
        "mean_p": t["p_given_told"].mean(),
    }).fillna({"told": 0, "bullseye": 0, "partial": 0})
    table[["told", "bullseye", "partial"]] = table[["told", "bullseye", "partial"]].astype(int)
    table["share"] = table["bullseye"] / table["told"].where(table["told"] > 0)
    fit = table[table["told"] >= MIN_FIT]
    if fit.empty:                                   # tiny test runs only
        fit = table[table["told"] > 0]
    a, b = prior or (fit_beta(fit["bullseye"], fit["told"]) if len(fit) else (1.0, 1.0))
    post_a, post_b = table["bullseye"] + a, table["told"] - table["bullseye"] + b
    table["posterior"] = post_a / (post_a + post_b)
    table["lo90"] = stats.beta.ppf(0.05, post_a, post_b)
    table["hi90"] = stats.beta.ppf(0.95, post_a, post_b)
    table = table.sort_values(["posterior", "told"], ascending=False)
    table.insert(0, "rank", range(1, len(table) + 1))
    return table.reset_index(), (a, b)


def dot_plot(table, label, n_col, title, path):
    t = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.24 * len(t) + 1.4), dpi=150)
    fig.patch.set_facecolor(common.SURFACE)
    common.style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=common.GRID, linewidth=0.8)
    y = np.arange(len(t))
    ax.hlines(y, t["lo90"], t["hi90"], color=common.BLUE_RAMP[2], linewidth=2.2)
    ax.plot(t["posterior"], y, "o", color=common.BLUE, markersize=5, zorder=3)
    ax.plot(t["share"], y, "|", color=common.ORANGE, markersize=9, zorder=3)
    ax.set_yticks(y, [f"{r[label]}  ({r[n_col]} / {r['told']})" for _, r in t.iterrows()],
                  fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of photos with a bullseye", color=common.INK_2, fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, color=common.INK)
    ax.text(0, -0.9 - 0.012 * len(t), "dot: posterior mean, bar: 90% credible interval, "
            "tick: raw share; label: bullseye / told photos", fontsize=7.5, color=common.INK_2,
            transform=ax.get_yaxis_transform())
    fig.tight_layout()
    fig.savefig(path, facecolor=common.SURFACE)
    plt.close(fig)


def _tile(job):
    project, row = job
    try:
        crop, _, _ = top_segment_crop(project, row["image_path"], row["mask_path"], TILE)
    except Exception:
        crop = Image.new("RGB", (TILE, TILE), (96, 96, 102))
    draw = ImageDraw.Draw(crop)
    text = f"{row['genus']} {row['bullseye__bullseye']:.2f}"
    draw.rectangle((0, TILE - 18, TILE, TILE), fill=(20, 20, 22))
    draw.text((4, TILE - 3), text[:28], fill=(240, 240, 240), font=_font(12), anchor="ls")
    return crop


def sheet(project, rows, title, path, pool):
    rows = rows.to_dict("records")
    tiles = list(pool.map(_tile, [(project, r) for r in rows], chunksize=4))
    n_rows = max(1, int(np.ceil(len(tiles) / COLS)))
    head = 40
    out = Image.new("RGB", (COLS * TILE, head + n_rows * TILE), (24, 24, 27))
    ImageDraw.Draw(out).text((10, head // 2), title, fill=(235, 235, 235), font=_font(20),
                             anchor="lm")
    for i, tile in enumerate(tiles):
        out.paste(tile, ((i % COLS) * TILE, head + (i // COLS) * TILE))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path, quality=88)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_args(parser)
    parser.add_argument("--workers", type=int, default=common.n_workers())
    parser.add_argument("--no-sheets", action="store_true")
    args = parser.parse_args()

    project = common.project_dir(args.project)
    df, folder, n_parts = load(project)
    ok = calls(df)
    ok.drop(columns=["image_path", "mask_path"]).to_csv(folder / "photos.csv", index=False)

    genus, g_prior = summarise(ok, ["genus", "family"])
    family, f_prior = summarise(ok, "family")
    bull_genera = genus[(genus["posterior"] >= 0.5)].groupby("family")["genus"]
    family["genera"] = family["family"].map(ok.groupby("family")["genus"].nunique())
    family["bullseye_genera"] = family["family"].map(bull_genera.size()).fillna(0).astype(int)
    ranked = genus[genus["told"] >= MIN_RANK_GENUS]
    top_genera = {fam: ", ".join(f"{r.genus} {r.posterior:.2f}" for r in g.head(5).itertuples())
                  for fam, g in ranked.groupby("family")}
    family["top_genera"] = family["family"].map(top_genera).fillna("")
    genus.to_csv(folder / "genus_bullseye.csv", index=False)
    family.to_csv(folder / "family_bullseye.csv", index=False)

    controls = genus[genus["genus"].isin(CONTROLS_YES + CONTROLS_NO)].copy()
    controls["expected"] = np.where(controls["genus"].isin(CONTROLS_YES), "yes", "no")
    controls = controls.sort_values(["expected", "rank"], ascending=[False, True])
    controls.to_csv(folder / "controls.csv", index=False)

    dot_plot(ranked.head(TOP_GENERA), "genus", "bullseye",
             f"Top {TOP_GENERA} genera by bullseye share (>= {MIN_RANK_GENUS} told photos)",
             folder / "top_genera.png")
    dot_plot(family[family["told"] >= MIN_RANK_FAMILY].head(TOP_FAMILIES), "family", "bullseye",
             f"Top {TOP_FAMILIES} families by bullseye share (>= {MIN_RANK_FAMILY} told photos)",
             folder / "top_families.png")

    counts = ok["call"].value_counts()
    print(f"photos     : {len(df):,} from {n_parts} shard file(s), {int((df['error'] != '').sum()):,} errors")
    print(f"calls      : " + ", ".join(f"{k} {v:,}" for k, v in counts.items()))
    print(f"flip rate  : {ok['bullseye__flip'].mean():.3f}   letter mass: "
          f"{ok['bullseye__mass'].median():.3f} median")
    print(f"prior      : genus Beta({g_prior[0]:.3f}, {g_prior[1]:.3f}), "
          f"family Beta({f_prior[0]:.3f}, {f_prior[1]:.3f})")
    print("\ntop genera:")
    print(ranked.head(25)[["rank", "genus", "family", "told", "bullseye", "share",
                           "posterior", "lo90", "hi90"]].to_string(index=False))
    print("\ntop families:")
    print(family[family["told"] >= MIN_RANK_FAMILY].head(20)[
        ["rank", "family", "told", "bullseye", "posterior", "genera", "bullseye_genera"]
    ].to_string(index=False))
    print("\ncontrols:")
    print(controls[["expected", "genus", "rank", "told", "bullseye", "posterior"]].to_string(index=False))

    if args.no_sheets:
        return
    sheets = folder / "sheets"
    with ProcessPoolExecutor(args.workers) as pool:
        top = ok.sort_values("bullseye__bullseye", ascending=False).head(100)
        sheet(project, top, "100 photos with the highest P(bullseye)", sheets / "top_photos.jpg", pool)
        yes = ok[ok["call"] == "bullseye"]
        sheet(project, yes.sample(min(100, len(yes)), random_state=0),
              "100 random photos called bullseye", sheets / "random_bullseye.jpg", pool)
        no = ok[ok["call"].isin(["partial", "none"])]
        sheet(project, no.sample(min(100, len(no)), random_state=0),
              "100 random photos called partial or none", sheets / "random_not.jpg", pool)
        for r in ranked.head(SHEET_GENERA).itertuples():
            rows = ok[ok["genus"] == r.genus].sort_values("bullseye__bullseye", ascending=False)
            sheet(project, rows.head(20),
                  f"#{r.rank} {r.genus} ({r.family}): {r.bullseye}/{r.told} bullseye, "
                  f"posterior {r.posterior:.2f}",
                  sheets / f"genus_{r.rank:03d}_{common.safe(r.genus)}.jpg", pool)
    print(f"\nwrote      : {folder}")


if __name__ == "__main__":
    main()
