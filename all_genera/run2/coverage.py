#!/usr/bin/env python3
"""
Which genera and families are thin: fewer than 30 phenovision hits, or fewer
than 30 photos with a SAM3 mask in run 2. Two interactive plots and a CSV.

    python all_genera/run2/coverage.py \\
        --genera data/all_genera_run2_index_genera.csv \\
        --metadata export/all_genera_run2/all_genera_segment_metadata.csv \\
        --taxa data/taxa.csv.gz \\
        --inat data/inat_photos.parquet \\
        --export export/all_genera_run2

  phenovision hits   the genus's flower-detected photos in the annotations
                     (flower_detected in the _genera.csv); a family's is the
                     sum over its genera
  photos with mask   run 2 photos with at least one mask >= the run's min score
  iNat observations  from --inat, rolled up to genus and family through the
                     taxa dump's ancestry. --inat is either iNat's
                     observations.csv.gz dump (one row per observation) or a
                     parquet. A parquet with an observation_uuid column counts
                     distinct observations; one without (inat_photos.parquet is
                     one row per photo) counts rows, and the plots say "photos".

A taxon is plotted if it is under the threshold on either count, coloured by
which: hits only, masks only, or both. Since a photo needs a hit to be run at
all, photos with mask <= hits, so "hits only" is normally empty; it is kept so
an empty legend entry shows that. Clicking a point opens its iNat taxon page.

Writes into --export:

    coverage_genera.html, coverage_families.html   the interactive plots
    low_hit_taxa.csv     every genus and family with fewer than --threshold
                         phenovision hits, with its counts and iNat link
"""

import argparse
import json
from pathlib import Path

import pandas as pd

ANGIOSPERMAE = 47125
TAXON_URL = "https://www.inaturalist.org/taxa/{}"
PLOTLY = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"
CATEGORIES = {
    "both": ("Fewer than {t} hits and fewer than {t} masked photos", "#d1495b"),
    "masks": ("Fewer than {t} masked photos only", "#edae49"),
    "hits": ("Fewer than {t} phenovision hits only", "#2a78d6"),
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--genera", required=True, help="<index stem>_genera.csv")
    parser.add_argument("--metadata", required=True, help="all_genera_segment_metadata.csv")
    parser.add_argument("--taxa", required=True, help="iNat taxa.csv.gz")
    parser.add_argument("--inat", required=True,
                        help="observations.csv.gz, or a parquet with taxon_id")
    parser.add_argument("--research-only", action="store_true",
                        help="Count only research-grade rows (needs quality_grade)")
    parser.add_argument("--threshold", type=int, default=30)
    parser.add_argument("--export", required=True)
    args = parser.parse_args()

    export = Path(args.export)
    export.mkdir(parents=True, exist_ok=True)
    t = args.threshold

    taxa = load_taxa(args.taxa)
    counts, unit = inat_counts(args.inat, args.research_only)
    print(f"iNat       : {int(counts.sum()):,} {unit} over {len(counts):,} taxa from {args.inat}")
    rolled = roll_up(counts, taxa)

    genera = pd.read_csv(args.genera)
    genera = genera[genera["angiosperm"].astype(str).str.lower() == "true"]
    meta = pd.read_csv(args.metadata, usecols=["genus", "family", "num_masks"])
    run = meta.groupby("genus").agg(photos_run=("num_masks", "size"),
                                    photos_with_mask=("num_masks", lambda s: int((s > 0).sum())))

    genus = genera.set_index("genus")[["family", "flower_detected"]].join(run)
    genus = genus.rename(columns={"flower_detected": "phenovision_hits"})
    genus[["photos_run", "photos_with_mask"]] = genus[["photos_run", "photos_with_mask"]] \
        .fillna(0).astype(int)
    family = genus.groupby("family")[["phenovision_hits", "photos_run",
                                      "photos_with_mask"]].sum()
    family["genera"] = genus.groupby("family").size()

    angio = taxa[taxa["angiosperm"]]
    for frame, rank in ((genus, "genus"), (family, "family")):
        ids = (angio[angio["rank"].isin([rank, rank + "hybrid"])]
               .sort_values("active", ascending=False)
               .drop_duplicates("name").set_index("name")["taxon_id"])
        frame["inat_taxon_id"] = frame.index.map(ids).astype("Int64")
        frame["inat_observations"] = frame["inat_taxon_id"].map(rolled).astype("Int64")
        frame["inat_url"] = [TAXON_URL.format(i) if pd.notna(i) else ""
                             for i in frame["inat_taxon_id"]]
        low_hits = frame["phenovision_hits"] < t
        low_masks = frame["photos_with_mask"] < t
        frame["category"] = ""
        frame.loc[low_hits & low_masks, "category"] = "both"
        frame.loc[low_hits & ~low_masks, "category"] = "hits"
        frame.loc[~low_hits & low_masks, "category"] = "masks"

    genus = genus.rename_axis("name").reset_index()
    family = family.rename_axis("name").reset_index()
    for frame, rank in ((genus, "genus"), (family, "family")):
        thin = frame[frame["category"] != ""]
        print(f"{rank:11s}: {len(frame):,} total; under {t}: "
              + ", ".join(f"{k} {int((thin['category'] == k).sum()):,}" for k in CATEGORIES))
        page = export / f"coverage_{'genera' if rank == 'genus' else 'families'}.html"
        page.write_text(html(thin, rank, unit, t))
        print(f"           : -> {page}")

    low = pd.concat([genus.assign(rank="genus"), family.assign(rank="family")],
                    ignore_index=True)
    low = low[low["phenovision_hits"] < t]
    cols = ["rank", "name", "family", "genera", "phenovision_hits", "photos_run",
            "photos_with_mask", "category", "inat_observations", "inat_taxon_id", "inat_url"]
    low = low.reindex(columns=cols).sort_values(["rank", "phenovision_hits", "name"],
                                                ascending=[False, True, True])
    low.to_csv(export / "low_hit_taxa.csv", index=False)
    print(f"low hits   : {len(low):,} taxa -> {export / 'low_hit_taxa.csv'}")


def load_taxa(path):
    taxa = pd.read_csv(path, sep="\t", dtype=str,
                       usecols=["taxon_id", "ancestry", "rank", "name", "active"])
    taxa["taxon_id"] = taxa["taxon_id"].astype("int64")
    taxa["ancestry"] = taxa["ancestry"].fillna("")
    taxa["angiosperm"] = ("/" + taxa["ancestry"] + "/").str.contains(
        f"/{ANGIOSPERMAE}/", regex=False)
    taxa["active"] = taxa["active"].str.lower() == "true"
    return taxa


def inat_counts(path, research_only):
    """Rows per taxon_id in the iNat source -> (Series, unit name)."""
    import duckdb
    path = str(path)
    con = duckdb.connect()
    if path.endswith(".parquet"):
        source = f"read_parquet('{path}')"
        columns = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()}
        if "observation_uuid" in columns:
            what, unit = "COUNT(DISTINCT observation_uuid)", "observations"
        else:
            what, unit = "COUNT(*)", "photos"
    else:
        source = f"read_csv('{path}', delim='\\t', header=true, all_varchar=true)"
        columns = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()}
        what, unit = "COUNT(*)", "observations"
    where = "WHERE quality_grade = 'research'" if research_only else ""
    if research_only and "quality_grade" not in columns:
        raise SystemExit(f"--research-only: no quality_grade column in {path}")
    if research_only:
        unit = "research-grade " + unit
    df = con.execute(f"SELECT TRY_CAST(taxon_id AS BIGINT) AS taxon_id, {what} AS n "
                     f"FROM {source} {where} GROUP BY 1").df().dropna()
    return df.set_index(df["taxon_id"].astype("int64"))["n"].astype("int64"), unit


def roll_up(counts, taxa):
    """taxon_id -> rows at that taxon or anywhere below it."""
    lineage = taxa.set_index("taxon_id")["ancestry"].reindex(counts.index).fillna("")
    frame = pd.DataFrame({"n": counts.to_numpy(),
                          "ids": (lineage + "/" + counts.index.astype(str)).str.split("/")})
    frame = frame.explode("ids")
    frame = frame[frame["ids"] != ""]
    return frame.groupby(frame["ids"].astype("int64"))["n"].sum()


def html(thin, rank, unit, t):
    plural = "genera" if rank == "genus" else "families"
    traces = []
    shown = thin.dropna(subset=["inat_observations"])
    shown = shown[shown["inat_observations"] > 0]
    for key, (label, color) in CATEGORIES.items():
        part = shown[shown["category"] == key]
        hover = [
            f"<b>{r['name']}</b>" + (f" ({r['family']})" if rank == "genus" else
                                     f" ({int(r['genera'])} genera)")
            + f"<br>phenovision hits: {int(r['phenovision_hits']):,}"
            f"<br>iNat {unit}: {int(r['inat_observations']):,}"
            f"<br>photos run: {int(r['photos_run']):,}, with mask: {int(r['photos_with_mask']):,}"
            for _, r in part.iterrows()]
        traces.append({
            "type": "scattergl" if len(shown) > 3000 else "scatter", "mode": "markers",
            "name": f"{label.format(t=t)} ({len(part):,})",
            "x": [int(v) for v in part["inat_observations"]],
            "y": [int(v) for v in part["phenovision_hits"]],
            "text": hover, "hoverinfo": "text", "customdata": list(part["inat_url"]),
            "marker": {"color": color, "size": 7, "opacity": 0.75,
                       "line": {"width": 0.5, "color": "#ffffff"}},
        })
    missing = len(thin) - len(shown)
    title = (f"{plural.capitalize()} under {t}: phenovision hits vs. iNat {unit}"
             f"<br><sub>{len(shown):,} {plural}; click a point to open it on iNaturalist"
             + (f"; {missing:,} with no iNat match not shown" if missing else "") + "</sub>")
    layout = {
        "title": {"text": title, "x": 0.01},
        "xaxis": {"title": f"Total iNat {unit} (log)", "type": "log"},
        "yaxis": {"title": "Phenovision hits (flower detected, log)", "type": "log"},
        "hovermode": "closest", "legend": {"orientation": "h", "y": -0.18},
        "margin": {"t": 80}, "template": "plotly_white",
    }
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Thin {plural}</title>
<script src="{PLOTLY}"></script>
<style>html,body{{margin:0;height:100%;background:#fff;font-family:sans-serif}}
#plot{{width:100%;height:100vh}}</style></head>
<body><div id="plot"></div><script>
const traces = {json.dumps(traces)};
const plot = document.getElementById("plot");
Plotly.newPlot(plot, traces, {json.dumps(layout)},
               {{responsive: true, scrollZoom: true, displaylogo: false}});
plot.on("plotly_click", e => {{
  const url = e.points[0].customdata;
  if (url) window.open(url, "_blank", "noopener");
}});
</script></body></html>
"""


if __name__ == "__main__":
    main()
