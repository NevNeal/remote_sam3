#!/usr/bin/env python3
"""
Interactive bullseye size-class map: every photo drawn, N per class clickable.
Clicking a highlighted point shows that photo's charisma output (cut-out |
per-pixel CLUT | petal map, as sample_pngs.py draws it) in a side panel.

    python charisma_gpu/class_map.py --root C:/Users/nevyn/Documents/rhirta_bullseye

Reads   root/maps/rbs_by_photo.csv (bullseye_maps.py) and root/segments/.
Writes  root/maps/class_samples/<photo_id>.png and
        root/maps/rhirta_bullseye_classes_interactive.html (open it from that
        folder; it loads the PNGs by relative path).
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from run import png_rows
from sample_pngs import draw_photo

HERE = Path(__file__).resolve().parent
NA = (-128, -60, 18, 58)
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
CLASSES = [  # name, lower bound, colour (reference blue ramp; grey for none), size
    ("none", 0.0, "#c9c8c3", 3),
    ("small (5–15%)", 0.05, "#86b6ef", 5),
    ("medium (15–30%)", 0.15, "#3987e5", 6),
    ("large (30–50%)", 0.30, "#1c5cab", 7),
    ("dominant (≥ 50%)", 0.50, "#0d366b", 8),
]

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>R. hirta bullseye classes</title>
<style>
  :root {{ --surface: #fcfcfb; --panel: #f3f2ee; --ink: #0b0b0b; --ink2: #52514e;
          --line: #d6d5d0; --accent: #2a78d6; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--surface); color: var(--ink);
         font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }}
  header {{ padding: 14px 18px 6px; }}
  h1 {{ margin: 0; font-size: 20px; }}
  header p {{ margin: 4px 0 0; color: var(--ink2); }}
  main {{ display: flex; gap: 14px; padding: 8px 18px 18px; height: calc(100vh - 78px); }}
  #map {{ flex: 1 1 62%; min-width: 0; min-height: 0; border: 1px solid var(--line);
          border-radius: 8px; overflow: hidden; }}
  aside {{ flex: 1 1 38%; min-width: 300px; background: var(--panel); border: 1px solid var(--line);
          border-radius: 8px; padding: 14px; overflow: auto; }}
  aside h2 {{ margin: 0 0 4px; font-size: 16px; }}
  .meta {{ color: var(--ink2); margin: 0 0 10px; }}
  .meta a {{ color: var(--accent); }}
  aside img {{ width: 100%; height: auto; border-radius: 6px; display: block; }}
  .hint {{ color: var(--ink2); }}
  .nav {{ display: flex; gap: 8px; margin: 10px 0 0; }}
  button {{ font: inherit; padding: 4px 10px; border: 1px solid var(--line); border-radius: 6px;
           background: var(--surface); color: var(--ink); cursor: pointer; }}
  @media (max-width: 900px) {{ main {{ flex-direction: column; height: auto; }}
                               #map {{ height: 70vh; }} }}
</style></head>
<body>
<header>
  <h1>Rudbeckia hirta bullseye size classes</h1>
  <p>Every photo is drawn; {n_click} ringed photos ({per} per class) are clickable and open
     their charisma output. Relative bullseye size = red on the petals ÷ (red on the petals +
     yellow ray). Click legend entries to hide classes.</p>
</header>
<main>
  <div id="map"></div>
  <aside id="panel">
    <h2 id="title">Click a ringed point</h2>
    <p class="meta" id="meta">The panel shows each segment of that photo as
      cut-out | per-pixel CLUT | petal map (cyan = disk found, red = petal bullseye).</p>
    <img id="img" alt="" hidden>
    <div class="nav" id="nav" hidden>
      <button id="prev">◀ previous in class</button><button id="next">next in class ▶</button>
    </div>
  </aside>
</main>
<script>{plotlyjs}</script>
<script>
const FIG = {fig_json};
const SAMPLES = {samples};
const byClass = {{}};
SAMPLES.forEach(s => (byClass[s.cls] = byClass[s.cls] || []).push(s));
let current = null;
function show(s) {{
  current = s;
  history.replaceState(null, "", "#" + s.photo_id);
  document.getElementById("title").textContent = "photo " + s.photo_id + " — " + s.cls;
  document.getElementById("meta").innerHTML =
    "relative bullseye size " + (100 * s.rbs).toFixed(1) + "% · " + s.segments +
    " usable segment(s) · " + s.lat.toFixed(3) + ", " + s.lon.toFixed(3) +
    " · <a href='https://www.inaturalist.org/photos/" + s.photo_id +
    "' target='_blank' rel='noopener'>iNaturalist photo</a>";
  const img = document.getElementById("img");
  img.src = s.png; img.alt = "charisma output for photo " + s.photo_id; img.hidden = false;
  document.getElementById("nav").hidden = false;
}}
function step(d) {{
  if (!current) return;
  const list = byClass[current.cls], i = list.indexOf(current);
  show(list[(i + d + list.length) % list.length]);
}}
document.getElementById("prev").onclick = () => step(-1);
document.getElementById("next").onclick = () => step(1);
// deep link: page.html#<photo_id> opens that photo
const linked = SAMPLES.find(x => "#" + x.photo_id === location.hash);
if (linked) show(linked);
document.addEventListener("keydown", e => {{
  if (e.key === "ArrowLeft") step(-1); if (e.key === "ArrowRight") step(1); }});
Plotly.newPlot("map", FIG.data, FIG.layout,
               {{responsive: true, scrollZoom: true, displaylogo: false}}).then(gd => {{
  gd.on("plotly_click", ev => {{
    const p = ev.points.find(q => q.customdata !== undefined);
    if (p) show(SAMPLES[p.customdata]);
  }});
}});
</script>
</body></html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--per-class", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clut", default=str(HERE / "clut_rhirta.csv"))
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = p.parse_args()
    root = Path(args.root)
    maps = root / "maps"
    photo = pd.read_csv(maps / "rbs_by_photo.csv")
    photo = photo[photo.longitude.between(NA[0] - 5, NA[1] + 5) &
                  photo.latitude.between(NA[2] - 3, NA[3] + 3)].copy()
    bounds = [c[1] for c in CLASSES[1:]]
    photo["cls_i"] = np.digitize(photo.rbs, bounds)

    picks = pd.concat([g.sample(min(args.per_class, len(g)), random_state=args.seed)
                       for _, g in photo.groupby("cls_i")])
    segs = png_rows(root / "segments").set_index("photo_id")
    out_dir = maps / "class_samples"
    out_dir.mkdir(exist_ok=True)
    jobs = [(pid, segs.loc[pid, "segment_files"], args.clut, str(out_dir))
            for pid in picks.photo_id if pid in segs.index
            and not (out_dir / f"{pid}.png").exists()]
    print(f"[class map] {len(picks)} clickable photos, drawing {len(jobs)} PNGs", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(draw_photo, jobs, chunksize=4))

    samples = []
    fig = go.Figure()
    for k, (name, _, colour, size) in enumerate(CLASSES):
        d = photo[photo.cls_i == k]
        fig.add_trace(go.Scattergeo(
            lon=d.longitude, lat=d.latitude, mode="markers", name=f"{name}  (n = {len(d):,})",
            legendgroup=name, hoverinfo="skip",
            marker=dict(size=size - 1 if k else 2.5, color=colour, opacity=0.55 if k else 0.5,
                        line=dict(width=0))))
    for k, (name, _, colour, size) in enumerate(CLASSES):
        d = picks[picks.cls_i == k]
        idx = list(range(len(samples), len(samples) + len(d)))
        for r in d.itertuples():
            samples.append(dict(photo_id=int(r.photo_id), rbs=float(r.rbs), cls=name,
                                segments=int(r.segments), lat=float(r.latitude),
                                lon=float(r.longitude), png=f"class_samples/{r.photo_id}.png"))
        fig.add_trace(go.Scattergeo(
            lon=d.longitude, lat=d.latitude, mode="markers", name=f"{name}: clickable",
            legendgroup=name, showlegend=False, customdata=idx,
            text=[f"photo {pid}<br>{name}<br>RBS {v:.1%}<br>click to open"
                  for pid, v in zip(d.photo_id, d.rbs)],
            hovertemplate="%{text}<extra></extra>",
            marker=dict(size=size + 5, color=colour, line=dict(width=2, color=INK))))
    fig.update_geos(scope="north america", projection_type="conic conformal",
                    projection_parallels=[33, 45], projection_rotation=dict(lon=-96),
                    lataxis_range=[18, 58], lonaxis_range=[-128, -60],
                    showland=True, landcolor="#f3f2ee", showlakes=True, lakecolor="#ffffff",
                    showocean=True, oceancolor="#ffffff", showsubunits=True,
                    subunitcolor="#a8a7a2", countrycolor=INK2, coastlinecolor=INK2)
    fig.update_layout(autosize=True, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=SURFACE,
                      legend=dict(title="relative bullseye size", x=0.01, y=0.02,
                                  bgcolor="rgba(252,252,251,0.9)", bordercolor="#d6d5d0",
                                  borderwidth=1, itemsizing="constant"),
                      hoverlabel=dict(bgcolor=SURFACE, font_color=INK))
    from plotly.offline import get_plotlyjs
    html = PAGE.format(plotlyjs=get_plotlyjs(), fig_json=fig.to_json(),
                       samples=json.dumps(samples), n_click=len(samples), per=args.per_class)
    out = maps / "rhirta_bullseye_classes_interactive.html"
    out.write_text(html, encoding="utf-8")
    print(f"[class map] wrote {out}")


if __name__ == "__main__":
    main()
