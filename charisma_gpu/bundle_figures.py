#!/usr/bin/env python3
"""
Copy every R. hirta figure into one shareable folder, with each interactive map
made fully self-contained: plotly.js and the basemap (topojson) inlined, and the
class map's sample PNGs embedded as data URIs. The HTML files then open in any
browser with no internet and no sibling folders.

    python charisma_gpu/bundle_figures.py --root C:/Users/nevyn/Documents/rhirta_bullseye \\
        --topojson-dir path/with/world_110m.json+north-america_110m.json --out OUT_DIR
"""

import argparse
import base64
import json
import re
import shutil
from pathlib import Path

GEO = ("world_110m", "north-america_110m")


def inline_geo(html, topo):
    assets = {name: json.loads((topo / f"{name}.json").read_text()) for name in GEO}
    tag = ("<script>window.PlotlyGeoAssets = {topojson: " + json.dumps(assets) + "};</script>")
    # before the first script, so plotly finds the assets instead of fetching them
    i = html.find("<script")
    return html[:i] + tag + "\n" + html[i:]


def inline_images(html, base):
    def repl(m):
        p = base / m.group(1)
        return '"data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode() + '"'
    return re.sub(r'"(class_samples/\d+\.png)"', repl, html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--topojson-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    maps, topo, out = Path(a.root) / "maps", Path(a.topojson_dir), Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    groups = {
        "1_bullseye_geography": maps,
        "2_varieties_cultivars_worldwide": maps / "variants",
        "3_environment_and_urban_rural_US": maps / "environment",
    }
    for folder, src in groups.items():
        dst = out / folder
        dst.mkdir(parents=True)
        for f in sorted(src.iterdir()):
            if f.suffix.lower() in (".png", ".pdf") and f.name != "timeline.png":
                shutil.copy2(f, dst / f.name)
            elif f.suffix.lower() == ".html":
                html = f.read_text(encoding="utf-8")
                html = inline_geo(html, topo)
                html = inline_images(html, src)
                (dst / f.name).write_text(html, encoding="utf-8")
    # nothing in the HTML may still point outside the file
    for h in out.rglob("*.html"):
        t = h.read_text(encoding="utf-8")
        assert "class_samples/" not in t, h
        ext = set(re.findall(r'(?:src|href)="(https?://[^"]+)"', t))
        print(f"{h.relative_to(out)}: {h.stat().st_size / 1e6:.1f} MB, external src/href: "
              f"{sorted(ext) or 'none'}")
    n = sum(1 for _ in out.rglob("*") if _.is_file())
    print(f"{n} files -> {out}")


if __name__ == "__main__":
    main()
