#!/usr/bin/env python3
"""
Photo atlas of the bullseye genera: every usable photo of each top genus,
cropped the way the model saw it, plus a PDF.

    python all_genera/pv99/analysis/bullseye_atlas.py
    python all_genera/pv99/analysis/bullseye_atlas.py --min-share 0.4 --per-genus 50 --no-pdf

Inputs (from bullseye_report.py): export/pv99/analysis/bullseye/photos.csv and
genus_bullseye.csv. A genus is in the atlas when it has >= --min-told usable
("told") photos and an adjusted share (the `posterior` column) >= --min-share;
the default 0.5 / 10 gives the 63 bullseye genera of the 2026-11-10 run.

For each genus, up to --per-genus of its told photos, highest P(bullseye)
first, so the photos the model rejected show at the end of each genus. Every
tile is the top segment's context crop (top_segment_crop, as the model saw
it), --tile px square, with P(bullseye) and the call printed on it.

Writes into export/pv99/analysis/bullseye/atlas/:

    <rank>_<Genus>/<n>_<photo_id>.jpg   the tiles
    atlas_index.csv                     one row per tile: genus, family, rank,
                                        photo_id, P(bullseye), call, tile path
    pv99_bullseye_atlas_full.pdf        one page per genus (unless --no-pdf)
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from bullseye_classify import top_segment_crop  # noqa: E402
from top_segments import _font  # noqa: E402

CALL_COLOUR = {"bullseye": (46, 160, 67), "partial": (210, 150, 30), "none": (150, 150, 150),
               "cant_tell": (110, 110, 110)}


def make_tile(job):
    project, row, path, size = job
    try:
        crop, _, _ = top_segment_crop(project, row["image_path"], row["mask_path"], size)
    except Exception as exc:
        return dict(photo_id=row["photo_id"], tile="", error=f"{type(exc).__name__}: {exc}"[:200])
    draw = ImageDraw.Draw(crop)
    bar = max(16, size // 12)
    draw.rectangle((0, size - bar, size, size), fill=(20, 20, 22))
    draw.rectangle((0, size - bar, 6, size), fill=CALL_COLOUR.get(row["call"], (110, 110, 110)))
    draw.text((10, size - bar // 2), f"{row['bullseye__bullseye']:.2f}  {row['call']}",
              fill=(240, 240, 240), font=_font(int(bar * 0.7)), anchor="lm")
    path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(path, quality=84)
    return dict(photo_id=row["photo_id"], tile=str(path), error="")


def build_pdf(index, genera, out_pdf, family):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=18, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13, spaceBefore=0, spaceAfter=1)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("s", parent=body, fontSize=8, leading=10.5, textColor=colors.HexColor("#555"))
    cell = ParagraphStyle("c", parent=body, fontSize=8, leading=9.5)
    red = '<font color="#c62828"><b>pv99</b></font>'

    def grid(paths, cols, size):
        cells = [RLImage(p, width=size, height=size) for p in paths]
        cells += [""] * (-len(cells) % cols)
        rows = [cells[i:i + cols] for i in range(0, len(cells), cols)]
        t = Table(rows, colWidths=[size + 2] * cols, rowHeights=[size + 2] * len(rows))
        t.setStyle(TableStyle([(k, (0, 0), (-1, -1), 1) for k in
                               ("LEFTPADDING", "RIGHTPADDING", "TOPPADDING", "BOTTOMPADDING")]))
        return t

    story = [Paragraph("Bullseye genera in pv99: full photo atlas", h1),
             Paragraph(f"Dataset: {red} SAM3 top segments &nbsp;|&nbsp; classifier: Qwen3-VL-8B zero-shot", small),
             Spacer(1, 8),
             Paragraph(
                 f"{len(index):,} photos of the {len(genera)} genera ranked as bullseye genera in the {red} "
                 "screen. Each genus gets one page with its usable photos, highest P(bullseye) first, so the "
                 "photos the model rejected sit at the end of the page. Each tile is the crop around the "
                 "photo's top SAM3 segment that the model saw. The strip on each tile gives P(bullseye) and "
                 "the call; its left edge is green for bullseye, amber for partial and grey for none.", body),
             Spacer(1, 4),
             Paragraph(
                 "<b>Adjusted share</b>: the share of a genus's usable photos called bullseye, pulled toward "
                 "the run-wide rate by the equivalent of about four extra photos, so a genus with few photos "
                 "cannot rank high on luck. It measures how consistently the model sees a bullseye, not a "
                 "verified trait.", body),
             Spacer(1, 10)]
    rows = [["#", "Genus", "Family", "Bullseye / usable", "Adjusted share"]]
    for g in genera.itertuples():
        rows.append([str(g.rank), Paragraph(f"<i>{g.genus}</i>", cell), g.family,
                     f"{g.bullseye}/{g.told}", f"{g.posterior:.2f}"])
    t = Table(rows, colWidths=[0.45 * inch, 1.7 * inch, 1.7 * inch, 1.2 * inch, 1.1 * inch],
              repeatRows=1)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 7.5), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ececea")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f5")]),
        ("ALIGN", (3, 1), (-1, -1), "RIGHT"), ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    story += [t, PageBreak()]

    fam = family.set_index("family")
    for g in genera.itertuples():
        tiles = index[(index["genus_rank"] == g.rank) & (index["tile"] != "")]["tile"].tolist()
        f = fam.loc[g.family]
        story += [Paragraph(f"#{g.rank} &nbsp;<i>{g.genus}</i> &nbsp;<font size=10 color='#555'>{g.family}</font>", h2),
                  Paragraph(f"{g.bullseye} of {g.told} usable photos called bullseye &nbsp;|&nbsp; adjusted share "
                            f"{g.posterior:.2f} (90% interval {g.lo90:.2f}-{g.hi90:.2f}) &nbsp;|&nbsp; "
                            f"family: {int(f.bullseye_genera)} bullseye genera of {int(f.genera)}", small),
                  Spacer(1, 4), grid(tiles, 7, 0.98 * inch), PageBreak()]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#777"))
        canvas.drawString(0.6 * inch, 0.4 * inch, "pv99 bullseye atlas")
        canvas.drawRightString(7.9 * inch, 0.4 * inch, str(doc.page))
        canvas.restoreState()

    SimpleDocTemplate(str(out_pdf), pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                      topMargin=0.45 * inch, bottomMargin=0.6 * inch,
                      title="pv99 bullseye atlas").build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_args(parser)
    parser.add_argument("--min-share", type=float, default=0.5, help="adjusted share cut")
    parser.add_argument("--min-told", type=int, default=10)
    parser.add_argument("--per-genus", type=int, default=49, help="7 x 7 fills one PDF page")
    parser.add_argument("--tile", type=int, default=300)
    parser.add_argument("--workers", type=int, default=common.n_workers())
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()

    project = common.project_dir(args.project)
    folder = common.analysis_dir(project, "bullseye")
    out = common.analysis_dir(project, "bullseye/atlas")
    genus = pd.read_csv(folder / "genus_bullseye.csv")
    family = pd.read_csv(folder / "family_bullseye.csv")
    genera = genus[(genus["told"] >= args.min_told) & (genus["posterior"] >= args.min_share)]
    genera = genera.sort_values("rank")

    photos = pd.read_csv(folder / "photos.csv")
    # Genus names can repeat across families (homonyms), so match on both.
    photos = photos[photos["told"].astype(str).str.lower().isin(["true", "1"])]
    photos = photos.merge(genera[["genus", "family", "rank"]].rename(columns={"rank": "genus_rank"}),
                          on=["genus", "family"])
    photos = photos.merge(common.load_metadata(project, ["photo_id", "image_path", "mask_path"]),
                          on="photo_id")
    photos = photos.sort_values(["genus_rank", "bullseye__bullseye"], ascending=[True, False])
    photos = photos.groupby("genus_rank").head(args.per_genus).copy()
    photos["n"] = photos.groupby("genus_rank").cumcount() + 1

    jobs = []
    for r in photos.to_dict("records"):
        path = out / f"{r['genus_rank']:03d}_{common.safe(r['genus'])}" / f"{r['n']:02d}_{r['photo_id']}.jpg"
        jobs.append((project, r, path, args.tile))
    print(f"[atlas] {len(genera)} genera, {len(jobs):,} photos -> {out}", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        made = pd.DataFrame(list(ex.map(make_tile, jobs, chunksize=8)))
    index = photos.drop(columns=["error"], errors="ignore").merge(made, on="photo_id")
    cols = ["genus_rank", "genus", "family", "n", "photo_id", "bullseye__bullseye", "call", "tile", "error"]
    index = index[cols].sort_values(["genus_rank", "n"])
    index.to_csv(out / "atlas_index.csv", index=False)
    bad = (index["error"] != "").sum()
    print(f"[atlas] tiles: {len(index) - bad:,} ok, {bad} failed", flush=True)

    if not args.no_pdf:
        pdf = out / "pv99_bullseye_atlas_full.pdf"
        build_pdf(index, genera, pdf, family)
        print(f"[atlas] pdf: {pdf} ({pdf.stat().st_size / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
