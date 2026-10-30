#!/usr/bin/env python3
"""
A punchy 3-page briefing on where Rudbeckia hirta has been introduced and on
its cultivars and "hybrids". Uses only facts already graded in
rhirta_variants_sources.pdf; see that file for the evidence.

    python charisma_gpu/introductions_summary_pdf.py --root C:/Users/nevyn/Documents/rhirta_bullseye

Writes root/maps/variants/rhirta_introductions_cultivars_summary.pdf (and
timeline.png beside it).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
import pandas as pd                                        # noqa: E402
from reportlab.lib import colors                           # noqa: E402
from reportlab.lib.pagesizes import letter                 # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import inch                       # noqa: E402
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)

NAVY = colors.HexColor("#2B3A55")
LIGHT = colors.HexColor("#EAEEF5")
BLUE = colors.HexColor("#2a78d6")
INK2 = colors.HexColor("#52514e")

ss = getSampleStyleSheet()
title = ParagraphStyle("title", parent=ss["Title"], textColor=NAVY, fontSize=22, leading=26,
                       alignment=0, spaceAfter=2)
kicker = ParagraphStyle("kicker", parent=ss["Normal"], textColor=INK2, fontSize=10.5,
                        spaceAfter=10)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=NAVY, fontSize=14.5, spaceBefore=8,
                    spaceAfter=4)
body = ParagraphStyle("body", parent=ss["Normal"], fontSize=10, leading=13.6, spaceAfter=5)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=13, bulletIndent=1, spaceAfter=3.5)
note = ParagraphStyle("note", parent=ss["Normal"], fontSize=8, textColor=INK2, leading=10.5)
cell = ParagraphStyle("cell", parent=ss["Normal"], fontSize=8.8, leading=11.2)
cellb = ParagraphStyle("cellb", parent=cell, textColor=colors.white, fontName="Helvetica-Bold")
statn = ParagraphStyle("statn", parent=ss["Normal"], fontSize=24, leading=27, textColor=BLUE,
                       fontName="Helvetica-Bold", alignment=1)
statl = ParagraphStyle("statl", parent=ss["Normal"], fontSize=8.6, leading=10.5, textColor=INK2,
                       alignment=1)

TIMELINE = [  # event year, label, label x, label y (labels staggered so none collide)
    (1935, "1930s: Blakeslee (Smith College)\ndoubles R. hirta's chromosomes\nwith colchicine",
     1937, 0.78),
    (1957, "1957: Burpee launches\nthe 'Gloriosa Daisy'", 1948, -0.72),
    (1960, "1960: Gloriosa\nGolden", 1956, 0.42),
    (1961, "1961: Gloriosa Double,\nAAS silver", 1964, -0.72),
    (1963, "1963: 'Pinwheel'\nbicolor", 1969, 0.42),
    (2002, "2002: 'Cherokee Sunset'\n(Thompson & Morgan), AAS", 1993, -0.72),
    (2003, "2003: 'Prairie Sun'\n(Benary), AAS", 1997, 0.42),
    (2011, "2009 or 2013: 'Denver Daisy'\n(Benary / Plant Select)", 2012, -0.72),
    (2014, "2014: Echibeckia 'Summerina'\nat the California\nSpring Trials", 2013, 0.78),
]


def timeline_png(path):
    fig, ax = plt.subplots(figsize=(7.6, 2.7), dpi=220)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axhline(0, color="#52514e", linewidth=1.2, zorder=1)
    ax.plot([1965, 2000], [0, 0], color="#d6d5d0", linewidth=5, zorder=0, solid_capstyle="round")
    for year, label, lx, ly in TIMELINE:
        ax.plot([year, lx], [0, ly * 0.62], color="#a8a7a2", linewidth=0.7, zorder=1)
        ax.scatter(year, 0, s=40, color="#2a78d6", edgecolors="#fcfcfb", linewidths=1.2,
                   zorder=3)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=6.4, color="#0b0b0b",
                linespacing=1.15)
    ax.text(1980, -0.13, "decades of strains and selections ('Indian Summer',\n'Toto', "
            "'Marmalade', 'Irish Eyes', ...)", ha="center", va="top", fontsize=6.2,
            color="#52514e", style="italic")
    ax.set_xlim(1925, 2025)
    ax.set_ylim(-1.05, 1.08)
    ax.set_yticks([])
    ax.set_xticks([1930, 1950, 1970, 1990, 2010])
    ax.tick_params(labelsize=7, colors="#52514e", length=0, pad=-2)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def table(rows, widths, header=True):
    data = [[Paragraph(c, cellb if (header and i == 0) else cell) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1 if header else 0)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY if header else colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


def stats_row(items):
    data = [[Paragraph(n, statn) for n, _ in items], [Paragraph(l, statl) for _, l in items]]
    t = Table(data, colWidths=[7.3 / len(items) * inch] * len(items))
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                           ("LINEAFTER", (0, 0), (-2, -1), 2, colors.white),
                           ("TOPPADDING", (0, 0), (-1, 0), 8),
                           ("BOTTOMPADDING", (0, 1), (-1, 1), 8)]))
    return t


def build(root):
    out_dir = root / "maps" / "variants"
    gbif = pd.read_csv(root / "metadata" / "gbif_rhirta_outside_us_ca.csv").dropna(
        subset=["lat", "lon"])
    eu = gbif[gbif.lon.between(-11, 45) & gbif.lat.between(35, 66)]
    ea = gbif[gbif.lon.between(100, 146) & gbif.lat.between(22, 50)]
    top = gbif.country.value_counts()
    timeline_png(out_dir / "timeline.png")
    s = []
    P = lambda t, st=body: s.append(Paragraph(t, st))
    B = lambda t: s.append(Paragraph(t, bullet, bulletText="•"))
    gap = lambda h=6: s.append(Spacer(1, h))

    # ── page 1: where it has gone ──────────────────────────────────────────
    P("Black-eyed Susans abroad and in the garden", title)
    P("A three-page briefing on where <i>Rudbeckia hirta</i> has been introduced, and on the "
      "cultivars and 'hybrids' people plant", kicker)
    s.append(stats_row([
        (f"{len(gbif):,}", "GBIF records outside<br/>the US and Canada"),
        (f"{gbif.country.nunique()}", "countries with<br/>records"),
        (f"{len(eu) / len(gbif):.0%}", "of those records<br/>are in Europe"),
        ("1957", "the 'Gloriosa Daisy'<br/>goes on sale"),
    ]))
    gap(8)
    P("The short version", h2)
    B("<b>A North American wildflower that Europe adopted.</b> Outside its native range, "
      "nine in ten records are European. East Asia is a distant second.")
    B("<b>It spread through gardens.</b> It was introduced to Europe as an ornamental, and "
      "it's also sown in wildflower mixes for re-naturalisation and erosion control. Expect "
      "garden forms and their escapes, not wild varieties.")
    B("<b>One breakthrough made the modern garden plant.</b> Colchicine-doubled tetraploids "
      "from the 1930s became Burpee's 1957 'Gloriosa Daisy'. Almost everything since is a "
      "strain or selection of that idea.")
    B("<b>True hybrids barely exist.</b> The 'hybrids' on sale are crosses with "
      "<i>Echinacea</i>, not with other <i>Rudbeckia</i>.")

    P("Where it has gone", h2)
    world = out_dir / "01_world.png"
    if world.exists():
        img = Image(str(world), width=6.5 * inch, height=6.5 * inch * 1122 / 2040)
        s.append(img)
    gap(4)
    B(f"<b>Europe: {len(eu):,} records.</b> Germany {top.get('DE', 0):,}, Belgium "
      f"{top.get('BE', 0):,}, Sweden {top.get('SE', 0):,}, the Netherlands {top.get('NL', 0):,}, "
      f"Russia {top.get('RU', 0):,}, Switzerland {top.get('CH', 0):,}, the UK "
      f"{top.get('GB', 0):,}. Euro+Med lists it as <b>naturalised</b> in many countries. "
      "Its status is uncertain in Great Britain, Latvia and Romania, and given only as "
      "introduced in Norway and Spain.")
    B(f"<b>East Asia: {len(ea):,} records.</b> South Korea {top.get('KR', 0):,}, Japan "
      f"{top.get('JP', 0):,}; also listed for China, Taiwan and Vietnam.")
    B("<b>Elsewhere, a trickle:</b> a few records each from Australia, New Zealand, Brazil, "
      "Argentina, Mexico, India and North Africa.")
    P("Read the numbers as where people photograph it, not how much grows there: 87% of "
      "records are photo observations, and 92% carry no establishment status.", note)

    s.append(PageBreak())
    # ── page 2: cultivars ──────────────────────────────────────────────────
    P("The cultivars: one breakthrough, then a long tail", title)
    P("Garden <i>R. hirta</i> is mostly the 'Gloriosa Daisy' line and what breeders built on it",
      kicker)
    s.append(Image(str(out_dir / "timeline.png"), width=7.3 * inch, height=7.3 * inch * 2.7 / 7.6))
    gap(6)
    s.append(table([
        ["Cultivar or strain", "Where it came from", "What's notable"],
        ["<b>Gloriosa Daisy</b> (strain)", "Blakeslee, Smith College (1930s); Burpee, 1957",
         "Tetraploid: chromosomes doubled with colchicine. Much larger heads, often doubled or "
         "with contrasting ray markings. Shown at the 1957 Philadelphia Flower Show."],
        ["Gloriosa Golden Daisy", "Burpee, 1960", "A golden single."],
        ["Gloriosa Double Daisy", "Burpee", "Double-flowered; All-America Selections silver "
                                            "medal, 1961."],
        ["'Pinwheel'", "Burpee, 1963", "A bicolor."],
        ["<b>'Cherokee Sunset'</b>", "Thompson &amp; Morgan, UK", "Mahogany and bronze "
         "colouring, about ten years in development; AAS winner (2002); good cut flower."],
        ["<b>'Prairie Sun'</b>", "Benary, Germany", "AAS winner (2003) and Fleuroselect Gold "
                                                    "Medal."],
        ["<b>'Denver Daisy'</b>", "Benary, via Plant Select (Denver)", "Made for Denver's 150th "
         "anniversary; sources say 2009 or 2013."],
        ["'Indian Summer'", "breeder not found", "Sturdy, large yellow flowers 10–14 weeks "
                                                 "from sowing."],
        ["'Toto', 'Double Gold', 'Marmalade', 'Kelvedon Star', 'Irish Eyes'",
         "breeders not found", "Widely grown names."],
    ], [1.75, 1.95, 3.6]))
    gap(6)
    P("What to take from it", h2)
    B("<b>Bigger and redder.</b> The garden line selected for large heads, doubles, bicolors "
      "and mahogany. Those are exactly the red-rayed flowers the bullseye analysis flags.")
    B("<b>Bred on both sides of the Atlantic.</b> Burpee (USA), Thompson &amp; Morgan (UK) "
      "and Benary (Germany) put the same cultivars into European seed catalogues as American "
      "ones. That fits Europe holding most introduced records.")
    B("<b>Prizes mark the waves.</b> AAS medals in 1961, 2002 and 2003 bracket the two big "
      "pushes: the first Gloriosa strains, and the 2000s cultivars.")

    s.append(PageBreak())
    # ── page 3: hybrids, meaning, open questions ───────────────────────────
    P("Hybrids, and what all this means for the bullseye data", title)
    P("Why 'hybrid' on a plant tag usually means <i>Echinacea</i>, and how garden plants "
      "show up in the photos", kicker)
    P("Hybrids: mostly marketing", h2)
    B("<b><i>Rudbeckia</i> × <i>Rudbeckia</i> almost never works.</b> In crosses of "
      "<i>R. hirta</i> cultivars with <i>R. fulgida</i>, <i>R. missouriensis</i> and "
      "<i>R. subtomentosa</i>, just one of 844 seedlings was a confirmed hybrid "
      "(<i>R. subtomentosa</i> × <i>R. hirta</i>). Mismatched chromosome numbers and "
      "pseudogamy, which gives maternal look-alikes, block the rest (Palmer et al. 2009).")
    B("<b>The 'hybrids' on sale are <i>Echinacea purpurea</i> × <i>R. hirta</i>.</b> They're "
      "sold as <b>Echibeckia 'Summerina'</b> (Orange, Yellow, Brown, Blazing Fire, Glow Sugar "
      "Shake) under patented 'ET RDB' lines; 'Summerina Orange' is 'ET-RDB 01', PP 25,221. "
      "Distribution is exclusive to Pacific Plug &amp; Liner. They're rated hardy to zone 6. "
      "A Dutch origin is reported but unconfirmed.")
    B("<b>So they stay in gardens.</b> No source suggests they escape. Expect them in nurseries "
      "and plantings, not on roadsides.")
    P("What it means for the bullseye photos", h2)
    s.append(table([
        ["Pattern in the data", "Most likely explanation"],
        ["61 flowers where red covers most of the ray, clustered in the Northeast and "
         "Great Lakes", "Garden cultivars: mahogany and bicolor Gloriosa-type plants. "
         "This is an inference; no photo was checked against garden records."],
        ["Up to 30–40% of photos with a small red base, in the Texas Hill Country / Edwards "
         "Plateau", "Wild variation. A wild Texas genebank collection (USDA GRIN) is described "
         "with a red-brown basal spot."],
        ["Records in Europe and East Asia", "Introduced garden forms and their escapes. Expect "
         "cultivar traits (large, doubled or bicolored heads) more often than in the wild "
         "native range."],
    ], [3.0, 4.3]))
    gap(4)
    P("Open questions", h2)
    B("<b>When did it first reach Europe and Asia?</b> No source here gives a date. Old herbarium "
      "specimens in GBIF (1,078 outside the US and Canada) could answer it.")
    B("<b>Is it invasive anywhere,</b> or only naturalised? Not established for any country, "
      "including Japan and Korea. The searches there turned up the related <i>R. laciniata</i>.")
    B("<b>How many red-rayed photos are cultivated plants?</b> Checking iNaturalist's "
      "'captive/cultivated' flag, or looking at the garden context in the photos, would test "
      "the cultivar reading.")
    gap(10)
    P("Sources: every claim here is graded and referenced in <i>rhirta_variants_sources.pdf</i> "
      "(same folder). Counts are from GBIF (9,530 records outside the US/Canada, fetched "
      "2026-10-30) and this study's iNaturalist analysis. Literature claims come from search "
      "summaries of Flora of North America, Kew POWO, Euro+Med, Palmer et al. 2009, USPTO "
      "patents, USDA GRIN, Wikipedia, extension services, nurseries and garden press. Check the "
      "originals before citing.", note)

    out = out_dir / "rhirta_introductions_cultivars_summary.pdf"
    SimpleDocTemplate(str(out), pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                      topMargin=0.5 * inch, bottomMargin=0.5 * inch,
                      title="Black-eyed Susans abroad and in the garden",
                      author="remote_sam3").build(s)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    build(Path(ap.parse_args().root))
