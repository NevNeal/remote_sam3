#!/usr/bin/env python3
"""
Sources and provenance for the R. hirta variants map (variant_map.py): what
each source is, how it was accessed, what was taken from it, where it is used,
and how much weight it can bear.

    python charisma_gpu/variant_sources_pdf.py --root C:/Users/nevyn/Documents/rhirta_bullseye

Writes root/maps/variants/rhirta_variants_sources.pdf.
"""

import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

NAVY = colors.HexColor("#2B3A55")
LIGHT = colors.HexColor("#EAEEF5")
GRADE_COLOURS = {"A": "#1c5cab", "B": "#3987e5", "C": "#86b6ef", "D": "#c9c8c3"}

ss = getSampleStyleSheet()
h1 = ParagraphStyle("h1", parent=ss["Heading1"], textColor=NAVY, fontSize=19, spaceAfter=4)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=NAVY, fontSize=13.5, spaceBefore=10,
                    spaceAfter=5)
sub = ParagraphStyle("sub", parent=ss["Normal"], textColor=colors.grey, fontSize=9.5,
                     spaceAfter=10)
body = ParagraphStyle("body", parent=ss["Normal"], fontSize=10, leading=14, spaceAfter=6)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=2, spaceAfter=3)
note = ParagraphStyle("note", parent=ss["Normal"], fontSize=8.5, textColor=colors.grey,
                      leading=11)
cell = ParagraphStyle("cell", parent=ss["Normal"], fontSize=8.3, leading=10.6)
cellb = ParagraphStyle("cellb", parent=cell, textColor=colors.white, fontName="Helvetica-Bold")
gradec = ParagraphStyle("gradec", parent=cell, fontName="Helvetica-Bold", alignment=1,
                        textColor=colors.white, fontSize=10)

story = []
P = lambda t, s=body: story.append(Paragraph(t, s))
B = lambda t: story.append(Paragraph(t, bullet, bulletText="•"))
gap = lambda h=6: story.append(Spacer(1, h))


def table(header, rows, widths, grade_col=None):
    data = [[Paragraph(h, cellb) for h in header]]
    for r in rows:
        data.append([Paragraph(str(c), gradec if i == grade_col else cell)
                     for i, c in enumerate(r)])
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), NAVY),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
             ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if grade_col is not None:
        for k, r in enumerate(rows, 1):
            g = str(r[grade_col])[0]
            style.append(("BACKGROUND", (grade_col, k), (grade_col, k),
                          colors.HexColor(GRADE_COLOURS.get(g, "#c9c8c3"))))
            style.append(("VALIGN", (grade_col, k), (grade_col, k), "MIDDLE"))
    t.setStyle(TableStyle(style))
    story.append(t)
    gap(8)


# ── the sources ──────────────────────────────────────────────────────────────
# id, source, grade, how accessed, what was taken, where used, limits
SOURCES = [
    ("S1", "<b>Flora of North America</b>, vol. 21, <i>Rudbeckia hirta</i> (text reproduced on EOL)",
     "B", "Search-result summary of the EOL page; FNA itself not opened",
     "Four varieties (hirta, angustifolia, floridana, pulcherrima). var. hirta: Pennsylvania to "
     "Georgia, sparingly north to Maine and west to Illinois. var. angustifolia: 'the South'. "
     "var. floridana: central Florida. var. pulcherrima: widespread, disturbed habitats.",
     "North America map: the variety ranges; page 4, row 1",
     "Ranges are prose, not maps; I drew the outlines (see D1-D4)."),
    ("S2", "<b>Compositae database</b> (citing Perdue's 1957 synopsis)", "B",
     "Search-result summary",
     "Perdue recognised the same four varieties; var. hirta Appalachians to Illinois plus the "
     "Atlantic coast; var. angustifolia across the Gulf Coastal Plain; var. floridana in "
     "<i>southern</i> Florida.",
     "Coastal-plain framing of var. angustifolia; var. floridana extent",
     "Disagrees with S1 on floridana (southern vs central Florida); I drew both (D3)."),
    ("S3", "<b>Wikipedia</b>, 'Rudbeckia hirta' (and mirrors)", "C", "Search-result summaries",
     "var. angustifolia South Carolina to Texas; var. pulcherrima Newfoundland to British "
     "Columbia, south to Alabama and New Mexico. Gloriosa daisies: tetraploid cultivars from "
     "Blakeslee's colchicine work, developed at Burpee, shown at the 1957 Philadelphia Flower "
     "Show; cultivars 'Indian Summer', 'Toto', 'Double Gold', 'Marmalade'. Naturalised in China.",
     "angustifolia's east-west limits; pulcherrima 'nearly throughout'; cultivar text (page 4)",
     "Tertiary source; used only where S1/S2 were silent or to corroborate."),
    ("S4", "<b>Kew, Plants of the World Online</b> (POWO)", "B",
     "Direct fetch <b>refused (HTTP 403)</b>; claims come from a search-result summary only",
     "Introduced in many European countries (e.g. Germany, France, Italy) and in Japan and "
     "Korea.",
     "World map subtitle ('outside North America every population is introduced'); Europe/East "
     "Asia caption",
     "I never saw POWO's area list, so nothing is drawn from it; country shading is GBIF only."),
    ("S5", "<b>Euro+Med PlantBase</b>", "B", "Search-result summary",
     "Country-level status in Europe: naturalised in many countries; uncertain degree of "
     "naturalisation in Great Britain, Latvia, Romania; introduced in Norway and Spain.",
     "Europe caption ('listed as naturalised in many countries')",
     "No per-country status layer was made from it."),
    ("S6", "<b>Wikispecies</b>; <b>iNaturalist</b> guide page", "C", "Search-result summary",
     "Introduced to Japan, Korea, Taiwan, Vietnam and other Asian and European countries; "
     "naturalised in western Europe and China.",
     "Corroborates the East Asia statements", "Lists, no dates or status detail."),
    ("S7", "<b>GBIF occurrence API</b>, backbone taxon 3114225 (<i>Rudbeckia hirta</i> L.)", "A",
     "Queried directly on 2026-10-30: a country facet (69,581 georeferenced records, 55 "
     "countries) and every georeferenced record without geospatial issues outside the US and "
     "Canada: <b>9,530 records, 53 countries</b>",
     "Records per country; each record's coordinates, year, basis of record, establishment "
     "means. Top: Germany 1,453, Belgium 1,278, Sweden 1,260, Netherlands 964, Russia 910, "
     "Switzerland 475, UK 411, France 358, Austria 350, South Korea 337, Belarus 290, Japan 281.",
     "World map country shading and dots; Europe (8,583) and East Asia (741) hexbins; page 4 "
     "numbers. Saved: metadata/gbif_rhirta_outside_us_ca.csv",
     "Not a DOI-citable GBIF download. 87% human observations (mostly iNaturalist), 11% "
     "specimens. 92% have no establishment status; 2 are flagged 'native reintroduced' "
     "(implausible). Cultivated plants and escapes are mixed. Effort follows observers. Years "
     "1831-2026, median 2021."),
    ("S8", "<b>iNaturalist open data</b>, local photo index (inat_photos.parquet)", "A",
     "Queried locally for research-grade photos identified below species",
     "Photos with coordinates: var. pulcherrima 1,053; var. floridana 397; var. hirta 53; "
     "var. angustifolia 23.",
     "North America map: the coloured variety points",
     "Community IDs to variety are rare and uneven; angustifolia and hirta are sparse. One row "
     "per photo, so multi-photo observations count more than once."),
    ("S9", "<b>This study</b>: GPU charisma run, relative bullseye size per photo "
     "(maps/rbs_by_photo.csv)", "A", "Own data",
     "33,615 North American photos. Half-degree cells with &ge; 15 photos where &ge; 20% have "
     "a bullseye; 61 flowers with RBS &ge; 50% (red covers most of the ray).",
     "North America map: hatched 'wild red-based bullseye' cells and the &times; 'likely "
     "cultivar' marks; page 4 rows 2 and 3",
     "Thresholds and the 'likely cultivar' reading are mine (D5, D6); no photo was checked "
     "against garden records."),
    ("S10", "<b>USDA GRIN</b> germplasm accession, wild Texas collection (TX-06)", "B",
     "Search-result summary of the accession page",
     "Rays yellow with a basal red-brown spot; heads to 10 cm.",
     "Page 4, row 2 (supports wild red-based forms in Texas)", "A single accession."),
    ("S11", "Government document describing var. pulcherrima as a weedy Great Plains variety "
     "(govdocs PDF in the search results)", "C",
     "Search-result summary; the exact document was not identified",
     "var. pulcherrima rays 'often purplish near the base'; the typical variety is an eastern "
     "wildflower.",
     "North America legend and page 4 ('often purplish near the ray base')",
     "Attribution uncertain; treat as unverified."),
    ("S12", "<b>Plant Care Today</b>, 'Gloriosa daisies: the beginnings'", "C",
     "Search-result summary",
     "Blakeslee doubled R. hirta chromosomes with colchicine from the 1930s; Burpee named it "
     "'Gloriosa Daisy' in the 1957 catalogue; Gloriosa Golden 1960; Gloriosa Double, AAS silver "
     "1961; 'Pinwheel' 1963.",
     "Stars at Smith College and Burpee; dates in the legend and page 4",
     "Popular history; dates not checked against Burpee catalogues."),
    ("S13", "<b>Palmer et al. 2009</b>, HortScience (crossability in Rudbeckia; NCSU PDF)", "B",
     "Search-result summary of the paper",
     "R. hirta cultivars crossed with R. fulgida, R. missouriensis, R. subtomentosa: of 844 "
     "seedlings only one confirmed interspecific hybrid (R. subtomentosa × R. hirta); ploidy "
     "differences; pseudogamy produces maternal-looking seedlings.",
     "Page 4, row 5 (hybrids)", "Paper not read in full."),
    ("S14", "<b>USPTO plant patents</b>, 'ET RDB' series (e.g. PP34593, PP34655, PP32892)", "B",
     "Search-result listing", "Parentage listed as Echinacea purpurea × Rudbeckia hirta.",
     "Page 4, row 5", "Patent texts not read."),
    ("S15", "<b>Monrovia</b>, <b>Laidback Gardener</b>, <b>Nursery Management</b> (Echibeckia "
     "'Summerina')", "C", "Search-result summaries",
     "'Summerina Orange' = R. hirta × E. purpurea, 'ET-RDB 01', PP 25,221; exclusive to Pacific "
     "Plug &amp; Liner; hardy to zone 6; a blog reports Dutch origin (unconfirmed).",
     "Star at Pacific Plug &amp; Liner (Watsonville, CA); page 4 row 5",
     "The star marks the distributor, not a confirmed breeding site; the Dutch origin is not "
     "mapped."),
    ("S16", "<b>UW-Madison Extension</b>, <b>Mississippi State Extension</b>, "
     "<b>Horticulture</b> magazine, retailer pages", "C", "Search-result summaries",
     "'Prairie Sun': Benary, AAS 2003 and Fleuroselect Gold. 'Cherokee Sunset': Thompson &amp; "
     "Morgan, AAS 2002. 'Denver Daisy': Benary, introduced through Plant Select for Denver's "
     "150th anniversary.",
     "Stars at Benary, Thompson &amp; Morgan and Denver Botanic Gardens; page 4 row 3",
     "'Denver Daisy' date given as 2009 by one source and 2013 by another; not resolved."),
]

DECISIONS = [
    ("D1", "Native range", "Lower 48 US states plus Canadian provinces (not the territories), "
     "clipped at 56°N.", "S1, S3 describe the range in words; the exact edge is mine."),
    ("D2", "var. hirta outline", "Union of 15 core states: PA, NJ, DE, MD, DC, VA, WV, NC, SC, "
     "GA, TN, KY, OH, IN, IL. 'Sparingly north to Maine' is not drawn.",
     "S1, S2. Whole states overstate the range (e.g. coastal Georgia)."),
    ("D3", "var. floridana outline", "Florida south of 29.6°N.",
     "S1 says central, S2 southern Florida; the cut-off covers both."),
    ("D4", "var. angustifolia outline", "A hand-placed inland edge of the Gulf and south Atlantic "
     "coastal plain (about 18 vertices), intersected with SC, GA, FL, AL, MS, LA and TX.",
     "S2 (Gulf Coastal Plain), S3 (SC to TX). The vertices are my approximation of the fall "
     "line, not a published boundary."),
    ("D5", "Wild red-based bullseye cells", "Half-degree cells with &ge; 15 photos and "
     "&ge; 20% of them at RBS &ge; 5%.", "S9. Other cell sizes or thresholds would move the "
     "edges (not tested here); the same Texas hotspot shows on the bullseye hexagon maps, "
     "which use a different grid."),
    ("D6", "'Likely cultivar'", "RBS &ge; 50%: red covers most of the ray.",
     "S9 plus S3/S12 (Gloriosa daisies include red and bicolored forms). Some wild plants may "
     "be this red; none were checked against garden records."),
    ("D7", "Breeding-centre locations", "Town-level coordinates of each organisation: "
     "Northampton MA, Warminster PA, Hann. Münden (DE), Ipswich (UK), Denver CO, Watsonville CA.",
     "S12, S15, S16. Placed at headquarters, not at specific trial grounds."),
    ("D8", "Map styling", "Country shading on a log scale; France's French Guiana excluded "
     "from France's colour; reference blue sequential ramp; three categorical colours for the "
     "varieties.", "Presentation only."),
]

LAYERS = [
    ("World", "Country shading (records per country)", "S7", "A"),
    ("World", "Black dots (individual records)", "S7", "A"),
    ("World", "Hatched native range", "S1, S3 + D1", "B/D"),
    ("World", "Orange stars (breeding / introduction)", "S12, S15, S16 + D7", "C/D"),
    ("World", "Subtitle: everything outside NA is introduced", "S4, S5, S6, S3", "B/C"),
    ("North America", "Native range fill (var. pulcherrima nearly throughout)", "S1, S3 + D1", "B/D"),
    ("North America", "var. hirta outline", "S1, S2 + D2", "B/D"),
    ("North America", "var. angustifolia outline", "S2, S3 + D4", "B/D"),
    ("North America", "var. floridana outline", "S1, S2 + D3", "B/D"),
    ("North America", "Coloured points (photos IDed to variety)", "S8", "A"),
    ("North America", "Hatched cells (wild red-based bullseye common)", "S9 + D5", "A/D"),
    ("North America", "× marks (red-rayed, likely cultivar)", "S9 + D6", "A/D"),
    ("North America", "Legend text 'often purplish near the ray base'", "S11", "C"),
    ("Europe / East Asia", "Hexbins of records", "S7", "A"),
    ("Europe / East Asia", "Captions on naturalised status", "S5, S4, S6", "B/C"),
    ("Page 4", "Row 1, wild regional varieties", "S1, S2, S3, S11", "B/C"),
    ("Page 4", "Row 2, wild red-based bullseyes", "S9, S10", "A/B"),
    ("Page 4", "Row 3, garden cultivars", "S3, S12, S16, S9", "C/A"),
    ("Page 4", "Row 4, introduced populations", "S7, S4, S5", "A/B"),
    ("Page 4", "Row 5, hybrids", "S13, S14, S15", "B/C"),
]

UNUSED = [
    ("Japanese search (Arage-hangonso = R. hirta, Hokkaido)", "Every result was about "
     "<i>R. laciniata</i> (O-hangonso), a different species and a designated invasive in Japan. "
     "Not used; whether "
     "<i>R. hirta</i> counts as invasive in Japan or Korea was not established."),
    ("GBIF 'cultivated' flag", "Only 8 records worldwide carry degreeOfEstablishment = cultivated, "
     "too few to separate garden plants from escapes. Not used."),
    ("'Cherry Brandy', 'Toto'", "Breeder and origin not found in the searches; listed as cultivar "
     "names only (S3), not mapped."),
    ("Echibeckia's reported Dutch origin", "A single blog's second-hand report; not mapped."),
    ("POWO's full native/introduced area list", "Page refused direct access (HTTP 403); the list "
     "was never seen in full."),
    ("Invasive status anywhere", "No source established invasiveness (as opposed to "
     "naturalisation) for R. hirta in any country; the maps say 'introduced' or 'naturalised' "
     "only."),
]

REFS = [
    ("S1", "Flora of North America, vol. 21, Rudbeckia hirta; text as reproduced by the "
     "Encyclopedia of Life", "https://api.eol.org/pages/467866/articles"),
    ("S2", "Compositae database, Rudbeckia hirta descriptions",
     "https://compositae.myspecies.info/taxonomy/term/1003/descriptions"),
    ("S3", "Wikipedia, 'Rudbeckia hirta'", "https://en.wikipedia.org/wiki/Rudbeckia_hirta"),
    ("S4", "Kew, Plants of the World Online",
     "https://powo.science.kew.org/taxon/urn:lsid:ipni.org:names:223740-2"),
    ("S5", "Euro+Med PlantBase",
     "https://europlusmed.org/cdm_dataportal/taxon/c7aa4620-1c81-4686-8fed-f8e30b133ccf"),
    ("S6", "Wikispecies; iNaturalist guide", "https://species.wikimedia.org/wiki/Rudbeckia_(Asteraceae) ; "
     "https://www.inaturalist.org/guide_taxa/870519"),
    ("S7", "GBIF occurrence search API, taxonKey 3114225 (accessed 2026-10-30)",
     "https://api.gbif.org/v1/occurrence/search?taxonKey=3114225 ; https://www.gbif.org/taxon/4TMB7"),
    ("S8", "iNaturalist open data (photos, observations, taxa), local build",
     "https://registry.opendata.aws/inaturalist-open-data/"),
    ("S9", "This study: charisma_gpu run and bullseye_maps.py", "rhirta_bullseye/maps/rbs_by_photo.csv"),
    ("S10", "USDA ARS GRIN-Global accession detail (wild Texas collection)",
     "https://npgstest.ars-grin.gov/gringlobal/accessiondetail?id=1897288"),
    ("S11", "Government document in search results (govdocs corpus)",
     "https://corpora.tika.apache.org/base/docs/govdocs1/021/021043.pdf"),
    ("S12", "Plant Care Today, 'Gloriosa daisies: the beginnings and here to stay'",
     "https://plantcaretoday.com/?p=95600"),
    ("S13", "Palmer et al. (2009). Crossability, cytogenetics, and reproductive pathways in "
     "Rudbeckia subgenus Rudbeckia. HortScience. (Title as given in the search results; authors "
     "and volume not checked.)",
     "https://mcilab.cals.ncsu.edu/wp-content/uploads/sites/69/2019/02/palmer-etal-2009.pdf"),
    ("S14", "USPTO plant patents, Rudbeckia 'ET RDB' series",
     "https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/PP34593 ; "
     "https://patents.google.com/patent/USPP34655"),
    ("S15", "Monrovia; Laidback Gardener (2015); Nursery Management",
     "https://monrovia.com/summerina-orange-echibeckia.html ; "
     "https://laidbackgardener.blog/2015/03/12/brand-new-perennial-for-your-summer-garden/ ; "
     "https://www.nurserymag.com/product/summerina-echibeckia-pacific-plug"),
    ("S16", "UW-Madison Extension; Mississippi State Extension P3506; Horticulture magazine",
     "https://hort.extension.wisc.edu/articles/prairie-sun-black-eyed-susan-rudbeckia-hirta ; "
     "https://extension.msstate.edu/sites/default/files/publications/P3506_web.pdf ; "
     "https://hortmag.com/plants-we-love-2/denver-daisy-brings-bright-color-for-summer"),
]


def build(out):
    P("Sources and how they were used", h1)
    P("Companion to <i>rhirta_variants_map.pdf</i>: where to expect Rudbeckia hirta varieties, "
      "cultivars and hybrids &nbsp;|&nbsp; prepared 2026-10-30", sub)

    P("1. How to read this", h2)
    P("Every element of the variants map comes from one of four kinds of evidence. Each source "
      "below carries a grade for how directly I saw it. That is a statement about my access, "
      "not about the source's quality: Flora of North America is authoritative, but I only saw "
      "a summary of it.")
    table(["Grade", "Meaning", "Sources"], [
        ["A", "Data I retrieved and processed myself; numbers can be re-run from the saved files",
         "S7 GBIF, S8 iNaturalist photo index, S9 this study"],
        ["B", "Authoritative source (flora, database, peer-reviewed paper, patent, genebank) seen "
              "only through a search-result summary or a reproduction",
         "S1 FNA, S2 Compositae db, S4 POWO, S5 Euro+Med, S10 GRIN, S13 Palmer et al., S14 USPTO"],
        ["C", "Secondary or popular source (encyclopedia, extension sheet, magazine, retailer, "
              "blog), seen through a search-result summary",
         "S3, S6, S11, S12, S15, S16"],
        ["D", "My own decision: an outline, threshold, location or interpretation",
         "D1-D8 (section 4)"],
    ], [0.6, 5.0, 3.6], grade_col=0)
    P("<b>How the literature was gathered.</b> I ran eight web searches. The search tool returns "
      "a short summary of each result page, written by the tool, plus the links. With one "
      "exception I did not open the pages myself. That exception was Kew POWO, and the site "
      "refused it (HTTP 403). So every grade-B and grade-C claim here comes from a summary. "
      "Where two summaries disagreed, I say so and say what I drew. Before citing any of them "
      "in a publication, open the original.")
    P("<b>Where the confidence lies.</b>")
    B("<b>Strongest:</b> where the plant has been recorded outside North America (GBIF, 9,530 "
      "records in 53 countries), and where red-based bullseyes are common (this study).")
    B("<b>Medium:</b> the variety ranges. The descriptions come from Perdue (1957) via Flora of "
      "North America, but the outlines on the map are mine.")
    B("<b>Weakest:</b> breeding locations, dates and hybrid origins. These come from popular "
      "sources, one date conflicts, and one origin is unconfirmed.")

    story.append(PageBreak())
    P("2. The sources", h2)
    table(["ID", "Source", "Gr.", "How I accessed it", "What I took from it", "Where it is used",
           "Limits"],
          [list(s) for s in SOURCES],
          [0.38, 1.45, 0.38, 1.25, 2.55, 1.6, 1.85], grade_col=2)

    story.append(PageBreak())
    P("3. Each map element and where it comes from", h2)
    table(["Map", "Element", "Sources and decisions", "Grade"],
          [list(r) for r in LAYERS], [1.5, 4.2, 2.4, 1.0], grade_col=3)

    P("4. Decisions I made", h2)
    P("These are not in any source. They are the choices needed to turn prose ranges and "
      "numbers into shapes on a map. Change any of them, and that part of the map changes "
      "with it.", body)
    table(["ID", "What", "Choice", "Basis and caveat"], [list(d) for d in DECISIONS],
          [0.4, 1.5, 3.9, 3.6])

    story.append(PageBreak())
    P("5. Conflicts, gaps, and what was not used", h2)
    table(["Source / claim", "Conflict"], [
        ["var. floridana range", "S1: central Florida; S2: southern Florida. Drawn as Florida "
                                 "south of 29.6°N to cover both (D3)."],
        ["'Denver Daisy' introduction year", "2009 in one source, 2013 in another, both tied to "
                                             "Denver's 150th anniversary. Not resolved; the map "
                                             "gives no year."],
        ["'Denver Daisy' parentage", "One source: native R. hirta × 'Prairie Sun'; a patent "
                                     "describes it as distinct from 'Prairie Sun'. Not used."],
        ["RHS Award of Garden Merit", "Wikipedia and a mirror disagree on whether both 'Indian "
                                      "Summer' and 'Toto' hold it. Not used."],
    ], [2.4, 6.8])
    table(["Not used", "Why"], [list(u) for u in UNUSED], [2.4, 6.8])

    P("6. Reproducing it", h2)
    P("Everything grade A can be rebuilt from saved files: "
      "<font face='Courier'>metadata/gbif_rhirta_outside_us_ca.csv</font> (S7), the local "
      "iNaturalist photo index (S8), and <font face='Courier'>maps/rbs_by_photo.csv</font> (S9). "
      "Run <font face='Courier'>charisma_gpu/variant_map.py</font> on remote_sam3's "
      "<i>charisma-gpu</i> branch. To cite S7 properly, request a GBIF download for taxonKey "
      "3114225, which gives a DOI. The API query used here does not.", body)

    story.append(PageBreak())
    P("7. References", h2)
    table(["ID", "Reference", "Link"], [[r[0], r[1], r[2]] for r in REFS], [0.45, 4.1, 4.65])
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.lightgrey, spaceBefore=6,
                            spaceAfter=6))
    P("Generated by charisma_gpu/variant_sources_pdf.py.", note)

    doc = SimpleDocTemplate(str(out), pagesize=landscape(letter), leftMargin=0.6 * inch,
                            rightMargin=0.6 * inch, topMargin=0.55 * inch,
                            bottomMargin=0.55 * inch,
                            title="R. hirta variants map: sources and how they were used",
                            author="remote_sam3")
    doc.build(story)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    build(Path(a.root) / "maps" / "variants" / "rhirta_variants_sources.pdf")
