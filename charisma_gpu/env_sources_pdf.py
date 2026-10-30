#!/usr/bin/env python3
"""
Data sources, methods and caveats for the environment and urban-rural
analysis of relative bullseye size (env_urban_maps.py).

    python charisma_gpu/env_sources_pdf.py --root C:/Users/nevyn/Documents/rhirta_bullseye

Writes root/maps/environment/rhirta_environment_sources.pdf.
"""

import argparse
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

NAVY = colors.HexColor("#2B3A55")
LIGHT = colors.HexColor("#EAEEF5")
ss = getSampleStyleSheet()
h1 = ParagraphStyle("h1", parent=ss["Heading1"], textColor=NAVY, fontSize=19, spaceAfter=3)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=NAVY, fontSize=13, spaceBefore=9,
                    spaceAfter=4)
sub = ParagraphStyle("sub", parent=ss["Normal"], textColor=colors.grey, fontSize=9.5,
                     spaceAfter=8)
body = ParagraphStyle("body", parent=ss["Normal"], fontSize=9.8, leading=13.4, spaceAfter=5)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=13, bulletIndent=1, spaceAfter=3)
note = ParagraphStyle("note", parent=ss["Normal"], fontSize=8, textColor=colors.grey,
                      leading=10.5)
cell = ParagraphStyle("cell", parent=ss["Normal"], fontSize=8.2, leading=10.4)
cellb = ParagraphStyle("cellb", parent=cell, textColor=colors.white, fontName="Helvetica-Bold")


def table(rows, widths):
    data = [[Paragraph(c, cellb if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


def build(root):
    env = root / "maps" / "environment"
    d = pd.read_csv(env / "rbs_us_covariates.csv")
    summ = pd.read_csv(env / "urban_rural_summary.csv")
    vif = pd.read_csv(env / "vif.csv")
    n_us = len(d)
    full = d.dropna(subset=["temp", "precip", "pseas", "srad", "elev", "uvb", "uvb_max",
                            "settlement"])
    s = []
    P = lambda t, st=body: s.append(Paragraph(t, st))
    B = lambda t: s.append(Paragraph(t, bullet, bulletText="•"))
    gap = lambda h=6: s.append(Spacer(1, h))

    P("Environment and urbanisation: data sources and methods", h1)
    P("Companion to <i>rhirta_bullseye_environment.pdf</i> and <i>rhirta_bullseye_urban_rural.pdf</i> "
      "&nbsp;|&nbsp; contiguous US &nbsp;|&nbsp; prepared 2026-10-30", sub)

    P("1. Data sources", h2)
    P("All four covariate datasets are standard, citable and openly downloadable, and each is "
      "widely used in published macroecology. I downloaded every one from its official host, "
      "and the files are kept in <font face='Courier'>rhirta_bullseye/environment_data/</font>.")
    s.append(table([
        ["Dataset", "What I used", "Resolution, period", "Citation", "Access and licence"],
        ["<b>WorldClim 2.1</b>, historical climate",
         "bio1 annual mean temperature; bio5 max temperature of the warmest month; bio12 annual "
         "precipitation; bio15 precipitation seasonality; monthly solar radiation (June–August "
         "mean); elevation",
         "5 arc-min (~9 km); 1970–2000 averages",
         "Fick, S.E. &amp; Hijmans, R.J. (2017) WorldClim 2: new 1-km spatial resolution climate "
         "surfaces for global land areas. <i>International Journal of Climatology</i> 37(12): "
         "4302–4315.",
         "geodata.ucdavis.edu (linked from worldclim.org), files wc2.1_5m_bio, _srad, _elev. "
         "Licence: an aggregator lists CC BY 4.0; <b>not confirmed on worldclim.org</b>."],
        ["<b>glUV</b>, global UV-B",
         "UVB1 annual mean UV-B; UVB3 mean UV-B of the highest month",
         "15 arc-min (~25 km); 2004–2013 (Aura-OMI erythemal daily dose)",
         "Beckmann, M. et al. (2014) glUV: a global UV-B radiation data set for macroecological "
         "studies. <i>Methods in Ecology and Evolution</i> 5: 372–383. "
         "doi:10.1111/2041-210X.12168",
         "ufz.de/gluv, ASCII grids, v1.0. The download page <b>states no licence</b>; cite the "
         "paper, and ask the UFZ contact before redistributing."],
        ["<b>GHS-SMOD R2023A</b>, degree of urbanisation",
         "Settlement class of each photo's 1 km cell, epoch 2020 (level 2: 7 classes)",
         "1 km, Mollweide (ESRI:54009); epoch 2020",
         "Schiavina, M., Melchiorri, M., Pesaresi, M. (2023) GHS-SMOD R2023A – GHS settlement "
         "layers, application of the Degree of Urbanisation methodology (stage I) to GHS-POP "
         "R2023A and GHS-BUILT-S R2023A, multitemporal (1975–2030). European Commission, JRC. "
         "doi:10.2905/A0DF7A6F-49DE-46EA-9BDE-563437A6E2BA",
         "jeodpp.jrc.ec.europa.eu, file GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V2_0. The data "
         "package states: free of charge, reuse authorised provided the source is acknowledged "
         "(Decision 2011/833/EU)."],
        ["<b>Degree of Urbanisation method</b>", "Definitions of the classes",
         "—",
         "European Commission &amp; Statistical Office of the EU (2021) <i>Applying the Degree of "
         "Urbanisation — a methodological manual to define cities, towns and rural areas for "
         "international comparisons</i>.",
         "Cited in the GHSL data package as the concept and method reference."],
        ["<b>Natural Earth</b> 1:50m", "US state boundaries (contiguous-US filter, Texas split, "
         "map outlines)", "1:50,000,000", "Natural Earth, naturalearthdata.com (public domain).",
         "Via cartopy."],
        ["<b>iNaturalist</b> photos (this study)", "Coordinates of each photo; relative bullseye "
         "size from the GPU charisma run", f"{n_us:,} US photos", "iNaturalist open data "
         "(registry.opendata.aws/inaturalist-open-data).",
         "Research-grade records. Per-photo CC licences were not checked."],
    ], [1.05, 1.55, 1.0, 2.15, 1.55]))
    gap(4)
    P("Why these: WorldClim and glUV are the climate and UV-B layers used in comparable studies "
      "of floral UV pigmentation. In particular, Koski &amp; Ashman (2015, <i>Nature Plants</i> "
      "1: 14007) found UV-B irradiance the best predictor of floral UV bullseye size in "
      "<i>Argentina anserina</i>, followed by temperature but not rainfall. GHS-SMOD applies the "
      "UN-endorsed Degree of Urbanisation definitions, so 'city / suburban / rural' has a "
      "published, reproducible meaning.", body)

    P("2. Methods", h2)
    B("<b>Response.</b> Relative bullseye size (RBS) = red on the petals ÷ (red on the petals + "
      "yellow ray), per photo, from the GPU charisma run with the disk step. 'Has a bullseye' = "
      "RBS ≥ 5%. 'Red-rayed' = RBS ≥ 50%, which I read as mostly garden cultivars (an "
      "inference, not verified).")
    B(f"<b>Sample.</b> Photos inside the contiguous US (Natural Earth): {n_us:,}. With every "
      f"covariate and a settlement class: {len(full):,}, of which "
      f"{int(full.has_bullseye.sum()):,} have a bullseye.")
    B("<b>Extraction.</b> Each covariate was sampled at the photo's coordinates (rasterio, nearest "
      "cell). GHS-SMOD is in Mollweide, so the coordinates were reprojected before sampling. "
      "Summer solar radiation is the mean of the June, July and August layers.")
    B("<b>Settlement classes.</b> City = class 30 (urban centre). Suburban / town = 23 dense, "
      "22 semi-dense urban cluster and 21 suburban or peri-urban. Rural = 13 rural cluster, "
      "12 low-density and 11 very-low-density rural. Class definitions were checked against the "
      "GHSL data package and the .clr file shipped with the raster.")
    B("<b>Gradients.</b> Decile bins of each covariate, with bootstrap 95% intervals (300 "
      "resamples). Shown for the whole US and without Texas.")
    vtxt = ", ".join(f"{r.variable} {r.VIF:.1f}" for r in vif.itertuples())
    B("<b>Models.</b> Covariates were standardised. Presence used a binomial GLM; size used a "
      "Gaussian GLM on the log-odds of RBS, bullseye flowers only (statsmodels). To limit "
      "collinearity, these covariates were used: temperature, precipitation, precipitation "
      f"seasonality, summer solar radiation, elevation and highest-month UV-B (VIF: {vtxt}). "
      "Each model was fitted for the whole US and without Texas.")

    s.append(PageBreak())
    P("3. What the figures show", h2)
    w = summ.set_index(["region", "settlement"])
    B("<b>Presence follows warm, seasonally dry, high-UV, low-elevation places, but that is "
      "Texas.</b> On the binned plots, prevalence rises to ~25% in the warmest, sunniest, "
      "highest-UV deciles of the whole US. Without Texas every gradient is flat at 3–5%.")
    B("<b>In the joint model</b> (whole US), presence rises with temperature and falls with "
      "precipitation, precipitation seasonality and elevation. Without Texas, the temperature "
      "and precipitation effects shrink to a quarter to a third. UV-B has no clear independent effect "
      "after temperature, and its sign flips between subsets.")
    B("<b>Size when present barely tracks climate.</b> The only interval clear of zero is "
      "summer solar radiation (negative, whole US), and that does not hold without Texas.")
    B("<b>This is the opposite of the UV-protection prediction for size.</b> Koski &amp; Ashman "
      "found bigger UV bullseyes under more UV-B. Here, visible-red bullseyes are <i>larger</i> "
      "in the cooler, lower-UV north. Note that a visible-red bullseye is not the same trait as "
      "a UV-absorbing bullseye; R. hirta's UV pattern was not measured.")
    B(f"<b>Urban vs rural.</b> Across the whole US, presence is similar in cities "
      f"({w.loc[('Whole US', 'City'), 'prevalence']:.1%}), suburbs "
      f"({w.loc[('Whole US', 'Suburban / town'), 'prevalence']:.1%}) and the countryside "
      f"({w.loc[('Whole US', 'Rural'), 'prevalence']:.1%}). But the two regions differ. In "
      f"Texas, bullseyes are commonest in the countryside "
      f"({w.loc[('Texas', 'Rural'), 'prevalence']:.1%} vs "
      f"{w.loc[('Texas', 'City'), 'prevalence']:.1%} in cities), consistent with wild variation. "
      f"In the rest of the US, they are commoner in cities "
      f"({w.loc[('Rest of US', 'City'), 'prevalence']:.1%}) and suburbs "
      f"({w.loc[('Rest of US', 'Suburban / town'), 'prevalence']:.1%}) than in the countryside "
      f"({w.loc[('Rest of US', 'Rural'), 'prevalence']:.1%}), consistent with planted garden "
      "forms.")
    B(f"<b>Red-rayed flowers</b> are rare everywhere (0.1–0.3% of photos). Outside Texas they "
      f"lean urban: cities {w.loc[('Rest of US', 'City'), 'red_rayed']:.2%} and suburbs "
      f"{w.loc[('Rest of US', 'Suburban / town'), 'red_rayed']:.2%}, against rural "
      f"{w.loc[('Rest of US', 'Rural'), 'red_rayed']:.2%}. The intervals overlap, so this is "
      "suggestive only.")

    P("4. Caveats a reviewer will raise", h2)
    B("<b>Spatial autocorrelation.</b> Photos cluster in space (Moran's I is significant out to "
      "about 300 km; see the correlogram), so GLM intervals are too narrow. For publication, "
      "use a spatial model: a GLMM with a spatial random effect, INLA/SPDE, or spatially "
      "blocked cross-validation.")
    B("<b>Collinearity.</b> Temperature, solar radiation and UV-B are strongly correlated "
      "(correlation matrix, VIF). Their separate effects cannot be fully separated with "
      "observational data across the US.")
    B("<b>Texas leverage.</b> Texas holds a large share of the bullseye photos. Results should "
      "always be shown with and without it, as here.")
    B("<b>Sampling bias.</b> iNaturalist effort follows people and roads, so urban and roadside "
      "plants are over-represented. One photo per row also lets multi-photo observations count "
      "more than once.")
    B("<b>Scale mismatch.</b> Covariates are 1–25 km cell averages. iNaturalist locations can be "
      "obscured or imprecise (up to tens of km), which blurs the 1 km settlement class most of "
      "all.")
    B("<b>Time mismatch.</b> The climate normals cover 1970–2000 and UV-B 2004–2013. Photos are "
      "mostly recent iNaturalist uploads, and settlement is the 2020 epoch.")
    B("<b>Trait measurement.</b> RBS is measured from phone photos under uncontrolled light. "
      "Garden cultivars are not flagged, and no per-photo cultivated status was used. "
      "iNaturalist's 'captive/cultivated' field would help.")

    P("5. Files", h2)
    P("<font face='Courier'>maps/environment/rbs_us_covariates.csv</font>: every US photo with "
      "its covariates and settlement class. Also <font face='Courier'>model_coefficients.csv</font>, "
      "<font face='Courier'>vif.csv</font> and <font face='Courier'>urban_rural_summary.csv</font>. "
      "Code: <font face='Courier'>charisma_gpu/env_urban_maps.py</font> on remote_sam3's "
      "<i>charisma-gpu</i> branch.", body)
    P("6. References", h2)
    for r in [
        "Beckmann, M. et al. (2014) glUV: a global UV-B radiation data set for macroecological "
        "studies. "
        "<i>Methods in Ecology and Evolution</i> 5: 372–383.",
        "European Commission &amp; Statistical Office of the European Union (2021) Applying the "
        "Degree of Urbanisation — a methodological manual. Publications Office of the EU.",
        "Fick, S.E. &amp; Hijmans, R.J. (2017) WorldClim 2. <i>International Journal of "
        "Climatology</i> 37: 4302–4315.",
        "Koski, M.H. &amp; Ashman, T.-L. (2015) Floral pigmentation patterns provide an example "
        "of Gloger's rule in plants. <i>Nature Plants</i> 1: 14007.",
        "Schiavina, M., Melchiorri, M. &amp; Pesaresi, M. (2023) GHS-SMOD R2023A. European "
        "Commission, JRC. doi:10.2905/A0DF7A6F-49DE-46EA-9BDE-563437A6E2BA.",
        "Natural Earth (naturalearthdata.com); iNaturalist open data (AWS Open Data Registry); "
        "software: rasterio, statsmodels, cartopy, matplotlib."]:
        s.append(Paragraph(r, bullet, bulletText="•"))
    gap(4)
    P("Verification note: the WorldClim, glUV and GHS-SMOD citations come from the providers' "
      "pages or files (GHS-SMOD from its data package). The Koski "
      "&amp; Ashman details come from search summaries. Check them against the papers before "
      "submission.", note)

    out = env / "rhirta_environment_sources.pdf"
    SimpleDocTemplate(str(out), pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                      topMargin=0.55 * inch, bottomMargin=0.55 * inch,
                      title="Environment and urbanisation: data sources and methods").build(s)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    build(Path(ap.parse_args().root))
