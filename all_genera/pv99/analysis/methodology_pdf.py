#!/usr/bin/env python3
"""
The methodology PDF: how each family's flower types were decided, built from
the analysis outputs themselves (every number in it is read from the CSVs).

    python all_genera/pv99/analysis/methodology_pdf.py

Writes export/pv99/analysis/flower_types_methodology.pdf.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, NextPageTemplate, PageBreak,
                                PageTemplate, Frame, BaseDocTemplate, Paragraph, Spacer,
                                Table, TableStyle)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import family_types as ft  # noqa: E402
import top_segments as ts  # noqa: E402
from morph_questions import PREAMBLE, QUESTIONS, prompt_text  # noqa: E402
from morph_classify import MODEL  # noqa: E402

INK = colors.HexColor(common.INK)
INK_2 = colors.HexColor(common.INK_2)
RULE = colors.HexColor(common.GRID)
HEAD_BG = colors.HexColor("#eef3fb")

GLOSSARY = [
    ("Flowering unit", [
        ("Solitary flower", "A flower borne singly on its own stalk (pedicel or peduncle), "
         "not grouped with others into a compact inflorescence."),
        ("Raceme / spike", "An unbranched indeterminate inflorescence: flowers along a single "
         "axis, stalked in a raceme (Lupinus, Digitalis), sessile in a spike (Plantago, "
         "Lavandula)."),
        ("Panicle / cyme / corymb", "Branched inflorescences: a panicle is a branched raceme; "
         "a cyme is determinate, each axis ending in a flower (Silene, Myosotis's coiled "
         "scorpioid cyme); a corymb is flat-topped with pedicels of unequal length."),
        ("Umbel", "An inflorescence whose pedicels all arise from one point, like umbrella "
         "ribs; compound in most Apiaceae, simple in Allium and Asclepias."),
        ("Capitulum (head)", "A dense head of sessile florets on a common receptacle, "
         "surrounded by an involucre of bracts (phyllaries), functioning as a single "
         "flower (a pseudanthium); diagnostic of Asteraceae, also Dipsacaceae."),
        ("Spadix and spathe", "A spike of minute flowers on a thick fleshy axis (spadix), "
         "subtended or enclosed by one large, often coloured bract (spathe); diagnostic of "
         "Araceae."),
        ("Spikelet", "The basic inflorescence unit of grasses and sedges: florets enclosed "
         "by chaffy bracts (glumes, lemma and palea) on a small axis (rachilla); no petals."),
        ("Catkin (ament)", "A pendulous or erect spike of small, unisexual, petal-less "
         "flowers, usually shed as a unit (Salix, Betula, Quercus)."),
        ("Cyathium", "The pseudanthium of Euphorbia: a cup-shaped involucre bearing nectar "
         "glands (often with petal-like appendages) that encloses one naked female flower "
         "and several single-stamen male flowers."),
    ]),
    ("Symmetry", [
        ("Actinomorphic (radial)", "Divisible into mirror halves along three or more planes "
         "through the centre (Ranunculus, Lilium, Rosa)."),
        ("Zygomorphic (bilateral)", "Divisible into mirror halves along exactly one plane, "
         "usually the median one (Pisum, orchids, Viola, Salvia)."),
        ("Disymmetric (biradial)", "Exactly two planes of symmetry at right angles "
         "(Dicentra; technically the cruciform Brassicaceae flower)."),
        ("Asymmetric", "No plane of symmetry (Canna, Centranthus)."),
    ]),
    ("Showy parts (perianth and display)", [
        ("Petals", "The inner perianth whorl (corolla), coloured and distinct from an outer "
         "green calyx of sepals."),
        ("Tepals", "Perianth members not differentiated into sepals and petals, all "
         "petal-like (Lilium, Tulipa, most petaloid monocots, Magnolia)."),
        ("Petaloid sepals or bracts", "Display by organs other than petals: coloured sepals "
         "(Clematis, Anemone, Aquilegia's outer whorl) or bracts subtending the flowers "
         "(Bougainvillea, Euphorbia pulcherrima, Cornus florida, Castilleja, bromeliads)."),
        ("Staminal display", "Display by many long, coloured stamens, the petals small or "
         "caducous (Callistemon, Eucalyptus, mimosoid legumes)."),
        ("Reduced (anemophilous)", "Small greenish or brownish flowers with the perianth "
         "reduced or absent, usually wind pollinated (grasses, sedges, catkin-bearers)."),
    ]),
    ("Petal fusion", [
        ("Polypetalous (choripetalous)", "Petals free from one another to the base."),
        ("Sympetalous (gamopetalous)", "Petals fused at least at the base into a tube, cup or "
         "bell, the free tips forming lobes."),
        ("Apetalous", "Petals (or tepals) absent."),
    ]),
    ("Flower form (corolla shape)", [
        ("Rotate / open", "Flat, wheel-, dish-, saucer- or star-shaped, with little or no "
         "tube; here also includes bowl- or cup-shaped (Solanum, Ranunculus, Rosa)."),
        ("Campanulate", "Bell-shaped: a broad tube flaring gradually to the lobes "
         "(Campanula)."),
        ("Urceolate", "Urn-shaped: inflated and contracted at the mouth (Vaccinium, Erica, "
         "Arbutus)."),
        ("Tubular / funnelform", "A long tube, cylindrical or widening gradually into a "
         "trumpet (Ipomoea, Fuchsia, Nicotiana)."),
        ("Salverform (hypocrateriform)", "A slender tube abruptly expanded into a flat limb "
         "at right angles to it (Phlox, Vinca, Primula)."),
        ("Bilabiate", "Two-lipped: the lobes grouped into an upper and a lower lip (Lamiaceae, "
         "Antirrhinum, Acanthaceae); personate if the lower lip closes the throat."),
        ("Papilionaceous", "The pea flower: an upper banner (standard), two lateral wings "
         "and a keel of the two lower petals, connate at their tips."),
        ("Orchid", "Three sepals, two lateral petals and a median petal modified into a lip "
         "(labellum), with stamens and style fused into a column."),
        ("Cruciform", "Four free petals in the form of a cross (Brassicaceae)."),
    ]),
    ("Accessory structures", [
        ("Spur", "A hollow, tubular, usually nectariferous extension of a petal or sepal "
         "(Aquilegia, Delphinium, Tropaeolum, Viola, Linaria, many orchids)."),
        ("Corona", "An additional ring, cup or fringe of tissue between the perianth and "
         "the stamens (Narcissus, Passiflora, Asclepias)."),
        ("Hood or galea", "A helmet formed by an upper sepal or petal (Aconitum, "
         "Pedicularis, many Lamiaceae upper lips, Pterostylis)."),
    ]),
    ("Merosity", [
        ("Merosity", "The number of parts in a floral whorl. Trimerous (3 or 6) is typical "
         "of monocots and magnoliids, tetramerous (4 or 8) and pentamerous (5 or 10) of "
         "eudicots; 'many' covers flowers with indefinite, numerous parts (Cactaceae, "
         "Aizoaceae staminodes, doubled forms)."),
    ]),
    ("Capitulum type (Asteraceae)", [
        ("Radiate", "Strap-shaped (ligulate) ray florets around a centre of tubular disc "
         "florets (Helianthus, Bellis)."),
        ("Discoid", "Disc florets only, no rays (Cirsium, Tanacetum)."),
        ("Ligulate", "Only strap-shaped florets, each with five teeth at the tip "
         "(Cichorieae: Taraxacum, Cichorium)."),
    ]),
]

REFERENCES = [
    "Judd, W. S., Campbell, C. S., Kellogg, E. A., Stevens, P. F. and Donoghue, M. J. (2016). "
    "Plant Systematics: A Phylogenetic Approach, 4th ed. Sinauer.",
    "Simpson, M. G. (2019). Plant Systematics, 3rd ed. Academic Press (ch. 9, Plant "
    "morphology: flowers and inflorescences).",
    "Christenhusz, M. J. M., Fay, M. F. and Chase, M. W. (2017). Plants of the World: An "
    "Illustrated Encyclopedia of Vascular Plants. Kew Publishing / University of Chicago Press.",
    "Heywood, V. H., Brummitt, R. K., Culham, A. and Seberg, O. (2007). Flowering Plant "
    "Families of the World. Kew Publishing.",
    "Stevens, P. F. (2001 onwards). Angiosperm Phylogeny Website, version 14. "
    "http://www.mobot.org/MOBOT/research/APweb/",
    "Angiosperm Phylogeny Group (2016). An update of the APG classification for the orders "
    "and families of flowering plants: APG IV. Botanical Journal of the Linnean Society 181: "
    "1-20.",
    "Endress, P. K. (1999). Symmetry in flowers: diversity and evolution. International "
    "Journal of Plant Sciences 160 (S6): S3-S23.",
    "Carion, N., et al. (2025). SAM 3: Segment Anything with Concepts. Meta AI.",
    "Qwen Team (2025). Qwen3-VL technical report. Alibaba Cloud.",
]


def styles():
    base = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=20, leading=24,
                                textColor=INK, alignment=TA_LEFT, spaceAfter=6),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=15, leading=19,
                             textColor=INK, spaceBefore=12, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=12, leading=15,
                             textColor=INK, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=9.5, leading=13,
                               textColor=INK, spaceAfter=5),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontSize=7.5, leading=9.5,
                                textColor=INK),
        "muted": ParagraphStyle("muted", parent=base["BodyText"], fontSize=8, leading=10.5,
                                textColor=INK_2),
        "mono": ParagraphStyle("mono", parent=base["Code"], fontSize=7.2, leading=9,
                               textColor=INK, backColor=colors.HexColor("#f4f4f2"),
                               borderPadding=4),
    }
    return s


def table(rows, widths, s, header=True, font=7.5):
    data = [[Paragraph(str(c), s["small"]) if not isinstance(c, Paragraph) else c
             for c in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), HEAD_BG)]
    t.setStyle(TableStyle(style))
    return t


def pic(path, max_w, max_h):
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    return Image(str(path), width=w * scale, height=h * scale)


def label_states(value):
    if not isinstance(value, str) or not value:
        return "(no reference)"
    if value == "-":
        return "n/a"
    return " | ".join(ft.LABELS.get(k, k) for k in value.split("|"))


def build(project, out_pdf):
    s = styles()
    A = common.analysis_dir(project)
    M = A / "morphology"
    meta = common.load_metadata(project, ["photo_id", "family", "genus", "num_masks",
                                          "conf_max"])
    segs = pd.read_csv(A / "top_segments" / "top_segments.csv")
    photos = pd.read_csv(M / "photos.csv")
    calls = pd.read_csv(M / "family_calls.csv")
    summ = pd.read_csv(M / "family_summary.csv")
    cal = pd.read_csv(M / "calibration.csv")
    reference = pd.read_csv(HERE / "family_reference.csv")
    order = segs.drop_duplicates("family").sort_values("family_rank")["family"].tolist()
    ok = photos[photos["error"].fillna("") == ""]

    n_photos = len(meta)
    n_masked = int((meta["num_masks"].fillna(0) > 0).sum())
    flips = {q["key"]: ok[f"{q['key']}__flip"].mean() for q in QUESTIONS}
    mass = {q["key"]: ok[f"{q['key']}__mass"].mean() for q in QUESTIONS}
    status_counts = calls["status"].value_counts()

    story = []
    P = lambda text, st="body": story.append(Paragraph(text, s[st]))  # noqa: E731

    # ── title and summary
    P("How the flower types of the 30 largest pv99 families were decided", "title")
    P(f"pv99 run (SAM3 'flower' segmentation of PhenoVision &ge; 0.99 photos) &middot; "
      f"generated {date.today().isoformat()} by all_genera/pv99/analysis/methodology_pdf.py",
      "muted")
    P("1. What was asked and what this document covers", "h1")
    P(f"The pv99 run segmented {n_photos:,} iNaturalist photos that PhenoVision scored "
      f"at least 0.99 for 'flower present' (up to 50 random photos per angiosperm genus) with "
      f"SAM3 and the text prompt 'flower'; {n_masked:,} ({100 * n_masked / n_photos:.1f}%) "
      f"got at least one mask at SAM3 score &ge; 0.5. The request was to take the 100 top "
      f"segments of each of the 30 most populous families and identify what types of flower "
      f"they show (zygomorphic, actinomorphic, spadix, and so on), using enough botanical "
      f"categories to describe the breadth of angiosperm variation, and to explain how each "
      f"decision was made.")
    P("Each family's call for each character combines three things, kept separate so that "
      "every call can be traced: (a) a <b>textbook expectation</b> for the family, written "
      "from standard systematic references; (b) <b>image evidence</b>: a vision-language "
      "model's answers to fixed botanical multiple-choice questions about each of the 100 "
      "segments; and (c) explicit <b>decision rules</b> that say when the images confirm, "
      "support, refine or contradict the textbook. Nothing is tuned by hand after the fact: "
      "the rules and thresholds are constants in family_types.py and are printed in section 6.")
    P(f"Outcome across {len(order)} families &times; {len(QUESTIONS)} characters: " +
      ", ".join(f"{v} {k}" for k, v in status_counts.items()) + ".")

    # ── data
    P("2. Which segments", "h1")
    P(f"<b>Families.</b> The 30 families with the most pv99 photos (photos per family, not "
      f"genera; the per-genus cap of 50 makes this close to a ranking by number of genera "
      f"with flowering photos). <b>Segment.</b> A photo's segment is its single most "
      f"confident SAM3 mask. <b>Top 100.</b> From each family's 200 photos with the highest "
      f"top-mask score, segments are taken in score order with at most "
      f"10 from any one genus, so that a family is not represented by a single showy genus; "
      f"if that leaves fewer than 100, the rest are filled in score order. The scores of the "
      f"chosen segments run from {segs['seg_score'].min():.3f} to "
      f"{segs['seg_score'].max():.3f} (median {segs['seg_score'].median():.3f}).")
    P(f"<b>What the model sees.</b> Not the cut-out alone but a context crop: the box around "
      f"the mask plus a margin of {ts.MARGIN:.0%} of its long side, never smaller than "
      f"{ts.MIN_FRAC:.0%} of the photo's short side, letterboxed to {ts.SIZE} px. The "
      f"context lets the model judge the inflorescence (an umbel, a spadix with its spathe, "
      f"a capitulum's involucre) that a tight cut-out would hide.")
    rows = [["#", "Family", "pv99 photos", "Segments", "Genera in top 100", "Score range"]]
    for fam in order:
        g = segs[segs["family"] == fam]
        rows.append([int(g["family_rank"].iloc[0]), fam, f"{int(g['family_photos'].iloc[0]):,}",
                     len(g), g["genus"].nunique(),
                     f"{g['seg_score'].min():.3f}-{g['seg_score'].max():.3f}"])
    story.append(table(rows, [0.3 * inch, 1.6 * inch, 0.9 * inch, 0.8 * inch, 1.2 * inch,
                              1.1 * inch], s))
    hist = A / "top_confidence_histogram.png"
    if hist.exists():
        story.append(Spacer(1, 8))
        story.append(pic(hist, 6.8 * inch, 3.6 * inch))
        P("Top mask confidence for every pv99 photo (the full run, not just the top "
          "segments).", "muted")

    # ── glossary
    story.append(PageBreak())
    P("3. The flower-type vocabulary", "h1")
    P("Eight characters, each a multiple-choice question to the model and a column of the "
      "textbook table. Together they cover the main axes along which angiosperm flowers "
      "vary in a photograph: how flowers are grouped (unit), their symmetry, which organs "
      "make them showy, whether the petals are fused, the shape of the corolla, accessory "
      "structures, the number of parts, and, for capitula, the head type. Ovary position, "
      "placentation and stamen number, which a photograph rarely shows, are left out. "
      "Definitions follow Judd et al. (2016) and Simpson (2019).")
    for group, terms in GLOSSARY:
        P(group, "h2")
        story.append(table([[f"<b>{t}</b>", d] for t, d in terms],
                           [1.6 * inch, 5.3 * inch], s, header=False))

    # ── textbook
    story.append(PageBreak())
    P("4. Evidence (a): the textbook expectation", "h1")
    P(f"family_reference.csv holds one row for each of {len(reference)} families (every "
      f"family that could plausibly be among the 30 largest). Each cell lists the states the "
      f"family shows, the commonest first, separated by '|'; '-' means the character does "
      f"not apply (e.g. symmetry of a grass spikelet, corolla form of an aroid spadix). The "
      f"states were written from the family descriptions in Judd et al. (2016), "
      f"Christenhusz et al. (2017), Heywood et al. (2007) and APweb, at the level of the "
      f"whole family, not of the genera that happen to be in the sample. A family with "
      f"several states (Fabaceae: papilionoid and mimosoid; Ranunculaceae: Ranunculus, "
      f"Aconitum and Delphinium) keeps all of them, and the images then say which the "
      f"sample shows. The notes column records the exceptions.")
    P("They are not ground truth. They are expectations at family rank, written from "
      "standard references; individual genera depart from them, which is why the image "
      "evidence and section 7's conflicts matter.", "muted")
    rows = [["Family"] + [q["label"] for q in QUESTIONS]]
    ref = reference.set_index("family")
    for fam in order:
        if fam in ref.index:
            rows.append([f"<b>{fam}</b>"] + [label_states(ref.at[fam, q["key"]])
                                             for q in QUESTIONS])
        else:
            rows.append([f"<b>{fam}</b>"] + ["(no reference row)"] + [""] * (len(QUESTIONS) - 1))
    story.append(table(rows, [1.0 * inch] + [0.74 * inch] * len(QUESTIONS), s))
    P("Notes on the families in the sample", "h2")
    for fam in order:
        if fam in ref.index and isinstance(ref.at[fam, "notes"], str):
            P(f"<b>{fam}.</b> {ref.at[fam, 'notes']}", "small")

    # ── model
    story.append(PageBreak())
    P("5. Evidence (b): what the images show", "h1")
    P(f"<b>Model.</b> {MODEL}, a vision-language model, run zero-shot (no training on these "
      f"photos or labels) in bf16 on one NVIDIA B200. <b>No free text.</b> For each question "
      f"the options are shown as letters and the model's next-token logits on those letters "
      f"are read directly; their softmax is the answer, a probability for every option. The "
      f"model never writes a sentence, so its answers cannot be misparsed. <b>Order "
      f"reversal.</b> Language models favour some answer positions. Every question is asked "
      f"twice, options in their written order and reversed, and the two probability vectors "
      f"(mapped back to the options) are averaged, so position bias cancels instead of "
      f"voting. 8 questions &times; 2 orders = 16 forward passes per segment, "
      f"{len(ok):,} segments.")
    P(f"<b>Abstaining.</b> Every question has a 'can't tell' option. A segment's call for a "
      f"character is its most probable real option, unless the abstain options together "
      f"hold &ge; {ft.ABSTAIN_CUT} of the probability; then it is 'can't tell' and it is "
      f"left out of that character's shares. <b>Letter mass</b> is the share of the model's "
      f"whole next-token distribution that fell on the option letters (near 1 means it was "
      f"answering the question as posed). <b>Flip rate</b> is how often the two orders "
      f"disagreed on the top option, a measure of how unstable that question is.")
    P("The prompt (shown for the symmetry question in written order):", "h2")
    story.append(Paragraph(prompt_text(QUESTIONS[1], list(range(len(QUESTIONS[1]["options"]))))
                           .replace("\n", "<br/>"), s["mono"]))
    story.append(Spacer(1, 6))
    P("How far to trust each question: calibration", "h2")
    P("Where the textbook gives a family a single state for a character (all Apiaceae are "
      "umbels, all Brassicaceae cruciform, all Orchidaceae zygomorphic), each segment's call "
      "can be scored against it. This measures the model per character on the families "
      "where the answer is not in doubt, and it is the main reason to believe, or not, the "
      "same question where the textbook is variable.")
    rows = [["Character", "Segments scored", "Told", "Abstained", "Match textbook",
             "Accuracy", "Flip rate", "Letter mass"]]
    for r in cal.itertuples():
        rows.append([r.label, r.segments, r.told, r.abstained, r.matches,
                     "" if pd.isna(r.accuracy) else f"{r.accuracy:.0%}",
                     f"{flips[r.character]:.0%}", f"{mass[r.character]:.2f}"])
    story.append(table(rows, [1.3 * inch, 0.9 * inch, 0.6 * inch, 0.75 * inch, 0.95 * inch,
                              0.7 * inch, 0.7 * inch, 0.8 * inch], s))

    # ── rules
    P("6. Evidence (c): the decision rules", "h1")
    P("For each family and character, let the <i>told share</i> of a state be the fraction "
      "of the family's segments called that state, among the segments not called 'can't "
      "tell'. Then:")
    rules = [
        ["n/a", "The textbook marks the character as not applicable. No call."],
        ["confirmed", f"One textbook state, and its told share is &ge; {ft.CONFIRM:.0%}. "
         f"Call = that state."],
        ["supported", f"One textbook state, told share &ge; {ft.SUPPORT:.0%} but < "
         f"{ft.CONFIRM:.0%}. Call = that state, flagged '?'."],
        ["conflict", f"One textbook state, told share < {ft.SUPPORT:.0%}. Call = the textbook "
         f"state (one family-level fact outweighs a sample of 100 photos of a few genera), "
         f"flagged '!' and listed in section 7 with the image majority, for review."],
        ["variable, confirmed", f"Several textbook states, together &ge; {ft.CONFIRM:.0%} "
         f"of the told segments. Call = the textbook states with told share &ge; "
         f"{ft.SEEN:.0%}, most frequent first: the images choose among what the family can "
         f"be."],
        ["variable, conflict", f"Several textbook states, together < {ft.CONFIRM:.0%}. "
         f"Call as above, flagged '!'."],
        ["image only", f"No textbook row. Call = the image majority if it holds &ge; "
         f"{ft.CONFIRM:.0%} of &ge; {ft.MIN_TOLD} told segments, else 'uncertain'."],
        ["reference only", "No segment could be told; the textbook states are kept."],
    ]
    story.append(table([["Status", "Rule"]] + rules, [1.4 * inch, 5.5 * inch], s))
    P("Why the textbook wins a conflict but the images win within it: a family-level "
      "statement such as 'Brassicaceae have four petals in a cross' is the summary of "
      "thousands of species, while 100 top-scoring segments are a biased sample (showy "
      "genera, the most flower-like masks) seen through a model that can be wrong. Where the "
      "family itself varies, though, the textbook cannot say which state this sample shows, "
      "and the images can.", "muted")

    # ── results
    story.append(NextPageTemplate("wide"))
    story.append(PageBreak())
    P("7. Results", "h1")
    story.append(pic(M / "flower_types_by_family.png", 10.0 * inch, 6.6 * inch))
    story.append(NextPageTemplate("portrait"))
    story.append(PageBreak())
    P("Each family's flower type", "h2")
    P("One line per family from the calls; '?' = supported, '!' = conflict (see below); "
      "characters that are n/a, uncertain, 'not a head' or 'no spur/corona/hood' are left "
      "out of the line.", "muted")
    rows = [["Family", "Flower type", "Confirmed"]]
    for r in summ.itertuples():
        rows.append([f"<b>{r.family}</b>", r.flower_type, r.characters_confirmed])
    story.append(table(rows, [1.2 * inch, 5.0 * inch, 0.7 * inch], s))
    bad = calls[calls["status"].str.contains("conflict|supported", regex=True)]
    P("Conflicts and weak support, with the reason", "h2")
    if bad.empty:
        P("None: every applicable character was confirmed.")
    else:
        rows = [["Family", "Character", "Textbook", "Status", "What the images showed"]]
        for r in bad.itertuples():
            rows.append([r.family, r.label, label_states(r.reference), r.status, r.rationale])
        story.append(table(rows, [1.0 * inch, 0.9 * inch, 1.3 * inch, 0.9 * inch,
                                  2.8 * inch], s))
        P("A conflict has one of three causes, which the contact sheets (section 8) "
          "distinguish: the sampled genera really depart from the family norm; SAM3's top "
          "segment is not one flower (a whole umbel or head segmented as one 'flower', or a "
          "single floret of an inflorescence); or the model misjudges the character, which "
          "the calibration table predicts per question.", "muted")

    # ── per family
    P("8. Family by family", "h1")
    P("Each family's 100 segments as cut-outs (the model saw them with context), with every "
      "character's textbook states, the share of told segments in each state, and the call.",
      "muted")
    shares = pd.read_csv(M / "family_shares.csv")
    for fam in order:
        sheet = A / "top_segments" / "sheets" / f"{common.safe(fam)}.jpg"
        block = [PageBreak(), Paragraph(fam, s["h2"])]
        if sheet.exists():
            block.append(pic(sheet, 6.9 * inch, 5.0 * inch))
        g = calls[calls["family"] == fam]
        rows = [["Character", "Textbook", "Images (told share)", "Can't tell", "Call", "Status"]]
        for r in g.itertuples():
            sh = shares[(shares["family"] == fam) & (shares["character"] == r.character)]
            told = sh[sh["option"] != "cant_tell"]
            tot = told["segments"].sum()
            top = told.sort_values("segments", ascending=False).head(3)
            txt = ", ".join(f"{ft.LABELS[o]} {n / tot:.0%}" for o, n in
                            zip(top["option"], top["segments"]) if n) if tot else "-"
            ct = int(sh[sh["option"] == "cant_tell"]["segments"].sum())
            rows.append([r.label, label_states(r.reference), txt, ct, label_states(r.call),
                         r.status])
        block.append(table(rows, [0.95 * inch, 1.45 * inch, 2.0 * inch, 0.55 * inch,
                                  1.15 * inch, 0.85 * inch], s))
        story += block

    # ── limitations and references
    story.append(PageBreak())
    P("9. Limitations", "h1")
    for text in [
        "<b>Biased sample.</b> The top 100 segments are the most confident 'flower' masks, "
        "so showy, well-lit, face-on flowers are over-represented and small or reduced "
        "flowers under-represented; shares describe this sample, not the family's species.",
        "<b>One mask is not always one flower.</b> SAM3's 'flower' can be an entire umbel, "
        "capitulum or spike, or a single floret of one. The context crop mitigates this for "
        "the unit question but symmetry and merosity of a floret inside a dense "
        "inflorescence are often not visible.",
        "<b>Zero-shot model.</b> The calibration table measures the model where the textbook "
        "is unambiguous; a character with low accuracy there (merosity typically) should be "
        "read from the textbook, not the images. A trained classifier on a few hundred "
        "hand-labelled segments would replace this step if a character matters.",
        "<b>Family-level textbook.</b> Large families (Fabaceae, Ranunculaceae, Plantaginaceae) "
        "are heterogeneous; listing several states loses which genera have which. "
        "Genus-level references would sharpen every 'variable' call.",
        "<b>Photographs.</b> 500-px iNaturalist 'medium' images; fine structures (spurs "
        "behind the flower, coronas, fused vs. free petal bases) are often too small.",
    ]:
        P(text)
    P("References", "h1")
    for ref_text in REFERENCES:
        P(ref_text, "small")
    P("Files", "h2")
    P("export/pv99/analysis/: figures/ (review figures, 30 per family), "
      "top_confidence_histogram.png, top_segments/ (cut-outs, context crops, contact sheets, "
      "top_segments.csv), morphology/ (photos.csv per-segment probabilities, segment_calls.csv, "
      "family_shares.csv, family_calls.csv, family_summary.csv, calibration.csv, "
      "flower_types_by_family.png). Code: all_genera/pv99/analysis/ on branch pv99-analysis.",
      "small")

    doc = BaseDocTemplate(str(out_pdf), pagesize=letter, leftMargin=0.7 * inch,
                          rightMargin=0.7 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                          title="pv99 flower types: methodology", author="pv99 analysis")
    pw, ph = letter
    lw, lh = landscape(letter)
    doc.addPageTemplates([
        PageTemplate("portrait", [Frame(0.7 * inch, 0.6 * inch, pw - 1.4 * inch,
                                        ph - 1.2 * inch)], pagesize=letter,
                     onPage=_footer),
        PageTemplate("wide", [Frame(0.5 * inch, 0.5 * inch, lw - 1.0 * inch,
                                    lh - 1.0 * inch)], pagesize=landscape(letter),
                     onPage=_footer),
    ])
    doc.build(story)


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(INK_2)
    w, _ = canvas._pagesize
    canvas.drawRightString(w - 0.5 * inch, 0.35 * inch, f"pv99 flower types · page {doc.page}")
    canvas.restoreState()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_args(parser)
    args = parser.parse_args()
    project = common.project_dir(args.project)
    out = common.analysis_dir(project) / "flower_types_methodology.pdf"
    build(project, out)
    print(f"pdf        : {out}")


if __name__ == "__main__":
    main()
