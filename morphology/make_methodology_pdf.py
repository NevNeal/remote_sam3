"""
Generate morphology_methodology.pdf: how the per-genus flower morphology calls
are made, from photo selection to the confidence labels, with every threshold
and the numbers behind the run-time estimate.

Run:  python morphology/make_methodology_pdf.py
Output: morphology/morphology_methodology.pdf
"""
import os
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image,
    KeepTogether,
)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from questions import LETTERS, PREAMBLE, QUESTIONS, prompt_text  # noqa: E402
import classify  # noqa: E402

OUT = os.path.join(HERE, "morphology_methodology.pdf")
FIGURE = os.path.join(HERE, "figures", "example_sheet_vicia.jpg")

# ---------------------------------------------------------------- styles
styles = getSampleStyleSheet()
NAVY = colors.HexColor("#2B3A55")
LIGHT = colors.HexColor("#EAEEF5")

h1 = ParagraphStyle("h1", parent=styles["Heading1"], textColor=NAVY,
                    fontSize=18, spaceAfter=4, spaceBefore=2)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=NAVY,
                    fontSize=13, spaceAfter=4, spaceBefore=12)
sub = ParagraphStyle("sub", parent=styles["Normal"], textColor=colors.grey,
                     fontSize=9, spaceAfter=10)
body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10,
                      leading=14, spaceAfter=6)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=2,
                        spaceAfter=3)
note = ParagraphStyle("note", parent=styles["Normal"], fontSize=8.5,
                      textColor=colors.grey, leading=11, spaceAfter=4)
mono = ParagraphStyle("mono", parent=styles["Code"], fontSize=8, leading=10,
                      backColor=LIGHT, borderPadding=6, spaceBefore=4, spaceAfter=10)
formula = ParagraphStyle("formula", parent=body, leftIndent=24, fontName="Courier",
                         fontSize=9, leading=13, spaceAfter=2)
cell = ParagraphStyle("cell", parent=styles["Normal"], fontSize=8.5, leading=11)
cellb = ParagraphStyle("cellb", parent=cell, textColor=colors.white,
                       fontName="Helvetica-Bold")

story = []


def P(text, style=body):
    story.append(Paragraph(text, style))


def B(text):
    story.append(Paragraph(text, bullet, bulletText="•"))


def F(text):
    story.append(Paragraph(text, formula))


def gap(h=6):
    story.append(Spacer(1, h))


def rule():
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.lightgrey,
                            spaceBefore=8, spaceAfter=8))


def table(header, rows, widths):
    data = [[Paragraph(h, cellb) for h in header]]
    data += [[Paragraph(str(c), cell) for c in r] for r in rows]
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    gap(8)


def esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------- title
P("Flower Morphology per Genus — Methodology", h1)
P("Zero-shot vision-language classification of 10 SAM3-cropped photos per angiosperm "
  "genus &nbsp;|&nbsp; remote_sam3, branch <i>flower-morphology</i> &nbsp;|&nbsp; "
  "input: all-genera run 2", sub)

P("This document describes how the pipeline in <font face='Courier'>morphology/</font> "
  "turns run 2's photos into a morphology call per genus: is the flowering structure "
  "an inflorescence, a capitulum, a spadix, a grass spikelet? Is the flower actinomorphic "
  "or zygomorphic? Every threshold below is the value in the code. Numbers marked "
  "<b>measured</b> come from data or a test run. Numbers marked <b>estimated</b> are "
  "projections that the first HiPerGator run will confirm or correct.")

# ---------------------------------------------------------------- 1
P("1. Approach in one paragraph", h2)
P("No labelled training set of flower morphology exists for these 10,734 genera, so the "
  "classifier is <b>zero-shot</b>. A general vision-language model (Qwen3-VL-8B-Instruct) "
  "is shown each photo with one multiple-choice question at a time. It is not asked to "
  "write an answer. A single forward pass gives its probability for the next token, and "
  "the probabilities on the option letters (A, B, C...) <i>are</i> the answer: a full "
  "distribution over the options rather than a single guess. Ten photos per genus are "
  "averaged into a genus-level call, with an agreement score that says how consistent "
  "the photos were.")

table(["Step", "What happens", "Where"], [
    ["1. Select", "10 photos per genus from run 2's metadata", "CPU, seconds"],
    ["2. Crop", "Box around each photo's SAM3 masks + margin, letterboxed to 512 × 512",
     "DataLoader workers"],
    ["3. Score", "6 questions × 1 forward pass each → probability per option", "1 B200"],
    ["4. Aggregate", "Mean probabilities per genus and family → call, share, agreement, "
                     "abstain, confidence", "CPU"],
    ["5. Validate", "Calls vs. 60 textbook genera (155 trait checks)", "CPU"],
    ["6. Sheets", "One contact sheet per genus for checking by eye", "CPU job"],
], [1.0, 4.2, 1.6])

# ---------------------------------------------------------------- 2
P("2. Photo selection", h2)
P("Source: run 2's <font face='Courier'>all_genera_segment_metadata.csv</font>, which "
  "has 235,262 photos (up to 30 per genus, chosen by phenovision probability tier). "
  "Each photo's SAM3 'flower' masks were made at a 0.5 score threshold, keeping the top 5. "
  "Per genus, up to 10 photos are taken in this order:")
B("<b>Masked photos first.</b> A photo is masked if it has at least one SAM3 mask and a mask file.")
B("<b>One per species before a second from any species</b>, so a genus is represented by "
  "as many of its species as possible.")
B("Within each round, by <b>highest mask confidence</b> (conf_max), then phenovision "
  "prediction probability, then photo_id, so the selection is deterministic.")
B("A genus with fewer than 10 masked photos is filled with its unmasked photos, which are "
  "shown uncropped.")
table(["Quantity (measured)", "Value"], [
    ["Genera", "10,734"],
    ["Photos selected", "90,605"],
    ["... of which have SAM3 masks (cropped)", "84,117"],
    ["Genera with the full 10 photos", "8,117"],
    ["Genera with no masked photo at all (all uncropped)", "459"],
], [4.6, 2.2])

# ---------------------------------------------------------------- 3
P("3. Cropping", h2)
P("The model sees the flowers, not the whole scene. For a masked photo:")
B("Take the union of the photo's kept masks (up to 5) and its bounding box, rescaled to the "
  f"image if the masks are at a different resolution.")
B(f"Grow the box by <b>{classify.MARGIN:.0%}</b> of its longer side on every edge to keep "
  "context (stems, neighbouring flowers, bracts).")
B(f"Never crop smaller than <b>{classify.MIN_FRAC:.0%}</b> of the photo's shorter side. A box at "
  "the photo's edge slides back inside rather than being clipped, so it keeps its full size.")
B(f"Letterbox to <b>{classify.SIZE} × {classify.SIZE}</b> on mid-grey, scaling up or down "
  "without distorting the aspect ratio. A fixed size gives every image the same "
  "<b>256 visual tokens</b> (Qwen3-VL: 16-px patches, 2 × 2 merge = 32 px per token), so "
  "batches need no padding.")
P("A photo with no mask, or whose mask file is missing, is letterboxed whole.", note)

# ---------------------------------------------------------------- 4
P("4. The questions", h2)
P("Six questions, each with up to 8 options. Options marked <i>abstain</i> mean \"can't "
  "tell from this photo\". They are recorded, but they never become a genus's call (§6). "
  "The text is in <font face='Courier'>morphology/questions.py</font>. Editing it changes "
  "the questions and the output columns, and no other code needs to change.")
rows = []
for q in QUESTIONS:
    opts = []
    for code, (key, _) in zip(LETTERS, q["options"]):
        k = f"<i>{key}</i>" if key in q["abstain"] else f"<b>{key}</b>"
        opts.append(f"{code}. {k}")
    rows.append([f"<b>{q['key']}</b><br/>{esc(q['text'])}", ", ".join(opts)])
table(["Question", "Options (italic = abstain)"], rows, [2.6, 4.2])

P("Every option carries a short description with textbook examples in the prompt "
  "(e.g. <i>spadix</i>: \"a fleshy spike of tiny flowers, usually with a hood- or leaf-like "
  "spathe, as in Arum, Anthurium or calla\"). The full prompt for the first question:")
story.append(Paragraph(esc(prompt_text(QUESTIONS[0])).replace("\n", "<br/>"), mono))

# ---------------------------------------------------------------- 5
P("5. Scoring a photo", h2)
P("The prompt is formatted with the model's chat template, with the image first, then "
  "the question, then the start of the assistant's turn. One forward pass (bfloat16, SDPA "
  "attention, only the last position's logits kept) gives logits <i>z</i> over the whole "
  "vocabulary for the first token of the answer. With <i>L</i> the token ids of the option "
  "letters (each letter is checked to be a single token):")
F("p_k  = exp(z_Lk) / sum_j exp(z_Lj)        probability of option k")
F("mass = sum_j exp(z_Lj) / sum_v exp(z_v)   share of all probability on the letters")
gap(4)
P("<b>p</b> is the photo's answer. <b>mass</b> is a sanity check. Near 1 means the model "
  "understood it was answering a lettered question. A low value means its next token was "
  "something else and the letter probabilities should be trusted less. The photo's own call "
  "is argmax p over all options, abstain included. photos.csv keeps every option's "
  "probability, so nothing is thrown away at this stage.")
P("Why letter probabilities instead of generated text: no parsing, no off-list answers, "
  "and no sampling randomness. A whole distribution comes out of one forward pass, and it "
  "is cheaper than generation, because nothing is decoded.", note)

# ---------------------------------------------------------------- 6
P("6. From photos to a genus call", h2)
P("For a genus with <i>n</i> readable photos and a question with options <i>k</i>, let "
  "<i>p<sub>ik</sub></i> be photo <i>i</i>'s probability for option <i>k</i> and <i>R</i> the "
  "non-abstain options:")
F("m_k       = (1/n) sum_i p_ik                     mean probability")
F("call c    = argmax over k in R of m_k")
F("share     = m_c / sum over k in R of m_k")
F("agreement = (1/n) #{ i : argmax over k in R of p_ik = c }")
F("abstain   = sum over k not in R of m_k")
gap(4)
P("Restricting the call to <i>R</i> matters. Ten photos of tiny flowers would otherwise all "
  "vote \"can't tell\". Instead, the call is the best real option, and <i>abstain</i> "
  "reports how much the model hedged.")
table(["Confidence", "Rule (applied in this order)"], [
    ["low", "abstain &gt;= 0.5, or agreement &lt; 0.5"],
    ["high", "share &gt;= 0.6 and agreement &gt;= 0.7 and abstain &lt; 0.3"],
    ["medium", "everything else"],
], [1.2, 5.6])
P("Each genus also gets a one-line <i>description</i> joining its calls, with medium "
  "calls marked '?' and low calls left out, e.g. <i>inflorescence · zygomorphic · petals · "
  "fused · lipped?</i>. Families get the same treatment, with all their genera's photos "
  "pooled. genus_probabilities.csv keeps every m<sub>k</sub> as features for a later "
  "supervised model.")

# ---------------------------------------------------------------- 7
P("7. Validation", h2)
P("<font face='Courier'>morphology/reference.csv</font> lists <b>60 genera</b> whose traits "
  "are textbook-certain. Only those traits are filled in, so there are <b>155 checks</b> in "
  "total: unit 28, symmetry 33, display 37, fusion 20, form 24, merosity 13. Examples: "
  "<i>Arum, Arisaema, Anthurium, Zantedeschia, Symplocarpus</i> → spadix; <i>Poa, Festuca, "
  "Bromus, Carex, Cyperus</i> → spikelet, reduced; <i>Salix, Betula, Alnus, Corylus, Populus"
  "</i> → catkin; <i>Helianthus, Taraxacum, Bellis, Cirsium, Aster</i> → head; <i>Vicia, "
  "Lathyrus, Lupinus, Pisum</i> → zygomorphic, pea-like; <i>Salvia, Lamium</i> → fused, "
  "two-lipped; <i>Campanula, Vaccinium, Erica</i> → fused, bell. Genera where the trait "
  "varies within the genus (e.g. wild vs. cultivated <i>Rosa</i> petal counts) are left "
  "blank for that trait. The summary prints accuracy per question and lists every miss.")
P("<b>Local test (measured).</b> The smaller Qwen3-VL-2B ran on an RTX 4060, on 10 photos "
  "each of <i>Vicia sativa</i> and <i>Chrysopsis scabrella</i> (an aster). It called "
  "<i>Vicia</i> zygomorphic, petals, pea-like (all high confidence) and <i>Chrysopsis</i> a "
  "head, actinomorphic. All 3 reference checks were correct. Its weakest question was petal "
  "count: it called <i>Vicia</i> 3/6 rather than 5, and that question had the lowest letter "
  "mass (0.72, against 0.88–0.97 for the others). The 8B model is expected to do better, "
  "but petal count should be read with care.")

if os.path.exists(FIGURE):
    img = Image(FIGURE)
    w = 6.8 * inch
    img.drawWidth, img.drawHeight = w, w * img.imageHeight / img.imageWidth
    story.append(KeepTogether([
        img,
        Paragraph("Figure 1. Contact sheet from the local test (Qwen3-VL-2B). Genus calls "
                  "with share, agreement and confidence on top. Each crop's own calls "
                  "beneath it. Photo 10 had no mask file in the test and is shown whole.",
                  note)]))

# ---------------------------------------------------------------- 8
P("8. Compute and run time", h2)
table(["Quantity", "Value", "Basis"], [
    ["Photos × questions", "90,605 × 6 = 543,630 forward passes", "measured"],
    ["Tokens per pass", "~490 (256 image + ~100–240 text)", "measured"],
    ["Tokens per photo, all questions", "~2,400", "measured"],
    ["Total prefill tokens", "~220 million", "derived"],
    ["2B on RTX 4060", "1.56 s per batch of 8 × 492 tokens (~25% of peak)", "measured"],
    ["8B on one B200", "~2 hours at similar utilisation", "estimated"],
    ["Job limit", "stops itself at 7.5 h, summarises, resubmit resumes", "by design"],
], [1.9, 3.1, 1.8])
P("<b>Parallelism.</b> Images are scored in batches of 64 per forward pass. A batch that "
  "runs out of memory splits in half and retries. Twelve DataLoader worker processes read "
  "the photos and masks from /blue, crop, and run the image preprocessing, once per batch, "
  "with 4 batches queued ahead of the GPU. Only the text is tokenised per question, on the "
  "main process. On the first batch the job checks that these hand-built inputs equal the "
  "processor's own output token for token, and falls back to the processor if they ever "
  "differ. One GPU. Each photo is appended to photos.csv as it finishes, so an interrupted "
  "job loses nothing.")

# ---------------------------------------------------------------- 9
P("9. Outputs", h2)
table(["File", "Contents"], [
    ["genus_morphology.csv", "Per genus: call, share, agreement, abstain, confidence per question; description"],
    ["genus_probabilities.csv", "Per genus: mean probability of every option (features for ML)"],
    ["family_morphology.csv", "The same calls per family"],
    ["photos.csv", "Per photo: every option's probability, the photo's call, letter mass, cropped flag"],
    ["validation.csv", "Every reference check: expected, called, share, agreement, correct"],
    ["sheets/&lt;Family&gt;/&lt;Genus&gt;.jpg", "10 crops with per-photo and genus calls"],
], [2.1, 4.7])

# ---------------------------------------------------------------- 10
P("10. Limitations", h2)
B("<b>Zero-shot is a first pass, not ground truth.</b> The model has never been trained "
  "on these definitions. Read validation.csv and a sample of sheets before relying on a "
  "question.")
B("<b>Probabilities are not calibrated.</b> A share of 0.8 is a relative strength, not an "
  "80% chance of being right. Calibration needs labelled genera.")
B("<b>Letter-position bias.</b> Language models favour some letters. If a question's calls "
  "pile up on option A, reorder the options and rerun to compare.")
B("<b>The crop decides what is seen.</b> If SAM3 masked one flower of an umbel, the crop "
  "may show a solitary flower. If it masked a whole head, individual florets may be too "
  "small to judge symmetry.")
B("<b>Ten photos, chosen by mask confidence.</b> The most confidently segmented photos may "
  "favour large, showy, face-on flowers. In a genus whose species differ, the call follows "
  "whichever form most of the sampled species share, and agreement drops.")
B("<b>Some questions are hierarchical.</b> Symmetry and petal count are about an individual "
  "flower and are often unknowable for heads, spikelets and catkins. Read them as such "
  "(their abstain values will be high).")

P("11. Next steps", h2)
B("Run the 2,000-photo test, then the full run. Read the validation accuracy per question.")
B("Hand-label a few hundred genera from the sheets, stratified by family, and measure "
  "accuracy and calibration properly.")
B("For questions that matter, train a small supervised classifier (logistic regression or "
  "gradient boosting) on genus_probabilities.csv using those labels. Or fine-tune on the "
  "image crops if the zero-shot features prove too weak.")

rule()
P("Generated by morphology/make_methodology_pdf.py. Thresholds are read from "
  "morphology/classify.py and the questions from morphology/questions.py, so this "
  "document matches the code it was generated from.", note)

doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=0.8 * inch,
                        rightMargin=0.8 * inch, topMargin=0.7 * inch,
                        bottomMargin=0.7 * inch,
                        title="Flower Morphology per Genus — Methodology",
                        author="remote_sam3")
doc.build(story)
print(OUT)
