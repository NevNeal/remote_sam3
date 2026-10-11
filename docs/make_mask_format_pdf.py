"""
Generate docs/mask_format.pdf - how the packed .npz mask format works.

Run:  python docs/make_mask_format_pdf.py
"""
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (HRFlowable, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "mask_format.pdf")

styles = getSampleStyleSheet()
NAVY = colors.HexColor("#2B3A55")
LIGHT = colors.HexColor("#EAEEF5")
h1 = ParagraphStyle("h1", parent=styles["Heading1"], textColor=NAVY, fontSize=18,
                    spaceAfter=2)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=NAVY, fontSize=12.5,
                    spaceBefore=10, spaceAfter=4)
sub = ParagraphStyle("sub", parent=styles["Normal"], textColor=colors.grey, fontSize=9,
                     spaceAfter=8)
body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=6)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=2, spaceAfter=3)
cell = ParagraphStyle("cell", parent=styles["Normal"], fontSize=9, leading=12)
cellb = ParagraphStyle("cellb", parent=cell, textColor=colors.white, fontName="Helvetica-Bold")
code = ParagraphStyle("code", parent=styles["Code"], fontSize=8.5, leading=11.5,
                      backColor=LIGHT, borderPadding=6, spaceBefore=4, spaceAfter=8)

story = []


def P(text, style=body):
    story.append(Paragraph(text, style))


def B(text):
    story.append(Paragraph(text, bullet, bulletText="•"))


def table(rows, widths):
    data = [[Paragraph(c, cellb if r == 0 else cell) for c in row]
            for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))


P("The packed mask format (.npz)", h1)
P("remote_sam3 &middot; segment.py --mask-format npz &middot; maskio.py", sub)
story.append(HRFlowable(width="100%", thickness=0.6, color=colors.lightgrey, spaceAfter=8))

P("<b>In one sentence:</b> instead of one uncompressed file per mask, each photo gets "
  "<b>one small file holding all of its masks</b>, stored at one bit per pixel and then "
  "compressed. Nothing is lost: loading the file gives back exactly the same masks.")

P("Why the old format was so big", h2)
P("The old format wrote one <font face='Courier'>.npy</font> per mask. Each was the whole "
  "image frame as <font face='Courier'>uint8</font>: one <b>byte</b> for every pixel, "
  "holding only a 0 or a 1, with no compression. For a 500 &times; 375 photo that is "
  "500 &times; 375 + a 128-byte header = <b>187,628 bytes per mask</b>, no matter how small "
  "the flower. The taxon 62741 run measured exactly that as its median mask size, and "
  "23.1 GB for 126,653 masks.")

P("What the new format does", h2)
table([
    ["Step", "What happens", "Size of one 500 &times; 375 photo with 3 masks"],
    ["0. Old format", "3 separate files, 1 byte per pixel", "562,884 bytes"],
    ["1. Stack", "The photo's N masks become one N &times; H &times; W array, in one file",
     "same bytes, 1 file instead of 3"],
    ["2. Pack bits", "A pixel is only 0 or 1, so 8 pixels fit in one byte "
                     "(<font face='Courier'>np.packbits</font>)", "70,313 bytes (8&times; smaller)"],
    ["3. Compress", "A mask is long runs of 0s and 1s, which deflate squeezes hard "
                    "(<font face='Courier'>np.savez_compressed</font>)",
     "about 2,000 bytes (measured, see below)"],
], [0.95 * inch, 3.25 * inch, 2.55 * inch])

P("What is inside the file", h2)
P("One file per photo, named <font face='Courier'>&lt;Genus_species_photoid&gt;_masks.npz</font>, "
  "in the same <font face='Courier'>masks/batch_NNNNN/</font> folders as before. It is a "
  "standard NumPy archive with three arrays:")
table([
    ["Array", "Type", "Contents"],
    ["packed", "uint8", "All N masks, flattened, 8 pixels per byte"],
    ["shape", "int32", "[N, H, W] - needed to unpack back to N masks of H &times; W"],
    ["scores", "float32", "[N] - each mask's SAM3 confidence, most confident first"],
], [0.95 * inch, 0.9 * inch, 4.9 * inch])
P("Masks are in the same order as before (most confident first), so mask <i>i</i> here is "
  "what used to be <font face='Courier'>_instance_i.npy</font>. The results CSV's "
  "<font face='Courier'>mask_paths</font> column holds this one path, and "
  "<font face='Courier'>mask_bytes</font> is this file's size. A photo with no detection "
  "gets no file, as before.")

P("How to read it", h2)
story.append(Preformatted(
    "from maskio import load_masks\n"
    "\n"
    "masks, scores = load_masks('masks/batch_00001/Rudbeckia_hirta_13098033_masks.npz')\n"
    "masks.shape    # (N, H, W), dtype bool   - masks[0] is the most confident\n"
    "scores         # array of N confidences\n"
    "\n"
    "# or with nothing but NumPy:\n"
    "import numpy as np\n"
    "d = np.load(path)\n"
    "masks = np.unpackbits(d['packed'], count=int(np.prod(d['shape'])))\n"
    "masks = masks.reshape(d['shape']).astype(bool)", code))

P("What it costs and what it saves", h2)
B("<b>Lossless.</b> Tested: the arrays loaded back are identical to the ones saved.")
B("<b>About 100&times; or more smaller.</b> Measured on real SAM3 masks from the earlier "
  "petal run, scaled to 500 px, 7 masks per photo: about 2 KB per photo against 1.17 MB "
  "as .npy (roughly 550&times;). Larger or more ragged masks compress less; expect "
  "100-500&times;.")
B("<b>Fewer files.</b> One per photo instead of one per mask (about a third as many), "
  "which is easier on /blue and faster to transfer.")
B("<b>Slightly more work to open.</b> A plain <font face='Courier'>np.load</font> no "
  "longer returns the mask directly; use <font face='Courier'>load_masks</font> or the "
  "three NumPy lines above.")
B("<b>Both formats still work.</b> render.py, review_figure.py and the metrics read "
  "either one, chosen by the file extension. The existing taxon 62741 run stays as .npy; "
  "new runs opt in with <font face='Courier'>MASK_FORMAT=npz</font>, which the "
  "all-genera job sets.")

SimpleDocTemplate(OUT, pagesize=letter, leftMargin=0.85 * inch, rightMargin=0.85 * inch,
                  topMargin=0.8 * inch, bottomMargin=0.8 * inch,
                  title="The packed mask format (.npz)").build(story)
print("wrote", OUT)
