"""Two alternative A1 poster designs (the classic one is make_poster.py), from the same final numbers and figures.

Run from the Project folder:  python src/shapes/make_poster_variants.py
Writes ../Report/final/poster_options/Poster_B_story.pdf and Poster_C_visual.pdf
"""
import json
import os

from PIL import Image as PI
from reportlab.graphics import renderPDF
from reportlab.graphics.barcode.qr import QrCodeWidget
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A1
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, Image, KeepInFrame, Paragraph, Spacer, Table, TableStyle

OUT = "../Report/final/poster_options"
FIG = "results/optionb/figs"
FONT = "C:/Windows/Fonts"
REPO = "https://github.com/bntshakya/AI-Assisted-Inverse-Design-of-Terahertz-Photonic-Devices-for-Next-Generation-6G-Systems"
for n, f in (("Body", "calibri.ttf"), ("Body-Bold", "calibrib.ttf"), ("Body-Italic", "calibrii.ttf"), ("Head", "cambriab.ttf")):
    pdfmetrics.registerFont(TTFont(n, os.path.join(FONT, f)))
W, H = A1
PETROL, TEAL, TEAL_L, AMBER, AMBER_L = (colors.HexColor(c) for c in ("#0b3c49", "#0d7a84", "#e3f1f2", "#e89b2d", "#fdf1dd"))
INK, MUTED, WHITE, CORAL = colors.HexColor("#16202a"), colors.HexColor("#566271"), colors.white, colors.HexColor("#c8553d")


def styles(scale=1.0):
    s = lambda v: v * scale
    return {
        "h": ParagraphStyle("h", fontName="Head", fontSize=s(33), leading=s(38), textColor=PETROL, spaceBefore=s(6), spaceAfter=s(8)),
        "p": ParagraphStyle("p", fontName="Body", fontSize=s(23), leading=s(30.5), textColor=INK, spaceAfter=s(8)),
        "pc": ParagraphStyle("pc", fontName="Body", fontSize=s(23), leading=s(30), textColor=INK, alignment=TA_CENTER),
        "b": ParagraphStyle("b", fontName="Body", fontSize=s(23), leading=s(30), textColor=INK, leftIndent=s(26), bulletIndent=s(6), spaceAfter=s(5)),
        "cap": ParagraphStyle("cap", fontName="Body-Italic", fontSize=s(18), leading=s(23), textColor=MUTED, spaceAfter=s(10)),
        "cell": ParagraphStyle("cell", fontName="Body", fontSize=s(20), leading=s(24), textColor=INK),
        "cellb": ParagraphStyle("cellb", fontName="Body-Bold", fontSize=s(20), leading=s(24), textColor=WHITE),
        "eq": ParagraphStyle("eq", fontName="Body", fontSize=s(25), leading=s(31), textColor=PETROL, alignment=TA_CENTER),
    }


def fig(name, width, st, caption=None, max_h=None):
    path = os.path.join(FIG, name)
    w, h = PI.open(path).size
    hh = width * h / w
    if max_h and hh > max_h:
        width, hh = max_h * w / h, max_h
    out = [Image(path, width=width, height=hh)]
    if caption:
        out.append(Paragraph(caption, st["cap"]))
    return out


def table(rows, widths, st):
    data = [[Paragraph(str(c), st["cellb"] if i == 0 else st["cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths)
    sty = [("BACKGROUND", (0, 0), (-1, 0), TEAL), ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#c7d3d6")),
           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]
    t.setStyle(TableStyle(sty))
    return [t, Spacer(1, 10)]


def soft(flow, width, color=TEAL_L, pad=16):
    t = Table([[flow]], colWidths=[width])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), color), ("LEFTPADDING", (0, 0), (-1, -1), pad),
                           ("RIGHTPADDING", (0, 0), (-1, -1), pad), ("TOPPADDING", (0, 0), (-1, -1), pad * 0.75),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), pad * 0.75)]))
    return [t, Spacer(1, 12)]


def qr(c, x, y, size, url=REPO, fg=PETROL):
    w = QrCodeWidget(url)
    w.barFillColor = fg
    b = w.getBounds()
    d = Drawing(size, size, transform=[size / (b[2] - b[0]), 0, 0, size / (b[3] - b[1]), 0, 0])
    d.add(w)
    renderPDF.draw(d, c, x, y)


def flow_into(c, items, x, y, w, h):
    Frame(x, y, w, h, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, showBoundary=0) \
        .addFromList([KeepInFrame(w, h, items, mode="shrink")], c)


def numbered(n, title, st, color=TEAL):
    return Paragraph(f'<font color="{color.hexval().replace("0x", "#")}">{n}</font>&#160;&#160;{title}', st["h"])


# ======================================================================== B: story
def poster_b(N):
    st = styles()
    c = canvas.Canvas(f"{OUT}/Poster_B_story.pdf", pagesize=A1)
    c.setTitle("THz valley photonic crystal inverse design - poster (story)")
    M, G = 24 * mm, 16 * mm
    # header
    hb = 175 * mm
    c.setFillColor(PETROL)
    c.rect(0, H - hb, W, hb, stroke=0, fill=1)
    img_w = 150 * mm
    c.setFillColor(WHITE)
    c.roundRect(W - M - img_w - 6 * mm, H - hb + 18 * mm, img_w + 12 * mm, hb - 36 * mm, 8 * mm, stroke=0, fill=1)
    cell = "../Report/final/slides/assets/freeform_cell.png"
    if os.path.exists(cell):
        cw, ch = PI.open(cell).size
        ih = hb - 48 * mm
        iw = ih * cw / ch
        c.drawImage(cell, W - M - img_w + (img_w - iw) / 2, H - hb + 24 * mm, iw, ih)
    tw = W - 2 * M - img_w - 20 * mm
    c.setFillColor(WHITE)
    lines = ["AI-Assisted Inverse Design of", "Terahertz Photonic Devices", "for Next-Generation 6G Systems"]
    fs = min(80, 80 * tw / max(pdfmetrics.stringWidth(l, "Head", 80) for l in lines))
    for i, l in enumerate(lines):
        c.setFont("Head", fs)
        c.drawString(M, H - 42 * mm - i * fs * 1.12, l)
    c.setFont("Body", 30)
    c.setFillColor(colors.HexColor("#cfe6e8"))
    c.drawString(M, H - hb + 40 * mm, "Binit Shakya (254161023)  \u00b7  M.Tech. Data Science  \u00b7  IIT Guwahati")
    c.drawString(M, H - hb + 24 * mm, "Supervisors: Prof. Debabrata Sikdar  \u00b7  Prof. Teena Sharma  \u00b7  Phase I, 2026")
    # hero strip
    sy = H - hb - 64 * mm
    c.setFillColor(AMBER_L)
    c.rect(0, sy, W, 64 * mm, stroke=0, fill=1)
    steps = [("Type the band gap", "you want (f_lower, f_upper)"), ("AI proposes", "fabricable hole shapes in 0.2 s"),
             ("Every design", "re-simulated"), (f"{N.get('verOK', '12')} of {N.get('verN', '14')}", "confirmed in COMSOL")]
    sw = (W - 2 * M) / 4
    for i, (a, b) in enumerate(steps):
        x = M + i * sw
        c.setFillColor(PETROL)
        c.setFont("Head", 38)
        c.drawCentredString(x + sw / 2, sy + 36 * mm, a)
        c.setFont("Body", 26)
        c.setFillColor(INK)
        c.drawCentredString(x + sw / 2, sy + 22 * mm, b.replace("f_lower, f_upper", "f lower, f upper"))
        if i < 3:
            c.setFillColor(AMBER)
            ax = x + sw - 6 * mm
            c.setLineWidth(5)
            c.setStrokeColor(AMBER)
            c.line(ax - 14 * mm, sy + 33 * mm, ax, sy + 33 * mm)
            p = c.beginPath()
            p.moveTo(ax + 6 * mm, sy + 33 * mm)
            p.lineTo(ax - 2 * mm, sy + 38 * mm)
            p.lineTo(ax - 2 * mm, sy + 28 * mm)
            p.close()
            c.drawPath(p, stroke=0, fill=1)
    # footer
    fb = 52 * mm
    c.setFillColor(PETROL)
    c.rect(0, 0, W, fb, stroke=0, fill=1)
    qs = 40 * mm
    c.setFillColor(WHITE)
    c.rect(W - M - qs - 4 * mm, 6 * mm, qs + 8 * mm, qs + 8 * mm, stroke=0, fill=1)
    qr(c, W - M - qs, 10 * mm, qs)
    c.setFont("Body-Bold", 26)
    c.setFillColor(WHITE)
    c.drawString(M, 31 * mm, "Code, trained models and results: scan the QR code (GitHub)")
    c.setFont("Body", 22)
    c.setFillColor(colors.HexColor("#cfe6e8"))
    c.drawString(M, 16 * mm, "Python \u00b7 PyTorch \u00b7 scikit-learn \u00b7 COMSOL Multiphysics 6.4  \u00b7  "
                             "design tool: python design.py 300 330  \u00b7  web app: streamlit run app.py")
    # columns
    top, bot = sy - 14 * mm, fb + 12 * mm
    lw = (W - 2 * M - G) * 0.53
    rw = W - 2 * M - G - lw
    left = [numbered(1, "Why", st),
            Paragraph("6G links near <b>0.3 THz</b> can travel along robust edges of a silicon <b>valley photonic crystal</b>: "
                      "a slab with two unequal air holes per cell. It only works if the cell's <b>band gap</b> sits exactly "
                      "at the right frequencies, which normally takes many slow COMSOL simulations to find.", st["p"]),
            *fig("unit_cells.png", lw, st, "Four hole families in this work: rounded and sharp triangles, circles, and freeform "
                 "three-fold shapes (dashed: one unit cell)."),
            numbered(2, "The idea: any hole in 10 numbers", st),
            *soft([Paragraph("r(&#952;) = r<sub>0</sub> + c<sub>3</sub> cos 3&#952; + s<sub>3</sub> sin 3&#952; + "
                             "c<sub>6</sub> cos 6&#952; + s<sub>6</sub> sin 6&#952;", st["eq"])], lw - 32),
            Paragraph("Five numbers per hole. Only multiples of 3 keep the 120\u00b0 symmetry the valley physics needs. "
                      "Circles and triangles are special cases; the AI can invent new shapes.", st["p"]),
            numbered(3, "How", st),
            *fig("pipeline.png", lw, st, "COMSOL calibrates a fast plane-wave (PWE) solver, which makes 6,000 training cells; "
                 "the AI turns a target into shapes; every design is checked and re-simulated."),
            numbered(4, "The AI: tandem network and CVAE", st),
            *fig("arch_tandem.png", lw, st, "Tandem network: the inverse network (left) is trained through the frozen forward "
                 "ensemble (right). CVAE + refine adds many designs per target."),
            Paragraph(f"Forward ensemble: 5 networks, 18 inputs \u2192 256 \u00d7 3 \u2192 2 band edges, "
                      f"<b>{N['fwdLo']} / {N['fwdUp']} GHz</b> error on unseen cells.", st["p"]),
            numbered(5, "How accuracy is measured", st),
            *soft([Paragraph("Errors in <b>GHz</b>, always on unseen data. A design counts if its <b>worse edge</b> is within "
                             "<b>1 GHz</b> after re-simulation: 0.3 % of 300 GHz, less than half of the narrowest 6G channel. "
                             "Not percent (3 GHz looks like 1 %) and not R\u00b2 (it stays near 1 even for 1 GHz errors).",
                             st["p"])], lw - 32)]
    right = [numbered(6, "Is the fast solver right?", st),
             *fig("final_pwe_vs_comsol.png", rw, st, f"PWE vs COMSOL on {N['pweN']} gapped cells: {N['pweLo']} / {N['pweUp']} "
                  f"GHz; with a learned COMSOL correction <b>{N['mfCorrLo']} / {N['mfCorrUp']} GHz</b>."),
             numbered(7, "Inverse design", st),
             *table([["Method", "Within 1 GHz", "Designs"], ["Tandem network", f"{N['tandemSucc']} \u00b1 {N['tandemSd']} %", "1"],
                     ["CVAE + refine (ours)", f"{N['crSucc']} \u00b1 {N['crSd']} %", f"~{N['crDistinct']}"]],
                    [rw * 0.5, rw * 0.28, rw * 0.22], st),
             *fig("showcase.png", rw, st, "Many different verified designs for one target (300 / 330 GHz).", max_h=95 * mm),
             numbered(8, "Proof in COMSOL", st),
             *fig("comsol_verify.png", rw, st, f"{N.get('verOK', '12')} of {N.get('verN', '14')} AI designs within 1 GHz of their "
                  "target (worst 1.05 GHz); misses share the known PWE offset."),
             numbered(9, "PWE or COMSOL data?", st),
             Paragraph(N["verdict"], st["p"]),
             numbered(10, "Next", st),
             Paragraph("Topological edge-state waveguide \u00b7 COMSOL in the loop (active learning) \u00b7 3D slab and "
                       "fabrication-tolerance checks.", st["p"])]
    flow_into(c, left, M, bot, lw, top - bot)
    flow_into(c, right, M + lw + G, bot, rw, top - bot)
    c.showPage()
    c.save()
    print("wrote", f"{OUT}/Poster_B_story.pdf")


# ======================================================================== C: visual
def poster_c(N):
    st = styles(1.05)
    c = canvas.Canvas(f"{OUT}/Poster_C_visual.pdf", pagesize=A1)
    c.setTitle("THz valley photonic crystal inverse design - poster (visual)")
    M = 26 * mm
    # headline
    c.setFillColor(PETROL)
    q = "Can AI design a 6G terahertz crystal?"
    fs = min(118, 118 * (W - 2 * M) / pdfmetrics.stringWidth(q, "Head", 118))
    c.setFont("Head", fs)
    c.drawString(M, H - 30 * mm - fs * 0.75, q)
    y = H - 30 * mm - fs * 0.75 - 26 * mm
    c.setFont("Body-Bold", 34)
    c.setFillColor(TEAL)
    c.drawString(M, y, "AI-Assisted Inverse Design of Terahertz Photonic Devices for Next-Generation 6G Systems")
    c.setFont("Body", 28)
    c.setFillColor(MUTED)
    c.drawString(M, y - 14 * mm, "Binit Shakya (254161023), M.Tech. Data Science, IIT Guwahati  \u00b7  Supervisors: "
                                 "Prof. Debabrata Sikdar, Prof. Teena Sharma  \u00b7  Phase I, 2026")
    # answer line
    y -= 34 * mm
    c.setFillColor(AMBER)
    c.roundRect(M, y - 6 * mm, 30 * mm, 16 * mm, 4 * mm, stroke=0, fill=1)
    c.setFillColor(WHITE)
    c.setFont("Head", 34)
    c.drawCentredString(M + 15 * mm, y - 1.5 * mm, "Yes.")
    c.setFillColor(INK)
    c.setFont("Body", 32)
    c.drawString(M + 38 * mm, y - 1 * mm, "Give it the band gap you want; it returns fabricable hole shapes in 0.2 s, checked by physics.")
    # giant numbers
    y -= 22 * mm
    nums = [(f"{N['tandemSucc']}%", "of unseen targets\nreached within 1 GHz"), (f"~{N['crDistinct']}", "different designs\nper target"),
            (f"{N.get('verOK', '12')}/{N.get('verN', '14')}", "AI designs confirmed\nin COMSOL"),
            (f"{N['mfCorrLo']}", "GHz error vs COMSOL\n(lower edge, corrected)")]
    nw = (W - 2 * M) / 4
    for i, (big, lab) in enumerate(nums):
        x = M + i * nw
        c.setFillColor(TEAL)
        c.setFont("Head", 120)
        c.drawCentredString(x + nw / 2, y - 42 * mm, big)
        c.setFillColor(INK)
        c.setFont("Body", 26)
        for k, l in enumerate(lab.split("\n")):
            c.drawCentredString(x + nw / 2, y - 56 * mm - k * 10 * mm, l)
    y -= 84 * mm
    # central figure
    c.setFillColor(TEAL_L)
    c.rect(0, y - 196 * mm, W, 196 * mm, stroke=0, fill=1)
    c.setFillColor(PETROL)
    c.setFont("Head", 40)
    c.drawString(M, y - 16 * mm, "The AI: a tandem network trained through a frozen physics surrogate")
    p = os.path.join(FIG, "arch_tandem.png")
    pw, ph = PI.open(p).size
    iw = W - 2 * M
    ih = iw * ph / pw
    if ih > 150 * mm:
        ih = 150 * mm
        iw = ih * pw / ph
    c.drawImage(p, (W - iw) / 2, y - 26 * mm - ih, iw, ih)
    c.setFont("Body-Italic", 22)
    c.setFillColor(MUTED)
    c.drawString(M, y - 190 * mm, "Trained on 6,000 cells from a fast plane-wave solver; CVAE + refine gives ~8 alternative "
                                  "designs per target; every design is re-simulated before it counts.")
    y -= 210 * mm
    # three panels
    pw3 = (W - 2 * M - 2 * 14 * mm) / 3
    ph3 = 142 * mm
    panels = [
        ("1  The shape", [Paragraph("Any three-fold hole in <b>10 numbers</b>:", st["p"]),
                          Paragraph("r(&#952;) = r<sub>0</sub> + c<sub>3</sub> cos 3&#952; + s<sub>3</sub> sin 3&#952;<br/>"
                                    "+ c<sub>6</sub> cos 6&#952; + s<sub>6</sub> sin 6&#952;", st["eq"]), Spacer(1, 6),
                          *fig("unit_cells.png", pw3, st, "Triangles and circles are special cases; freeform shapes are new.")]),
        ("2  The physics check", [*fig("final_pwe_vs_comsol.png", pw3, st), Paragraph(
            f"Fast solver vs COMSOL on <b>{N['pweN']}</b> cells: {N['pweLo']} / {N['pweUp']} GHz, and <b>{N['mfCorrLo']} / "
            f"{N['mfCorrUp']} GHz</b> after a correction learned from COMSOL.", st["p"])]),
        ("3  The proof", [*fig("comsol_verify.png", pw3, st), Paragraph(
            f"<b>{N.get('verOK', '12')} of {N.get('verN', '14')}</b> AI designs hit their target within 1 GHz in COMSOL "
            "(worst 1.05 GHz). Misses share the known solver offset.", st["p"])]),
    ]
    for i, (t, items) in enumerate(panels):
        x = M + i * (pw3 + 14 * mm)
        c.setFillColor(PETROL)
        c.setFont("Head", 38)
        c.drawString(x, y - 12 * mm, t)
        flow_into(c, items, x, y - ph3, pw3, ph3 - 22 * mm)
    y -= ph3 + 8 * mm
    # showcase strip
    c.setFillColor(PETROL)
    c.setFont("Head", 38)
    c.drawString(M, y - 12 * mm, "Many answers for one target (300 / 330 GHz)")
    p = os.path.join(FIG, "showcase.png")
    pw, ph = PI.open(p).size
    avail = y - 22 * mm - 78 * mm
    ih = min(avail, (W - 2 * M) * 0.70 * ph / pw)
    iw = ih * pw / ph
    c.drawImage(p, M, y - 22 * mm - ih, iw, ih)
    # learned + next + QR beside the showcase
    bx = M + iw + 16 * mm
    bw = W - M - bx
    learn = [Paragraph("What we learned", st["h"]),
             Paragraph("\u2022 Cheap physics data + a few COMSOL runs beat COMSOL data alone (0.25 vs 0.27 GHz on freeform "
                       "cells), and work with no freeform COMSOL runs at all.", st["p"]),
             Paragraph("\u2022 Judge a design with a simulator, never with the network that made it.", st["p"]),
             Paragraph("\u2022 Run simple baselines first: on 2 inputs a cubic fit beat the inherited network 100\u00d7.", st["p"]),
             Paragraph("Next", st["h"]),
             Paragraph("Edge-state waveguide \u00b7 COMSOL in the loop \u00b7 3D slab and tolerance checks.", st["p"])]
    flow_into(c, learn, bx, y - 22 * mm - ih, bw, ih)
    # footer with QR
    qs = 52 * mm
    qr(c, W - M - qs, 14 * mm, qs)
    c.setFont("Body-Bold", 28)
    c.setFillColor(PETROL)
    c.drawRightString(W - M - qs - 8 * mm, 46 * mm, "Code, models, results \u2192")
    c.setFont("Body", 22)
    c.setFillColor(MUTED)
    c.drawString(M, 20 * mm, "How accuracy is measured: errors in GHz on unseen data; a design counts if its worse edge is "
                             "within 1 GHz after re-simulation.")
    c.drawString(M, 11 * mm, "Python \u00b7 PyTorch \u00b7 scikit-learn \u00b7 COMSOL Multiphysics 6.4")
    c.showPage()
    c.save()
    print("wrote", f"{OUT}/Poster_C_visual.pdf")


def main():
    os.makedirs(OUT, exist_ok=True)
    N = json.load(open("results/optionb/final_numbers.json"))
    poster_b(N)
    poster_c(N)


if __name__ == "__main__":
    main()
