"""A1 portrait poster for the Phase I evaluation, built from the final numbers and figures.

Run from the Project folder after make_final_numbers.py and make_final_figures.py:
    python src/shapes/make_poster.py
Writes ../Report/final/254161023_MTP1_Poster.pdf
"""
import json
import os

from PIL import Image as PI
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A1
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, Image, KeepInFrame, Paragraph, Spacer, Table, TableStyle

OUT = "../Report/final/254161023_MTP1_Poster.pdf"
FIG = "results/optionb/figs"
FONT = "C:/Windows/Fonts"
for n, f in (("Body", "calibri.ttf"), ("Body-Bold", "calibrib.ttf"), ("Body-Italic", "calibrii.ttf")):
    pdfmetrics.registerFont(TTFont(n, os.path.join(FONT, f)))
INK, ACC, SOFT, MUTED = colors.HexColor("#16202a"), colors.HexColor("#0d7a84"), colors.HexColor("#e3f1f2"), colors.HexColor("#566271")
W, H = A1
M = 22 * mm
GAP = 14 * mm
COLW = (W - 2 * M - 2 * GAP) / 3

st = {"h": ParagraphStyle("h", fontName="Body-Bold", fontSize=34, leading=40, textColor=ACC, spaceBefore=10, spaceAfter=10),
      "p": ParagraphStyle("p", fontName="Body", fontSize=23, leading=31, textColor=INK, spaceAfter=10),
      "b": ParagraphStyle("b", fontName="Body", fontSize=23, leading=31, textColor=INK, leftIndent=26, bulletIndent=6, spaceAfter=6),
      "cap": ParagraphStyle("cap", fontName="Body-Italic", fontSize=18, leading=23, textColor=MUTED, spaceAfter=12),
      "big": ParagraphStyle("big", fontName="Body-Bold", fontSize=58, leading=62, textColor=ACC, alignment=TA_CENTER),
      "bigl": ParagraphStyle("bigl", fontName="Body", fontSize=19, leading=23, textColor=INK, alignment=TA_CENTER),
      "cell": ParagraphStyle("cell", fontName="Body", fontSize=19, leading=23, textColor=INK),
      "cellb": ParagraphStyle("cellb", fontName="Body-Bold", fontSize=19, leading=23, textColor=colors.white)}


def P(t, s="p"):
    return Paragraph(t, st[s])


def bullets(items):
    return [Paragraph(i, st["b"], bulletText="•") for i in items]


def fig(name, width, caption=None):
    path = os.path.join(FIG, name)
    if not os.path.exists(path):
        return [P(f"[{name} missing]", "cap")]
    w, h = PI.open(path).size
    out = [Image(path, width=width, height=width * h / w)]
    if caption:
        out.append(P(caption, "cap"))
    return out


def table(rows, widths):
    data = [[Paragraph(str(c), st["cellb"] if i == 0 else st["cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths)
    sty = [("BACKGROUND", (0, 0), (-1, 0), ACC), ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#c7ccd3")),
           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]
    for i in range(2, len(rows), 2):
        sty.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f3f6f8")))
    t.setStyle(TableStyle(sty))
    return [t, Spacer(1, 12)]


def box(flow, color=SOFT):
    t = Table([[flow]], colWidths=[COLW])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), color), ("BOX", (0, 0), (-1, -1), 1.2, ACC),
                           ("LEFTPADDING", (0, 0), (-1, -1), 16), ("RIGHTPADDING", (0, 0), (-1, -1), 16),
                           ("TOPPADDING", (0, 0), (-1, -1), 12), ("BOTTOMPADDING", (0, 0), (-1, -1), 12)]))
    return [t, Spacer(1, 14)]


def main():
    N = json.load(open("results/optionb/final_numbers.json"))
    c = canvas.Canvas(OUT, pagesize=A1)
    c.setTitle("AI-Assisted Inverse Design of THz Valley Photonic Crystals - poster")
    # title band
    band = 150 * mm
    c.setFillColor(ACC)
    c.rect(0, H - band, W, band, stroke=0, fill=1)
    c.setFillColor(colors.white)
    l1, l2 = "AI-Assisted Inverse Design of Terahertz Photonic Devices", "for Next-Generation 6G Systems"
    fs = min(76, 76 * (W - 2 * M) / pdfmetrics.stringWidth(l1, "Body-Bold", 76))
    c.setFont("Body-Bold", fs)
    c.drawCentredString(W / 2, H - 58 * mm, l1)
    c.drawCentredString(W / 2, H - 58 * mm - 1.2 * fs, l2)
    c.setFont("Body", 34)
    c.drawCentredString(W / 2, H - 118 * mm, "Binit Shakya (254161023) · M.Tech. Data Science · IIT Guwahati")
    c.drawCentredString(W / 2, H - 134 * mm, "Supervisors: Prof. Debabrata Sikdar, Prof. Teena Sharma · Phase I (Semester III, 2026)")

    # headline numbers row
    top = H - band - 12 * mm
    tiles = [(f"{N['tandemSucc']}%", "of unseen targets reached within 1 GHz (tandem, PWE-checked)"),
             (f"{N['crDistinct']}", "different valid designs per target (CVAE + refine)"),
             (f"{N.get('verOK', '?')} / {N.get('verN', '?')}", "AI designs within 1 GHz of their target in COMSOL"),
             (f"{N['mfCorrLo']} / {N['mfCorrUp']}", "GHz error vs COMSOL with the learned correction")]
    tw = (W - 2 * M - 3 * GAP) / 4
    th = 62 * mm
    for i, (big, lab) in enumerate(tiles):
        x = M + i * (tw + GAP)
        c.setFillColor(SOFT)
        c.setStrokeColor(ACC)
        c.roundRect(x, top - th, tw, th, 8 * mm, stroke=1, fill=1)
        f = Frame(x, top - th, tw, th, leftPadding=10, rightPadding=10, topPadding=18, bottomPadding=6, showBoundary=0)
        f.addFromList([P(big, "big"), Spacer(1, 6), P(lab, "bigl")], c)
    ytop = top - th - 14 * mm
    ybot = 26 * mm
    colh = ytop - ybot

    col1 = [P("1 · The problem", "h"),
            P("6G links will use carriers near <b>0.3 THz</b>. A silicon <b>valley photonic crystal</b> (a slab with two "
              "unequal air holes per cell) guides them along robust edges, but only if its <b>band gap</b> sits at the "
              "right frequencies."),
            *box([P("<b>Goal:</b> given a target gap (f<sub>lower</sub>, f<sub>upper</sub>), find several fabricable hole "
                    "shapes whose simulated band edges are both within <b>1 GHz</b>, with far fewer COMSOL runs than a sweep.")]),
            P("2 · Any three-fold hole in 10 numbers", "h"),
            P("Each hole: r(θ) = r<sub>0</sub> + c<sub>3</sub> cos 3θ + s<sub>3</sub> sin 3θ + c<sub>6</sub> cos 6θ + "
              "s<sub>6</sub> sin 6θ. Only multiples of 3 keep the 120° symmetry the valley physics needs. Circles and "
              "triangles are special cases; most designs are new shapes."),
            *fig("fourier_descriptor.png", COLW, "Exact triangles and their Fourier versions (top); shapes the description "
                 "can make (bottom)."),
            P("3 · Data", "h"),
            *bullets([f"<b>6,000</b> cells from a fast plane-wave-expansion (PWE) solver (0.2 s each)",
                      f"<b>{N['nComsol']}</b> COMSOL cells ({N['nFree']} freeform, {N['nSharp']} sharp and {N['nRound']} "
                      f"rounded triangles, {N['nCircle']} circles, {N['nVerify']} AI designs)",
                      "Band edges only at M and K: checked on every full-path cell"]),
            P("4 · How accuracy is measured", "h"),
            *box(bullets(["<b>Forward</b>: mean error in GHz on unseen shapes, and the 95 % bound",
                          "<b>Inverse</b>: re-simulate each design; error = worse of its two edges; <b>success</b> = below 1 GHz",
                          "<b>Why</b>: GHz, not % (3 GHz looks like 1 % at 300 GHz); both edges must be right; "
                          "1 GHz ≈ 0.3 % and below half the narrowest 6G channel"]))]
    col2 = [P("5 · Method", "h"),
            *fig("pipeline.png", COLW, "COMSOL calibrates PWE; PWE makes the training data; the AI proposes shapes; "
                 "every design is re-simulated."),
            *bullets(["<b>Forward ensemble</b>: 5 networks, 18 Fourier inputs → 2 band edges",
                      "<b>Tandem network</b>: target → one design (inverse net trained through the frozen forward net)",
                      "<b>CVAE + refine</b>: many designs per target, polished by gradient steps",
                      "<b>COMSOL correction</b>: a Gaussian process learns COMSOL − PWE (multi-fidelity)"]),
            P("6 · Is the fast solver right?", "h"),
            *fig("final_pwe_vs_comsol.png", COLW, f"PWE vs COMSOL on {N['pweN']} gapped cells: {N['pweLo']} / {N['pweUp']} "
                 f"GHz (f<sub>lower</sub> / f<sub>upper</sub>); with the correction {N['mfCorrLo']} / {N['mfCorrUp']} GHz."),
            P("7 · Inverse design results", "h"),
            *table([["Inverse method", "Success", "Designs"],
                    ["Tandem network", f"{N['tandemSucc']} ± {N['tandemSd']} %", "1"],
                    ["CVAE + refine (ours)", f"{N['crSucc']} ± {N['crSd']} %", f"~{N['crDistinct']}"]],
                   [COLW * 0.5, COLW * 0.28, COLW * 0.22]),
            P(f"100 unseen targets, every design re-simulated, 5 repeated runs. Forward network: {N['fwdLo']} / {N['fwdUp']} "
              f"GHz mean error, 95 % within {N['fwdPlo']} / {N['fwdPup']} GHz."),
            *fig("showcase.png", COLW, "Different PWE-verified designs for the same target, 300 / 330 GHz: choices for "
                 "fabrication.")]
    col3 = [P("8 · Proof in COMSOL", "h"),
            *fig("comsol_verify.png", COLW, f"{N.get('verOK', '?')} of {N.get('verN', '?')} AI designs within 1 GHz in COMSOL; "
                 "misses are the known PWE offset, which the correction removes."),
            P("9 · Do we still need PWE?", "h"),
            *fig("comsol_vs_pwe.png", COLW, f"{N['nFreeTest']} freeform COMSOL cells, cross-validated."),
            P(N["verdict"]),
            *box(bullets(["<b>Freeform shapes</b>: the AI designs any three-fold hole, not just two triangle sizes",
                          f"<b>Checked by physics</b>: {N['tandemSucc']} % of unseen targets met within 1 GHz; "
                          f"{N.get('verOK', '?')} of {N.get('verN', '?')} designs confirmed in COMSOL",
                          "<b>Cheap + accurate</b>: thousands of fast PWE cells, corrected by COMSOL runs"]), color=colors.HexColor("#eaf6ee")),
            P("10 · Next (Phase II)", "h"),
            *bullets(["Design the <b>topological edge-state waveguide</b> (two crystals side by side)",
                      "<b>Active learning</b> with COMSOL in the loop", "3D slab and fabrication-tolerance checks"])]
    for i, flow in enumerate((col1, col2, col3)):
        x = M + i * (COLW + GAP)
        Frame(x, ybot, COLW, colh, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, showBoundary=0) \
            .addFromList([KeepInFrame(COLW, colh, flow, mode="shrink")], c)
    c.setFont("Body", 18)
    c.setFillColor(MUTED)
    c.drawCentredString(W / 2, 12 * mm, "Code: Python, PyTorch, scikit-learn · Simulations: COMSOL Multiphysics 6.4 · "
                                        "Web app: streamlit run app.py")
    c.showPage()
    c.save()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
