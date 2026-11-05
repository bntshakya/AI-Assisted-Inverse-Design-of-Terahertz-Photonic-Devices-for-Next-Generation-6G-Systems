"""Build the option-B study guide PDF -> docs/OptionB_Study_Guide.pdf

Run from the Project folder after validate_pwe.py, gen_dataset.py, run_optionb.py, make_figures.py:
  python src/shapes/build_guide.py
Numbers in the results chapters are read from results/*.json, so rebuilding after new data
(e.g. the circle COMSOL sweep) updates them.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,  # noqa: E402
                                SimpleDocTemplate, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents  # noqa: E402

OUT = "docs/OptionB_Study_Guide.pdf"
FIG = "results/optionb/figs"
EQ_DIR = "results/optionb/eq"
FONT_DIR = "C:/Windows/Fonts"

# ------------------------------------------------------------------------------ fonts and styles
for name, f in (("Body", "calibri.ttf"), ("Body-Bold", "calibrib.ttf"), ("Body-Italic", "calibrii.ttf"),
                ("Body-BoldItalic", "calibriz.ttf"), ("Mono", "consola.ttf"), ("Sym", "segoeui.ttf")):
    pdfmetrics.registerFont(TTFont(name, os.path.join(FONT_DIR, f)))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-BoldItalic")

INK, ACCENT, MUTED = colors.HexColor("#1f2933"), colors.HexColor("#2563a8"), colors.HexColor("#6b7280")
S = {
    "title": ParagraphStyle("title", fontName="Body-Bold", fontSize=24, leading=29, textColor=INK, alignment=TA_CENTER),
    "subtitle": ParagraphStyle("subtitle", fontName="Body", fontSize=13, leading=17, textColor=MUTED, alignment=TA_CENTER),
    "h1": ParagraphStyle("h1", fontName="Body-Bold", fontSize=17, leading=21, textColor=ACCENT, spaceBefore=6, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="Body-Bold", fontSize=13, leading=16, textColor=INK, spaceBefore=10, spaceAfter=4),
    "h3": ParagraphStyle("h3", fontName="Body-Bold", fontSize=11, leading=14, textColor=INK, spaceBefore=6, spaceAfter=2),
    "p": ParagraphStyle("p", fontName="Body", fontSize=10.5, leading=14.5, textColor=INK, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", fontName="Body", fontSize=10.5, leading=14, textColor=INK, leftIndent=14,
                             bulletIndent=4, spaceAfter=2),
    "cap": ParagraphStyle("cap", fontName="Body-Italic", fontSize=9, leading=11.5, textColor=MUTED, spaceAfter=8),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=8.3, leading=10.5, textColor=INK, backColor=colors.HexColor("#f3f4f6"),
                           borderPadding=5, leftIndent=4, spaceBefore=3, spaceAfter=7),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=9, leading=11, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Body-Bold", fontSize=9, leading=11, textColor=colors.white),
    "box": ParagraphStyle("box", fontName="Body", fontSize=10.2, leading=13.8, textColor=INK),
    "toc1": ParagraphStyle("toc1", fontName="Body-Bold", fontSize=11, leading=15, leftIndent=10),
    "toc2": ParagraphStyle("toc2", fontName="Body", fontSize=10, leading=13, leftIndent=26),
}
W = A4[0] - 4.2 * cm


# ------------------------------------------------------------------------------ helpers
def eq(tex, size=13, name=None):
    """Render a LaTeX-like formula (matplotlib mathtext) to PNG and return a centred Image."""
    os.makedirs(EQ_DIR, exist_ok=True)
    name = name or f"eq{abs(hash(tex)) % 10 ** 10}.png"
    path = os.path.join(EQ_DIR, name)
    plt.rcParams["mathtext.fontset"] = "cm"
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=size)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04, transparent=True)
    plt.close(fig)
    from PIL import Image as PI
    w, h = PI.open(path).size
    scale = min(1.0, (W - 10) / (w / 300 * 72))  # never wider than the text column
    img = Image(path, width=w / 300 * 72 * scale, height=h / 300 * 72 * scale)
    img.hAlign = "CENTER"
    return [Spacer(1, 2), img, Spacer(1, 4)]


def P(t):
    return Paragraph(t, S["p"])


def H1(t):
    return Paragraph(t, S["h1"])


def H2(t):
    return Paragraph(t, S["h2"])


def H3(t):
    return Paragraph(t, S["h3"])


def bullets(items):
    return [Paragraph(i, S["bullet"], bulletText="•") for i in items]


def numbered(items):
    return [Paragraph(i, S["bullet"], bulletText=f"{k}.") for k, i in enumerate(items, 1)]


def code(t):
    return Preformatted(t.strip("\n"), S["code"])


def fig(name, width=W, caption=None):
    path = os.path.join(FIG, name)
    if not os.path.exists(path):
        return [P(f"<i>[figure {name} not generated yet: run make_figures.py]</i>")]
    from PIL import Image as PI
    w, h = PI.open(path).size
    img = Image(path, width=width, height=width * h / w)
    out = [img]
    if caption:
        out.append(Paragraph(caption, S["cap"]))
    return [KeepTogether(out)]


def table(rows, widths=None, header=True, font=9):
    data = [[Paragraph(str(c), S["cellb"] if (header and i == 0) else S["cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c7ccd3")),
          ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), ACCENT)]
    for i in range(1 if header else 0, len(rows)):
        if i % 2 == 0:
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f5f7fa")))
    t.setStyle(TableStyle(st))
    return [t, Spacer(1, 8)]


def box(title, text, color="#e8f0fb"):
    inner = [Paragraph(f"<b>{title}</b>", S["box"]), Spacer(1, 2)]
    inner += [Paragraph(x, S["box"]) for x in (text if isinstance(text, list) else [text])]
    t = Table([[inner]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(color)),
                           ("BOX", (0, 0), (-1, -1), 0.6, ACCENT), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return [Spacer(1, 3), t, Spacer(1, 8)]


def load(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except OSError:
        return default


def f2(x, d=2):
    return f"{x:.{d}f}"


class Doc(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in ("h1", "h2"):
            level = 0 if flowable.style.name == "h1" else 1
            text = flowable.getPlainText()
            key = f"sec{abs(hash(text))}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
            self.notify("TOCEntry", (level, text, self.page, key))


def on_page(canv, doc):
    canv.saveState()
    canv.setFont("Body", 8.5)
    canv.setFillColor(MUTED)
    if doc.page > 1:
        canv.drawString(2.1 * cm, A4[1] - 1.3 * cm, "AI-assisted inverse design of THz valley photonic crystals — option B study guide")
        canv.drawRightString(A4[0] - 2.1 * cm, 1.2 * cm, f"{doc.page}")
        canv.setStrokeColor(colors.HexColor("#d5dae1"))
        canv.line(2.1 * cm, A4[1] - 1.45 * cm, A4[0] - 2.1 * cm, A4[1] - 1.45 * cm)
    canv.restoreState()


# ------------------------------------------------------------------------------ content
def story():
    val = load("results/pwe_validation.json", {})
    res = load("results/optionb/results.json", {})
    eps = val.get("eps_si", 8.914)
    pf = val.get("per_family", {})
    nf = len(pd.read_csv("data/pwe/fourier.csv")) if os.path.exists("data/pwe/fourier.csv") else 0
    nl = len(pd.read_csv("data/pwe/library.csv")) if os.path.exists("data/pwe/library.csv") else 0
    cs = circle_stats()

    s = []
    # ---------------------------------------------------------------- title page
    s += [Spacer(1, 3.2 * cm), Paragraph("AI-Assisted Inverse Design of<br/>Terahertz Valley Photonic Crystals", S["title"]),
          Spacer(1, 10), Paragraph("Study guide: freeform hole shapes (option B), the data, the physics, the code, the results", S["subtitle"]),
          Spacer(1, 26), Paragraph("M.Tech project P29 — Binit Shakya, IIT Guwahati<br/>Supervisors: Dr. Debabrata Sikdar, Dr. Teena Sharma", S["subtitle"]),
          Spacer(1, 8), Paragraph("Prepared 28 October 2026, for the first-week-of-November evaluation", S["subtitle"]),
          Spacer(1, 1.2 * cm)]
    s += fig("unit_cells.png", W, "The four kinds of unit cell this guide talks about. Grey = silicon, white = air holes, "
             "dashed rhombus = one unit cell with its two holes A and B.")
    s += [PageBreak(), Paragraph("Contents", ParagraphStyle("h1x", parent=S["h1"]))]
    toc = TableOfContents()
    toc.levelStyles = [S["toc1"], S["toc2"]]
    s += [toc, PageBreak()]

    # ---------------------------------------------------------------- 1 project in one page
    s += [H1("1. The project in one page")]
    s += [P("<b>Goal.</b> 6G links will use carrier frequencies around 300 GHz (the IEEE 802.15.3d band is 252.72–321.84 GHz). "
            "On a chip, terahertz waves can be guided with almost no loss and around sharp bends by <i>topological</i> "
            "waveguides made from <b>valley photonic crystals (VPCs)</b>: a silicon slab with a periodic pattern of air holes. "
            "The holes decide the <b>photonic band gap</b> — the frequency range in which no wave can travel through the "
            "crystal — and the topological edge channel lives inside that gap. So the design question is:"),
          box("The inverse-design question", "Given a target band gap (lower edge f<sub>lower</sub>, upper edge f<sub>upper</sub>, in GHz), "
              "<b>what hole shapes produce it?</b> The usual way is trial and error in COMSOL. The project replaces that with "
              "machine learning that goes directly from target to shape."),
          fig("pipeline.png", W, "The option-B pipeline. Orange = COMSOL (slow, accurate), blue = fast solver and data, green = "
              "machine learning, purple = checks and the loop back."),
          P("<b>Where the project was.</b> Until now the hole shape was fixed (a triangle with rounded corners) and only its two "
            "side lengths L1, L2 were designed, from 64 COMSOL simulations. With 2 inputs and 2 outputs that problem turned out to "
            "be one-to-one, so a plain Gaussian-process fit beat every neural network (chapter 4)."),
          P("<b>What option B adds.</b> The hole may now have <i>any</i> shape, as long as it keeps the three-fold symmetry the "
            "physics needs. Every shape — circle, rounded or sharp triangle, rotated triangle, hexagon-like, trefoil, or a mix with "
            "a different shape in each hole — is written as the same list of 18 numbers (a Fourier description of the hole outline). "
            "Now many different shapes give the same band gap, which is exactly the situation the tandem network and generative "
            "models were invented for."),
          H3("What was built and found (details in the chapters)")]
    s += bullets([
        "<b>The real hole geometry</b> of the old COMSOL data, worked out from the group's slides and confirmed by simulation: the "
        "rounded hole is a triangle of side L <i>intersected with a disk</i> of radius 2L/(3√3) centred on the hole (chapter 3).",
        f"<b>A fast band solver</b> (plane-wave expansion, ~0.15 s per shape instead of COMSOL's seconds-to-minutes) that reproduces "
        f"both triangle COMSOL datasets to within about 1 GHz once a single number is fitted: the silicon permittivity, ε = {eps:.2f} "
        f"(not 11.7). " + (f"It then predicts the new circle dataset, which it was never fitted on, just as well "
                           f"(MAE {cs['mae'][0]:.2f} / {cs['mae'][1]:.2f} GHz). " if cs else "") + "That strongly suggests the COMSOL 2D model uses an effective index of n ≈ {eps ** 0.5:.2f} for the 200 µm slab — "
        "please confirm in COMSOL (chapter 6).",
        "<b>The 2-k-point shortcut still holds</b> for freeform shapes: in every one of several hundred random shapes the band edges "
        "sit exactly at the M or K point, checked against the full k-path and a 2D grid over the Brillouin zone. So COMSOL needs only "
        "2 eigen-solves per shape (chapter 6).",
        f"<b>A new dataset</b> of {nf + nl:,} fabricable unit cells ({nf:,} freeform Fourier shapes and {nl:,} mixed shapes) labelled "
        "by the fast solver (chapter 7).",
        "<b>A forward surrogate</b> (shape → band edges) that works for any hole family, a <b>multi-fidelity correction</b> learned "
        "from the COMSOL datasets, and <b>four inverse-design methods</b> compared honestly: every proposed shape is checked for "
        "fabrication rules and re-simulated (chapters 8–10).",
        "<b>A design gallery:</b> several different, verified hole shapes for one target band gap, exported in COMSOL's sweep format "
        "so you can confirm them (chapter 10).",
        (f"<b>The round-cell (circular-hole) data is in:</b> {cs['n']} COMSOL cells at k = 1, 2, {cs['n_gap']} with a gap "
         f"(up to {cs['gap_max']:.1f} GHz). It joins the test set and the multi-fidelity correction (chapters 3, 6, 9)."
         if cs else
         "<b>Ready for the round-cell (circular-hole) data:</b> one command adds it, and it is checked against the solver at once "
         "(chapter 11).")])
    s += [PageBreak()]

    # ---------------------------------------------------------------- 2 physics
    s += [H1("2. Physics you need")]
    s += [H2("2.1 Photonic crystals and band diagrams"),
          P("A photonic crystal is a material whose refractive index repeats periodically, here silicon with air holes repeated "
            "on a lattice with period a = 242.5 µm. Light in such a structure obeys Maxwell's equations; in 2D, for the polarization "
            "used by your COMSOL model (magnetic field H<sub>z</sub> out of the plane, electric field in the plane — called TE "
            "here), they reduce to one scalar <i>master equation</i>:")]
    s += eq(r"-\nabla\cdot\left(\frac{1}{\varepsilon(\mathbf{r})}\,\nabla H_z(\mathbf{r})\right)=\left(\frac{\omega}{c}\right)^2 H_z(\mathbf{r})")
    s += [P("This is an eigenvalue problem: only certain frequencies ω are allowed. Because ε(r) is periodic, Bloch's theorem says "
            "the solutions are plane waves times periodic functions,")]
    s += eq(r"H_z(\mathbf{r}) = e^{i\mathbf{k}\cdot\mathbf{r}}\,u_{\mathbf{k}}(\mathbf{r}),\qquad u_{\mathbf{k}}(\mathbf{r}+\mathbf{R})=u_{\mathbf{k}}(\mathbf{r})")
    s += [P("so we only need to solve inside <b>one unit cell</b> with a phase shift e<super>ik·R</super> between opposite faces "
            "(COMSOL's <i>Floquet periodicity</i>). For each wave vector <b>k</b> we get a list of frequencies f<sub>1</sub>(k) &lt; "
            "f<sub>2</sub>(k) &lt; ... — the <b>bands</b>. Plotting them along a path of k values gives the <b>band diagram</b>. "
            "If no band exists between two frequencies for <i>any</i> k, that range is a <b>complete band gap</b>:"),]
    s += eq(r"f_{\rm lower}=\max_{\mathbf{k}} f_1(\mathbf{k}),\qquad f_{\rm upper}=\min_{\mathbf{k}} f_2(\mathbf{k}),\qquad \Delta f = f_{\rm upper}-f_{\rm lower}")
    s += [P("These two numbers are the targets of the whole project. If Δf &lt; 0 the bands overlap and there is no complete gap.")]
    s += [H2("2.2 The lattice, the two holes and the k-path"),
          P("The lattice is triangular with primitive vectors a<sub>1</sub> = a(1, 0) and a<sub>2</sub> = a(1/2, √3/2). Each cell "
            "holds two holes, at the two sites of a honeycomb: A = (a<sub>1</sub>+a<sub>2</sub>)/3 and B = 2(a<sub>1</sub>+a<sub>2</sub>)/3. "
            "Their distance is a/√3 = 140.0 µm — which is why your L values are multiples of 140 µm. Each A hole has three "
            "nearest B neighbours at 120° from each other, and vice versa."),
          P("The reciprocal lattice has vectors b<sub>1</sub>, b<sub>2</sub> with a<sub>i</sub>·b<sub>j</sub> = 2πδ<sub>ij</sub>. "
            "Its Wigner–Seitz cell, the <b>first Brillouin zone</b>, is a hexagon with special points:")]
    s += eq(r"\Gamma=(0,0),\qquad M=\left(\frac{\pi}{a},\frac{\pi}{\sqrt{3}a}\right),\qquad K=\left(\frac{4\pi}{3a},0\right)")
    s += [P("Your COMSOL sweep parameter k runs along Γ → M → K → Γ, with k = 0 at Γ, 1 at M, 2 at K and 3 back at Γ (281 points, "
            "0.1 to 2.9). Corners K and K′ of the hexagon are the two <b>valleys</b>.")]
    s += [H2("2.3 Dirac cones and the valley-Hall effect"),
          P("When the two holes are identical the cell has six-fold rotation symmetry (C<sub>6</sub>), which includes inversion "
            "(r → −r). Then bands 1 and 2 touch at K and K′ in a cone-shaped crossing, a <b>Dirac point</b> — see the red dashed "
            "curves in the group's band diagram, and the dotted grey curve in figure 6.1. Near K the two bands behave like a 2×2 "
            "Dirac Hamiltonian:")]
    s += eq(r"H(\mathbf{q}) = v_D\,(q_x\sigma_x + q_y\sigma_y) + m\,\sigma_z,\qquad \Delta f \propto 2|m|")
    s += [P("Making the holes different (L1 ≠ L2) breaks inversion but keeps three-fold symmetry (C<sub>3</sub>). This adds the "
            "'mass' m, which opens a gap at K and K′. Its sign decides the <b>valley Chern number</b> C<sub>K</sub> = sgn(m)/2. "
            "Swapping which hole is big flips the sign: VPC-A and VPC-B in the group's slides are the same crystal with the holes "
            "swapped, with opposite valley Chern numbers. Where they meet, the difference in topology forces a guided "
            "<b>edge state</b> inside the gap that cannot back-scatter at sharp 120° bends — the reason VPCs are attractive for 6G "
            "chips."),
          box("Two rules that follow (they shape option B)", [
              "1. <b>Each hole must keep C<sub>3</sub> symmetry</b> about its own centre. A random blob breaks C<sub>3</sub>; the "
              "Dirac points are then no longer protected at K and the gap you get is not a valley-Hall gap.",
              "2. <b>The two holes must differ</b> (that is what breaks inversion). Swapping them gives exactly the same band "
              "edges but the opposite topology, so the dataset can be doubled for free and the design network should treat a "
              "swapped design as 'the same answer'."])]
    s += [H2("2.4 Two useful invariances"),
          P("<b>Scale.</b> For a 2D crystal of non-dispersive materials, scaling every length by s divides every frequency by s. "
            "A design found for a = 242.5 µm can be moved to another 6G channel by scaling a and the holes together. "
            "<b>Mirror.</b> Reflecting the whole crystal about an A–B bond line maps K to K′; by time reversal f(K) = f(K′), so the "
            "band edges do not change. In the Fourier description this flips the sign of all sine coefficients (chapter 5)."),
          PageBreak()]

    # ---------------------------------------------------------------- 3 the data
    s += [H1("3. The data you have: the COMSOL datasets")]
    s += [H2("3.1 What COMSOL computed"),
          P("The COMSOL 6.4 model is 2D: a single unit cell with Floquet-periodic boundaries, Electromagnetic Waves (emw), "
            "eigenfrequency study with 2 eigenfrequencies, swept over two size parameters and k. The two original sweeps use the "
            "same 8 × 8 grid (L1 = 140–189 µm, L2 = 91–140 µm, 7 µm steps) and the full 281-point k-path. On 28 October two "
            "more runs were added, both at k = 1 and 2 only (section 3.5):"),
          ]
    rows = [["File", "Holes", "k-points", "Gapped cells (Δf > 0.1 GHz)", "Gap range"],
            ["parameter_sweep_values (... r = 2L/(3√3)).csv", "rounded triangles", "281", "61 of 64", "0.6–43.6 GHz"],
            ["r1=r2=0 ; varied L1 & L2 (Parametric sweep).csv", "sharp triangles", "281", "63 of 64", "3.0–40.2 GHz"]]
    if cs:
        rows += [["r=a_divby(3sqrt(3))_(Parametric sweep at k=1,2).csv (also split into k=1 and k=2 files)",
                  "circles (7 × 7 radii)", "2", f"{cs['n_gap']} of {cs['n']}", f"{cs['gap_min']:.1f}–{cs['gap_max']:.1f} GHz"],
                 ["parameter_sweep_values (... r = 2L/(3√3)) (k=1,2).csv", "rounded triangles again, finer mesh", "2",
                  "61 of 64", "a check of row 1"]]
    s += table(rows, [5.6 * cm, 3.0 * cm, 1.6 * cm, 3.3 * cm, 2.6 * cm])
    s += [H2("3.2 Why 29.5 million rows are only 128 designs"),
          P("The export has one row <b>per mesh node</b>: X, Y, L1, L2, k, λ, emw.freq. The eigenfrequency is a single number "
            "for the whole cell, so it is repeated at all ~820 nodes. Counting carefully:")]
    s += table([["Level", "Count", "Meaning"],
                ["rows in each file", "~14.7 million", "nodes × k-points × 2 eigenfrequencies × 64 geometries"],
                ["unique values per file", "35,968 = 64 × 281 × 2", "one frequency per (geometry, k, band)"],
                ["designs per file", "64", "one (L1, L2) each → one (f_lower, f_upper)"],
                ["both files", "128 designs (124 with a gap)", "the training set for the old problem"]],
               [4 * cm, 5 * cm, 7 * cm])
    s += [P("src/utils/process_bands.py reduces both files to bands.csv (35,968 rows each file: geometry, k, f1, f2) and "
            "band_edges.csv (128 rows). Always split train/test <i>by geometry</i>, never by row, or the same design leaks into "
            "both sets.")]
    s += [H2("3.3 The exact hole shapes (worked out for option B)"),
          P("To put old and new shapes in one description we need the old outlines exactly. The dataset title says r = 2L/(3√3); "
            "the slides show the corner arcs. A normal fillet of radius r on a triangle of side L is impossible here (the largest "
            "fillet that fits is L/(2√3) = 0.29 L, but 2/(3√3) = 0.385). Measuring the schematic in the group's slides shows the "
            "corner arcs are centred on the <i>hole centre</i>: the rounded hole is the triangle <b>intersected with a disk</b>. "
            "In polar form, around the hole centre:")]
    s += eq(r"r_{\rm sharp}(\theta)=\frac{h}{\cos\theta_f},\qquad r_{\rm rounded}(\theta)=\min\left(\frac{h}{\cos\theta_f},\ \rho\right),\qquad h=\frac{L}{2\sqrt{3}},\ \ \rho=\frac{2L}{3\sqrt{3}}=\frac{4h}{3}")
    s += [P("θ<sub>f</sub> is the angle to the nearest flat side. The flat sides face the three neighbouring holes and the "
            "corners point into the hexagon centres, which leaves a silicon wall of at least 45 µm everywhere in the sweep. The "
            "fast solver of chapter 6 reproduces COMSOL to within ~1 GHz with exactly this geometry — a strong check that it is "
            "right. (Ask the person who built the model to confirm; the formula is in src/shapes/geometry.py.)")]
    s += [H2("3.4 Where the band edges sit"),
          P("In all 128 old cells band 1 peaks at K. Band 2 bottoms out at M (rounded, 63 of 64) or K (sharp, 63 of 64). "
            "So computing only k = 1 (M) and k = 2 (K) reproduces every band edge exactly — 140 times cheaper than 281 k-points. "
            "Chapter 6 shows this stays true for freeform shapes.")]
    s += circle_data_section(cs)
    s += [PageBreak()]

    # ---------------------------------------------------------------- 4 stage 1 recap
    s += [H1("4. Stage 1 recap: the L1/L2 benchmark")]
    s += [P("Before option B, three inverse-design methods were compared on the old data (10 random splits, errors judged by a "
            "per-family Gaussian-process 'oracle' standing in for COMSOL; src/benchmark/)."),
          *bullets(["<b>Direct inverse NN</b>: a network trained to map target → (L1, L2) directly.",
                    "<b>Tandem NN</b> (Liu et al. 2018): first train a forward network (L1, L2) → edges, freeze it, then train the "
                    "inverse network through it so the loss is measured on the <i>response</i>, not on the geometry.",
                    "<b>GP + gradient</b>: fit a Gaussian process forward model, then solve for the geometry with multi-start "
                    "least squares."]),
          ]
    s += table([["Task", "Method", "Rounded only", "Both families, rounded test", "Both families, sharp test"],
                ["edges (one-to-one)", "direct", "0.508", "0.577", "0.537"], ["", "tandem", "0.642", "0.700", "0.697"],
                ["", "GP + gradient", "<b>0.007</b>", "<b>0.058</b>", "<b>0.060</b>"],
                ["gap only (many-to-one)", "direct", "1.671 (max 30.5)", "1.443", "0.633"],
                ["", "tandem", "0.165", "0.170", "0.090"], ["", "GP + gradient", "<b>0.001</b>", "<b>0.033</b>", "<b>0.007</b>"]],
               [3.4 * cm, 2.6 * cm, 3 * cm, 3.5 * cm, 3.5 * cm])
    s += [Paragraph("Mean error of the achieved target, GHz.", S["cap"]),
          box("Lessons that motivate option B", [
              "• With 2 design numbers and 2 targets the map is one-to-one (its Jacobian never changes sign), so there is no "
              "non-uniqueness for a tandem to solve, and 61 smooth points are best handled by a Gaussian process.",
              "• The tandem's advantage appears as soon as the problem is many-to-one (gap-only task: 6–10× better than the direct "
              "network). Option B makes the real problem many-to-one: 10 or more shape numbers for 2 targets.",
              "• 64 points per family cannot support freeform shapes. More data is needed — chapter 6 shows how to get it cheaply."])]
    s += [PageBreak()]

    # ---------------------------------------------------------------- 5 option B shape description
    s += [H1("5. Option B: describing any C3 hole with a few numbers")]
    s += [H2("5.1 The symmetry selection rule"),
          P("Describe a hole by its outline in polar form r(θ) around its own centre, with θ = 0 pointing at a neighbouring hole "
            "of the other type (the hole's <b>local frame</b>). Any closed star-shaped outline has a Fourier series")]
    s += eq(r"r(\theta)=r_0+\sum_{n\geq1}\left[c_n\cos n\theta + s_n \sin n\theta\right]")
    s += [P("C<sub>3</sub> symmetry means r(θ + 120°) = r(θ). Substituting, each term picks up a factor e<super>i2πn/3</super>, "
            "which equals 1 only when n is a multiple of 3. So <b>only harmonics n = 3, 6, 9, 12, ... are allowed</b>; a cos 2θ "
            "term, for example, would stretch the hole into an oval and break C<sub>3</sub>:")]
    s += eq(r"r(\theta)=r_0+c_3\cos3\theta+s_3\sin3\theta+c_6\cos6\theta+s_6\sin6\theta\ (+\,c_9,s_9,c_{12},s_{12})")
    s += [P("<b>What each number does.</b> r<sub>0</sub> is the mean radius (sets the hole size and air fraction). c<sub>3</sub>, "
            "s<sub>3</sub> make the hole triangular: amplitude √(c<sub>3</sub>²+s<sub>3</sub>²), and the phase rotates it "
            "(c<sub>3</sub> &lt; 0, s<sub>3</sub> = 0 is a triangle whose flat side faces the neighbour, like the old holes). "
            "c<sub>6</sub>, s<sub>6</sub> sharpen or flatten the corners (hexagon-like, star-like). Higher harmonics add finer "
            "detail. Using (c<sub>n</sub>, s<sub>n</sub>) instead of (amplitude, angle) avoids the ambiguity that a triangle "
            "rotated by 60° equals a flipped triangle, so every shape has exactly one description.")]
    s += fig("fourier_descriptor.png", W, "Top: the exact outlines of the old rounded and sharp triangles (black) and their Fourier "
             "reconstructions with harmonics up to 3, 6 and 12. Bottom: shapes from the same formula. The red one uses cos 2θ and "
             "breaks C3.")
    s += [H2("5.2 One descriptor for every dataset"),
          P("Each hole is turned into 9 numbers (r<sub>0</sub>, c<sub>3</sub>, s<sub>3</sub>, c<sub>6</sub>, s<sub>6</sub>, "
            "c<sub>9</sub>, s<sub>9</sub>, c<sub>12</sub>, s<sub>12</sub>, in µm), computed numerically from the exact outline. "
            "A unit cell is 18 numbers (hole A then hole B). The old shapes become:")]
    s += table([["Old hole", "r0 / L", "c3 / r0", "c6 / r0", "c9 / r0", "c12 / r0"],
                ["rounded triangle (∩ disk)", "0.339", "−0.160", "0.002", "0.019", "−0.011"],
                ["sharp triangle", "0.363", "−0.275", "0.104", "−0.054", "0.032"],
                ["circle, radius R", "R / R = 1", "0", "0", "0", "0"]], [4.6 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm])
    s += [P("So the 'corner type' flag of stage 1 is no longer needed: a sharp triangle is simply a hole with a stronger 3rd "
            "harmonic and a non-zero 6th harmonic. A surrogate trained on these 18 numbers can, in principle, learn from rounded, "
            "sharp, circular and freeform holes at once, and a circle-hole COMSOL sweep adds directly to it.")]
    s += [H2("5.3 The design space for inverse design"),
          P("For designing new holes we use the first two harmonics for each hole: <b>10 numbers per cell</b> "
            "[r<sub>0</sub>, c<sub>3</sub>, s<sub>3</sub>, c<sub>6</sub>, s<sub>6</sub>]<sub>A</sub> + [...]<sub>B</sub>. They are "
            "sampled in a box that contains the old triangles: r<sub>0</sub> in [25, 75] µm, √(c<sub>3</sub>²+s<sub>3</sub>²) ≤ 0.35 "
            "r<sub>0</sub>, √(c<sub>6</sub>²+s<sub>6</sub>²) ≤ 0.12 r<sub>0</sub>, phases uniform. The networks work with "
            "u in (0, 1)<super>10</super> (r<sub>0</sub> = 25 + 50u, c<sub>3</sub> = 0.35 r<sub>0</sub>(2u − 1), ...), so a "
            "sigmoid output is always a shape inside the box. A second 'library' design space lets each hole be a circle, a "
            "rounded or sharp triangle (any size and rotation) or a Fourier shape — the 'mix of 3–4 shapes'.")]
    s += [H2("5.4 Fabrication and validity rules"),
          P("Random coefficients can produce cells that cannot be made, or that are not a valley crystal at all. Every cell is "
            "checked (geometry.check_cell) before it is simulated or proposed:")]
    s += bullets(["silicon wall between any two holes (A–B neighbours, A–A and B–B second neighbours, periodic images) "
                  "<b>≥ 25 µm</b> — the old data never goes below 45 µm;",
                  "hole radius everywhere <b>≥ 10 µm</b> (no slivers; also guarantees a simple, non-self-intersecting outline);",
                  "radius of curvature of convex corners <b>≥ 8 µm</b> (deep reactive-ion etching cannot cut sharper corners);",
                  "the holes do not overlap, and each stays C<sub>3</sub> by construction."])
    s += [P("About 40–60% of random draws pass. These limits are parameters (check_cell(min_wall=..., ...)); set them to your "
            "fabrication process. For inverse design the same rules are written as a smooth penalty P(u) (models.penalty) that "
            "is zero only for comfortably valid cells, so gradients can push a design back into the valid region.")]
    s += [H2("5.5 Different designs that are physically the same"),
          P("Two operations never change the band edges: swapping hole A and hole B (chapter 2.3) and mirroring the cell (all "
            "sine coefficients change sign). When we count 'how many different shapes reach a target' we treat these as the same "
            "design, so the diversity numbers in chapter 10 are honest."), PageBreak()]

    # ---------------------------------------------------------------- 6 PWE
    pr, ps = pf.get("rounded", {}), pf.get("sharp", {})
    s += [H1("6. A fast simulator: plane-wave expansion (PWE)")]
    s += [H2("6.1 Why"),
          P("Freeform shapes need thousands of labelled examples; 128 COMSOL runs are far too few, and thousands of COMSOL runs "
            "take days. Your model is 2D, and for a 2D photonic crystal the classic plane-wave expansion method is fast and "
            "accurate. It is used here as a <b>cheap, slightly less accurate simulator</b>: it labels thousands of shapes, the "
            "machine learning is trained on them, and only the final designs go back to COMSOL."),
          H2("6.2 How it works"),
          P("Expand the periodic part of the field in plane waves e<super>i(k+G)·r</super> over reciprocal lattice vectors "
            "G = m b<sub>1</sub> + n b<sub>2</sub>, and expand 1/ε the same way. The master equation becomes a matrix eigenproblem:")]
    s += eq(r"\sum_{\mathbf{G}'}(\mathbf{k}+\mathbf{G})\cdot(\mathbf{k}+\mathbf{G}')\;\left[\varepsilon^{-1}\right]_{\mathbf{G}\mathbf{G}'}\;h_{\mathbf{G}'}=\left(\frac{\omega}{c}\right)^2 h_{\mathbf{G}}")
    s += [P("The permittivity map of the cell is drawn on a 96 × 96 grid with 4 × 4 sub-pixel averaging (so smooth shape "
            "changes give smooth frequency changes) and Fourier-transformed with an FFT:")]
    s += eq(r"\varepsilon_{\mathbf{G}}=\frac{1}{A_{\rm cell}}\int_{\rm cell}\varepsilon(\mathbf{r})\,e^{-i\mathbf{G}\cdot\mathbf{r}}\,d^2r")
    s += [P("[ε<super>−1</super>] is the inverse of the matrix ε<sub>G−G′</sub> (Ho, Chan and Soukoulis 1990), which "
            "converges much faster than transforming 1/ε directly. With 439 plane waves the band edges change by less than "
            "0.4 GHz when going to 1,251 plane waves. One cell takes ≈ 0.05 s for the matrix plus ≈ 0.06 s per k-point; with M "
            "and K only, about 0.17 s on one CPU core, and the generator runs 10 cells in parallel. Code: src/shapes/pwe.py.")]
    s += [H2("6.3 Checking it against the COMSOL datasets"),
          P(f"With the slides' value ε = 11.7 the solver is about 13% too low at every k and for both shape families — the same "
            f"ratio everywhere, which is the signature of a different permittivity, not a wrong geometry. Fitting ε on every 4th "
            f"geometry gives <b>ε = {eps:.3f} (n = {eps ** 0.5:.3f})</b>; then all 128 cells are reproduced:")]
    if pr:
        s += table([["Family", "cells", "MAE f_lower", "MAE f_upper", "MAE gap", "gap correlation"],
                    ["rounded", pr["n_gapped"], f2(pr["MAE_f_lower_GHz"]), f2(pr["MAE_f_upper_GHz"]), f2(pr["MAE_gap_GHz"]), f2(pr["corr_gap"], 5)],
                    ["sharp", ps["n_gapped"], f2(ps["MAE_f_lower_GHz"]), f2(ps["MAE_f_upper_GHz"]), f2(ps["MAE_gap_GHz"]), f2(ps["corr_gap"], 4)]]
                   + ([["circle (not used to fit ε)", cs["n_gap"], f2(cs["mae"][0]), f2(cs["mae"][1]), f2(cs["mae"][2]),
                        f2(cs["corr"], 5)]] if cs else []),
                   [3.6 * cm, 1.4 * cm, 2.4 * cm, 2.4 * cm, 2.2 * cm, 2.8 * cm])
        s += [Paragraph("Errors in GHz (about 0.1–0.3% of the frequency).", S["cap"])]
        if cs:
            s += [P(f"The circle row is the real test. ε was fitted on triangles only, yet the {cs['n_gap']} gapped circle cells "
                    f"come out within {cs['max_err'][0]:.2f} GHz (f<sub>lower</sub>) and {cs['max_err'][1]:.2f} GHz "
                    f"(f<sub>upper</sub>) at worst. PWE is slightly high for circles ({cs['bias'][0]:+.2f} / "
                    f"{cs['bias'][1]:+.2f} GHz on average), so its gaps are {abs(cs['bias'][2]):.2f} GHz too narrow. It agrees with "
                    f"COMSOL on which cells have a gap in {cs['n'] - cs['disagree']} of {cs['n']} cells; the exception has a "
                    "gap close to zero.")]
    s += fig("pwe_validation.png", W, "(a) Full band diagram: COMSOL (black), PWE with the fitted ε (orange dashes, on top of "
             "black) and with ε = 11.7 (grey). (b, c) Band edges and gap widths of all 128 triangle cells"
             + (", plus the gapped circle cells (red), which were not used to fit ε." if cs else "."))
    s += box("What the ε = 8.9 result means — please check", [
        f"A 2D model that should mimic a 200 µm silicon slab often uses the slab's <i>effective index</i> instead of bulk silicon. "
        f"n ≈ {eps ** 0.5:.2f} is a plausible effective index for this slab near 300 GHz. Open the COMSOL model → Materials (or the "
        "emw node) and check the relative permittivity / refractive index. If it is 11.7, something else differs (for example an "
        "out-of-plane wave number), and the fitted ε simply absorbs it — the comparison above shows the PWE reproduces the model "
        "either way. In the thesis, describe it as 'calibrated to COMSOL'.",
        f"Published support: Shalaev et al. (Nature Nanotechnology 2019) reduce a silicon membrane to a 2D COMSOL model "
        f"in exactly this way and use n<sub>eff</sub> = 2.965 — close to the n = {eps ** 0.5:.3f} fitted here. Cite them "
        "when you explain the 2D model."], "#fff4e5")
    s += mk_section()
    s += [PageBreak()]

    # ---------------------------------------------------------------- 7 dataset
    s += [H1("7. The option-B dataset")]
    s += [P(f"src/shapes/gen_dataset.py draws random cells, keeps the valid ones and labels them with the PWE solver at M and K "
            f"(10 processes in parallel). Two sets were generated: <b>fourier</b> ({nf:,} cells, both holes Fourier shapes) and "
            f"<b>library</b> ({nl:,} cells, each hole independently a circle, rounded triangle, sharp triangle or Fourier shape, with "
            "random size and rotation). Each row stores the two hole families and parameters, the 18 Fourier features, f<sub>lower</sub>, "
            "f<sub>upper</sub>, the gap, where each edge sits (M or K), the thinnest wall and the air fraction.")]
    s += fig("dataset.png", W, "Top: random cells from the fourier (first row) and library (second row) sets. Bottom left: every "
             "cell as a point (f_lower, f_upper); cells above the diagonal have a complete gap. The COMSOL cells (orange, green) sit "
             "inside the new cloud. Bottom right: gap distribution.")
    s += [P("Two things to notice. First, freeform shapes reach much wider gaps than the old triangles (about 70 GHz versus 44 GHz) "
            "and a wider range of centre frequencies, which is the point of a larger design space. Second, a sizeable fraction of "
            "cells have no complete gap (Δf &lt; 0). They are kept for training: they teach the surrogate where gaps close, which "
            "the inverse methods need to know."), PageBreak()]

    # RESULTS-SECTIONS
    s += results_sections(res)
    s += closing_sections(res)
    return s


def circle_stats():
    """Numbers for the round-cell COMSOL sweep (data/comsol/extra_circle.csv), or {} if it is not there."""
    path = "data/comsol/extra_circle.csv"
    if not os.path.exists(path):
        return {}
    e = pd.read_csv(path)
    e["R1"] = [json.loads(p)["R"] for p in e.paramsA]
    e["R2"] = [json.loads(p)["R"] for p in e.paramsB]
    e["gap_pwe"] = e.f_upper_pwe - e.f_lower_pwe
    g = e[e.gap > 0.1]
    err = g[["f_lower_pwe", "f_upper_pwe", "gap_pwe"]].to_numpy() - g[["f_lower", "f_upper", "gap"]].to_numpy()
    eq_ = e[np.isclose(e.R1, e.R2)]
    thr = g.assign(d=g.R1 - g.R2).groupby("R1").d.min()
    return dict(n=len(e), n_gap=len(g), gap_min=g.gap.min(), gap_max=g.gap.max(),
                r1=(e.R1.min(), e.R1.max()), r2=(e.R2.min(), e.R2.max()),
                mae=np.abs(err).mean(0), bias=err.mean(0), max_err=np.abs(err).max(0),
                corr=np.corrcoef(g.gap, g.gap_pwe)[0, 1], disagree=int(((e.gap > 0.1) != (e.gap_pwe > 0.1)).sum()),
                gap_equal=float(eq_.gap.iloc[0]) if len(eq_) else None, thr=(thr.min(), thr.max()),
                table=e.pivot_table(index="R1", columns="R2", values="gap"))


def circle_data_section(cs):
    rr = load("results/optionb/comsol_rerun_check.json", {})
    s = [H2("3.5 The round-cell dataset and the k = 1, 2 re-run")]
    if not cs:
        return s + [P("The slides also describe a cell with two circular holes (r = a/(3√3) = 46.7 µm when symmetric). A "
                      "circle is the simplest C<sub>3</sub> shape, so it slots straight into option B; chapter 11 has the "
                      "one-line command to add it.")]
    s += [P(f"<b>Circles.</b> The round cell from the slides: two circular holes, r1 = {cs['r1'][0]:.1f}–{cs['r1'][1]:.1f} µm "
            f"on site A and r2 = {cs['r2'][0]:.1f}–{cs['r2'][1]:.1f} µm on site B, in 7 steps of 2.33 µm each. That is "
            "r1 = 1.0–1.3 and r2 = 0.7–1.0 times a/(3√3) = 46.7 µm — the same layout as the L1/L2 triangle grid. A circle is "
            "the simplest C<sub>3</sub> shape (every Fourier coefficient except r<sub>0</sub> is zero), so it slots straight "
            "into option B. The sweep was run at k = 1 and 2 only, as section 6.4 recommends. Gap widths in GHz "
            "(negative = no complete gap):")]
    t = cs["table"]
    rows = [["r1 \\ r2 (µm)"] + [f"{c:.1f}" for c in t.columns]]
    rows += [[f"{r:.1f}"] + [f"{v:.1f}" for v in t.loc[r]] for r in t.index]
    s += table(rows, [2.4 * cm] + [2.0 * cm] * len(t.columns), font=8.5)
    s += bullets([
        f"The gap grows with the difference of the radii, as for the triangles, up to {cs['gap_max']:.1f} GHz on this grid "
        "(the triangles reach 44 GHz).",
        f"Equal circles (r1 = r2 = 46.7 µm) have <b>no</b> complete gap ({cs['gap_equal']:.1f} GHz). The two bands still touch "
        "at K (the Dirac point, 326.8 GHz), but band 2 at M is lower (316.1 GHz), so the bands overlap in frequency. A gap "
        f"opens only when r1 exceeds r2 by about {cs['thr'][0]:.0f}–{cs['thr'][1]:.0f} µm, which is why {cs['n'] - cs['n_gap']} "
        f"of the {cs['n']} cells have none. Circles need a larger size difference than triangles to open a gap.",
        "In every circle cell band 1 tops out at K and band 2 bottoms out at M — COMSOL itself agrees with the shortcut.",
        f"The fast solver, fitted only on triangles, predicts these cells to within {cs['mae'][0]:.2f} / {cs['mae'][1]:.2f} GHz "
        "(section 6.3): the circle model uses the same permittivity as the triangle models."])
    if rr:
        md = rr["max_abs_diff_GHz"]
        s += [P(f"<b>Rounded triangles, re-run.</b> The rounded sweep was repeated at k = 1 and 2 only, with a finer mesh "
                f"(about {round(rr['mesh_nodes_per_solution_new'], -2):,.0f} mesh-node rows per solution instead of about 820). Its band edges match "
                f"the old {rr['n_k_old']}-point sweep for all {rr['n_cells']} cells to within {max(md['lower'], md['upper']):.3f} "
                f"GHz, and the gap to within {md['gap']:.3f} GHz. That confirms two things at once: COMSOL's two-point route "
                "gives the same band edges as the full path, and the old mesh was already converged (to about 0.01%). It is "
                "not added as training data, because the same cells are already in the dataset.")]
    return s


def _where(d):
    """{'2.0': 128} or {'K': 600, 'M': 58} -> 'K (128)'."""
    names = {"1.0": "M", "2.0": "K", "1": "M", "2": "K"}
    items = sorted(((names.get(str(k), str(k)), v) for k, v in d.items() if v), key=lambda kv: -kv[1])
    return ", ".join(f"{k if k in ('M', 'K') else 'k = ' + k} ({v})" for k, v in items)


def _dev(x):
    """Largest band-edge difference as text: 0 is exact, tiny values are below the 0.001 GHz print precision."""
    return "0 (exact)" if x == 0 else ("&lt; 0.001 GHz" if x < 5e-4 else f"{x:.3f} GHz")


def mk_section():
    """Section 6.4: why COMSOL needs only k = 1 (M) and k = 2 (K)."""
    mk = load("results/optionb/mk_check.json", {})
    co, mx, ci = mk.get("comsol", {}), mk.get("pwe_mixed", {}), mk.get("pwe_circles", {})
    rr = load("results/optionb/comsol_rerun_check.json", {})
    s = [H2("6.4 Why COMSOL needs only k = 1 (M) and k = 2 (K)"),
         P("The band edges are defined over <i>all</i> wavevectors: f<sub>lower</sub> is the highest point of band 1 and "
           "f<sub>upper</sub> the lowest point of band 2 anywhere in the Brillouin zone. Your COMSOL sweep samples 281 points "
           "along the path Γ → M → K → Γ (k = 0 → 1 → 2 → 3). The shortcut used everywhere in this project goes further and "
           "solves only at M and K, about 140 times less work. This section explains why that is justified, what the "
           "literature does and does not promise, and how it was checked."),
         H3("What the literature says")]
    s += bullets([
        "<b>Why K.</b> With two identical holes the honeycomb crystal has six-fold symmetry and bands 1 and 2 touch at K "
        "(a Dirac point). Making the holes different breaks inversion symmetry, the touching point splits and the gap opens "
        "<i>at K</i>, so the top of band 1 sits at K. This is the standard valley-Hall picture (Ma and Shvets 2016; "
        "Shalaev et al. 2019; Yang et al. 2020).",
        "<b>Why M.</b> Band 2 is not necessarily lowest at K: away from K it can dip, and its lowest point can be M instead. "
        "In your COMSOL data it is M for the rounded triangles (63 of 64 cells) and K for the sharp ones (63 of 64). "
        "Shalaev et al. (2019) chose triangular holes precisely because they push band 2 up at M. With circular holes in "
        "their silicon slab, band 2 dipped between K and M and the gap became smaller and indirect — the case a two-point "
        "shortcut would miss.",
        "<b>No guarantee.</b> Band-structure codes usually scan only the edge of the irreducible Brillouin zone, but that "
        "is a rule of thumb, not a theorem. Harrison, Kuchment, Sobolev and Winn (2007) built periodic structures whose band "
        "edges lie deep inside the zone. Craster et al. (2012) warn of the same for waves in periodic media. Maurin et al. "
        "(2018) ran hundreds of simulations covering all 17 two-dimensional symmetry groups and found the edge-only search "
        "reliable for highly symmetric lattices but less so for low-symmetry ones.",
        "<b>Where your crystals stand.</b> The old triangle cells have three mirror lines, and with time reversal their "
        "irreducible zone is the triangle Γ–M–K: the COMSOL path is exactly its boundary. A freeform hole with sine terms "
        "has no mirror line (plane group p3, rotations only — the low-symmetry case). Its irreducible zone is the whole "
        "60° wedge Γ–K–K′, which contains points the COMSOL path never visits. So the shortcut has to be <i>checked</i>, "
        "especially for freeform shapes and for circles."])
    s += [H3("Three checks")]
    if co and mx and ci:
        s += [P("Each check compares the band edges from M and K alone with the band edges from a much denser set of "
                "wavevectors. The script is src/shapes/check_mk_shortcut.py; the numbers are in results/optionb/mk_check.json.")]
        s += [KeepTogether(table([["Check", "Cells", "Compared against", "Largest difference", "Top of band 1 at", "Bottom of band 2 at"],
                    ["COMSOL: rounded and sharp triangles", co["n_cells"], f"the full {co['n_k']}-point COMSOL path",
                     _dev(co['max_dev_GHz']), _where(co["lower_at"]), _where(co["upper_at"])],
                    ["PWE: random mixed cells (freeform, rotated triangles, circles)", mx["n_cells"],
                     "29-point path + 27-point grid over the wedge", _dev(mx['max_dev_GHz']),
                     _where(mx["lower_at"]), _where(mx["upper_at"])],
                    [f"PWE: circle pairs, radii {ci['r_range_um'][0]}–{ci['r_range_um'][1]} µm", ci["n_cells"],
                     f"{int(round(2.96 / ci['path_step'])) + 1}-point path + {ci['wedge_points']}-point wedge grid",
                     _dev(ci['max_dev_GHz']), _where(ci["lower_at"]), _where(ci["upper_at"])]]
                   + ([["COMSOL: rounded triangles re-run at k = 1, 2 only (finer mesh)", rr["n_cells"],
                        f"the old {rr['n_k_old']}-point sweep", f"{max(rr['max_abs_diff_GHz'].values()):.3f} GHz (mesh)",
                        "–", "–"]] if rr else []),
                   [3.9 * cm, 1.2 * cm, 3.6 * cm, 2.2 * cm, 2.6 * cm, 3.0 * cm], font=8.5))]
        s += [Paragraph(f"In total {mk['n_pwe_total']} PWE cells ({mk['n_circle_pairs_total']} of them circle pairs) and "
                        f"all {co['n_cells']} COMSOL cells: no band edge lies anywhere other than M or K.", S["cap"])]
    else:
        s += [P("<i>Run src/shapes/check_mk_shortcut.py, then rebuild this guide.</i>")]
    wm = mk.get("wedge_map", {})
    # K and K' are equivalent (time reversal), so either corner counts
    K2 = [wm["K"][0] / 2, wm["K"][0] * 3 ** 0.5 / 2] if wm else None
    at_K = bool(wm) and (np.allclose(wm["band1_max_at"], wm["K"], atol=1e-6) or np.allclose(wm["band1_max_at"], K2, atol=1e-6))
    at_M = bool(wm) and np.allclose(wm["band2_min_at"], wm["M"], atol=1e-6)
    s += fig("mk_shortcut.png", W,
             "(a–c) Bands along the COMSOL path for three kinds of cell. Downward triangle = highest point of band 1, "
             "upward triangle = lowest point of band 2, green = the gap. In (a) the thick grey line underneath is COMSOL itself. (d, e) The same two bands "
             "of the freeform cell over the whole wedge Γ–K–K′ (black triangle; grey hexagon = Brillouin zone). "
             + ("The highest point of band 1 is a K corner (K and K′ give the same frequency) and the lowest point of band 2 "
                "is M, the middle of the K–K′ edge — nowhere inside the wedge." if at_K and at_M else
                "The marked points show where the extremes of this cell lie."))
    s += [P("How to read the figure. Along the path (top row) band 1 rises to its maximum at K and, in these three cells, "
            "band 2 falls to its minimum at M (for sharp triangles it is usually K; either way a point the shortcut "
            "solves). The bottom row answers the question the path "
            "cannot: could a band edge hide <i>off</i> the path, inside the wedge? The colour maps show it does not — "
            "band 1 rises steadily towards the corners K and K′, and band 2 is lowest at M on the zone edge.")]
    if ci:
        s += box("For your round-cell (circle) data", [
            f"All {ci['n_cells']} circle pairs tested (radii {ci['r_range_um'][0]}–{ci['r_range_um'][1]} µm, walls ≥ 20 µm) "
            f"have the top of band 1 at K and the bottom of band 2 at M, with a largest difference of {_dev(ci['max_dev_GHz'])}. The dip "
            "between K and M that Shalaev et al. saw for circles in a 3D slab does not appear in your 2D model.",
            f"{ci['n_no_gap']} of these pairs (nearly equal radii) have no complete gap at all: band 2 at M falls below band 1 "
            "at K. The two-point shortcut still reports this correctly, as a negative gap.",
            "Your COMSOL circle sweep (section 3.5, k = 1 and 2 only) agrees with the fast solver at M and K to within about "
            "1 GHz, and in COMSOL too band 1 tops out at K and band 2 bottoms out at M in every cell. The one direct COMSOL check "
            "still missing is a full-path run of 2–3 circle cells (a few minutes); do it once for the thesis."], "#e9f5ec")
    if co and mx and ci:
        s += box("How to write it in the thesis", [
            "“Band extrema of two-dimensional photonic crystals usually, but not always, lie on the boundary of the "
            "irreducible Brillouin zone [Harrison 2007; Craster 2012; Maurin 2018]. We therefore verified this for our "
            f"structures. For all {co['n_cells']} COMSOL cells, the band edges at M and K equal those over the full "
            f"{co['n_k']}-point path. For {mk['n_pwe_total']} plane-wave cells, including {mk['n_circle_pairs_total']} "
            "with circular holes, they equal those over a dense path and over the full irreducible zone to within "
            f"{max(co['max_dev_GHz'], mx['max_dev_GHz'], ci['max_dev_GHz']):.3f} GHz. The gap opens at K through "
            "inversion-symmetry breaking, and the second band near M limits the upper edge [Shalaev 2019].”"])
    return s


def closing_sections(res):
    cs = circle_stats()
    s = [H1("11. How to run everything, and the COMSOL side")]
    s += [H2("11.1 Commands"),
          P("All scripts run from the Project folder with the project's virtual environment "
            "(<font name='Mono'>.venv\\Scripts\\activate</font>). Times are for this laptop (12 cores, RTX 3050)."),
          code("""
python src/utils/process_bands.py                         # raw COMSOL CSVs -> bands.csv, band_edges.csv (once)
python src/shapes/validate_pwe.py                         # fit eps, check PWE vs 128 COMSOL cells   (~1 min)
python src/shapes/gen_dataset.py --kind fourier --n 4000  # freeform C3 cells, PWE labels           (~3 min)
python src/shapes/gen_dataset.py --kind library --n 2000  # mixed-shape cells                        (~1 min)
python src/shapes/add_comsol_dataset.py "<csv>" --family circle   # add a COMSOL sweep (section 11.4)
python src/shapes/check_mk_shortcut.py                    # band edges only at M and K? (6.4)      (~3 min)
python src/shapes/run_optionb.py --showcase 300 330       # surrogate, multi-fidelity, inverse, gallery (~15 min)
python src/shapes/make_figures.py                         # figures for this guide
python src/shapes/build_guide.py                          # this PDF
""")]
    s += [H2("11.2 Building a Fourier-hole model in COMSOL"),
          P("To verify designs (or to generate high-fidelity freeform data), the hole outline must be driven by global "
            "parameters. Work on a copy of the existing model:")]
    s += numbered([
        "<b>Definitions → Parameters:</b> add rA0, c3A, s3A, c6A, s6A, rB0, c3B, s3B, c6B, s6B (all in [um]); the hole "
        "centres xA, yA, xB, yB as used by your model; and thA = the direction from hole A to one of its B neighbours "
        "(30[deg] in this guide's convention: A = (a/2, a/(2√3)), B = (a, a/√3)), thB = thA + 180[deg].",
        "<b>Geometry → More Primitives → Parametric Curve</b> (hole A): parameter s from 0 to 2*pi, with<br/>"
        "<font name='Mono' size='8'>x: xA + (rA0 + c3A*cos(3*s) + s3A*sin(3*s) + c6A*cos(6*s) + s6A*sin(6*s))*cos(s + thA)</font><br/>"
        "<font name='Mono' size='8'>y: yA + (rA0 + c3A*cos(3*s) + s3A*sin(3*s) + c6A*cos(6*s) + s6A*sin(6*s))*sin(s + thA)</font>",
        "<b>Conversions → Convert to Solid</b> on the curve. Repeat steps 2–3 for hole B with the B parameters and thB.",
        "If your cell cuts holes at its boundary (the slides' hexagonal cell has the holes on its corners), copy each hole to "
        "every equivalent corner with <b>Transforms → Copy</b> (displacements = lattice vectors), then subtract all holes from "
        "the cell with <b>Booleans → Difference</b>. Keep the Floquet periodicity, materials and mesh settings unchanged.",
        "<b>Sanity check:</b> set all c and s to 0 (circles) and compare with the round-cell data; or set c3 = −0.160·r0 and "
        "r0 = 0.339·L to approximate a rounded triangle and compare with the old data.",
        "<b>Eigenfrequency step:</b> ask for 4 eigenfrequencies searched around a frequency below band 1 (e.g. 200 GHz), so "
        "bands 1 and 2 are always among them; the parsers keep the two lowest per k. Use only k = 1 and k = 2 (chapter 6.4)."])
    s += [H2("11.3 Verifying the gallery designs"),
          P("run_optionb.py writes <font name='Mono'>data/pwe/showcase_for_comsol.txt</font> in COMSOL's Load-from-File format "
            "(10 shape parameters in metres plus k, every design listed once per k) and "
            "<font name='Mono'>data/pwe/showcase_designs.csv</font> with the PWE band edges to compare against. In "
            "<b>Study → Parametric Sweep</b> choose <i>Specified combinations</i>, <i>Load from File</i>, set the unit of the "
            "ten length parameters to m, run, then <b>Results → Derived Values → Global Evaluation</b> of emw.freq (GHz) and export "
            "the table. Expect agreement within ~1–2 GHz; larger differences mean the PWE calibration does not carry over to "
            "freeform shapes and the multi-fidelity correction needs freeform COMSOL points.")]
    s += [H2("11.4 Adding a new COMSOL sweep (how the round-cell data went in)"),
          P("<font name='Mono' size=8>python src/shapes/add_comsol_dataset.py \"data/comsol/r=a_divby(3sqrt(3))_(Parametric "
            "sweep at k=1,2).csv\" --family circle</font>"),
          P("<font name='Mono' size=8>python src/shapes/check_comsol_rerun.py \"data/comsol/parameter_sweep_values (Varied L1, "
            "L2 and r = 2Lby 3sqrt(3)) (k=1,2).csv\" --corner 1</font>"),
          P("The first command reads the per-mesh-node export in chunks (columns X, Y, r1, r2, k, λ, freq; use --usecols if "
            "different), keeps the two lowest eigenfrequencies per k, writes <font name='Mono'>data/comsol/extra_circle.csv</font> "
            "in the shared format and prints the PWE–COMSOL error. The second compares a k = 1, 2 re-run with the old full-path "
            "sweep of the same cells. Then rerun run_optionb.py, make_figures.py and build_guide.py: new cells automatically join "
            "the COMSOL test set and the multi-fidelity correction. The same route works for any future sweep (for example the "
            "gallery designs) with --family rounded, sharp or circle; freeform designs need a parser for the 10-parameter table.")]
    s += [H2("11.5 Code map (src/shapes)")]
    s += table([["File", "Purpose"],
                ["geometry.py", "lattice, hole families, C3 Fourier descriptor, validity checks"],
                ["pwe.py", "plane-wave-expansion band solver"],
                ["validate_pwe.py", "fit ε, compare with both COMSOL datasets"],
                ["gen_dataset.py", "sample valid cells, label with PWE, check the M/K shortcut"],
                ["dataset.py", "one table for COMSOL + PWE data with the 18 features"],
                ["add_comsol_dataset.py", "add a new COMSOL sweep (e.g. circles)"],
                ["check_comsol_rerun.py", "compare a k = 1, 2 COMSOL re-run with an old full-path sweep"],
                ["check_mk_shortcut.py", "check that band edges sit only at M and K (section 6.4)"],
                ["models.py", "surrogate ensemble, validity penalty, tandem, CVAE, multi-start"],
                ["run_optionb.py", "all experiments, gallery, COMSOL export"],
                ["make_figures.py, build_guide.py", "figures and this PDF"]], [5 * cm, 11.5 * cm])
    s += [PageBreak()]

    s += [H1("12. Limitations — say these before an examiner does")]
    s += bullets([
        "<b>Low-fidelity labels.</b> The 6,000 cells are labelled by the 2D PWE solver, calibrated with one number (ε) on "
        "triangles only. Freeform shapes are not yet confirmed in COMSOL; the gallery file is the first check.",
        "<b>2D is itself an approximation.</b> Both COMSOL and PWE treat the 200 µm slab as a 2D crystal with an effective "
        "permittivity. A real slab has a light cone, radiation loss and an effective index that depends on frequency and on "
        "the hole pattern. Final designs need a 3D (slab) check, as in the group's device simulations.",
        "<b>Gap is not topology.</b> A C3-symmetric cell with unequal holes and a gap at K is expected to be valley-Hall, but "
        "the valley Chern sign, band ordering and the existence of edge states for a chosen interface are not computed yet.",
        "<b>Fabrication rules</b> (25 µm walls, 8 µm corners) are placeholders; replace them with your process limits.",
        "<b>Targets were drawn from the same distribution as the data</b>, so they are reachable. A real request can be "
        "unreachable; the surrogate's uncertainty and the nearest achievable target should be reported then.",
        "<b>Correction learned from simple shapes only.</b> The multi-fidelity correction has seen rounded triangles, sharp "
        "triangles and circles; applying it to freeform shapes is extrapolation until freeform COMSOL data (the gallery "
        "designs) are added.",
        ("<b>Several splits, one dataset.</b> The benchmark was repeated with different random splits (chapter 10, "
         "repeatability table), but all runs use the same 6,000 PWE cells and the same PWE 'truth'; COMSOL verification of "
         "freeform designs is still missing."
         if load("results/optionb/seed_summary.json", {}).get("n", 0) >= 2 else
         "<b>Single split.</b> The final surrogate and the inverse benchmark use one random split; repeat with several "
         "seeds for the thesis tables (run_optionb.py --seed N).")])
    s += [H1("13. What to do next")]
    s += [H3("Before the November evaluation")]
    s += numbered([
        "Confirm the permittivity in the COMSOL model (chapter 6) and the hole construction (chapter 3.3).",
        "Run the gallery designs in COMSOL (chapter 11.3) — even 4–8 designs turn 'PWE says' into 'COMSOL confirms'.",
        ("Run 2–3 circle cells on the full k-path once in COMSOL (section 6.4) — the round-cell data is already in (chapter 3.5)."
         if cs else "Add the round-cell data (chapter 11.4) and rebuild this guide."),
        "Presentation storyline: problem → stage-1 audit (one-to-one, GP wins) → why freeform shapes (non-uniqueness) → symmetry "
        "rule and Fourier descriptor → validated fast solver → dataset → surrogate and learning curve → four inverse methods "
        "compared with physics verification → gallery of different shapes for one target → limitations and plan."])
    s += [H3("Next semester")]
    s += bullets([
        "<b>COMSOL in the loop (active learning):</b> pick the next COMSOL runs where the surrogate is most uncertain or where the "
        "inverse designs land (MPh scripts in src/comsol already run COMSOL from Python).",
        "<b>Proper multi-fidelity model</b> (co-kriging / Δ-learning network) trained on PWE + COMSOL across all shape families.",
        "<b>Topology-aware design:</b> compute the valley Chern number from the PWE eigenvectors (Berry curvature near K) and "
        "add it as a constraint; then design the VPC-A/VPC-B interface and its edge-state band.",
        "<b>Richer shapes</b> (harmonics 9 and 12, or free-form pixels with a diffusion prior, as in Zhang et al. 2026) and a "
        "3D slab check; fabrication and THz measurement of one design."])
    s += [PageBreak()]

    s += [H1("14. Glossary")]
    s += table([["Term", "Meaning in this project"],
                ["unit cell", "smallest repeating tile of the crystal; here a rhombus with two holes (A, B)"],
                ["Bloch / Floquet", "wave = plane wave × periodic function; COMSOL's periodic boundary with phase e^(ik·R)"],
                ["k-point, k-path", "wave vector; the band diagram follows Γ → M → K → Γ (k = 0, 1, 2, 3)"],
                ["band edge", "f_lower = highest point of band 1; f_upper = lowest point of band 2"],
                ["band gap Δf", "f_upper − f_lower; negative means the bands overlap (no complete gap)"],
                ["Dirac point", "point where bands 1 and 2 touch at K when both holes are equal (C6 symmetry)"],
                ["valley, valley Chern number", "K and K′; topological index ±1/2 whose sign flips when the holes are swapped"],
                ["C3 / C6 symmetry", "unchanged by 120° / 60° rotation"],
                ["C3 Fourier descriptor", "r(θ) = r0 + Σ c_n cos nθ + s_n sin nθ with n = 3, 6, 9, 12; 9 numbers per hole"],
                ["PWE", "plane-wave expansion: Maxwell's equations as a matrix eigenproblem in a plane-wave basis"],
                ["effective index / ε", "a 2D stand-in for the slab: ε ≈ 8.9 (n ≈ 2.99) instead of bulk 11.7"],
                ["fidelity", "how close a simulator is to reality; PWE low, COMSOL (2D) higher"],
                ["surrogate / forward model", "network or GP that predicts band edges from shape in milliseconds"],
                ["inverse model", "anything that goes from target band edges to a shape"],
                ["non-uniqueness", "many shapes give the same target; makes direct inverse regression fail"],
                ["tandem network", "inverse network trained through a frozen forward network (loss on the response)"],
                ["CVAE", "conditional variational autoencoder; samples many designs for one target"],
                ["ensemble", "several networks trained from different seeds; their spread estimates uncertainty"],
                ["Gaussian process (GP)", "Bayesian regression with a kernel; strong on small data, gives uncertainty"],
                ["multi-fidelity correction", "learn COMSOL − PWE and add it to cheap predictions"],
                ["validity penalty", "smooth cost that is zero only for fabricable cells"],
                ["MAE", "mean absolute error, here always in GHz"]], [4.3 * cm, 12.2 * cm])
    s += [H1("15. Test yourself")]
    qa = [("Why did the tandem network not beat a Gaussian process in stage 1?",
           "With two design numbers and two targets the map was one-to-one and smooth; there was no non-uniqueness to resolve, "
           "and 61 points favour a GP."),
          ("Why can a hole not have a cos 2θ term?", "It is not invariant under 120° rotation, so it breaks C3 and the valley physics."),
          ("What are f_lower and f_upper and where are they found?", "Max of band 1 (always at K here) and min of band 2 (at M "
           "for rounded triangles, circles and most freeform cells; at K for most sharp triangles); the gap is their "
           "difference."),
          ("Why may COMSOL be run at only k = 1 and 2, and when could that fail?", "The gap opens at K and band 2 bottoms "
           "out at M or K; this was checked on all COMSOL cells and on 700+ PWE cells over the whole irreducible zone. It "
           "is not a theorem: a new hole family could put an edge between K and M (as circles do in some 3D slabs), so "
           "check a few cells of each new family on the full path."),
          ("Why does swapping holes A and B not change the band edges?", "Swapping is equivalent to inversion of the crystal; "
           "together with time reversal the frequencies are unchanged — but the valley Chern number flips."),
          ("Why fit ε when the slides say 11.7?", "All frequencies were 13% off by the same factor; one permittivity explains it "
           "(likely an effective index in the 2D COMSOL model). After fitting, both triangle datasets agree within ~1 GHz — and so "
           "does the circle dataset, which was not used in the fit."),
          ("What does the validity penalty do in the tandem and multi-start methods?", "It pushes designs back into the "
           "fabricable region during training/optimization; an exact check is still applied afterwards."),
          ("Why verify with the PWE solver rather than with the surrogate?", "An optimizer can exploit surrogate errors; only "
           "an independent simulation shows whether the target is really met."),
          ("What would make the multi-fidelity correction trustworthy for freeform shapes?", "COMSOL runs of freeform (and "
           "circle) cells, e.g. the gallery designs, added to the correction's training data."),
          ("How do you move a design to a different 6G channel?", "Scale the lattice constant and the holes by s; all frequencies "
           "divide by s (2D, non-dispersive).")]
    for q, a in qa:
        s += [Paragraph(f"<b>Q.</b> {q}", S["p"]), Paragraph(f"<i>A.</i> {a}", ParagraphStyle("a", parent=S["p"], leftIndent=14,
                                                                                             textColor=MUTED))]
    s += [PageBreak(), H1("16. References")]
    refs = [
        "D. Liu, Y. Tan, E. Khoram, Z. Yu, \"Training deep neural networks for the inverse design of nanophotonic structures\", "
        "ACS Photonics 5, 1365 (2018) — the tandem network. arxiv.org/abs/1710.04724",
        "J. D. Joannopoulos, S. G. Johnson, J. N. Winn, R. D. Meade, <i>Photonic Crystals: Molding the Flow of Light</i>, 2nd ed., "
        "Princeton (2008) — band structure, master equation, plane waves. ab-initio.mit.edu/book",
        "K. M. Ho, C. T. Chan, C. M. Soukoulis, \"Existence of a photonic gap in periodic dielectric structures\", "
        "Phys. Rev. Lett. 65, 3152 (1990) — inverse-matrix rule for PWE.",
        "Y. Yang et al., \"Terahertz topological photonics for on-chip communication\", Nature Photonics 14, 446 (2020) — the "
        "a = 242.5 µm silicon VPC platform.",
        "T. Ma, G. Shvets, \"All-Si valley-Hall photonic topological insulator\", New J. Phys. 18, 025012 (2016).",
        "M. I. Shalaev, W. Walasik, A. Tsukernik, Y. Xu, N. M. Litchinitser, \"Robust topologically protected transport in "
        "photonic crystals at telecommunication wavelengths\", Nature Nanotechnology 14, 31–34 (2019), "
        "doi:10.1038/s41565-018-0297-6; arXiv:1712.07284 — triangular vs circular holes, band 2 at M, 2D COMSOL model with "
        "n<sub>eff</sub> = 2.965 (sections 6.3, 6.4).",
        "J. M. Harrison, P. Kuchment, A. Sobolev, B. Winn, \"On occurrence of spectral edges for periodic operators inside "
        "the Brillouin zone\", J. Phys. A: Math. Theor. 40, 7597–7618 (2007); arXiv:math-ph/0702035 — band edges need "
        "not lie on the zone boundary.",
        "R. V. Craster, T. Antonakakis, M. Makwana, S. Guenneau, \"Dangers of using the edges of the Brillouin zone\", "
        "Phys. Rev. B 86 (2012).",
        "F. Maurin, C. Claeys, E. Deckers, W. Desmet, \"Probability that a band-gap extremum is located on the irreducible "
        "Brillouin-zone contour for the 17 different plane crystallographic lattices\", Int. J. Solids Struct. 135, 26–36 "
        "(2018), doi:10.1016/j.ijsolstr.2017.11.006.",
        "A. Kumar et al., \"Terahertz topological photonic integrated circuits for 6G and beyond: a perspective\", "
        "J. Appl. Phys. 132(14) (2022).",
        "K. Sohn, H. Lee, X. Yan, \"Learning structured output representation using deep conditional generative models\", "
        "NeurIPS (2015) — conditional VAE.",
        "W. Ma, F. Cheng, Y. Xu, Q. Wen, Y. Liu, \"Probabilistic representation and inverse design of metamaterials based on a "
        "deep generative model with semi-supervised learning\", Adv. Mater. 31, 1901111 (2019).",
        "B. Lakshminarayanan, A. Pritzel, C. Blundell, \"Simple and scalable predictive uncertainty estimation using deep "
        "ensembles\", NeurIPS (2017).",
        "C. E. Rasmussen, C. K. I. Williams, <i>Gaussian Processes for Machine Learning</i>, MIT Press (2006). gaussianprocess.org/gpml",
        "M. C. Kennedy, A. O'Hagan, \"Predicting the output from a complex computer code when fast approximations are "
        "available\", Biometrika 87, 1 (2000) — multi-fidelity modelling.",
        "R. E. Christiansen, O. Sigmund, \"Inverse design in photonics by topology optimization: tutorial\", JOSA B 38, 496 (2021) "
        "— in your Reading Material folder.",
        "Label-efficient generative inverse design and design-rule discovery in freeform topological photonics, "
        "arXiv:2609.28401 (2026) — C3-symmetric freeform valley crystals with a diffusion prior.",
        "Valley photonic crystal design trading band-gap width against valley Chern number, arXiv:2601.22958 (2026).",
        "Inverse design of dual-band valley-Hall photonic crystals, arXiv:2503.22206 (2025).",
        "IEEE Std 802.15.3d-2017 — 252.72–321.84 GHz wireless links.",
        "COMSOL 6.4 documentation: Parametric Curve, Parametric Sweep (Load from File), Global Evaluation. doc.comsol.com"]
    s += numbered(refs)
    return s


def comsol_freeform_section():
    """COMSOL runs of freeform (Fourier-hole) cells: validation of PWE, the gallery check, PWE vs COMSOL training."""
    import glob
    paths = sorted(glob.glob("data/comsol/extra_fourier*.csv"))
    if not paths:
        return []
    e = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    g = e[e.gap > 0.1]
    err = e[["f_lower_pwe", "f_upper_pwe"]].to_numpy() - e[["f_lower", "f_upper"]].to_numpy()
    s = [H2("9.1 Freeform shapes in COMSOL"),
         P("The hole outline was rebuilt in COMSOL from the Fourier formula (Parametric Curve + Convert to Solid, with the "
           "Air material following the holes; data/comsol/new/FOURIER_COMSOL_STEPS.txt) and random freeform cells were run "
           f"at k = 1 and 2, exported as Global Evaluation tables. So far: <b>{len(e)} valid freeform cells</b> "
           f"({len(g)} with a gap). Two plain-circle cells in batch 1 reproduced the earlier circle sweep within 0.013 GHz, "
           "so the Fourier-hole model is set up correctly."),
         P(f"<b>The PWE solver is accurate for freeform shapes too:</b> PWE − COMSOL = {abs(err).mean(0)[0]:.2f} / "
           f"{abs(err).mean(0)[1]:.2f} GHz on average (f<sub>lower</sub> / f<sub>upper</sub>), largest "
           f"{abs(err).max(0)[0]:.2f} / {abs(err).max(0)[1]:.2f} GHz — the same accuracy as for triangles and circles, with "
           "no refitting. This was the largest open question of option B."),
         box("A limitation of the COMSOL cell", [
             "In the rhombic COMSOL unit cell each hole centre is only 70 µm from two cell edges. A hole that reaches further "
             "is cut off at the edge in COMSOL (the part outside is not wrapped around to the other side), so COMSOL "
             "simulates a different shape. In batch 1, 105 of 202 random cells did this and disagreed with PWE by up to "
             "88 GHz; they are removed automatically (src/comsol/import_global_eval.py) and later batches only contain holes "
             "reaching at most 68 µm. PWE has no such limit. To allow larger holes in COMSOL, the holes would have to be "
             "copied by the lattice vectors and clipped to the cell."], "#fff4e5")]
    sc = pd.read_csv("data/pwe/showcase_designs.csv") if os.path.exists("data/pwe/showcase_designs.csv") else None
    if sc is not None:
        keys = ("r0", "c3", "s3", "c6", "s6")
        A = np.array([[json.loads(a)[k] for k in keys] + [json.loads(b)[k] for k in keys]
                      for a, b in zip(e.paramsA, e.paramsB)])
        names = ["rA0", "c3A", "s3A", "c6A", "s6A", "rB0", "c3B", "s3B", "c6B", "s6B"]
        rows = [["gallery design", "target (GHz)", "PWE (GHz)", "COMSOL (GHz)", "COMSOL − target"]]
        for i in range(len(sc)):
            j = np.where(np.abs(A - sc.loc[i, [n + "_um" for n in names]].to_numpy(float)).max(1) < 0.01)[0]
            if len(j):
                r = e.iloc[j[0]]
                rows.append([str(i + 1), "300 / 330", f"{sc.f_lower_pwe[i]:.2f} / {sc.f_upper_pwe[i]:.2f}",
                             f"{r.f_lower:.2f} / {r.f_upper:.2f}", f"{r.f_lower - 300:+.2f} / {r.f_upper - 330:+.2f}"])
        if len(rows) > 1:
            s += [H3("The first COMSOL check of AI-designed shapes")]
            s += table(rows, [3 * cm, 2.8 * cm, 3.4 * cm, 3.4 * cm, 3.4 * cm])
            s += [P("Shapes proposed by the inverse methods for a 300 / 330 GHz gap and simulated in COMSOL hit the target "
                    "within about 1 GHz. 23 of the 24 gallery designs fit inside the COMSOL cell and can be checked the same "
                    "way.")]
    cmp_ = load("results/optionb/comsol_vs_pwe.json", {})
    if cmp_:
        nice = {"pwe_solver": "PWE solver, no ML", "nn_pwe": "NN trained on 6,000 PWE cells",
                "gp_comsol_free": "GP on COMSOL freeform cells only", "gp_comsol_all": "GP on all COMSOL cells",
                "nn_comsol_all": "NN on all COMSOL cells", "nn_pwe+corr": "PWE-trained NN + COMSOL correction",
                "solver+corr": "PWE solver + COMSOL correction"}
        need = {"pwe_solver": "none", "nn_pwe": "none"}
        rows = [["predictor", "COMSOL data used", "MAE f_lower", "MAE f_upper", "MAE gap"]]
        for m, v in cmp_["methods"].items():
            rows.append([nice.get(m, m), need.get(m, "yes"), f"{v['mae_f_lower']:.2f}", f"{v['mae_f_upper']:.2f}",
                         f"{v['mae_gap']:.2f}"])
        s += [H3("Do we still need PWE? Training on PWE vs on COMSOL")]
        s += [P(f"Every predictor below is scored on the {cmp_['n_free']} COMSOL freeform cells with "
                f"{cmp_['folds']}-fold cross-validation (each cell is predicted by models that never saw it; "
                f"{len(cmp_['seeds'])} repetitions). 'All COMSOL cells' adds the {cmp_['n_other']} triangle and circle "
                "cells to the training folds. GHz, src/shapes/compare_training.py:")]
        s += table(rows, [5.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm])
        best = min(cmp_["methods"], key=lambda m: np.mean([cmp_["methods"][m]["mae_f_lower"], cmp_["methods"][m]["mae_f_upper"]]))
        s += fig("comsol_vs_pwe.png", W, "(a) Each predictor's error on held-out COMSOL freeform cells. (b) Error vs the "
                 "number of COMSOL freeform cells used for training (the triangle and circle COMSOL cells are always "
                 "included).")
        M = cmp_["methods"]
        avg = {m: (v["mae_f_lower"] + v["mae_f_upper"]) / 2 for m, v in M.items()}
        pwe_only = min(avg[m] for m in ("pwe_solver", "nn_pwe"))
        com_only = min(avg[m] for m in M if "comsol" in m)
        both = min(avg[m] for m in M if "corr" in m)
        lc = cmp_["learning_curve"]
        zero = {m: float(np.mean(c[0]["mae"])) for m, c in lc.items()}
        if both < com_only and pwe_only < com_only:
            verdict = (f"The COMSOL-only models are much worse (best {com_only:.2f} GHz, {com_only / both:.0f}× the error of "
                       "PWE + correction): about a hundred cells are far too few to learn 10 shape numbers (chapter 8). "
                       "So <b>PWE is still needed</b>: the cheap PWE data supply the shape knowledge and the COMSOL cells "
                       "correct the small remaining bias — the multi-fidelity idea of chapter 9.")
        else:
            verdict = (f"Here a COMSOL-only model is competitive (best {com_only:.2f} GHz vs {both:.2f} GHz with PWE + "
                       "correction), so with this much COMSOL data the PWE stage matters less.")
        s += [P(f"<b>Best: {nice.get(best, best)}</b> ({avg[best]:.2f} GHz averaged over both edges). PWE alone gives "
                f"{pwe_only:.2f} GHz; adding the COMSOL correction lowers that to {both:.2f} GHz. " + verdict)]
        s += [P(f"The learning curve (b) holds a surprise: the correction already reaches "
                f"{min(zero['nn_pwe+corr'], zero['solver+corr']):.2f} GHz with <b>zero</b> freeform COMSOL cells, because "
                "the PWE-vs-COMSOL offset learned on the triangle and circle cells carries over to freeform shapes. The "
                f"GP on COMSOL alone needs every freeform cell to get down to {lc['gp_comsol_all'][-1]['mae'][0]:.1f} / "
                f"{lc['gp_comsol_all'][-1]['mae'][1]:.1f} GHz. Adding the triangle cells even hurts it "
                f"({avg['gp_comsol_all']:.2f} vs {avg['gp_comsol_free']:.2f} GHz), because triangles say little about "
                "freeform shapes. More freeform batches (2–5) will mainly help the COMSOL-only line; rerun "
                "compare_training.py after each import to see it.")]
    return s


def seed_table(names):
    """Mean ± standard deviation over repeated runs (summarize_seeds.py), if more than one run exists."""
    sm = load("results/optionb/seed_summary.json", {})
    if sm.get("n", 0) < 2:
        return []

    def pm(d, j=None, scale=1.0, fmt="{:.2f}"):
        m, sd = (d["mean"], d["std"]) if j is None else (d["mean"][j], d["std"][j])
        return f"{fmt.format(scale * m)} ± {fmt.format(scale * sd)}"

    rows = [["method", "within 1 GHz (%)", "MAE f_lower (GHz)", "MAE f_upper (GHz)", "distinct designs (of 8)"]]
    for n in names:
        v = sm["inverse"][n]
        rows.append([NICE[n], pm(v["success_rate_within_1GHz"], scale=100, fmt="{:.0f}"), pm(v["verified_mae_GHz"], 0),
                     pm(v["verified_mae_GHz"], 1), pm(v["distinct"], fmt="{:.1f}") if "distinct" in v else "1 (by design)"])
    out = [H3(f"Repeatability: {sm['n']} different random splits")]
    out += [P("Every number above comes from one random choice of training and test cells and one set of initial network "
              f"weights. The whole experiment was repeated with {sm['n']} different random seeds (seeds {sm['seeds']}; "
              "src/shapes/summarize_seeds.py). Mean ± standard deviation over the runs — use these in the thesis:")]
    out += table(rows, [3.4 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm, 3.4 * cm])
    fw, mf = sm["forward_mae_all"], sm["mf_pwe_plus_gp_cv_mae"]
    out += [Paragraph(f"Forward surrogate MAE on held-out PWE cells: {pm(fw, 0)} / {pm(fw, 1)} GHz (f_lower / f_upper). "
                      f"Multi-fidelity, PWE + GP vs COMSOL (10-fold CV): {pm(mf, 0)} / {pm(mf, 1)} GHz.", S["cap"])]
    return out


NICE = {"tandem": "Tandem NN", "cvae": "Conditional VAE", "multistart": "Multi-start gradient", "cvae_refine": "CVAE + refine"}


def results_sections(res):
    if not res or "inverse" not in res:
        return [H1("8.–10. Results"), P("<i>Run src/shapes/run_optionb.py, then rebuild this guide.</i>")]
    lc, ft, mf, inv, sc = res["learning_curve"], res["forward_test"], res["multifidelity"], res["inverse"], res["showcase"]
    fin = load("results/optionb/final_comsol.json", None)   # final study with all COMSOL data, if run
    if fin:
        a = fin["mf_cv10"]["all"]
        mf = dict(mf, n=fin["n_gapped"], final=True, pwe_raw_mae=a["pwe"]["mae"], pwe_plus_gp_cv_mae=a["pwe_plus_corr"]["mae"],
                  datasets={"comsol_" + r["family"]: r["gapped"] for r in fin["families"]},
                  gap_mae={"pwe_raw": None, "pwe_plus_gp": None},
                  leave_family_out={"comsol_" + k: {"pwe_raw_mae": v["pwe"]["mae"], "pwe_plus_gp_mae": v["pwe_plus_corr"]["mae"]}
                                    for k, v in fin["mf_leave_family_out"].items()})
    fz = lambda v: "" if v is None else f2(v)
    s = [H1("8. Forward surrogate: shape → band edges")]
    s += [H2("8.1 The model"),
          P("Input: the 18 Fourier features of the cell (µm), standardised. Output: (f<sub>lower</sub>, f<sub>upper</sub>) in GHz, "
            "standardised. Each network is a multilayer perceptron with three hidden layers of 256 SiLU units, trained with AdamW "
            "and a one-cycle learning-rate schedule (400 epochs, batch 128) on 90% of the 6,000 PWE cells — <i>including</i> cells "
            "without a gap, because they teach the model where gaps close. Five networks with different random seeds form a "
            "<b>deep ensemble</b>:")]
    s += eq(r"\hat{\mathbf{y}}(\mathbf{x})=\frac{1}{M}\sum_{m=1}^{M}F_m(\mathbf{x}),\qquad \hat{\sigma}(\mathbf{x})=\mathrm{std}_m\,F_m(\mathbf{x}),\qquad M=5")
    s += [P("The mean is the prediction; the spread is a cheap uncertainty estimate (large where the members disagree, i.e. far "
            "from the data). The baseline is a Gaussian process with an automatic-relevance (one length scale per feature) kernel:")]
    s += eq(r"k(\mathbf{x},\mathbf{x}')=\sigma_f^2\exp\left(-\frac{1}{2}\sum_{i=1}^{18}\frac{(x_i-x'_i)^2}{\ell_i^2}\right)+\sigma_n^2\,\delta_{\mathbf{x}\mathbf{x}'}")
    s += [H2("8.2 How many simulations are needed?"),
          P("The learning curve answers the most practical question of the project: how many labelled cells (PWE now, COMSOL "
            "later) does a freeform surrogate need? Same held-out test cells for every row; mean absolute error in GHz for "
            "(f<sub>lower</sub>, f<sub>upper</sub>).")]
    rows = [["training cells", "NN ensemble MAE", "GP MAE", "NN time (s)", "GP time (s)"]]
    for r in lc:
        rows.append([r["n_train"], f"{r['nn_mae'][0]:.2f} / {r['nn_mae'][1]:.2f}",
                     f"{r['gp_mae'][0]:.2f} / {r['gp_mae'][1]:.2f}" if "gp_mae" in r else "— (too slow)",
                     r["nn_s"], r.get("gp_s", "—")])
    s += table(rows, [3 * cm, 3.8 * cm, 3.8 * cm, 2.8 * cm, 2.8 * cm])
    gp_rows = [r for r in lc if "gp_mae" in r]
    small = gp_rows[0]
    big_gp = gp_rows[-1]
    nn_at = {r["n_train"]: np.mean(r["nn_mae"]) for r in lc}
    s += [P(f"With {small['n_train']} cells the GP is about as good as or better than the network "
            f"({np.mean(small['gp_mae']):.2f} vs {np.mean(small['nn_mae']):.2f} GHz on average) — the stage-1 lesson again: small "
            f"data favours GPs. With more data the network keeps improving (at {big_gp['n_train']} cells: NN "
            f"{np.mean(next(r['nn_mae'] for r in lc if r['n_train'] == big_gp['n_train'])):.2f} vs GP {np.mean(big_gp['gp_mae']):.2f} GHz; "
            f"with all {lc[-1]['n_train']:,} cells the network reaches {nn_at[lc[-1]['n_train']]:.2f} GHz), while an exact GP "
            "grows as n³ in cost. <b>Planning rule:</b> about 500 COMSOL cells already give ~1 GHz accuracy over this whole "
            "freeform space. With the 2-k-point shortcut that is 1,000 eigen-solves — time your 3-geometry check run to "
            "know whether that is an evening or a weekend.")]
    s += [H2("8.3 Accuracy of the final ensemble")]
    s += table([["test set", "MAE f_lower", "MAE f_upper", "MAE gap", "max error"],
                [f"all {ft['n_test']} held-out PWE cells", f2(ft["mae_all"][0]), f2(ft["mae_all"][1]), f2(ft["gap_mae"]),
                 f"{ft['max_err'][0]:.1f} / {ft['max_err'][1]:.1f}"],
                ["   only cells with a gap", f2(ft["mae_gapped"][0]), f2(ft["mae_gapped"][1]), "", ""],
                *[[f"   {k}", f2(v[0]), f2(v[1]), "", ""] for k, v in ft["by_dataset"].items()],
                *([] if mf.get("final") else [[f"{mf['n']} gapped COMSOL cells", f2(mf["nn_raw_mae"][0]), f2(mf["nn_raw_mae"][1]), "", ""]])],
               [5.2 * cm, 2.6 * cm, 2.6 * cm, 2.4 * cm, 3.6 * cm])
    s += [Paragraph(f"GHz. Correlation between the ensemble spread and the actual error on the test set: "
                    f"{ft['corr_std_abs_err']:.2f} (positive = the uncertainty is informative).", S["cap"])]
    s += fig("forward_mf.png", W, "(a) Learning curve, log–log. (b) Surrogate vs PWE on held-out cells. (c) Error of the PWE "
             "solver against COMSOL before (grey) and after (green) the multi-fidelity correction, per dataset and band edge.")
    s += [P("Note the last row. A network trained <i>only</i> on PWE labels of random cells predicts the COMSOL cells within "
            "about a gigahertz. Its training data contained triangles of random size and rotation, but none of these exact "
            "cells, and no COMSOL labels at all. Part of that error comes from the network and part is the PWE–COMSOL "
            "difference, which chapter 9 corrects.")]
    s += [PageBreak()]

    # ------------------------------------------------------------ 9
    s += [H1("9. Using the COMSOL datasets: multi-fidelity correction")]
    s += [P("We have two simulators: PWE (fast, slightly biased) and COMSOL (slow, the reference). The standard trick "
            "(Kennedy and O'Hagan 2000) is to model the high-fidelity answer as the low-fidelity answer plus a smooth correction "
            "learned from the few cells where both exist. The correction δ is a Gaussian process fitted to the differences "
            "f<sub>COMSOL</sub>(x<sub>i</sub>) − f<sub>PWE</sub>(x<sub>i</sub>) at the COMSOL cells, using the same 18 features:")]
    s += eq(r"f_{\rm COMSOL}(\mathbf{x})\;\approx\;f_{\rm PWE}(\mathbf{x})+\delta(\mathbf{x}),\qquad \delta(\mathbf{x})\sim\mathcal{GP}")
    fams = ", ".join(f"{k.replace('comsol_', '')} {v}" for k, v in mf.get("datasets", {}).items())
    s += [P(f"All COMSOL datasets are used here ({fams}; {mf['n']} gapped cells). This is where using several datasets "
            "matters most: one family alone cannot tell whether the PWE bias depends on the shape. Results (10-fold "
            "cross-validation, so every number is on cells the correction did not see):")]
    s += table([["predictor", "MAE f_lower", "MAE f_upper", "MAE gap"],
                ["PWE alone (ε fitted)", f2(mf["pwe_raw_mae"][0]), f2(mf["pwe_raw_mae"][1]), fz(mf["gap_mae"]["pwe_raw"])],
                ["PWE + GP correction", f2(mf["pwe_plus_gp_cv_mae"][0]), f2(mf["pwe_plus_gp_cv_mae"][1]), fz(mf["gap_mae"]["pwe_plus_gp"])],
                *([] if mf.get("final") else [
                    ["NN surrogate alone", f2(mf["nn_raw_mae"][0]), f2(mf["nn_raw_mae"][1]), ""],
                    ["NN surrogate + GP correction", f2(mf["nn_plus_gp_cv_mae"][0]), f2(mf["nn_plus_gp_cv_mae"][1]), ""]])],
               [6 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm])
    lofo = mf.get("leave_family_out", {})
    if lofo:
        rows = [["held-out family", "PWE alone", "PWE + correction learned on the other families"]]
        for d, v in lofo.items():
            rows.append([d.replace("comsol_", ""), f"{v['pwe_raw_mae'][0]:.2f} / {v['pwe_raw_mae'][1]:.2f}",
                         f"{v['pwe_plus_gp_mae'][0]:.2f} / {v['pwe_plus_gp_mae'][1]:.2f}"])
        s += [H3("Does the correction transfer to a family it has not seen?")] + table(rows, [4 * cm, 4 * cm, 8.5 * cm])
        helped = [d.replace("comsol_", "") for d, v in lofo.items() if np.mean(v["pwe_plus_gp_mae"]) < np.mean(v["pwe_raw_mae"])]
        hurt = [d.replace("comsol_", "") for d, v in lofo.items() if np.mean(v["pwe_plus_gp_mae"]) >= np.mean(v["pwe_raw_mae"])]
        bias_txt = ""
        if os.path.exists("results/optionb/mf.npz") and not mf.get("final"):
            z = np.load("results/optionb/mf.npz", allow_pickle=True)
            err = z["pwe"] - z["y"]
            bias_txt = (" Mean PWE − COMSOL bias (f<sub>lower</sub> / f<sub>upper</sub>): " + "; ".join(
                f"{d.replace('comsol_', '')} {err[z['dataset'] == d].mean(0)[0]:+.2f} / {err[z['dataset'] == d].mean(0)[1]:+.2f} GHz"
                for d in np.unique(z["dataset"])) + ".")
        if not hurt:
            verdict = ("Learned on the other families, the correction still helps on every held-out family, so part of the "
                       "PWE bias is shared between shapes.")
        elif not helped:
            verdict = ("Within a family the correction works well, but learned on the other families it does <b>not</b> "
                       "help on any held-out family: the bias depends on the shape.")
        else:
            verdict = (f"Learned on the other families, the correction helps for {', '.join(helped)} but <b>not</b> for "
                       f"{', '.join(hurt)}: part of the bias is shared, part is shape-specific.")
        s += [P("MAE (f<sub>lower</sub> / f<sub>upper</sub>), GHz. " + verdict + bias_txt + " This is the honest reason to "
                "add COMSOL cells of each new shape family — now the gallery designs — before trusting corrected predictions "
                "for freeform shapes.")]
    s += comsol_freeform_section()
    s += [PageBreak()]

    # ------------------------------------------------------------ 10
    s += [H1("10. Inverse design: from target band edges to shapes")]
    s += [H2("10.1 The problem"),
          P("Given a target t = (f<sub>lower</sub>, f<sub>upper</sub>), find designs u in (0, 1)<super>10</super> (chapter 5.3; "
            "φ(u) turns u into the 18 features) that the surrogate F maps onto t while staying fabricable. All four methods "
            "approximately solve")]
    s += eq(r"\mathbf{u}^\star=\arg\min_{\mathbf{u}\in(0,1)^{10}}\ \sum_{j=1}^{2}\left(\frac{F_j(\phi(\mathbf{u}))-t_j}{s_j}\right)^2+\lambda\,P(\mathbf{u})")
    s += [P("where s<sub>j</sub> is the standard deviation of each target in the data and P the validity penalty. With 10 unknowns "
            "and 2 equations there is typically an 8-dimensional family of solutions — this is the non-uniqueness that makes "
            "plain 'target → design' regression fail (it averages different valid answers into an invalid one).")]
    s += [H2("10.2 Method 1 — tandem network"),
          P("An inverse network G<sub>θ</sub>: t → u (two inputs, three hidden layers of 256, ten sigmoid outputs) is trained "
            "<i>through the frozen surrogate</i> on the targets of the training cells:")]
    s += eq(r"\mathcal{L}_{\rm tandem}(\theta)=\frac{1}{N}\sum_{i=1}^{N}\left[\left\|\frac{F(\phi(G_\theta(\mathbf{t}_i)))-\mathbf{t}_i}{\mathbf{s}}\right\|^2+\lambda\,P(G_\theta(\mathbf{t}_i))\right]")
    s += [P("Because the loss is measured on the response, the network may pick any one of the valid designs and is never "
            "penalised for not matching 'the' training geometry — Liu et al.'s fix for non-uniqueness. Limitation: it is a "
            "function, so it returns exactly one design per target.")]
    s += [H2("10.3 Method 2 — conditional variational autoencoder (CVAE)"),
          P("A generative model learns the whole <i>distribution</i> of designs that produce a target. An encoder q(z | u, t) "
            "compresses a training design into a 4-number latent code z; a decoder D(z, t) rebuilds the design from the code and "
            "the target. Training maximises the evidence lower bound:")]
    s += eq(r"\mathcal{L}_{\rm CVAE}=\mathbb{E}_{q(\mathbf{z}|\mathbf{u},\mathbf{t})}\left[\frac{\|\mathbf{u}-D(\mathbf{z},\mathbf{t})\|^2}{\sigma^2}\right]+\beta\,\mathrm{KL}\left(q(\mathbf{z}|\mathbf{u},\mathbf{t})\,\|\,\mathcal{N}(\mathbf{0},I)\right)")
    s += [P("At design time the encoder is dropped: draw z<super>(1)</super>, ..., z<super>(64)</super> from N(0, I) and decode "
            "each with the target — 64 different candidate shapes in one pass. The CVAE is trained on real (design, response) "
            "pairs only, so its samples look like designs from the dataset; it does not use the surrogate.")]
    s += [H2("10.4 Method 3 — multi-start gradient search"),
          P("No inverse network at all: start from 64 random designs, and run 300 Adam steps on the objective of 10.1 directly, "
            "back-propagating through the frozen surrogate (the design is parametrised by logit(u) so it stays inside the box). "
            "Different starts end in different solutions. This is the neural-network version of stage 1's 'GP + gradient', the "
            "method that won there.")]
    s += [H2("10.5 Method 4 — CVAE + refine"),
          P("Use the 64 CVAE samples as starting points for 100 gradient steps: the CVAE supplies realistic, diverse starts and "
            "the gradient makes them hit the target precisely.")]
    s += [H2("10.6 How the methods were judged"),
          P("100 targets were taken from <b>held-out</b> Fourier cells with a gap above 2 GHz (so every target is reachable and "
            "no method saw it). For each method and target the candidates are ranked by surrogate error; the best one that "
            "passes the <b>exact</b> fabrication check is kept and <b>re-simulated with the PWE solver</b>. The error reported is "
            "|PWE − target|, never the surrogate's own opinion. 'Distinct designs' counts, for 10 targets, how many of the 8 best "
            "different candidates (after removing A↔B-swapped and mirrored duplicates) are verified within 1 GHz.")]
    names = [n for n in ("tandem", "cvae", "multistart", "cvae_refine") if n in inv]
    rows = [["method", "valid design found", "verified MAE f_lower / f_upper", "median", "within 1 GHz", "distinct designs (of 8)", "ms per target"]]
    for n in names:
        v = inv[n]
        rows.append([NICE[n], v["valid_design_found"], f"{v['verified_mae_GHz'][0]:.2f} / {v['verified_mae_GHz'][1]:.2f}",
                     f"{v['verified_median_GHz'][0]:.2f} / {v['verified_median_GHz'][1]:.2f}",
                     f"{100 * v['success_rate_within_1GHz']:.0f}%",
                     f"{v['distinct_verified_designs_per_target']:.1f}" if "distinct_verified_designs_per_target" in v else "1 (by design)",
                     f"{1000 * v['seconds_per_target']:.0f}"])
    s += table(rows, [3.2 * cm, 2.1 * cm, 3.2 * cm, 2.4 * cm, 1.8 * cm, 2.2 * cm, 1.8 * cm])
    s += fig("inverse.png", W, "(a) PWE-verified error per method. (b) Fraction of targets reached within a given error. "
             "(c) Number of different verified shapes per target.")
    s += seed_table(names)
    best = max(names, key=lambda n: inv[n]["success_rate_within_1GHz"])
    most = max([n for n in names if "distinct_verified_designs_per_target" in inv[n]],
               key=lambda n: inv[n]["distinct_verified_designs_per_target"])
    s += [H2("10.7 What the comparison shows")]
    t, ms, cr, cv = inv["tandem"], inv["multistart"], inv["cvae_refine"], inv["cvae"]
    s += bullets([
        f"<b>Most accurate: {NICE[best]}</b> ({100 * inv[best]['success_rate_within_1GHz']:.0f}% of targets within 1 GHz after "
        f"physics verification; mean error {inv[best]['verified_mae_GHz'][0]:.2f} / {inv[best]['verified_mae_GHz'][1]:.2f} GHz). "
        "Now that the problem is many-to-one, the base-paper method does what it was designed for — compare stage 1, where it "
        "was the worst method on the one-to-one task.",
        f"<b>Why does the per-target search lose to the tandem?</b> Multi-start optimisation has a lower <i>surrogate</i> error "
        f"but a higher <i>verified</i> error ({ms['verified_mae_GHz'][0]:.2f} / {ms['verified_mae_GHz'][1]:.2f} GHz). Searching "
        "hard against a surrogate finds the places where the surrogate is wrong in your favour (often near the edge of the "
        "data). The tandem network learns one smooth map from all training targets, which keeps its designs where the "
        "surrogate is reliable — an implicit regulariser. This 'surrogate exploitation' effect is worth a slide; the remedy is "
        "to penalise surrogate uncertainty (ensemble spread) during the search.",
        f"<b>Best overall trade-off: CVAE + refine</b> ({100 * cr['success_rate_within_1GHz']:.0f}% within 1 GHz, "
        f"{cr['distinct_verified_designs_per_target']:.1f} of 8 candidates are different verified designs). The CVAE proposes "
        "realistic shapes from the data distribution, so the short refinement does not wander into unreliable regions.",
        f"<b>Most different answers: {NICE[most]}</b> ({inv[most]['distinct_verified_designs_per_target']:.1f} of 8). The "
        "tandem returns exactly one design per target. Having several gives a real choice: thickest walls, smoothest corners, "
        "or the best fabrication tolerance.",
        f"<b>The plain CVAE</b> reaches {100 * cv['success_rate_within_1GHz']:.0f}% within 1 GHz: its samples are diverse but only "
        "approximately on target. That is why the 'refine' step helps.",
        f"<b>Speed:</b> the tandem needs {1000 * t['seconds_per_target']:.1f} ms per target (one forward pass); multi-start "
        f"{1000 * ms['seconds_per_target']:.0f} ms. COMSOL would take minutes per evaluation.",
        "All errors here are against PWE. Against COMSOL, add the PWE–COMSOL difference from chapter 9 (about 0.3–1 GHz) until "
        "the gallery designs are checked in COMSOL."])
    s += [PageBreak()]
    s += [H2("10.8 The design gallery: many shapes, one target")]
    s += [P(f"Target f<sub>lower</sub> = {sc['target'][0]:.0f} GHz, f<sub>upper</sub> = {sc['target'][1]:.0f} GHz (a "
            f"{sc['target'][1] - sc['target'][0]:.0f} GHz gap overlapping the upper half of the IEEE 802.15.3d band). "
            f"{sc['n_candidates']} candidates were generated by the CVAE, multi-start search and CVAE + refine; after the exact "
            f"fabrication check, duplicate removal and PWE verification, <b>{sc['n_verified_within_1GHz']} genuinely different "
            "designs hit the target within 1 GHz</b>:")]
    s += fig("showcase.png", W, "Verified designs for the same target. Each panel: tiled unit cells; the dashed rhombus is one "
             "cell. The title gives the PWE-verified band edges and the thinnest silicon wall.")
    if sc["designs"]:
        rows = [["#", "hole A: r0, c3, s3, c6, s6 (µm)", "hole B: r0, c3, s3, c6, s6 (µm)", "PWE edges (GHz)", "wall"]]
        for i, d in enumerate(sc["designs"][:10]):
            p = d["p"]
            rows.append([i, ", ".join(f"{v:.1f}" for v in p[:5]), ", ".join(f"{v:.1f}" for v in p[5:]),
                         f"{d['pwe'][0]:.1f} / {d['pwe'][1]:.1f}", f"{d['min_wall']:.0f}"])
        s += table(rows, [0.7 * cm, 5.4 * cm, 5.4 * cm, 2.7 * cm, 1.3 * cm])
        big = np.median([max(d["p"][0], d["p"][5]) for d in sc["designs"]])
        small = np.median([min(d["p"][0], d["p"][5]) for d in sc["designs"]])
        s += [P("These numbers are exactly what goes into the COMSOL Parametric Curve of chapter 11.2 "
                "(data/pwe/showcase_for_comsol.txt has them in metres, ready for Load from File)."),
              P(f"<b>Read the table like a physicist.</b> Every design pairs a large hole (median r<sub>0</sub> ≈ {big:.0f} µm) with "
                f"a small one (≈ {small:.0f} µm, at the lower edge of the design box, 25 µm). A 30 GHz gap this low in frequency "
                "needs strong inversion breaking, so the hole sizes are nearly fixed. The freedom left over is in the shape "
                "harmonics (c<sub>3</sub>, s<sub>3</sub>, c<sub>6</sub>, s<sub>6</sub>), which is where the designs differ: "
                "that is the non-uniqueness, made visible. Allowing smaller holes (a wider design box) would give the optimiser "
                "more room.")]
    s += [PageBreak()]
    return s


def flatten(items):
    out = []
    for x in items:
        out += flatten(x) if isinstance(x, (list, tuple)) else [x]
    return out


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc = Doc(OUT, pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=1.9 * cm, bottomMargin=1.8 * cm,
              title="Option B study guide: AI-assisted inverse design of THz valley photonic crystals",
              author="Binit Shakya (prepared with Claude)")
    doc.multiBuild(flatten(story()), onFirstPage=on_page, onLaterPages=on_page)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
