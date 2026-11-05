"""Build a short, plain-language PWE cheat sheet -> docs/PWE_Cheat_Sheet.pdf

For explaining PWE to the guide/examiners without the maths. Reuses the styles of build_guide.py.
Run from the Project folder:  python src/shapes/build_pwe_cheatsheet.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from build_guide import (A4, MUTED, S, H1, H2, P, W, box, bullets, cm, fig, load, numbered,  # noqa: E402
                         table)
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer  # noqa: E402

OUT = "docs/PWE_Cheat_Sheet.pdf"


def footer(c, doc):
    c.saveState()
    c.setFont("Body", 8.5)
    c.setFillColor(MUTED)
    c.drawString(2.1 * cm, 1.1 * cm, "PWE cheat sheet: what to say about the fast solver")
    c.drawRightString(A4[0] - 2.1 * cm, 1.1 * cm, str(doc.page))
    c.restoreState()


def story():
    cmp_ = load("results/optionb/comsol_vs_pwe.json", {})
    m = cmp_.get("methods", {})
    solver = m.get("pwe_solver", {"mae_f_lower": 0.51, "mae_f_upper": 0.40})
    corr = m.get("solver+corr", {"mae_f_lower": 0.31, "mae_f_upper": 0.36})
    nfree = cmp_.get("n_free", 97)
    FN = load("results/optionb/final_numbers.json", {})
    g = lambda k, d="?": FN.get(k, d)

    s = [Paragraph("PWE in plain words", S["title"]), Spacer(1, 4),
         Paragraph("What it is, why we use it, and what to say when someone asks", S["subtitle"]), Spacer(1, 14)]

    s += box("The one sentence to remember",
             "<b>PWE is a fast, approximate version of what COMSOL does.</b> It takes a hole shape and gives back the "
             "band frequencies. We used it to make thousands of training examples cheaply, then corrected its small "
             "error with real COMSOL runs.")

    s += [H1("1. What problem does it solve?")]
    s += [P("A guitar string can only ring at certain notes, set by its length and tension. Light inside your photonic "
            "crystal is the same: it can only travel at certain frequencies, set by the pattern of holes. Between the "
            "allowed frequencies there is a range where light cannot travel at all. That range is the <b>band gap</b>, "
            "and its two edges (f_lower and f_upper) are the numbers the whole project is about.")]
    s += [P("Finding those allowed frequencies for a given hole shape is called <b>solving for the band structure</b>. "
            "COMSOL does it. PWE does exactly the same job, a different way.")]

    s += [H1("2. How it works, in three steps (no maths)")]
    s += numbered([
        "<b>Draw the cell as a picture.</b> The computer makes a pixel image of one unit cell: silicon is black, the "
        "air holes are white (left panel below).",
        "<b>Describe the picture as a mix of simple waves.</b> Just as a song can be written as a mix of pure notes, "
        "the hole pattern is written as a mix of simple ripple patterns (middle panel). This is the 'plane waves' in "
        "the name <i>plane-wave expansion</i>.",
        "<b>Ask which light waves fit.</b> With the crystal written as waves, finding the allowed light frequencies "
        "becomes one table of numbers (a matrix) that the computer solves in a fraction of a second. Out come the "
        "bands (right panel), the same curves COMSOL gives.",
    ])
    s += fig("pwe_demo.png", W, "Left: the unit cell as pixels. Middle: how strongly each simple wave is present "
             "(bright = strong). Right: the bands PWE finds (solid) on top of COMSOL's (dashed). They overlap; the "
             "green strip is the band gap.")
    s += box("COMSOL vs PWE in one line each",
             ["<b>COMSOL</b> cuts the cell into thousands of tiny triangles (the mesh) and solves on those. "
              "Accurate, flexible, slow.",
              "<b>PWE</b> describes the cell as a few hundred simple waves and solves on those. Very fast, and accurate "
              "for a simple 2D cell like ours.",
              "Both are standard textbook methods. PWE is the method in the well-known free MIT program MPB and in the "
              "textbook by Joannopoulos et al., <i>Photonic Crystals: Molding the Flow of Light</i> (2008)."])

    s += [PageBreak(), H1("3. Why we used it")]
    s += table([["", "COMSOL", "PWE (our code)"],
                ["time per shape", "much longer (you run batches of ~200 overnight / for hours)", "about 0.17 s"],
                ["shapes we have", f"{g('nComsol')} (triangles, circles, freeform)", "6,000"],
                ["accuracy", "the reference", f"{g('pweLo')} / {g('pweUp')} GHz from COMSOL ({g('pweN')} cells)"]],
               [3.2 * cm, 6.8 * cm, W - 10 * cm])
    s += [P("The AI needs thousands of examples to learn how 10 shape numbers change the band gap. Making 6,000 in "
            "COMSOL would take weeks. PWE made them in about 20 minutes.")]
    s += [P(f"We tested this directly (study guide section 9.1). Scored on {nfree} freeform shapes simulated in COMSOL:")]
    s += table([["approach", "error vs COMSOL (GHz)"],
                [f"best AI trained only on COMSOL data", f"{g('comsolLo')} / {g('comsolUp')}"],
                ["PWE alone", f"{solver['mae_f_lower']:.2f} / {solver['mae_f_upper']:.2f}"],
                ["PWE + a correction learned from COMSOL (what we use)",
                 f"<b>{corr['mae_f_lower']:.2f} / {corr['mae_f_upper']:.2f}</b>"]],
               [W - 5 * cm, 5 * cm])
    s += [P("So each tool does what it is good at: PWE gives <b>quantity</b>, COMSOL gives <b>accuracy</b>, and the "
            "correction combines them. This is called <b>multi-fidelity</b> learning: cheap low-accuracy data plus a "
            "little expensive high-accuracy data.")]

    s += [H1("4. How do we know PWE is right?")]
    s += fig("pwe_validation.png", W, "(a) Full bands for one cell: COMSOL (black) and PWE (orange dashes) lie on top "
             "of each other. (b), (c) Every gapped COMSOL cell: PWE (up) vs COMSOL (across). Points on the diagonal "
             "line mean they agree.")
    s += bullets([
        f"{g('pweN')} gapped COMSOL cells of four shape families: band edges within {g('pweLo')} / {g('pweUp')} GHz "
        "of COMSOL on average.",
        "It was tuned on only 128 triangle cells, so all the others (circles, freeform shapes, the new triangles) are "
        "an independent test.",
        "For scale: the gap itself is up to about 40 GHz wide and sits near 300 to 350 GHz, so 0.5 GHz is about "
        "0.15 percent.",
    ])
    s += box("The one setting we tuned (in case someone asks about 11.7)",
             "Silicon's textbook permittivity is 11.7. With 11.7, PWE's bands come out too low (grey dotted line in "
             "panel (a)). With 8.91 they match COMSOL exactly. The likely reason is that the COMSOL model is 2D and uses "
             "an <i>effective</i> value for a thin silicon slab (refractive index about 2.99 instead of 3.42). "
             "Shalaev et al. (Nature Nanotechnology, 2019) used 2.965 in the same kind of 2D model. We fitted this "
             "one number on the triangles only; circles and freeform shapes then matched without any extra tuning. "
             "(Check the material value in your COMSOL model so you can say it confidently.)", color="#fff5e6")

    s += [PageBreak(), H1("5. Questions your guide may ask, with answers")]
    qa = [
        ("What is PWE?",
         "A plane-wave expansion solver: a standard, fast way to compute the band structure of a periodic structure. "
         "It does the same job as COMSOL's eigenfrequency study, using waves instead of a mesh."),
        ("Why not just use COMSOL for everything?",
         f"Speed. The AI needs thousands of examples; PWE makes one in 0.2 s. I also tested a COMSOL-only model on "
         f"{g('nFreeTest')} freeform COMSOL shapes: best {g('comsolLo')} / {g('comsolUp')} GHz, against "
         f"{g('corrLo')} / {g('corrUp')} GHz for PWE plus the COMSOL correction."),
        ("How do you know PWE is accurate?",
         f"I compared it with COMSOL on {g('pweN')} gapped cells of four shape families: {g('pweLo')} / {g('pweUp')} GHz "
         "on average, and the gap widths follow COMSOL closely."),
        ("Are 439 plane waves enough?",
         "I re-solved 225 COMSOL cells with 177 to 1,011 plane waves. At 439 the edges are within ~0.5 GHz of the finest "
         "setting, and since the permittivity was calibrated at 439, the agreement with COMSOL (~0.5 GHz) is as good as "
         "with 1,011 waves, which are 5 times slower."),
        ("Why permittivity 8.91 and not 11.7?",
         "The COMSOL model is 2D, so it uses an effective index for the slab. 8.91 (n = 2.99) is the value that matches "
         "COMSOL; Shalaev et al. 2019 used n = 2.965 in a similar 2D model. I fitted it once on triangles and it "
         "transferred to circles and freeform shapes."),
        ("Did you write the solver?",
         "It is part of my project code (src/shapes/pwe.py), written with AI coding help. The method itself is standard "
         "and well documented (Joannopoulos et al.; the MIT program MPB uses the same idea). My contribution is the AI "
         "design pipeline that uses it, and checking it against COMSOL."),
        ("What are its limits?",
         "2D, one polarisation, effective index: a small error (~0.5 GHz), so final designs are checked in COMSOL "
         "and the error is corrected with COMSOL data."),
        ("Is the final design checked in COMSOL?",
         f"Yes. {g('verN')} AI designs for four targets were simulated in COMSOL; {g('verOK')} hit both edges within "
         f"1 GHz (worst {g('verWorst')} GHz)."),
        ("How is accuracy measured?",
         "In GHz on shapes the model never saw. A design's error is the worse of its two edges after re-simulation; "
         "below 1 GHz counts as success. Percentages would hide errors at 300 GHz."),
    ]
    for q, a in qa:
        s += [Paragraph(f"<b>Q: {q}</b>", S["p"]), Paragraph(f"A: {a}", S["bullet"]), Spacer(1, 1)]

    s += [H1("6. A 60-second explanation you can say")]
    s += box("Say it like this",
             "\"To train the AI I need thousands of examples of hole shapes and their band gaps. COMSOL is accurate but "
             "too slow for that, so I used a plane-wave expansion solver. It computes the same bands in a fraction of "
             f"a second by describing the crystal as a sum of simple waves. I checked it against COMSOL on {g('pweN')} "
             "cells: it agrees to under a GHz. Then I used my COMSOL runs to learn a small correction, which "
             f"brings the error down to about {g('mfCorrLo')} / {g('mfCorrUp')} GHz. Finally, {g('verOK')} of {g('verN')} "
             "AI designs simulated in COMSOL hit their target within 1 GHz.\"", color="#eaf6ee")
    return s


def main():
    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=1.9 * cm,
                            bottomMargin=1.8 * cm, title="PWE in plain words", author="Binit Shakya")
    doc.build(story(), onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
