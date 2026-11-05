"""Build the beginner's learning guide -> docs/Learn_From_Zero.pdf

Run from the Project folder (after the exercise scripts have made learn/figs/*.png):
    python learn/run_all.py          # runs every exercise, then this builder
or  python learn/build_learn.py      # only rebuild the PDF
Project numbers (inverse-design table, multi-fidelity) are read from results/optionb/results.json.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "shapes"))
import build_guide as bg  # noqa: E402  (fonts, styles and helpers of the option-B guide)
from build_guide import (A4, MUTED, P, H1, H2, H3, Image, KeepTogether, PageBreak, Paragraph,  # noqa: E402
                         ParagraphStyle, S, Spacer, TableOfContents, W, box, bullets, cm, code, eq, numbered, table)

OUT = "docs/Learn_From_Zero.pdf"
LFIG = os.path.join(HERE, "figs")
ANSWERS = []   # (part, question, answer) collected for the answer section


def fig(path, width=W, caption=None):
    if not os.path.exists(path):
        return [P(f"<i>[missing figure {path}: run python learn/run_all.py]</i>")]
    from PIL import Image as PI
    w, h = PI.open(path).size
    out = [Image(path, width=width, height=width * h / w)]
    if caption:
        out.append(Paragraph(caption, S["cap"]))
    return [KeepTogether(out)]


def lfig(name, width=W, caption=None):
    return fig(os.path.join(LFIG, name), width, caption)


def numbered_from(items, start):
    return [Paragraph(t, S["bullet"], bulletText=f"{k}.") for k, t in enumerate(items, start)]


def exercise(n, script, what, tries):
    scripts = [x.strip() for x in script.replace(" and ", ", ").split(",")]
    runs = "<br/>".join(f"<font name='Mono' size=9>python learn\\{x}</font>" for x in scripts)
    return box(f"Exercise {n} — {script}", [
        f"<b>Run:</b><br/>{runs}",
        f"<b>What it shows:</b> {what}",
        "<b>Then try:</b> " + tries], "#eef6ee")


def check(part, qa):
    """Questions printed now, answers collected for the back of the guide."""
    out = [H3("Check yourself")]
    for q, a in qa:
        ANSWERS.append((part, q, a))
    out += numbered([q for q, _ in qa])
    return out


def on_page(canv, doc):
    canv.saveState()
    canv.setFont("Body", 8.5)
    canv.setFillColor(MUTED)
    if doc.page > 1:
        canv.drawString(2.1 * cm, A4[1] - 1.3 * cm, "Learning the project from zero — waves, Fourier, PWE and the machine learning")
        canv.drawRightString(A4[0] - 2.1 * cm, 1.2 * cm, f"{doc.page}")
        canv.setStrokeColor(bg.colors.HexColor("#d5dae1"))
        canv.line(2.1 * cm, A4[1] - 1.45 * cm, A4[0] - 2.1 * cm, A4[1] - 1.45 * cm)
    canv.restoreState()


def results():
    try:
        with open("results/optionb/results.json") as f:
            return json.load(f)
    except OSError:
        return {}


# =============================================================================================== content
def part0():
    s = [Spacer(1, 2.6 * cm),
         Paragraph("Learning the Project from Zero", S["title"]), Spacer(1, 8),
         Paragraph("Waves, Fourier series, the plane-wave (PWE) solver and every machine-learning method in the "
                   "project — step by step, with worked examples and exercises you run yourself", S["subtitle"]),
         Spacer(1, 22),
         Paragraph("M.Tech project P29 — Binit Shakya, IIT Guwahati", S["subtitle"]),
         Spacer(1, 6), Paragraph("Companion to the option-B study guide (docs/OptionB_Study_Guide.pdf)", S["subtitle"]),
         Spacer(1, 1.0 * cm)]
    s += fig("results/optionb/figs/pipeline.png", W, "The whole project on one page. By the end of this guide you will "
             "understand every box: what it computes, how, and why.")
    s += [PageBreak(), Paragraph("Contents", ParagraphStyle("h1x", parent=S["h1"]))]
    toc = TableOfContents()
    toc.levelStyles = [S["toc1"], S["toc2"]]
    s += [toc, PageBreak()]

    s += [H1("0. How to use this guide")]
    s += [P("This guide assumes nothing beyond school mathematics. Every idea comes in four steps: <b>the idea in words</b>, "
            "<b>a worked example with real numbers</b> (do it with pen and paper), <b>an exercise script</b> you run and "
            "change, and <b>check-yourself questions</b> (answers at the back, part 13). Read the parts in order — each one "
            "uses the previous ones. Parts 1–6 are the physics and the simulator, parts 7–10 the machine learning.")]
    s += [H2("0.1 Running an exercise (do this once now)")]
    s += numbered([
        "Open a terminal: in VS Code use <i>Terminal → New Terminal</i>, or open <i>Windows Terminal / PowerShell</i>.",
        "Go to the project folder and switch on the project's Python (the virtual environment):"])
    s += [code('cd "E:\\MTech\\MTP project\\Project"\n.venv\\Scripts\\activate          # the prompt now starts with (.venv)\n'
               'python learn\\ex01_waves.py')]
    s += numbered([
        "The numbers appear in the terminal. The picture is saved in <font name='Mono'>learn\\figs\\</font> — open "
        "it in VS Code or the file explorer.",
        "To experiment: open the script in VS Code. The settings are the CAPITAL names near the top (for example "
        "<font name='Mono'>F_GHZ = 300.0</font>). Change one, save (Ctrl+S), run the same command again (press the "
        "Up arrow in the terminal to repeat it). Compare with the previous output.",
        "If something breaks, undo your change (Ctrl+Z) and save. You cannot damage the project: the exercises only read "
        "it."])
    s += [H2("0.2 The exercises")]
    s += table([["#", "script", "topic", "part"],
                ["1", "ex01_waves.py", "frequency, wavelength, why a = 242.5 µm", "2"],
                ["2", "ex02_fourier_series.py", "Fourier series of the permittivity pattern", "3"],
                ["3", "ex03_hole_shapes.py", "describing a hole with r(θ); the 3-fold rule; the old triangle's numbers", "4"],
                ["4", "ex04_pwe_1d.py", "the plane-wave method in 1D, by hand and by computer, vs the exact answer", "5"],
                ["5", "ex05_pwe_2d.py", "the project's real 2D solver on circular holes", "6"],
                ["6", "ex06_design_to_band_edges.py", "10 design numbers → holes → fabrication check → band edges", "6"],
                ["7", "ex07_surrogate_gp_vs_nn.py", "learning a function: Gaussian process vs neural network", "7"],
                ["8", "ex08_tandem_toy.py", "why a direct inverse fails; the tandem network", "8"],
                ["9", "ex09_cvae_toy.py", "conditional VAE: many designs for one target", "8"],
                ["10", "ex10_multistart_toy.py", "gradient search from many starts; surrogate exploitation", "8"],
                ["11", "ex11_multifidelity_toy.py", "cheap simulator + learned correction", "9"]],
               [0.8 * cm, 4.6 * cm, 9.6 * cm, 1.2 * cm])
    s += [P("<font name='Mono'>python learn\\run_all.py</font> runs all of them and rebuilds this PDF (about 4 minutes).")]
    s += [PageBreak()]
    return s


def part1():
    s = [H1("1. The project in plain words")]
    s += [P("<b>The device.</b> A thin slab of silicon with a regular pattern of holes, the same platform as the "
            "terahertz chip of Yang et al. (Nature Photonics 2020). Terahertz waves around 300 GHz (wavelength about 1 mm in "
            "air) travel inside the slab. The pattern repeats every <b>a = 242.5 µm</b> and each repeating tile (the "
            "<i>unit cell</i>) contains <b>two holes</b>, A and B.")]
    s += [P("<b>What the pattern does.</b> For a certain range of frequencies, the <i>band gap</i>, waves cannot travel "
            "through the patterned silicon at all. If holes A and B are different, the crystal becomes a <i>valley photonic "
            "crystal</i>: at a boundary between two such crystals (one with A big, one with B big) a wave can travel along the "
            "boundary, even around sharp corners, without bouncing back. That is useful for 6G on-chip links. The band gap "
            "is described by two numbers: its bottom edge <b>f<sub>lower</sub></b> and its top edge <b>f<sub>upper</sub></b>.")]
    s += [P("<b>The forward problem</b> (easy, done by simulation): given the hole shapes, compute f<sub>lower</sub> and "
            "f<sub>upper</sub>. COMSOL does this accurately but slowly; our own PWE solver does it in 0.2 seconds.")]
    s += [P("<b>The inverse problem</b> (the goal of the project): given the f<sub>lower</sub> and f<sub>upper</sub> you "
            "want, find hole shapes that produce them. Simulators cannot run backwards, so we train machine-learning "
            "models to do it. In option B the holes are not limited to triangles: any shape with three-fold symmetry is "
            "allowed, described by a Fourier series (part 4). Then <b>many different shapes give the same band edges</b>, "
            "which makes the inverse problem hard and interesting (part 8).")]
    s += box("The chain of ideas in this guide", [
        "waves and band gaps (2) → Fourier series (3) → hole shapes as Fourier series (4) → the PWE solver in 1D (5) and "
        "2D (6) → learning a function from examples (7) → running it backwards: tandem, CVAE, multi-start (8) → "
        "combining cheap and expensive simulations (9) → the full pipeline on one example (10)."])
    s += [PageBreak()]
    return s


def part2():
    s = [H1("2. Waves and photonic crystals")]
    s += [H2("2.1 Frequency, wavelength, speed"),
          P("A wave repeats in time and in space. <b>Frequency f</b> = how many times per second it repeats (unit Hz; "
            "1 GHz = 10<super>9</super> Hz). <b>Wavelength λ</b> = the distance between two crests. They are tied together "
            "by the speed of the wave:")]
    s += eq(r"c = f\,\lambda \qquad\Rightarrow\qquad \lambda=\frac{c}{f}")
    s += [P("<b>Worked example.</b> f = 300 GHz = 3×10<super>11</super> Hz, c = 2.998×10<super>8</super> m/s. "
            "λ = 2.998×10<super>8</super> / 3×10<super>11</super> = 9.99×10<super>−4</super> m ≈ <b>1.0 mm = 999 µm</b>."),
          P("Inside a material light is slower by the <b>refractive index n</b>: speed c/n, wavelength λ/n. The "
            "<b>permittivity</b> ε (how strongly the material responds to an electric field) is ε = n<super>2</super>. "
            "Bulk silicon: n = 3.42, ε = 11.7. In this project's 2D model the effective values are n ≈ 2.99, ε ≈ 8.91 "
            "(part 6.5). With n = 2.99: λ in silicon = 999 / 2.99 = <b>334 µm</b>.")]
    s += lfig("ex01_waves.png", W, "Exercise 1: a 300 GHz wave in air and in silicon. The grey lines are 242.5 µm apart — "
              "the crystal period is about a quarter of the air wavelength.")
    s += [P("The ratio a/λ is called the <b>normalised frequency</b>: 242.5 / 999 = 0.243. Photonic crystals do interesting "
            "things when a/λ is roughly 0.2–0.5, i.e. when the pattern is comparable to the wavelength. That is why "
            "a = 242.5 µm was chosen for 300 GHz. Scaling the whole crystal by 2 divides every frequency by 2.")]
    s += [H2("2.2 Why a periodic pattern blocks some frequencies"),
          P("Every boundary between silicon and air reflects a little of the wave. In a periodic structure there are many "
            "boundaries, equally spaced. At most frequencies the small reflections arrive with random phases and cancel. "
            "At special frequencies — when the spacing is about half a wavelength inside the material — all reflections "
            "arrive <b>in step</b> and add up to a total reflection. The wave cannot enter: that frequency range is a "
            "<b>band gap</b>. This is the same effect that makes soap bubbles and butterfly wings coloured (Bragg "
            "reflection).")]
    s += [H2("2.3 The crystal and its unit cell")]
    s += lfig("concept_lattice_bz.png", W, "Left: the crystal in real space. Translating by a1 or a2 maps the pattern onto "
              "itself. Right: the same crystal in wavevector space (section 2.4).")
    s += [P("The lattice vectors are <b>a1 = a(1, 0)</b> and <b>a2 = a(1/2, √3/2)</b>, both of length a = 242.5 µm, 60° "
            "apart. The unit cell is the rhombus they span. Hole A sits at one third of the long diagonal, hole B at two "
            "thirds, so every A has three B neighbours at a distance a/√3 = 140 µm — a honeycomb, like graphene.")]
    s += [H2("2.4 The wavevector k and the Brillouin zone"),
          P("A wave travelling in some direction is described by its <b>wavevector k</b>: it points in the direction of "
            "travel and its length is |k| = 2π/λ (how fast the phase turns per µm). In a crystal, k and k + G behave "
            "identically when G is a <i>reciprocal lattice vector</i> (part 6.1), so all distinct waves are found inside one "
            "hexagon in k-space: the <b>Brillouin zone</b>. Special points get names: <b>Γ</b> (k = 0, the centre), "
            "<b>K</b> (a corner) and <b>M</b> (the middle of an edge). Because of the crystal's symmetry, the small orange "
            "triangle Γ–M–K already contains every distinct case."),
          P("Your COMSOL model walks along the triangle's edge with one number, also called k: <b>k = 0 at Γ, 1 at M, 2 at "
            "K, 3 back at Γ</b>. So 'k = 1 and 2' in your data simply means 'the points M and K'.")]
    s += [H2("2.5 The band diagram, f_lower and f_upper"),
          P("For every wavevector k the crystal allows only certain frequencies, the <b>bands</b>. Plotting them along "
            "Γ → M → K → Γ gives the <b>band diagram</b> (figure in part 6, exercise 5). Band 1 is the lowest, band 2 the "
            "next. If the highest point of band 1 is below the lowest point of band 2 there is a <b>complete band gap</b>:")]
    s += eq(r"f_{\rm lower}=\max_{\mathbf{k}}\,f_1(\mathbf{k}),\qquad f_{\rm upper}=\min_{\mathbf{k}}\,f_2(\mathbf{k}),"
            r"\qquad \Delta f = f_{\rm upper}-f_{\rm lower}")
    s += [P("In this project band 1 always peaks at K and band 2 bottoms out at M (or K), so two simulations per design "
            "are enough — the 'M/K shortcut', checked in section 6.4 of the option-B guide. If Δf &lt; 0 the bands overlap "
            "and there is no complete gap.")]
    s += [H2("2.6 What makes it a valley crystal"),
          P("With two <i>identical</i> holes the pattern has six-fold symmetry and bands 1 and 2 touch at K (a "
            "<b>Dirac point</b>). Making the holes different breaks the symmetry between A and B; the touching point "
            "splits and a gap opens <i>at K</i>. The two corner types K and K′ ('valleys') then carry opposite "
            "topological charge, which is what protects the boundary waves. The bigger the difference between A and B, the "
            "wider the gap. Keeping three-fold symmetry is essential, which is why every hole shape in option B is three-fold "
            "symmetric (part 4).")]
    s += exercise(1, "ex01_waves.py", "wavelength in air and silicon at 300 GHz and the normalised frequency a/λ.",
                  "F_GHZ = 252.72 and 321.84 (the IEEE 802.15.3d band edges); N_SI = 3.42.")
    s += check("2", [
        ("What is the wavelength in air at 250 GHz?", "λ = c/f = 2.998×10<super>8</super>/2.5×10<super>11</super> = 1.2 mm."),
        ("Why does a periodic structure have band gaps?", "Reflections from the many equally spaced boundaries add up in "
         "step at certain frequencies, giving total reflection."),
        ("What do k = 1 and k = 2 mean in your COMSOL files?", "The high-symmetry points M and K of the Brillouin zone."),
        ("Define f<sub>lower</sub> and f<sub>upper</sub>.", "f<sub>lower</sub> = highest frequency of band 1 over all k; f<sub>upper</sub> = lowest frequency of "
         "band 2 over all k; the gap lies between them."),
        ("If you make the whole crystal 10% larger, what happens to the gap?", "All frequencies fall by a factor 1.1 "
         "(divide by 1.1)."),
        ("Why do the two holes have to be different?", "Identical holes give a Dirac point (no gap) at K; different holes "
         "break the A/B symmetry and open the valley gap.")])
    s += [PageBreak()]
    return s


def part3():
    s = [H1("3. Fourier series from the beginning")]
    s += [H2("3.1 The idea"),
          P("Any pattern that repeats — a musical note, a heartbeat, the silicon/air pattern of a crystal — can be built "
            "by adding up simple waves (cosines and sines) whose frequencies are 1, 2, 3, ... times the repetition "
            "frequency. A chord is a sum of pure notes; a Fourier series is a recipe telling you <i>how much</i> of each "
            "pure note to add. The 'how much' numbers are the <b>Fourier coefficients</b>.")]
    s += [H2("3.2 The formula (real form)"),
          P("For a pattern g(x) that repeats every a:")]
    s += eq(r"g(x)=a_0+\sum_{m=1}^{\infty}\left[a_m\cos(2\pi m x/a)+b_m\sin(2\pi m x/a)\right]")
    s += eq(r"a_0=\frac{1}{a}\int_0^a g\,dx,\qquad a_m=\frac{2}{a}\int_0^a g(x)\cos(2\pi m x/a)\,dx,\qquad "
            r"b_m=\frac{2}{a}\int_0^a g(x)\sin(2\pi m x/a)\,dx")
    s += [P("In words: <b>a<sub>0</sub> is the average</b> of the pattern; a<sub>m</sub> measures how much the pattern "
            "looks like the m-th cosine; b<sub>m</sub> the same for the m-th sine.")]
    s += [H2("3.3 Why the formula works (orthogonality)"),
          P("Multiply two cosines and average over one period. If they have <i>different</i> m the result is 0; if they "
            "have the same m it is 1/2. Example: the average of cos²(2πx) over a period is 1/2 (because cos² = (1 + cos 2·2πx)/2 "
            "and the cosine part averages to 0), while the average of cos(2πx)·cos(4πx) is 0. So when you multiply g(x) by "
            "cos(2πmx/a) and average, every term of the series vanishes except the m-th, which leaves a<sub>m</sub>/2. "
            "That is where the factor 2 in the formula comes from.")]
    s += [H2("3.4 Worked example: the permittivity of a layered crystal"),
          P("The 1D crystal of part 5: period a = 1, silicon (ε = 8.914) for −F/2 &lt; x &lt; F/2, air (ε = 1) elsewhere, "
            "with silicon fraction F = 0.5. Do this with pen and paper:")]
    s += numbered([
        "<b>Average.</b> a<sub>0</sub> = (silicon fraction)·ε<sub>Si</sub> + (air fraction)·1 = 0.5·8.914 + 0.5·1 = "
        "<b>4.957</b>.",
        "<b>Split the integral.</b> Write ε(x) = 1 + (ε<sub>Si</sub> − 1)·[1 inside silicon, 0 outside]. The constant 1 "
        "gives zero for every m ≥ 1 (a cosine averages to zero over a period). So only the silicon part remains:"])
    s += eq(r"a_m = 2(\varepsilon_{\rm Si}-1)\int_{-F/2}^{F/2}\cos(2\pi m x)\,dx = 2(\varepsilon_{\rm Si}-1)\,"
            r"\frac{\sin(\pi m F)}{\pi m}")
    s += numbered_from([
        "<b>m = 1:</b> a<sub>1</sub> = 2·7.914·sin(π/2)/π = 15.828/3.1416 = <b>5.038</b>.",
        "<b>m = 2:</b> sin(π) = 0, so a<sub>2</sub> = <b>0</b> (with F = 0.5 all even terms vanish).",
        "<b>m = 3:</b> a<sub>3</sub> = 15.828·sin(3π/2)/(3π) = −15.828/9.425 = <b>−1.679</b>.",
        "<b>Sines:</b> the pattern is symmetric about x = 0 (an even function) and sines are odd, so every "
        "b<sub>m</sub> = 0."], 3)
    s += [P("Exercise 2 computes the same numbers with the computer (FFT) and prints them next to the formula: 4.957, "
            "5.038, 0, −1.679, 0, 1.008 — identical.")]
    s += lfig("ex02_fourier_series.png", W, "Exercise 2: adding more and more cosines reproduces the sharp silicon/air "
              "pattern. Near the jumps a small overshoot remains (the Gibbs effect); smooth shapes need far fewer terms.")
    s += [H2("3.5 The complex form (used by the PWE solver)"),
          P("Euler's formula e<super>iθ</super> = cos θ + i sin θ lets us write cosine and sine together. The series "
            "becomes a sum over positive <i>and negative</i> m:")]
    s += eq(r"g(x)=\sum_{m=-\infty}^{\infty}c_m\,e^{i 2\pi m x/a},\qquad c_m=\frac{1}{a}\int_0^a g(x)\,e^{-i2\pi m x/a}\,dx")
    s += [P("For a symmetric (even) real pattern: c<sub>0</sub> = a<sub>0</sub> and c<sub>m</sub> = c<sub>−m</sub> = "
            "a<sub>m</sub>/2. Example: c<sub>1</sub> = 5.038/2 = 2.519 — you will meet this exact number inside the PWE "
            "matrix in part 5. Why bother with complex numbers? Because derivatives become multiplications: "
            "d/dx e<super>iqx</super> = iq·e<super>iqx</super>. That turns a differential equation into an ordinary matrix "
            "problem.")]
    s += [H2("3.6 The FFT"),
          P("The <b>Fast Fourier Transform</b> is the computer's way of getting all c<sub>m</sub> at once from N equally "
            "spaced samples of the pattern: in code, <font name='Mono'>np.fft.fft(samples) / N</font>. The 2D version "
            "(<font name='Mono'>np.fft.fft2</font>) does the same for a 2D picture of the unit cell — that is step 4 of the "
            "project's solver.")]
    s += exercise(2, "ex02_fourier_series.py", "the coefficients by formula and by FFT; the pattern rebuilt from 1, 3, 10 "
                  "and 50 cosines.", "FILL = 0.2 and 0.8 (which coefficients vanish now?); N_TERMS = (1, 2, 5, 200).")
    s += check("3", [
        ("What does a<sub>0</sub> represent?", "The average value of the pattern."),
        ("Why are all b<sub>m</sub> zero for the layer pattern centred at x = 0?", "The pattern is even (symmetric) and sines are odd; "
         "their product averages to zero."),
        ("Compute a<sub>1</sub> for F = 0.25 (ε_Si = 8.914).", "a<sub>1</sub> = 2·7.914·sin(π/4)/π = 15.828·0.7071/3.1416 = 3.563."),
        ("How are c<sub>1</sub> and a<sub>1</sub> related for an even pattern?", "c_1 = c<sub>−1</sub> = a<sub>1</sub>/2."),
        ("Why do sharp edges need many Fourier terms?", "A sudden jump contains fine detail, i.e. high frequencies; only "
         "high-m cosines can build it.")])
    s += [PageBreak()]
    return s


def part4():
    s = [H1("4. Describing a hole shape with a Fourier series")]
    s += [H2("4.1 A hole as a function r(θ)")]
    s += lfig("concept_polar.png", W, "Left: from the hole centre, look in direction θ and measure the distance r to the "
              "edge. Right: the same numbers plotted against θ — a curve that repeats every 360° (here even every 120°).")
    s += [P("Stand at the centre of a hole and look in direction θ. The distance to the edge is r(θ). Going once around "
            "(θ from 0 to 360°) you come back to the start, so <b>r(θ) is a repeating pattern with period 360°</b> — exactly "
            "what part 3 says can be written as a Fourier series, now in the angle θ instead of position x:")]
    s += eq(r"r(\theta)=r_0+\sum_{n}\left[c_n\cos n\theta+s_n\sin n\theta\right]")
    s += [P("r<sub>0</sub> is the average radius (the size), c<sub>n</sub> and s<sub>n</sub> describe the shape. A "
            "<b>circle</b> is just r(θ) = r<sub>0</sub>: every other coefficient is zero.")]
    s += [H2("4.2 Worked example: adding c3 cos 3θ"),
          P("Take r<sub>0</sub> = 50 µm and c<sub>3</sub> = 10 µm. Fill in the table yourself, then compare with exercise 3:")]
    s += table([["θ", "3θ", "cos 3θ", "r = 50 + 10 cos 3θ (µm)"],
                ["0°", "0°", "1", "60 (a bump)"], ["30°", "90°", "0", "50"], ["60°", "180°", "−1", "40 (a dent)"],
                ["90°", "270°", "0", "50"], ["120°", "360°", "1", "60 (the next bump)"], ["180°", "540°", "−1", "40"]],
               [2.5 * cm, 2.5 * cm, 2.5 * cm, 6 * cm])
    s += [P("Three bumps (0°, 120°, 240°) and three dents: a <b>rounded triangle</b>. Negative c<sub>3</sub> swaps bumps "
            "and dents (the triangle points the other way). s<sub>3</sub> sin 3θ is the same shape turned by 30°. Adding "
            "c<sub>6</sub> cos 6θ sharpens or flattens the corners; c<sub>9</sub>, c<sub>12</sub> add finer detail.")]
    s += lfig("ex03_hole_shapes.png", W, "Exercise 3: circle; ± c3; c3 with c6; the sine version; the forbidden cos 2θ "
              "term (two-fold, not three-fold); and the old rounded triangle rebuilt from its first Fourier terms. Red line = "
              "θ = 0, the direction towards the neighbouring hole.")
    s += [H2("4.3 The three-fold rule, step by step"),
          P("Valley crystals need each hole to look the same after a turn of 120°: r(θ + 120°) = r(θ) for every θ. Test one "
            "term:")]
    s += eq(r"\cos\left(n(\theta+120^\circ)\right)=\cos\left(n\theta+n\cdot120^\circ\right)")
    s += [P("This equals cos nθ for every θ only if the extra angle n·120° is a whole number of turns (a multiple of 360°):")]
    s += table([["n", "n·120°", "whole turns?", "allowed?"],
                ["1", "120°", "no", "no"], ["2", "240°", "no", "no"], ["3", "360°", "1", "yes"], ["4", "480°", "no", "no"],
                ["6", "720°", "2", "yes"], ["9", "1080°", "3", "yes"], ["12", "1440°", "4", "yes"]],
               [2 * cm, 3 * cm, 3 * cm, 3 * cm])
    s += [P("So <b>only n = 3, 6, 9, 12, ...</b> may appear. Exercise 3 confirms it numerically: the shapes built from "
            "n = 3 and 6 change by 0.000 µm when turned by 120°, the cos 2θ shape by 8.66 µm. This is a <b>symmetry "
            "selection rule</b>: the physics forbids most coefficients, so the description is short.")]
    s += [H2("4.4 Nine numbers per hole, eighteen per cell"),
          P("The project keeps n = 3, 6, 9, 12, each with a cosine and a sine, plus r<sub>0</sub>: <b>1 + 2·4 = 9 numbers "
            "per hole</b>, in the order [r0, c3, s3, c6, s6, c9, s9, c12, s12]. Two holes give <b>18 numbers</b>: the "
            "input of every machine-learning model. The same 18 numbers describe a circle, a sharp triangle, a rounded "
            "triangle or any freeform shape — that is why one model can learn from all datasets together."),
          P("Each hole measures θ from its own direction 'towards a neighbour of the other type' (θ = 0). For hole A that "
            "direction is 30° in the lab; for hole B it is 210° — the two face each other.")]
    s += [H2("4.5 Worked example: the old rounded triangle"),
          P("The old COMSOL holes are triangles of side L cut by a circle of radius 2L/(3√3) (option-B guide, 3.3). A "
            "flat side faces θ = 0, so r is <i>smallest</i> at θ = 0 and largest at the corners (60°, 180°, 300°). "
            "Exercise 3 evaluates r(θ) at 3,600 angles for L = 168 µm and applies the formulas of part 3:")]
    s += table([["n", "0", "1, 2", "3", "4, 5", "6", "7, 8", "9", "10, 11", "12"],
                ["c_n (µm)", "56.87", "0", "−9.08", "0", "0.10", "0", "1.11", "0", "−0.63"]],
               [2 * cm] + [1.6 * cm] * 9)
    s += [P("Everything not a multiple of 3 is zero, as the rule says. c<sub>3</sub> is negative because the flat side "
            "(small r) is at θ = 0. Dividing: r<sub>0</sub>/L = 0.339 and c<sub>3</sub>/r<sub>0</sub> = −0.160 — any "
            "rounded triangle of any size has these ratios.")]
    s += [H2("4.6 The design space and the fabrication rules"),
          P("For inverse design the project uses <b>10 numbers</b>: [r0, c3, s3, c6, s6] for each hole (c9 to s12 set to "
            "zero to keep shapes smooth). Allowed ranges: r0 from 25 to 75 µm, |c3|, |s3| up to 0.35·r0, |c6|, |s6| up to "
            "0.12·r0. The optimisers work with u in (0, 1)<super>10</super> and convert, e.g. r0 = 25 + 50u, so every u "
            "inside the box is a legal shape."),
          P("A shape must also be <b>fabricable</b>: silicon walls between holes at least 25 µm, radius never below "
            "10 µm, corners not sharper than an 8 µm radius, holes not overlapping. <font name='Mono'>geometry.check_cell"
            "</font> tests this. Exercise 6 shows a design with r0 = 95 µm failing: its thinnest wall is 10 µm.")]
    s += exercise(3, "ex03_hole_shapes.py", "the table of 4.2, the 120° symmetry test, and the Fourier numbers of the old "
                  "rounded triangle.", "C3 = −0.2·R0; C2 = 0.3·R0 (look at the symmetry test); L_TRI = 140.")
    s += check("4", [
        ("Which Fourier terms can a three-fold symmetric hole have, and why?", "n = 3, 6, 9, ...: only then is n·120° a whole "
         "number of turns, so the term is unchanged by a 120° rotation."),
        ("What shape is r = 40 + 0·cos 3θ?", "A circle of radius 40 µm."),
        ("How do you rotate a c3 cos 3θ shape by 30°?", "Use s3 sin 3θ instead (or a mix of c3 and s3)."),
        ("Why is c3 of the old rounded triangle negative?", "θ = 0 points at a flat side, where r is smallest; a negative "
         "cos 3θ term is smallest at θ = 0."),
        ("How many input numbers do the models get, and why that many?", "18: 9 per hole (r0 plus cosine and sine of n = 3, "
         "6, 9, 12) for 2 holes.")])
    s += [PageBreak()]
    return s


def part5():
    s = [H1("5. The plane-wave expansion (PWE) method in 1D")]
    s += [P("This is the heart of the project's fast simulator. We first do it for a 1D crystal (layers), small enough to "
            "write out by hand; part 6 only adds a second dimension.")]
    s += [H2("5.1 Eigenvalue problems in one paragraph"),
          P("For a square matrix A, an <b>eigenvector</b> v is a direction that A only stretches: A v = λ v; the stretch "
            "factor λ is the <b>eigenvalue</b>. Example: A = [[2, 1], [1, 2]]. Try v = (1, 1): A v = (3, 3) = 3·v, so λ = 3. "
            "Try v = (1, −1): A v = (1, −1), so λ = 1. A computer finds all of them with "
            "<font name='Mono'>np.linalg.eigvalsh(A)</font>. In photonics the 'stretch factors' are the allowed "
            "frequencies (squared), and the eigenvectors are the shapes of the waves.")]
    s += [H2("5.2 The wave equation in the crystal"),
          P("For a wave crossing the layers, Maxwell's equations reduce to one equation for the magnetic field H(x):")]
    s += eq(r"-\frac{d}{dx}\left[\frac{1}{\varepsilon(x)}\,\frac{dH}{dx}\right]=\left(\frac{\omega}{c}\right)^2 H(x),\qquad \omega=2\pi f")
    s += [P("Read it as: 'an operation on H gives back H times a number'. That is an eigenvalue problem: the allowed "
            "(ω/c)² are the eigenvalues. The only thing describing the crystal is ε(x).")]
    s += [H2("5.3 Bloch's theorem: what a wave in a crystal looks like")]
    s += lfig("concept_bloch.png", W * 0.85, "A Bloch wave: an ordinary wave e<super>ikx</super> multiplied by a pattern u(x) that repeats "
              "every unit cell.")
    s += [P("Because the crystal repeats, every allowed wave is a plane wave times a periodic function: "
            "H(x) = e<super>ikx</super>·u(x), with u(x + a) = u(x). And a periodic function has a Fourier series (part 3)! "
            "So, with G<sub>m</sub> = 2πm/a:")]
    s += eq(r"H(x)=e^{ikx}\sum_m h_m e^{iG_m x}=\sum_m h_m\,e^{i(k+G_m)x}")
    s += [P("The field is a sum of <b>plane waves</b> with wavevectors k + G<sub>m</sub> — hence the name "
            "'plane-wave expansion'. The unknowns are the amplitudes h<sub>m</sub>.")]
    s += [H2("5.4 Turning the equation into a matrix"),
          P("Also expand 1/ε(x) = Σ η<sub>m</sub> e<super>iG<sub>m</sub>x</super>. Now substitute, one step at a time:")]
    s += numbered([
        "dH/dx: each term e<super>i(k+G<sub>m'</sub>)x</super> becomes i(k+G<sub>m'</sub>)·e<super>i(k+G<sub>m'</sub>)x</super>.",
        "Multiply by 1/ε: η<sub>j</sub>e<super>iG<sub>j</sub>x</super> times e<super>i(k+G<sub>m'</sub>)x</super> gives "
        "e<super>i(k+G<sub>j+m'</sub>)x</super>, because exponents add (G<sub>j</sub> + G<sub>m'</sub> = G<sub>j+m'</sub>).",
        "Differentiate again: another factor i(k+G<sub>m</sub>) for the term that now has index m = j + m'.",
        "Collect all terms in front of the same plane wave e<super>i(k+G<sub>m</sub>)x</super> (that is j = m − m') and "
        "compare both sides. The two factors of i give i·i = −1, which cancels the minus sign of the equation:"])
    s += eq(r"\sum_{m'}(k+G_m)\;\eta_{m-m'}\;(k+G_{m'})\;h_{m'}=\left(\frac{\omega}{c}\right)^2 h_m")
    s += [P("That is <b>matrix × vector = number × vector</b>: the matrix entries are "
            "M<sub>mm'</sub> = (k+G<sub>m</sub>) η<sub>m−m'</sub> (k+G<sub>m'</sub>). Keep the plane waves m = −M … M and "
            "you have a (2M+1) × (2M+1) eigenvalue problem. Solve it at one k: the eigenvalues give the frequencies of all "
            "bands at that k. Repeat for many k: you have the band diagram.")]
    s += [H2("5.5 Ho's rule: invert the ε matrix"),
          P("There are two ways to get the η matrix. The naive way: Fourier coefficients of 1/ε directly. "
            "<b>Ho's rule</b> (Ho, Chan and Soukoulis 1990): build the matrix of ε coefficients [c<sub>m−m'</sub>] (a "
            "<i>Toeplitz</i> matrix — constant along each diagonal) and <b>invert</b> it. Both are exact with infinitely "
            "many plane waves, but with a finite number Ho's rule handles the sharp silicon/air jumps far better — see 5.7. "
            "The project's solver uses Ho's rule.")]
    s += [H2("5.6 Worked example with only 3 plane waves"),
          P("Silicon fraction 0.5, ε = 8.914, units a = 1, wave at the zone edge k = π. Plane waves m = −1, 0, 1 so "
            "k + G = π − 2π, π, π + 2π = <b>−π, π, 3π</b>.")]
    s += numbered([
        "Coefficients of ε (part 3.5): c<sub>0</sub> = 4.957, c<sub>±1</sub> = 2.519, c<sub>±2</sub> = 0.",
        "Toeplitz matrix [c<sub>m−m'</sub>] (row m, column m'): [[4.957, 2.519, 0], [2.519, 4.957, 2.519], [0, 2.519, 4.957]].",
        "Invert it (computer, or by hand with patience) to get η.",
        "Multiply entry (m, m') by (k+G<sub>m</sub>)(k+G<sub>m'</sub>). Exercise 4 prints the result: "
        "[[3.055, 2.093, −3.191], [2.093, 4.118, −6.278], [−3.191, −6.278, 27.491]].",
        "Eigenvalues: 1.311, 3.791, 29.56. These are (ω/c)², so ω/c = 1.145, 1.947, ...; the normalised frequency "
        "u = a/λ = (ω/c)/(2π) = <b>0.182</b> and <b>0.310</b>."])
    s += [P("The exact answer for this crystal (a textbook formula, also in exercise 4) has its first gap between u = 0.1816 "
            "and 0.2990. With just <b>3 plane waves</b> the lower edge is already right to 0.3% and the upper to 4%. For "
            "the real a = 242.5 µm multiply u by c/a = 1236 GHz: the gap is 224.5 – 369.7 GHz.")]
    s += lfig("ex04_pwe_1d.png", W, "Exercise 4: (a) the bands for all k with 41 plane waves (blue) lie exactly on the exact "
              "solution (black dots); green = the band gap. (b) Gap edges vs the number of plane waves: Ho's rule (solid) "
              "converges almost at once, the naive rule (dashed) slowly.")
    s += [H2("5.7 Convergence: how many plane waves?")]
    s += table([["plane waves", "Ho: lower", "Ho: upper", "naive: lower", "naive: upper"],
                ["3", "0.1822", "0.3099", "0.2175", "0.4377"], ["5", "0.1816", "0.2999", "0.1910", "0.3625"],
                ["11", "0.1816", "0.2991", "0.1856", "0.3184"], ["41", "0.1816", "0.2990", "0.1826", "0.3040"],
                ["81", "0.1816", "0.2990", "0.1821", "0.3015"], ["exact", "0.1816", "0.2990", "0.1816", "0.2990"]],
               [3 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm])
    s += [P("Lesson: more plane waves = more accuracy but a bigger matrix (cost grows like the cube of its size). Ho's rule "
            "gives the accuracy cheaply, which is what makes 0.2 seconds per design possible in 2D.")]
    s += exercise(4, "ex04_pwe_1d.py", "the 3 × 3 matrices of 5.6, the convergence table and the band diagram.",
                  "FILL = 0.3; EPS_SI = 11.7 (does the gap get wider?); M_PLOT = 2.")
    s += check("5", [
        ("What does Bloch's theorem say?", "A wave in a periodic crystal is a plane wave e<super>ikx</super> times a function with the "
         "period of the crystal."),
        ("Why does a derivative become a multiplication in PWE?", "d/dx of e<super>iqx</super> is iq·e<super>iqx</super>: each plane wave is only "
         "multiplied by iq."),
        ("Where do the matrix's off-diagonal entries come from?", "From the Fourier coefficients of 1/ε (η<sub>m−m'</sub>): the "
         "pattern couples plane waves whose G differ by G<sub>m−m'</sub>."),
        ("What do the eigenvalues of the PWE matrix mean?", "(ω/c)² of the allowed waves at that k; take the square root and "
         "divide by 2π (times c/a) to get frequencies."),
        ("What is Ho's rule and why use it?", "Invert the Toeplitz matrix of ε's coefficients instead of using 1/ε's "
         "coefficients; it converges much faster for sharp material boundaries."),
        ("With u = 0.25 and a = 242.5 µm, what is the frequency?", "f = u·c/a = 0.25·1236 GHz = 309 GHz.")])
    s += [PageBreak()]
    return s


def part6():
    s = [H1("6. The 2D solver of the project (src/shapes/pwe.py)")]
    s += [H2("6.1 Reciprocal lattice vectors"),
          P("In 2D the periodic part of the field repeats along a1 and a2, so its Fourier series uses vectors "
            "G = m·b1 + n·b2 with two whole numbers m, n. The b's are chosen so that b<sub>i</sub>·a<sub>j</sub> = 2π if "
            "i = j and 0 otherwise:")]
    s += eq(r"\mathbf{b}_1=\frac{2\pi}{a}\left(1,\,-\frac{1}{\sqrt{3}}\right),\qquad \mathbf{b}_2=\frac{2\pi}{a}\left(0,\,\frac{2}{\sqrt{3}}\right)")
    s += [P("<b>Check one by hand:</b> a1·b1 = a(1, 0)·(2π/a)(1, −1/√3) = 2π (correct). a2·b1 = a(1/2, √3/2)·(2π/a)(1, −1/√3) = "
            "2π(1/2 − 1/2) = 0 (correct). The Brillouin zone of part 2.4 is the hexagon around the origin of this G-lattice; "
            "M = (b1 + b2)/2 and K = (4π/3a, 0).")]
    s += [H2("6.2 The 2D equation"),
          P("For the polarisation used here (TE: magnetic field H<sub>z</sub> out of the plane, electric field in the "
            "plane) the 1D equation keeps its form; the products of wavenumbers become dot products of vectors:")]
    s += eq(r"\sum_{\mathbf{G}'}(\mathbf{k}+\mathbf{G})\cdot(\mathbf{k}+\mathbf{G}')\,\left[\varepsilon^{-1}\right]_{\mathbf{G}\mathbf{G}'}"
            r"\,h_{\mathbf{G}'}=\left(\frac{\omega}{c}\right)^2h_{\mathbf{G}}")
    s += [H2("6.3 What the code does, step by step")]
    s += numbered([
        "<b>Grid.</b> The unit cell is divided into 96 × 96 pixels, each further split into 4 × 4 sub-pixels.",
        "<b>Holes.</b> For every sub-pixel the code knows its distance ρ and angle θ from the nearest hole A and B (computed "
        "once). A sub-pixel is air if ρ &lt; r(θ) — <i>this is where the Fourier hole shape of part 4 enters</i>. Any "
        "family works: circle, triangle or the 9 Fourier numbers.",
        "<b>Smooth ε.</b> Each pixel's ε is the average of its 16 sub-pixels (edge pixels get in-between values), so a "
        "tiny change of shape gives a tiny change of frequency — important for gradient-based design.",
        "<b>FFT.</b> <font name='Mono'>np.fft.fft2</font> of the ε picture gives the coefficients ε<sub>G</sub>.",
        "<b>Toeplitz and Ho's rule.</b> Keep the 439 vectors G with |G| ≤ 11|b1|. Build the 439 × 439 matrix "
        "[ε<sub>G−G'</sub>] and invert it — once per design.",
        "<b>Per k-point.</b> Multiply entry (G, G') by (k+G)·(k+G'), find the 2 smallest eigenvalues "
        "(<font name='Mono'>scipy.linalg.eigh</font>), convert: f = c·√λ / (2π).",
        "<b>Band edges.</b> Do step 6 at M and at K only: f<sub>lower</sub> = max(band 1 at M, at K), "
        "f<sub>upper</sub> = min(band 2 at M, at K)."])
    s += [P("The core of steps 4–6 in the real code (simplified):")]
    s += [code("""eps = self.eps_grid(holeA, holeB)              # steps 1-3: 96 x 96 picture of eps
eg = np.fft.fft2(eps) / eps.size                # step 4: Fourier coefficients
T = eg[self.dm, self.dn]                        # step 5: Toeplitz matrix eps_(G-G')
inv = np.linalg.inv(T)                          #         Ho's rule
kv = kvec + self.G                              # step 6: all vectors k + G
M = (kv @ kv.T) * inv                           #         (k+G).(k+G') [eps^-1]
lam = eigh(M, eigvals_only=True, subset_by_index=[0, 1])   # two lowest bands
f_GHz = np.sqrt(lam) * 1e6 * C0 / (2 * np.pi) / 1e9          # lengths are in um""")]
    s += [H2("6.4 Accuracy and speed"),
          P("Exercise 5 repeats the convergence test of part 5 in 2D (circles of 60 and 35 µm). From 91 to 693 plane waves "
            "the gap changes from 22.95 to 23.82 GHz; the project uses 439 (23.72 GHz), within about 0.1–0.4 GHz of the "
            "converged value. Setting up the solver takes 0.17 s, one design at M and K about 0.2 s. Against COMSOL the "
            "solver is within about 1 GHz for all 177 COMSOL cells (option-B guide, 6.3).")]
    s += [H2("6.5 Calibration: why ε = 8.91 and not 11.7"),
          P("With bulk silicon's ε = 11.7 every PWE frequency came out about 13% too low compared with COMSOL — the same "
            "ratio for every design and every k. A constant ratio means a different material constant, not a wrong geometry. "
            "Fitting ε gives 8.914 (n = 2.986), after which all triangle cells agree within about 1 GHz, and the circle "
            "cells (never used in the fit) agree just as well. The likely reason: the 2D COMSOL model uses an "
            "<i>effective index</i> for the thin slab; Shalaev et al. (2019) used n<sub>eff</sub> = 2.965 the same way. "
            "Please confirm in the COMSOL material settings.")]
    s += lfig("ex05_pwe_2d.png", W, "Exercise 5: left, the ε picture the solver builds (dark = silicon). Middle, equal "
              "circles: bands 1 and 2 touch at K (Dirac point, 331.6 GHz), and band 2 at M (321.5 GHz) dips below it — no "
              "complete gap. Right, unequal circles: a 23.7 GHz valley gap opens.")
    s += [H2("6.6 From design numbers to band edges"),
          P("Exercise 6 runs the complete forward chain that the inverse methods must invert: 10 design numbers → two "
            "Fourier holes → fabrication check → PWE band edges. Holes with the <i>same average radii</i> (55 and 35 µm) but "
            "different shapes give gaps from 10.4 to 15.4 GHz — the shape matters, which is the whole point of option B.")]
    s += lfig("ex06_design_to_band_edges.png", W, "Exercise 6: five designs. The last one breaks the fabrication rules "
              "(10 µm wall) and is never simulated.")
    s += exercise(5, "ex05_pwe_2d.py and ex06_design_to_band_edges.py", "the solver's ε picture, band diagrams for equal "
                  "and unequal holes, convergence, and the full forward chain.",
                  "R_A, R_B = 55, 40 in ex05; add your own design to DESIGNS in ex06 and predict the gap before running.")
    s += check("6", [
        ("What is a reciprocal lattice vector?", "A vector G = m b1 + n b2 with b<sub>i</sub>·a<sub>j</sub> = 2πδ<sub>ij</sub>; the Fourier series of "
         "anything periodic in the crystal uses exactly these wavevectors."),
        ("Where does the hole shape enter the PWE solver?", "When building the ε picture: a sub-pixel is air if its distance "
         "to the hole centre is smaller than r(θ)."),
        ("Why average 16 sub-pixels per pixel?", "So ε changes smoothly when the shape changes; frequencies then change "
         "smoothly too, which gradient-based design needs."),
        ("What is computed once per design and what once per k-point?", "The inverse ε matrix once per design; the "
         "eigenvalue problem once per k-point."),
        ("Why was ε fitted?", "Bulk-silicon ε made every frequency 13% low by the same factor; fitting one number "
         "(8.914) matched COMSOL for all cells, consistent with an effective index in the 2D model.")])
    s += [PageBreak()]
    return s


def part7():
    s = [H1("7. Machine learning: learning a function from examples")]
    s += [H2("7.1 What a model and training are"),
          P("A <b>model</b> is a formula with adjustable numbers called <b>parameters</b>. <b>Training</b> = choosing the "
            "parameters so the model's outputs match known examples. The mismatch is measured by a <b>loss</b>, usually the "
            "mean squared error (MSE) = average of (prediction − truth)²."),
          P("<b>Worked example: gradient descent by hand.</b> Model y = w·x, one parameter w. Examples: (x, y) = (1, 2) and "
            "(2, 4) — the right answer is w = 2. Loss L(w) = ½[(w·1 − 2)² + (w·2 − 4)²]. Its slope (derivative) is "
            "dL/dw = (w − 2)·1 + (2w − 4)·2 = 5w − 10. <b>Gradient descent</b>: repeatedly step downhill, "
            "w ← w − (learning rate)·dL/dw. Start at w = 0 with learning rate 0.1:")]
    s += table([["step", "w", "slope 5w − 10", "new w = w − 0.1·slope"],
                ["1", "0", "−10", "1.0"], ["2", "1.0", "−5", "1.5"], ["3", "1.5", "−2.5", "1.75"], ["4", "1.75", "−1.25", "1.875"]],
               [2 * cm, 2.5 * cm, 4 * cm, 6 * cm])
    s += [P("w approaches 2. A neural network does exactly this, with about 137,000 parameters instead of one; the "
            "computer finds all the slopes at once with <i>backpropagation</i> (PyTorch's <font name='Mono'>loss.backward()"
            "</font>). <b>Adam</b> is an improved gradient descent that adapts the step size of every parameter.")]
    s += [H2("7.2 Training, validation, test — and honest errors"),
          P("If you judge a model on the examples it learned from, it looks perfect even when it has only memorised them "
            "(<b>overfitting</b>). So the data are split: <b>training</b> (to fit), <b>validation</b> (to choose settings) and "
            "<b>test</b> (touched once, at the end). The project reports the <b>mean absolute error (MAE) in GHz</b> on test "
            "cells — a number you can compare with the 1 GHz agreement between PWE and COMSOL. Repeating with different "
            "random splits (the 'seeds' you ran overnight) shows how much a result depends on luck.")]
    s += [H2("7.3 Neural networks")]
    s += lfig("concept_nn.png", W * 0.9, "A multi-layer perceptron (MLP). Every line is a weight; every neuron adds up its "
              "weighted inputs and applies a bend (the activation).")
    s += [P("Each neuron computes output = SiLU(w<sub>1</sub>·in<sub>1</sub> + w<sub>2</sub>·in<sub>2</sub> + … + b). "
            "SiLU(z) = z/(1 + e<super>−z</super>) is a smooth bend; without bends the whole network would be a straight-line "
            "formula. The project's forward network: 18 inputs → 3 hidden layers of 256 → 2 outputs. "
            "<b>Count the parameters:</b> 18·256 + 256 = 4,864; two layers of 256·256 + 256 = 65,792 each; 256·2 + 2 = 514; "
            "total <b>136,962</b>. Inputs and outputs are normalised (minus mean, divided by standard deviation) so all "
            "numbers are of similar size.")]
    s += [P("<b>Ensemble.</b> Train 5 networks from different random starts. Their average is the prediction; their "
            "disagreement is an uncertainty estimate — where they disagree, the data were thin.")]
    s += [H2("7.4 Gaussian processes"),
          P("A <b>Gaussian process (GP)</b> works on one assumption: <i>similar inputs give similar outputs</i>. "
            "'Similar' is defined by a <b>kernel</b>, here the RBF kernel")]
    s += eq(r"k(\mathbf{x},\mathbf{x}')=\sigma^2\exp\left(-\sum_j\frac{(x_j-x'_j)^2}{2\ell_j^2}\right)")
    s += [P("A prediction at a new x is a weighted combination of the training outputs, with large weights for nearby "
            "examples. Because it knows how far x is from the data, a GP also gives a <b>standard deviation</b>: small near "
            "examples, large far away. Each length scale l<sub>j</sub> says how far one must move along feature j before "
            "the output changes ('ARD'); a huge l<sub>j</sub> means feature j barely matters. The cost: training needs to "
            "invert an n × n matrix, so time grows like n³ — fine for 100s of examples, slow for 1000s.")]
    s += lfig("ex07_surrogate.png", W, "Exercise 7: with 8 examples the GP (orange) is better than the network ensemble (blue) "
              "and its uncertainty band shows where it is guessing. With 60 examples both are accurate.")
    s += [P("Exercise 7 prints: with 8 examples GP error 0.045 vs network 0.080; with 60 examples 0.008 vs 0.010. The "
            "project's learning curve shows the same pattern at a larger scale: the GP wins with few hundred cells, the "
            "network keeps improving with thousands (option-B guide, chapter 8). This is also why, in stage 1 (only L1 "
            "and L2, 61 cells), a GP beat the neural networks.")]
    s += exercise(7, "ex07_surrogate_gp_vs_nn.py", "GP and network ensemble learning the same function from 8 and 60 "
                  "examples, with uncertainty bands.", "N_SMALL = 4; HIDDEN = 8; STEPS = 300 (under-trained).")
    s += check("7", [
        ("What is a loss function?", "A number measuring how wrong the model is on the examples, e.g. the mean squared "
         "error; training minimises it."),
        ("Do one more gradient-descent step of the table in 7.1.", "w = 1.875, slope = 5·1.875 − 10 = −0.625, new w = "
         "1.875 + 0.0625 = 1.9375."),
        ("Why keep a test set aside?", "To measure the error on cells the model never saw; errors on training cells hide "
         "overfitting."),
        ("What does the spread of an ensemble tell you?", "How uncertain the prediction is: large spread = the networks "
         "disagree = little data nearby."),
        ("When would you prefer a GP over a neural network?", "With few examples (tens to a few hundred) and when you want "
         "a calibrated uncertainty."),
        ("Why does the project's network have SiLU between layers?", "Without a non-linear activation the stacked layers "
         "collapse into one linear formula, which cannot fit curved relations.")])
    s += [PageBreak()]
    return s


def part8(res):
    s = [H1("8. Inverse design: running the model backwards")]
    s += [H2("8.1 The problem, and why it is hard"),
          P("The forward surrogate (part 7) answers 'shape → band edges' instantly. Inverse design asks 'band edges → "
            "shape'. With 10 design numbers and only 2 targets, <b>many shapes give the same band edges</b> (the "
            "option-B gallery found 24 different verified shapes for one target). The toy used in exercises 8–10 has the "
            "same structure: design x = (x1, x2) in a square, response y = x1² + x2². For y = 0.5 every point on the circle "
            "of radius √0.5 is a correct design.")]
    s += lfig("concept_inverse_methods.png", W, "The four inverse methods compared in the project. All of them use the "
              "forward surrogate; all their designs are re-checked with the fabrication rules and the PWE solver.")
    s += [H2("8.2 Why the direct inverse fails"),
          P("Obvious idea: train a network on examples (y → x). But for y = 0.5 it has seen (0.7, 0), (−0.7, 0), (0, 0.7), "
            "(0, −0.7) … and minimising the squared error makes it predict their <b>average</b>, near (0, 0) — whose "
            "response is y = 0, nowhere near 0.5. Exercise 8 shows exactly this: every direct design lands at the centre, "
            "average miss 0.52.")]
    s += [H2("8.3 The tandem network (Liu et al. 2018)"),
          P("Two steps. (1) Train the <b>forward</b> network x → y as usual. (2) <b>Freeze</b> it and put a new inverse "
            "network in front of it. Train only the inverse network, with the loss measured on the <i>response</i>:")]
    s += eq(r"\mathcal{L}=|F(G(\mathbf{t}))-\mathbf{t}|^2+\lambda\,P(G(\mathbf{t}))")
    s += [P("G is the inverse network, F the frozen forward network, t the target, P the fabrication penalty. The "
            "inverse network is no longer forced to reproduce one particular stored design, only <i>some</i> design that "
            "works, so it settles on one consistent family of answers. Exercise 8: average miss 0.005 instead of 0.52.")]
    s += [P("In the project: G outputs u in (0, 1)<super>10</super> through a sigmoid (always a legal shape), and the "
            "penalty pushes it away from thin walls. Strengths: instant (one network evaluation) and the most accurate "
            "method. Weakness: <b>one design per target</b> — no alternatives.")]
    s += lfig("ex08_tandem.png", W, "Exercise 8: left, direct-inverse designs (red) collapse to the centre; tandem designs "
              "(blue) sit on the correct circles. Right: achieved vs target response.")
    s += [H2("8.4 Conditional variational autoencoder (CVAE)"),
          P("Goal: <b>many</b> designs per target. The trick is a short random code z (here 1 number, in the project 4) that "
            "selects <i>which</i> of the correct designs you get."),
          P("<b>Training</b> uses two networks: an <b>encoder</b> reads a known design and its response (x, y) and outputs a "
            "code z; a <b>decoder</b> reads (z, y) and must rebuild x. The loss has two parts:")]
    s += eq(r"\mathcal{L}=|\hat{\mathbf{x}}-\mathbf{x}|^2+\beta\;{\rm KL}\left(q(z|\mathbf{x},y)\;\Vert\;\mathcal{N}(0,1)\right)")
    s += [P("The first term says 'rebuild the design'; the second says 'make the codes look like a standard normal "
            "distribution N(0, 1)'.")]
    s += [P("The KL term (Kullback–Leibler divergence) pushes the codes to fill a standard normal distribution with no "
            "gaps, so that <i>any</i> random code is meaningful later. For a normal distribution with mean μ and variance "
            "σ² it is ½(μ² + σ² − 1 − ln σ²), zero exactly when μ = 0 and σ = 1. To train through the random sampling we "
            "write z = μ + σ·(random normal number) — the <b>reparameterisation trick</b>."),
          P("<b>Using it:</b> throw the encoder away. For a target y, draw 300 random codes z and decode each: 300 designs. "
            "Exercise 9: all 300 hit y within 0.02 on average and are spread all around the circle (357°)."),
          P("<b>β is a balance.</b> Too large and the codes carry no information, so designs ignore the target "
            "(the exercise with β = 0.5 misses by about 0.44). Too small and random codes fall in gaps and give poor designs. "
            "β depends on how the data are scaled: the project uses β = 0.5 on normalised data with a 4-number code.")]
    s += lfig("ex09_cvae.png", W * 0.55, "Exercise 9: 300 CVAE designs for each of three targets; dashed circles are the sets "
              "of correct designs.")
    s += [H2("8.5 Multi-start gradient search"),
          P("No inverse network at all. Start from 64 random designs and move each one downhill on "
            "(F(design) − target)² + λ·penalty, using the forward network's gradient — gradient descent (7.1) on the "
            "<i>design</i> instead of on the network's parameters. The design is written as u = sigmoid(w) so it can never "
            "leave the allowed box. Different starts end at different correct designs (exercise 10, part 1). Cost: a few "
            "hundred forward evaluations per start — about 0.6 s per target in the project.")]
    s += lfig("ex10_multistart.png", W, "Exercise 10: left, 30 searches from random starts (orange) each end at a different "
              "correct design (green). Right, searching on a weak surrogate: it believes every design is perfect (blue), the "
              "re-simulation disagrees (orange).")
    s += [H2("8.6 The trap: surrogate exploitation"),
          P("The forward network is not the truth; it has small errors. An optimiser that searches hard finds exactly the "
            "places where the network is wrong <i>in its favour</i>. Exercise 10, part 2: with a weak surrogate the search "
            "reports a perfect 0.000 error, but the true error is 0.051 on average and up to 0.157. Two consequences in the "
            "project: (1) <b>every design is re-simulated with PWE</b> before it counts; (2) searching harder is not always "
            "better — the tandem, which cannot search, came out most accurate.")]
    s += [H2("8.7 CVAE + refine"),
          P("Combine the two: take the CVAE's designs as good, varied starting points and run a short gradient search "
            "(100 steps) from each. The CVAE supplies diversity, the search supplies accuracy, and starting close to an "
            "answer keeps the search short, which limits exploitation.")]
    s += [H2("8.8 How the project compares them")]
    s += numbered([
        "Take 100 test cells with a gap that no model saw; their band edges are the targets.",
        "Each method proposes designs; the surrogate ranks them; the exact fabrication check removes illegal ones.",
        "The chosen design is <b>re-simulated with PWE</b>; the error |PWE − target| in GHz is what counts.",
        "Score: share of targets hit within 1 GHz, MAE, how many <i>different</i> verified designs per target (up to 8), "
        "and time per target."])
    inv = res.get("inverse", {})
    if inv:
        nice = {"tandem": "Tandem NN", "cvae": "CVAE", "multistart": "Multi-start", "cvae_refine": "CVAE + refine"}
        rows = [["method", "within 1 GHz", "MAE f_lower / f_upper (GHz)", "different designs (of 8)", "s per target"]]
        for k, v in inv.items():
            rows.append([nice.get(k, k), f"{100 * v['success_rate_within_1GHz']:.0f}%",
                         f"{v['verified_mae_GHz'][0]:.2f} / {v['verified_mae_GHz'][1]:.2f}",
                         f"{v.get('distinct_verified_designs_per_target', 1.0):.1f}", f"{v['seconds_per_target']:.2f}"])
        s += table(rows, [3.2 * cm, 2.4 * cm, 4.6 * cm, 3.6 * cm, 2.4 * cm])
        sm = {}
        if os.path.exists("results/optionb/seed_summary.json"):
            with open("results/optionb/seed_summary.json") as f:
                sm = json.load(f)
        if sm.get("n", 0) >= 2:
            sr = {k: v["success_rate_within_1GHz"] for k, v in sm["inverse"].items()}
            txt = "; ".join(f"{nice.get(k, k)} {100 * v['mean']:.0f} ± {100 * v['std']:.0f}%" for k, v in sr.items())
            s += [Paragraph(f"Project results, seed 0 (results/optionb/results.json). Repeated with {sm['n']} random splits, "
                            f"the share within 1 GHz is (mean ± spread): {txt}. The ranking does not depend on luck.", S["cap"])]
        else:
            s += [Paragraph("Project results (results/optionb/results.json), seed 0.", S["cap"])]
    s += [P("<b>How to read it:</b> the tandem is the most accurate but gives one design; the CVAE alone is varied but less "
            "accurate; multi-start is accurate and varied but slower; CVAE + refine gives nearly 8 different verified "
            "designs per target at 97% success — the best trade-off. For a thesis, 'best' depends on what the designer "
            "needs: one answer fast, or many to choose from (for example for fabrication tolerance).")]
    s += exercise("8–10", "ex08_tandem_toy.py, ex09_cvae_toy.py, ex10_multistart_toy.py",
                  "direct inverse vs tandem; CVAE sampling; multi-start search and surrogate exploitation.",
                  "in ex09 BETA = 0.5 and 0.002; in ex10 N_TRAIN = 400 (does the exploitation gap shrink?).")
    s += check("8", [
        ("Why does a direct inverse network fail when many designs give the same response?", "Minimising squared error "
         "makes it predict the average of all correct designs, which is usually not a correct design."),
        ("In the tandem, which network learns in step 2, and what is the loss?", "Only the inverse network; the loss is "
         "(forward(inverse(target)) − target)² plus the fabrication penalty."),
        ("What is the code z of a CVAE for?", "It selects which of the many correct designs is produced; random codes give "
         "different designs for the same target."),
        ("What happens if β is too large?", "The KL term dominates, the codes carry no information about the design, and "
         "designs stop matching the target well."),
        ("What is surrogate exploitation and how does the project guard against it?", "Optimisers find designs where the "
         "surrogate's error flatters them; every design is re-simulated with PWE before it is counted."),
        ("Why combine CVAE and gradient search?", "The CVAE gives varied starting points, the short search makes each "
         "accurate; together accurate AND diverse.")])
    s += [PageBreak()]
    return s


def part9(res):
    mf = res.get("multifidelity", {})
    s = [H1("9. Multi-fidelity: cheap simulations corrected by expensive ones")]
    s += [P("The project has two simulators: PWE (0.2 s, about 1 GHz off) and COMSOL (accurate, minutes). The "
            "Kennedy–O'Hagan idea: model the expensive answer as the cheap answer plus a correction δ, and learn δ with a "
            "Gaussian process from the few cells where both exist:")]
    s += eq(r"f_{\rm COMSOL}(\mathbf{x})\approx f_{\rm PWE}(\mathbf{x})+\delta(\mathbf{x}),\qquad \delta\sim\mathcal{GP}")
    s += [P("Why it works: the cheap simulator already captures the hard, wiggly part; the difference is usually smooth "
            "and small, and a GP learns smooth things from very few points. Exercise 11 makes this concrete: with only 4 "
            "expensive points, a GP on the expensive function alone is badly wrong (error 3.99), the cheap simulator alone "
            "is off by 0.75, and cheap + learned correction is essentially exact.")]
    s += lfig("ex11_multifidelity.png", W * 0.8, "Exercise 11: four expensive runs (black dots) are not enough to learn the "
              "function (orange), but they are plenty to learn the smooth difference between cheap and expensive (green).")
    if mf:
        lofo = mf.get("leave_family_out", {})
        txt = (f"In the project the correction GP uses the 18 shape features and the {mf['n']} COMSOL cells with a gap. "
               f"On cells it did not see (10-fold cross-validation) it reduces the f<sub>lower</sub> error from "
               f"{mf['pwe_raw_mae'][0]:.2f} to {mf['pwe_plus_gp_cv_mae'][0]:.2f} GHz and the gap error from "
               f"{mf['gap_mae']['pwe_raw']:.2f} to {mf['gap_mae']['pwe_plus_gp']:.2f} GHz.")
        if lofo:
            parts = [f"{k.replace('comsol_', '')} {v['pwe_raw_mae'][0]:.2f} → {v['pwe_plus_gp_mae'][0]:.2f}"
                     for k, v in lofo.items()]
            txt += (" The catch: when a whole shape family is left out and predicted from the others, the correction "
                    f"does not always help (f<sub>lower</sub> GHz: {'; '.join(parts)}). The correction depends on the "
                    "shape, so it needs COMSOL cells of each new family — for freeform shapes, the gallery designs.")
        s += [P(txt)]
    s += exercise(11, "ex11_multifidelity_toy.py", "cheap simulator, expensive simulator and the learned correction.",
                  "make the bias wigglier (add 2·sin(10x) in cheap()); move the 4 expensive points.")
    s += check("9", [
        ("Write the multi-fidelity model in one line.", "expensive(x) ≈ cheap(x) + δ(x), with δ learned by a GP from the "
         "cells where both exist."),
        ("Why can the correction be learned from few points?", "The difference between the simulators is smooth and "
         "simple even when the function itself is not."),
        ("Why does the project's correction not transfer to an unseen shape family?", "The PWE–COMSOL difference depends on "
         "the hole shape; a family it never saw can have a different bias.")])
    s += [PageBreak()]
    return s


def part10(res):
    sc = res.get("showcase", {})
    s = [H1("10. The whole pipeline on one example")]
    s += [P("A designer asks for a band gap from <b>300 to 330 GHz</b> (inside the 6G band). Here is everything that "
            "happens, with the part of this guide that explains each step:")]
    s += numbered([
        "<b>Data (parts 4, 6).</b> 6,000 random fabricable cells were described by their 18 Fourier numbers and labelled "
        "with PWE at M and K (about 4 minutes in total). The PWE solver was calibrated (ε = 8.914) and checked against all "
        "177 COMSOL cells.",
        "<b>Forward surrogate (7).</b> An ensemble of 5 networks learned 18 numbers → (f<sub>lower</sub>, "
        "f<sub>upper</sub>), with about 0.13 / 0.17 GHz error on unseen cells.",
        "<b>Inverse models (8).</b> The tandem and the CVAE were trained using the surrogate.",
        "<b>Proposals (8).</b> For the target (300, 330): 512 CVAE designs, 256 multi-start designs and 256 CVAE + refine "
        "designs.",
        "<b>Checks (4.6, 8.6).</b> Every candidate is ranked by the surrogate, checked against the fabrication rules, and "
        "re-simulated with PWE; only those within 1 GHz of both targets are kept, and near-duplicates (including the "
        "same shape with A and B swapped or mirrored) are removed.",
        f"<b>Result.</b> {sc.get('n_verified_within_1GHz', 24)} different verified shapes, all giving 300 / 330 GHz to "
        "within 1 GHz (option-B guide, figure 'showcase'). They are exported to "
        "<font name='Mono'>data/pwe/showcase_for_comsol.txt</font>.",
        "<b>Final word (9).</b> Simulate some of them in COMSOL. Their COMSOL results also extend the multi-fidelity "
        "correction to freeform shapes — the active-learning loop of the pipeline figure."])
    s += fig("results/optionb/figs/showcase.png", W, "Different hole shapes, same band gap: the non-uniqueness that the "
             "inverse methods of part 8 are built to handle.")
    s += [PageBreak()]
    return s


def part11():
    s = [H1("11. Glossary")]
    rows = [["term", "meaning"],
            ["a, a1, a2", "lattice constant (242.5 µm) and lattice vectors; the pattern repeats along them"],
            ["unit cell", "the smallest tile that repeats; here a rhombus with two holes A and B"],
            ["permittivity ε, index n", "material constants, ε = n²; silicon 11.7 bulk, 8.914 fitted here"],
            ["wavevector k", "direction and phase rate of a wave, |k| = 2π/λ"],
            ["reciprocal lattice G", "G = m b1 + n b2; the wavevectors of anything periodic in the crystal"],
            ["Brillouin zone", "hexagon of distinct wavevectors; Γ centre, M edge middle, K corner"],
            ["band, band diagram", "allowed frequencies for each k; plotted along Γ–M–K–Γ"],
            ["band gap, f_lower, f_upper", "frequencies with no allowed wave; its bottom and top edges"],
            ["Dirac point", "where bands 1 and 2 touch at K (equal holes)"],
            ["valley photonic crystal", "honeycomb crystal with unequal holes: gap at K and K′, protected edge waves"],
            ["Fourier series / coefficients", "a repeating pattern written as a sum of cosines and sines / their amounts"],
            ["FFT", "fast algorithm computing all Fourier coefficients from samples"],
            ["C3 symmetry, selection rule", "unchanged by a 120° turn; allows only harmonics n = 3, 6, 9, 12"],
            ["Bloch wave", "plane wave × a function with the crystal's period"],
            ["PWE", "plane-wave expansion: field as plane waves, Maxwell's equation as a matrix eigenproblem"],
            ["Toeplitz matrix, Ho's rule", "matrix constant along diagonals [ε<sub>G−G'</sub>]; inverting it gives fast convergence"],
            ["eigenvalue / eigenvector", "A v = λ v; here λ = (ω/c)² and v the wave's plane-wave amounts"],
            ["TE polarisation", "magnetic field out of the plane, electric field in the plane"],
            ["COMSOL", "finite-element simulator; accurate, slow; the reference (high fidelity)"],
            ["surrogate", "fast learned model that imitates a simulator"],
            ["parameter, loss, gradient descent", "adjustable numbers of a model; error measure; downhill updates"],
            ["MLP, SiLU, Adam", "layered neural network; its smooth activation; its optimiser"],
            ["ensemble", "several networks; mean = prediction, spread = uncertainty"],
            ["Gaussian process, kernel, ARD", "regression from similarity; the similarity function; one length scale per input"],
            ["overfitting, test set", "memorising instead of learning; data kept aside to measure honest errors"],
            ["MAE", "mean absolute error, always in GHz here"],
            ["inverse design, non-uniqueness", "target → design; many designs give the same target"],
            ["tandem network", "inverse net trained through a frozen forward net, loss on the response"],
            ["CVAE, latent code z, β, KL", "generative model; random code picks the design; weight of the KL term; distance "
             "between distributions"],
            ["multi-start gradient search", "many downhill searches on the design from random starts"],
            ["surrogate exploitation", "optimiser finding the surrogate's errors instead of real solutions"],
            ["multi-fidelity correction", "cheap result + GP-learned difference to the expensive one"],
            ["seed", "the number that fixes the random choices (data split, initial weights)"]]
    s += table(rows, [5 * cm, 11.5 * cm])
    s += [PageBreak()]
    return s


def part12():
    s = [H1("12. Formula sheet")]
    s += [P("<b>Waves</b>")] + eq(r"c=f\lambda,\quad \lambda_{\rm material}=\lambda/n,\quad \varepsilon=n^2,\quad "
                                  r"f[{\rm GHz}]=u\cdot\frac{c}{a}=u\cdot1236\ (a=242.5\,\mu m)")
    s += [P("<b>Band edges</b>")] + eq(r"f_{\rm lower}=\max_{\mathbf{k}}f_1,\quad f_{\rm upper}=\min_{\mathbf{k}}f_2,\quad "
                                       r"\Delta f=f_{\rm upper}-f_{\rm lower}")
    s += [P("<b>Fourier series</b>")] + eq(r"g(x)=\sum_m c_m e^{i2\pi mx/a},\quad c_m=\frac{1}{a}\int_0^a g\,e^{-i2\pi mx/a}dx,"
                                           r"\quad c_{\pm m}=\frac{a_m}{2}")
    s += [P("<b>Hole shape (C3)</b>")] + eq(r"r(\theta)=r_0+\sum_{n=3,6,9,12}\left[c_n\cos n\theta+s_n\sin n\theta\right]")
    s += [P("<b>Layer coefficients</b>")] + eq(r"a_0=\varepsilon_{\rm air}+(\varepsilon_{\rm Si}-\varepsilon_{\rm air})F,\quad "
                                               r"a_m=2(\varepsilon_{\rm Si}-\varepsilon_{\rm air})\frac{\sin\pi mF}{\pi m}")
    s += [P("<b>PWE (1D and 2D, TE)</b>")] + eq(r"\sum_{\mathbf{G}'}(\mathbf{k}+\mathbf{G})\cdot(\mathbf{k}+\mathbf{G}')"
                                                r"\left[\varepsilon^{-1}\right]_{\mathbf{G}\mathbf{G}'}h_{\mathbf{G}'}"
                                                r"=\frac{\omega^2}{c^2}h_{\mathbf{G}},\quad [\varepsilon^{-1}]=[\varepsilon_{\mathbf{G}-\mathbf{G}'}]^{-1}")
    s += [P("<b>Reciprocal vectors</b>")] + eq(r"\mathbf{b}_1=\frac{2\pi}{a}(1,-1/\sqrt{3}),\ \mathbf{b}_2=\frac{2\pi}{a}(0,2/\sqrt{3}),"
                                               r"\ \mathbf{a}_i\cdot\mathbf{b}_j=2\pi\delta_{ij}")
    s += [P("<b>Gradient descent, MSE</b>")] + eq(r"w\leftarrow w-\eta\,\frac{\partial\mathcal{L}}{\partial w},\qquad "
                                                  r"\mathcal{L}=\frac{1}{N}\sum_i(\hat y_i-y_i)^2")
    s += [P("<b>Tandem and CVAE</b>")] + eq(r"\mathcal{L}_{\rm tandem}=|F(G(\mathbf{t}))-\mathbf{t}|^2+\lambda P,\qquad "
                                            r"\mathcal{L}_{\rm CVAE}=|\hat{\mathbf{x}}-\mathbf{x}|^2+\beta\,\frac{1}{2}"
                                            r"\sum(\mu^2+\sigma^2-1-\ln\sigma^2)")
    s += [P("<b>Multi-fidelity</b>")] + eq(r"f_{\rm COMSOL}\approx f_{\rm PWE}+\delta,\qquad \delta\sim\mathcal{GP}(0,k)")
    s += [PageBreak()]
    return s


def part13():
    s = [H1("13. Answers to the check-yourself questions")]
    cur = None
    for part, q, a in ANSWERS:
        if part != cur:
            s += [H3(f"Part {part}")]
            cur = part
        s += [Paragraph(f"<b>Q.</b> {q}", S["p"]),
              Paragraph(f"<i>A.</i> {a}", ParagraphStyle("ans", parent=S["p"], leftIndent=14, textColor=MUTED))]
    s += [H2("Where to read more")]
    s += bullets([
        "J. D. Joannopoulos, S. G. Johnson, J. N. Winn, R. D. Meade, <i>Photonic Crystals: Molding the Flow of Light</i>, "
        "2nd ed. (2008), free at ab-initio.mit.edu/book — chapters 2–5 (waves, symmetry, band gaps) and appendix D "
        "(numerical methods including plane waves).",
        "K. M. Ho, C. T. Chan, C. M. Soukoulis, Phys. Rev. Lett. 65, 3152 (1990) — the inverse-matrix rule.",
        "D. Liu et al., ACS Photonics 5, 1365 (2018) — the tandem network.",
        "K. Sohn, H. Lee, X. Yan, NeurIPS (2015) — conditional VAE.",
        "C. E. Rasmussen, C. K. I. Williams, <i>Gaussian Processes for Machine Learning</i> (2006), free at "
        "gaussianprocess.org/gpml — chapter 2.",
        "M. C. Kennedy, A. O'Hagan, Biometrika 87, 1 (2000) — multi-fidelity modelling.",
        "The option-B study guide (docs/OptionB_Study_Guide.pdf) — the project's results and the COMSOL side."])
    return s


def story():
    res = results()
    s = []
    s += part0() + part1() + part2() + part3() + part4() + part5() + part6() + part7() + part8(res) + part9(res)
    s += part10(res) + part11() + part12() + part13()
    return s


def main():
    ANSWERS.clear()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc = bg.Doc(OUT, pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=1.9 * cm, bottomMargin=1.8 * cm,
                 title="Learning the project from zero", author="Binit Shakya (prepared with Claude)")
    doc.multiBuild(bg.flatten(story()), onFirstPage=on_page, onLaterPages=on_page)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
