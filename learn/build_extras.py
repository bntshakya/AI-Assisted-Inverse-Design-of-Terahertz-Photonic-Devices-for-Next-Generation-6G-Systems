"""Viva / interview material -> study website data + PDFs.

Run from the Project folder (after src/shapes/ablation_architecture.py and make_arch_figures.py):
    python learn/build_extras.py
Writes:
  learn/site/web_data.json   key "extras" (Q&A, why-choices, metrics, formulas, interview guide, viva day, ablation)
  learn/site/arch_*.png, ablation.png
  docs/Viva_QA.pdf, docs/Architecture_and_Choices.pdf, docs/Interview_Guide.pdf, docs/Formula_Sheet.pdf,
  docs/Viva_Card.pdf, docs/Presenter_Script.pdf
"""
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "shapes"))
import viva_content as V  # noqa: E402

SITE = "learn/site/web_data.json"
FIGS = "results/optionb/figs"
ABL = "results/optionb/ablation.json"
DECK = "../Report/final/254161023_MTP1_Presentation.pptx"


# ------------------------------------------------------------------ architecture-test text
def f2(x):
    return f"{x:.2f}"


def ablation_text(ab):
    P = {}
    A = {(r["depth"], r["width"]): r for r in ab.get("A_depth_width", [])}
    if (3, 256) in A and len(A) >= 12:
        ref = A[(3, 256)]
        d = {k: A[(k, 256)]["mae_mean"] for k in range(1, 6) if (k, 256) in A}
        best_key = min(A, key=lambda k: A[k]["mae_mean"])
        best = A[best_key]
        noise = max(ref["mae_seed_std"], best["mae_seed_std"], 0.005)
        diff = ref["mae_mean"] - best["mae_mean"]
        lst = ", ".join(f"{k} layer{'s' if k > 1 else ''} {f2(v)}" for k, v in d.items())
        P["why_depth"] = (f"I tested it on the validation split (single networks, 3 random seeds each, scored in GHz, mean of "
                          f"both edges). With 256 neurons per layer: {lst} GHz. ")
        if best_key == (3, 256) or diff <= 2 * noise:
            P["why_depth"] += (f"Going from 1 to 3 layers cuts the error by {100 * (1 - d[3] / d[1]):.0f} %; bigger networks "
                               f"(best: {best_key[0]} x {best_key[1]}, {f2(best['mae_mean'])} GHz) are no better than "
                               f"3 x 256 within the seed-to-seed noise (about {f2(noise)} GHz). So 3 x 256 is the smallest "
                               "size that is as good as the bigger ones.")
            P["why_depth_short"] = (f"Validation test: 1 layer {f2(d[1])} GHz, 3 layers {f2(d[3])} GHz; deeper or wider "
                                    "networks were no better within the seed noise.")
        else:
            P["why_depth"] += (f"Going from 1 to 3 layers cuts the error by {100 * (1 - d[3] / d[1]):.0f} %. The best network "
                               f"in the test was {best_key[0]} x {best_key[1]} at {f2(best['mae_mean'])} GHz, "
                               f"{f2(diff)} GHz better than 3 x 256 ({f2(ref['mae_mean'])} GHz) with "
                               f"{best['params'] / ref['params']:.1f}x the weights. 3 x 256 was fixed as a standard size "
                               "before this test, and all reported results use it. The gain would not change any "
                               "conclusion: the forward error (0.13 / 0.17 GHz for the ensemble) is already far below the "
                               "1 GHz design criterion and below the PWE-COMSOL difference, which dominates the final "
                               "error. Using a larger network is an easy, free improvement for Semester IV.")
            P["why_depth_short"] = (f"Validation test: 1 layer {f2(d[1])} GHz, 3 layers {f2(d[3])} GHz; larger sizes "
                                    f"(best {best_key[0]} x {best_key[1]}) gain {f2(diff)} GHz, which changes no conclusion; "
                                    "an easy upgrade later.")
        w = {k: A[(3, k)]["mae_mean"] for k in (64, 128, 256, 512) if (3, k) in A}
        P["why_depth_push"] = ("Width at 3 layers: " + ", ".join(f"{k} neurons {f2(v)}" for k, v in w.items()) + " GHz. ")
        if (1, 512) in A and (3, 64) in A:
            P["why_depth_push"] += (f"Depth matters more than width: one layer of 512 ({f2(A[(1, 512)]['mae_mean'])} GHz, "
                                    f"{A[(1, 512)]['params']:,} weights) is worse than three layers of 64 "
                                    f"({f2(A[(3, 64)]['mae_mean'])} GHz, {A[(3, 64)]['params']:,} weights).")
        unstable = [k for k, r in A.items() if r["mae_seed_std"] >= 0.1]
        if unstable:
            k = unstable[0]
            P["why_depth_push"] += (f" Bigger is not free: {k[0]} x {k[1]} ({A[k]['params']:,} weights) failed to train on one of "
                                    f"its three seeds (error {f2(A[k]['mae_mean'])} &#177; {f2(A[k]['mae_seed_std'])} GHz); very large "
                                    "networks need a smaller learning rate or more care.")
        P["why_tandem_short"] = ("Same building block as the forward network (3 x 256, SiLU), so the inverse network has "
                                 "about the same capacity as the function it inverts.")
    B = {r["activation"]: r["mae_mean"] for r in ab.get("B_activation", [])}
    if len(B) == 4:
        order = sorted(B, key=B.get)
        P["why_act"] = ("Tested at 3 x 256 on the validation split (3 seeds): " +
                        ", ".join(f"{k} {f2(B[k])}" for k in order) + " GHz. ")
        if order[0] in ("SiLU", "GELU") and B["SiLU"] - B[order[0]] <= 0.01:
            P["why_act"] += ("Smooth activations (SiLU, GELU) are the most accurate here; SiLU is as good as the best and "
                             "cheaper than GELU.")
        elif B["SiLU"] <= B["ReLU"]:
            P["why_act"] += (f"SiLU beats ReLU by {f2(B['ReLU'] - B['SiLU'])} GHz; the band edges change smoothly with the "
                             "shape, and a smooth activation represents that better.")
        else:
            P["why_act"] += (f"{order[0]} came out {f2(B['SiLU'] - B[order[0]])} GHz better than SiLU here; SiLU was chosen "
                             "for its smooth gradients, which the inverse methods rely on.")
        P["why_act_short"] = "Validation test: " + ", ".join(f"{k} {f2(B[k])}" for k in order) + " GHz; smooth, so clean gradients for inverse design."
    C = {r["members"]: r for r in ab.get("C_ensemble", [])}
    if 1 in C and 5 in C:
        k = {m: C[m]["mae_mean"] for m in C}
        g5 = 100 * (1 - k[5] / k[1])
        P["why_ens"] = ("Measured on the validation split: " + ", ".join(f"{m} network{'s' if m > 1 else ''} {f2(v)}"
                                                                         for m, v in k.items()) +
                        f" GHz. Averaging cancels part of each network's random error: 5 networks are {g5:.0f} % more "
                        "accurate than one, and going to 10 doubles the cost for "
                        + (f"only {f2(k[5] - k[10])} GHz more" if k.get(10) and k[5] - k[10] >= 0.005 else "almost nothing (under 0.01 GHz)")
                        + ". The spread between the 5 also serves as an uncertainty estimate.")
        if C[5].get("corr_spread_error") is not None:
            c5 = C[5]
            P["why_spread"] = (f"Where the 5 networks disagree, the prediction is more likely to be wrong: on the validation "
                               f"cells the 10 % with the largest spread have an average error of {f2(c5['err_top10pct_spread'])} "
                               f"GHz against {f2(c5['err_rest'])} GHz for the rest (correlation {c5['corr_spread_error']:.2f}). "
                               "So the spread flags risky predictions; it is shown in the app as the ensemble spread.")
            P["why_ens_push"] = (f"The 10 % of cells with the largest spread have {c5['err_top10pct_spread'] / max(c5['err_rest'], 1e-9):.1f}x "
                                 "the average error of the rest, so the spread is a useful warning light.")
        P["why_ens_short"] = (f"Validation test: 1 network {f2(k[1])} GHz, 5 networks {f2(k[5])} GHz; the spread flags risky "
                              "predictions; 10 would double the cost for little gain.")
    D = {r["epochs"]: r for r in ab.get("D_epochs", [])}
    if len(D) == 4:
        e = {m: D[m]["mae_mean"] for m in D}
        P["why_epochs"] = ("Validation error by training length: " + ", ".join(f"{m} epochs {f2(v)}" for m, v in e.items()) +
                           f" GHz (about {D[400]['train_s']:.0f} s vs {D[800]['train_s']:.0f} s per network for 400 and 800). ")
        if e[400] - e[800] <= 0.01:
            P["why_epochs"] += "400 is where the curve flattens: doubling to 800 adds almost nothing."
        else:
            P["why_epochs"] += (f"So 400 epochs was not fully converged: 800 epochs is {f2(e[400] - e[800])} GHz better "
                                f"({100 * (1 - e[800] / e[400]):.0f} %). 400 was a time trade-off (the whole pipeline is retrained "
                                "after every new COMSOL export). Like the network size, this changes no conclusion: the "
                                "forward error is already far below the 1 GHz criterion and below the PWE-COMSOL difference. "
                                "Training longer is an easy improvement for Semester IV.")
        P["why_epochs_short"] = ("Validation test: " + ", ".join(f"{m}: {f2(v)}" for m, v in e.items()) + " GHz; " +
                                 ("400 is where the curve flattens." if e[400] - e[800] <= 0.01 else
                                  "800 would be better: 400 was a speed trade-off (an easy upgrade)."))
    E = ab.get("E_tandem", [])
    lam = {r["penalty_weight"]: r for r in E if r["depth"] == 3 and r["width"] == 256}
    if len(lam) >= 3:
        pct = lambda x: f"{100 * x:.1f} %" if abs(100 * x - round(100 * x)) > 0.01 else f"{100 * x:.0f} %"
        P["why_lambda"] = ("The penalty pushes designs away from thin walls and tiny radii while training. Tested on 200 "
                           "validation targets (designs checked with the exact rules, edges by the forward network): " +
                           "; ".join(f"&#955; = {l:g}: {pct(lam[l]['valid_share'])} of designs fabricable, "
                                     f"{pct(lam[l]['within_1GHz_and_valid'])} fabricable and within 1 GHz"
                                     for l in sorted(lam)) + ". ")
        top = max(r["within_1GHz_and_valid"] for r in lam.values())
        winners = sorted(l for l in lam if lam[l]["within_1GHz_and_valid"] >= top - 1e-9)
        if winners == [1e-3]:
            P["why_lambda"] += "&#955; = 0.001 gives the most usable designs."
        elif 1e-3 in winners:
            others = [l for l in winners if l != 1e-3]
            P["why_lambda"] += (f"&#955; = 0.001 and {', '.join(f'{l:g}' for l in others)} are equally good on this measure; "
                                f"without the penalty only {pct(lam[0.0]['valid_share']) if 0.0 in lam else 'fewer'} of designs "
                                "can be fabricated. The smaller weight disturbs the band-edge accuracy least "
                                f"(worst-edge error {lam[1e-3]['worst_edge_mean']:.3f} GHz"
                                + (f" vs {lam[others[0]]['worst_edge_mean']:.3f}" if others else "") + "), so 0.001 was kept.")
        else:
            P["why_lambda"] += (f"Here &#955; = {winners[0]:g} gave the most usable designs; 0.001 is close, and the exact "
                                "fabrication check afterwards removes any invalid design anyway.")
        P["why_lambda_short"] = ("Validation test: " + ", ".join(f"&#955; {l:g} -> {pct(lam[l]['within_1GHz_and_valid'])} usable"
                                                                for l in sorted(lam)) + ".")
        sizes = [r for r in E if r["penalty_weight"] == 1e-3]
        if sizes:
            P["why_tandem_short"] = ("Validation test (share of targets with a fabricable design within 1 GHz): " +
                                     ", ".join(f"{r['depth']} x {r['width']}: {pct(r['within_1GHz_and_valid'])}" for r in sizes)
                                     + ".")
    F = ab.get("F_cvae", [])
    if len(F) >= 5:
        row = lambda lt, bt: next(r for r in F if r["latent"] == lt and abs(r["beta"] - bt) < 1e-9)
        lat = [row(l, 0.5) for l in (2, 4, 8)]
        bet = [row(4, b) for b in (0.1, 0.5, 1.0)]
        P["why_cvae"] = ("Tested on 100 validation targets (64 samples each, before refining; scored by the forward network and the exact "
                         "fabrication rules). Latent size: " +
                         "; ".join(f"{r['latent']} numbers: {r['distinct_within_1GHz_of_8']:.1f} distinct designs within "
                                   f"1 GHz (of 8), {100 * r['samples_within_1GHz']:.0f} % of samples within 1 GHz" for r in lat) +
                         ". &#946;: " + "; ".join(f"{r['beta']:g}: {r['distinct_within_1GHz_of_8']:.1f} designs, "
                                                  f"{100 * r['samples_within_1GHz']:.0f} % of samples" for r in bet) + ". ")
        dist = lambda r: r["distinct_within_1GHz_of_8"]
        best_lat = max(lat, key=dist)
        if best_lat["latent"] > 4 and dist(best_lat) - dist(lat[1]) > 0.5:
            P["why_cvae"] += (f"A larger code helps: latent size {best_lat['latent']} gives {dist(best_lat):.1f} designs against "
                              f"{dist(lat[1]):.1f} for the 4 used. That fits the geometry of the problem: with the 2 band edges "
                              "fixed, the 10 design numbers leave 8 free directions, and a 4-number code cannot cover them all. "
                              "4 was a standard small choice made early; the refine step makes up for most of the difference "
                              "(CVAE + refine already gives about 8 designs per target at 97 %), so a larger latent code is an "
                              "easy upgrade for Semester IV. ")
        else:
            P["why_cvae"] += ("The latent size needs to be big enough to hold the free directions left once the 2 band edges "
                              "are fixed, but each extra dimension also makes random sampling less precise. ")
        best_b = max(bet, key=dist)
        if best_b["beta"] == 0.5:
            P["why_cvae"] += ("&#946; = 0.5 is the best of the three: a small &#946; lets the codes drift away from the normal "
                              "distribution, so random codes decode to poorer designs; a large &#946; squeezes them together "
                              "and the designs lose variety.")
        else:
            P["why_cvae"] += (f"For &#946;, {best_b['beta']:g} came out best ({dist(best_b):.1f} designs vs {dist(bet[1]):.1f} at the "
                              "0.5 used): &#946; trades how closely the codes follow the normal distribution (needed so that "
                              "random codes decode to valid designs) against how much detail they keep.")
        P["why_latent_short"] = ("Validation test: " + ", ".join(f"{r['latent']} -> {dist(r):.1f} designs" for r in lat) +
                                 " within 1 GHz per target (before refining)" +
                                 ("; a larger code would help, refine makes up for most of it." if best_lat["latent"] > 4 else "."))
        P["why_beta_short"] = ("Validation test: " + ", ".join(f"&#946; {r['beta']:g} -> {r['distinct_within_1GHz_of_8']:.1f} designs"
                                                               for r in bet) + " (before refining); trades accuracy of samples against variety.")
    return P


# ------------------------------------------------------------------ text conversion
def uni(t):
    """Numeric HTML entities -> real characters (searchable text); &lt; &gt; &amp; stay as they are."""
    return re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), t)


def to_html(t):
    t = uni(t)
    t = re.sub(r"\bf_(lower|upper)\b", r"f<sub>\1</sub>", t)
    return t.replace("<super>", "<sup>").replace("</super>", "</sup>")


def to_pdf(t):
    t = uni(t)
    t = re.sub(r"\bf_(lower|upper)\b", r"f<sub>\1</sub>", t)
    return t.replace("<sup>", "<super>").replace("</sup>", "</super>")


def build_extras(P):
    fill = lambda t: V.fill(t, P) if t else t
    pending = "This answer is filled in from the architecture test (results/optionb/ablation.json)."
    qa = []
    for topic, lvl, q, a, push in V.QA:
        a2, p2 = fill(a), fill(push)
        if "{" in a2:
            a2 = pending
        if p2 and "{" in p2:
            p2 = ""
        qa.append({"t": topic, "lvl": lvl, "q": to_html(q), "a": to_html(a2), "p": to_html(p2)})
    why = [{"g": g, "s": s, "v": to_html(v), "w": to_html(fill(w) if "{" not in fill(w) else "see the architecture test")}
           for g, s, v, w in V.WHY_ITEMS]
    iv = V.INTERVIEW
    return {
        "topics": V.TOPICS, "qa": qa, "why": why,
        "metrics": [{"m": m, "what": to_html(w), "why": to_html(y), "ex": to_html(e)} for m, w, y, e in V.METRICS_ALT],
        "formulas": [{"n": n, "f": to_html(f), "r": to_html(r)} for n, f, r in V.FORMULAS],
        "interview": {"pitch30": iv["pitch30"], "pitch2": [to_html(x) for x in iv["pitch2"]],
                      "deep5": iv["deep5"], "stories": iv["stories"], "bullets": iv["bullets"], "skills": iv["skills"],
                      "questions": iv["questions"], "roles": iv["roles"], "linkedin": iv["linkedin"], "github_line": iv["github_line"]},
        "vivaday": V.VIVA_DAY,
        "why_text": {k: to_html(v) for k, v in P.items()},
        "arch": {"forward": 136962, "forward5": 684810, "tandem": 134922, "enc": 136968, "dec": 135946},
    }


# ------------------------------------------------------------------ PDFs
def pdfs(P, ab):
    from build_guide import A4, S, W, cm, colors  # registers the Calibri fonts  # noqa: F401
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)
    from PIL import Image as PI

    TEAL, INK, MUTED, LIGHT = colors.HexColor("#0d7a84"), colors.HexColor("#16202a"), colors.HexColor("#566271"), colors.HexColor("#e6f0f1")
    st = {
        "title": ParagraphStyle("t", fontName="Body-Bold", fontSize=22, leading=27, textColor=colors.HexColor("#0b3c49")),
        "sub": ParagraphStyle("s", fontName="Body", fontSize=11.5, leading=15, textColor=MUTED),
        "h1": ParagraphStyle("h1", fontName="Body-Bold", fontSize=15, leading=19, textColor=TEAL, spaceBefore=10, spaceAfter=6),
        "h2": ParagraphStyle("h2", fontName="Body-Bold", fontSize=12, leading=15, textColor=INK, spaceBefore=6, spaceAfter=3),
        "q": ParagraphStyle("q", fontName="Body-Bold", fontSize=11, leading=14, textColor=colors.HexColor("#0b3c49"), spaceBefore=7),
        "a": ParagraphStyle("a", fontName="Body", fontSize=10.3, leading=13.8, textColor=INK, spaceBefore=2),
        "push": ParagraphStyle("p", fontName="Body-Italic", fontSize=9.6, leading=12.6, textColor=MUTED, leftIndent=10, spaceBefore=2),
        "p": ParagraphStyle("pp", fontName="Body", fontSize=10.3, leading=14, textColor=INK, spaceAfter=4),
        "bul": ParagraphStyle("b", fontName="Body", fontSize=10.3, leading=13.8, textColor=INK, leftIndent=13, bulletIndent=3, spaceAfter=2),
        "cell": ParagraphStyle("c", fontName="Body", fontSize=9.3, leading=11.8, textColor=INK),
        "cellb": ParagraphStyle("cb", fontName="Body-Bold", fontSize=9.3, leading=11.8, textColor=colors.white),
        "big": ParagraphStyle("big", fontName="Body-Bold", fontSize=13, leading=16, textColor=TEAL, alignment=TA_CENTER),
        "cap": ParagraphStyle("cap", fontName="Body-Italic", fontSize=9, leading=11.5, textColor=MUTED, spaceAfter=6),
    }

    def tbl(rows, widths, header=True, font=None):
        data = [[Paragraph(to_pdf(str(c)), st["cellb"] if header and i == 0 else st["cell"]) for c in r] for i, r in enumerate(rows)]
        t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        s = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d3d6")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
        if header:
            s.append(("BACKGROUND", (0, 0), (-1, 0), TEAL))
        for i in range(1 if header else 0, len(rows)):
            if i % 2 == 0:
                s.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f2f7f8")))
        t.setStyle(TableStyle(s))
        return t

    def callout(title, text, bg=LIGHT):
        inner = [Paragraph(f"<b>{to_pdf(title)}</b>", st["a"])] + [Paragraph(to_pdf(x), st["a"]) for x in (text if isinstance(text, list) else [text])]
        t = Table([[inner]], colWidths=[W])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LEFTPADDING", (0, 0), (-1, -1), 9),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 6),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
        return [Spacer(1, 4), t, Spacer(1, 6)]

    def image(path, width):
        w, h = PI.open(path).size
        return Image(path, width=width, height=width * h / w)

    def footer(label):
        def f(c, doc):
            c.saveState()
            c.setFont("Body", 8.5)
            c.setFillColor(MUTED)
            c.drawString(2.0 * cm, 1.1 * cm, label)
            c.drawRightString(A4[0] - 2.0 * cm, 1.1 * cm, str(doc.page))
            c.restoreState()
        return f

    def build(path, story, label):
        doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=2.0 * cm, rightMargin=2.0 * cm, topMargin=1.8 * cm,
                                bottomMargin=1.8 * cm, title=label, author="Binit Shakya")
        doc.build(story, onFirstPage=footer(label), onLaterPages=footer(label))
        print("wrote", path)

    fill = lambda t: V.fill(t, P) if t else t
    topics = dict(V.TOPICS)
    # ---- Q&A
    s = [Paragraph("Viva questions and answers", st["title"]), Spacer(1, 4),
         Paragraph(f"{len(V.QA)} likely questions with short answers you can say out loud. Read the <b>MUST</b> ones first "
                   "(about half), say each answer aloud before reading it, and use the 'if they push' line for follow-ups. "
                   "All numbers are the final ones (4,198 COMSOL cells).", st["sub"]), Spacer(1, 6)]
    s += callout("How to answer any question", ["1. Answer the question in one sentence first. 2. Give one number as evidence. "
                                                "3. Stop, and let them ask more. If you do not know: say what you do know and "
                                                "how you would check the rest; never guess a number."])
    for key, name in V.TOPICS:
        items = [x for x in V.QA if x[0] == key]
        s.append(Paragraph(name, st["h1"]))
        for topic, lvl, q, a, push in items:
            a2, p2 = fill(a), fill(push)
            tag = '<font color="#c8553d">MUST</font>' if lvl == "must" else '<font color="#8a949c">nice</font>'
            block = [Paragraph(f"{tag} &#160; {to_pdf(q)}", st["q"]),
                     Paragraph(to_pdf(a2 if "{" not in a2 else "(filled in from the architecture test)"), st["a"])]
            if p2 and "{" not in p2:
                block.append(Paragraph("If they push: " + to_pdf(p2), st["push"]))
            s.append(KeepTogether(block))
    build("docs/Viva_QA.pdf", s, "Viva questions and answers  ·  Binit Shakya  ·  M.Tech. Phase I")

    # ---- architecture and choices
    s = [Paragraph("The networks and why they look like this", st["title"]), Spacer(1, 4),
         Paragraph("Diagrams of every network, the reason behind each setting, and the architecture test that backs the "
                   "choices up. The test trained on the training split and scored on the validation split; the test split "
                   "behind the reported results was never used.", st["sub"]), Spacer(1, 8)]
    for fn, title, cap in (("arch_tandem.png", "1. The tandem network", "The inverse network (left, trained) proposes a design; the frozen forward ensemble (right) predicts its band edges; the loss compares them with the target. Only the inverse network learns."),
                           ("arch_forward.png", "2. The forward ensemble", "Five identical networks from different random starts; their average is the prediction and their spread the uncertainty."),
                           ("arch_cvae.png", "3. The conditional VAE (+ refine)", "The encoder is only needed for training; to design, random codes z go through the decoder, and each design is polished by gradient steps.")):
        s += [Paragraph(title, st["h1"]), image(os.path.join(FIGS, fn), W), Paragraph(cap, st["cap"])]
    s.append(PageBreak())
    s.append(Paragraph("Every setting and why", st["h1"]))
    s.append(tbl([["Part", "Setting", "Value", "Why"]] + [[g, n, v, fill(w) if "{" not in fill(w) else "see the test below"]
                                                          for g, n, v, w in V.WHY_ITEMS], [2.4 * cm, 3.0 * cm, 3.3 * cm, W - 8.7 * cm]))
    s.append(Paragraph("The architecture test", st["h1"]))
    for key, title in (("why_depth", "Depth and width"), ("why_act", "Activation"), ("why_ens", "Ensemble size"),
                       ("why_spread", "Does the spread mean anything?"), ("why_epochs", "Training length"),
                       ("why_lambda", "Tandem fabrication penalty"), ("why_cvae", "CVAE latent size and beta")):
        if key in P:
            s += [Paragraph(title, st["h2"]), Paragraph(to_pdf(P[key]), st["p"])]
    if os.path.exists(os.path.join(FIGS, "ablation.png")):
        s += [image(os.path.join(FIGS, "ablation.png"), W),
              Paragraph("Validation error (mean of both edges) of single networks: (a) by depth and width, (b) by number of "
                        "networks averaged, (c) by training length. Error bars: spread over 3 seeds.", st["cap"])]
    A = ab.get("A_depth_width", [])
    if A:
        s.append(Paragraph("All depth / width results", st["h2"]))
        s.append(tbl([["Hidden layers", "Width", "Weights", "Validation MAE f_lower / f_upper (GHz)", "Mean", "Seed spread"]] +
                     [[r["depth"], r["width"], f"{r['params']:,}", f"{r['mae'][0]:.3f} / {r['mae'][1]:.3f}",
                       f"{r['mae_mean']:.3f}", f"{r['mae_seed_std']:.3f}"] for r in A],
                     [2.4 * cm, 1.6 * cm, 2.2 * cm, 5.0 * cm, 2.0 * cm, 2.3 * cm]))
    build("docs/Architecture_and_Choices.pdf", s, "Networks and choices  ·  Binit Shakya  ·  M.Tech. Phase I")

    # ---- interview guide
    iv = V.INTERVIEW
    s = [Paragraph("Explaining the project in a job interview", st["title"]), Spacer(1, 4),
         Paragraph("Ready-made versions of the story for different lengths and audiences, the interview stories it gives you, "
                   "and resume lines.", st["sub"]), Spacer(1, 6)]
    s += callout("The 30-second version", iv["pitch30"])
    s.append(Paragraph("The 2-minute version", st["h1"]))
    s += [Paragraph(to_pdf(x), st["p"]) for x in iv["pitch2"]]
    s.append(Paragraph("The 5-minute technical walk-through", st["h1"]))
    s.append(tbl([["Part", "What to say"]] + [[a, b] for a, b in iv["deep5"]], [3.2 * cm, W - 3.2 * cm]))
    for k, (title, sit, task, act, res) in enumerate(iv["stories"]):
        head = [Paragraph("Interview stories (Situation, Task, Action, Result)", st["h1"])] if k == 0 else []
        s.append(KeepTogether(head + [Paragraph(title, st["h2"]),
                                      tbl([["Situation", sit], ["Task", task], ["Action", act], ["Result", res]],
                                          [2.4 * cm, W - 2.4 * cm], header=False)]))
    s.append(Paragraph("Resume bullet points", st["h1"]))
    s += [Paragraph(to_pdf(b), st["bul"], bulletText="•") for b in iv["bullets"]]
    s.append(Paragraph("Skills this project shows", st["h1"]))
    s.append(tbl([["Area", "Evidence"]] + [list(x) for x in iv["skills"]], [3.6 * cm, W - 3.6 * cm]))
    s.append(Paragraph("Common interview questions, answered with this project", st["h1"]))
    s.append(tbl([["Question", "Answer"]] + [list(x) for x in iv["questions"]], [5.0 * cm, W - 5.0 * cm]))
    s.append(Paragraph("Portfolio / LinkedIn post", st["h1"]))
    s += callout("Ready to post (edit to taste)", iv["linkedin"])
    s += callout("One-line project description (GitHub, resume header)", iv["github_line"])
    s.append(Paragraph("What to lead with, by role", st["h1"]))
    s.append(tbl([["Role", "Lead with"]] + [list(x) for x in iv["roles"]], [3.8 * cm, W - 3.8 * cm]))
    build("docs/Interview_Guide.pdf", s, "Interview guide  ·  Binit Shakya")

    # ---- formula sheet
    s = [Paragraph("Formula sheet", st["title"]), Spacer(1, 4),
         Paragraph("The equations an examiner might ask you to write, each with a plain-English reading.", st["sub"]), Spacer(1, 8)]
    s.append(tbl([["What", "Formula", "Read it as"]] + [[n, f, r] for n, f, r in V.FORMULAS],
                 [3.2 * cm, 6.8 * cm, W - 10.0 * cm]))
    build("docs/Formula_Sheet.pdf", s, "Formula sheet  ·  Binit Shakya  ·  M.Tech. Phase I")

    # ---- one-page viva card
    nums = [("4,198", "COMSOL cells (1,064 rounded, 1,064 sharp, 1,049 circles, 1,007 freeform, 14 AI designs)"),
            ("0.79 / 0.34 GHz", "PWE vs COMSOL (f_lower / f_upper), 3,267 gapped cells"),
            ("0.14 / 0.23 GHz", "after the learned COMSOL correction (10-fold CV)"),
            ("0.13 / 0.17 GHz", "forward network on 600 unseen PWE cells"),
            ("99 %  /  97 %, ~8", "tandem  /  CVAE + refine: targets within 1 GHz, designs per target"),
            ("12 of 14", "AI designs within 1 GHz in COMSOL (worst 1.05 GHz)"),
            ("0.25 vs 0.27 GHz", "freeform: PWE + correction vs COMSOL-only network"),
            ("1 GHz", "success threshold: 0.3 % of 300 GHz, < 1/2 of a 2.16 GHz channel")]
    s = [Paragraph("Viva card: read this just before you go in", st["title"]), Spacer(1, 6)]
    s += callout("Your opening (30 seconds)", V.VIVA_DAY[0][1])
    s.append(tbl([["Number", "What it is"]] + [list(x) for x in nums], [4.2 * cm, W - 4.2 * cm]))
    s.append(Spacer(1, 6))
    s.append(tbl([["The networks", ""],
                  ["Forward", "5 x MLP 18 -> 256 -> 256 -> 256 -> 2, SiLU; average = prediction, spread = uncertainty"],
                  ["Tandem", "inverse MLP 2 -> 256 x 3 -> 10 trained through the frozen forward net; one design, 99 %"],
                  ["CVAE + refine", "latent 4, &#946; 0.5; 64 samples + 100 gradient steps; ~8 designs, 97 %, 0.2 s"],
                  ["Correction", "Gaussian process learns COMSOL - PWE; prediction = PWE + &#916;"],
                  ["Accuracy", "GHz on unseen data; designs re-simulated; worse edge < 1 GHz counts"],
                  ["Why these sizes?", "Tested on the validation split: depth and SiLU matter most (ReLU 3x worse); a bigger "
                                       "network or longer training gains a few hundredths of a GHz, which changes no conclusion"]],
                 [3.0 * cm, W - 3.0 * cm]))
    s += callout("If you get stuck", [V.VIVA_DAY[2][1], V.VIVA_DAY[3][1]], bg=colors.HexColor("#fdf3e3"))
    build("docs/Viva_Card.pdf", s, "Viva card  ·  Binit Shakya")

    # ---- presenter script from the deck's speaker notes
    if os.path.exists(DECK):
        from pptx import Presentation
        prs = Presentation(DECK)
        s = [Paragraph("Presenter script", st["title"]), Spacer(1, 4),
             Paragraph("What to say on each slide (the speaker notes of the presentation). About 40 seconds per slide; "
                       "practise it out loud twice.", st["sub"]), Spacer(1, 6)]
        for i, sl in enumerate(prs.slides, 1):
            title = next((sh.text_frame.text for sh in sl.shapes if sh.is_placeholder and sh.has_text_frame and sh.text_frame.text.strip()), f"Slide {i}")
            note = sl.notes_slide.notes_text_frame.text if sl.has_notes_slide else ""
            s.append(KeepTogether([Paragraph(f"{i}. {to_pdf(title)}", st["q"]), Paragraph(to_pdf(note), st["a"])]))
        build("docs/Presenter_Script.pdf", s, "Presenter script  ·  Binit Shakya")


def pct_tex(x):
    v = 100 * x
    return (f"{v:.1f}" if abs(v - round(v)) > 0.01 else f"{v:.0f}") + "\\%"


def write_ablation_tex(ab, path="../Report/final/ablation.tex"):
    """Appendix table of the architecture test (validation split) for the report."""
    A = ab.get("A_depth_width", [])
    if not A:
        return
    widths = sorted({r["width"] for r in A})
    depths = sorted({r["depth"] for r in A})
    cell = {(r["depth"], r["width"]): r for r in A}
    rows = []
    for d in depths:
        vals = []
        for w in widths:
            r = cell.get((d, w))
            if not r:
                vals.append("")
                continue
            v = f"{r['mae_mean']:.2f}" + ("$^\\ast$" if r["mae_seed_std"] >= 0.1 else "")
            vals.append(f"\\textbf{{{v}}}" if (d, w) == (3, 256) else v)
        rows.append(f"{d} & " + " & ".join(vals) + " \\\\")
    lines = [r"\begin{table}[htbp]",
             r"  \caption{Architecture test on the validation split (single networks, 3 seeds; validation MAE in GHz, mean of "
             r"both edges). Bold: the network used; $^\ast$: unstable (one seed failed to train). The test split was not used.}",
             r"  \label{tab:abl}", r"  \centering", r"  \small",
             r"  \begin{tabular}{l" + "c" * len(widths) + "}", r"    \toprule",
             "    Hidden layers & " + " & ".join(f"width {w}" for w in widths) + r" \\", r"    \midrule"]
    lines += ["    " + r for r in rows]
    lines += [r"    \bottomrule", r"  \end{tabular}"]
    extra = []
    B = {r["activation"]: r["mae_mean"] for r in ab.get("B_activation", [])}
    if B:
        extra.append("Activation (3 $\\times$ 256) & " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(B.items(), key=lambda x: x[1])))
    C = ab.get("C_ensemble", [])
    if C:
        extra.append("Networks in the ensemble & " + ", ".join(f"{r['members']}: {r['mae_mean']:.2f}" for r in C))
    D = ab.get("D_epochs", [])
    if D:
        extra.append("Training epochs & " + ", ".join(f"{r['epochs']}: {r['mae_mean']:.2f}" for r in D))
    E = [r for r in ab.get("E_tandem", []) if r["depth"] == 3 and r["width"] == 256]
    if E:
        extra.append("Tandem penalty $\\lambda$ (fabricable and $<$1\\,GHz) & " +
                     ", ".join(f"{r['penalty_weight']:g}: {pct_tex(r['within_1GHz_and_valid'])}" for r in E))
    F = [r for r in ab.get("F_cvae", []) if abs(r["beta"] - 0.5) < 1e-9]
    if F:
        extra.append("CVAE latent size (distinct designs $<$1\\,GHz, of 8) & " +
                     ", ".join(f"{r['latent']}: {r['distinct_within_1GHz_of_8']:.1f}" for r in F))
    if extra:
        lines += [r"  \\[6pt]", r"  \begin{tabular}{p{0.38\linewidth}p{0.55\linewidth}}", r"    \toprule",
                  r"    Setting & Validation result \\", r"    \midrule"]
        lines += ["    " + e + r" \\" for e in extra]
        lines += [r"    \bottomrule", r"  \end{tabular}"]
    lines.append(r"\end{table}")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("wrote", path)


def main():
    ab = json.load(open(ABL)) if os.path.exists(ABL) else {}
    write_ablation_tex(ab)
    P = ablation_text(ab)
    ex = build_extras(P)
    ex["ablation"] = {k: v for k, v in ab.items() if k[:2] in ("A_", "B_", "C_", "D_", "E_", "F_")}
    data = json.load(open(SITE))
    data["extras"] = ex
    with open(SITE, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"patched {SITE}: extras with {len(ex['qa'])} questions, {len(P)} test texts")
    for fn in ("arch_tandem.png", "arch_forward.png", "arch_cvae.png", "ablation.png"):
        if os.path.exists(os.path.join(FIGS, fn)):
            shutil.copy(os.path.join(FIGS, fn), os.path.join("learn/site", fn))
    if "--no-pdf" not in sys.argv:
        pdfs(P, ab)


if __name__ == "__main__":
    main()
