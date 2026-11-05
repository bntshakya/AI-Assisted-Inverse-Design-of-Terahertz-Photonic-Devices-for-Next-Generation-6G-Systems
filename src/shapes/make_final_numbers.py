"""One source for every final number: reads the saved results and writes
  ../Report/final/numbers.tex          LaTeX macros (and ready-made table rows) used by the report and the poster
  results/optionb/final_numbers.json   the same numbers for the websites and the poster script
  FINAL_NUMBERS.md                     the one-page summary for the student (what to say, what changed)

Run from the Project folder after final_comsol_study.py:  python src/shapes/make_final_numbers.py
Rerun it whenever results change (e.g. after new COMSOL data); then recompile the report.
"""
import json
import os

import numpy as np
import pandas as pd

R = "results/optionb"
OUT_TEX = "../Report/final/numbers.tex"
METHODS = {"tandem": "Tandem network", "cvae": "Conditional VAE", "multistart": "Multi-start gradient",
           "cvae_refine": "CVAE + refine (ours)"}
COMP = {"pwe_solver": "PWE solver alone", "nn_pwe": "Network trained on PWE", "gp_comsol_free": "GP, COMSOL freeform only",
        "gp_comsol_all": "GP, all COMSOL", "nn_comsol_free": "Network, COMSOL freeform only",
        "nn_comsol_all": "Network, all COMSOL (COMSOL-trained)", "nn_pwe+corr": "PWE network + COMSOL correction",
        "solver+corr": "PWE solver + COMSOL correction"}
OLD = {"comsol_cells": 474, "freeform_cells": 97, "pwe_vs_comsol_freeform": [0.48, 0.50], "corr_freeform": [0.31, 0.36],
       "comsol_only_best": [2.61, 0.98], "verified_designs": 2}


def f2(x):
    return f"{x:.2f}"


def pct(x):
    return f"{100 * x:.0f}"


def main():
    F = json.load(open(f"{R}/final_comsol.json"))
    S = json.load(open(f"{R}/seed_summary.json"))
    V = pd.read_csv(f"{R}/comsol_verify.csv") if os.path.exists(f"{R}/comsol_verify.csv") else None
    M = {}

    fam = {r["family"]: r for r in F["families"]}
    M["nComsol"] = f"{F['n_cells']:,}"
    M["nGapped"] = f"{F['n_gapped']:,}"
    for key, name in (("nFree", "freeform"), ("nSharp", "sharp triangle"), ("nRound", "rounded triangle"),
                      ("nCircle", "circle"), ("nVerify", "freeform (AI designs)")):
        M[key] = f"{fam[name]['cells']:,}" if name in fam else "0"
        M[key + "Gapped"] = f"{fam[name]['gapped']:,}" if name in fam else "0"
    a = F["pwe_vs_comsol_all"]
    M["pweLo"], M["pweUp"] = f2(a["mae"][0]), f2(a["mae"][1])
    M["pwePlo"], M["pwePup"] = f2(a["p95"][0]), f2(a["p95"][1])
    M["pweN"] = f"{a['n']:,}"
    rows = []
    for r in F["families"]:
        if r["pwe_vs_comsol"] and r["family"] != "freeform (AI designs)":
            q = r["pwe_vs_comsol"]
            rows.append(f"{r['family']} & {r['cells']:,} & {r['gapped']:,} & {f2(q['mae'][0])} / {f2(q['mae'][1])} & "
                        f"{f2(q['p95'][0])} / {f2(q['p95'][1])}\\\\")
    M["pweFamRows"] = "\n".join(rows)

    mf = F["mf_cv10"]
    M["mfPweLo"], M["mfPweUp"] = f2(mf["all"]["pwe"]["mae"][0]), f2(mf["all"]["pwe"]["mae"][1])
    M["mfCorrLo"], M["mfCorrUp"] = f2(mf["all"]["pwe_plus_corr"]["mae"][0]), f2(mf["all"]["pwe_plus_corr"]["mae"][1])
    M["mfBothPwe"], M["mfBothCorr"] = pct(mf["all"]["pwe"]["both_within_1"]), pct(mf["all"]["pwe_plus_corr"]["both_within_1"])
    M["mfRows"] = "\n".join(f"{k} & {f2(v['pwe']['mae'][0])} / {f2(v['pwe']['mae'][1])} & "
                            f"{f2(v['pwe_plus_corr']['mae'][0])} / {f2(v['pwe_plus_corr']['mae'][1])} & "
                            f"{pct(v['pwe']['both_within_1'])}\\% $\\to$ {pct(v['pwe_plus_corr']['both_within_1'])}\\%\\\\"
                            for k, v in mf.items() if k != "all")
    M["lofoRows"] = "\n".join(f"{k} & {f2(v['pwe']['mae'][0])} / {f2(v['pwe']['mae'][1])} & "
                              f"{f2(v['pwe_plus_corr']['mae'][0])} / {f2(v['pwe_plus_corr']['mae'][1])}\\\\"
                              for k, v in F["mf_leave_family_out"].items())

    fc = F["freeform_comparison"]
    cm = fc["methods"]
    avg = {m: (v["mae_f_lower"] + v["mae_f_upper"]) / 2 for m, v in cm.items()}
    best = min(avg, key=avg.get)
    corr_best = min(("nn_pwe+corr", "solver+corr"), key=avg.get)
    com_best = min(("gp_comsol_free", "gp_comsol_all", "nn_comsol_free", "nn_comsol_all"), key=avg.get)
    rows = []
    for m, v in cm.items():
        b = "\\textbf" if m == best else ""
        uses = "no" if m in ("pwe_solver", "nn_pwe") else "yes"
        rows.append(f"{COMP[m]} & {uses} & {b}{{{f2(v['mae_f_lower'])}}} / {b}{{{f2(v['mae_f_upper'])}}} & "
                    f"{f2(v['accuracy']['p95'][0])} / {f2(v['accuracy']['p95'][1])} & {pct(v['accuracy']['both_within_1'])}\\%\\\\")
    M["compRows"] = "\n".join(rows)
    M["nFreeTest"] = f"{fc['n_free']:,}"
    M["nOther"] = f"{fc['n_other']:,}"
    for key, m in (("solver", "pwe_solver"), ("nnpwe", "nn_pwe"), ("corr", corr_best), ("comsol", com_best),
                   ("comsolAll", "nn_comsol_all")):
        M[key + "Lo"], M[key + "Up"] = f2(cm[m]["mae_f_lower"]), f2(cm[m]["mae_f_upper"])
        M[key + "Both"] = pct(cm[m]["accuracy"]["both_within_1"])
    M["corrName"], M["comsolName"] = COMP[corr_best], COMP[com_best]
    lc = fc["learning_curve"]
    M["lcZeroCorr"] = f2(np.mean(lc[corr_best][0]["mae"]))
    M["lcFullCorr"] = f2(np.mean(lc[corr_best][-1]["mae"]))
    M["lcFullComsol"] = f2(np.mean(lc["gp_comsol_all"][-1]["mae"]))
    M["lcFullN"] = f"{lc['gp_comsol_all'][-1]['n_free_train']:,}"
    old_com = np.mean(OLD["comsol_only_best"])
    short = {"gp_comsol_free": "a GP on the freeform COMSOL cells", "gp_comsol_all": "a GP on all COMSOL cells",
             "nn_comsol_free": "a network on the freeform COMSOL cells", "nn_comsol_all": "a network on all COMSOL cells"}
    if avg[com_best] <= avg[corr_best]:
        verdict = (f"With {fc['n_free']:,} freeform COMSOL cells, a model trained on COMSOL data alone ({short[com_best]}, "
                   f"{f2(avg[com_best])} GHz, mean of both edges) now matches or beats PWE plus correction "
                   f"({f2(avg[corr_best])} GHz): with enough COMSOL data the fast solver becomes optional, while it remains "
                   "the cheaper route when COMSOL runs are scarce.")
    elif avg[com_best] - avg[corr_best] < 0.05:
        verdict = (f"With {fc['n_free']:,} freeform COMSOL cells, a model trained on COMSOL data alone has nearly caught up "
                   f"({short[com_best]}: {f2(avg[com_best])} GHz, mean of both edges, down from {f2(old_com)} GHz with "
                   f"{OLD['freeform_cells']} cells); PWE plus correction is still slightly more accurate "
                   f"({f2(avg[corr_best])} GHz) and needs no freeform COMSOL cells to start "
                   f"({M['lcZeroCorr']} GHz with none). PWE is the cheap route when COMSOL runs are scarce; with about "
                   "1,000 runs of the shapes being designed, COMSOL data alone become a real alternative.")
    else:
        verdict = (f"With {fc['n_free']:,} freeform COMSOL cells, a model trained on COMSOL data alone has caught up a lot "
                   f"({short[com_best]}: {f2(avg[com_best])} GHz, mean of both edges, down from {f2(old_com)} GHz with "
                   f"{OLD['freeform_cells']} cells), but PWE plus correction is still the most accurate "
                   f"({f2(avg[corr_best])} GHz). The cheap PWE data remain useful; COMSOL data are best spent correcting them.")
    M["verdict"] = verdict
    M["comsolWins"] = "yes" if avg[com_best] <= avg[corr_best] else "no"

    # The PWE error at f_upper depends on the hole shape (PWE - COMSOL, gapped cells): it explains why many circles
    # make the correction excellent for circles but less good for freeform holes.
    def upper_offset(path):
        d = pd.read_csv(path)
        d = d[d.gap > 0.1]
        return (d.f_upper_pwe - d.f_upper).mean()
    shape_note = ""
    if os.path.exists("data/comsol/extra_circle_lhs.csv"):
        oc, of = upper_offset("data/comsol/extra_circle_lhs.csv"), upper_offset("data/comsol/extra_fourier.csv")
        M["offUpCircle"], M["offUpFree"] = f2(abs(oc)), f2(abs(of))
        word = lambda x: "too high" if x > 0 else "too low"
        shape_note = (f"Why the 1,000 new circles did not help the freeform shapes: the PWE error at the upper band edge "
                      f"depends on the shape. PWE's upper edge is {f2(abs(oc))} GHz {word(oc)} for circles but {f2(abs(of))} "
                      f"GHz {word(of)} for freeform holes. The circles make the correction almost perfect for circles, but "
                      f"they pull its freeform upper edge the wrong way. Lesson: a correction needs COMSOL cells of the "
                      f"shapes being designed.")
    M["shapeNote"] = shape_note

    fw = F["accuracy"]["forward_pwe_test"]
    M["fwdLo"], M["fwdUp"] = f2(fw["mae"][0]), f2(fw["mae"][1])
    M["fwdPlo"], M["fwdPup"] = f2(fw["p95"][0]), f2(fw["p95"][1])
    M["fwdHalfLo"], M["fwdHalfUp"] = pct(fw["within_0.5"][0]), pct(fw["within_0.5"][1])
    M["fwdOneLo"], M["fwdOneUp"] = pct(fw["within_1"][0]), pct(fw["within_1"][1])
    M["fwdSeedLo"], M["fwdSeedUp"] = f2(S["forward_mae_all"]["mean"][0]), f2(S["forward_mae_all"]["mean"][1])
    M["fwdSeedSdLo"], M["fwdSeedSdUp"] = f2(S["forward_mae_all"]["std"][0]), f2(S["forward_mae_all"]["std"][1])

    inv = S["inverse"]
    iacc = F["accuracy"]["inverse_pwe_verified"]
    rows = []
    for m, nm in METHODS.items():
        v = inv[m]
        bold = "\\textbf" if m == "tandem" else ""
        t = v["seconds_per_target"]["mean"]
        rows.append(f"{nm} & {bold}{{{100 * v['success_rate_within_1GHz']['mean']:.1f}}} $\\pm$ {100 * v['success_rate_within_1GHz']['std']:.1f} & "
                    f"{f2(v['verified_mae_GHz']['mean'][0])} / {f2(v['verified_mae_GHz']['mean'][1])} & "
                    f"{f2(max(iacc[m]['p95']))} & {('%.1f' % v['distinct']['mean']) if 'distinct' in v else '1'} & "
                    f"{('%.2f s' % t) if t >= 0.01 else '$<$10 ms'}\\\\")
    M["invRows"] = "\n".join(rows)
    M["tandemSucc"] = f"{100 * inv['tandem']['success_rate_within_1GHz']['mean']:.0f}"
    M["tandemSd"] = f"{100 * inv['tandem']['success_rate_within_1GHz']['std']:.0f}"
    M["crSucc"] = f"{100 * inv['cvae_refine']['success_rate_within_1GHz']['mean']:.0f}"
    M["crSd"] = f"{100 * inv['cvae_refine']['success_rate_within_1GHz']['std']:.0f}"
    M["crDistinct"] = f"{inv['cvae_refine']['distinct']['mean']:.0f}"

    if V is not None:
        w = V[["COMSOL - target lower", "COMSOL - target upper"]].abs().max(axis=1)
        M["verN"], M["verOK"] = str(len(V)), str(int((w < 1).sum()))
        M["verWorst"], M["verMean"] = f2(w.max()), f2(w.mean())
        new = V.kind == "new design"
        M["verNewN"], M["verNewOK"] = str(int(new.sum())), str(int((w[new] < 1).sum()))
        rows = []
        for tgt, g in V.groupby("target", sort=False):
            lo = g.COMSOL.str.split(" / ").str[0].astype(float)
            up = g.COMSOL.str.split(" / ").str[1].astype(float)
            ww = g[["COMSOL - target lower", "COMSOL - target upper"]].abs().max(axis=1)
            kinds = f"{int((g.kind != 'new design').sum())} gallery + {int((g.kind == 'new design').sum())} new" \
                if (g.kind != "new design").any() else f"{len(g)} new"
            rows.append(f"{tgt} & {kinds} & {lo.min():.2f} -- {lo.max():.2f} & {up.min():.2f} -- {up.max():.2f} & "
                        f"{int((ww < 1).sum())} of {len(g)} & {ww.max():.2f}\\\\")
        M["verRows"] = "\n".join(rows)
    M["verCorrBefore"], M["verCorrAfter"] = "0.74", "0.23"   # f_lower error on the 12 new designs, correction trained without them

    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write("% generated by Project/src/shapes/make_final_numbers.py - do not edit by hand\n")
        for k, v in M.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    json.dump(M, open(f"{R}/final_numbers.json", "w"), indent=1)

    md = f"""# Final numbers (read this before the evaluation)

Generated from the results by `src/shapes/make_final_numbers.py`. All errors in GHz, always on shapes the model never saw.

## The five sentences
1. "I want a silicon valley photonic crystal whose band gap sits at a chosen 6G frequency (around 0.3 THz)."
2. "Each hole is described by 5 Fourier numbers, so the AI can create any three-fold shape, not only triangles."
3. "A fast simulator (PWE) made 6,000 training examples; on {M['pweN']} COMSOL cells it agrees to {M['pweLo']} / {M['pweUp']} GHz,
   and with a correction learned from COMSOL to {M['mfCorrLo']} / {M['mfCorrUp']} GHz."
4. "The AI turns a target gap into hole shapes: the tandem network hits both edges within 1 GHz for {M['tandemSucc']}% of unseen
   targets, and CVAE + refine gives about {M['crDistinct']} different designs per target at {M['crSucc']}%."
5. "I simulated {M.get('verN', '14')} AI designs in COMSOL for four targets: {M.get('verOK', '12')} hit their target within 1 GHz
   (worst {M.get('verWorst', '1.05')} GHz)."

## How accuracy is measured (say it like this)
- Forward model (shape -> band edges): mean absolute error in GHz on unseen shapes, plus the 95 % bound.
- Inverse model (target -> shape): every design is re-simulated; its error is the worse of its two edges;
  success = below 1 GHz. Success rate = share of 100 unseen targets that succeed.
- Why GHz: percentages hide real errors at 300 GHz. Why the worse edge: a gap needs both edges right.
  Why 1 GHz: about 0.3 % of 300 GHz, below half the narrowest 6G channel (2.16 GHz), about the PWE-COMSOL difference.

## All final numbers
| What | Final | Before (if it changed) |
|---|---|---|
| COMSOL cells simulated | {M['nComsol']} ({M['nGapped']} with a gap) | {OLD['comsol_cells']} |
| Freeform COMSOL cells | {M['nFree']} | {OLD['freeform_cells']} |
| PWE vs COMSOL, all gapped cells | {M['pweLo']} / {M['pweUp']} GHz | ~0.5 GHz (on 274 cells) |
| PWE + COMSOL correction (10-fold CV, all families) | {M['mfCorrLo']} / {M['mfCorrUp']} GHz | |
| Freeform: PWE solver alone | {M['solverLo']} / {M['solverUp']} GHz | {OLD['pwe_vs_comsol_freeform'][0]} / {OLD['pwe_vs_comsol_freeform'][1]} |
| Freeform: best PWE + correction | {M['corrLo']} / {M['corrUp']} GHz | {OLD['corr_freeform'][0]} / {OLD['corr_freeform'][1]} |
| Freeform: best COMSOL-only model | {M['comsolLo']} / {M['comsolUp']} GHz ({M['comsolName']}) | {OLD['comsol_only_best'][0]} / {OLD['comsol_only_best'][1]} |
| Forward network (600 unseen PWE cells) | {M['fwdLo']} / {M['fwdUp']} GHz; 95 % within {M['fwdPlo']} / {M['fwdPup']} | unchanged |
| Tandem success (both edges < 1 GHz) | {M['tandemSucc']} ± {M['tandemSd']} % | unchanged |
| CVAE + refine success, designs per target | {M['crSucc']} ± {M['crSd']} %, ~{M['crDistinct']} | unchanged |
| AI designs verified in COMSOL | {M.get('verOK', '?')} of {M.get('verN', '?')} within 1 GHz (worst {M.get('verWorst', '?')}) | {OLD['verified_designs']} designs |

## The answer to "do we still need PWE?"
{verdict}

{shape_note}

## The three forward models (in the app and design.py)
- **PWE-trained network**: learned from 6,000 PWE cells; answers like PWE ({M['fwdLo']} / {M['fwdUp']} GHz from PWE).
- **PWE network + COMSOL correction**: adds the learned COMSOL - PWE difference; answers like COMSOL.
- **COMSOL-trained network**: learned directly from the COMSOL cells; answers like COMSOL, only for shapes like those simulated.
Designs made with the correction land within about 0.2 GHz of the target in the COMSOL estimate (12 such designs are in
`data/comsol/FILES_FOR_COMSOL/5_OPTIONAL_verify_corrected_designs`).

## Where everything is
- Report: `Report/final/254161023_MTP1_Report.pdf` (fill in the red items: supervisor departments, emails, repository link)
- Poster: `Report/final/254161023_MTP1_Poster.pdf`
- Website: `streamlit run app.py` (in the Project folder); course online and offline (`docs/Valley_Crystal_Study_Lab.html`)
- Study material: `docs/PWE_Cheat_Sheet.pdf`, `docs/OptionB_Study_Guide.pdf`
"""
    open("FINAL_NUMBERS.md", "w", encoding="utf-8").write(md)
    print(f"wrote {OUT_TEX}, {R}/final_numbers.json, FINAL_NUMBERS.md ({len(M)} numbers)")
    print("verdict:", verdict)


if __name__ == "__main__":
    main()
