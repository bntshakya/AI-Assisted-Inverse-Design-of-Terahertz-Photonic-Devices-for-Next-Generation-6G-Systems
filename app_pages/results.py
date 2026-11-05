import os

import pandas as pd
import streamlit as st

import designer as dz
from app_common import FIG, fmt2, numbers

NICE = {"pwe_solver": "PWE solver alone", "nn_pwe": "network trained on PWE", "gp_comsol_free": "GP on COMSOL freeform only",
        "gp_comsol_all": "GP on all COMSOL", "nn_comsol_free": "network on COMSOL freeform only",
        "nn_comsol_all": "network on all COMSOL (COMSOL-trained model)", "nn_pwe+corr": "PWE network + COMSOL correction",
        "solver+corr": "PWE solver + COMSOL correction"}


def pct(x):
    return f"{100 * x:.1f} %"


def render():
    N = numbers()
    f, s, main = N["final"], N["seeds"], N["main"]
    st.title("Final results")
    st.caption("All errors in GHz, always on shapes the model never saw. See 'How accuracy is measured' for what each "
               "number means.")

    st.header("1. Forward model: shape → band edges")
    if f:
        a = f["accuracy"]["forward_pwe_test"]
        st.markdown(f"Network trained on 5,400 PWE cells, tested on **{a['n']} unseen PWE cells**:")
        st.dataframe(pd.DataFrame([{"edge": e, "mean error": a["mae"][i], "median": a["median"][i], "95 % within": a["p95"][i],
                                    "max": a["max"][i], "within 0.5 GHz": pct(a["within_0.5"][i]),
                                    "within 1 GHz": pct(a["within_1"][i])} for i, e in enumerate(["f_lower", "f_upper"])]),
                     hide_index=True)
    if s:
        st.caption(f"Over 5 repeated runs: {fmt2(s['forward_mae_all']['mean'])} ± {fmt2(s['forward_mae_all']['std'])} GHz.")

    st.header("2. Inverse design: target → shapes")
    if s:
        inv = s["inverse"]
        acc = f["accuracy"]["inverse_pwe_verified"] if f else {}
        st.markdown("100 unseen targets, every design re-simulated with PWE, 5 repeated runs (mean ± spread):")
        st.dataframe(pd.DataFrame([{
            "method": dz.METHODS[m],
            "success (both edges < 1 GHz)": f"{100 * v['success_rate_within_1GHz']['mean']:.1f} ± {100 * v['success_rate_within_1GHz']['std']:.1f} %",
            "mean error f_lower / f_upper": fmt2(v["verified_mae_GHz"]["mean"]),
            "95 % of worst-edge errors below": f"{max(acc[m]['p95']):.2f}" if m in acc else "-",
            "different designs (of 8)": f"{v['distinct']['mean']:.1f}" if "distinct" in v else "1",
            "time per target": f"{v['seconds_per_target']['mean']:.2f} s" if v['seconds_per_target']['mean'] >= 0.01 else "< 10 ms"}
            for m, v in inv.items()]), hide_index=True)
    if os.path.exists(f"{FIG}/inverse.png"):
        st.image(f"{FIG}/inverse.png")

    st.header("3. Final proof: AI designs simulated in COMSOL")
    if N["verify"] is not None:
        v = N["verify"]
        worst = v[["COMSOL - target lower", "COMSOL - target upper"]].abs().max(1)
        st.markdown(f"**{int((worst < 1).sum())} of {len(v)}** designs within 1 GHz of their target in COMSOL; worst "
                    f"**{worst.max():.2f} GHz**, mean worst-edge error **{worst.mean():.2f} GHz**. The misses are "
                    "systematic: COMSOL puts f_lower 0.5-1 GHz below PWE (the known PWE offset).")
        st.dataframe(v, hide_index=True)

    if f:
        st.header("4. Is the fast solver (PWE) accurate?")
        st.markdown(f"PWE against COMSOL on **{f['pwe_vs_comsol_all']['n']:,} gapped COMSOL cells**: mean error "
                    f"**{fmt2(f['pwe_vs_comsol_all']['mae'])} GHz**.")
        st.dataframe(pd.DataFrame([{"shape family": r["family"], "COMSOL cells": r["cells"], "with a gap": r["gapped"],
                                    "PWE mean error": fmt2(r["pwe_vs_comsol"]["mae"]) if r["pwe_vs_comsol"] else "-",
                                    "95 % within": fmt2(r["pwe_vs_comsol"]["p95"]) if r["pwe_vs_comsol"] else "-"}
                                   for r in f["families"]]), hide_index=True)
        if N["conv"]:
            st.caption("Convergence: with 439 plane waves the band edges are within about 0.5 GHz of 1,011 waves; the "
                       "calibrated permittivity absorbs this nearly uniform shift.")

        st.header("5. Correcting PWE with COMSOL (multi-fidelity)")
        mf = f["mf_cv10"]
        st.markdown("COMSOL ≈ PWE + a Gaussian-process correction learned from COMSOL cells; 10-fold cross-validation:")
        st.dataframe(pd.DataFrame([{"family": k, "PWE alone": fmt2(v["pwe"]["mae"]), "PWE + correction": fmt2(v["pwe_plus_corr"]["mae"]),
                                    "both edges < 1 GHz": f"{pct(v['pwe']['both_within_1'])} → {pct(v['pwe_plus_corr']['both_within_1'])}"}
                                   for k, v in mf.items()]), hide_index=True)

        st.header("6. Do we still need PWE?")
        fc = f["freeform_comparison"]
        st.markdown(f"Each of the **{fc['n_free']:,} freeform COMSOL cells** predicted by models that never saw it "
                    "(5-fold cross-validation, 3 repeats):")
        st.dataframe(pd.DataFrame([{"predictor": NICE.get(m, m), "uses COMSOL data": "no" if m in ("pwe_solver", "nn_pwe") else "yes",
                                    "mean error f_lower / f_upper": f"{v['mae_f_lower']:.2f} / {v['mae_f_upper']:.2f}",
                                    "95 % within": fmt2(v["accuracy"]["p95"]),
                                    "both edges < 1 GHz": pct(v["accuracy"]["both_within_1"])}
                                   for m, v in fc["methods"].items()]), hide_index=True)
        if os.path.exists(f"{FIG}/comsol_vs_pwe.png"):
            st.image(f"{FIG}/comsol_vs_pwe.png", caption="(a) errors of each predictor; (b) error against the number of "
                     "freeform COMSOL cells used for training.")
    else:
        st.info("The final COMSOL study has not been run yet (src/shapes/final_comsol_study.py).")

    st.header("7. Figures")
    for fig, cap in (("pwe_validation.png", "PWE against COMSOL"), ("mk_shortcut.png", "Band edges sit at M or K"),
                     ("forward_mf.png", "Forward model and multi-fidelity correction"), ("showcase.png", "Designs for 300 / 330 GHz")):
        if os.path.exists(f"{FIG}/{fig}"):
            st.image(f"{FIG}/{fig}", caption=cap)
