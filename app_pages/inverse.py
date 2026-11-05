import numpy as np
import pandas as pd
import streamlit as st

import designer as dz
from app_common import badges, band_fig, bundle, cell_fig, expected_error, fmt2

MODEL_HELP = {"pwe": "Ranks and refines designs with the network trained on PWE data. Matches the PWE check closely; "
                     "COMSOL usually gives f_lower about 0.5-1 GHz lower.",
              "corr": "Adds the learned COMSOL correction to the predictions, so designs aim at what COMSOL would give. "
                      "The PWE check then differs by about the PWE-COMSOL offset.",
              "comsol": "Ranks and refines designs with the network trained directly on COMSOL data. Aims at COMSOL; "
                        "reliable only for shapes like those simulated in COMSOL (holes within 68 µm)."}


def render():
    B = bundle()
    st.title("Inverse designer")
    st.markdown("Type the band gap you want. The AI proposes hole shapes; each one is **checked with the PWE solver**, "
                "so every design comes with a real error in GHz (the worse of its two band edges).")
    c1, c2, c3 = st.columns([1, 1, 2.4])
    lo = c1.number_input("target f_lower (GHz)", 270.0, 420.0, 300.0, 0.5)
    up = c2.number_input("target f_upper (GHz)", 275.0, 440.0, 330.0, 0.5)
    methods = c3.multiselect("inverse methods", list(dz.METHODS), default=["tandem", "cvae_refine"],
                             format_func=dz.METHODS.get,
                             help="Tandem: one design, the most accurate. CVAE + refine: several different designs. "
                                  "The other two are the comparison methods from the report.")
    models = dz.available_models(B)
    model = st.radio("forward model used to choose the designs", models, horizontal=True, format_func=dz.MODELS.get,
                     help="\n\n".join(f"**{dz.MODELS[m]}**: {MODEL_HELP[m]}" for m in models))
    o1, o2, o3 = st.columns(3)
    k = o1.slider("designs per method", 1, 8, 3, help="The tandem network always gives exactly one.")
    verify = o2.toggle("check each design with PWE", True, help="About 0.2 s per design. The project only reports "
                       "PWE-checked numbers.")
    comsol_only = o3.toggle("only shapes that fit the COMSOL cell", True, help="Hole reach ≤ 68 µm, so the design can "
                            "be simulated in your COMSOL model.")
    b1, b2, _ = st.columns([1, 1.3, 4])
    go = b1.button("Design", type="primary")
    if b2.button("Try new samples"):
        st.session_state.seed = st.session_state.get("seed", 0) + 1
        go = True
    if go:
        if up <= lo:
            st.error("f_upper must be larger than f_lower.")
        elif not methods:
            st.error("Pick at least one method.")
        else:
            with st.spinner("Designing and checking…"):
                st.session_state.res = dz.design(B, (lo, up), methods, k, None, comsol_only, verify,
                                                 st.session_state.get("seed", 0), model=model)
                st.session_state.target = (lo, up)
    res = st.session_state.get("res")
    if res is None:
        return
    T = st.session_state.target
    if not res:
        st.warning("No valid design found. Try a target inside the training range (gaps of about 2-40 GHz between "
                   "roughly 290 and 380 GHz), or allow shapes that cross the COMSOL cell.")
        return
    st.subheader(f"{len(res)} designs for {T[0]:g} / {T[1]:g} GHz")
    aim_comsol = res[0]["model"] in ("corr", "comsol") and B["corr"] is not None
    rows = []
    for i, d in enumerate(res):
        r = {"#": i + 1, "method": dz.METHODS[d["method"]], "predicted": fmt2(d["pred"]),
             "PWE check": fmt2(d["pwe"]) if "pwe" in d else "-",
             "PWE error (GHz)": round(d["err_pwe"], 2) if "pwe" in d else None}
        err = d.get("err_pwe")
        if aim_comsol and "pwe" in d:
            delta = dz.predict(B, d["p"][None], "corr")[0][0] - dz.predict(B, d["p"][None], "pwe")[0][0]
            est = d["pwe"] + delta
            err = float(np.abs(est - np.asarray(T)).max())
            r["COMSOL estimate"] = fmt2(est)
            r["est. error (GHz)"] = round(err, 2)
        r["within 1 GHz"] = ("yes" if err < 1 else "no") if err is not None else "-"
        r.update({"wall (µm)": round(d["wall"], 1), "reach (µm)": round(d["reach"], 1),
                  "fits COMSOL": "yes" if d["fits_comsol"] else "no"})
        rows.append(r)
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    if aim_comsol:
        st.info("These designs aim at COMSOL, which sits about 0.5-1 GHz from PWE. 'COMSOL estimate' = the PWE check plus "
                "the learned COMSOL correction; 'within 1 GHz' is judged on that estimate. Only a COMSOL run gives the final word.")
    mae, p95, what = expected_error(res[0]["model"])
    if mae:
        st.caption(f"Predicted = {dz.MODELS[res[0]['model']]} (typical error {fmt2(mae)} GHz against {what}). "
                   "PWE check = the design re-simulated with the physics solver; its error is the worse of the two edges.")
    cols = st.columns(4)
    for i, d in enumerate(res):
        with cols[i % 4]:
            st.pyplot(cell_fig(d["p"], title=f"#{i + 1} {dz.METHODS[d['method']]}"), clear_figure=True)
            if "pwe" in d:
                st.markdown(f"PWE **{fmt2(d['pwe'])}** GHz")
    st.subheader("Look closer at one design")
    sel = st.selectbox("design", list(range(len(res))), format_func=lambda i: f"#{i + 1} - {dz.METHODS[res[i]['method']]}")
    d = res[sel]
    l, r = st.columns([1, 1.4])
    with l:
        st.markdown("**The 10 design numbers (µm)**")
        st.dataframe(pd.DataFrame([np.round(d["p"][:5], 2), np.round(d["p"][5:], 2)], index=["hole A", "hole B"],
                                  columns=["r0", "c3", "s3", "c6", "s6"]))
        st.markdown(badges(dz.check(d["p"])[1]))
    with r:
        if st.button("Simulate the full band diagram with PWE (about 3 s)"):
            with st.spinner("Solving 31 k-points…"):
                fig, fl, fu = band_fig(d["p"])
            st.pyplot(fig, clear_figure=True)
            st.markdown(f"f_lower **{fl:.2f}**, f_upper **{fu:.2f}**, gap **{fu - fl:.2f} GHz**")
    st.subheader("Send designs to COMSOL")
    pick = st.multiselect("designs to include", list(range(len(res))), default=[i for i, x in enumerate(res) if x["fits_comsol"]],
                          format_func=lambda i: f"#{i + 1}")
    if pick:
        st.download_button("Download COMSOL sweep file (.txt)", dz.comsol_sweep([res[i]["p"] for i in pick]),
                           file_name=f"comsol_designs_{T[0]:g}_{T[1]:g}.txt")
        st.caption("In COMSOL: Study 1 > Parametric Sweep > Specified combinations > Load from File; unit m for the "
                   "shape rows; export emw.freq with Global Evaluation.")
