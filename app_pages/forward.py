import numpy as np
import pandas as pd
import streamlit as st

import designer as dz
from app_common import PRESETS, RANGES, ACCURACY_TEXT, badges, band_fig_holes, bundle, expected_error, fmt2, holes_fig
from geometry import cell_features, check_cell

FAMS = ["freeform (Fourier)", "circles", "rounded triangles", "sharp triangles"]


def comsol_coverage(fam, a, b, reach):
    if fam == "freeform (Fourier)":
        return reach <= dz.COMSOL_REACH
    if fam == "circles":
        return 46 <= a <= 65 and 28 <= b <= 47
    return 140 <= a <= 189 and 91 <= b <= 140


def render():
    B = bundle()
    st.title("Forward predictor")
    st.markdown("Pick or draw a unit cell. Three models predict its band edges; each is shown with its **expected "
                "accuracy**, and you can check the shape with the **PWE solver** to see the true error for this shape.")
    fam = st.radio("hole family", FAMS, horizontal=True)
    c_in, c_out = st.columns([1.1, 1.3])
    with c_in:
        if fam == "freeform (Fourier)":
            pre = st.selectbox("start from", list(PRESETS))
            if st.button("Load this shape") or "fA_r0" not in st.session_state:
                for h, vals in zip("AB", PRESETS[pre]):
                    for (nm, (lo, hi)), v in zip(RANGES.items(), vals):
                        st.session_state[f"f{h}_{nm}"] = float(np.clip(v, lo, hi))
            ca, cb = st.columns(2)
            for h, col in (("A", ca), ("B", cb)):
                with col:
                    st.markdown(f"**Hole {h}**")
                    for nm, (lo, hi) in RANGES.items():
                        st.slider(nm, lo, hi, step=0.1, key=f"f{h}_{nm}")
            keys = ["r0", "c3", "s3", "c6", "s6"]
            hA = ("fourier", {k: st.session_state[f"fA_{k}"] for k in keys})
            hB = ("fourier", {k: st.session_state[f"fB_{k}"] for k in keys})
            a, b = hA[1]["r0"], hB[1]["r0"]
        elif fam == "circles":
            a = st.slider("hole A radius (µm)", 25.0, 75.0, 55.0, 0.5)
            b = st.slider("hole B radius (µm)", 25.0, 75.0, 38.0, 0.5)
            hA, hB = ("circle", {"R": a}), ("circle", {"R": b})
        else:
            f = "rounded" if fam == "rounded triangles" else "sharp"
            a = st.slider("hole A side L1 (µm)", 90.0, 200.0, 168.0, 1.0)
            b = st.slider("hole B side L2 (µm)", 80.0, 200.0, 119.0, 1.0)
            hA, hB = (f, {"L": a}), (f, {"L": b})
    ok, info = check_cell(hA, hB)
    reach = max(info["rmax_A"], info["rmax_B"])
    feat = cell_features(hA, hB)
    with c_out:
        st.pyplot(holes_fig(hA, hB, 3.4), clear_figure=True)
        st.markdown(badges(info))
        rows = []
        for m in dz.available_models(B):
            mu, sd = dz.predict_features(B, feat[None], m)
            mae, p95, what = expected_error(m)
            rows.append({"model": dz.MODELS[m], "f_lower (GHz)": round(mu[0][0], 2), "f_upper (GHz)": round(mu[0][1], 2),
                         "gap (GHz)": round(mu[0][1] - mu[0][0], 2),
                         "networks disagree by": f"± {np.max(sd[0]):.2f}" if m != "corr" else "-",
                         "typical error": fmt2(mae) if mae else "-", "95 % within": fmt2(p95) if p95 else "-",
                         "answers like": "PWE" if m == "pwe" else "COMSOL"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        notes = []
        if not ok:
            notes.append("This cell fails a fabrication check, so every prediction is less reliable.")
        if "comsol" in dz.available_models(B) and not comsol_coverage(fam, a, b, reach):
            notes.append("This shape is outside what was simulated in COMSOL, so the COMSOL-trained model is extrapolating.")
        for n in notes:
            st.warning(n)
        if st.button("Check with PWE (true error for this shape)", type="primary"):
            with st.spinner("Solving the band diagram…"):
                fig, fl, fu = band_fig_holes(hA, hB)
            st.pyplot(fig, clear_figure=True)
            mu = dz.predict_features(B, feat[None], "pwe")[0][0]
            st.markdown(f"PWE: f_lower **{fl:.2f}**, f_upper **{fu:.2f}**, gap **{fu - fl:.2f} GHz**. "
                        f"PWE-trained network error for this shape: **{abs(mu[0] - fl):.2f} / {abs(mu[1] - fu):.2f} GHz**. "
                        "(COMSOL-level models aim at COMSOL, which usually sits about 0.5 GHz from PWE.)")
    with st.expander("Why does a forward prediction not come with its own exact accuracy?"):
        st.markdown(ACCURACY_TEXT)
