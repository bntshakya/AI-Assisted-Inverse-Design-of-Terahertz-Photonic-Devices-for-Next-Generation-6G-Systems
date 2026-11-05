import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app_common import TEAL, holes_fig, numbers

SETS = {"PWE freeform (4,000)": "data/pwe/fourier.csv", "PWE shape library (2,000)": "data/pwe/library.csv",
        "COMSOL freeform": "data/comsol/extra_fourier.csv", "COMSOL AI designs (verification)": "data/comsol/extra_verify.csv",
        "COMSOL sharp triangles": "data/comsol/extra_sharp_lhs.csv", "COMSOL rounded triangles": "data/comsol/extra_rounded_lhs2.csv",
        "COMSOL rounded triangles (first 200)": "data/comsol/extra_rounded_lhs.csv", "COMSOL circles (grid)": "data/comsol/extra_circle.csv",
        "COMSOL circles (new)": "data/comsol/extra_circle_lhs.csv"}


@st.cache_data(show_spinner=False)
def load(path):
    return pd.read_csv(path)


def render():
    N = numbers()
    st.title("The datasets")
    st.markdown("One example = one unit cell (two holes) and its two band edges, f_lower and f_upper.")
    rows = [{"dataset": "PWE freeform", "cells": 4000, "made by": "PWE (computer)", "used for": "training the networks"},
            {"dataset": "PWE shape library", "cells": 2000, "made by": "PWE (computer)", "used for": "training the networks"}]
    if N["final"]:
        for r in N["final"]["families"]:
            rows.append({"dataset": f"COMSOL {r['family']}", "cells": r["cells"], "made by": "COMSOL (you)",
                         "used for": "checking PWE, correction, COMSOL-trained model, final proof"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("PWE cells are split 80/10/10 (train / validate / test); COMSOL cells are evaluated with cross-validation. "
               "The inherited 8 x 8 COMSOL grids of rounded and sharp triangles (64 + 64 cells) are counted with their families.")

    avail = {k: v for k, v in SETS.items() if os.path.exists(v)}
    name = st.selectbox("browse a dataset", list(avail))
    df = load(avail[name])
    g = df.f_upper - df.f_lower
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.metric("cells", f"{len(df):,}")
        st.metric("with a band gap", f"{int((g > 0.1).sum()):,}")
        fig, ax = plt.subplots(figsize=(4.2, 2.6))
        ax.hist(g, bins=40, color=TEAL)
        ax.set_xlabel("gap width (GHz)")
        ax.set_ylabel("cells")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    with c2:
        seed = st.number_input("show other examples", 0, 999, 0)
        smp = df.sample(min(6, len(df)), random_state=int(seed))
        cols = st.columns(3)
        for i, (_, r) in enumerate(smp.iterrows()):
            hA = (r.famA, json.loads(r.paramsA))
            hB = (r.famB, json.loads(r.paramsB))
            with cols[i % 3]:
                st.pyplot(holes_fig(hA, hB, 2.2, title=f"{r.famA} / {r.famB}"), clear_figure=True)
                st.caption(f"{r.f_lower:.1f} / {r.f_upper:.1f} GHz")
