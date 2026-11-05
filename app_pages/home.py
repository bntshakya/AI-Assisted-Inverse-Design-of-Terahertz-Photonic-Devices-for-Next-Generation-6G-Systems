import os

import streamlit as st

from app_common import FIG, numbers


def render():
    N = numbers()
    f, s = N["final"], N["seeds"]
    st.title("AI-assisted inverse design of terahertz valley photonic crystals")
    st.caption("Binit Shakya · M.Tech. Data Science (P29), IIT Guwahati · supervisors Prof. Debabrata Sikdar and "
               "Prof. Teena Sharma")
    st.markdown("**Give a target band gap; the AI designs the hole shapes of a silicon photonic crystal that produce it, "
                "and every design is checked by physics simulation.**")

    c = st.columns(4)
    if s:
        c[0].metric("targets reached within 1 GHz", f"{100 * s['inverse']['tandem']['success_rate_within_1GHz']['mean']:.0f} %",
                    "tandem network, PWE-checked", delta_color="off")
        c[1].metric("different designs per target", f"{s['inverse']['cvae_refine']['distinct']['mean']:.0f}",
                    "CVAE + refine", delta_color="off")
    if N["verify"] is not None:
        v = N["verify"]
        worst = v[["COMSOL - target lower", "COMSOL - target upper"]].abs().max(1)
        c[2].metric("AI designs within 1 GHz in COMSOL", f"{int((worst < 1).sum())} of {len(v)}",
                    f"worst {worst.max():.2f} GHz", delta_color="off")
    if f:
        c[3].metric("COMSOL cells simulated", f"{f['n_cells']:,}", f"{f['n_gapped']:,} with a band gap", delta_color="off")

    st.subheader("The project in one minute")
    st.markdown("""
- **Why:** 6G links will use frequencies near **0.3 THz**. A silicon *valley photonic crystal* (a slab with a
  repeating pattern of air holes) guides such waves robustly, but only if its **band gap** sits at the right frequencies.
- **The problem:** going from a wanted band gap (f_lower, f_upper) to the hole shapes is hard; simulators only go the
  other way, and slowly.
- **The idea:** describe each hole by 5 Fourier numbers (any three-fold shape, including circles and triangles), make
  thousands of training examples with a fast solver (PWE, checked against COMSOL), and train neural networks that
  turn a target into shapes. Every design is re-simulated before it counts.
""")
    if os.path.exists(f"{FIG}/pipeline.png"):
        st.image(f"{FIG}/pipeline.png", caption="The pipeline: COMSOL calibrates a fast solver, which generates the training "
                 "data; the AI proposes shapes; every design is checked by simulation.")
    st.subheader("Use this site")
    st.markdown("""
| Page | What you can do |
|---|---|
| **Learn** | The full course from zero (22 chapters, with live demos) |
| **Inverse designer** | Type a target band gap and get hole shapes, checked with PWE, downloadable for COMSOL |
| **Forward predictor** | Draw a shape and get its band edges from three different models, with their accuracy |
| **Results** | Every final number and figure |
| **How accuracy is measured** | What each number means and why it is measured that way |
| **Data** | Browse every dataset |
| **About** | Project structure, commands, credits |
""")
