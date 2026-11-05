"""Shared pieces of the Streamlit website (app.py + app_pages/): paths, cached models, final numbers, drawings."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
if os.path.join(ROOT, "src", "shapes") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src", "shapes"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import designer as dz  # noqa: E402

BLUE, ORANGE, TEAL, GREY = "#2b5bd0", "#cf6d1e", "#0d7a84", "#6b7280"
FIG = "results/optionb/figs"

PRESETS = {"rounded triangles (old data)": ([47.4, -7.6, 0, 0.1, 0], [34.0, -5.4, 0, 0.1, 0]),
           "circles": ([55, 0, 0, 0, 0], [38, 0, 0, 0, 0]),
           "sharper triangles": ([50, -14, 0, 5, 0], [34, -9, 0, 3, 0]),
           "rotated triangle + hexagon": ([47.4, -3.8, -6.6, 0, 0], [36, 0, 0, -3, 0]),
           "trefoil + circle": ([45, 9, 0, 3, 0], [35, 0, 0, 0, 0])}
RANGES = {"r0": (25.0, 75.0), "c3": (-26.0, 26.0), "s3": (-26.0, 26.0), "c6": (-9.0, 9.0), "s6": (-9.0, 9.0)}


@st.cache_resource(show_spinner="Loading the trained networks…")
def bundle():
    return dz.load_bundle()


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except OSError:
        return None


@st.cache_data(show_spinner=False)
def numbers():
    """Every final number the pages show, read from the saved results."""
    out = {"final": load_json("results/optionb/final_comsol.json"),
           "seeds": load_json("results/optionb/seed_summary.json"),
           "main": load_json("results/optionb/results.json"),
           "conv": load_json("results/optionb/pwe_convergence.json"),
           "verify": None}
    if os.path.exists("results/optionb/comsol_verify.csv"):
        out["verify"] = pd.read_csv("results/optionb/comsol_verify.csv")
    return out


def fmt2(v):
    return " / ".join(f"{x:.2f}" for x in v)


def holes_fig(hA, hB, size=2.8, title=None):
    fig, ax = plt.subplots(figsize=(size, size))
    dz.draw_holes(ax, hA, hB, title=title)
    fig.tight_layout(pad=0.2)
    return fig


def cell_fig(p, size=2.8, title=None):
    fig, ax = plt.subplots(figsize=(size, size))
    dz.draw_cell(ax, p, title=title)
    fig.tight_layout(pad=0.2)
    return fig


def band_fig_holes(hA, hB, nk=31):
    ks = np.linspace(0, 3, nk)
    b = dz.solver().bands(hA, hB, ks)
    fl, fu = b[:, 0].max(), b[:, 1].min()
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    if fu > fl:
        ax.axhspan(fl, fu, color=TEAL, alpha=0.15, lw=0)
    ax.plot(ks, b[:, 0], color=BLUE, lw=2, label="band 1")
    ax.plot(ks, b[:, 1], color=ORANGE, lw=2, label="band 2")
    for k in (1, 2):
        ax.axvline(k, color="0.8", ls=":", lw=1)
    ax.set_xticks([0, 1, 2, 3], ["Γ", "M", "K", "Γ"])
    ax.set_ylabel("frequency (GHz)")
    ax.set_ylim(0, 560)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig, fl, fu


def band_fig(p):
    return band_fig_holes(*dz.p_to_holes(np.asarray(p, float)))


def badges(info):
    def b(ok, text):
        return f":{'green' if ok else 'red'}[{'✔' if ok else '✘'} {text}]"
    reach = max(info["rmax_A"], info["rmax_B"])
    curv = min(info.get("curv_radius_A", np.inf), info.get("curv_radius_B", np.inf))
    return "  ".join([b(info["min_wall"] >= 25, f"wall {info['min_wall']:.0f} µm"),
                      b(min(info["rmin_A"], info["rmin_B"]) >= 10, f"min radius {min(info['rmin_A'], info['rmin_B']):.0f} µm"),
                      b(curv >= 8, "corners ok" if curv >= 8 else f"corner radius {curv:.0f} µm"),
                      b(reach <= dz.COMSOL_REACH, f"reach {reach:.0f} µm ({'fits' if reach <= dz.COMSOL_REACH else 'crosses'} COMSOL cell)")])


def expected_error(model):
    """Typical error of a forward model, from the final results: (mean, 95th percentile) per edge, and what it is
    measured against."""
    N = numbers()
    f = N["final"]
    if model == "pwe":
        if f:
            a = f["accuracy"]["forward_pwe_test"]
            return a["mae"], a["p95"], "PWE answers, 600 unseen PWE cells"
        return [0.14, 0.18], None, "PWE answers, 600 unseen PWE cells"
    if not f:
        return None, None, ""
    key = {"corr": "nn_pwe+corr", "comsol": "nn_comsol_all"}[model]
    a = f["freeform_comparison"]["methods"][key]["accuracy"]
    return a["mae"], a["p95"], f"COMSOL answers, {f['freeform_comparison']['n_free']} freeform COMSOL cells (cross-validated)"


ACCURACY_TEXT = """
**Forward model** (shape → f_lower, f_upper). For a new shape nobody knows the true answer until it is simulated,
so a single prediction cannot carry an exact accuracy. What it carries instead:
- the **expected accuracy**, measured once on shapes the model never saw (mean absolute error and the 95 % bound, in GHz);
- an **uncertainty** for this particular shape: how much the 5 networks of the ensemble disagree;
- on request, the **true error** for this shape: simulate it with PWE (about 0.2 s) and compare.

**Inverse model** (target → shapes). Each proposed shape is simulated with PWE right away, so every design does come
with a real error: max(|f_lower − target|, |f_upper − target|), the worse of the two edges. A design **succeeds** when
that error is below **1 GHz**. Over 100 unseen targets this gives the **success rate**. The final proof is
simulating designs in **COMSOL** and counting how many land within 1 GHz of their target.

**Why these rules**
- **GHz, not percent:** at 300 GHz even a 3 GHz miss looks like a tiny 1 %, so percentages hide real errors.
  R² is also misleading: it mostly reflects how spread out the data are.
- **The worse of the two edges:** a band gap is only useful if both edges are right.
- **1 GHz tolerance:** about 0.3 % of 300 GHz, below half the narrowest IEEE 802.15.3d channel (2.16 GHz),
  and about the size of the remaining PWE–COMSOL difference; demanding more would be false precision.
- **Unseen data and an independent simulator:** a model is never judged on shapes it learned from, and a design is
  never judged by the network that proposed it.
"""
