"""Figures for the final report, poster and websites (all COMSOL data):
  final_pwe_vs_comsol.png   PWE against COMSOL for every gapped COMSOL cell, by shape family
  comsol_verify.png         the AI designs simulated in COMSOL: COMSOL result against target
  accuracy_cdf.png          share of predictions / designs within a given error (forward, correction, inverse)
Also refreshes comsol_vs_pwe.png (make_figures.fig_comsol_vs_pwe) and copies every report figure to Report/final/figures.

Run from the Project folder after final_comsol_study.py:  python src/shapes/make_final_figures.py
"""
import glob
import json
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import make_figures  # noqa: E402

R, FIG = "results/optionb", "results/optionb/figs"
COL = {"rounded triangle": "#cf6d1e", "sharp triangle": "#23884f", "circle": "#c23b3b", "freeform": "#2b5bd0",
       "freeform (AI designs)": "#7b3fbf"}
FAMILY = {"comsol_fourier": "freeform", "comsol_verify": "freeform (AI designs)", "comsol_circle": "circle",
          "comsol_circle_lhs": "circle", "comsol_rounded_lhs": "rounded triangle", "comsol_rounded_lhs2": "rounded triangle",
          "comsol_sharp_lhs": "sharp triangle"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def comsol_with_pwe():
    parts = []
    for f in glob.glob("data/comsol/extra_*.csv"):
        t = pd.read_csv(f)
        if "f_lower_pwe" in t:
            parts.append(t[["id", "dataset", "f_lower", "f_upper", "f_lower_pwe", "f_upper_pwe"]])
    t = pd.concat(parts, ignore_index=True)
    cache = f"{R}/comsol_pwe_cache.csv"
    if os.path.exists(cache):          # the inherited 8 x 8 grids (PWE computed by final_comsol_study.py)
        from dataset import load_comsol
        com = load_comsol()
        grid = com[com.dataset.isin(["comsol_rounded", "comsol_sharp"])][["id", "dataset", "f_lower", "f_upper"]]
        grid = grid.merge(pd.read_csv(cache), on="id")
        t = pd.concat([t, grid], ignore_index=True)
    t["family"] = t.dataset.map(FAMILY).fillna(t.dataset.str.replace("comsol_", "").map(
        {"rounded": "rounded triangle", "sharp": "sharp triangle"}))
    return t[(t.f_upper - t.f_lower) > 0.1]


def fig_pwe_vs_comsol():
    t = comsol_with_pwe()
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    for fam, g in t.groupby("family"):
        for j, e in enumerate(("f_lower", "f_upper")):
            axs[j].scatter(g[e], g[e + "_pwe"], s=4, alpha=0.5, color=COL.get(fam, "grey"), label=f"{fam} ({len(g)})")
        axs[2].scatter(g.f_upper - g.f_lower, g.f_upper_pwe - g.f_lower_pwe, s=4, alpha=0.5, color=COL.get(fam, "grey"))
    for ax, name in zip(axs, ("(a) f_lower", "(b) f_upper", "(c) gap width")):
        lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, "k-", lw=0.8)
        ax.set_xlabel("COMSOL (GHz)")
        ax.set_ylabel("PWE (GHz)")
        ax.set_title(name)
    axs[0].legend(fontsize=7, frameon=False, markerscale=3)
    fig.suptitle(f"PWE against COMSOL on {len(t):,} gapped COMSOL cells", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/final_pwe_vs_comsol.png", dpi=160)
    plt.close(fig)


def fig_verify():
    p = f"{R}/comsol_verify.csv"
    if not os.path.exists(p):
        return
    v = pd.read_csv(p)
    lo = v["COMSOL - target lower"].to_numpy()
    up = v["COMSOL - target upper"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    x = np.arange(len(v))
    ax.axhspan(-1, 1, color="#0d7a84", alpha=0.12, lw=0, label="within 1 GHz")
    ax.scatter(x, lo, color="#2b5bd0", label="f_lower: COMSOL − target", zorder=3)
    ax.scatter(x, up, color="#cf6d1e", marker="s", label="f_upper: COMSOL − target", zorder=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, [f"{t}\n{'chk' if k != 'new design' else ''}" for t, k in zip(v.target, v.kind)], fontsize=6.5)
    ax.set_ylabel("error (GHz)")
    w = np.maximum(np.abs(lo), np.abs(up))
    ax.set_title(f"AI designs simulated in COMSOL: {int((w < 1).sum())} of {len(v)} within 1 GHz (worst {w.max():.2f} GHz)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{FIG}/comsol_verify.png", dpi=160)
    plt.close(fig)


def fig_accuracy():
    F = json.load(open(f"{R}/final_comsol.json"))
    fp = np.load(f"{R}/forward_parity.npz", allow_pickle=True)
    iv = np.load(f"{R}/inverse_errors.npz", allow_pickle=True)
    cv = np.load(f"{R}/comsol_vs_pwe.npz", allow_pickle=True)
    curves = [("forward network vs PWE (600 test cells)", np.abs(fp["mu"] - fp["y"]).max(1), "#2b5bd0"),
              ("PWE solver vs COMSOL (freeform)", np.abs(cv["pwe_solver"] - cv["y"]).max(1), "#6b7280"),
              ("PWE solver + correction vs COMSOL (freeform)", np.abs(cv["solver_plus_corr"] - cv["y"]).max(1), "#0d7a84"),
              ("tandem designs, PWE-verified (100 targets)", np.nanmax(iv["tandem"], 1), "#cf6d1e"),
              ("CVAE + refine designs, PWE-verified", np.nanmax(iv["cvae_refine"], 1), "#7b3fbf")]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for lab, e, c in curves:
        e = np.sort(e[~np.isnan(e)])
        ax.plot(e, np.arange(1, len(e) + 1) / len(e), color=c, lw=1.8, label=lab)
    ax.axvline(1.0, color="k", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("worse of the two edge errors (GHz)")
    ax.set_ylabel("share of cells / targets")
    ax.set_title("How often is the error below a given value? (dotted: 1 GHz criterion)")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/accuracy_cdf.png", dpi=160)
    plt.close(fig)


def main():
    fig_pwe_vs_comsol()
    fig_verify()
    make_figures.fig_comsol_vs_pwe()
    fig_accuracy()
    dst = "../Report/final/figures"
    os.makedirs(dst, exist_ok=True)
    for f in ("pipeline.png", "comsol_vs_pwe.png", "inverse.png", "fourier_descriptor.png", "showcase.png",
              "final_pwe_vs_comsol.png", "comsol_verify.png", "accuracy_cdf.png"):
        if os.path.exists(f"{FIG}/{f}"):
            shutil.copy(f"{FIG}/{f}", f"{dst}/{f}")
    print("figures written and copied to", dst)


if __name__ == "__main__":
    main()
