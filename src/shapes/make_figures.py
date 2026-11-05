"""Figures for the option-B study guide -> results/optionb/figs/*.png

Run from the Project folder:  python src/shapes/make_figures.py
A figure is skipped (with a message) when its input file does not exist yet.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from geometry import A1, A2, SITE_A, SITE_B, boundary, fourier_features, radius  # noqa: E402

FIG = "results/optionb/figs"
RES = "results/optionb"
C = {"blue": "#2563a8", "orange": "#d9711c", "green": "#2f8f4e", "purple": "#7a4fb5", "red": "#c23b3b",
     "gray": "#6b7280", "si": "#cfd6df"}
DS_COLOR = {"pwe_fourier": C["blue"], "pwe_library": C["purple"], "comsol_rounded": C["orange"],
            "comsol_sharp": C["green"], "comsol_circle": C["red"]}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight", "axes.titlesize": 10})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("wrote", name)


def draw_cell(ax, holeA, holeB, title=None, n_rep=1, show_cell=True, labels=True):
    """Silicon (grey) with air holes (white), tiled around one rhombic unit cell."""
    corners = np.array([[0, 0], A1, A1 + A2, A2])
    lo, hi = -0.35 * A1[0], 1.85 * A1[0]
    ax.add_patch(plt.Rectangle((lo, -0.45 * A1[0]), hi - lo, 1.75 * A1[0], color=C["si"], zorder=0))
    for i in range(-n_rep - 1, n_rep + 2):
        for j in range(-n_rep - 1, n_rep + 2):
            off = i * A1 + j * A2
            for hole, site in ((holeA, "A"), (holeB, "B")):
                ax.add_patch(Polygon(boundary(*hole, site) + off, closed=True, fc="white", ec="#333", lw=0.6, zorder=1))
    if show_cell:
        ax.add_patch(Polygon(corners, closed=True, fill=False, ec=C["blue"], lw=1.2, ls="--", zorder=2))
    if labels:
        ax.text(*SITE_A, "A", ha="center", va="center", fontsize=8, color=C["red"], zorder=3)
        ax.text(*SITE_B, "B", ha="center", va="center", fontsize=8, color=C["red"], zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.45 * A1[0], 1.3 * A1[0])
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)


# ----------------------------------------------------------------------------- 1 unit cells
def fig_unit_cells():
    cells = [(("rounded", {"L": 168.0}), ("rounded", {"L": 119.0}), "(a) rounded triangles\nL1 = 168, L2 = 119 um (old data)"),
             (("sharp", {"L": 168.0}), ("sharp", {"L": 119.0}), "(b) sharp triangles\nL1 = 168, L2 = 119 um (old data)"),
             (("circle", {"R": 60.7}), ("circle", {"R": 32.7}), "(c) circles (round cell)\nR1 = 60.7, R2 = 32.7 um"),
             (("fourier", {"r0": 55, "c3": -9, "s3": 5, "c6": 4, "s6": -2}),
              ("fourier", {"r0": 38, "c3": 7, "s3": 3, "c6": -2}), "(d) freeform C3 Fourier holes\n(option B)")]
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.2))
    for ax, (a, b, t) in zip(axs, cells):
        draw_cell(ax, a, b, t)
    save(fig, "unit_cells.png")


# ----------------------------------------------------------------------------- 2 Fourier descriptor
def _recon(fam, par, nmax):
    f = fourier_features(fam, par)
    th = np.linspace(0, 2 * np.pi, 721)
    r = np.full_like(th, f[0])
    for i, n in enumerate((3, 6, 9, 12)):
        if n <= nmax:
            r += f[1 + 2 * i] * np.cos(n * th) + f[2 + 2 * i] * np.sin(n * th)
    return th, r


def fig_fourier():
    fig = plt.figure(figsize=(11, 5.6))
    th = np.linspace(0, 2 * np.pi, 721)
    for k, (fam, title) in enumerate((("rounded", "rounded triangle, L = 140 um"), ("sharp", "sharp triangle, L = 140 um"))):
        ax = fig.add_subplot(2, 4, 1 + 2 * k)
        ax.plot(np.degrees(th), radius(fam, {"L": 140}, th), color="k", lw=1.6, label="exact r(theta)")
        for nmax, col in ((3, C["orange"]), (6, C["blue"]), (12, C["green"])):
            ax.plot(np.degrees(th), _recon(fam, {"L": 140}, nmax)[1], color=col, lw=1, ls="--", label=f"harmonics <= {nmax}")
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
        ax.set_xlabel("theta (deg, 0 = towards the other hole)")
        ax.set_ylabel("r (um)")
        ax.set_title(title)
        if k == 0:
            ax.legend(fontsize=7, frameon=False)
        ax2 = fig.add_subplot(2, 4, 2 + 2 * k, projection="polar")
        ax2.plot(th, radius(fam, {"L": 140}, th), color="k", lw=1.6)
        ax2.plot(*_recon(fam, {"L": 140}, 6), color=C["blue"], ls="--", lw=1)
        ax2.plot(*_recon(fam, {"L": 140}, 12), color=C["green"], ls="--", lw=1)
        ax2.set_yticklabels([])
        ax2.set_title("shape in the plane", fontsize=8)
    shapes = [("circle", {"r0": 45}), ("rounded-triangle-like", {"r0": 45, "c3": -7}),
              ("sharp-triangle-like", {"r0": 45, "c3": -12, "c6": 4.5}), ("hexagon-like", {"r0": 45, "c6": -3}),
              ("rotated triangle", {"r0": 45, "c3": -6, "s3": 7}), ("trefoil", {"r0": 45, "c3": 4, "c6": 5}),
              ("star-ish", {"r0": 45, "c3": -14, "s6": 5})]
    for i, (name, par) in enumerate(shapes):
        ax = fig.add_axes([0.02 + i * 0.122, 0.0, 0.11, 0.33], projection="polar")
        ax.fill(th, radius("fourier", par, th), color=C["blue"], alpha=0.25)
        ax.plot(th, radius("fourier", par, th), color=C["blue"], lw=1.2)
        ax.set_ylim(0, 70)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_title(name, fontsize=7.5)
    ax = fig.add_axes([0.02 + 7 * 0.122, 0.0, 0.11, 0.33], projection="polar")
    r_bad = 45 + 9 * np.cos(2 * th)
    ax.fill(th, r_bad, color=C["red"], alpha=0.25)
    ax.plot(th, r_bad, color=C["red"], lw=1.2)
    ax.set_ylim(0, 70)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title("cos 2theta: breaks C3 (not allowed)", fontsize=7.5, color=C["red"])
    fig.subplots_adjust(bottom=0.42, wspace=0.45)
    save(fig, "fourier_descriptor.png")


# ----------------------------------------------------------------------------- 3 PWE validation
def fig_pwe_validation():
    path = "data/pwe/validation.csv"
    if not os.path.exists(path):
        return print("skip pwe_validation (run validate_pwe.py)")
    v = pd.read_csv(path)
    eps = json.load(open("results/pwe_validation.json"))["eps_si"]
    from pwe import Solver
    b = pd.read_csv("data/comsol/bands.csv")
    g = b[(b.corner == 1) & (abs(b.L1_um - 168.01) < 0.01) & (abs(b.L2_um - 119.01) < 0.01)].sort_values("k")
    ks = np.linspace(0.05, 2.95, 59)
    s = Solver(eps_si=eps)
    pw = s.bands(("rounded", {"L": 168.01}), ("rounded", {"L": 119.01}), ks)
    pw11 = Solver(eps_si=11.7).bands(("rounded", {"L": 168.01}), ("rounded", {"L": 119.01}), ks)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    ax = axs[0]
    for j in (1, 2):
        ax.plot(g.k, g[f"f{j}_GHz"], color="k", lw=2, label="COMSOL" if j == 1 else None)
        ax.plot(ks, pw[:, j - 1], color=C["orange"], ls="--", lw=1.4, label=f"PWE, eps = {eps:.2f}" if j == 1 else None)
        ax.plot(ks, pw11[:, j - 1], color=C["gray"], ls=":", lw=1.2, label="PWE, eps = 11.7" if j == 1 else None)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Gamma", "M", "K", "Gamma"])
    ax.set_ylim(150, 450)
    ax.set_ylabel("frequency (GHz)")
    ax.set_title("(a) bands, rounded L1 = 168, L2 = 119 um")
    ax.legend(fontsize=7, frameon=False, loc="upper center")
    for ax, q, t in ((axs[1], ("f_lower", "f_upper"), "(b) band edges, all gapped COMSOL cells"), (axs[2], ("gap",), "(c) gap width")):
        for corner, col, lab in ((1, C["orange"], "rounded"), (0, C["green"], "sharp")):
            w = v[(v.corner == corner) & (v.gap > 0.1)]
            for qq, mk in zip(q, ("o", "^")):
                ax.scatter(w[qq], w[f"{qq}_pwe"], s=10, color=col, marker=mk, alpha=0.8,
                           label=f"{lab} {qq}" if len(q) > 1 else lab)
        if os.path.exists("data/comsol/extra_circle.csv"):  # never used to fit eps: an out-of-family test
            w = pd.read_csv("data/comsol/extra_circle.csv")
            w = w[w.gap > 0.1].assign(gap_pwe=lambda x: x.f_upper_pwe - x.f_lower_pwe)
            for qq, mk in zip(q, ("o", "^")):
                ax.scatter(w[qq], w[f"{qq}_pwe"], s=12, color=C["red"], marker=mk, alpha=0.8,
                           label=f"circle {qq} (not fitted)" if len(q) > 1 else "circle (not fitted)")
        lim = ax.get_xlim()
        ax.plot(lim, lim, color="k", lw=0.8)
        ax.set_xlabel("COMSOL (GHz)")
        ax.set_ylabel("PWE (GHz)")
        ax.set_title(t)
        ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    save(fig, "pwe_validation.png")


# ----------------------------------------------------------------------------- 4 dataset
def fig_dataset():
    files = {"pwe_fourier": "data/pwe/fourier.csv", "pwe_library": "data/pwe/library.csv"}
    if not all(os.path.exists(f) for f in files.values()):
        return print("skip dataset (run gen_dataset.py)")
    d = {k: pd.read_csv(f) for k, f in files.items()}
    e = pd.read_csv("data/comsol/band_edges.csv")
    from dataset import holes
    fig = plt.figure(figsize=(11, 6.4))
    rows = pd.concat([d["pwe_fourier"].sample(6, random_state=3), d["pwe_library"].sample(6, random_state=4)])
    for i, (_, r) in enumerate(rows.iterrows()):
        ax = fig.add_axes([0.01 + (i % 6) * 0.165, 0.70 - (i // 6) * 0.30, 0.16, 0.28])
        hA, hB = holes(r)
        draw_cell(ax, hA, hB, n_rep=1, show_cell=False, labels=False)
        ax.set_title(f"{r.famA}/{r.famB}  gap {r.gap:.1f} GHz", fontsize=7)
    ax = fig.add_axes([0.06, 0.06, 0.4, 0.32])
    for k, x in d.items():
        ax.scatter(x.f_lower, x.f_upper, s=2, alpha=0.35, color=DS_COLOR[k], label=f"{k} ({len(x)})")
    for corner, k in ((1, "comsol_rounded"), (0, "comsol_sharp")):
        w = e[e.corner == corner]
        ax.scatter(w.f_lower, w.f_upper, s=9, color=DS_COLOR[k], label=f"{k} ({len(w)})", edgecolor="k", lw=0.3)
    circ = pd.read_csv("data/comsol/extra_circle.csv") if os.path.exists("data/comsol/extra_circle.csv") else None
    if circ is not None:
        ax.scatter(circ.f_lower, circ.f_upper, s=9, color=DS_COLOR["comsol_circle"], label=f"comsol_circle ({len(circ)})",
                   edgecolor="k", lw=0.3)
    lim = [260, 440]
    ax.plot(lim, lim, color="k", lw=0.8)
    ax.text(400, 385, "gap = 0", rotation=33, fontsize=7)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("f_lower (GHz)")
    ax.set_ylabel("f_upper (GHz)")
    ax.legend(fontsize=6.5, frameon=False, markerscale=3)
    ax = fig.add_axes([0.56, 0.06, 0.4, 0.32])
    bins = np.linspace(-30, 80, 45)
    for k, x in d.items():
        ax.hist(x.gap, bins=bins, alpha=0.5, color=DS_COLOR[k], label=k)
    gaps = e.gap if circ is None else pd.concat([e.gap, circ.gap])
    ax.hist(gaps, bins=bins, alpha=0.8, color=C["orange"], label=f"COMSOL (all {len(gaps)})")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("gap = f_upper - f_lower (GHz); < 0 means no complete gap")
    ax.set_ylabel("cells")
    ax.legend(fontsize=7, frameon=False)
    save(fig, "dataset.png")


# ----------------------------------------------------------------------------- 5 forward + mf
def fig_forward():
    p = f"{RES}/results.json"
    if not os.path.exists(p):
        return print("skip forward/mf/inverse figures (run run_optionb.py)")
    res = json.load(open(p))
    lc = res["learning_curve"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
    ax = axs[0]
    n = [r["n_train"] for r in lc]
    ax.plot(n, [np.mean(r["nn_mae"]) for r in lc], "o-", color=C["blue"], label="neural-network ensemble")
    ng = [r["n_train"] for r in lc if "gp_mae" in r]
    ax.plot(ng, [np.mean(r["gp_mae"]) for r in lc if "gp_mae" in r], "s--", color=C["orange"], label="Gaussian process")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training cells")
    ax.set_ylabel("test MAE, mean of both edges (GHz)")
    ax.set_title("(a) how much data is needed")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3, which="both")
    z = np.load(f"{RES}/forward_parity.npz", allow_pickle=True)
    ax = axs[1]
    for j, (lab, col) in enumerate((("f_lower", C["blue"]), ("f_upper", C["orange"]))):
        ax.scatter(z["y"][:, j], z["mu"][:, j], s=3, alpha=0.5, color=col, label=lab)
    lim = ax.get_xlim()
    ax.plot(lim, lim, "k", lw=0.8)
    ax.set_xlabel("PWE truth (GHz)")
    ax.set_ylabel("surrogate (GHz)")
    ft = res["forward_test"]
    ax.set_title(f"(b) held-out cells: MAE {ft['mae_all'][0]:.2f} / {ft['mae_all'][1]:.2f} GHz")
    ax.legend(fontsize=7, frameon=False, markerscale=3)
    m = np.load(f"{RES}/mf.npz", allow_pickle=True)
    ax = axs[2]
    err_raw = (m["pwe"] - m["y"])
    err_cor = (m["pwe"] + m["corr"] - m["y"])
    labels, data = [], []
    for d in np.unique(m["dataset"]):
        s = m["dataset"] == d
        for j, q in enumerate(("lower", "upper")):
            labels += [f"{d.replace('comsol_', '')}\n{q}\nraw", "corr."]
            data += [err_raw[s, j], err_cor[s, j]]
    bp = ax.boxplot(data, widths=0.6, showfliers=False, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(C["si"] if i % 2 == 0 else C["green"])
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=6)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("PWE - COMSOL (GHz)")
    ax.set_title("(c) multi-fidelity correction (10-fold CV)")
    fig.tight_layout()
    save(fig, "forward_mf.png")


def fig_inverse():
    p = f"{RES}/results.json"
    if not os.path.exists(p):
        return
    res = json.load(open(p))
    inv = res["inverse"]
    z = np.load(f"{RES}/inverse_errors.npz")
    names = ["tandem", "cvae", "multistart", "cvae_refine"]
    nice = {"tandem": "tandem NN", "cvae": "conditional VAE", "multistart": "multi-start\ngradient", "cvae_refine": "CVAE +\nrefine"}
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
    ax = axs[0]
    x = np.arange(len(names))
    for j, (lab, col) in enumerate((("f_lower", C["blue"]), ("f_upper", C["orange"]))):
        ax.bar(x + (j - 0.5) * 0.36, [inv[n]["verified_mae_GHz"][j] for n in names], 0.36, color=col, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([nice[n] for n in names], fontsize=7.5)
    ax.set_ylabel("PWE-verified error (GHz)")
    ax.set_title("(a) mean error over 100 targets")
    ax.legend(fontsize=7, frameon=False)
    ax = axs[1]
    for n, col in zip(names, (C["gray"], C["purple"], C["blue"], C["green"])):
        e = np.nanmax(z[n], 1)
        e = np.sort(np.where(np.isnan(e), 1e3, e))
        ax.plot(e, np.arange(1, len(e) + 1) / len(e), color=col, lw=1.5, label=nice[n].replace("\n", " "))
    ax.axvline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(0.01, 30)
    ax.set_xlabel("worse of the two edge errors (GHz)")
    ax.set_ylabel("fraction of targets")
    ax.set_title("(b) cumulative error")
    ax.legend(fontsize=7, frameon=False)
    ax = axs[2]
    vals = [inv[n].get("distinct_verified_designs_per_target", 1.0 if n == "tandem" else 0) for n in names]
    ax.bar(x, vals, color=[C["gray"], C["purple"], C["blue"], C["green"]])
    ax.set_xticks(x)
    ax.set_xticklabels([nice[n] for n in names], fontsize=7.5)
    ax.set_ylabel("distinct verified designs (max 8)")
    ax.set_title("(c) how many different answers")
    fig.tight_layout()
    save(fig, "inverse.png")


def fig_showcase():
    p = f"{RES}/results.json"
    if not os.path.exists(p):
        return
    sc = json.load(open(p)).get("showcase")
    if not sc or not sc["designs"]:
        return print("skip showcase (no designs)")
    from models import p_to_holes
    ds = sc["designs"][:8]
    ncol = 4
    nrow = int(np.ceil(len(ds) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(11, 3.0 * nrow))
    axs = np.atleast_2d(axs)
    for ax in axs.flat:
        ax.axis("off")
    for ax, d in zip(axs.flat, ds):
        hA, hB = p_to_holes(np.array(d["p"]))
        draw_cell(ax, hA, hB, n_rep=1, labels=False)
        ax.set_title(f"PWE: {d['pwe'][0]:.1f} / {d['pwe'][1]:.1f} GHz, wall {d['min_wall']:.0f} um", fontsize=7.5)
    fig.suptitle(f"Different shapes, same target band edges {sc['target'][0]:.0f} / {sc['target'][1]:.0f} GHz "
                 f"({sc['n_verified_within_1GHz']} distinct designs verified within 1 GHz)", fontsize=10)
    save(fig, "showcase.png")


# ----------------------------------------------------------------------------- 6 pipeline diagram
def _box(ax, xy, w, h, text, col):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02", fc=col, ec="k", lw=0.8, alpha=0.9))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=7.6, wrap=True)


def _arrow(ax, a, b, text=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, lw=1, color="k"))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.02, text, fontsize=6.5, ha="center", color=C["gray"])


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(-0.015, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    hi, lo, ml, ve = "#fde2c4", "#d6e6f7", "#dff0e3", "#eadff5"
    _box(ax, (0.01, 0.70), 0.17, 0.22, "COMSOL datasets\nrounded (64) + sharp (64)\n+ circle (49)\nhigh fidelity, slow", hi)
    _box(ax, (0.24, 0.70), 0.17, 0.22, "PWE solver\n(2D plane waves)\ncalibrate eps = 8.91\nerror < 1 GHz", lo)
    _box(ax, (0.47, 0.70), 0.17, 0.22, "C3 Fourier shapes\n+ shape library\nvalidity checks\n6,000 cells, M & K", lo)
    _box(ax, (0.70, 0.70), 0.17, 0.22, "forward surrogate\n18 features -> edges\nNN ensemble (vs GP)", ml)
    _box(ax, (0.70, 0.30), 0.17, 0.24, "inverse design\ntandem NN | CVAE |\nmulti-start gradient |\nCVAE + refine", ml)
    _box(ax, (0.47, 0.30), 0.17, 0.24, "validity check +\nPWE verification\n(+ multi-fidelity\ncorrection)", ve)
    _box(ax, (0.24, 0.30), 0.17, 0.24, "COMSOL verification\nof the best designs\n(showcase file)", hi)
    _box(ax, (0.01, 0.30), 0.17, 0.24, "add to data,\nretrain\n(active learning loop)", ve)
    _arrow(ax, (0.18, 0.81), (0.24, 0.81))
    _arrow(ax, (0.41, 0.81), (0.47, 0.81))
    _arrow(ax, (0.64, 0.81), (0.70, 0.81))
    _arrow(ax, (0.785, 0.70), (0.785, 0.54))
    _arrow(ax, (0.70, 0.42), (0.64, 0.42))
    _arrow(ax, (0.47, 0.42), (0.41, 0.42))
    _arrow(ax, (0.24, 0.42), (0.18, 0.42))
    _arrow(ax, (0.095, 0.54), (0.095, 0.70))
    ax.text(0.5, 0.12, "target band edges (f_lower, f_upper)  ->  several different fabricable hole shapes that give them",
            ha="center", fontsize=9, style="italic")
    _arrow(ax, (0.95, 0.42), (0.87, 0.42))  # the target enters the inverse-design box
    ax.text(0.955, 0.42, "target\n(f_lower, f_upper)", ha="left", va="center", fontsize=7.6, style="italic")
    save(fig, "pipeline.png")


# ----------------------------------------------------------------------------- why only M and K
def fig_mk_shortcut():
    path = os.path.join(RES, "mk_examples.npz")
    if not os.path.exists(path):
        return print("skip mk_shortcut (run check_mk_shortcut.py)")
    import matplotlib.tri as mtri
    from geometry import A_UM
    z = np.load(path)
    ks, paths, names = z["ks"], z["paths"], [str(n) for n in z["names"]]
    cells = [[tuple(h) for h in json.loads(str(c))] for c in z["cells"]]
    b = pd.read_csv("data/comsol/bands.csv")
    fig = plt.figure(figsize=(11, 6.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.42, wspace=0.3)
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        bd = paths[i]
        lo, up = bd[:, 0].max(), bd[:, 1].min()
        if up > lo:
            ax.axhspan(lo, up, color=C["green"], alpha=0.15, lw=0)
            ax.text(2.97, (lo + up) / 2, f"gap {up - lo:.1f} GHz", ha="right", va="center", fontsize=7, color=C["green"])
        if i == 0:  # the same cell from COMSOL, underneath
            L1, L2 = cells[0][0][1]["L"], cells[0][1][1]["L"]
            g = b[(b.corner == 1) & np.isclose(b.L1_um, L1) & np.isclose(b.L2_um, L2)].sort_values("k")
            ax.plot(g.k, g.f1_GHz, color="k", lw=3, alpha=0.25, label="COMSOL")
            ax.plot(g.k, g.f2_GHz, color="k", lw=3, alpha=0.25)
        ax.plot(ks, bd[:, 0], color=C["blue"], lw=1.3, label="band 1 (PWE)")
        ax.plot(ks, bd[:, 1], color=C["orange"], lw=1.3, label="band 2 (PWE)")
        i1, i2 = bd[:, 0].argmax(), bd[:, 1].argmin()
        ax.plot(ks[i1], bd[i1, 0], "v", color=C["blue"], ms=8, mec="k", mew=0.6, zorder=5)
        ax.plot(ks[i2], bd[i2, 1], "^", color=C["orange"], ms=8, mec="k", mew=0.6, zorder=5)
        for x in (1, 2):
            ax.axvline(x, color=C["gray"], lw=0.6, ls=":")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["Γ", "M\n(k = 1)", "K\n(k = 2)", "Γ"])
        ax.set_xlim(0, 3)
        ax.set_ylim(max(0, lo - 140), up + 140)
        ax.set_ylabel("frequency (GHz)")
        ax.set_title(f"({'abc'[i]}) {names[i]}", fontsize=8.5)
        if i == 0:
            ax.legend(fontsize=7, frameon=False, loc="lower right")

    # band maps over the irreducible wedge of the freeform cell (k in units of 2*pi/a)
    s = A_UM / (2 * np.pi)
    w, wb = z["wedge"] * s, z["wedge_bands"]
    K, M = z["K"] * s, z["M"] * s
    K2 = np.array([K[0] / 2, K[0] * np.sqrt(3) / 2])
    hexagon = np.array([[K[0] * np.cos(t), K[0] * np.sin(t)] for t in np.deg2rad(np.arange(0, 361, 60))])
    tri = mtri.Triangulation(w[:, 0], w[:, 1])
    for j, (band, mark, lab) in enumerate(((0, "v", "band 1"), (1, "^", "band 2"))):
        ax = fig.add_subplot(gs[1, j])
        tp = ax.tripcolor(tri, wb[:, band], shading="gouraud", cmap="viridis")
        ax.plot(hexagon[:, 0], hexagon[:, 1], color=C["gray"], lw=0.8)
        ax.plot([0, K[0], K2[0], 0], [0, K[1], K2[1], 0], color="k", lw=0.8)
        idx = wb[:, band].argmax() if band == 0 else wb[:, band].argmin()
        ax.plot(*w[idx], mark, color="white", ms=10, mec="k", mew=0.8, zorder=5)
        for p, t, off in ((np.zeros(2), "Γ", (-0.07, -0.03)), (K, "K", (0.03, -0.03)), (M, "M", (0.045, -0.01)),
                          (K2, "K′", (-0.16, -0.01))):
            ax.text(p[0] + off[0], p[1] + off[1], t, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-0.75, 0.8)
        ax.set_ylim(-0.7, 0.72)
        ax.set_xlabel("k_x  (2π/a)")
        if j == 0:
            ax.set_ylabel("k_y  (2π/a)")
        ext = "▼ = highest" if band == 0 else "▲ = lowest"
        ax.set_title(f"({'de'[j]}) {lab} over the wedge, {ext}", fontsize=8.5)
        fig.colorbar(tp, ax=ax, shrink=0.8, label=f"{lab} (GHz)")
    ax = fig.add_subplot(gs[1, 2])
    draw_cell(ax, *cells[2], title="(f) the freeform cell of (c), (d), (e)")
    save(fig, "mk_shortcut.png")


def fig_comsol_vs_pwe():
    p = f"{RES}/comsol_vs_pwe.json"
    if not os.path.exists(p):
        return print("skip comsol_vs_pwe (run compare_training.py)")
    r = json.load(open(p))
    nice = {"pwe_solver": "PWE solver\n(no ML)", "nn_pwe": "NN on\nPWE", "gp_comsol_free": "GP on COMSOL\nfreeform",
            "gp_comsol_all": "GP on all\nCOMSOL", "nn_comsol_free": "NN on COMSOL\nfreeform", "nn_comsol_all": "NN on all\nCOMSOL",
            "nn_pwe+corr": "NN on PWE\n+ correction", "solver+corr": "PWE solver\n+ correction"}
    col = {"pwe_solver": C["blue"], "nn_pwe": C["blue"], "gp_comsol_free": C["orange"], "gp_comsol_all": C["orange"],
           "nn_comsol_free": C["orange"], "nn_comsol_all": C["orange"], "nn_pwe+corr": C["green"], "solver+corr": C["green"]}
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    ms = list(r["methods"])
    x = np.arange(len(ms))
    lo = [r["methods"][m]["mae_f_lower"] for m in ms]
    up = [r["methods"][m]["mae_f_upper"] for m in ms]
    ax.bar(x - 0.18, lo, 0.36, color=[col[m] for m in ms], alpha=0.95, label="f_lower")
    ax.bar(x + 0.18, up, 0.36, color=[col[m] for m in ms], alpha=0.5, hatch="//", label="f_upper")
    ax.set_xticks(x)
    ax.set_xticklabels([nice[m] for m in ms], fontsize=7)
    ax.set_ylabel("MAE vs COMSOL (GHz)")
    ax.set_title(f"(a) predicting {r['n_free']} COMSOL freeform cells (blue: PWE only, orange: COMSOL only, "
                 "green: both)", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False)
    ax = axs[1]
    for m, c in (("gp_comsol_all", C["orange"]), ("nn_pwe+corr", C["green"]), ("solver+corr", C["purple"])):
        pts = r["learning_curve"][m]
        ax.plot([q["n_free_train"] for q in pts], [np.mean(q["mae"]) for q in pts], "o-", color=c, label=nice[m].replace("\n", " "))
    ax.axhline(np.mean([r["methods"]["pwe_solver"]["mae_f_lower"], r["methods"]["pwe_solver"]["mae_f_upper"]]),
               color=C["blue"], ls="--", lw=1, label="PWE solver alone")
    ax.set_xlabel("COMSOL freeform cells used for training")
    ax.set_ylabel("MAE, mean of both edges (GHz)")
    ax.set_yscale("log")
    ax.set_title("(b) how many COMSOL freeform runs are needed", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "comsol_vs_pwe.png")


if __name__ == "__main__":
    fig_unit_cells()
    fig_fourier()
    fig_pipeline()
    fig_pwe_validation()
    fig_mk_shortcut()
    fig_comsol_vs_pwe()
    fig_dataset()
    fig_forward()
    fig_inverse()
    fig_showcase()
