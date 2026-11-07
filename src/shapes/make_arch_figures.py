"""Architecture diagrams of the networks, drawn from the real layer sizes in models.py.

Run from the Project folder:  python src/shapes/make_arch_figures.py
Writes results/optionb/figs/arch_tandem.png, arch_forward.png, arch_cvae.png, arch_all.png (stacked, for the report).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(__file__))
import models as M  # noqa: E402

OUT = "results/optionb/figs"
TEAL, TEAL_L, GREY, GREY_L, AMBER, INK, MUTED = "#0d7a84", "#d7eef0", "#8a949c", "#eceff1", "#e89b2d", "#16202a", "#566271"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def n_params(i, o, h=256, depth=3):
    return sum(p.numel() for p in M.mlp(i, o, h, depth).parameters())


def column(ax, x, size, y0, h, color, label, max_show=7, r=0.085, sub=None):
    """A layer as a column of neurons (at most max_show drawn, with a gap marker)."""
    shown = min(size, max_show)
    ys = [y0 + h / 2 + (k - (shown - 1) / 2) * (h / max(shown, 4)) for k in range(shown)]
    gap = size > max_show
    for k, y in enumerate(ys):
        if gap and k == shown // 2:
            ax.text(x, y, "⋮", ha="center", va="center", fontsize=13, color=color)
            continue
        ax.add_patch(plt.Circle((x, y), r, facecolor="white", edgecolor=color, lw=1.6, zorder=3))
    ax.text(x, y0 - 0.12, label, ha="center", va="top", fontsize=9.5, color=INK, fontweight="bold")
    if sub:
        ax.text(x, y0 - 0.42, sub, ha="center", va="top", fontsize=8.5, color=MUTED)
    return [y for k, y in enumerate(ys) if not (gap and k == shown // 2)]


def connect(ax, x1, ys1, x2, ys2, color):
    for a in ys1:
        for b in ys2:
            ax.plot([x1, x2], [a, b], color=color, lw=0.35, alpha=0.45, zorder=1)


def mlp_drawing(ax, xs, sizes, labels, y0, h, color, subs=None):
    cols = []
    for k, (x, s, lab) in enumerate(zip(xs, sizes, labels)):
        cols.append(column(ax, x, s, y0, h, color, lab, sub=subs[k] if subs else None))
    for k in range(len(xs) - 1):
        connect(ax, xs[k], cols[k], xs[k + 1], cols[k + 1], color)
    return cols


def box(ax, x, y, w, h, text, fc, ec, fs=9.5, bold=False, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.3, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", zorder=4, linespacing=1.35)


def arrow(ax, p1, p2, color=INK, style="-|>", lw=1.4, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=5))


def polyarrow(ax, pts, color, ls="--", lw=1.1):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, zorder=5)
    arrow(ax, pts[-2], pts[-1], color=color, ls=ls, lw=lw)


def tandem(ax):
    ax.set_xlim(-0.3, 15.2)
    ax.set_ylim(-2.0, 4.45)
    ax.axis("off")
    y0, h = 0.25, 2.6
    # trainable inverse network
    ax.add_patch(FancyBboxPatch((0.15, -0.65), 4.55, 4.15, boxstyle="round,pad=0.02,rounding_size=0.15", fc=TEAL_L, ec=TEAL,
                                lw=1.2, alpha=0.55, zorder=0))
    ax.text(2.42, 3.33, f"INVERSE NETWORK  (trained)  ·  {n_params(2, 10):,} weights", ha="center", fontsize=9.5,
            color=TEAL, fontweight="bold")
    mlp_drawing(ax, [0.65, 1.5, 2.38, 3.26, 4.2], [2, 256, 256, 256, 10],
                ["target", "256", "256", "256", "10"], y0, h, TEAL,
                ["f_lower,\nf_upper", "SiLU", "SiLU", "SiLU", "raw outputs"])
    box(ax, 5.05, 1.05, 1.25, 1.0, "sigmoid\nu in (0, 1)", "white", TEAL, fs=9)
    box(ax, 6.55, 0.8, 1.68, 1.5, "map to the\nallowed box\nr0 = 25 + 50u\nc3 = 0.35 r0 (2u−1)", "white", TEAL, fs=8.1)
    arrow(ax, (4.35, 1.55), (5.03, 1.55))
    arrow(ax, (6.32, 1.55), (6.53, 1.55))
    box(ax, 8.45, 1.05, 1.15, 1.0, "design p\n10 numbers\n(µm)", AMBER + "33", AMBER, fs=8.8, bold=True)
    arrow(ax, (8.25, 1.55), (8.43, 1.55))
    # frozen forward ensemble
    ax.add_patch(FancyBboxPatch((9.95, -0.65), 3.65, 4.15, boxstyle="round,pad=0.02,rounding_size=0.15", fc=GREY_L, ec=GREY,
                                lw=1.2, zorder=0))
    for k in (2, 1):
        ax.add_patch(FancyBboxPatch((9.95 + 0.07 * k, -0.65 + 0.07 * k), 3.65, 4.15,
                                    boxstyle="round,pad=0.02,rounding_size=0.15", fc="none", ec=GREY, lw=0.8, zorder=0))
    ax.text(11.77, 3.33, "FORWARD NETWORK  (frozen)  × 5, averaged", ha="center", fontsize=9.5, color=MUTED,
            fontweight="bold")
    mlp_drawing(ax, [10.3, 11.1, 11.9, 12.7, 13.35], [18, 256, 256, 256, 2], ["18", "256", "256", "256", "2"],
                y0, h, GREY, ["features", "SiLU", "SiLU", "SiLU", "edges"])
    arrow(ax, (9.62, 1.55), (10.18, 1.55))
    box(ax, 13.85, 0.85, 1.25, 1.4, "loss\n(pred − target)²\n+ λ·penalty", "white", INK, fs=8.4)
    arrow(ax, (13.47, 1.55), (13.83, 1.55))
    # target goes to the loss; penalty from the design
    polyarrow(ax, [(0.65, 1.85), (-0.05, 1.85), (-0.05, 3.85), (14.47, 3.85), (14.47, 2.27)], TEAL)
    ax.text(7.2, 3.97, "the same target is what the loss compares the prediction with", ha="center", fontsize=8.5,
            color=TEAL)
    polyarrow(ax, [(9.03, 1.03), (9.03, -0.9), (14.47, -0.9), (14.47, 0.83)], AMBER)
    ax.text(11.75, -1.12, "fabrication penalty on the design (thin walls, tiny radii), λ = 0.001", ha="center",
            fontsize=8.5, color="#9a6110")
    # gradients
    arrow(ax, (14.45, -1.45), (2.4, -1.45), color="#c8553d", lw=1.6)
    ax.text(8.4, -1.85, "gradients flow back through the frozen forward network; only the inverse network's weights change",
            ha="center", fontsize=8.8, color="#c8553d")


def forward(ax):
    ax.set_xlim(-0.2, 11.5)
    ax.set_ylim(-1.3, 4.1)
    ax.axis("off")
    y0, h = 0.25, 2.6
    box(ax, 0.0, 0.95, 1.7, 1.2, "unit cell\n(any hole shape)", "white", INK, fs=9)
    box(ax, 2.05, 0.95, 1.6, 1.2, "18 Fourier\nfeatures\n(9 per hole, µm)", AMBER + "33", AMBER, fs=8.8, bold=True)
    arrow(ax, (1.72, 1.55), (2.03, 1.55))
    ax.add_patch(FancyBboxPatch((4.0, -0.6), 4.2, 4.05, boxstyle="round,pad=0.02,rounding_size=0.15", fc=TEAL_L, ec=TEAL,
                                lw=1.2, alpha=0.55, zorder=0))
    for k in (2, 1):
        ax.add_patch(FancyBboxPatch((4.0 + 0.07 * k, -0.6 + 0.07 * k), 4.2, 4.05,
                                    boxstyle="round,pad=0.02,rounding_size=0.15", fc="none", ec=TEAL, lw=0.8, zorder=0))
    ax.text(6.1, 3.62, f"one MLP: {n_params(18, 2):,} weights  ·  5 of them, different random starts",
            ha="center", fontsize=9.3, color=TEAL, fontweight="bold")
    mlp_drawing(ax, [4.4, 5.3, 6.2, 7.1, 7.85], [18, 256, 256, 256, 2], ["18", "256", "256", "256", "2"], y0, h, TEAL,
                ["input", "SiLU", "SiLU", "SiLU", "output"])
    arrow(ax, (3.67, 1.55), (4.28, 1.55))
    box(ax, 8.6, 0.8, 1.35, 1.5, "average\nof 5\n= prediction", "white", INK, fs=9)
    box(ax, 10.15, 0.8, 1.3, 1.5, "spread\nof 5\n= uncertainty", "white", INK, fs=9)
    arrow(ax, (7.97, 1.55), (8.58, 1.55))
    ax.text(5.75, -1.05, "inputs and outputs are standardised (mean 0, spread 1) with training-set statistics; "
            "loss = mean squared error; AdamW, one-cycle schedule, 400 epochs, batch 128", ha="center", fontsize=8.5,
            color=MUTED)


def cvae(ax):
    ax.set_xlim(-0.3, 15.2)
    ax.set_ylim(-1.45, 4.35)
    ax.axis("off")
    y0, h = 0.25, 2.6
    ax.add_patch(FancyBboxPatch((0.15, -0.65), 4.6, 4.15, boxstyle="round,pad=0.02,rounding_size=0.15", fc=GREY_L, ec=GREY,
                                lw=1.2, zorder=0))
    ax.text(2.45, 3.33, f"ENCODER (training only)  ·  {n_params(12, 8):,} weights", ha="center", fontsize=9.5,
            color=MUTED, fontweight="bold")
    mlp_drawing(ax, [0.6, 1.5, 2.4, 3.3, 4.25], [12, 256, 256, 256, 8], ["12", "256", "256", "256", "8"], y0, h, GREY,
                ["design\n+ target", "SiLU", "SiLU", "SiLU", "μ, log σ²"])
    box(ax, 5.1, 0.85, 1.6, 1.4, "z = μ + σ·ε\n4 numbers\n(latent code)", AMBER + "33", AMBER, fs=8.8,
        bold=True)
    arrow(ax, (4.4, 1.55), (5.08, 1.55))
    ax.add_patch(FancyBboxPatch((7.0, -0.65), 4.65, 4.15, boxstyle="round,pad=0.02,rounding_size=0.15", fc=TEAL_L, ec=TEAL,
                                lw=1.2, alpha=0.55, zorder=0))
    ax.text(9.32, 3.33, f"DECODER (used to design)  ·  {n_params(6, 10):,} weights", ha="center", fontsize=9.5,
            color=TEAL, fontweight="bold")
    mlp_drawing(ax, [7.4, 8.3, 9.2, 10.1, 11.05], [6, 256, 256, 256, 10], ["6", "256", "256", "256", "10"], y0, h, TEAL,
                ["z\n+ target", "SiLU", "SiLU", "SiLU", "sigmoid\n→ u"])
    arrow(ax, (6.72, 1.55), (7.28, 1.55))
    box(ax, 11.95, 0.8, 1.4, 1.5, "design\n10 numbers", "white", TEAL, fs=9, bold=True)
    arrow(ax, (11.2, 1.55), (11.93, 1.55))
    box(ax, 13.6, 0.65, 1.5, 1.8, "+ refine\n100 gradient\nsteps on the\nfrozen\nforward net", "white", INK, fs=8.3)
    arrow(ax, (13.37, 1.55), (13.58, 1.55))
    ax.text(7.4, -0.95, "training loss = reconstruction error of the design + β·KL(z ‖ normal), β = 0.5;   "
            "to design: draw z from a normal distribution 64 times → 64 different designs for one target",
            ha="center", fontsize=8.6, color=MUTED)


def save(fn, w, h, name):
    fig, ax = plt.subplots(figsize=(w, h))
    fn(ax)
    fig.savefig(f"{OUT}/{name}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", f"{OUT}/{name}")


def main():
    os.makedirs(OUT, exist_ok=True)
    save(tandem, 15, 6.0, "arch_tandem.png")
    save(forward, 11.5, 5.0, "arch_forward.png")
    save(cvae, 15, 5.6, "arch_cvae.png")
    fig, axs = plt.subplots(3, 1, figsize=(15, 16.5), gridspec_kw={"height_ratios": [6.0, 5.0, 5.6]})
    for ax, fn, t in zip(axs, (tandem, forward, cvae), ("(a) Tandem network: inverse network trained through the frozen "
                                                        "forward ensemble", "(b) Forward ensemble",
                                                        "(c) Conditional VAE (+ refine)")):
        fn(ax)
        ax.set_title(t, loc="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/arch_all.png", dpi=170, bbox_inches="tight", facecolor="white")
    print("wrote", f"{OUT}/arch_all.png")


if __name__ == "__main__":
    main()
