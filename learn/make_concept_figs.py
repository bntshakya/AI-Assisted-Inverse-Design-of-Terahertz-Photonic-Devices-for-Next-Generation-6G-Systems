"""Concept diagrams for the learning PDF (no physics computed here, just drawings).
Run:  python learn/make_concept_figs.py   -> learn/figs/concept_*.png
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "shapes"))
from geometry import A1, A2, SITE_A, SITE_B, boundary  # noqa: E402

FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9})


def arrow(ax, a, b, col="k", lw=1.2, text=None, off=(0, 0)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=lw, color=col))
    if text:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, color=col, fontsize=9)


def box(ax, xy, w, h, text, col):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02", fc=col, ec="k", lw=0.8))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=8.5)


# ---------------------------------------------------------------- 1. lattice and Brillouin zone
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axs[0]
ax.add_patch(plt.Rectangle((-300, -300), 1200, 1200, color="#cfd6df"))
holeA, holeB = ("circle", {"R": 55}), ("circle", {"R": 35})
for i in range(-2, 4):
    for j in range(-2, 4):
        off = i * A1 + j * A2
        for hole, site in ((holeA, "A"), (holeB, "B")):
            ax.add_patch(Polygon(boundary(*hole, site) + off, fc="white", ec="k", lw=0.6))
cell = np.array([[0, 0], A1, A1 + A2, A2])
ax.add_patch(Polygon(cell, fill=False, ec="tab:blue", lw=1.5, ls="--"))
arrow(ax, (0, 0), A1, "tab:red", 1.8, "a1", (-10, -28))
arrow(ax, (0, 0), A2, "tab:red", 1.8, "a2", (-40, 0))
ax.text(*SITE_A, "A", ha="center", va="center", color="tab:red", fontsize=10, weight="bold")
ax.text(*SITE_B, "B", ha="center", va="center", color="tab:red", fontsize=10, weight="bold")
ax.set_xlim(-150, 520)
ax.set_ylim(-120, 400)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("real space: silicon (grey) with holes A and B\nblue dashed = one unit cell, a = |a1| = |a2| = 242.5 um",
             fontsize=9)
ax = axs[1]
K = 4 * np.pi / 3
hexagon = np.array([[K * np.cos(t), K * np.sin(t)] for t in np.deg2rad(np.arange(0, 361, 60))])
ax.fill(hexagon[:, 0], hexagon[:, 1], color="#eef3fa", ec="k", lw=1)
M = np.array([np.pi, np.pi / np.sqrt(3)])
Kp = np.array([K, 0])
ax.fill([0, M[0], Kp[0]], [0, M[1], Kp[1]], color="#fde2c4", ec="tab:orange", lw=1)
for p, t, o in ((np.zeros(2), "Γ (k = 0, 3)", (-1.6, -0.5)), (M, "M (k = 1)", (0.15, 0.1)), (Kp, "K (k = 2)", (0.12, -0.35)),
                (np.array([K / 2, K * np.sqrt(3) / 2]), "K′", (0.1, 0.1))):
    ax.plot(*p, "o", color="k", ms=4)
    ax.text(p[0] + o[0], p[1] + o[1], t, fontsize=9)
arrow(ax, (0, 0), M, "tab:blue", 1.6)
arrow(ax, M, Kp, "tab:blue", 1.6)
arrow(ax, Kp, (0.05, 0), "tab:blue", 1.6)
ax.set_aspect("equal")
ax.set_xlim(-5, 6)
ax.set_ylim(-4.7, 4.7)
ax.set_xlabel("k_x (units of 1/a)")
ax.set_ylabel("k_y (units of 1/a)")
ax.set_title("wavevector space: the hexagon = Brillouin zone\norange = irreducible part, blue = COMSOL k-path", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "concept_lattice_bz.png"), dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 2. polar description of a hole
th = np.linspace(0, 2 * np.pi, 720)
r = 50 - 9 * np.cos(3 * th) + 3 * np.cos(6 * th)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.6]})
ax = axs[0]
ax.fill(r * np.cos(th), r * np.sin(th), color="white", ec="k", lw=1.4)
ax.set_facecolor("#cfd6df")
t0 = np.deg2rad(40)
r0 = 50 - 9 * np.cos(3 * t0) + 3 * np.cos(6 * t0)
ax.plot([0, r0 * np.cos(t0)], [0, r0 * np.sin(t0)], color="tab:red", lw=1.5)
ax.plot([0, 70], [0, 0], color="k", lw=0.8, ls=":")
ax.text(r0 * np.cos(t0) / 2 - 12, r0 * np.sin(t0) / 2 + 4, "r(θ)", color="tab:red", fontsize=11)
ax.add_patch(matplotlib.patches.Arc((0, 0), 30, 30, theta1=0, theta2=40, color="tab:blue"))
ax.text(18, 5, "θ", color="tab:blue", fontsize=11)
ax.text(47, -24, "θ = 0:\ntowards the\nother hole", fontsize=7)
ax.plot(0, 0, "k+", ms=8)
ax.set_xlim(-70, 75)
ax.set_ylim(-70, 70)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("a hole: distance r from the centre to the edge,\nfor every direction θ", fontsize=9)
ax = axs[1]
ax.plot(np.degrees(th), r, color="k", lw=1.6, label="r(θ) of this hole")
ax.axhline(50, color="tab:gray", ls=":", label="r0 = 50 (average)")
ax.plot(np.degrees(th), 50 - 9 * np.cos(3 * th), color="tab:orange", lw=1, label="r0 + c3 cos 3θ  (c3 = -9)")
for x in (120, 240):
    ax.axvline(x, color="0.85", lw=0.8)
ax.set_xticks(range(0, 361, 60))
ax.set_xlabel("θ (degrees)")
ax.set_ylabel("r (um)")
ax.set_title("the same hole unrolled: a curve that repeats every 120°\n-> only harmonics n = 3, 6, 9, ... can appear", fontsize=9)
ax.set_ylim(38, 70)
ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "concept_polar.png"), dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 3. a neural network
fig, ax = plt.subplots(figsize=(8, 3.3))
layers = [4, 6, 6, 2]
names = ["inputs\n(18 shape numbers;\n4 drawn)", "hidden layer 1\n(256 neurons;\n6 drawn)", "hidden layer 2", "outputs\nf_lower, f_upper"]
pos = []
for i, n in enumerate(layers):
    ys = np.linspace(-1, 1, n) * (n / 6)
    pos.append([(i * 3, y) for y in ys])
for a, b in zip(pos[:-1], pos[1:]):
    for p in a:
        for q in b:
            ax.plot([p[0], q[0]], [p[1], q[1]], color="0.8", lw=0.6, zorder=0)
for i, ps in enumerate(pos):
    for p in ps:
        ax.add_patch(plt.Circle(p, 0.17, color=["tab:blue", "tab:gray", "tab:gray", "tab:orange"][i], zorder=2))
    ax.text(i * 3, -1.55, names[i], ha="center", va="top", fontsize=8)
ax.text(4.5, 1.35, "each neuron: output = SiLU( w1·in1 + w2·in2 + ... + b )\ntraining = adjusting all w and b to reduce the error",
        ha="center", fontsize=8.5, color="tab:red")
ax.set_xlim(-1, 10)
ax.set_ylim(-2.6, 1.8)
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "concept_nn.png"), dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 4. the four inverse methods
fig, axs = plt.subplots(2, 2, figsize=(11, 5.2))
B, G, O, P = "#d6e6f7", "#dff0e3", "#fde2c4", "#eadff5"
ax = axs[0, 0]
box(ax, (0.02, 0.38), 0.22, 0.25, "target\n(f_lower, f_upper)", O)
box(ax, (0.38, 0.38), 0.24, 0.25, "inverse net\n(trained)", G)
box(ax, (0.74, 0.38), 0.24, 0.25, "design\n(10 numbers)", B)
arrow(ax, (0.26, 0.5), (0.36, 0.5))
arrow(ax, (0.64, 0.5), (0.72, 0.5))
box(ax, (0.48, 0.02), 0.5, 0.22, "frozen forward net\n-> predicted band edges", P)
arrow(ax, (0.86, 0.36), (0.86, 0.26))
ax.text(0.02, 0.1, "loss = (predicted edges - target)²\nonly the inverse net learns", fontsize=8, color="tab:red")
ax.set_title("(a) tandem network: 1 design per target, instant", fontsize=9)
ax = axs[0, 1]
box(ax, (0.02, 0.55), 0.22, 0.25, "target", O)
box(ax, (0.02, 0.15), 0.22, 0.25, "random code z\n~ N(0, 1)", "#f3f4f6")
box(ax, (0.38, 0.35), 0.24, 0.3, "decoder\n(trained with\nan encoder)", G)
box(ax, (0.74, 0.35), 0.24, 0.3, "design\n(different for\neach z)", B)
arrow(ax, (0.26, 0.67), (0.36, 0.55))
arrow(ax, (0.26, 0.27), (0.36, 0.45))
arrow(ax, (0.64, 0.5), (0.72, 0.5))
ax.set_title("(b) conditional VAE: many designs per target, instant", fontsize=9)
ax = axs[1, 0]
box(ax, (0.02, 0.38), 0.22, 0.25, "64 random\nstarting designs", B)
box(ax, (0.38, 0.38), 0.3, 0.25, "walk downhill on\n(forward(design) - target)²\n+ fabrication penalty", G)
box(ax, (0.78, 0.38), 0.2, 0.25, "64 designs", B)
arrow(ax, (0.26, 0.5), (0.36, 0.5))
arrow(ax, (0.7, 0.5), (0.76, 0.5))
ax.text(0.38, 0.15, "uses the gradient of the forward net; no training", fontsize=8, color="tab:red")
ax.set_title("(c) multi-start gradient search: many designs, ~0.6 s", fontsize=9)
ax = axs[1, 1]
box(ax, (0.02, 0.38), 0.24, 0.25, "CVAE samples\n(good guesses)", G)
box(ax, (0.38, 0.38), 0.3, 0.25, "short gradient\nsearch from each", G)
box(ax, (0.78, 0.38), 0.2, 0.25, "accurate AND\ndifferent designs", B)
arrow(ax, (0.28, 0.5), (0.36, 0.5))
arrow(ax, (0.7, 0.5), (0.76, 0.5))
ax.text(0.02, 0.15, "every method's designs are then checked: fabrication rules + PWE re-simulation", fontsize=8,
        color="tab:red")
ax.set_title("(d) CVAE + refine: best of both, ~0.2 s", fontsize=9)
for ax in axs.flat:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.9)
    ax.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "concept_inverse_methods.png"), dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 5. Bloch wave in 1D
x = np.linspace(0, 6, 2000)
k = 2 * np.pi / 2.6
u = 1 + 0.35 * np.cos(2 * np.pi * x) + 0.1 * np.cos(4 * np.pi * x)
fig, axs = plt.subplots(3, 1, figsize=(8, 4.2), sharex=True)
axs[0].plot(x, np.cos(k * x), color="tab:blue")
axs[0].set_ylabel("plane wave\ncos(kx)", fontsize=8)
axs[1].plot(x, u, color="tab:orange")
axs[1].set_ylabel("periodic part\nu(x), period a", fontsize=8)
axs[2].plot(x, np.cos(k * x) * u, color="k")
axs[2].set_ylabel("Bloch wave\n= product", fontsize=8)
for ax in axs:
    for i in range(7):
        ax.axvline(i, color="0.85", lw=0.8, zorder=0)
    ax.set_yticks([])
axs[2].set_xlabel("x (units of a); grey lines = unit cells")
fig.suptitle("Bloch's theorem: a wave in a crystal = an ordinary wave × a pattern that repeats every cell", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "concept_bloch.png"), dpi=170)
plt.close(fig)
print("saved learn/figs/concept_*.png")
