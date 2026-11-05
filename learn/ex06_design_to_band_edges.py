"""Exercise 6 - the whole forward chain: 10 design numbers -> two holes -> fabrication check -> band edges.

This is exactly what every inverse-design method in the project has to "invert".
A design is p = [rA0, c3A, s3A, c6A, s6A,  rB0, c3B, s3B, c6B, s6B]  (all in um).

Run:  python learn/ex06_design_to_band_edges.py      (about 5 seconds)
Try:  add your own design to DESIGNS; make rA0 = 90 to see the fabrication check fail.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "shapes"))
from geometry import A1, A2, boundary, cell_features, check_cell  # noqa: E402
from pwe import Solver  # noqa: E402

FIG = os.path.join(HERE, "figs")
DESIGNS = {   # same average radii (55 um and 35 um), different shapes
    "two circles": [55, 0, 0, 0, 0, 35, 0, 0, 0, 0],
    "A triangle-like (c3 = -9)": [55, -9, 0, 0, 0, 35, 0, 0, 0, 0],
    "A flipped (c3 = +9)": [55, 9, 0, 0, 0, 35, 0, 0, 0, 0],
    "both shaped": [55, -9, 0, 3, 0, 35, 6, 0, 0, 0],
    "too big (rA0 = 95)": [95, 0, 0, 0, 0, 35, 0, 0, 0, 0],
}


def to_holes(p):
    a = {"r0": p[0], "c3": p[1], "s3": p[2], "c6": p[3], "s6": p[4]}
    b = {"r0": p[5], "c3": p[6], "s3": p[7], "c6": p[8], "s6": p[9]}
    return ("fourier", a), ("fourier", b)


s = Solver(eps_si=8.914)
fig, axs = plt.subplots(1, len(DESIGNS), figsize=(3 * len(DESIGNS), 3.2))
print("design                       valid  thinnest wall  f_lower  f_upper    gap   (GHz)")
for ax, (name, p) in zip(axs, DESIGNS.items()):
    hA, hB = to_holes(p)
    ok, info = check_cell(hA, hB)
    if ok:
        lo, up = s.band_edges(hA, hB)
        txt = f"{lo:7.1f}  {up:7.1f}  {up - lo:6.1f}"
    else:
        txt = "   (not solved: fails the fabrication rules)"
    print(f"{name:28s} {str(ok):5s}  {info['min_wall']:9.1f} um  {txt}")
    ax.add_patch(plt.Rectangle((-150, -150), 700, 700, color="#cfd6df"))
    for i in range(-1, 3):
        for j in range(-1, 3):
            off = i * A1 + j * A2
            for hole, site in ((hA, "A"), (hB, "B")):
                ax.add_patch(Polygon(boundary(*hole, site) + off, fc="white", ec="k", lw=0.6))
    ax.set_xlim(-60, 420)
    ax.set_ylim(-60, 330)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(name + ("" if ok else "\nINVALID"), fontsize=8, color="k" if ok else "red")

f18 = cell_features(*to_holes(DESIGNS["both shaped"]))
print("\nthe 18 machine-learning features of 'both shaped' (um):")
print("  hole A:", np.round(f18[:9], 2))
print("  hole B:", np.round(f18[9:], 2))
os.makedirs(FIG, exist_ok=True)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "ex06_design_to_band_edges.png"), dpi=160)
print("saved learn/figs/ex06_design_to_band_edges.png")
