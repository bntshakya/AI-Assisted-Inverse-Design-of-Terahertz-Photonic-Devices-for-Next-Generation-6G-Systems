"""Exercise 5 - the real 2D solver of the project (src/shapes/pwe.py) on circular holes.

Same idea as exercise 4, but the crystal is 2D (honeycomb of two holes A and B per cell), so
G = m b1 + n b2 has two integers and the matrix is about 439 x 439 instead of 41 x 41.
We look at (1) the permittivity map the solver builds, (2) bands along the COMSOL path
Gamma -> M -> K -> Gamma for EQUAL holes (Dirac point at K, no gap opens at K) and UNEQUAL holes
(a valley gap opens at K), (3) how the band edges change with the number of plane waves.

Run:  python learn/ex05_pwe_2d.py        (about 20 seconds)
Try:  R_A, R_B = 55, 40 ; or put a triangle: ("rounded", {"L": 168}) for hole A.
"""
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "shapes"))
from geometry import A1, A2  # noqa: E402
from pwe import Solver  # noqa: E402

FIG = os.path.join(HERE, "figs")
EPS = 8.914                       # calibrated silicon permittivity (study PDF, part 6)
EQUAL = (("circle", {"R": 48.0}), ("circle", {"R": 48.0}))
UNEQUAL = (("circle", {"R": 60.0}), ("circle", {"R": 35.0}))

t0 = time.time()
s = Solver(eps_si=EPS)
print(f"solver ready in {time.time() - t0:.2f} s: {s.n} x {s.n} grid, {s.n_pw} plane waves "
      f"-> a {s.n_pw} x {s.n_pw} matrix per k-point")

ks = np.round(np.arange(0.04, 2.97, 0.04), 2)   # COMSOL-style path: 0 Gamma, 1 M, 2 K, 3 Gamma (hits 1 and 2 exactly)
res = {}
for name, cell in (("equal", EQUAL), ("unequal", UNEQUAL)):
    t0 = time.time()
    b = s.bands(*cell, ks)
    lo, up = b[:, 0].max(), b[:, 1].min()
    res[name] = b
    print(f"{name:8s} holes: band 1 top {lo:.1f} GHz at k = {ks[b[:, 0].argmax()]:.2f}, "
          f"band 2 bottom {up:.1f} GHz at k = {ks[b[:, 1].argmin()]:.2f}, gap {up - lo:+.1f} GHz "
          f"({time.time() - t0:.1f} s for {len(ks)} k-points)")
    fK, fM = s.bands(*cell, [2.0]), s.bands(*cell, [1.0])
    print(f"          at K: {fK[0, 0]:.2f} and {fK[0, 1]:.2f} GHz;  at M: {fM[0, 0]:.2f} and {fM[0, 1]:.2f} GHz")

print("\nconvergence (unequal holes): plane waves -> band edges at M/K")
for g in (5, 7, 9, 11, 14):
    sg = Solver(eps_si=EPS, g_cut=g)
    lo, up = sg.band_edges(*UNEQUAL)
    print(f"  {sg.n_pw:4d} plane waves: f_lower {lo:7.2f}  f_upper {up:7.2f}  gap {up - lo:6.2f} GHz")

fig = plt.figure(figsize=(12, 3.6))
ax = fig.add_axes([0.05, 0.14, 0.25, 0.74])
eg = s.eps_grid(*UNEQUAL)
u = (np.arange(s.n + 1)) / s.n                       # pixel corners in cell coordinates
U, V = np.meshgrid(u, u, indexing="ij")
X, Y = U * A1[0] + V * A2[0], U * A1[1] + V * A2[1]  # -> micrometres (the cell is a rhombus)
ax.pcolormesh(X, Y, eg, cmap="Greys", shading="flat")
ax.set_aspect("equal")
ax.set_title("eps on the 96 x 96 grid of one cell\n(dark = silicon, white = air)", fontsize=8)
ax.set_xlabel("x (um)")
ax.set_ylabel("y (um)")
for i, (name, b) in enumerate(res.items()):
    ax = fig.add_axes([0.40 + i * 0.31, 0.12, 0.26, 0.78])
    lo, up = b[:, 0].max(), b[:, 1].min()
    if up > lo:
        ax.axhspan(lo, up, color="tab:green", alpha=0.2)
        ax.text(0.05, (lo + up) / 2, f"gap {up - lo:.1f} GHz", fontsize=8, color="tab:green", va="center")
    ax.plot(ks, b, color="tab:blue")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Γ", "M", "K", "Γ"])
    ax.set_ylim(150, 500)
    ax.set_ylabel("frequency (GHz)")
    cellname = "R_A = R_B = 48 um" if name == "equal" else "R_A = 60, R_B = 35 um"
    ax.set_title(f"{name} circles ({cellname})", fontsize=9)
os.makedirs(FIG, exist_ok=True)
fig.savefig(os.path.join(FIG, "ex05_pwe_2d.png"), dpi=160)
print("saved learn/figs/ex05_pwe_2d.png")
