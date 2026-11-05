"""Step-by-step walk-through of the PWE solver on one dataset cell.
Run from the Project folder: python src/shapes/pwe_demo.py"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh

sys.path.insert(0, str(Path(__file__).parent))
from pwe import Solver, kpath, C0  # noqa: E402
from geometry import radius  # noqa: E402

df = pd.read_csv("data/comsol/band_edges.csv")
row = df[(df.corner == 1) & (df.L1_um.round() == 168) & (df.L2_um.round() == 112)].iloc[0]
holeA, holeB = ("rounded", {"L": row.L1_um}), ("rounded", {"L": row.L2_um})
print(f"STEP 0  COMSOL cell: rounded triangles, L1 = {row.L1_um:.1f} um, L2 = {row.L2_um:.1f} um")
print(f"  COMSOL answer: f_lower={row.f_lower:.2f} GHz, f_upper={row.f_upper:.2f} GHz, gap={row.gap:.2f} GHz")

S = Solver(eps_si=8.914)
eps = S.eps_grid(holeA, holeB)
print(f"\nSTEP 1  picture: {eps.shape} pixels, eps from {eps.min():.2f} to {eps.max():.2f}")
print(f"  air pixels {np.mean(eps == 1):.1%}, silicon pixels {np.mean(eps == 8.914):.1%}, edge pixels (in between) {np.mean((eps > 1) & (eps < 8.914)):.1%}")

eg = np.fft.fft2(eps) / eps.size
print("\nSTEP 2  picture as a sum of waves (Fourier coefficients eps_G):")
for (m, n) in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (5, 3)]:
    print(f"  G = {m} b1 + {n} b2 : |eps_G| = {abs(eg[m, n]):.3f}")
print("  (G=0 is the average eps; higher G = finer detail, shrinking quickly)")

T = eg[S.dm, S.dn]
inv = np.linalg.inv(T)
print(f"\nSTEP 3  {S.n_pw} plane waves -> eps matrix {T.shape}, inverted (Ho's rule)")

res = {}
for name, k in (("M", 1.0), ("K", 2.0)):
    kv = kpath(k) + S.G
    Mmat = (kv @ kv.T) * inv
    lam = eigh(Mmat, eigvals_only=True, subset_by_index=[0, 3])
    f = np.sqrt(lam) * 1e6 * C0 / (2 * np.pi) / 1e9
    res[name] = f
    print(f"\nSTEP 4  k = {name}: matrix {Mmat.shape}, lowest 4 eigenvalues -> frequencies {np.round(f, 2)} GHz")

fl, fu = res["K"][0], res["M"][1]
print(f"\nSTEP 5  f_lower = band 1 at K = {fl:.2f} GHz   (COMSOL {row.f_lower:.2f})")
print(f"        f_upper = band 2 at M = {fu:.2f} GHz   (COMSOL {row.f_upper:.2f})")
print(f"        gap = {fu - fl:.2f} GHz   (COMSOL {row.gap:.2f})")

ks = np.linspace(0, 3, 61)
b = S.bands(holeA, holeB, ks)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
ax[0].imshow(eps.T, origin="lower", cmap="gray_r")
ax[0].set_title("Step 1: unit cell as pixels\n(black = silicon, white = air, skewed axes)")
ax[0].set_xlabel("pixel along a1"); ax[0].set_ylabel("pixel along a2")
mag = np.abs(np.fft.fftshift(eg))
c = S.n // 2
ax[1].imshow(np.log10(mag[c - 12:c + 13, c - 12:c + 13].T + 1e-6), origin="lower", cmap="viridis",
             extent=[-12.5, 12.5, -12.5, 12.5])
ax[1].set_title("Step 2: Fourier coefficients log10|eps_G|\n(centre = average; outer = fine detail)")
ax[1].set_xlabel("m (along b1)"); ax[1].set_ylabel("n (along b2)")
cb = pd.read_csv("data/comsol/bands.csv")
cb = cb[(cb.corner == 1) & (cb.L1_um.round() == 168) & (cb.L2_um.round() == 112)].sort_values("k")
ax[2].plot(cb.k, cb.f1_GHz, "k--", lw=1, label="COMSOL"); ax[2].plot(cb.k, cb.f2_GHz, "k--", lw=1)
ax[2].plot(ks, b[:, 0], "C0", label="PWE band 1"); ax[2].plot(ks, b[:, 1], "C1", label="PWE band 2")
ax[2].axhspan(fl, fu, color="C2", alpha=0.25, label=f"gap {fu - fl:.1f} GHz")
ax[2].plot([2], [fl], "ko"); ax[2].plot([1], [fu], "ko")
ax[2].set_xticks([0, 1, 2, 3], ["Γ", "M", "K", "Γ"])
ax[2].set_ylabel("frequency (GHz)"); ax[2].legend(); ax[2].set_title("Steps 4-5: PWE (solid) vs COMSOL (dashed)")
plt.tight_layout()
out = Path("results/optionb/figs/pwe_demo.png")
plt.savefig(out, dpi=130)
print("\nsaved", out)
