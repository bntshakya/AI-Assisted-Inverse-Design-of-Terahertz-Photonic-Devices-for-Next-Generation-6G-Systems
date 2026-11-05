"""Exercise 1 - waves: frequency, wavelength and why the lattice constant is 242.5 um.

Run (from the Project folder, with the virtual environment active):
    python learn/ex01_waves.py
Try: change F_GHZ to 252.72 or 321.84 (the edges of the IEEE 802.15.3d band), or N_SI to 3.42
(bulk silicon) and run again. Watch how the wavelength inside silicon changes.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
C0 = 299_792_458.0      # speed of light in vacuum, m/s
F_GHZ = 300.0           # frequency we care about (0.3 THz)
N_SI = 2.99             # effective refractive index of the silicon slab (bulk silicon: 3.42)
A_UM = 242.5            # lattice constant of the crystal, um

f = F_GHZ * 1e9
lam_air = C0 / f * 1e6          # wavelength in air, um
lam_si = lam_air / N_SI         # wavelength inside silicon, um
print(f"frequency           f = {F_GHZ:.1f} GHz = {f:.3e} Hz")
print(f"wavelength in air   lambda = c / f = {lam_air:.1f} um")
print(f"wavelength in Si    lambda / n = {lam_si:.1f} um   (n = {N_SI})")
print(f"lattice constant    a = {A_UM} um")
print(f"normalised freq.    a / lambda_air = {A_UM / lam_air:.3f}   <- photonic crystals work when this is ~0.2-0.5")
print(f"half a wavelength in Si = {lam_si / 2:.1f} um   <- compare with the hole-to-hole distance 140 um")

x = np.linspace(0, 1500, 3000)
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(x, np.sin(2 * np.pi * x / lam_air), label=f"in air: wavelength {lam_air:.0f} um")
ax.plot(x, np.sin(2 * np.pi * x / lam_si) - 2.6, label=f"in silicon: wavelength {lam_si:.0f} um")
for k in range(int(1500 / A_UM) + 1):
    ax.axvline(k * A_UM, color="0.8", lw=0.8, zorder=0)
ax.set_yticks([])
ax.set_xlabel("position x (um); grey lines = one lattice constant a = 242.5 um apart")
ax.set_title(f"A {F_GHZ:.0f} GHz wave: the crystal period is comparable to the wavelength")
ax.legend(loc="upper right", fontsize=8, frameon=False)
os.makedirs(FIG, exist_ok=True)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "ex01_waves.png"), dpi=160)
print("saved learn/figs/ex01_waves.png")
