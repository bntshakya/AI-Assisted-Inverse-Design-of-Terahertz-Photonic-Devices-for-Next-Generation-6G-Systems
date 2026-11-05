"""Exercise 3 - describing a hole shape with a Fourier series in the angle.

A hole is drawn around its centre in polar form: the distance r from the centre to the edge,
for every direction theta. Any closed, star-shaped hole is a function r(theta) that repeats
every 360 degrees, so it has a Fourier series:
    r(theta) = r0 + sum_n [ c_n cos(n theta) + s_n sin(n theta) ]
Valley crystals need three-fold symmetry (looks the same after 120 degrees), which allows only
n = 3, 6, 9, 12, ...  This script shows it, and computes the numbers for the project's old
rounded triangle.

Run:  python learn/ex03_hole_shapes.py
Try:  change C3 to -0.2, or set C2 = 0.1 (a forbidden term) and look at the symmetry test.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
R0 = 50.0          # average radius, um
C3 = 0.20 * R0     # strength of the 3-fold term
C6 = 0.05 * R0     # strength of the 6-fold term
C2 = 0.10 * R0     # a 2-fold term: NOT allowed (breaks the 3-fold symmetry)
L_TRI = 168.0      # side of the old rounded triangle, um


def r_fourier(theta, r0, terms):
    """terms = {n: (c_n, s_n)}"""
    r = np.full_like(theta, r0)
    for n, (c, s) in terms.items():
        r = r + c * np.cos(n * theta) + s * np.sin(n * theta)
    return r


def r_rounded_triangle(theta, L):
    """Old COMSOL hole: triangle of side L cut by a disk of radius 2L/(3 sqrt3) (see the study PDF)."""
    h = L / (2 * np.sqrt(3))                                  # centre-to-side distance
    tf = (theta + np.pi / 3) % (2 * np.pi / 3) - np.pi / 3    # angle to the nearest side's normal
    return np.minimum(h / np.cos(tf), 2 * L / (3 * np.sqrt(3)))


th = np.linspace(0, 2 * np.pi, 3600, endpoint=False)

# --- 1. a worked table: r at a few angles for r0 + c3 cos(3 theta)
print("theta(deg)   cos(3 theta)   r = r0 + c3 cos(3 theta)   (r0 = 50, c3 = 10)")
for d in (0, 30, 60, 90, 120, 180):
    t = np.deg2rad(d)
    print(f"{d:8d}   {np.cos(3 * t):12.3f}   {R0 + C3 * np.cos(3 * t):12.1f}")

# --- 2. symmetry test: does the shape look the same after turning by 120 degrees?
shapes = {
    "circle": {},
    "+ c3 cos 3theta": {3: (C3, 0)},
    "- c3 cos 3theta": {3: (-C3, 0)},
    "+ c3 and c6": {3: (C3, 0), 6: (C6, 0)},
    "+ c3 sin 3theta (rotated)": {3: (0, C3)},
    "c2 cos 2theta (forbidden)": {2: (C2, 0)},
}
print("\nshape                         max change of r after a 120-degree turn (um)")
for name, terms in shapes.items():
    r = r_fourier(th, R0, terms)
    turned = r_fourier(th + 2 * np.pi / 3, R0, terms)
    print(f"{name:28s}  {np.abs(r - turned).max():8.3f}")

# --- 3. the 9 numbers of the old rounded triangle (L = 168 um), by numerical integration
r = r_rounded_triangle(th, L_TRI)
print(f"\nrounded triangle, L = {L_TRI} um: Fourier coefficients (um)")
print("  n     c_n (cos)   s_n (sin)")
for n in range(0, 13):
    c = r.mean() if n == 0 else 2 * np.mean(r * np.cos(n * th))
    s = 0.0 if n == 0 else 2 * np.mean(r * np.sin(n * th))
    flag = "  <- allowed (multiple of 3)" if n % 3 == 0 and n else ""
    print(f"{n:3d}  {c:10.3f}  {s:10.3f}{flag}")
r0 = r.mean()
print(f"r0 / L = {r0 / L_TRI:.3f},  c3 / r0 = {2 * np.mean(r * np.cos(3 * th)) / r0:.3f}")

# --- figure
fig, axs = plt.subplots(1, 7, figsize=(14, 2.6), subplot_kw={"aspect": "equal"})
for ax, (name, terms) in zip(axs, shapes.items()):
    rr = r_fourier(th, R0, terms)
    ax.fill(rr * np.cos(th), rr * np.sin(th), color="#dbe7f5", ec="k", lw=1)
    ax.plot([0, 70], [0, 0], color="r", lw=0.8)
    ax.set_title(name, fontsize=8)
ax = axs[-1]
recon = {n: (2 * np.mean(r * np.cos(n * th)), 2 * np.mean(r * np.sin(n * th))) for n in (3, 6, 9, 12)}
for N, col in ((3, "tab:orange"), (12, "tab:green")):
    rr = r_fourier(th, r0, {n: v for n, v in recon.items() if n <= N})
    ax.plot(rr * np.cos(th), rr * np.sin(th), color=col, lw=1, label=f"up to n = {N}")
ax.plot(r * np.cos(th), r * np.sin(th), "k", lw=1.6, label="exact")
ax.set_title("old rounded triangle", fontsize=8)
ax.legend(fontsize=6, frameon=False, loc="lower left")
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
os.makedirs(FIG, exist_ok=True)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "ex03_hole_shapes.png"), dpi=160)
print("saved learn/figs/ex03_hole_shapes.png  (red line = direction theta = 0, towards the neighbour hole)")
