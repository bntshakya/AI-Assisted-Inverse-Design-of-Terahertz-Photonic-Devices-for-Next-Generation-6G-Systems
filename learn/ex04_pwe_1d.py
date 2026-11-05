"""Exercise 4 - the plane-wave expansion (PWE) method in 1D, step by step.

The crystal: silicon and air layers, period a. A wave travels across the layers (along x).
Its magnetic field obeys   - d/dx [ (1/eps(x)) dH/dx ] = (omega/c)^2 H.
PWE: write H as plane waves with wavevector k + G (G = 2 pi m / a), write eps as its Fourier
series (exercise 2), and the equation becomes a matrix eigenvalue problem. The eigenvalues are
(omega/c)^2, i.e. the allowed frequencies at this k.  We compare with the exact textbook answer.

Units: lengths in units of a, frequency as u = a / lambda = omega a / (2 pi c). For the real
crystal (a = 242.5 um) multiply u by c / a = 1236 GHz.

Run:  python learn/ex04_pwe_1d.py
Try:  FILL = 0.3; EPS_SI = 11.7; M_PLOT = 2 (only 5 plane waves - how wrong is it?)
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
EPS_SI, EPS_AIR = 8.914, 1.0
FILL = 0.5          # silicon fraction of the period
M_PLOT = 20         # plane waves m = -M..M  (2M + 1 of them) for the band plot
GHZ_PER_U = 299792458.0 / 242.5e-6 / 1e9   # u -> GHz for a = 242.5 um


def c_eps(m, inverse=False):
    """Complex Fourier coefficient of eps(x) (or of 1/eps) for exp(i 2 pi m x); silicon centred at x = 0."""
    e1, e2 = (1 / EPS_SI, 1 / EPS_AIR) if inverse else (EPS_SI, EPS_AIR)
    m = np.asarray(m, dtype=float)
    out = (e1 - e2) * np.sin(np.pi * m * FILL) / (np.pi * np.where(m == 0, 1, m))
    return np.where(m == 0, e2 + (e1 - e2) * FILL, out)


def pwe_matrix(k, M, rule="ho"):
    """The PWE matrix at wavevector k (units 1/a) with 2M + 1 plane waves."""
    m = np.arange(-M, M + 1)
    G = 2 * np.pi * m
    diff = m[:, None] - m[None, :]
    if rule == "ho":                     # Ho's rule: invert the Toeplitz matrix of eps (used in pwe.py)
        eta = np.linalg.inv(c_eps(diff))
    else:                                # naive rule: Fourier coefficients of 1/eps directly
        eta = c_eps(diff, inverse=True)
    kg = k + G
    return kg[:, None] * eta * kg[None, :]


def pwe_bands(k, M, rule="ho", n_bands=4):
    lam = np.linalg.eigvalsh(pwe_matrix(k, M, rule))[:n_bands]
    return np.sqrt(np.clip(lam, 0, None)) / (2 * np.pi)      # u = (omega/c) / (2 pi), with a = 1


def exact_rhs(u):
    """Exact dispersion of a two-layer crystal:  cos(k a) = RHS(u)."""
    n1, n2, d1, d2 = np.sqrt(EPS_SI), np.sqrt(EPS_AIR), FILL, 1 - FILL
    w = 2 * np.pi * u
    return (np.cos(n1 * w * d1) * np.cos(n2 * w * d2)
            - 0.5 * (n1 / n2 + n2 / n1) * np.sin(n1 * w * d1) * np.sin(n2 * w * d2))


def exact_first_gap():
    """First gap: where RHS(u) < -1 (no real k exists)."""
    u = np.linspace(1e-4, 1, 200001)
    r = exact_rhs(u) + 1
    s = np.where(np.diff(np.sign(r)) != 0)[0]
    lo = brentq(lambda x: exact_rhs(x) + 1, u[s[0]], u[s[0] + 1])
    hi = brentq(lambda x: exact_rhs(x) + 1, u[s[1]], u[s[1] + 1])
    return lo, hi


# ---- step 1: the matrix by hand-size (3 plane waves) at the zone edge k = pi / a
np.set_printoptions(precision=3, suppress=True)
print("Fourier coefficients of eps:  c0 = %.3f  c1 = c-1 = %.3f  c2 = c-2 = %.3f"
      % tuple(c_eps([0, 1, 2])))
print("\nWith 3 plane waves (m = -1, 0, 1) at k = pi/a:")
print("eps Toeplitz matrix [c_{m-m'}]:\n", c_eps(np.subtract.outer([-1, 0, 1], [-1, 0, 1])))
print("PWE matrix (k+G) [eps^-1] (k+G'):\n", pwe_matrix(np.pi, 1))
lam = np.linalg.eigvalsh(pwe_matrix(np.pi, 1))
print("eigenvalues (omega/c)^2:", lam, "->  u = a/lambda:", np.sqrt(lam) / (2 * np.pi))

# ---- step 2: convergence of the first gap with the number of plane waves
lo_ex, hi_ex = exact_first_gap()
print(f"\nexact first gap:  u = {lo_ex:.4f} to {hi_ex:.4f}   ({lo_ex * GHZ_PER_U:.1f} to {hi_ex * GHZ_PER_U:.1f} GHz for a = 242.5 um)")
print(" plane waves   Ho rule: lower  upper      naive rule: lower  upper")
conv = []
for M in (1, 2, 3, 5, 10, 20, 40):
    h = pwe_bands(np.pi, M, "ho")
    n = pwe_bands(np.pi, M, "naive")
    conv.append((2 * M + 1, h[0], h[1], n[0], n[1]))
    print(f"{2 * M + 1:11d}   {h[0]:.4f}  {h[1]:.4f}          {n[0]:.4f}  {n[1]:.4f}")
conv = np.array(conv)

# ---- figure: (a) bands, (b) convergence
ks = np.linspace(0, np.pi, 121)
bands = np.array([pwe_bands(k, M_PLOT) for k in ks])
uu = np.linspace(1e-3, 0.95, 4000)
rhs = exact_rhs(uu)
ok = np.abs(rhs) <= 1
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axs[0]
ax.axhspan(lo_ex, hi_ex, color="tab:green", alpha=0.15)
ax.plot(ks / np.pi, bands, color="tab:blue")
ax.plot(np.arccos(rhs[ok]) / np.pi, uu[ok], ".", ms=1.2, color="k")
ax.text(0.03, (lo_ex + hi_ex) / 2, "band gap: no wave can travel", fontsize=8, va="center", color="tab:green")
ax.set_xlabel("wavevector k (units of pi/a);  0 = centre, 1 = zone edge")
ax.set_ylabel("frequency u = a / lambda")
ax.set_ylim(0, 0.9)
ax.set_title(f"(a) bands: PWE with {2 * M_PLOT + 1} plane waves (blue) vs exact (black dots)", fontsize=9)
ax = axs[1]
ax.plot(conv[:, 0], conv[:, 1], "o-", color="tab:blue", label="lower edge, Ho rule")
ax.plot(conv[:, 0], conv[:, 2], "s-", color="tab:orange", label="upper edge, Ho rule")
ax.plot(conv[:, 0], conv[:, 3], "o--", color="tab:blue", alpha=0.45, label="lower edge, naive rule")
ax.plot(conv[:, 0], conv[:, 4], "s--", color="tab:orange", alpha=0.45, label="upper edge, naive rule")
ax.axhline(lo_ex, color="k", lw=0.8)
ax.axhline(hi_ex, color="k", lw=0.8)
ax.set_xscale("log")
ax.set_xticks(conv[:, 0])
ax.set_xticklabels([str(int(n)) for n in conv[:, 0]])
ax.minorticks_off()
ax.set_xlabel("number of plane waves")
ax.set_ylabel("gap edge u")
ax.set_title("(b) convergence: black lines = exact answer", fontsize=9)
ax.legend(fontsize=7, frameon=False)
os.makedirs(FIG, exist_ok=True)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "ex04_pwe_1d.png"), dpi=160)
print("saved learn/figs/ex04_pwe_1d.png")
