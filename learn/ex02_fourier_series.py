"""Exercise 2 - Fourier series of a repeating pattern (the permittivity of a layered crystal).

The pattern: a 1D crystal with period a. Silicon (eps = EPS_SI) fills a fraction FILL of each period,
air (eps = 1) the rest. We write eps(x) as a sum of cosines and check the coefficients two ways:
by the formula worked out by hand in the study PDF, and numerically with the FFT.

Run:  python learn/ex02_fourier_series.py
Try:  FILL = 0.2 or 0.8; N_TERMS = (1, 2, 5, 50); watch how many terms a sharp edge needs.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
EPS_SI, EPS_AIR = 8.914, 1.0   # the fitted silicon permittivity of this project
FILL = 0.5                     # fraction of each period that is silicon
N_TERMS = (1, 3, 10, 50)       # how many cosines to add up in the plot


def eps_of_x(x):
    """Silicon where |x - nearest whole number| < FILL/2, air elsewhere (period a = 1)."""
    d = np.abs(x - np.round(x))
    return np.where(d < FILL / 2, EPS_SI, EPS_AIR)


def coeff_formula(m):
    """Hand-derived coefficient of cos(2 pi m x): see section 3.4 of the study PDF."""
    if m == 0:
        return EPS_AIR + (EPS_SI - EPS_AIR) * FILL
    return 2 * (EPS_SI - EPS_AIR) * np.sin(np.pi * m * FILL) / (np.pi * m)


# numerical coefficients: sample one period and use the FFT
N = 4096
xs = (np.arange(N) + 0.5) / N - 0.5
e = eps_of_x(xs)
fft = np.fft.fft(np.fft.ifftshift(e)) / N           # complex coefficients c_m
print(" m   formula a_m    FFT a_m      (a_m = coefficient of cos(2 pi m x))")
for m in range(0, 7):
    a_num = fft[0].real if m == 0 else 2 * fft[m].real
    print(f"{m:2d}  {coeff_formula(m):10.4f}  {a_num:10.4f}")


def partial_sum(x, n_terms):
    return sum(coeff_formula(m) * np.cos(2 * np.pi * m * x) for m in range(n_terms + 1))


x = np.linspace(-1, 1, 2000)
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.plot(x, eps_of_x(x), color="k", lw=2.2, label="the real pattern eps(x)")
for n in N_TERMS:
    ax.plot(x, partial_sum(x, n), lw=1, label=f"sum of {n} cosine{'s' if n > 1 else ''}")
ax.set_xlabel("x (in units of the period a)")
ax.set_ylabel("permittivity eps")
ax.set_title("Fourier series: more terms -> closer to the sharp silicon/air pattern")
ax.legend(fontsize=7, frameon=False, ncol=2)
os.makedirs(FIG, exist_ok=True)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "ex02_fourier_series.png"), dpi=160)
print("saved learn/figs/ex02_fourier_series.png")
