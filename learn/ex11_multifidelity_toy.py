"""Exercise 11 - multi-fidelity: many cheap answers + a few expensive ones.

The project has a cheap simulator (PWE, 0.2 s, about 1 GHz off) and an expensive one (COMSOL).
Idea (Kennedy and O'Hagan 2000): expensive(x) = cheap(x) + delta(x), where the DIFFERENCE delta is
smooth and simple, so a Gaussian process can learn it from very few expensive points.

Here: "expensive" = the Forrester test function (a standard benchmark), "cheap" = the same function
with a smooth bias added. We only have 4 expensive points.

Run:  python learn/ex11_multifidelity_toy.py
Try:  make the bias wigglier (e.g. + 2 sin(10 x)) and see the correction need more points.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel  # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
X_HI = np.array([0.0, 0.4, 0.6, 1.0])        # where we can afford the expensive simulation


def expensive(x):
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


def cheap(x):
    return expensive(x) + 3.0 * x - 1.5        # biased but cheap: available everywhere


def gp(x, y):
    k = ConstantKernel(1.0) * RBF(0.3, (1e-2, 1e1)) + WhiteKernel(1e-6, (1e-9, 1e-2))
    return GaussianProcessRegressor(k, normalize_y=True, random_state=0).fit(x[:, None], y)


xg = np.linspace(0, 1, 400)
only_hi = gp(X_HI, expensive(X_HI)).predict(xg[:, None])
delta = gp(X_HI, expensive(X_HI) - cheap(X_HI)).predict(xg[:, None])
corrected = cheap(xg) + delta
truth = expensive(xg)
for name, pred in (("cheap simulator alone", cheap(xg)), ("GP on the 4 expensive points", only_hi),
                   ("cheap + learned correction", corrected)):
    print(f"{name:30s} mean error {np.abs(pred - truth).mean():6.3f}")

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(xg, truth, "k", lw=2.2, label="expensive (truth)")
ax.plot(xg, cheap(xg), color="tab:gray", ls=":", label="cheap (biased)")
ax.plot(xg, only_hi, color="tab:orange", label="GP from 4 expensive points only")
ax.plot(xg, corrected, color="tab:green", ls="--", label="cheap + GP correction (multi-fidelity)")
ax.plot(X_HI, expensive(X_HI), "o", color="k", label="the 4 expensive runs")
ax.set_xlabel("design x")
ax.set_ylabel("response")
ax.legend(fontsize=7, frameon=False)
ax.set_title("Multi-fidelity: the correction is simple even when the function is not", fontsize=9)
os.makedirs(FIG, exist_ok=True)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "ex11_multifidelity.png"), dpi=160)
print("saved learn/figs/ex11_multifidelity.png")
