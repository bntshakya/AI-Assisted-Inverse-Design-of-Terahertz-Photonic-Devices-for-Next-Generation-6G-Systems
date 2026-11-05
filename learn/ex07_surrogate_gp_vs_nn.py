"""Exercise 7 - a surrogate model: learn a function from examples. Gaussian process vs neural network.

The project's forward surrogate learns  18 shape numbers -> (f_lower, f_upper).  Here the same
idea in 1D so you can see it: learn y = f(x) from a few examples, then predict everywhere.
Both models also say how UNSURE they are: the GP from its kernel, the network ensemble from the
disagreement between 5 networks trained from different random starts.

Run:  python learn/ex07_surrogate_gp_vs_nn.py     (about 30 seconds)
Try:  N_SMALL = 4; change the kernel length scale; HIDDEN = 8.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel  # noqa: E402

torch.set_num_threads(4)
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
N_SMALL, N_BIG = 8, 60
HIDDEN, STEPS = 32, 3000


def f_true(x):
    return np.sin(3 * x) + 0.5 * x


def fit_gp(x, y):
    k = ConstantKernel(1.0) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1e-4, (1e-8, 1e-1))
    return GaussianProcessRegressor(k, normalize_y=True, random_state=0).fit(x[:, None], y)


def fit_nn(x, y, seed):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(1, HIDDEN), torch.nn.SiLU(), torch.nn.Linear(HIDDEN, HIDDEN),
                              torch.nn.SiLU(), torch.nn.Linear(HIDDEN, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    X, Y = torch.tensor(x[:, None], dtype=torch.float32), torch.tensor(y[:, None], dtype=torch.float32)
    for _ in range(STEPS):                       # gradient descent on the mean squared error
        opt.zero_grad()
        loss = ((net(X) - Y) ** 2).mean()
        loss.backward()
        opt.step()
    return net


rng = np.random.default_rng(0)
xg = np.linspace(0, 3, 400)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, n in zip(axs, (N_SMALL, N_BIG)):
    x = np.sort(rng.uniform(0, 3, n))
    y = f_true(x) + rng.normal(0, 0.02, n)
    gp = fit_gp(x, y)
    mu, sd = gp.predict(xg[:, None], return_std=True)
    nets = [fit_nn(x, y, s) for s in range(5)]
    with torch.no_grad():
        P = np.stack([m(torch.tensor(xg[:, None], dtype=torch.float32)).numpy()[:, 0] for m in nets])
    print(f"{n:3d} examples:  GP error {np.abs(mu - f_true(xg)).mean():.3f}   "
          f"NN-ensemble error {np.abs(P.mean(0) - f_true(xg)).mean():.3f}   (mean absolute error over the whole range)")
    ax.plot(xg, f_true(xg), "k", lw=2, label="true function")
    ax.plot(xg, mu, color="tab:orange", label="GP mean")
    ax.fill_between(xg, mu - 2 * sd, mu + 2 * sd, color="tab:orange", alpha=0.2, label="GP ± 2 std")
    ax.plot(xg, P.mean(0), color="tab:blue", label="NN ensemble mean")
    ax.fill_between(xg, P.mean(0) - 2 * P.std(0), P.mean(0) + 2 * P.std(0), color="tab:blue", alpha=0.15,
                    label="NN ± 2 std")
    ax.plot(x, y, "o", color="k", ms=4, label="training examples")
    ax.set_title(f"{n} training examples")
    ax.set_xlabel("input x")
axs[0].set_ylabel("output y")
axs[0].legend(fontsize=7, frameon=False, loc="upper left")
os.makedirs(FIG, exist_ok=True)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "ex07_surrogate.png"), dpi=160)
print("saved learn/figs/ex07_surrogate.png")
