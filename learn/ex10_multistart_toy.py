"""Exercise 10 - multi-start gradient search, and the danger of "exploiting" a surrogate.

Part 1. No network for the inverse at all: start from 30 random designs and walk each one downhill
on the error (forward(x) - target)^2, using the gradient. Each start ends at a (different) correct
design. The design is written as x = tanh(w) so it can never leave the square [-1, 1]^2 (the project
does the same with a sigmoid and adds a fabrication penalty).

Part 2. In the project the forward map is a TRAINED network, not the truth. Search hard enough and
the optimiser finds designs where the network is wrong in a flattering way: the network says
"perfect", the real simulation disagrees. Here the "truth" is a little bumpier (like a real band
structure) and the network has only N_TRAIN = 40 examples, to make that visible. That is why every design in the project is re-checked with the PWE solver.

Run:  python learn/ex10_multistart_toy.py     (about 20 seconds)
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
torch.manual_seed(1)
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
TARGET, N_STARTS, STEPS = 0.5, 30, 300
N_TRAIN = 40     # examples for the weak surrogate in part 2 (try 400)


def f_true(x):
    return (x ** 2).sum(-1)


def search(forward, n_starts, steps, keep_path=False):
    w = torch.randn(n_starts, 2, requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.05)
    path = []
    for _ in range(steps):
        x = torch.tanh(w)
        loss = ((forward(x) - TARGET) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if keep_path:
            path.append(torch.tanh(w).detach().numpy().copy())
    return torch.tanh(w).detach(), np.array(path)


# Part 1: search on the true function
x_end, path = search(lambda x: f_true(x), N_STARTS, STEPS, keep_path=True)
print(f"part 1: {N_STARTS} starts, all end with |y - target| < {np.abs(f_true(x_end.numpy()) - TARGET).max():.1e}")

# Part 2: a bumpier "truth" (like a real band structure) and a surrogate trained on only 40 examples
def f_bumpy(x):
    lib = torch if isinstance(x, torch.Tensor) else np
    return (x ** 2).sum(-1) + 0.08 * lib.sin(7 * x[..., 0]) * lib.sin(7 * x[..., 1])


Xs = torch.rand(N_TRAIN, 2) * 2 - 1
net = torch.nn.Sequential(torch.nn.Linear(2, 64), torch.nn.SiLU(), torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1))
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
for _ in range(3000):
    opt.zero_grad()
    l_ = ((net(Xs)[:, 0] - f_bumpy(Xs)) ** 2).mean()
    l_.backward()
    opt.step()
for p in net.parameters():
    p.requires_grad_(False)
xs2, _ = search(lambda x: net(x)[:, 0], 200, 600)
pred, true = net(xs2)[:, 0].numpy(), f_bumpy(xs2.numpy())
print(f"part 2: surrogate says mean error {np.abs(pred - TARGET).mean():.3f}; "
      f"true (re-simulated) mean error {np.abs(true - TARGET).mean():.3f}; worst {np.abs(true - TARGET).max():.3f}")

fig, axs = plt.subplots(1, 2, figsize=(10, 4.4))
ax = axs[0]
ax.add_patch(plt.Circle((0, 0), np.sqrt(TARGET), fill=False, color="k", lw=1.2, ls="--"))
for i in range(N_STARTS):
    ax.plot(path[:, i, 0], path[:, i, 1], color="tab:blue", lw=0.7, alpha=0.7)
ax.plot(path[0, :, 0], path[0, :, 1], "o", color="tab:orange", ms=3, label="starts")
ax.plot(path[-1, :, 0], path[-1, :, 1], "s", color="tab:green", ms=4, label="ends")
ax.set_aspect("equal")
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)
ax.set_title("part 1: 30 searches, 30 different correct designs", fontsize=9)
ax.legend(fontsize=7, loc="lower left")
ax = axs[1]
ax.hist(np.abs(pred - TARGET), bins=30, alpha=0.6, label="error the surrogate believes")
ax.hist(np.abs(true - TARGET), bins=30, alpha=0.6, label="real error (re-simulated)")
ax.set_xlabel("|y - target|")
ax.set_ylabel("designs")
ax.set_title("part 2: searching on a weak surrogate", fontsize=9)
ax.legend(fontsize=7)
os.makedirs(FIG, exist_ok=True)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "ex10_multistart.png"), dpi=160)
print("saved learn/figs/ex10_multistart.png")
