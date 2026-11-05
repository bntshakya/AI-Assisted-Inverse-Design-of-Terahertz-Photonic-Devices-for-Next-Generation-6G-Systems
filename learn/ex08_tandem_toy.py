"""Exercise 8 - why a direct inverse network fails and how the TANDEM network fixes it.

Toy design problem with the same difficulty as the project: many designs give the same answer.
Design x = (x1, x2) in the square [-1, 1]^2.  Response y = x1^2 + x2^2 (distance from centre, squared).
For a target y every point on a circle of radius sqrt(y) is a correct design - infinitely many.

1. Direct inverse network: learn y -> x from examples. For y = 0.5 it has seen (0.7, 0), (-0.7, 0),
   (0, 0.7) ... and learns their AVERAGE, which is near (0, 0) - a terrible design (y = 0).
2. Tandem (Liu et al. 2018): first learn the forward map x -> y; freeze it; then train the inverse
   network with loss = ( forward(inverse(y)) - y )^2.  It is judged on the RESPONSE, not on matching one
   particular design, so it is free to pick any one correct design and stick to it.

Run:  python learn/ex08_tandem_toy.py     (about 30 seconds)
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
torch.manual_seed(0)
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")


def f_true(x):            # works for numpy and torch
    return (x ** 2).sum(-1)


def mlp(n_in, n_out, out_act=None):
    layers = [torch.nn.Linear(n_in, 64), torch.nn.SiLU(), torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, n_out)]
    if out_act:
        layers.append(out_act)
    return torch.nn.Sequential(*layers)


def train(net, loss_fn, steps=3000, lr=2e-3):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    return loss.item()


# training data: random designs and their responses
X = torch.rand(3000, 2) * 2 - 1
Y = f_true(X)[:, None]

# 1) direct inverse: y -> x, loss on the design
direct = mlp(1, 2, torch.nn.Tanh())
train(direct, lambda: ((direct(Y) - X) ** 2).mean())

# 2) tandem: forward first, then inverse through the frozen forward
forward = mlp(2, 1)
train(forward, lambda: ((forward(X) - Y) ** 2).mean())
for p in forward.parameters():
    p.requires_grad_(False)
inverse = mlp(1, 2, torch.nn.Tanh())        # tanh keeps designs inside the square
train(inverse, lambda: ((forward(inverse(Y)) - Y) ** 2).mean())

T = torch.linspace(0.05, 1.0, 40)[:, None]   # targets
with torch.no_grad():
    xd, xt = direct(T), inverse(T)
yd, yt = f_true(xd.numpy()), f_true(xt.numpy())    # check designs with the TRUE function ("re-simulate")
print("target y   direct design -> achieved y     tandem design -> achieved y")
for i in range(0, 40, 6):
    print(f"  {T[i, 0]:.2f}     ({xd[i, 0]:+.2f}, {xd[i, 1]:+.2f}) -> {yd[i]:.2f}"
          f"            ({xt[i, 0]:+.2f}, {xt[i, 1]:+.2f}) -> {yt[i]:.2f}")
print(f"mean |achieved - target|: direct {np.abs(yd - T[:, 0].numpy()).mean():.3f}, tandem {np.abs(yt - T[:, 0].numpy()).mean():.3f}")

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
ax = axs[0]
for r in np.sqrt(T[::6, 0].numpy()):
    ax.add_patch(plt.Circle((0, 0), r, fill=False, color="0.8", lw=0.8))
ax.plot(xd[:, 0], xd[:, 1], "o", color="tab:red", ms=4, label="direct inverse: designs")
ax.plot(xt[:, 0], xt[:, 1], "s", color="tab:blue", ms=4, label="tandem: designs")
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)
ax.set_aspect("equal")
ax.set_title("designs proposed for 40 targets\n(grey circles = all correct designs for some targets)", fontsize=9)
ax.legend(fontsize=7, loc="lower left")
ax = axs[1]
ax.plot(T, yd, "o", color="tab:red", label="direct inverse")
ax.plot(T, yt, "s", color="tab:blue", label="tandem")
ax.plot([0, 1], [0, 1], "k", lw=0.8)
ax.set_xlabel("target y")
ax.set_ylabel("achieved y (true function)")
ax.set_title("does the design hit the target?", fontsize=9)
ax.legend(fontsize=8)
os.makedirs(FIG, exist_ok=True)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "ex08_tandem.png"), dpi=160)
print("saved learn/figs/ex08_tandem.png")
