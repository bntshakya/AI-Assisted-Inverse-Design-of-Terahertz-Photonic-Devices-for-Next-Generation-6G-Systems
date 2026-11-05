"""Exercise 9 - a conditional variational autoencoder (CVAE): MANY designs for one target.

Same toy as exercise 8 (design x in [-1,1]^2, response y = x1^2 + x2^2).
The tandem gives ONE design per target. A CVAE learns the whole SET of designs that give y:
  encoder:  (x, y) -> a short code z (here 1 number) that says "which of the correct designs"
  decoder:  (z, y) -> x
Training makes the codes look like a standard normal distribution. Afterwards we throw the encoder
away, draw random codes z ~ N(0, 1) and decode them with the target y: every z gives a different design.

Run:  python learn/ex09_cvae_toy.py     (about 40 seconds)
Try:  BETA = 0.5 (codes squeezed too hard: designs miss the target) or 0.002; LATENT = 2.
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
LATENT, BETA, STEPS = 1, 0.02, 8000


def mlp(n_in, n_out):
    return torch.nn.Sequential(torch.nn.Linear(n_in, 64), torch.nn.SiLU(), torch.nn.Linear(64, 64), torch.nn.SiLU(),
                               torch.nn.Linear(64, n_out))


enc = mlp(3, 2 * LATENT)          # (x1, x2, y) -> (mean, log-variance) of z
dec = mlp(LATENT + 1, 2)          # (z, y) -> (x1, x2)
opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=2e-3)
X = torch.rand(4000, 2) * 2 - 1
Y = (X ** 2).sum(1, keepdim=True)
for step in range(STEPS):
    idx = torch.randint(0, len(X), (256,))
    x, y = X[idx], Y[idx]
    mu, logvar = enc(torch.cat([x, y], 1)).chunk(2, 1)
    z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)         # the "reparameterisation trick"
    xr = dec(torch.cat([z, y], 1))
    recon = ((xr - x) ** 2).sum(1).mean()                           # rebuild the design
    kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(1).mean()   # keep codes close to N(0, 1)
    loss = recon + BETA * kl
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 1500 == 0:
        print(f"step {step:5d}: reconstruction {recon.item():.4f}, KL {kl.item():.3f}")

fig, ax = plt.subplots(figsize=(5, 5))
print("\ntarget y   designs drawn   mean |achieved y - target|   spread of designs (angle range, degrees)")
for t, col in ((0.15, "tab:red"), (0.5, "tab:blue"), (0.85, "tab:green")):
    with torch.no_grad():
        z = torch.randn(300, LATENT)
        xs = dec(torch.cat([z, torch.full((300, 1), t)], 1)).numpy()
    ach = (xs ** 2).sum(1)
    ang = np.degrees(np.arctan2(xs[:, 1], xs[:, 0]))
    print(f"   {t:.2f}        300           {np.abs(ach - t).mean():.3f}                       {np.ptp(ang):.0f}")
    ax.plot(xs[:, 0], xs[:, 1], ".", ms=3, color=col, label=f"300 CVAE designs for y = {t}")
    ax.add_patch(plt.Circle((0, 0), np.sqrt(t), fill=False, color=col, lw=0.8, ls="--"))
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)
ax.set_aspect("equal")
ax.set_title("CVAE: many different designs per target\n(dashed = all correct designs)", fontsize=9)
ax.legend(fontsize=7, loc="lower left")
os.makedirs(FIG, exist_ok=True)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "ex09_cvae.png"), dpi=160)
print("saved learn/figs/ex09_cvae.png")
