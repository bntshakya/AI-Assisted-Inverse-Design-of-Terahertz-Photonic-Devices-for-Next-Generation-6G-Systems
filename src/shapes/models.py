"""Neural networks for option B: shape descriptor -> band edges, and the three inverse methods.

Design space for inverse design (10 numbers, both holes are C3 Fourier shapes):
  p = [r0, c3, s3, c6, s6]_A + [r0, c3, s3, c6, s6]_B   (um)
Networks work in u in (0, 1)^10 (u_to_p): r0 = 25 + 50 u, c3 = 0.35 r0 (2u - 1), c6 = 0.12 r0 (2u - 1), ...
so every output of a sigmoid is a shape inside the sampled box.

Forward surrogate (Ensemble): 18 Fourier features -> (f_lower, f_upper); 5 MLPs, mean = prediction,
spread = uncertainty. It accepts ANY hole family because every family maps to the same 18 features.

Inverse methods (all use the frozen forward surrogate and the same validity penalty):
  Tandem       one network target -> design, trained through the frozen surrogate (Liu et al. 2018)
  CVAE         conditional variational autoencoder: target + random latent z -> design; sampling z
               gives MANY different designs for one target (Sohn et al. 2015; Ma et al. 2019)
  multistart   gradient descent on the design itself from many random starts (no inverse network)
"""
import numpy as np
import torch
import torch.nn as nn

from geometry import A1, A2, FRAME_DEG, SITE_A, SITE_B

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
R0 = (25.0, 75.0)
A3MAX, A6MAX = 0.35, 0.12
N_DESIGN = 10
MIN_WALL, MIN_R, MIN_CURV_R = 25.0, 10.0, 8.0


# ------------------------------------------------------------------------- design <-> features
def u_to_p(u):
    lib = torch if torch.is_tensor(u) else np
    cols = []
    for h in (0, 5):
        r0 = R0[0] + (R0[1] - R0[0]) * u[:, h]
        cols += [r0] + [m * r0 * (2 * u[:, h + j] - 1) for j, m in ((1, A3MAX), (2, A3MAX), (3, A6MAX), (4, A6MAX))]
    return lib.stack(cols, 1) if lib is torch else np.stack(cols, 1)


def p_to_u(p):
    cols = []
    for h in (0, 5):
        r0 = p[:, h]
        cols += [(r0 - R0[0]) / (R0[1] - R0[0])] + [(p[:, h + j] / (m * r0) + 1) / 2
                                                    for j, m in ((1, A3MAX), (2, A3MAX), (3, A6MAX), (4, A6MAX))]
    return np.clip(np.stack(cols, 1), 1e-4, 1 - 1e-4)


def p_to_feat(p):
    lib = torch if torch.is_tensor(p) else np
    z = lib.zeros((p.shape[0], 4), dtype=p.dtype) if lib is np else torch.zeros(p.shape[0], 4, device=p.device)
    return lib.cat([p[:, :5], z, p[:, 5:], z], 1) if lib is torch else np.concatenate([p[:, :5], z, p[:, 5:], z], 1)


def p_to_holes(p_row):
    keys = ["r0", "c3", "s3", "c6", "s6"]
    return (("fourier", dict(zip(keys, map(float, p_row[:5])))), ("fourier", dict(zip(keys, map(float, p_row[5:])))))


# ------------------------------------------------------------------------- differentiable validity
_TH = torch.linspace(0, 2 * np.pi, 73)[:-1]
_LAT = [torch.tensor(v, dtype=torch.float32) for v in (A1, A2, A2 - A1)]


def _r_and_curv(ph, th):
    r0, c3, s3, c6, s6 = (ph[:, i:i + 1] for i in range(5))
    c3t, s3t, c6t, s6t = torch.cos(3 * th), torch.sin(3 * th), torch.cos(6 * th), torch.sin(6 * th)
    r = r0 + c3 * c3t + s3 * s3t + c6 * c6t + s6 * s6t
    dr = -3 * c3 * s3t + 3 * s3 * c3t - 6 * c6 * s6t + 6 * s6 * c6t
    d2r = -9 * c3 * c3t - 9 * s3 * s3t - 36 * c6 * c6t - 36 * s6 * s6t
    curv = (r ** 2 + 2 * dr ** 2 - r * d2r) / (r ** 2 + dr ** 2).clamp_min(1e-6) ** 1.5
    return r, curv


def _boundary(ph, site):
    th = _TH.to(ph.device)
    r, _ = _r_and_curv(ph, th)
    c = torch.tensor(SITE_A if site == "A" else SITE_B, dtype=torch.float32, device=ph.device)
    ang = th + np.deg2rad(FRAME_DEG[site])
    return torch.stack([c[0] + r * torch.cos(ang), c[1] + r * torch.sin(ang)], -1)


def penalty(p, margin=2.0):
    """Soft version of geometry.check_cell (um^2 scale); 0 when the cell is comfortably valid."""
    a, b = _boundary(p[:, :5], "A"), _boundary(p[:, 5:], "B")
    walls = [torch.cdist(a, b).amin((1, 2))]
    for lat in _LAT:
        lat = lat.to(p.device)
        walls += [torch.cdist(a, a + lat).amin((1, 2)), torch.cdist(b, b + lat).amin((1, 2))]
    wall = torch.stack(walls, 1).amin(1)
    th = _TH.to(p.device)
    rA, kA = _r_and_curv(p[:, :5], th)
    rB, kB = _r_and_curv(p[:, 5:], th)
    rmin = torch.minimum(rA.amin(1), rB.amin(1))
    kmax = torch.maximum(kA.amax(1), kB.amax(1))
    pen = torch.relu(MIN_WALL + margin - wall) ** 2 + torch.relu(MIN_R + margin - rmin) ** 2
    pen = pen + (100 * torch.relu(kmax - 1 / (MIN_CURV_R + margin))) ** 2
    return pen


# ------------------------------------------------------------------------- forward surrogate
def mlp(i, o, h=256, depth=3):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


def _train(net, params, loss_fn, n, epochs, batch, lr, seed):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    steps = epochs * max(1, (n + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            loss = loss_fn(perm[i:i + batch].to(DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()


class Ensemble:
    def __init__(self, n_members=5, hidden=256, epochs=400, batch=128, lr=2e-3):
        self.n_members, self.hidden, self.epochs, self.batch, self.lr = n_members, hidden, epochs, batch, lr

    def fit(self, X, Y, seed=0):
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-3
        self.ym, self.ys = Y.mean(0), Y.std(0)
        Xn = torch.tensor((X - self.xm) / self.xs, dtype=torch.float32, device=DEV)
        Yn = torch.tensor((Y - self.ym) / self.ys, dtype=torch.float32, device=DEV)
        self.nets = []
        for m in range(self.n_members):
            torch.manual_seed(seed * 100 + m)
            net = mlp(X.shape[1], Y.shape[1], self.hidden).to(DEV)
            _train(net, net.parameters(), lambda idx: ((net(Xn[idx]) - Yn[idx]) ** 2).mean(),
                   len(Xn), self.epochs, self.batch, self.lr, seed * 100 + m)
            net.eval()
            for q in net.parameters():
                q.requires_grad_(False)
            self.nets.append(net)
        self._t = {k: torch.tensor(v, dtype=torch.float32, device=DEV) for k, v in
                   dict(xm=self.xm, xs=self.xs, ym=self.ym, ys=self.ys).items()}
        return self

    def torch_forward(self, feat):
        """feat (n, 18) in um (torch) -> mean prediction in GHz (torch, differentiable)."""
        t = self._t
        xn = (feat - t["xm"]) / t["xs"]
        return torch.stack([n(xn) for n in self.nets]).mean(0) * t["ys"] + t["ym"]

    def predict(self, X):
        with torch.no_grad():
            xn = torch.tensor((X - self.xm) / self.xs, dtype=torch.float32, device=DEV)
            out = torch.stack([n(xn) for n in self.nets]).cpu().numpy() * self.ys + self.ym
        return out.mean(0), out.std(0)


# ------------------------------------------------------------------------- inverse methods
def _design_loss(fwd, u, t, ys, lam):
    p = u_to_p(u)
    err = ((fwd.torch_forward(p_to_feat(p)) - t) / ys) ** 2
    return err.sum(1), lam * penalty(p)


class Tandem:
    """Inverse network trained through the frozen forward surrogate. One design per target."""

    def __init__(self, lam=1e-3, epochs=300, batch=128, lr=2e-3):
        self.lam, self.epochs, self.batch, self.lr = lam, epochs, batch, lr

    def fit(self, fwd, T, seed=0):
        torch.manual_seed(seed)
        self.tm, self.ts = T.mean(0), T.std(0)
        Tt = torch.tensor(T, dtype=torch.float32, device=DEV)
        tn = (Tt - torch.tensor(self.tm, device=DEV, dtype=torch.float32)) / torch.tensor(self.ts, device=DEV, dtype=torch.float32)
        ys = torch.tensor(fwd.ys, dtype=torch.float32, device=DEV)
        self.net = mlp(2, N_DESIGN, 256).to(DEV)

        def loss_fn(idx):
            u = torch.sigmoid(self.net(tn[idx]))
            e, pen = _design_loss(fwd, u, Tt[idx], ys, self.lam)
            return (e + pen).mean()

        _train(self.net, self.net.parameters(), loss_fn, len(Tt), self.epochs, self.batch, self.lr, seed)
        return self

    def propose(self, T):
        with torch.no_grad():
            tn = torch.tensor((T - self.tm) / self.ts, dtype=torch.float32, device=DEV)
            return u_to_p(torch.sigmoid(self.net(tn))).cpu().numpy()[:, None, :]  # (n_targets, 1, 10)


class CVAE:
    """p(design | target): encoder q(z | u, t), decoder u = D(z, t). Sample z ~ N(0, I) to get many designs."""

    def __init__(self, latent=4, beta=0.5, epochs=500, batch=128, lr=2e-3):
        self.latent, self.beta, self.epochs, self.batch, self.lr = latent, beta, epochs, batch, lr

    def fit(self, U, T, seed=0):
        torch.manual_seed(seed)
        self.tm, self.ts = T.mean(0), T.std(0)
        Ut = torch.tensor(U, dtype=torch.float32, device=DEV)
        tn = torch.tensor((T - self.tm) / self.ts, dtype=torch.float32, device=DEV)
        self.enc = mlp(N_DESIGN + 2, 2 * self.latent, 256).to(DEV)
        self.dec = mlp(self.latent + 2, N_DESIGN, 256).to(DEV)

        def loss_fn(idx):
            mu, logv = self.enc(torch.cat([Ut[idx], tn[idx]], 1)).chunk(2, 1)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logv)
            u = torch.sigmoid(self.dec(torch.cat([z, tn[idx]], 1)))
            rec = ((u - Ut[idx]) ** 2).sum(1) / 0.01  # unit-variance-ish scaling of u errors
            kl = 0.5 * (mu ** 2 + logv.exp() - 1 - logv).sum(1)
            return (rec + self.beta * kl).mean()

        _train(self.enc, list(self.enc.parameters()) + list(self.dec.parameters()), loss_fn,
               len(Ut), self.epochs, self.batch, self.lr, seed)
        return self

    def propose(self, T, n_samples=64, seed=0):
        g = torch.Generator(device=DEV).manual_seed(seed)
        with torch.no_grad():
            tn = torch.tensor((T - self.tm) / self.ts, dtype=torch.float32, device=DEV)
            tn = tn.repeat_interleave(n_samples, 0)
            z = torch.randn(len(tn), self.latent, generator=g, device=DEV)
            u = torch.sigmoid(self.dec(torch.cat([z, tn], 1)))
            return u_to_p(u).cpu().numpy().reshape(len(T), n_samples, N_DESIGN)


def multistart(fwd, T, n_starts=64, steps=300, lr=0.05, lam=1e-3, seed=0, U0=None):
    """Gradient descent on the design (logit of u) through the surrogate, from many starts.
    U0 (n_targets, n_starts, 10) can seed it, e.g. with CVAE samples ("CVAE + refine")."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    n = len(T)
    Tt = torch.tensor(T, dtype=torch.float32, device=DEV).repeat_interleave(n_starts, 0)
    ys = torch.tensor(fwd.ys, dtype=torch.float32, device=DEV)
    if U0 is None:
        u0 = torch.rand(n * n_starts, N_DESIGN, generator=g, device=DEV) * 0.9 + 0.05
    else:
        u0 = torch.tensor(U0.reshape(-1, N_DESIGN), dtype=torch.float32, device=DEV).clamp(0.02, 0.98)
    logit = torch.log(u0 / (1 - u0)).requires_grad_(True)
    opt = torch.optim.Adam([logit], lr=lr)
    for _ in range(steps):
        e, pen = _design_loss(fwd, torch.sigmoid(logit), Tt, ys, lam)
        loss = (e + pen).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = u_to_p(torch.sigmoid(logit))
    return p.cpu().numpy().reshape(n, n_starts, N_DESIGN)
