"""Inverse-design methods.

x = (L1, L2[, corner]): the first two columns are the free geometry; any further columns
(corner type) are a condition the designer fixes, so inverse methods take it as input and
only design (L1, L2).
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import least_squares
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

N_GEO = 2  # designable columns of x


class Norm:
    def __init__(self, a):
        self.m, self.s = a.mean(0), a.std(0) + 1e-12

    def fwd(self, a):
        return (a - self.m) / self.s

    def inv(self, a):
        return a * self.s + self.m


def _t(a):
    return torch.tensor(a, dtype=torch.float32)


def _mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))


def _fit(params, loss_fn, epochs=3000, lr=5e-3):
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn().backward()
        opt.step()
        sched.step()


def _target_fn(task):
    """task 'edges': target = (f_lower, f_upper); task 'gap': target = f_upper - f_lower."""
    if task == "edges":
        return lambda y: y
    return lambda y: y[..., 1:2] - y[..., 0:1]


class BoundedInverse(nn.Module):
    """(target, condition) -> geometry, squashed into the training box (normalised coords)."""

    def __init__(self, i, lo, hi):
        super().__init__()
        self.net = _mlp(i, N_GEO)
        self.lo, self.hi = lo, hi

    def forward(self, tc):
        return self.lo + (self.hi - self.lo) * torch.sigmoid(self.net(tc))


class NNForward:
    """Forward surrogate x -> (f_lower, f_upper); frozen after training."""

    def __init__(self, x, y, seed):
        torch.manual_seed(seed)
        self.xn, self.yn = Norm(x), Norm(y)
        X, Y = _t(self.xn.fwd(x)), _t(self.yn.fwd(y))
        self.net = _mlp(x.shape[1], 2)
        _fit(self.net.parameters(), lambda: ((self.net(X) - Y) ** 2).mean())
        for p in self.net.parameters():
            p.requires_grad_(False)

    def torch_predict(self, xn):  # normalised in, normalised out
        return self.net(xn)


class _NNInverse:
    """Shared plumbing: normalised (target, condition) in, physical x (geometry + condition) out."""

    def _setup(self, x, y, task, xn):
        self.tf = _target_fn(task)
        t = self.tf(y)
        self.tn, self.xn = Norm(t), xn
        Xn = _t(xn.fwd(x))
        self.T, self.Xg, self.C = _t(self.tn.fwd(t)), Xn[:, :N_GEO], Xn[:, N_GEO:]
        self.net = BoundedInverse(self.T.shape[1] + self.C.shape[1],
                                  self.Xg.min(0).values, self.Xg.max(0).values)
        return t

    def propose(self, target, cond):
        cn = _t(self.xn.fwd(np.c_[np.zeros((len(cond), N_GEO)), cond])[:, N_GEO:])
        with torch.no_grad():
            g = self.net(torch.cat([_t(self.tn.fwd(target)), cn], 1))
        x = self.xn.inv(torch.cat([g, cn], 1).numpy())
        return [s[None] for s in x]


class DirectInverse(_NNInverse):
    name = "Direct inverse NN"

    def __init__(self, x, y, task, seed, fwd=None):
        torch.manual_seed(seed)
        self._setup(x, y, task, Norm(x))
        TC = torch.cat([self.T, self.C], 1)
        _fit(self.net.parameters(), lambda: ((self.net(TC) - self.Xg) ** 2).mean())


class TandemInverse(_NNInverse):
    name = "Tandem NN (Liu 2018)"

    def __init__(self, x, y, task, seed, fwd):
        torch.manual_seed(seed + 1)
        t = self._setup(x, y, task, fwd.xn)
        TC = torch.cat([self.T, self.C], 1)
        ym, ys = _t(fwd.yn.m), _t(fwd.yn.s)
        t_phys, var = _t(t), float(t.var()) + 1e-9

        def loss():
            xn = torch.cat([self.net(TC), self.C], 1)
            yp = fwd.torch_predict(xn) * ys + ym  # physical GHz
            return ((self.tf(yp) - t_phys) ** 2).mean() / var

        _fit(self.net.parameters(), loss)


def make_gp(x, y):
    xn = Norm(x)
    k = ConstantKernel(1.0) * RBF(np.ones(x.shape[1]), (1e-2, 1e2)) + WhiteKernel(1e-6, (1e-10, 1e-1))
    gps = [GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=3, random_state=0)
           .fit(xn.fwd(x), y[:, j]) for j in range(y.shape[1])]

    def predict(xq):
        z = xn.fwd(np.atleast_2d(xq))
        return np.stack([g.predict(z) for g in gps], 1)

    return predict


class GPGradient:
    name = "GP + gradient (multi-start)"

    def __init__(self, x, y, task, seed, fwd=None, n_starts=12):
        self.pred, self.tf, self.n_starts = make_gp(x, y), _target_fn(task), n_starts
        self.gn = Norm(x[:, :N_GEO])
        self.lo, self.hi = self.gn.fwd(x[:, :N_GEO].min(0)), self.gn.fwd(x[:, :N_GEO].max(0))
        self.rng = np.random.default_rng(seed)
        self.scale = self.tf(y).std(0) + 1e-9

    def propose(self, target, cond):
        """Per target: every multi-start solution that reaches the best optimum.

        Least-squares in normalised coordinates; plain L-BFGS on the squared error
        stalls on this ill-conditioned objective (3 GHz errors vs 0.01 GHz here).
        """
        out = []
        for t, c in zip(target, cond):
            full = lambda z: np.r_[self.gn.inv(z), c]
            res = lambda z: ((self.tf(self.pred(full(z))) - t) / self.scale).ravel()
            runs = [least_squares(res, self.rng.uniform(self.lo, self.hi), bounds=(self.lo, self.hi),
                                  xtol=1e-12, ftol=1e-12, gtol=1e-12) for _ in range(self.n_starts)]
            best = min(r.cost for r in runs)
            out.append(np.array([full(r.x) for r in runs if r.cost <= best + 1e-8]))
        return out
