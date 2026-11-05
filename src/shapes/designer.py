"""Shared design helpers for design.py and the Streamlit app: load the saved networks, predict band edges with any of
the three forward models, propose designs with any of the four inverse methods, check them, verify them with PWE,
draw them, and write COMSOL sweep files.

Forward models ("model" argument):
  "pwe"     network trained on the 6,000 PWE cells (PWE-level answers)
  "corr"    the same network + the COMSOL correction (a Gaussian process on COMSOL - network) -> COMSOL-level
  "comsol"  network trained directly on all COMSOL cells -> COMSOL-level (only inside the COMSOL-simulated shapes)

The networks are trained and saved once by src/shapes/save_models.py (models/optionb/), so designing takes seconds.
"""
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from geometry import A1, A2, check_cell, boundary  # noqa: E402
from models import DEV, Ensemble, CVAE, Tandem, mlp, multistart, p_to_feat, p_to_holes, p_to_u  # noqa: E402
from pwe import Solver  # noqa: E402
from gen_dataset import eps_from_validation  # noqa: E402
from run_optionb import _distinct  # noqa: E402

MODEL_DIR = "models/optionb"
METHODS = {"tandem": "Tandem network", "cvae": "Conditional VAE", "multistart": "Multi-start gradient",
           "cvae_refine": "CVAE + refine"}
MODELS = {"pwe": "PWE-trained network", "corr": "PWE network + COMSOL correction", "comsol": "COMSOL-trained network"}
NAMES = ["rA0", "c3A", "s3A", "c6A", "s6A", "rB0", "c3B", "s3B", "c6B", "s6B"]
COMSOL_REACH = 68.0  # um: further out a hole crosses the edge of the COMSOL unit cell


class LightGP:
    """Mean-only Gaussian-process predictor (kernel, training inputs, weights): small to save, fast to use."""

    def __init__(self, gpr, xm, xs):
        self.kernel, self.X, self.alpha = gpr.kernel_, gpr.X_train_, gpr.alpha_.ravel()
        self.ym, self.ys = float(np.ravel(gpr._y_train_mean)[0]), float(np.ravel(gpr._y_train_std)[0])
        self.xm, self.xs = xm, xs

    def predict(self, X):
        K = self.kernel((np.asarray(X, float) - self.xm) / self.xs, self.X)
        return K @ self.alpha * self.ys + self.ym


# ------------------------------------------------------------------------------------------- save / load
def _sd(net):
    return {k: v.detach().cpu() for k, v in net.state_dict().items()}


def _ens_dict(e):
    return {"nets": [_sd(n) for n in e.nets], "norm": dict(xm=e.xm, xs=e.xs, ym=e.ym, ys=e.ys)}


def save_bundle(fwd, tandem, cvae, corr, info, path=MODEL_DIR, fwd_comsol=None):
    os.makedirs(path, exist_ok=True)
    d = {"fwd": [_sd(n) for n in fwd.nets], "fwd_norm": dict(xm=fwd.xm, xs=fwd.xs, ym=fwd.ym, ys=fwd.ys),
         "tandem": _sd(tandem.net), "tandem_norm": dict(tm=tandem.tm, ts=tandem.ts),
         "cvae_enc": _sd(cvae.enc), "cvae_dec": _sd(cvae.dec), "cvae_norm": dict(tm=cvae.tm, ts=cvae.ts),
         "latent": cvae.latent, "info": info}
    if fwd_comsol is not None:
        d["fwd_comsol"] = _ens_dict(fwd_comsol)
    torch.save(d, os.path.join(path, "bundle.pt"))
    with open(os.path.join(path, "correction.pkl"), "wb") as f:
        pickle.dump(corr, f)


def _load_net(sd, i, o):
    net = mlp(i, o, 256).to(DEV)
    net.load_state_dict(sd)
    net.eval()
    for q in net.parameters():
        q.requires_grad_(False)
    return net


def _load_ens(nets, norm):
    e = Ensemble.__new__(Ensemble)
    e.nets = [_load_net(sd, 18, 2) for sd in nets]
    for k, v in norm.items():
        setattr(e, k, v)
    e._t = {k: torch.tensor(v, dtype=torch.float32, device=DEV) for k, v in norm.items()}
    return e


def load_bundle(path=MODEL_DIR):
    f = os.path.join(path, "bundle.pt")
    if not os.path.exists(f):
        raise FileNotFoundError(f"{f} not found: run  python src/shapes/save_models.py  once (about 10 minutes)")
    b = torch.load(f, map_location="cpu", weights_only=False)
    fwd = _load_ens(b["fwd"], b["fwd_norm"])
    fwd_c = _load_ens(b["fwd_comsol"]["nets"], b["fwd_comsol"]["norm"]) if "fwd_comsol" in b else None
    tan = Tandem.__new__(Tandem)
    tan.net = _load_net(b["tandem"], 2, 10)
    tan.tm, tan.ts = b["tandem_norm"]["tm"], b["tandem_norm"]["ts"]
    cv = CVAE.__new__(CVAE)
    cv.latent = b["latent"]
    cv.enc = _load_net(b["cvae_enc"], 12, 2 * cv.latent)
    cv.dec = _load_net(b["cvae_dec"], cv.latent + 2, 10)
    cv.tm, cv.ts = b["cvae_norm"]["tm"], b["cvae_norm"]["ts"]
    corr = None
    cf = os.path.join(path, "correction.pkl")
    if os.path.exists(cf):
        with open(cf, "rb") as fh:
            corr = pickle.load(fh)
    return {"fwd": fwd, "fwd_comsol": fwd_c, "tandem": tan, "cvae": cv, "corr": corr, "info": b.get("info", {})}


def available_models(B):
    return [m for m in MODELS if m == "pwe" or (m == "corr" and B["corr"] is not None)
            or (m == "comsol" and B["fwd_comsol"] is not None)]


_SOLVER = None


def solver():
    global _SOLVER
    if _SOLVER is None:
        _SOLVER = Solver(eps_si=eps_from_validation())
    return _SOLVER


# ------------------------------------------------------------------------------------------- predict / design
def _correction(B, feat):
    c = B["corr"]
    if "gps" in c and isinstance(c["gps"][0], LightGP):
        return np.stack([g.predict(feat) for g in c["gps"]], 1)
    z = (feat - c["xm"]) / c["xs"]                       # older format: sklearn GPs on normalised inputs
    return np.stack([g.predict(z) for g in c["gps"]], 1)


def predict(B, P, model="pwe", corrected=None):
    """Band edges (GHz) for designs P (n, 10): mean and ensemble spread for the chosen forward model."""
    if corrected is not None:
        model = "corr" if corrected else "pwe"
    feat = p_to_feat(np.asarray(P, float).reshape(-1, 10))
    return predict_features(B, feat, model)


def predict_features(B, feat, model="pwe"):
    """Same as predict, from the 18 Fourier features (any hole family)."""
    feat = np.asarray(feat, float).reshape(-1, 18)
    if model == "comsol" and B["fwd_comsol"] is not None:
        return B["fwd_comsol"].predict(feat)
    mu, sd = B["fwd"].predict(feat)
    if model == "corr" and B["corr"] is not None:
        mu = mu + _correction(B, feat)
    return mu, sd


def _refine_net(B, model):
    return B["fwd_comsol"] if (model == "comsol" and B["fwd_comsol"] is not None) else B["fwd"]


def propose(B, method, target, n=64, seed=0, model="pwe", target_net=None):
    """Candidate designs. `target` is used by the PWE-level generators (tandem, CVAE); `target_net` (default: target)
    is what the gradient refinement aims the chosen network at."""
    T = np.array([target], float)
    Tn = T if target_net is None else np.array([target_net], float)
    if method == "tandem":
        return B["tandem"].propose(T)[0]
    if method == "cvae":
        return B["cvae"].propose(T, n, seed=seed)[0]
    net = _refine_net(B, model)
    if method == "multistart":
        return multistart(net, Tn, n, seed=seed)[0]
    if method == "cvae_refine":
        c = B["cvae"].propose(T, n, seed=seed)
        return multistart(net, Tn, n, steps=100, U0=p_to_u(c.reshape(-1, 10)).reshape(1, n, 10), seed=seed)[0]
    raise ValueError(method)


def pwe_level_target(B, target, model, seed=0):
    """For COMSOL-aimed models: the PWE-level target that should land on `target` after the PWE-COMSOL offset."""
    T = np.asarray(target, float)
    if model == "pwe":
        return T
    c0 = B["cvae"].propose(T[None], 64, seed=seed)[0]
    return T - (predict(B, c0, model)[0] - predict(B, c0, "pwe")[0]).mean(0)


def design(B, target, methods=("cvae_refine",), k=4, corrected=None, comsol_only=True, verify=True, seed=0,
           n_candidates=64, model="pwe"):
    """Up to k distinct, fabricable designs per method, ranked by the chosen model's predicted error;
    optionally verified with the PWE solver."""
    if corrected is not None:
        model = "corr" if corrected else model
    out = []
    T = np.asarray(target, float)
    Tp = pwe_level_target(B, T, model, seed)            # where the PWE-level generators should aim
    Tn = Tp if model == "corr" else T                    # corr refines the PWE network; comsol refines its own network
    for m in methods:
        cands = propose(B, m, Tp, n_candidates, seed, model, target_net=Tn)
        pred, sd = predict(B, cands, model)
        order = np.argsort(np.abs(pred - T).max(1))
        us, got = [], 0
        for j in order[:400]:
            if got == (1 if m == "tandem" else k):
                break
            p = cands[j]
            ok, info = check_cell(*p_to_holes(p))
            reach = max(info["rmax_A"], info["rmax_B"])
            if not ok or (comsol_only and reach > COMSOL_REACH):
                continue
            u = p_to_u(p[None])[0]
            if not _distinct(u, us):
                continue
            us.append(u)
            got += 1
            d = {"method": m, "model": model, "p": p, "pred": pred[j], "spread": sd[j],
                 "err_pred": float(np.abs(pred[j] - T).max()), "wall": float(info["min_wall"]), "reach": float(reach),
                 "fits_comsol": reach <= COMSOL_REACH}
            if verify:
                v = np.array(solver().band_edges(*p_to_holes(p)))
                d["pwe"], d["err_pwe"] = v, float(np.abs(v - T).max())
            out.append(d)
    return out


def band_diagram(p, nk=31):
    ks = np.linspace(0, 3, nk)
    return ks, solver().bands(*p_to_holes(p), ks)


def check(p):
    ok, info = check_cell(*p_to_holes(np.asarray(p, float)))
    return ok, info


# ------------------------------------------------------------------------------------------- drawing / export
SILICON, AIR, CELL = "#4a5666", "#f8fafc", "#0d7a84"


def draw_holes(ax, hA, hB, tiles=1, title=None):
    """Top view of the crystal around one unit cell (dashed rhombus); hA, hB = (family, params)."""
    from matplotlib.patches import Polygon
    ax.set_facecolor(SILICON)
    R = range(-tiles - 1, tiles + 2)
    for i in R:
        for j in R:
            off = i * A1 + j * A2
            for site, h in (("A", hA), ("B", hB)):
                ax.add_patch(Polygon(boundary(*h, site, n=240) + off, closed=True, fc=AIR, ec="none"))
    rh = np.array([[0, 0], A1, A1 + A2, A2, [0, 0]])
    ax.plot(rh[:, 0], rh[:, 1], "--", color=CELL, lw=1.5)
    c = (A1 + A2) / 2
    half = 215 if tiles else 165
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def draw_cell(ax, p, tiles=1, title=None):
    hA, hB = p_to_holes(np.asarray(p, float))
    draw_holes(ax, hA, hB, tiles, title)


def comsol_sweep(P):
    """Text for COMSOL Parametric Sweep > Load from File (values in metres, every design at k = 1 and 2)."""
    P = np.asarray(P, float).reshape(-1, 10)
    lines = [f'{nm} "' + " ".join(f"{v:.6g}" for v in np.repeat(P[:, j] * 1e-6, 2)) + '"' for j, nm in enumerate(NAMES)]
    lines.append('k "' + " ".join(["1 2"] * len(P)) + '"')
    return "\n".join(lines) + "\n"
