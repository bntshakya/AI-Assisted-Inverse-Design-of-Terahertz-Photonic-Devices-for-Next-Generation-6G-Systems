"""Option B experiments: freeform C3 hole shapes -> band edges, and back.

Run from the Project folder (after gen_dataset.py and validate_pwe.py):
  python src/shapes/run_optionb.py                      # everything (~15 min on a laptop GPU)
  python src/shapes/run_optionb.py --showcase 300 330   # target band edges for the design gallery (GHz)

Stages
 1 forward   surrogate (18 Fourier features -> f_lower, f_upper) trained on PWE data (fourier + library),
             learning curve vs a Gaussian-process baseline, accuracy on held-out PWE cells and on the
             128 COMSOL cells (both old datasets, never seen in training).
 2 mf        multi-fidelity: learn the COMSOL - PWE difference with a GP on the 18 features, using the
             COMSOL datasets (rounded + sharp, + any extra_*.csv such as the circle cells).
 3 inverse   inverse design in the 10-number Fourier space for 100 held-out targets: tandem NN,
             conditional VAE, multi-start gradient search, CVAE + refine. Every chosen design is checked
             for fabrication validity and re-solved with the PWE solver (the "truth" at low fidelity).
 4 showcase  many different shapes for ONE target, verified, exported for COMSOL.
Writes results/optionb/*.json|npz and data/pwe/showcase_for_comsol.txt.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

sys.path.insert(0, os.path.dirname(__file__))
from dataset import holes, load_comsol, load_pwe  # noqa: E402
from gen_dataset import eps_from_validation  # noqa: E402
from geometry import FEATURE_NAMES, check_cell  # noqa: E402
from models import (CVAE, Ensemble, Tandem, multistart, p_to_feat, p_to_holes, p_to_u,  # noqa: E402
                    u_to_p)
from pwe import Solver  # noqa: E402

OUT = "results/optionb"
SEED = 0  # set by --seed: data split, network initialisation and test targets
P_IDX = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]  # Fourier design numbers inside the 18 features
TOL_GHZ = 1.0


def split(n, seed=None):
    seed = SEED if seed is None else seed
    perm = np.random.default_rng(seed).permutation(n)
    a, b = int(0.8 * n), int(0.9 * n)
    return perm[:a], perm[a:b], perm[b:]


def mae(a, b):
    return np.abs(np.asarray(a) - np.asarray(b)).mean(0).round(3).tolist()


def gp_fit_predict(Xtr, ytr, Xte):
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-3
    out = []
    for j in range(ytr.shape[1]):
        k = ConstantKernel(1.0) * RBF(np.ones(Xtr.shape[1]), (1e-2, 1e3)) + WhiteKernel(1e-3, (1e-8, 1e1))
        gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=0, random_state=0)
        gp.fit((Xtr - xm) / xs, ytr[:, j])
        out.append(gp.predict((Xte - xm) / xs))
    return np.stack(out, 1)


# ----------------------------------------------------------------------------------- 1 forward
def stage_forward(pwe, com, res):
    X, Y = pwe[FEATURE_NAMES].to_numpy(float), pwe[["f_lower", "f_upper"]].to_numpy(float)
    tr, va, te = split(len(pwe))
    trv = np.r_[tr, va]
    curve = []
    for n in (125, 250, 500, 1000, 2000, len(trv)):
        idx = trv[:n]
        steps = int(np.ceil(n / 128))
        t0 = time.time()
        nn_pred = Ensemble(n_members=3, epochs=int(np.clip(8000 / steps, 150, 3000))).fit(X[idx], Y[idx], seed=SEED).predict(X[te])[0]
        row = {"n_train": int(n), "nn_mae": mae(nn_pred, Y[te]), "nn_s": round(time.time() - t0, 1)}
        if n <= 1000:
            t0 = time.time()
            row["gp_mae"] = mae(gp_fit_predict(X[idx], Y[idx], X[te]), Y[te])
            row["gp_s"] = round(time.time() - t0, 1)
        curve.append(row)
        print("  learning curve", row, flush=True)
    res["learning_curve"] = curve

    t0 = time.time()
    fwd = Ensemble(n_members=5, epochs=400).fit(X[trv], Y[trv], seed=SEED)
    res["forward_train_s"] = round(time.time() - t0, 1)
    mu, sd = fwd.predict(X[te])
    gapped = pwe.gap.to_numpy()[te] > 0.1
    res["forward_test"] = {
        "n_test": int(len(te)), "mae_all": mae(mu, Y[te]), "mae_gapped": mae(mu[gapped], Y[te][gapped]),
        "max_err": np.abs(mu - Y[te]).max(0).round(2).tolist(),
        "gap_mae": float(np.abs((mu[:, 1] - mu[:, 0]) - (Y[te, 1] - Y[te, 0])).mean().round(3)),
        "by_dataset": {d: mae(mu[pwe.dataset.to_numpy()[te] == d], Y[te][pwe.dataset.to_numpy()[te] == d])
                       for d in pwe.dataset.unique()},
        "corr_std_abs_err": float(np.corrcoef(sd.mean(1), np.abs(mu - Y[te]).mean(1))[0, 1].round(3))}
    print("  forward test", res["forward_test"], flush=True)
    np.savez(f"{OUT}/forward_parity.npz", y=Y[te], mu=mu, sd=sd, dataset=pwe.dataset.to_numpy()[te])
    return fwd, tr, va, te


# ----------------------------------------------------------------------------------- 2 multi-fidelity
def stage_mf(fwd, com, res):
    solver = Solver(eps_si=eps_from_validation())
    com = com[com.gap > 0.1].reset_index(drop=True)
    pw = np.array([solver.band_edges(*holes(r)) for _, r in com.iterrows()])
    Xc = com[FEATURE_NAMES].to_numpy(float)
    Yc = com[["f_lower", "f_upper"]].to_numpy(float)
    nn = fwd.predict(Xc)[0]
    delta = Yc - pw
    fam = com.dataset.to_numpy()
    out = {"n": int(len(com)), "datasets": {d: int((fam == d).sum()) for d in np.unique(fam)}}

    # 10-fold CV of the correction GP
    rng = np.random.default_rng(0)
    folds = rng.permutation(len(com)) % 10
    corr = np.zeros_like(delta)
    for f in range(10):
        trn, tst = folds != f, folds == f
        corr[tst] = gp_fit_predict(Xc[trn], delta[trn], Xc[tst])
    out["pwe_raw_mae"] = mae(pw, Yc)
    out["pwe_plus_gp_cv_mae"] = mae(pw + corr, Yc)
    out["nn_raw_mae"] = mae(nn, Yc)
    out["nn_plus_gp_cv_mae"] = mae(nn + corr, Yc)
    out["gap_mae"] = {"pwe_raw": float(np.abs((pw[:, 1] - pw[:, 0]) - com.gap).mean().round(3)),
                      "pwe_plus_gp": float(np.abs((pw + corr)[:, 1] - (pw + corr)[:, 0] - com.gap).mean().round(3))}
    # leave one family out: is the correction transferable to a shape family it has not seen?
    lofo = {}
    for d in np.unique(fam):
        trn, tst = fam != d, fam == d
        if trn.sum() < 10:
            continue
        c = gp_fit_predict(Xc[trn], delta[trn], Xc[tst])
        lofo[d] = {"pwe_raw_mae": mae(pw[tst], Yc[tst]), "pwe_plus_gp_mae": mae(pw[tst] + c, Yc[tst])}
    out["leave_family_out"] = lofo
    res["multifidelity"] = out
    print("  multi-fidelity", json.dumps(out), flush=True)
    np.savez(f"{OUT}/mf.npz", y=Yc, pwe=pw, nn=nn, corr=corr, dataset=fam)


# ----------------------------------------------------------------------------------- 3 inverse
def _variants(u):
    """Designs with identical band edges: swap holes A <-> B, and mirror (all sine terms flip)."""
    sw = np.r_[u[5:], u[:5]]
    mir = lambda v: np.where(np.isin(np.arange(10), [2, 4, 7, 9]), 1 - v, v)
    return [u, sw, mir(u), mir(sw)]


def _distinct(u, chosen, min_dist=0.08):
    return all(min(np.linalg.norm(v - c) for v in _variants(u)) > min_dist for c in chosen)


def select(cands, T, fwd, k=1, max_checks=64):
    """Per target: rank candidates by surrogate error, keep the first k that pass the exact
    validity check and are distinct. Returns list (per target) of (p, predicted (f_lower, f_upper))."""
    n_t, n_s, _ = cands.shape
    pred = fwd.predict(p_to_feat(cands.reshape(-1, 10)))[0].reshape(n_t, n_s, 2)
    err = np.abs(pred - T[:, None]).max(-1)
    picks = []
    for i in range(n_t):
        got, us = [], []
        for j in np.argsort(err[i])[:max_checks]:
            p = cands[i, j]
            if not check_cell(*p_to_holes(p))[0]:
                continue
            u = p_to_u(p[None])[0]
            if _distinct(u, us):
                got.append((p, pred[i, j]))
                us.append(u)
            if len(got) == k:
                break
        picks.append(got)
    return picks


def verify(solver, p):
    return np.array(solver.band_edges(*p_to_holes(p)))


def stage_inverse(fwd, pwe, tr, te, res, args):
    solver = Solver(eps_si=eps_from_validation())
    is4 = (pwe.dataset == "pwe_fourier").to_numpy()
    gap = pwe.gap.to_numpy()
    X, Y = pwe[FEATURE_NAMES].to_numpy(float), pwe[["f_lower", "f_upper"]].to_numpy(float)
    rng = np.random.default_rng(1 + SEED)
    te_t = te[is4[te] & (gap[te] > 2.0)]
    te_t = rng.choice(te_t, size=min(100, len(te_t)), replace=False)
    T = Y[te_t]
    tr_g = tr[gap[tr] > 0.5]
    tr4 = tr[is4[tr]]

    t0 = time.time()
    tandem = Tandem().fit(fwd, Y[tr_g], seed=SEED)
    t_tandem = time.time() - t0
    t0 = time.time()
    cvae = CVAE().fit(p_to_u(X[tr4][:, P_IDX]), Y[tr4], seed=SEED)
    t_cvae = time.time() - t0
    print(f"  trained tandem ({t_tandem:.0f} s) and CVAE ({t_cvae:.0f} s)", flush=True)

    methods = {}
    t0 = time.time(); methods["tandem"] = (tandem.propose(T), time.time() - t0)
    t0 = time.time(); c = cvae.propose(T, 64, seed=SEED); methods["cvae"] = (c, time.time() - t0)
    t0 = time.time(); methods["multistart"] = (multistart(fwd, T, 64, seed=SEED), time.time() - t0)
    t0 = time.time()
    methods["cvae_refine"] = (multistart(fwd, T, 64, steps=100, U0=p_to_u(c.reshape(-1, 10)).reshape(c.shape), seed=SEED),
                              time.time() - t0 + methods["cvae"][1])

    inv, per_target = {}, {}
    for name, (cands, secs) in methods.items():
        picks = select(cands, T, fwd, k=1)
        errs, valid = [], 0
        for i, got in enumerate(picks):
            if not got:
                errs.append([np.nan, np.nan])
                continue
            valid += 1
            errs.append(np.abs(verify(solver, got[0][0]) - T[i]))
        errs = np.array(errs)
        ok = ~np.isnan(errs[:, 0])
        inv[name] = {"valid_design_found": f"{valid}/{len(T)}",
                     "verified_mae_GHz": np.nanmean(errs, 0).round(3).tolist(),
                     "verified_median_GHz": np.nanmedian(errs, 0).round(3).tolist(),
                     "success_rate_within_1GHz": float(((errs.max(1) < TOL_GHZ) & ok).mean().round(3)),
                     "seconds_per_target": round(secs / len(T), 4)}
        per_target[name] = errs
        print(f"  {name}: {inv[name]}", flush=True)

    # diversity: how many DIFFERENT verified designs per target (first 10 targets)
    for name in ("cvae", "multistart", "cvae_refine"):
        cands = methods[name][0][:10]
        picks = select(cands, T[:10], fwd, k=8)
        counts = []
        for i, got in enumerate(picks):
            counts.append(sum(np.abs(verify(solver, p) - T[i]).max() < TOL_GHZ for p, _ in got))
        inv[name]["distinct_verified_designs_per_target"] = float(np.mean(counts))
        print(f"  {name} diversity: {counts}", flush=True)
    res["inverse"] = inv
    np.savez(f"{OUT}/inverse_errors.npz", T=T, **per_target)
    return cvae


# ----------------------------------------------------------------------------------- 4 showcase
def stage_showcase(fwd, cvae, res, target):
    solver = Solver(eps_si=eps_from_validation())
    T = np.array([target], dtype=float)
    c = cvae.propose(T, 512, seed=7)
    ms = multistart(fwd, T, 256, seed=7)
    ref = multistart(fwd, T, 256, steps=150, U0=p_to_u(c[:, :256].reshape(-1, 10)).reshape(1, 256, 10), seed=7)
    cands = np.concatenate([c, ms, ref], 1)
    picks = select(cands, T, fwd, k=24, max_checks=600)[0]
    designs = []
    for p, pred in picks:
        v = verify(solver, p)
        if np.abs(v - T[0]).max() < TOL_GHZ:
            ok, info = check_cell(*p_to_holes(p))
            designs.append({"p": np.round(p, 3).tolist(), "pred": np.round(pred, 2).tolist(),
                            "pwe": np.round(v, 2).tolist(), "min_wall": round(float(info["min_wall"]), 1)})
    res["showcase"] = {"target": list(target), "n_candidates": int(cands.shape[1]),
                       "n_verified_within_1GHz": len(designs), "designs": designs}
    print(f"  showcase {target}: {len(designs)} distinct verified designs", flush=True)

    # COMSOL verification file (Parametric Sweep > Load from File, Specified combinations), metres
    names = ["rA0", "c3A", "s3A", "c6A", "s6A", "rB0", "c3B", "s3B", "c6B", "s6B"]
    P = np.array([d["p"] for d in designs]).reshape(-1, 10)
    if len(P):
        ks = (1.0, 2.0)
        with open("data/pwe/showcase_for_comsol.txt", "w") as f:
            for j, nm in enumerate(names):
                f.write(f'{nm} "{" ".join(f"{v * 1e-6:.6g}" for v in np.repeat(P[:, j], len(ks)))}"\n')
            f.write(f'k "{" ".join(str(k) for k in np.tile(ks, len(P)))}"\n')
        pd.DataFrame(P, columns=[n + "_um" for n in names]).assign(
            f_lower_pwe=[d["pwe"][0] for d in designs], f_upper_pwe=[d["pwe"][1] for d in designs]
        ).to_csv("data/pwe/showcase_designs.csv", index_label="design")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--showcase", type=float, nargs=2, default=(300.0, 330.0), metavar=("F_LOWER", "F_UPPER"))
    ap.add_argument("--seed", type=int, default=0, help="repeat with a different split / initialisation")
    ap.add_argument("--out", default=None, help="output folder (default results/optionb)")
    ap.add_argument("--no-showcase", action="store_true", help="skip the gallery (keeps data/pwe files untouched)")
    args = ap.parse_args()
    global OUT, SEED
    SEED = args.seed
    OUT = args.out or OUT
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    pwe, com = load_pwe(), load_comsol()
    print(f"PWE cells: {pwe.dataset.value_counts().to_dict()}; COMSOL cells: {com.dataset.value_counts().to_dict()}")
    res = {"device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")}
    t0 = time.time()
    fwd, tr, va, te = stage_forward(pwe, com, res)
    stage_mf(fwd, com, res)
    cvae = stage_inverse(fwd, pwe, np.r_[tr, va], te, res, args)
    if not args.no_showcase:
        stage_showcase(fwd, cvae, res, args.showcase)
    res["seed"] = SEED
    res["total_minutes"] = round((time.time() - t0) / 60, 1)
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"done in {res['total_minutes']} min -> {OUT}/results.json")


if __name__ == "__main__":
    main()
