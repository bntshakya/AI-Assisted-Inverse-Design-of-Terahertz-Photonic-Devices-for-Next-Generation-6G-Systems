"""Final COMSOL study with all COMSOL data (about 3,200 cells): the numbers for the report, website and app.

1. data summary per shape family
2. PWE vs COMSOL accuracy per family (the solver check)
3. multi-fidelity correction (COMSOL = PWE + GP correction): 10-fold cross-validation per family, leave-one-family-out
4. "do we still need PWE?" on the 1,007 freeform COMSOL cells (5-fold CV x 3 seeds): PWE alone, network on PWE,
   COMSOL-only GP / network, PWE + correction; plus a learning curve over the number of freeform COMSOL cells
5. accuracy statistics (mean, 95th percentile, share within 0.5 / 1 GHz) for the forward network (PWE test set),
   for the COMSOL-level predictors, for inverse design (PWE-verified) and for the COMSOL-verified designs

Gaussian processes: ARD kernel hyperparameters are fitted once per task on a training subset (seed 0, fold 0,
at most 700 cells) and then kept fixed for all folds, so thousands of cells stay fast. Networks: ensembles of 3.

Run from the Project folder:  python src/shapes/final_comsol_study.py      (about 1-1.5 h)
Writes results/optionb/final_comsol.json and results/optionb/comsol_vs_pwe.json (same format as before, for figures)
"""
import glob
import json
import os
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from dataset import holes, load_comsol, load_pwe  # noqa: E402
from gen_dataset import eps_from_validation  # noqa: E402
from geometry import FEATURE_NAMES  # noqa: E402
from models import Ensemble  # noqa: E402
from pwe import Solver  # noqa: E402

OUT = "results/optionb/final_comsol.json"
SEEDS, FOLDS = (0, 1, 2), 5
FAMILY = {"comsol_fourier": "freeform", "comsol_verify": "freeform (AI designs)", "comsol_circle": "circle",
          "comsol_circle_lhs": "circle", "comsol_rounded": "rounded triangle", "comsol_rounded_lhs": "rounded triangle",
          "comsol_rounded_lhs2": "rounded triangle", "comsol_sharp": "sharp triangle", "comsol_sharp_lhs": "sharp triangle"}


def stats(err):
    """err (n, 2) absolute errors in GHz -> dict of accuracy numbers per edge + worst edge."""
    err = np.abs(np.asarray(err, float))
    worst = err.max(1)
    return {"mae": err.mean(0).round(3).tolist(), "median": np.median(err, 0).round(3).tolist(),
            "p95": np.percentile(err, 95, axis=0).round(3).tolist(), "max": err.max(0).round(3).tolist(),
            "within_0.5": (err < 0.5).mean(0).round(4).tolist(), "within_1": (err < 1.0).mean(0).round(4).tolist(),
            "both_within_1": float((worst < 1.0).mean().round(4)), "worst_edge_mean": float(worst.mean().round(3)),
            "n": int(len(err))}


def fit_kernels(X, Y, n_sub=700, seed=0):
    idx = np.random.default_rng(seed).choice(len(X), min(n_sub, len(X)), replace=False)
    xm, xs = X.mean(0), X.std(0) + 1e-3
    out = []
    for j in range(Y.shape[1]):
        k = ConstantKernel(1.0) * RBF(np.ones(X.shape[1]), (1e-2, 1e3)) + WhiteKernel(1e-3, (1e-8, 1e1))
        out.append(GaussianProcessRegressor(k, normalize_y=True, random_state=0).fit((X[idx] - xm) / xs, Y[idx, j]).kernel_)
    return out


def gp_fixed(Xtr, Ytr, Xte, kernels):
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-3
    return np.stack([GaussianProcessRegressor(k, optimizer=None, normalize_y=True).fit((Xtr - xm) / xs, Ytr[:, j])
                     .predict((Xte - xm) / xs) for j, k in enumerate(kernels)], 1)


def nn(Xtr, Ytr, Xte, seed, target_steps=5000):
    spe = int(np.ceil(len(Xtr) / 128))
    ep = int(np.clip(target_steps / spe, 150, 1500))
    return Ensemble(n_members=3, epochs=ep).fit(Xtr, Ytr, seed=seed).predict(Xte)[0]


def pwe_for(com):
    """PWE band edges for every COMSOL cell: from the imported tables, the rest computed now (cached)."""
    known = {}
    for f in glob.glob("data/comsol/extra_*.csv"):
        t = pd.read_csv(f)
        if "f_lower_pwe" in t:
            known.update({i: (a, b) for i, a, b in zip(t.id, t.f_lower_pwe, t.f_upper_pwe)})
    cache = "results/optionb/comsol_pwe_cache.csv"
    if os.path.exists(cache):
        t = pd.read_csv(cache)
        known.update({i: (a, b) for i, a, b in zip(t.id, t.f_lower_pwe, t.f_upper_pwe)})
    miss = [i for i in com.id if i not in known]
    if miss:
        s = Solver(eps_si=eps_from_validation())
        rows = com.set_index("id").loc[miss]
        new = {i: s.band_edges(*holes(r)) for i, r in rows.iterrows()}
        known.update(new)
        old = pd.read_csv(cache) if os.path.exists(cache) else pd.DataFrame(columns=["id", "f_lower_pwe", "f_upper_pwe"])
        pd.concat([old, pd.DataFrame([(i, a, b) for i, (a, b) in new.items()], columns=old.columns)]).to_csv(cache, index=False)
        print(f"  computed PWE for {len(miss)} cells", flush=True)
    return np.array([known[i] for i in com.id], float)


def main():
    t0 = time.time()
    com = load_comsol().reset_index(drop=True)
    com["family"] = com.dataset.map(FAMILY).fillna(com.dataset)
    X = com[FEATURE_NAMES].to_numpy(float)
    Y = com[["f_lower", "f_upper"]].to_numpy(float)
    P = pwe_for(com)
    gapped = (com.gap > 0.1).to_numpy()
    res = {"n_cells": int(len(com)), "n_gapped": int(gapped.sum())}

    # 1 + 2: data summary and PWE accuracy per family (gapped cells)
    fam_rows = []
    for fam, g in com.groupby("family"):
        m = (com.family == fam).to_numpy() & gapped
        fam_rows.append({"family": fam, "cells": int(len(g)), "gapped": int(m.sum()),
                         "pwe_vs_comsol": stats(P[m] - Y[m]) if m.any() else None})
    allm = gapped & (com.family != "freeform (AI designs)").to_numpy()
    res["families"] = fam_rows
    res["pwe_vs_comsol_all"] = stats(P[allm] - Y[allm])
    print(f"PWE vs COMSOL on {allm.sum()} gapped cells: {res['pwe_vs_comsol_all']['mae']} GHz", flush=True)

    # 3: multi-fidelity correction, 10-fold CV on all gapped cells, per family; leave-one-family-out
    Xg, Yg, Pg, fg = X[gapped], Y[gapped], P[gapped], com.family.to_numpy()[gapped]
    kr = fit_kernels(Xg, Yg - Pg)
    corr = np.zeros_like(Yg)
    folds = np.random.default_rng(0).permutation(len(Xg)) % 10
    for f in range(10):
        tr, te = folds != f, folds == f
        corr[te] = Pg[te] + gp_fixed(Xg[tr], (Yg - Pg)[tr], Xg[te], kr)
    res["mf_cv10"] = {fam: {"pwe": stats(Pg[fg == fam] - Yg[fg == fam]), "pwe_plus_corr": stats(corr[fg == fam] - Yg[fg == fam])}
                      for fam in np.unique(fg)}
    res["mf_cv10"]["all"] = {"pwe": stats(Pg - Yg), "pwe_plus_corr": stats(corr - Yg)}
    lofo = {}
    for fam in ("circle", "rounded triangle", "sharp triangle", "freeform"):
        te = fg == fam
        if te.sum() and (~te).sum():
            pred = Pg[te] + gp_fixed(Xg[~te], (Yg - Pg)[~te], Xg[te], kr)
            lofo[fam] = {"pwe": stats(Pg[te] - Yg[te]), "pwe_plus_corr": stats(pred - Yg[te])}
    res["mf_leave_family_out"] = lofo
    print(f"multi-fidelity done ({(time.time() - t0) / 60:.1f} min): all {res['mf_cv10']['all']['pwe_plus_corr']['mae']}", flush=True)

    # 4: freeform comparison
    free = (com.dataset == "comsol_fourier").to_numpy()
    Xf, Yf, Pf = X[free], Y[free], P[free]
    Xo, Yo, Po = X[~free], Y[~free], P[~free]
    pwe = load_pwe()
    Xp, Yp = pwe[FEATURE_NAMES].to_numpy(float), pwe[["f_lower", "f_upper"]].to_numpy(float)
    methods = ["pwe_solver", "nn_pwe", "gp_comsol_free", "gp_comsol_all", "nn_comsol_free", "nn_comsol_all",
               "nn_pwe+corr", "solver+corr"]
    preds = {m: np.zeros((len(SEEDS), len(Xf), 2)) for m in methods}
    fr0 = np.random.default_rng(0).permutation(len(Xf)) % FOLDS != 0
    K = {"free": fit_kernels(Xf[fr0], Yf[fr0]),
         "all": fit_kernels(np.r_[Xo, Xf[fr0]], np.r_[Yo, Yf[fr0]]),
         "res_solver": fit_kernels(np.r_[Xo, Xf[fr0]], np.r_[Yo - Po, Yf[fr0] - Pf[fr0]])}
    fracs = (0.0, 0.1, 0.25, 0.5, 1.0)
    curve = {m: {f: [] for f in fracs} for m in ("gp_comsol_all", "nn_pwe+corr", "solver+corr")}
    for si, seed in enumerate(SEEDS):
        nnp = Ensemble(n_members=3, epochs=400).fit(Xp, Yp, seed=seed)
        pn_f, pn_o = nnp.predict(Xf)[0], nnp.predict(Xo)[0]
        if si == 0:
            K["res_nn"] = fit_kernels(np.r_[Xo, Xf[fr0]], np.r_[Yo - pn_o, Yf[fr0] - pn_f[fr0]])
        folds = np.random.default_rng(seed).permutation(len(Xf)) % FOLDS
        for f in range(FOLDS):
            tr, te = folds != f, folds == f
            Pm = preds
            Pm["pwe_solver"][si, te] = Pf[te]
            Pm["nn_pwe"][si, te] = pn_f[te]
            Pm["gp_comsol_free"][si, te] = gp_fixed(Xf[tr], Yf[tr], Xf[te], K["free"])
            Xa, Ya = np.r_[Xo, Xf[tr]], np.r_[Yo, Yf[tr]]
            Pm["gp_comsol_all"][si, te] = gp_fixed(Xa, Ya, Xf[te], K["all"])
            Pm["nn_comsol_free"][si, te] = nn(Xf[tr], Yf[tr], Xf[te], seed)
            Pm["nn_comsol_all"][si, te] = nn(Xa, Ya, Xf[te], seed)
            Pm["nn_pwe+corr"][si, te] = pn_f[te] + gp_fixed(Xa, Ya - np.r_[pn_o, pn_f[tr]], Xf[te], K["res_nn"])
            Pm["solver+corr"][si, te] = Pf[te] + gp_fixed(Xa, Ya - np.r_[Po, Pf[tr]], Xf[te], K["res_solver"])
            idx = np.where(tr)[0]
            for frac in fracs:
                k = idx[np.random.default_rng(seed + 100 * f).permutation(len(idx))[:int(round(frac * len(idx)))]]
                Xa2, Ya2 = np.r_[Xo, Xf[k]], np.r_[Yo, Yf[k]]
                curve["gp_comsol_all"][frac].append(np.abs(gp_fixed(Xa2, Ya2, Xf[te], K["all"]) - Yf[te]).mean(0))
                curve["nn_pwe+corr"][frac].append(np.abs(pn_f[te] + gp_fixed(Xa2, Ya2 - np.r_[pn_o, pn_f[k]], Xf[te], K["res_nn"]) - Yf[te]).mean(0))
                curve["solver+corr"][frac].append(np.abs(Pf[te] + gp_fixed(Xa2, Ya2 - np.r_[Po, Pf[k]], Xf[te], K["res_solver"]) - Yf[te]).mean(0))
            print(f"  seed {seed} fold {f} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
        print("seed", seed, {m: np.abs(preds[m][si] - Yf).mean(0).round(3).tolist() for m in methods}, flush=True)

    def per_seed(m):
        e = [np.abs(preds[m][si] - Yf).mean(0) for si in range(len(SEEDS))]
        g = [np.abs((preds[m][si][:, 1] - preds[m][si][:, 0]) - (Yf[:, 1] - Yf[:, 0])).mean() for si in range(len(SEEDS))]
        return e, g

    comp = {}
    for m in methods:
        e, g = per_seed(m)
        comp[m] = {"mae_f_lower": float(np.mean([x[0] for x in e])), "mae_f_upper": float(np.mean([x[1] for x in e])),
                   "std_f_lower": float(np.std([x[0] for x in e])), "std_f_upper": float(np.std([x[1] for x in e])),
                   "mae_gap": float(np.mean(g)), "accuracy": stats(preds[m].mean(0) - Yf)}
    n_tr = int(round(len(Xf) * (FOLDS - 1) / FOLDS))
    lc = {m: [{"n_free_train": int(round(f * n_tr)), "mae": np.mean(v, 0).round(3).tolist()} for f, v in c.items()]
          for m, c in curve.items()}
    res["freeform_comparison"] = {"n_free": int(len(Xf)), "n_other": int(len(Xo)), "methods": comp, "learning_curve": lc}

    # 5: accuracy statistics from the saved main results
    acc = {}
    fp = np.load("results/optionb/forward_parity.npz", allow_pickle=True)
    acc["forward_pwe_test"] = stats(fp["mu"] - fp["y"])
    iv = np.load("results/optionb/inverse_errors.npz", allow_pickle=True)
    acc["inverse_pwe_verified"] = {k: stats(iv[k][~np.isnan(iv[k][:, 0])]) for k in ("tandem", "cvae", "multistart", "cvae_refine")}
    if os.path.exists("results/optionb/comsol_verify.csv"):
        cv = pd.read_csv("results/optionb/comsol_verify.csv")
        acc["comsol_verified_designs"] = stats(cv[["COMSOL - target lower", "COMSOL - target upper"]].to_numpy())
        new = cv[cv.kind == "new design"]
        acc["comsol_verified_new_designs"] = stats(new[["COMSOL - target lower", "COMSOL - target upper"]].to_numpy())
    res["accuracy"] = acc
    res["minutes"] = round((time.time() - t0) / 60, 1)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)

    # same format as compare_training.py, so the existing figures and pages pick it up
    old = "results/optionb/comsol_vs_pwe.json"
    if os.path.exists(old) and not os.path.exists("results/optionb/comsol_vs_pwe_97cells.json"):
        shutil.copy(old, "results/optionb/comsol_vs_pwe_97cells.json")
    with open(old, "w") as fh:
        json.dump({"n_free": int(len(Xf)), "n_other": int(len(Xo)), "seeds": list(SEEDS), "folds": FOLDS,
                   "methods": {m: {k: v for k, v in comp[m].items() if k != "accuracy"} for m in methods},
                   "learning_curve": lc, "minutes": res["minutes"], "source": "final_comsol_study.py"}, fh, indent=2)
    np.savez(old.replace(".json", ".npz"), y=Yf, **{m.replace("+", "_plus_"): preds[m].mean(0) for m in methods})
    print(json.dumps({m: [round(comp[m]["mae_f_lower"], 3), round(comp[m]["mae_f_upper"], 3)] for m in methods}, indent=1))
    print(f"done in {res['minutes']} min -> {OUT}")


if __name__ == "__main__":
    main()
