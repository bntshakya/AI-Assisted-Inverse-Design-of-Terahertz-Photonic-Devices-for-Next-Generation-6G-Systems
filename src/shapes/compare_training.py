"""Do we still need PWE? Compare training on PWE data, on COMSOL data, and on both - scored on COMSOL freeform cells.

Test cells: the COMSOL freeform (Fourier-hole) cells (data/comsol/extra_fourier*.csv). They are split into 5 folds;
each fold is predicted by models that never saw it (repeated with several seeds). Predictors:
  pwe_solver        the PWE solver itself (no machine learning)               - needs no COMSOL data
  nn_pwe            NN ensemble trained on the 6,000 PWE cells                   - needs no COMSOL data
  gp_comsol_free    Gaussian process trained on COMSOL freeform cells only
  gp_comsol_all     Gaussian process trained on all COMSOL cells (triangles, circles, freeform)
  nn_comsol_all     NN ensemble trained on all COMSOL cells
  nn_pwe+corr       nn_pwe + GP correction learned from all COMSOL cells (multi-fidelity)
  solver+corr       pwe_solver + GP correction learned from all COMSOL cells (multi-fidelity)
A learning curve repeats the main ones with only part of the COMSOL freeform training cells, to show how many
COMSOL runs each approach needs.

Run from the Project folder:  python src/shapes/compare_training.py   (about 10 minutes on the laptop GPU)
Writes results/optionb/comsol_vs_pwe.json
"""
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_comsol, load_pwe  # noqa: E402
from geometry import FEATURE_NAMES  # noqa: E402
from models import Ensemble  # noqa: E402
from run_optionb import gp_fit_predict  # noqa: E402

OUT = "results/optionb/comsol_vs_pwe.json"
SEEDS = (0, 1, 2)
FOLDS = 5


def mae(a, b):
    return np.abs(np.asarray(a) - np.asarray(b)).mean(0)


def main():
    t0 = time.time()
    com = load_comsol()
    free = com[com.dataset.str.startswith("comsol_fourier")].reset_index(drop=True)
    other = com[~com.dataset.str.startswith("comsol_fourier")].reset_index(drop=True)
    if len(free) < 10:
        sys.exit("fewer than 10 COMSOL freeform cells: import a Fourier batch first")
    import pandas as pd
    extra = pd.concat([pd.read_csv(p) for p in sorted(__import__("glob").glob("data/comsol/extra_fourier*.csv"))])
    pwe_solver = extra.set_index("id").loc[free.id, ["f_lower_pwe", "f_upper_pwe"]].to_numpy(float)
    Xf, Yf = free[FEATURE_NAMES].to_numpy(float), free[["f_lower", "f_upper"]].to_numpy(float)
    Xo, Yo = other[FEATURE_NAMES].to_numpy(float), other[["f_lower", "f_upper"]].to_numpy(float)
    # PWE solver values for the other COMSOL cells (for the solver+corr residuals)
    from pwe import Solver
    from gen_dataset import eps_from_validation
    from dataset import holes
    s = Solver(eps_si=eps_from_validation())
    pwe_other = np.array([s.band_edges(*holes(r)) for _, r in other.iterrows()])
    print(f"COMSOL freeform cells: {len(free)}; other COMSOL cells: {len(other)} ({other.dataset.value_counts().to_dict()})",
          flush=True)

    pwe = load_pwe()
    Xp, Yp = pwe[FEATURE_NAMES].to_numpy(float), pwe[["f_lower", "f_upper"]].to_numpy(float)
    res = {"n_free": len(free), "n_other": len(other), "n_pwe": len(pwe), "seeds": list(SEEDS), "folds": FOLDS}
    methods = ["pwe_solver", "nn_pwe", "gp_comsol_free", "gp_comsol_all", "nn_comsol_all", "nn_pwe+corr", "solver+corr"]
    scores = {m: [] for m in methods}
    curve_fracs = (0.0, 0.25, 0.5, 1.0)
    curve = {m: {f: [] for f in curve_fracs} for m in ("gp_comsol_all", "nn_pwe+corr", "solver+corr")}
    preds_all = {m: np.zeros((len(SEEDS), len(free), 2)) for m in methods}

    for si, seed in enumerate(SEEDS):
        nn_pwe = Ensemble(n_members=5, epochs=400).fit(Xp, Yp, seed=seed)
        p_nn_free = nn_pwe.predict(Xf)[0]
        p_nn_other = nn_pwe.predict(Xo)[0]
        folds = np.random.default_rng(seed).permutation(len(free)) % FOLDS
        P = {m: np.zeros_like(Yf) for m in methods}
        for f in range(FOLDS):
            tr, te = folds != f, folds == f
            P["pwe_solver"][te] = pwe_solver[te]
            P["nn_pwe"][te] = p_nn_free[te]
            P["gp_comsol_free"][te] = gp_fit_predict(Xf[tr], Yf[tr], Xf[te])
            Xa, Ya = np.r_[Xo, Xf[tr]], np.r_[Yo, Yf[tr]]
            P["gp_comsol_all"][te] = gp_fit_predict(Xa, Ya, Xf[te])
            P["nn_comsol_all"][te] = Ensemble(n_members=5, epochs=1500).fit(Xa, Ya, seed=seed).predict(Xf[te])[0]
            P["nn_pwe+corr"][te] = p_nn_free[te] + gp_fit_predict(Xa, Ya - np.r_[p_nn_other, p_nn_free[tr]], Xf[te])
            P["solver+corr"][te] = pwe_solver[te] + gp_fit_predict(Xa, Ya - np.r_[pwe_other, pwe_solver[tr]], Xf[te])
            # learning curve: only part of the freeform training cells (the other COMSOL families always included)
            idx_tr = np.where(tr)[0]
            for frac in curve_fracs:
                k = idx_tr[np.random.default_rng(seed + 100 * f).permutation(len(idx_tr))[:int(round(frac * len(idx_tr)))]]
                Xa2, Ya2 = np.r_[Xo, Xf[k]], np.r_[Yo, Yf[k]]
                curve["gp_comsol_all"][frac].append(mae(gp_fit_predict(Xa2, Ya2, Xf[te]), Yf[te]))
                curve["nn_pwe+corr"][frac].append(mae(p_nn_free[te] + gp_fit_predict(
                    Xa2, Ya2 - np.r_[p_nn_other, p_nn_free[k]], Xf[te]), Yf[te]))
                curve["solver+corr"][frac].append(mae(pwe_solver[te] + gp_fit_predict(
                    Xa2, Ya2 - np.r_[pwe_other, pwe_solver[k]], Xf[te]), Yf[te]))
        for m in methods:
            scores[m].append(mae(P[m], Yf).tolist() + [float(np.abs((P[m][:, 1] - P[m][:, 0]) - (Yf[:, 1] - Yf[:, 0])).mean())])
            preds_all[m][si] = P[m]
        print(f"seed {seed}: " + "; ".join(f"{m} {np.round(scores[m][-1], 2).tolist()}" for m in methods), flush=True)

    res["methods"] = {m: {"mae_f_lower": float(np.mean([x[0] for x in scores[m]])),
                          "mae_f_upper": float(np.mean([x[1] for x in scores[m]])),
                          "mae_gap": float(np.mean([x[2] for x in scores[m]])),
                          "std_f_lower": float(np.std([x[0] for x in scores[m]])),
                          "std_f_upper": float(np.std([x[1] for x in scores[m]]))} for m in methods}
    n_tr_free = int(round(len(free) * (FOLDS - 1) / FOLDS))
    res["learning_curve"] = {m: [{"n_free_train": int(round(f * n_tr_free)),
                                  "mae": np.mean(v, 0).round(3).tolist()} for f, v in c.items()] for m, c in curve.items()}
    res["minutes"] = round((time.time() - t0) / 60, 1)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    np.savez(OUT.replace(".json", ".npz"), y=Yf, **{m.replace("+", "_plus_"): preds_all[m].mean(0) for m in methods})
    print(json.dumps(res["methods"], indent=1))
    print(json.dumps(res["learning_curve"], indent=1))
    print(f"done in {res['minutes']} min -> {OUT}")


if __name__ == "__main__":
    main()
