"""Benchmark of inverse-design methods on the COMSOL band-edge data (both corner types).

Run from the Project folder (after src/utils/process_bands.py):  python src/benchmark/run.py

Each seed splits every corner type into train/test (same fraction). Two training configs:
  rounded : trained on rounded-corner geometries only, inputs (L1, L2)
  both    : trained on rounded + sharp geometries, inputs (L1, L2, corner)
Both configs are tested on the same held-out rounded geometries, so comparing them shows
whether the sharp-corner data helps; "both" is also tested on held-out sharp geometries.

Scoring uses a GP oracle per corner type, fitted on all its points (leave-one-out error is
printed), as a stand-in for COMSOL; designs worth reporting must be re-simulated in COMSOL.
"""
import json
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, "src/benchmark")
from data import CORNER, gap_of, load_corner, random_split, with_corner
from methods import DirectInverse, GPGradient, NNForward, TandemInverse, make_gp

warnings.filterwarnings("ignore")  # sklearn GP hyper-parameter convergence chatter

N_SEEDS = 10
FRAC_TRAIN = 0.74  # 45 of 61 rounded, as before
METHODS = [DirectInverse, TandemInverse, GPGradient]


def loo_mae(x, y):
    errs = []
    for i in range(len(x)):
        m = np.arange(len(x)) != i
        errs.append(np.abs(make_gp(x[m], y[m])(x[i]) - y[i])[0])
    return np.mean(errs, 0)


def main():
    data = {c: load_corner(c) for c in CORNER}
    oracles = {c: make_gp(*data[c]) for c in CORNER}
    for c, (x, y) in data.items():
        print(f"{c}: {len(x)} geometries, oracle LOO MAE [f_lower, f_upper] GHz:",
              np.round(loo_mae(x, y), 4), flush=True)

    def score(sols, corner):
        """Oracle response of each candidate; geometry columns only."""
        return oracles[corner](sols[:, :2])

    res = {}
    for task in ("edges", "gap"):
        for seed in range(N_SEEDS):
            split = {c: random_split(len(data[c][0]), FRAC_TRAIN, seed) for c in CORNER}
            tr = {c: (data[c][0][split[c][0]], data[c][1][split[c][0]]) for c in CORNER}
            te = {c: (data[c][0][split[c][1]], data[c][1][split[c][1]]) for c in CORNER}
            configs = {
                "rounded": (tr["rounded"][0], tr["rounded"][1], ["rounded"], False),
                "both": (np.r_[with_corner(tr["rounded"][0], "rounded"), with_corner(tr["sharp"][0], "sharp")],
                         np.r_[tr["rounded"][1], tr["sharp"][1]], ["rounded", "sharp"], True),
            }
            for cfg, (x, y, test_corners, conditioned) in configs.items():
                fwd = NNForward(x, y, seed)
                models = [M(x, y, task, seed, fwd=fwd) for M in METHODS]
                for corner in test_corners:
                    xt, yt = te[corner]
                    tgt = yt if task == "edges" else gap_of(yt)
                    cond = np.full((len(xt), 1), CORNER[corner], float) if conditioned else np.zeros((len(xt), 0))
                    for M, m in zip(METHODS, models):
                        t0 = time.time()
                        sols = m.propose(tgt, cond)
                        dt = (time.time() - t0) / len(xt)
                        r = res.setdefault((task, cfg, corner, M.name), {"err": [], "geo": [], "span": [], "t": []})
                        for t, s, x_true in zip(tgt, sols, xt):
                            yp = score(s, corner)
                            err = np.abs((yp if task == "edges" else gap_of(yp)) - t).max(1)
                            r["err"].append(err.min())
                            r["geo"].append(np.abs(s[np.argmin(err), :2] - x_true).mean())
                            r["span"].append(np.ptp(s[:, :2], 0).max() if len(s) > 1 else 0.0)
                            r["t"].append(dt)
            print(f"task={task} seed={seed} done", flush=True)

    hdr = "%-6s %-8s %-8s %-28s %9s %9s %11s %9s"
    print("\n" + hdr % ("task", "train", "test", "method", "err GHz", "max GHz", "geo MAE um", "span um"))
    out = {}
    for (task, cfg, corner, name), r in sorted(res.items()):
        e, g, s = (np.array(r[k]) for k in ("err", "geo", "span"))
        print("%-6s %-8s %-8s %-28s %9.3f %9.2f %11.2f %9.2f" % (task, cfg, corner, name, e.mean(), e.max(),
                                                               g.mean(), s.mean()))
        out[f"{task}/train={cfg}/test={corner}/{name}"] = dict(
            err_mean=e.mean(), err_max=e.max(), geo_mae_um=g.mean(), span_um=s.mean(),
            ms_per_target=1e3 * np.mean(r["t"]), n=len(e))
    json.dump(out, open("results/benchmark.json", "w"), indent=2)


if __name__ == "__main__":
    main()
