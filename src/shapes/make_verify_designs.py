"""AI designs for several targets, PWE-verified and inside the COMSOL cell, as one COMSOL sweep file.

Trains the forward ensemble and CVAE once (same settings and seed as run_optionb.py), then for each target:
CVAE samples + gradient refinement -> rank by the surrogate -> exact fabrication check -> hole reach <= 68 um
(so the hole stays inside the COMSOL cell) -> distinct -> PWE verification. Keeps up to 3 designs per target
whose PWE band edges are within 0.5 GHz of the target. The two gallery designs already simulated in COMSOL are
put first as check cells.

Run from the Project folder:  python src/shapes/make_verify_designs.py
Writes data/comsol/new/sweep_verify_designs.txt (Load from File; rA0..s6B in metres, k = 1, 2)
   and data/comsol/new/verify_designs.csv (target, design numbers in um, PWE band edges, reach, wall)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_pwe  # noqa: E402
from gen_dataset import eps_from_validation  # noqa: E402
from geometry import FEATURE_NAMES, check_cell  # noqa: E402
from models import CVAE, Ensemble, multistart, p_to_feat, p_to_holes, p_to_u  # noqa: E402
from pwe import Solver  # noqa: E402
from run_optionb import P_IDX, _distinct, split  # noqa: E402

TARGETS = [(300.0, 330.0), (305.0, 325.0), (310.0, 340.0), (320.0, 335.0)]
PER_TARGET, TOL, MAX_REACH = 3, 0.5, 68.0
NAMES = ["rA0", "c3A", "s3A", "c6A", "s6A", "rB0", "c3B", "s3B", "c6B", "s6B"]
OUT = "data/comsol/new"


def main():
    t0 = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    pwe = load_pwe()
    X, Y = pwe[FEATURE_NAMES].to_numpy(float), pwe[["f_lower", "f_upper"]].to_numpy(float)
    tr, va, te = split(len(pwe), seed=0)
    trv = np.r_[tr, va]
    fwd = Ensemble(n_members=5, epochs=400).fit(X[trv], Y[trv], seed=0)
    is4 = (pwe.dataset == "pwe_fourier").to_numpy()
    cv = CVAE().fit(p_to_u(X[trv[is4[trv]]][:, P_IDX]), Y[trv[is4[trv]]], seed=0)
    print(f"models trained ({time.time() - t0:.0f} s)", flush=True)
    solver = Solver(eps_si=eps_from_validation())

    rows = []
    sc = pd.read_csv("data/pwe/showcase_designs.csv")
    comsol_known = [(299.10, 330.25), (299.11, 330.06)]
    for i in range(2):  # check cells: already simulated in COMSOL
        p = sc.loc[i, [n + "_um" for n in NAMES]].to_numpy(float)
        rows.append(dict(kind="check (already in COMSOL)", target_lower=300.0, target_upper=330.0,
                         **dict(zip(NAMES, np.round(p, 3))), f_lower_pwe=sc.f_lower_pwe[i], f_upper_pwe=sc.f_upper_pwe[i],
                         f_lower_comsol_earlier=comsol_known[i][0], f_upper_comsol_earlier=comsol_known[i][1]))
    for tgt in TARGETS:
        T = np.array([tgt])
        c = cv.propose(T, 512, seed=11)
        ref = multistart(fwd, T, 256, steps=150, U0=p_to_u(c[:, :256].reshape(-1, 10)).reshape(1, 256, 10), seed=11)
        cands = np.concatenate([c, ref], 1)[0]
        pred = fwd.predict(p_to_feat(cands))[0]
        order = np.argsort(np.abs(pred - T[0]).max(1))
        kept, us, tried = [], [], 0
        for j in order:
            if len(kept) == PER_TARGET or tried >= 80:
                break
            p = cands[j]
            ok, info = check_cell(*p_to_holes(p))
            if not ok or max(info["rmax_A"], info["rmax_B"]) > MAX_REACH:
                continue
            u = p_to_u(p[None])[0]
            if not _distinct(u, us):
                continue
            tried += 1
            v = np.array(solver.band_edges(*p_to_holes(p)))
            if np.abs(v - T[0]).max() <= TOL:
                kept.append(p)
                us.append(u)
                rows.append(dict(kind="new design", target_lower=tgt[0], target_upper=tgt[1],
                                 **dict(zip(NAMES, np.round(p, 3))), f_lower_pwe=round(v[0], 3), f_upper_pwe=round(v[1], 3),
                                 reach_um=round(max(info["rmax_A"], info["rmax_B"]), 1), wall_um=round(info["min_wall"], 1)))
        print(f"  target {tgt}: {len(kept)} designs kept after {tried} PWE checks", flush=True)

    df = pd.DataFrame(rows)
    df.insert(0, "id", [f"v{i:02d}" for i in range(len(df))])
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "verify_designs.csv"), index=False)
    with open(os.path.join(OUT, "sweep_verify_designs.txt"), "w") as f:
        for nm in NAMES:
            f.write(f'{nm} "' + " ".join(f"{v:.6g}" for v in np.repeat(df[nm].to_numpy(float) * 1e-6, 2)) + '"\n')
        f.write('k "' + " ".join(["1 2"] * len(df)) + '"\n')
    print(df[["id", "kind", "target_lower", "target_upper", "f_lower_pwe", "f_upper_pwe"]].to_string(index=False))
    print(f"wrote {OUT}/sweep_verify_designs.txt ({len(df)} cells) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
