"""Compare a COMSOL re-run at k = 1, 2 only with the old full-path sweep of the same cells.

The rounded-triangle cells were re-simulated on 28 Oct 2026 at k = 1 and 2 only, with a finer mesh
(about 5,500 mesh nodes per solution instead of about 820). Comparing the two tests at once
(a) that COMSOL's two-k-point route gives the same band edges as the 281-point path, and
(b) how converged the old mesh was.

Run from the Project folder:
  python src/shapes/check_comsol_rerun.py "data/comsol/<re-run export>.csv" --corner 1
Writes results/optionb/comsol_rerun_check.json and data/comsol/rerun_vs_old_<corner>.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--corner", type=int, default=1, help="1 rounded, 0 sharp (as in data/comsol/bands.csv)")
    args = ap.parse_args()

    raw = pd.read_csv(args.csv, comment="%", header=None, usecols=[0, 2, 3, 4, 6], names=["X", "L1", "L2", "k", "f"])
    d = raw[["L1", "L2", "k", "f"]].copy()
    d[["L1", "L2"]] = (d[["L1", "L2"]] * 1e6).round(2)        # metres -> um
    d = d.round({"f": 6}).drop_duplicates()
    rows_per_state = len(raw) / len(d)                        # one row per mesh node and eigen-solution
    d["band"] = d.groupby(["L1", "L2", "k"]).f.rank(method="first").astype(int)
    d = d[d.band <= 2]
    new = d.pivot_table(index=["L1", "L2"], columns=["k", "band"], values="f")

    old = pd.read_csv("data/comsol/bands.csv")
    old = old[old.corner == args.corner]
    out = []
    for (l1, l2), r in new.iterrows():
        o = old[np.isclose(old.L1_um, l1, atol=0.02) & np.isclose(old.L2_um, l2, atol=0.02)]
        if o.empty:
            continue
        lo_new, up_new = max(r[(1, 1)], r[(2, 1)]), min(r[(1, 2)], r[(2, 2)])
        lo_old, up_old = o.f1_GHz.max(), o.f2_GHz.min()
        out.append(dict(L1=l1, L2=l2, f_lower_new=lo_new, f_upper_new=up_new, f_lower_old=lo_old, f_upper_old=up_old,
                        d_lower=lo_new - lo_old, d_upper=up_new - up_old, d_gap=(up_new - lo_new) - (up_old - lo_old)))
    c = pd.DataFrame(out)
    c.to_csv(f"data/comsol/rerun_vs_old_{args.corner}.csv", index=False)
    summary = {
        "file": os.path.basename(args.csv), "corner": args.corner, "n_cells": int(len(c)),
        "n_k_new": int(new.columns.get_level_values(0).nunique()), "n_k_old": int(old.k.nunique()),
        "mesh_nodes_per_solution_new": int(round(rows_per_state)),
        "max_abs_diff_GHz": {q: float(c[f"d_{q}"].abs().max()) for q in ("lower", "upper", "gap")},
        "mean_diff_GHz": {q: float(c[f"d_{q}"].mean()) for q in ("lower", "upper", "gap")}}
    os.makedirs("results/optionb", exist_ok=True)
    with open("results/optionb/comsol_rerun_check.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
