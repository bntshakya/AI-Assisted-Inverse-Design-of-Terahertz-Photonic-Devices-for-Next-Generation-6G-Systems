"""Random (Latin-hypercube) geometries for a new, lightweight COMSOL sweep.

Run from the Project folder:
  python src/comsol/make_samples.py --n 500

Writes to data/comsol/new/:
  samples.csv          id, L1_um, L2_um, is_check       master list for run_mph.py / parse_table.py
  sweep_load_file.txt  for Study > Parametric Sweep > Load from File (Sweep type: Specified combinations).
                       Every geometry appears once per k in --k, so all three lists have equal length.
                       Values are in METRES (the unit column is ignored when COMSOL loads .txt files).

The first --n-check rows are geometries from the existing 8x8 grid, so the new runs can be
compared with data/comsol/bands.csv before trusting the rest.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import qmc

OUT_DIR = "data/comsol/new"
# k = 1 is M and k = 2 is K on the Gamma-M-K-Gamma path of the existing model: in all 64
# rounded-corner geometries band 1 peaks at k = 2 and band 2 bottoms out at k = 1 (63 of 64).
DEFAULT_K = (1.0, 2.0)
CHECK_GEOMETRIES = [(168.01, 119.01), (147.01, 105.01), (189.01, 91.0)]  # on the old grid, all gapped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500, help="number of random geometries")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--l1", type=float, nargs=2, default=(140.0, 189.0), metavar=("MIN", "MAX"), help="um")
    p.add_argument("--l2", type=float, nargs=2, default=(91.0, 140.0), metavar=("MIN", "MAX"), help="um")
    p.add_argument("--k", type=float, nargs="+", default=DEFAULT_K, help="k values per geometry")
    p.add_argument("--n-check", type=int, default=len(CHECK_GEOMETRIES))
    p.add_argument("--names", nargs=3, default=("L1", "L2", "k"), help="COMSOL parameter names")
    args = p.parse_args()

    lo, hi = np.array([args.l1[0], args.l2[0]]), np.array([args.l1[1], args.l2[1]])
    lhs = (qmc.scale(qmc.LatinHypercube(d=2, seed=args.seed).random(args.n), lo, hi).round(3)
           if args.n else np.empty((0, 2)))  # --n 0: check geometries only
    check = np.array(CHECK_GEOMETRIES[:args.n_check]).reshape(-1, 2)
    geo = np.r_[check, lhs]
    samples = pd.DataFrame({
        "id": [f"chk{i}" for i in range(len(check))] + [f"s{i:05d}" for i in range(len(lhs))],
        "L1_um": geo[:, 0], "L2_um": geo[:, 1], "is_check": [1] * len(check) + [0] * len(lhs),
    })

    os.makedirs(OUT_DIR, exist_ok=True)
    samples.to_csv(os.path.join(OUT_DIR, "samples.csv"), index=False)

    nk = len(args.k)
    l1 = np.repeat(samples.L1_um.to_numpy() * 1e-6, nk)
    l2 = np.repeat(samples.L2_um.to_numpy() * 1e-6, nk)
    kk = np.tile(args.k, len(samples))
    fmt = lambda a: " ".join(f"{v:.9g}" for v in a)
    with open(os.path.join(OUT_DIR, "sweep_load_file.txt"), "w") as f:
        for name, vals in zip(args.names, (l1, l2, kk)):
            f.write(f'{name} "{fmt(vals)}"\n')

    print(f"{len(samples)} geometries ({len(check)} check + {len(lhs)} random) x {nk} k-points "
          f"= {len(samples) * nk} eigen-solves")
    print(f"wrote {OUT_DIR}/samples.csv and {OUT_DIR}/sweep_load_file.txt")


if __name__ == "__main__":
    main()
