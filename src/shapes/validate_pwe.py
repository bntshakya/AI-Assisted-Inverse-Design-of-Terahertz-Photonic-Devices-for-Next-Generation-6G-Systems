"""Check the PWE solver against the COMSOL band edges (both datasets) and calibrate eps.

Run from the Project folder:  python src/shapes/validate_pwe.py

1. Fits ONE number, the silicon permittivity eps used by the COMSOL 2D model, on every 4th
   geometry (32 of 128). The slides say 11.7 (bulk Si); the COMSOL frequencies are reproduced only
   with eps ~ 8.9 (n ~ 2.98), i.e. the 2D model most likely uses an effective index for the
   200 um slab. Confirm this in COMSOL: Materials > Relative permittivity / refractive index.
2. Compares PWE and COMSOL band edges for all 128 geometries (f_lower, f_upper, gap in GHz).

Writes data/pwe/validation.csv and results/pwe_validation.json (eps is read by the other scripts).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(__file__))
from pwe import Solver

EDGES = "data/comsol/band_edges.csv"
OUT_CSV = "data/pwe/validation.csv"
OUT_JSON = "results/pwe_validation.json"
FAMILY = {1: "rounded", 0: "sharp"}


def pwe_edges(df, eps):
    s = Solver(eps_si=eps)
    out = []
    for r in df.itertuples():
        fam = FAMILY[r.corner]
        out.append(s.band_edges((fam, {"L": r.L1_um}), (fam, {"L": r.L2_um})))
    return np.array(out)


def main():
    df = pd.read_csv(EDGES)
    calib = df.iloc[::4]

    def loss(eps):
        p = pwe_edges(calib, eps)
        return np.mean((p / calib[["f_lower", "f_upper"]].values - 1) ** 2)

    eps = minimize_scalar(loss, bounds=(8.0, 10.0), method="bounded", options={"xatol": 1e-3}).x
    print(f"fitted eps = {eps:.3f} (n = {np.sqrt(eps):.3f}) on {len(calib)} geometries", flush=True)

    p = pwe_edges(df, eps)
    df["f_lower_pwe"], df["f_upper_pwe"] = p[:, 0], p[:, 1]
    df["gap_pwe"] = df.f_upper_pwe - df.f_lower_pwe
    df["calib"] = df.index.isin(calib.index).astype(int)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    summary = {"eps_si": eps, "n_si": float(np.sqrt(eps)), "n_geometries": len(df), "per_family": {}}
    for corner, g in df.groupby("corner"):
        g = g[g.gap > 0.1]
        err = {q: (g[f"{q}_pwe"] - g[q]) for q in ("f_lower", "f_upper", "gap")}
        summary["per_family"][FAMILY[corner]] = {
            "n_gapped": len(g),
            **{f"MAE_{q}_GHz": float(e.abs().mean()) for q, e in err.items()},
            **{f"bias_{q}_GHz": float(e.mean()) for q, e in err.items()},
            "corr_gap": float(np.corrcoef(g.gap, g.gap_pwe)[0, 1]),
        }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
