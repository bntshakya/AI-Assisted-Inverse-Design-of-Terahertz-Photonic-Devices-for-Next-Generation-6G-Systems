"""Is the PWE solver converged? Re-solve COMSOL cells with coarser and finer settings.

For each setting (pixel grid, plane-wave cut-off) the band edges of all COMSOL triangle cells and all COMSOL
freeform cells are computed with the same permittivity (8.914). Reported per setting: number of plane waves,
time per cell, mean and max change of the band edges against the finest setting, and the error against COMSOL.

Run from the Project folder:  python src/shapes/check_pwe_convergence.py
Writes results/optionb/pwe_convergence.json
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dataset import holes, load_comsol  # noqa: E402
from gen_dataset import eps_from_validation  # noqa: E402
from pwe import Solver  # noqa: E402

OUT = "results/optionb/pwe_convergence.json"
SETTINGS = [(64, 7.0), (96, 9.0), (96, 11.0), (128, 14.0), (160, 17.0)]  # (pixels per side, cut-off); (96, 11) = used
USED = (96, 11.0)


def main():
    com = load_comsol()
    com = com[com.dataset.isin(["comsol_rounded", "comsol_sharp", "comsol_fourier"])].reset_index(drop=True)
    H = [holes(r) for _, r in com.iterrows()]
    Y = com[["f_lower", "f_upper"]].to_numpy(float)
    fam = np.where(com.dataset == "comsol_fourier", "freeform", "triangles")
    eps = eps_from_validation()
    print(f"{len(com)} COMSOL cells ({dict(zip(*np.unique(fam, return_counts=True)))}), eps = {eps}", flush=True)
    edges = {}
    rows = []
    for n_grid, g_cut in SETTINGS:
        s = Solver(n_grid=n_grid, g_cut=g_cut, eps_si=eps)
        t0 = time.time()
        E = np.array([s.band_edges(*h) for h in H])
        dt = (time.time() - t0) / len(H)
        edges[(n_grid, g_cut)] = E
        rows.append({"n_grid": n_grid, "g_cut": g_cut, "n_plane_waves": int(s.n_pw), "seconds_per_cell": round(dt, 3),
                     "used_in_project": (n_grid, g_cut) == USED})
        print(f"  {n_grid} px, cut {g_cut}: {s.n_pw} plane waves, {dt:.2f} s/cell", flush=True)
    ref = edges[SETTINGS[-1]]
    for row, key in zip(rows, SETTINGS):
        E = edges[key]
        d = np.abs(E - ref)
        row["change_vs_finest_mean_GHz"] = np.round(d.mean(0), 3).tolist()
        row["change_vs_finest_max_GHz"] = np.round(d.max(0), 3).tolist()
        for f in ("triangles", "freeform"):
            m = fam == f
            row[f"mae_vs_comsol_{f}_GHz"] = np.round(np.abs(E[m] - Y[m]).mean(0), 3).tolist()
    res = {"n_cells": len(com), "eps": eps, "reference": "finest setting", "settings": rows}
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(rows, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
