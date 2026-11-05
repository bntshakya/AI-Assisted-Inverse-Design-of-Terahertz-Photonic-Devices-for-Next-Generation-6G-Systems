"""Why COMSOL needs only k = 1 (M) and k = 2 (K): three independent checks.

1. COMSOL: for all 128 old cells, the band edges from k = 1 and 2 are compared with the full 281-point
   path (data/comsol/bands.csv).
2. PWE, mixed shapes: cells in data/pwe/*.csv that gen_dataset.py --check also solved on a 29-point path
   and on a grid over the irreducible wedge (columns dev_*).
3. PWE, circle pairs (for the round-cell COMSOL data): a grid of radii, each solved on a fine path
   (step 0.02) and a grid over the wedge. Shalaev et al. (Nat. Nanotech. 2019) report that circular holes
   can push the band-2 minimum between K and M in a silicon slab, so circles are tested separately.

It also stores band diagrams of three example cells and a band map over the whole wedge for the figure.

Outputs: results/optionb/mk_check.json, mk_circle_check.csv, mk_examples.npz
Run from the Project folder:  python src/shapes/check_mk_shortcut.py      (about 4 minutes)
"""
import json
import os
import sys
import time

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, os.path.dirname(__file__))
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dataset import holes  # noqa: E402
from pwe import K_PT, M_PT, Solver, kpath, wedge_grid  # noqa: E402

OUT = "results/optionb"
PATH_FINE = np.round(np.arange(0.02, 2.99, 0.02), 4)   # circle check (Gamma itself has a zero mode)
PATH_FIG = np.round(np.arange(0.01, 3.0, 0.01), 4)     # example band diagrams
_S = None


def eps_fitted():
    with open("results/pwe_validation.json") as f:
        return json.load(f)["eps_si"]


def _init(eps):
    global _S
    _S = Solver(eps_si=eps)


def _bands(task):
    """task = (holeA, holeB, list of wavevectors) -> bands (n_k, 2)."""
    hA, hB, kvecs = task
    return _S.bands_kvec(hA, hB, kvecs)


def _circle(rr):
    rA, rB = rr
    hA, hB = ("circle", {"R": rA}), ("circle", {"R": rB})
    inv = _S.inv_eps_matrix(hA, hB)
    mk = _S.bands_kvec(hA, hB, [M_PT, K_PT], inv=inv)
    path = _S.bands_kvec(hA, hB, [kpath(k) for k in PATH_FINE], inv=inv)
    zone = _S.bands_kvec(hA, hB, wedge_grid(10), inv=inv)
    lo_mk, up_mk = mk[:, 0].max(), mk[:, 1].min()
    lo, up = max(path[:, 0].max(), zone[:, 0].max()), min(path[:, 1].min(), zone[:, 1].min())
    return dict(rA=rA, rB=rB, f_lower_MK=lo_mk, f_upper_MK=up_mk, f_lower_full=lo, f_upper_full=up,
                dev_lower=lo - lo_mk, dev_upper=up_mk - up,
                k_lower=float(PATH_FINE[path[:, 0].argmax()]), k_upper=float(PATH_FINE[path[:, 1].argmin()]),
                gap=up_mk - lo_mk)


def check_comsol():
    b = pd.read_csv("data/comsol/bands.csv")
    rows = []
    for (corner, l1, l2), g in b.groupby(["corner", "L1_um", "L2_um"]):
        mk = g[np.isclose(g.k, 1.0) | np.isclose(g.k, 2.0)]
        rows.append(dict(corner=corner, dev_lower=g.f1_GHz.max() - mk.f1_GHz.max(),
                         dev_upper=mk.f2_GHz.min() - g.f2_GHz.min(),
                         k_lower=float(g.k.iloc[g.f1_GHz.argmax()]), k_upper=float(g.k.iloc[g.f2_GHz.argmin()]),
                         n_k=len(g)))
    d = pd.DataFrame(rows)
    counts = lambda x: {str(k): int(v) for k, v in x.round(2).value_counts().items()}  # noqa: E731
    return dict(n_cells=len(d), n_k=int(d.n_k.median()),
                max_dev_GHz=float(max(d.dev_lower.abs().max(), d.dev_upper.abs().max())),
                lower_at=counts(d.k_lower), upper_at=counts(d.k_upper),
                upper_at_rounded=counts(d[d.corner == 1].k_upper), upper_at_sharp=counts(d[d.corner == 0].k_upper))


def check_pwe_mixed():
    files = ["data/pwe/pilot_fourier.csv", "data/pwe/pilot_library.csv", "data/pwe/fourier.csv", "data/pwe/library.csv"]
    d = pd.concat([pd.read_csv(f) for f in files if os.path.exists(f)])
    d = d.dropna(subset=["dev_path_lower"])
    dev = d[["dev_path_lower", "dev_path_upper", "dev_zone_lower", "dev_zone_upper"]].abs().max(axis=1)
    pair = d.famA + "-" + d.famB
    circ = pair == "circle-circle"
    has_circle = (d.famA == "circle") | (d.famB == "circle")
    return dict(n_cells=int(len(d)), max_dev_GHz=float(dev.max()), n_circle_circle=int(circ.sum()),
                n_with_a_circle=int(has_circle.sum()), n_fourier_fourier=int((pair == "fourier-fourier").sum()),
                lower_at={"K": int((d.k_lower == 2).sum()), "M": int((d.k_lower == 1).sum())},
                upper_at={"M": int((d.k_upper == 1).sum()), "K": int((d.k_upper == 2).sum())})


def pick_examples():
    """A rounded COMSOL cell, a circle pair and a strongly mirror-less freeform cell."""
    e = pd.read_csv("data/comsol/band_edges.csv")
    r = e[e.corner == 1].sort_values("gap").iloc[-1]
    rounded = (("rounded", {"L": float(r.L1_um)}), ("rounded", {"L": float(r.L2_um)}))
    circle = (("circle", {"R": 70.0}), ("circle", {"R": 35.0}))
    f = pd.read_csv("data/pwe/fourier.csv")
    f = f[f.gap > 15]
    skew = f.A_s3.abs() + f.B_s3.abs() + f.A_s6.abs() + f.B_s6.abs()
    free = holes(f.loc[skew.idxmax()])
    names = [f"rounded triangles, L1 = {r.L1_um:.0f}, L2 = {r.L2_um:.0f} um",
             "circles, R_A = 70, R_B = 35 um", "freeform (no mirror symmetry)"]
    return names, [rounded, circle, free]


def main():
    t0 = time.time()
    eps = eps_fitted()
    summary = {"eps_si": eps, "comsol": check_comsol(), "pwe_mixed": check_pwe_mixed()}
    print("COMSOL:", summary["comsol"])
    print("PWE mixed:", summary["pwe_mixed"])

    pairs = [(a, b) for b in range(20, 65, 5) for a in range(b, 100, 5) if a + b <= 120]
    names, cells = pick_examples()
    wedge = wedge_grid(24)
    with Pool(max(1, (os.cpu_count() or 2) - 2), initializer=_init, initargs=(eps,)) as pool:
        circ = pd.DataFrame(pool.map(_circle, pairs))
        # band diagrams: split each path into chunks so all workers stay busy
        chunks = np.array_split(PATH_FIG, 8)
        tasks = [(hA, hB, [kpath(k) for k in c]) for hA, hB in cells for c in chunks]
        tasks += [(cells[2][0], cells[2][1], list(w)) for w in np.array_split(wedge, 8)]
        out = pool.map(_bands, tasks)
    paths = [np.vstack(out[8 * i:8 * i + 8]) for i in range(3)]
    wedge_bands = np.vstack(out[24:])

    circ.to_csv(os.path.join(OUT, "mk_circle_check.csv"), index=False)
    summary["pwe_circles"] = dict(
        n_cells=len(circ), r_range_um=[int(circ[["rA", "rB"]].min().min()), int(circ[["rA", "rB"]].max().max())],
        path_step=0.02, wedge_points=len(wedge_grid(10)),
        max_dev_GHz=float(max(circ.dev_lower.abs().max(), circ.dev_upper.abs().max())),
        lower_at=circ.k_lower.value_counts().to_dict(), upper_at=circ.k_upper.value_counts().to_dict(),
        n_no_gap=int((circ.gap <= 0).sum()), gap_max_GHz=float(circ.gap.max()))
    ex = []
    for name, (hA, hB), b in zip(names, cells, paths):
        ex.append(dict(name=name, k_lower=float(PATH_FIG[b[:, 0].argmax()]), k_upper=float(PATH_FIG[b[:, 1].argmin()]),
                       f_lower=float(b[:, 0].max()), f_upper=float(b[:, 1].min())))
    summary["examples"] = ex
    wb_lo, wb_up = wedge_bands[:, 0].argmax(), wedge_bands[:, 1].argmin()
    summary["wedge_map"] = dict(points=len(wedge), band1_max_at=[float(x) for x in wedge[wb_lo]],
                                band2_min_at=[float(x) for x in wedge[wb_up]],
                                K=[float(x) for x in K_PT], M=[float(x) for x in M_PT])
    np.savez(os.path.join(OUT, "mk_examples.npz"), ks=PATH_FIG, paths=np.array(paths), names=np.array(names),
             cells=np.array([json.dumps(c) for c in cells]), wedge=wedge, wedge_bands=wedge_bands, K=K_PT, M=M_PT)
    summary["n_pwe_total"] = summary["pwe_mixed"]["n_cells"] + len(circ)
    summary["n_circle_pairs_total"] = summary["pwe_mixed"]["n_circle_circle"] + len(circ)
    with open(os.path.join(OUT, "mk_check.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("PWE circles:", summary["pwe_circles"])
    print("examples:", ex)
    print("wedge map:", summary["wedge_map"])
    print(f"done in {time.time() - t0:.0f} s -> {OUT}/mk_check.json")


if __name__ == "__main__":
    main()
