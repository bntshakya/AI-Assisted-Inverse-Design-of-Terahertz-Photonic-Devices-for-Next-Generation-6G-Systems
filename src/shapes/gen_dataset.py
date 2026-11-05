"""Generate a labelled dataset of freeform C3 hole shapes with the PWE solver (option B).

Run from the Project folder, e.g.
  python src/shapes/gen_dataset.py --kind fourier --n 3000            # both holes = C3 Fourier shapes
  python src/shapes/gen_dataset.py --kind library --n 1500            # each hole = circle / rounded
                                                                       #   / sharp triangle / Fourier,
                                                                       #   any size and rotation
  python src/shapes/gen_dataset.py --kind fourier --n 300 --check 1   # pilot: also full k-path + 2D zone

Kinds
  fourier  r(theta) = r0 + c3 cos3t + s3 sin3t + c6 cos6t + s6 sin6t for EACH hole (10 numbers per cell).
           Sampled as r0 in [25, 75] um, |c3, s3| = a3 r0 with a3 in [0, 0.35] and a random phase,
           |c6, s6| = a6 r0 with a6 in [0, 0.12] and an independent random phase. These ranges contain
           the old rounded (c3/r0 = -0.16) and sharp (-0.275, c6/r0 = 0.10) triangles.
  library  each hole independently picks circle / rounded / sharp / fourier with a random size and
           rotation: the "mix of 3-4 different shapes" design space.
Only cells passing geometry.check_cell (walls >= 25 um, radius >= 10 um, corner radius >= 8 um) are kept.

Labels: f_lower = max over k of band 1, f_upper = min over k of band 2, using M and K
(the COMSOL shortcut). With --check f, a random fraction f of the cells is also solved on the
full Gamma-M-K-Gamma path (29 points) and on a 2D grid of the irreducible zone, and the
difference is stored (dev_path_*, dev_zone_* in GHz) to test whether the shortcut still holds.

Writes data/pwe/<kind>.csv: id, kind, famA, famB, paramsA, paramsB (JSON), 18 Fourier features,
f_lower, f_upper, gap, k_lower, k_upper, min_wall, air_frac (+ check columns).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one BLAS thread per worker process
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from geometry import FEATURE_NAMES, air_fraction, cell_features, check_cell
from pwe import K_PT, M_PT, Solver, kpath, wedge_grid

OUT_DIR = "data/pwe"
PATH_K = np.round(np.linspace(0.1, 2.9, 29), 4)


def eps_from_validation(default=8.914):
    try:
        with open("results/pwe_validation.json") as f:
            return json.load(f)["eps_si"]
    except OSError:
        return default


# ------------------------------------------------------------------------------- sampling
def sample_fourier_hole(rng, r0=(25.0, 75.0), a3=(0.0, 0.35), a6=(0.0, 0.12)):
    r = rng.uniform(*r0)
    A3, A6 = rng.uniform(*a3) * r, rng.uniform(*a6) * r
    p3, p6 = rng.uniform(0, 2 * np.pi, 2)
    return ("fourier", {"r0": r, "c3": A3 * np.cos(p3), "s3": A3 * np.sin(p3),
                        "c6": A6 * np.cos(p6), "s6": A6 * np.sin(p6)})


def sample_library_hole(rng):
    fam = rng.choice(["circle", "rounded", "sharp", "fourier"])
    if fam == "circle":
        return ("circle", {"R": rng.uniform(25.0, 75.0)})
    if fam in ("rounded", "sharp"):
        return (fam, {"L": rng.uniform(80.0, 200.0), "rot": rng.uniform(-60.0, 60.0)})
    return sample_fourier_hole(rng)


def sample_cells(kind, n, seed):
    rng = np.random.default_rng(seed)
    cells, tried = [], 0
    draw = sample_fourier_hole if kind == "fourier" else sample_library_hole
    while len(cells) < n:
        tried += 1
        hA, hB = draw(rng), draw(rng)
        ok, info = check_cell(hA, hB)
        if ok:
            cells.append((hA, hB, info["min_wall"]))
    print(f"{n} valid cells from {tried} draws ({100 * n / tried:.0f}% accepted)", flush=True)
    return cells


# ------------------------------------------------------------------------------- solving
_SOLVER = None


def _init(eps):
    global _SOLVER
    _SOLVER = Solver(eps_si=eps)


def _solve(task):
    i, hA, hB, check = task
    s = _SOLVER
    inv = s.inv_eps_matrix(hA, hB)
    mk = s.bands_kvec(hA, hB, [M_PT, K_PT], inv=inv)
    out = {"f1_M": mk[0, 0], "f2_M": mk[0, 1], "f1_K": mk[1, 0], "f2_K": mk[1, 1]}
    if check:
        path = s.bands_kvec(hA, hB, [kpath(k) for k in PATH_K], inv=inv)
        zone = s.bands_kvec(hA, hB, wedge_grid(6), inv=inv)
        out.update(path_lower=path[:, 0].max(), path_upper=path[:, 1].min(),
                   path_k_lower=PATH_K[path[:, 0].argmax()], path_k_upper=PATH_K[path[:, 1].argmin()],
                   zone_lower=max(zone[:, 0].max(), path[:, 0].max()),
                   zone_upper=min(zone[:, 1].min(), path[:, 1].min()))
    return i, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["fourier", "library"], default="fourier")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--check", type=float, default=0.1, help="fraction also solved on path + 2D zone")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    eps = eps_from_validation()
    cells = sample_cells(args.kind, args.n, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    tasks = [(i, hA, hB, rng.random() < args.check) for i, (hA, hB, _) in enumerate(cells)]
    t0 = time.time()
    res = {}
    with Pool(args.workers, initializer=_init, initargs=(eps,)) as pool:
        for j, (i, out) in enumerate(pool.imap_unordered(_solve, tasks, chunksize=8)):
            res[i] = out
            if (j + 1) % 250 == 0:
                print(f"  {j + 1}/{len(tasks)}  {time.time() - t0:.0f} s", flush=True)
    print(f"solved {len(tasks)} cells in {time.time() - t0:.0f} s with {args.workers} workers, eps = {eps:.3f}")

    rows = []
    for i, (hA, hB, wall) in enumerate(cells):
        o = res[i]
        f_lower, f_upper = max(o["f1_M"], o["f1_K"]), min(o["f2_M"], o["f2_K"])
        row = {"id": f"{args.kind}{args.seed}_{i:05d}", "kind": args.kind, "famA": hA[0], "famB": hB[0],
               "paramsA": json.dumps({k: round(float(v), 4) for k, v in hA[1].items()}),
               "paramsB": json.dumps({k: round(float(v), 4) for k, v in hB[1].items()})}
        row.update(dict(zip(FEATURE_NAMES, np.round(cell_features(hA, hB), 4))))
        row.update(f_lower=f_lower, f_upper=f_upper, gap=f_upper - f_lower,
                   k_lower=1.0 if o["f1_M"] > o["f1_K"] else 2.0,
                   k_upper=1.0 if o["f2_M"] < o["f2_K"] else 2.0,
                   min_wall=wall, air_frac=air_fraction(hA, hB))
        if "path_lower" in o:
            row.update(dev_path_lower=o["path_lower"] - f_lower, dev_path_upper=o["path_upper"] - f_upper,
                       dev_zone_lower=o["zone_lower"] - f_lower, dev_zone_upper=o["zone_upper"] - f_upper,
                       path_k_lower=o["path_k_lower"], path_k_upper=o["path_k_upper"])
        rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = args.out or os.path.join(OUT_DIR, f"{args.kind}.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} cells, {int((df.gap > 0.1).sum())} with gap > 0.1 GHz; "
          f"gap {df.gap.min():.1f} to {df.gap.max():.1f} GHz")
    if "dev_path_lower" in df:
        c = df.dropna(subset=["dev_path_lower"])
        for q in ("path", "zone"):
            lo, up = c[f"dev_{q}_lower"], -c[f"dev_{q}_upper"]  # both >= 0 by construction
            print(f"  {q} check on {len(c)} cells: band edge moved by > 0.1 GHz in "
                  f"{int(((lo > 0.1) | (up > 0.1)).sum())}; max shift lower {lo.max():.2f}, upper {up.max():.2f} GHz")


if __name__ == "__main__":
    main()
