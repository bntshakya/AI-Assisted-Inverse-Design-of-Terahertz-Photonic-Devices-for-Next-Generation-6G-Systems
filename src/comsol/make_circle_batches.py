"""1000 random circle-pair cells for COMSOL, split into 5 Parametric Sweep load files.

Run from the Project folder:  python src/comsol/make_circle_batches.py
Writes data/comsol/new/sweep_circles_1000_batch1..5.txt (names r1 r2 k, values in metres, every cell at k = 1 and 2)
and samples_circles_1000.csv. Batch 1 starts with 3 cells from the old 7 x 7 circle grid as checks
(import_global_eval.py compares them with data/comsol/extra_circle.csv and drops them).
Range: r1 (hole A) 46.67-65 um, r2 (hole B) 28-46.67 um; every hole stays inside the COMSOL cell (reach < 68 um).
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import qmc

OUT = "data/comsol/new"
N, BATCH, SEED = 1000, 200, 21
CHECKS = [(53.668, 42.002), (60.667, 35.002), (46.669, 32.668)]  # on the old circle grid


def main():
    lhs = qmc.scale(qmc.LatinHypercube(d=2, seed=SEED).random(N), [46.67, 28.0], [65.0, 46.67]).round(3)
    df = pd.DataFrame({"id": [f"c{i:05d}" for i in range(N)], "r1_um": lhs[:, 0], "r2_um": lhs[:, 1], "is_check": 0})
    chk = pd.DataFrame({"id": [f"chk{i}" for i in range(len(CHECKS))], "r1_um": [c[0] for c in CHECKS],
                        "r2_um": [c[1] for c in CHECKS], "is_check": 1})
    df = pd.concat([chk, df], ignore_index=True)
    df["batch"] = [1] * len(CHECKS) + [1 + i // BATCH for i in range(N)]
    df.to_csv(os.path.join(OUT, "samples_circles_1000.csv"), index=False)
    fmt = lambda a: " ".join(f"{v:.9g}" for v in a)
    for b, g in df.groupby("batch"):
        r1 = np.repeat(g.r1_um.to_numpy() * 1e-6, 2)
        r2 = np.repeat(g.r2_um.to_numpy() * 1e-6, 2)
        k = np.tile([1.0, 2.0], len(g))
        with open(os.path.join(OUT, f"sweep_circles_1000_batch{b}.txt"), "w") as f:
            for name, vals in (("r1", r1), ("r2", r2), ("k", k)):
                f.write(f'{name} "{fmt(vals)}"\n')
        print(f"batch {b}: {len(g)} cells ({int(g.is_check.sum())} check) -> {2 * len(g)} solves")


if __name__ == "__main__":
    main()
