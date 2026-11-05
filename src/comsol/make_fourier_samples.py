"""Random freeform (C3 Fourier) unit cells for a COMSOL Parametric Sweep at k = 1 and 2.

Each cell = 10 numbers [rA0, c3A, s3A, c6A, s6A, rB0, c3B, s3B, c6B, s6B] (um), drawn exactly like the PWE
'fourier' dataset (src/shapes/models.py: u_to_p) and kept only if it passes the fabrication check.
Batch 1 starts with check cells: 2 circle cells from your COMSOL circle sweep (all c, s = 0) and 2 gallery
designs, so the COMSOL Fourier-hole model can be validated before the rest is trusted.

Run from the Project folder:
    python src/comsol/make_fourier_samples.py --n 1000 --batch 200
Writes data/comsol/new/sweep_fourier_batch<i>.txt (Load from File, values in metres; every cell twice: k = 1, 2)
and data/comsol/new/samples_fourier.csv (the same cells in um, with an id).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shapes"))
from geometry import check_cell  # noqa: E402
from models import p_to_holes, u_to_p  # noqa: E402

NAMES = ["rA0", "c3A", "s3A", "c6A", "s6A", "rB0", "c3B", "s3B", "c6B", "s6B"]
OUT = "data/comsol/new"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max-reach", type=float, default=68.0,
                    help="largest distance (um) a hole may reach from its centre. In the rhombic COMSOL cell each hole "
                         "centre is 70 um from two cell edges; larger holes would cross the edge (wrong in COMSOL)")
    ap.add_argument("--first-batch", type=int, default=1, help="number of the first batch file")
    ap.add_argument("--no-checks", action="store_true", help="do not add the check cells")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    checks = [("chk_circle_53.67_42.00", [53.670, 0, 0, 0, 0, 42.002, 0, 0, 0, 0]),
              ("chk_circle_60.67_35.00", [60.670, 0, 0, 0, 0, 35.002, 0, 0, 0, 0])]
    if os.path.exists("data/pwe/showcase_designs.csv"):
        sc = pd.read_csv("data/pwe/showcase_designs.csv")
        for i in range(min(2, len(sc))):
            checks.append((f"chk_gallery_{i}", sc.loc[i, [n + "_um" for n in NAMES]].astype(float).tolist()))

    rows = []
    while len(rows) < args.n:
        p = u_to_p(rng.random((1, 10)))[0]
        ok, info = check_cell(*p_to_holes(p))
        if ok and max(info["rmax_A"], info["rmax_B"]) <= args.max_reach:
            rows.append((f"f{len(rows):05d}", p.round(3).tolist()))
    if args.no_checks:
        checks = []
    for name, p in checks:
        info = check_cell(*p_to_holes(np.array(p)))[1]
        if max(info["rmax_A"], info["rmax_B"]) > args.max_reach:
            print(f"note: check cell {name} reaches {max(info['rmax_A'], info['rmax_B']):.1f} um (crosses the cell edge)")
    cells = checks + rows
    df = pd.DataFrame([dict(id=i, **dict(zip(NAMES, p))) for i, p in cells])
    df["is_check"] = df.id.str.startswith("chk")
    os.makedirs(OUT, exist_ok=True)
    tag = "" if args.first_batch == 1 else f"_from_batch{args.first_batch}"
    df.to_csv(os.path.join(OUT, f"samples_fourier{tag}.csv"), index=False)

    body = df[~df.is_check]
    for b in range(int(np.ceil(len(body) / args.batch))):
        part = body.iloc[b * args.batch:(b + 1) * args.batch]
        if b == 0 and len(checks):
            part = pd.concat([df[df.is_check], part])
        path = os.path.join(OUT, f"sweep_fourier_batch{b + args.first_batch}.txt")
        with open(path, "w") as f:
            for nm in NAMES:
                vals = np.repeat(part[nm].to_numpy() * 1e-6, 2)
                f.write(f'{nm} "' + " ".join(f"{v:.6g}" for v in vals) + '"\n')
            f.write('k "' + " ".join(["1 2"] * len(part)) + '"\n')
        print(f"{path}: {len(part)} cells, {2 * len(part)} solves")
    print(f"{len(df)} cells in total ({int(df.is_check.sum())} check cells) -> {OUT}/samples_fourier.csv")


if __name__ == "__main__":
    main()
