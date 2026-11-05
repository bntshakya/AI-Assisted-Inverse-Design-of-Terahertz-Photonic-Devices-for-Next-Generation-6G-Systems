"""Import a COMSOL Global Evaluation table (one row per shape, k and eigenfrequency; no X, Y) into the project.

Input: the table exported from Results > Derived Values > Global Evaluation (emw.freq) after a k = 1, 2 sweep, e.g.
    % L1 (m),L2 (m),k,Eigenfrequency (GHz),Frequency (GHz)                       (triangles or circles: 2 numbers)
    % rA0 (m),c3A (m),...,s6B (m),k,Eigenfrequency (GHz),Frequency (GHz)          (freeform: 10 numbers)
Run from the Project folder (several files, e.g. all batches, can be given at once):
    python src/comsol/import_global_eval.py data/comsol/new/global_eval_batch1.csv --family rounded
    python src/comsol/import_global_eval.py data/comsol/new/global_eval_fourier_batch1.csv --family fourier

What it does
  1. band edges per shape: f_lower = max(band 1 at k=1, k=2), f_upper = min(band 2 at k=1, k=2)
  2. check cells: triangles already in data/comsol/band_edges.csv, or freeform cells that are plain circles
     (all c, s = 0) and exist in data/comsol/extra_circle.csv, are compared with the earlier COMSOL data, then dropped
  3. every new shape is solved with the PWE solver and the PWE - COMSOL error is printed
  4. writes data/comsol/extra_<name>.csv in the shared format, so run_optionb.py / dataset.py pick it up
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shapes"))
from gen_dataset import eps_from_validation  # noqa: E402
from pwe import Solver  # noqa: E402

FAMILIES = ("rounded", "sharp", "circle", "fourier")
CORNER = {"rounded": 1, "sharp": 0}
FOURIER_KEYS = ["r0", "c3", "s3", "c6", "s6"]


def read(path):
    head = None
    with open(path, encoding="utf8", errors="replace") as f:
        for line in f:
            if line.startswith("%"):
                head = line[1:].strip()
            else:
                break
    names = [c.strip() for c in head.split(",")]
    d = pd.read_csv(path, comment="%", header=None, names=names)
    ki = next(i for i, c in enumerate(names) if c.split(" ")[0] == "k")
    pcols = names[:ki]
    fcol = ([c for c in names if c.lower().startswith("frequency")] or [names[-1]])[0]
    out = pd.DataFrame({f"p{j}": d[c].astype(float) for j, c in enumerate(pcols)})
    out["k"] = d[names[ki]].astype(float)
    out["f"] = pd.to_numeric(d[fcol], errors="coerce")
    out = out.dropna()
    p = [f"p{j}" for j in range(len(pcols))]
    if any("(m)" in c for c in pcols) or out[p].abs().to_numpy().max() < 1e-2:
        out[p] = out[p] * 1e6                                               # metres -> um
    out[p] = out[p].round(3)
    print(f"{os.path.basename(path)}: {len(out)} rows, parameters {pcols}, k column {names[ki]!r}, value {fcol!r}")
    return out, p


def holes_of(family, row):
    if family == "fourier":
        return (("fourier", dict(zip(FOURIER_KEYS, row[:5]))), ("fourier", dict(zip(FOURIER_KEYS, row[5:10]))))
    key = "R" if family == "circle" else "L"
    return ((family, {key: row[0]}), (family, {key: row[1]}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tables", nargs="+")
    ap.add_argument("--family", choices=FAMILIES, default="rounded")
    ap.add_argument("--max-reach", type=float, default=69.5, help="fourier: drop holes reaching this far (um)")
    ap.add_argument("--name", default=None, help="dataset name (default: <family>_lhs, or 'fourier' for freeform)")
    args = ap.parse_args()
    name = args.name or ("fourier" if args.family == "fourier" else f"{args.family}_lhs")
    parts = [read(t) for t in args.tables]
    pc = parts[0][1]
    if args.family == "fourier" and len(pc) != 10:
        sys.exit(f"expected 10 shape parameters for --family fourier, found {len(pc)}")
    d = pd.concat([x for x, _ in parts]).drop_duplicates()

    n = d.groupby(pc + ["k"]).size()
    if n.min() < 2:
        sys.exit(f"some (shape, k) have fewer than 2 eigenfrequencies: {n.value_counts().to_dict()}")
    d = d.sort_values("f")
    d["band"] = d.groupby(pc + ["k"]).cumcount() + 1
    b = d[d.band <= 2].pivot_table(index=pc, columns=["k", "band"], values="f")
    P = np.array(b.index.tolist(), dtype=float).reshape(len(b), len(pc))
    e = pd.DataFrame(P, columns=pc)
    e["f_lower"] = np.maximum(b[(1.0, 1)], b[(2.0, 1)]).to_numpy()
    e["f_upper"] = np.minimum(b[(1.0, 2)], b[(2.0, 2)]).to_numpy()
    e["gap"] = e.f_upper - e.f_lower
    print(f"{len(e)} shapes, {int((e.gap > 0.1).sum())} with a gap; gap {e.gap.min():.1f} to {e.gap.max():.1f} GHz")

    # check cells: compare with earlier COMSOL data, then drop
    keep = np.ones(len(e), bool)
    if args.family in CORNER:
        old = pd.read_csv("data/comsol/band_edges.csv")
        old = old[old.corner == CORNER[args.family]]
        ref = [(r.L1_um, r.L2_um, r.f_lower, r.f_upper) for r in old.itertuples()]
    elif os.path.exists("data/comsol/extra_circle.csv"):
        c = pd.read_csv("data/comsol/extra_circle.csv")
        ref = [(json.loads(a)["R"], json.loads(bb)["R"], lo, up) for a, bb, lo, up in
               zip(c.paramsA, c.paramsB, c.f_lower, c.f_upper)]
    else:
        ref = []
    for i, r in e.iterrows():
        v = r[pc].to_numpy()
        if args.family == "fourier":
            if np.abs(np.r_[v[1:5], v[6:10]]).max() > 1e-6:
                continue                                    # not a plain circle
            a1, a2 = v[0], v[5]
        else:
            a1, a2 = v[0], v[1]
        for x1, x2, lo, up in ref:
            if abs(x1 - a1) < 0.02 and abs(x2 - a2) < 0.02:
                keep[i] = False
                print(f"  check {a1:.2f}/{a2:.2f} um: new - earlier COMSOL = {r.f_lower - lo:+.3f} / {r.f_upper - up:+.3f} "
                      "GHz (f_lower / f_upper)")
    e = e[keep].reset_index(drop=True)
    print(f"{len(e)} new shapes after removing the check cells")
    if args.family == "fourier":
        # the rhombic COMSOL cell: each hole centre is 70 um from two cell edges; larger holes are cut off there
        from geometry import check_cell
        reach = np.array([max(check_cell(*holes_of("fourier", row))[1][k] for k in ("rmax_A", "rmax_B"))
                          for row in e[pc].to_numpy()])
        bad = reach >= args.max_reach
        if bad.any():
            print(f"dropping {int(bad.sum())} shapes whose holes reach >= {args.max_reach} um (they cross the COMSOL cell "
                  "edge, so COMSOL simulated a different shape)")
        e = e[~bad].reset_index(drop=True)

    # PWE comparison
    s = Solver(eps_si=eps_from_validation())
    pw = np.array([s.band_edges(*holes_of(args.family, row)) for row in e[pc].to_numpy()])
    e["f_lower_pwe"], e["f_upper_pwe"] = pw[:, 0], pw[:, 1]
    g = e[e.gap > 0.1]
    if len(g):
        err = g[["f_lower_pwe", "f_upper_pwe"]].to_numpy() - g[["f_lower", "f_upper"]].to_numpy()
        gerr = (g.f_upper_pwe - g.f_lower_pwe) - g.gap
        print(f"PWE - COMSOL on {len(g)} gapped shapes: mean {err.mean(0).round(2)}, MAE {np.abs(err).mean(0).round(2)}, "
              f"max {np.abs(err).max(0).round(2)} GHz (f_lower, f_upper); gap MAE {gerr.abs().mean():.2f} GHz")

    hs = [holes_of(args.family, row) for row in e[pc].to_numpy()]
    out = pd.DataFrame({
        "id": [f"comsol_{name}_{i:05d}" for i in range(len(e))],
        "dataset": "comsol_" + name, "famA": args.family, "famB": args.family,
        "paramsA": [json.dumps({k: round(float(v), 4) for k, v in h[0][1].items()}) for h in hs],
        "paramsB": [json.dumps({k: round(float(v), 4) for k, v in h[1][1].items()}) for h in hs],
        "f_lower": e.f_lower, "f_upper": e.f_upper, "gap": e.gap,
        "f_lower_pwe": e.f_lower_pwe, "f_upper_pwe": e.f_upper_pwe})
    path = f"data/comsol/extra_{name}.csv"
    out.to_csv(path, index=False)
    print("wrote", path)


if __name__ == "__main__":
    main()
