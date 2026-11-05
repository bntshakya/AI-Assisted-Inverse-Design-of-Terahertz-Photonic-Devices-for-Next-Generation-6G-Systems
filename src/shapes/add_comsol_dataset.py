"""Add a new COMSOL sweep (e.g. the round/circular-hole cells) to the shape-agnostic dataset.

Run from the Project folder, e.g. for circular holes with radii r1 (hole A) and r2 (hole B):
  python src/shapes/add_comsol_dataset.py "data/comsol/<your circle export>.csv" --family circle --name circle

Input: a COMSOL export with one row per mesh node, like the two existing files:
  X, Y, <param 1>, <param 2>, k, lambda, emw.freq      ('%' header lines are skipped automatically)
--usecols gives the 0-based positions of (param1, param2, k, freq) if the layout differs.
param1 belongs to hole A and param2 to hole B; their meaning depends on --family:
  circle  -> radius R          rounded / sharp -> side length L
Lengths in metres are converted to um automatically (values < 1e-2 are taken as metres).

Writes data/comsol/extra_<name>.csv in the unified table format (dataset.py picks it up), then
runs the PWE solver on every geometry and prints the PWE-vs-COMSOL error. That tells you at once
whether the new model uses the same permittivity (the fitted eps ~ 8.91) as the old ones.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from process_bands import count_header_lines  # noqa: E402
from pwe import Solver  # noqa: E402
from gen_dataset import eps_from_validation  # noqa: E402

PARAM = {"circle": "R", "rounded": "L", "sharp": "L"}


def read_states(path, usecols, chunksize=2_000_000):
    skip = count_header_lines(path)
    with open(path, encoding="utf8", errors="replace") as f:
        head = [next(f).rstrip() for _ in range(skip)]
    print("header:", *head[-2:], sep="\n  ")
    parts = []
    for chunk in pd.read_csv(path, skiprows=skip, header=None, usecols=usecols, chunksize=chunksize):
        chunk.columns = ["p1", "p2", "k", "freq"]
        chunk["freq"] = pd.to_numeric(chunk["freq"], errors="coerce")
        chunk = chunk.dropna().round({"p1": 9, "p2": 9, "k": 4, "freq": 6}).drop_duplicates()
        parts.append(chunk)
        print(f"  chunk {len(parts)}", flush=True)
    return pd.concat(parts).drop_duplicates()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--family", choices=list(PARAM), default="circle")
    p.add_argument("--name", default=None, help="dataset name (default: the family)")
    p.add_argument("--usecols", type=int, nargs=4, default=(2, 3, 4, 6))
    args = p.parse_args()
    name = args.name or args.family

    s = read_states(args.csv, list(args.usecols))
    g = s.groupby(["p1", "p2", "k"])["freq"]
    n = g.size()
    if n.min() < 2:
        sys.exit(f"some (p1, p2, k) have fewer than 2 eigenfrequencies: {n.value_counts().to_dict()}")
    # the two lowest eigenfrequencies at each k are bands 1 and 2 (works with >= 2 per k)
    bands = g.apply(lambda f: pd.Series(np.sort(f.to_numpy())[:2], index=["f1", "f2"])).unstack().reset_index()
    edges = bands.groupby(["p1", "p2"]).agg(f_lower=("f1", "max"), f_upper=("f2", "min")).reset_index()
    edges["gap"] = edges.f_upper - edges.f_lower
    scale = 1e6 if edges[["p1", "p2"]].to_numpy().max() < 1e-2 else 1.0
    edges[["p1", "p2"]] = (edges[["p1", "p2"]] * scale).round(3)
    print(f"{len(edges)} geometries, k values {sorted(bands.k.unique())[:5]}..., "
          f"{int((edges.gap > 0.1).sum())} with gap > 0.1 GHz")

    key = PARAM[args.family]
    out = pd.DataFrame({
        "id": [f"comsol_{name}_{a:.2f}_{b:.2f}" for a, b in zip(edges.p1, edges.p2)],
        "dataset": "comsol_" + name, "famA": args.family, "famB": args.family,
        "paramsA": [json.dumps({key: float(v)}) for v in edges.p1],
        "paramsB": [json.dumps({key: float(v)}) for v in edges.p2],
        "f_lower": edges.f_lower, "f_upper": edges.f_upper, "gap": edges.gap})

    solver = Solver(eps_si=eps_from_validation())
    pw = np.array([solver.band_edges((args.family, {key: a}), (args.family, {key: b}))
                   for a, b in zip(edges.p1, edges.p2)])
    out["f_lower_pwe"], out["f_upper_pwe"] = pw[:, 0], pw[:, 1]
    path = f"data/comsol/extra_{name}.csv"
    out.to_csv(path, index=False)
    ok = out.gap > 0.1
    err = out.loc[ok, ["f_lower_pwe", "f_upper_pwe"]].to_numpy() - out.loc[ok, ["f_lower", "f_upper"]].to_numpy()
    print(f"wrote {path}")
    print(f"PWE - COMSOL on gapped cells: mean {err.mean(0).round(2)} GHz, MAE {np.abs(err).mean(0).round(2)} GHz"
          " (f_lower, f_upper). A few GHz or less = same model settings as the old sweeps.")


if __name__ == "__main__":
    main()
