"""Compare the COMSOL results of the AI designs (sweep_verify_designs.txt) with their targets and with PWE.

Run from the Project folder after importing the export:
    python src/comsol/import_global_eval.py data/comsol/new/global_eval_verify_designs.csv --family fourier --name verify
    python src/comsol/compare_verify.py
Writes results/optionb/comsol_verify.csv and prints the table.
"""
import argparse
import json

import numpy as np
import pandas as pd

NAMES = ["rA0", "c3A", "s3A", "c6A", "s6A", "rB0", "c3B", "s3B", "c6B", "s6B"]
KEYS = ["r0", "c3", "s3", "c6", "s6"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="data/comsol/new/verify_designs.csv")
    ap.add_argument("--got", default="data/comsol/extra_verify.csv")
    ap.add_argument("--out", default="results/optionb/comsol_verify.csv")
    a = ap.parse_args()
    plan = pd.read_csv(a.plan)
    got = pd.read_csv(a.got)
    P = np.array([[json.loads(a)[k] for k in KEYS] + [json.loads(b)[k] for k in KEYS]
                  for a, b in zip(got.paramsA, got.paramsB)])
    rows = []
    for _, r in plan.iterrows():
        p = r[NAMES].to_numpy(float)
        j = np.where(np.abs(P - p).max(1) < 0.01)[0]
        if not len(j):
            continue
        g = got.iloc[j[0]]
        rows.append({"id": r.id, "kind": r.kind, "target": f"{r.target_lower:g} / {r.target_upper:g}",
                     "PWE": f"{r.f_lower_pwe:.2f} / {r.f_upper_pwe:.2f}", "COMSOL": f"{g.f_lower:.2f} / {g.f_upper:.2f}",
                     "COMSOL - target lower": round(g.f_lower - r.target_lower, 2),
                     "COMSOL - target upper": round(g.f_upper - r.target_upper, 2)})
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(df.to_string(index=False))
    new = df[~df.kind.str.startswith("check")]
    if len(new):
        worst = np.maximum(new["COMSOL - target lower"].abs(), new["COMSOL - target upper"].abs())
        print(f"\n{len(new)} new designs: {int((worst < 1).sum())} within 1 GHz of the target in COMSOL "
              f"(worst {worst.max():.2f} GHz, mean {worst.mean():.2f} GHz)")


if __name__ == "__main__":
    main()
