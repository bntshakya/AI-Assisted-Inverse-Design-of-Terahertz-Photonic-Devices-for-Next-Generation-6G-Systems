"""Combine repeated runs of run_optionb.py (different random splits) into mean +- standard deviation.

Seed 0 is results/optionb/results.json; further seeds are results/optionb/seeds/seed<N>/results.json, made by
  python src/shapes/run_optionb.py --seed N --out results/optionb/seeds/seedN --no-showcase
Run from the Project folder:  python src/shapes/summarize_seeds.py   -> results/optionb/seed_summary.json
"""
import glob
import json
import os

import numpy as np

OUT = "results/optionb/seed_summary.json"


def load_all():
    runs = {}
    if os.path.exists("results/optionb/results.json"):
        runs[0] = json.load(open("results/optionb/results.json"))
    for path in sorted(glob.glob("results/optionb/seeds/seed*/results.json")):
        r = json.load(open(path))
        runs[int(r.get("seed", os.path.basename(os.path.dirname(path))[4:]))] = r
    return runs


def ms(values):
    """mean and sample standard deviation of a list of numbers or of equal-length lists."""
    a = np.array(values, dtype=float)
    return {"mean": np.round(a.mean(0), 4).tolist(), "std": np.round(a.std(0, ddof=1) if len(a) > 1 else 0 * a.mean(0), 4).tolist()}


def main():
    runs = load_all()
    seeds = sorted(runs)
    out = {"seeds": seeds, "n": len(seeds)}
    out["forward_mae_all"] = ms([runs[s]["forward_test"]["mae_all"] for s in seeds])
    out["forward_gap_mae"] = ms([runs[s]["forward_test"]["gap_mae"] for s in seeds])
    mf = [runs[s]["multifidelity"] for s in seeds]
    out["mf_pwe_raw_mae"] = ms([m["pwe_raw_mae"] for m in mf])
    out["mf_pwe_plus_gp_cv_mae"] = ms([m["pwe_plus_gp_cv_mae"] for m in mf])
    out["mf_nn_raw_mae"] = ms([m["nn_raw_mae"] for m in mf])
    out["mf_gap_raw"] = ms([m["gap_mae"]["pwe_raw"] for m in mf])
    out["mf_gap_gp"] = ms([m["gap_mae"]["pwe_plus_gp"] for m in mf])
    curve = {}
    for s in seeds:
        for row in runs[s]["learning_curve"]:
            c = curve.setdefault(row["n_train"], {"nn": [], "gp": []})
            c["nn"].append(np.mean(row["nn_mae"]))
            if "gp_mae" in row:
                c["gp"].append(np.mean(row["gp_mae"]))
    out["learning_curve"] = {str(n): {"nn": ms(c["nn"]), **({"gp": ms(c["gp"])} if c["gp"] else {})}
                             for n, c in sorted(curve.items())}
    inv = {}
    for name in runs[seeds[0]]["inverse"]:
        rows = [runs[s]["inverse"][name] for s in seeds]
        inv[name] = {"success_rate_within_1GHz": ms([r["success_rate_within_1GHz"] for r in rows]),
                     "verified_mae_GHz": ms([r["verified_mae_GHz"] for r in rows]),
                     "seconds_per_target": ms([r["seconds_per_target"] for r in rows])}
        if "distinct_verified_designs_per_target" in rows[0]:
            inv[name]["distinct"] = ms([r["distinct_verified_designs_per_target"] for r in rows])
    out["inverse"] = inv
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"{len(seeds)} runs (seeds {seeds}) -> {OUT}")
    for name, v in inv.items():
        sr, mae = v["success_rate_within_1GHz"], v["verified_mae_GHz"]
        d = v.get("distinct", {"mean": 1.0, "std": 0.0})
        print(f"  {name:12s} within 1 GHz {100 * sr['mean']:.0f} ± {100 * sr['std']:.0f}%   MAE {mae['mean'][0]:.3f} ± "
              f"{mae['std'][0]:.3f} / {mae['mean'][1]:.3f} ± {mae['std'][1]:.3f} GHz   distinct {d['mean']:.1f} ± {d['std']:.1f}")


if __name__ == "__main__":
    main()
