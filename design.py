"""Instant inverse design from the command line, using the saved networks (no training).

    python design.py 300 330                      # 4 designs with CVAE + refine, each checked with PWE
    python design.py 310 340 --method all --n 3   # compare all four methods
    python design.py 300 330 --model corr         # aim at COMSOL: PWE network + learned COMSOL correction
    python design.py 300 330 --model comsol       # aim at COMSOL: network trained directly on COMSOL data

Prints a table, saves a picture of the designs, a CSV, and a COMSOL sweep file to results/designs/.
Needs models/optionb/ (made once by: python src/shapes/save_models.py).
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "shapes"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import designer as dz  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Target band edges (GHz) -> hole shapes")
    ap.add_argument("f_lower", type=float)
    ap.add_argument("f_upper", type=float)
    ap.add_argument("--method", default="cvae_refine", choices=list(dz.METHODS) + ["all"])
    ap.add_argument("--n", type=int, default=4, help="designs per method (tandem always gives 1)")
    ap.add_argument("--model", default="pwe", choices=["pwe", "corr", "comsol"],
                    help="forward model used to pick designs: pwe (PWE-trained), corr (+ COMSOL correction), comsol (COMSOL-trained)")
    ap.add_argument("--comsol", action="store_true", help="same as --model corr")
    ap.add_argument("--allow-big", action="store_true", help="also allow holes that cross the COMSOL cell edge")
    ap.add_argument("--no-verify", action="store_true", help="skip the PWE check (faster)")
    ap.add_argument("--seed", type=int, default=0, help="change to get other samples")
    a = ap.parse_args()
    if a.f_upper <= a.f_lower:
        sys.exit("f_upper must be larger than f_lower")
    t0 = time.time()
    B = dz.load_bundle()
    methods = list(dz.METHODS) if a.method == "all" else [a.method]
    model = "corr" if a.comsol else a.model
    res = dz.design(B, (a.f_lower, a.f_upper), methods, a.n, None, not a.allow_big, not a.no_verify, a.seed, model=model)
    if not res:
        sys.exit("no valid design found: try a target inside the training range (e.g. 300 330) or --allow-big")

    rows = []
    for i, d in enumerate(res):
        r = {"#": i + 1, "method": dz.METHODS[d["method"]],
             "predicted": f"{d['pred'][0]:.2f} / {d['pred'][1]:.2f}", "pred. error": round(d["err_pred"], 2)}
        if "pwe" in d:
            r["PWE check"] = f"{d['pwe'][0]:.2f} / {d['pwe'][1]:.2f}"
            r["PWE error"] = round(d["err_pwe"], 2)
        r.update({"wall um": round(d["wall"], 1), "reach um": round(d["reach"], 1), "fits COMSOL": d["fits_comsol"]})
        rows.append(r)
    table = pd.DataFrame(rows)
    print(f"\nTarget {a.f_lower} / {a.f_upper} GHz  (model: {dz.MODELS[model]}; "
          f"networks trained {B['info'].get('trained', '?')})\n")
    print(table.to_string(index=False))

    out = os.path.join("results", "designs")
    os.makedirs(out, exist_ok=True)
    tag = f"{a.f_lower:g}_{a.f_upper:g}"
    n = len(res)
    cols = min(4, n)
    fig, axs = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(3 * cols, 3 * int(np.ceil(n / cols))), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for i, d in enumerate(res):
        ax = axs.ravel()[i]
        ax.axis("on")
        lab = f"PWE {d['pwe'][0]:.1f}/{d['pwe'][1]:.1f}" if "pwe" in d else f"pred {d['pred'][0]:.1f}/{d['pred'][1]:.1f}"
        dz.draw_cell(ax, d["p"], title=f"#{i + 1} {dz.METHODS[d['method']]}\n{lab} GHz")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"design_{tag}.png"), dpi=130)
    P = np.array([d["p"] for d in res])
    pd.concat([table, pd.DataFrame(P.round(3), columns=[x + "_um" for x in dz.NAMES])], axis=1).to_csv(
        os.path.join(out, f"design_{tag}.csv"), index=False)
    with open(os.path.join(out, f"design_{tag}_comsol.txt"), "w") as f:
        f.write(dz.comsol_sweep(P))
    print(f"\nsaved results/designs/design_{tag}.png, .csv and _comsol.txt  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
