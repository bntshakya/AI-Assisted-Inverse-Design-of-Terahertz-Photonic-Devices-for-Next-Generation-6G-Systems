"""Drive COMSOL from Python (MPh) to compute band edges for every geometry in samples.csv.

Needs COMSOL 6.x installed and licensed on this machine, and `pip install mph` in the .venv.

One-time model prep in the COMSOL GUI (File > Save As a copy, e.g. data/comsol/new/vpc_light.mph):
  - Study 1 > Parametric Sweep: right-click > Disable. Python sets L1, L2 and k itself.
  - Eigenfrequency step: leave the number of eigenfrequencies and the search shift as they are,
    so bands 1 and 2 mean the same as in the old data.
  - Optional, faster and smaller: Eigenfrequency step > Values of Dependent Variables >
    Store fields in output: For selections, with no selection.

Run from the Project folder:
  python src/comsol/run_mph.py --model data/comsol/new/vpc_light.mph --check    # check geometries only
  python src/comsol/run_mph.py --model data/comsol/new/vpc_light.mph            # everything
  # parallel: one process per shard, each with its own COMSOL client and some of the cores
  python src/comsol/run_mph.py --model ... --shard 0/2 --cores 4
  python src/comsol/run_mph.py --model ... --shard 1/2 --cores 4
  python src/comsol/run_mph.py --collect     # merge results into bands_new.csv / band_edges_new.csv

Raw eigenfrequencies are appended to data/comsol/new/results_mph*.csv after every geometry,
so a run can be stopped and restarted; finished ids are skipped.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from bands_to_edges import compare_with_old, to_bands, to_edges
from make_samples import DEFAULT_K, OUT_DIR


def eigenfrequencies(model, expr):
    """All eigenfrequencies of the last solve, GHz, real part, ascending."""
    f = np.asarray(model.evaluate(expr, "GHz"))
    return np.sort(np.real(np.atleast_1d(f)).ravel())


def solve_all(args):
    import mph  # imported here so --collect works without COMSOL

    samples = pd.read_csv(os.path.join(OUT_DIR, "samples.csv"))
    if args.check:
        samples = samples[samples.is_check == 1]
    i, n = (int(v) for v in args.shard.split("/"))
    samples = samples.iloc[i::n]
    out = os.path.join(OUT_DIR, f"results_mph_shard{i}of{n}.csv")
    done = set(pd.read_csv(out).id) if os.path.exists(out) else set()
    todo = samples[~samples.id.isin(done)]
    print(f"{len(todo)} geometries to run ({len(done)} already in {out})", flush=True)
    if todo.empty:
        return

    client = mph.start(cores=args.cores)
    model = client.load(args.model)
    names = dict(zip(("L1", "L2", "k"), args.names))
    print("model parameters:", {k: model.parameter(v) for k, v in names.items()}, flush=True)

    for _, s in todo.iterrows():
        t0 = time.time()
        rows = []
        try:
            model.parameter(names["L1"], f"{s.L1_um}[um]")
            model.parameter(names["L2"], f"{s.L2_um}[um]")
            for k in args.k:
                model.parameter(names["k"], f"{k}")
                model.solve(args.study)
                f = eigenfrequencies(model, args.expr)
                rows += [dict(id=s.id, L1_um=s.L1_um, L2_um=s.L2_um, k=k, freq_GHz=v) for v in f]
        except Exception as e:  # keep going; the id stays missing and is retried next run
            print(f"{s.id} FAILED: {e}", flush=True)
            continue
        pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        freqs = {k: [round(r["freq_GHz"], 3) for r in rows if r["k"] == k] for k in args.k}
        print(f"{s.id} L1={s.L1_um} L2={s.L2_um} {freqs} ({time.time() - t0:.1f} s)", flush=True)


def collect(args):
    files = glob.glob(os.path.join(OUT_DIR, "results_mph_shard*.csv"))
    if not files:
        sys.exit(f"no results_mph_shard*.csv in {OUT_DIR}")
    long = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    bands = to_bands(long[["L1_um", "L2_um", "k", "freq_GHz"]], args.corner)
    edges = to_edges(bands)
    bands.to_csv(os.path.join(OUT_DIR, "bands_new.csv"), index=False)
    edges.to_csv(os.path.join(OUT_DIR, "band_edges_new.csv"), index=False)
    print(f"{len(edges)} geometries, {int((edges.gap > 0.1).sum())} with gap > 0.1 GHz "
          f"-> {OUT_DIR}/band_edges_new.csv")
    compare_with_old(edges)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="path to the prepared .mph file")
    p.add_argument("--study", default=None, help="study tag/name to solve (default: all studies)")
    p.add_argument("--expr", default="emw.freq", help="global eigenfrequency expression")
    p.add_argument("--names", nargs=3, default=("L1", "L2", "k"), help="COMSOL parameter names")
    p.add_argument("--k", type=float, nargs="+", default=DEFAULT_K)
    p.add_argument("--corner", type=int, default=1, help="1 rounded, 0 sharp (label in the output)")
    p.add_argument("--cores", type=int, default=None)
    p.add_argument("--shard", default="0/1", help="i/n: run every n-th geometry starting at i")
    p.add_argument("--check", action="store_true", help="run only the check geometries")
    p.add_argument("--collect", action="store_true", help="merge shard results into band tables")
    args = p.parse_args()
    if args.collect:
        collect(args)
    elif args.model:
        solve_all(args)
    else:
        p.error("give --model to run COMSOL, or --collect to merge results")


if __name__ == "__main__":
    main()
