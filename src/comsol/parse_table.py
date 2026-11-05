"""Parse a COMSOL Global Evaluation table export (GUI route) into band tables.

Run from the Project folder:
  python src/comsol/parse_table.py data/comsol/new/global_eval.txt [--corner 1]

Expected input: Results > Derived Values > Global Evaluation of emw.freq (unit GHz) on the
parametric-sweep dataset, exported as a table (.txt or .csv). COMSOL writes '%' header lines,
the last of which holds the column names, e.g. "L1 (m)  L2 (m)  k (1)  emw.freq (GHz)".
Column layouts differ between versions, so the parser looks columns up by name (override with
--l1-col/--l2-col/--k-col/--freq-col) and prints what it picked. Send the first lines of a real
export if it guesses wrong.

Writes data/comsol/new/bands_new.csv and band_edges_new.csv (same columns as process_bands.py).
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from bands_to_edges import compare_with_old, to_bands, to_edges
from make_samples import OUT_DIR

LENGTH_TO_UM = {"m": 1e6, "mm": 1e3, "um": 1.0, "µm": 1.0}
FREQ_TO_GHZ = {"Hz": 1e-9, "kHz": 1e-6, "MHz": 1e-3, "GHz": 1.0, "THz": 1e3}


def read_table(path):
    with open(path, encoding="utf8", errors="replace") as f:
        lines = f.read().splitlines()
    header = [l for l in lines if l.startswith("%")]
    data = [l for l in lines if l.strip() and not l.startswith("%")]
    if not header or not data:
        sys.exit("no '%' header line or no data rows found")
    sep = "," if "," in data[0] else r"\t|\s{2,}|\s+"
    names = [c.strip() for c in re.split(sep if sep == "," else r"\t|\s{2,}", header[-1].lstrip("% ").strip())]
    rows = [[c.strip() for c in re.split(sep, l.strip())] for l in data]
    width = max(len(r) for r in rows)
    if len(names) != width:
        sys.exit(f"header has {len(names)} names but rows have {width} values:\n{header[-1]}\n{data[0]}")
    return pd.DataFrame(rows, columns=names)


def to_real(col):
    """COMSOL writes complex eigenvalues as e.g. 4.96E11+7.58E4i; keep the real part."""
    s = col.astype(str).str.replace("i", "j", regex=False)
    return s.map(lambda v: complex(v.replace(" ", "")).real).astype(float)


def unit_of(name):
    m = re.search(r"\(([^)]*)\)", name)
    return m.group(1).strip() if m else None


def pick(df, pattern, given, what):
    if given:
        return given
    hits = [c for c in df.columns if re.search(pattern, c)]
    if not hits:
        sys.exit(f"no column for {what} (pattern {pattern!r}) in {list(df.columns)}; use the --*-col options")
    return hits[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("table")
    p.add_argument("--corner", type=int, default=1, help="1 rounded, 0 sharp (label in the output)")
    p.add_argument("--l1-col")
    p.add_argument("--l2-col")
    p.add_argument("--k-col")
    p.add_argument("--freq-col")
    args = p.parse_args()

    df = read_table(args.table)
    cols = {
        "L1": pick(df, r"^L1\b", args.l1_col, "L1"),
        "L2": pick(df, r"^L2\b", args.l2_col, "L2"),
        "k": pick(df, r"^k\b", args.k_col, "k"),
        "freq": pick(df, r"emw\.freq|^freq|[Ee]igenfrequency", args.freq_col, "frequency"),
    }
    print("using columns:", cols)

    long = pd.DataFrame({key: to_real(df[c]) for key, c in cols.items()})
    for key in ("L1", "L2"):
        u = unit_of(cols[key])
        scale = LENGTH_TO_UM.get(u) if u else (1e6 if long[key].max() < 1e-2 else 1.0)
        if scale is None:
            sys.exit(f"unknown length unit {u!r} in {cols[key]!r}")
        long[key] *= scale
    u = unit_of(cols["freq"])
    fscale = FREQ_TO_GHZ.get(u) if u else (1e-9 if long["freq"].max() > 1e6 else 1.0)
    if fscale is None:
        sys.exit(f"unknown frequency unit {u!r} in {cols['freq']!r}")
    long["freq"] *= fscale
    long = long.rename(columns={"L1": "L1_um", "L2": "L2_um", "freq": "freq_GHz"})
    print(f"{len(long)} rows; L1 {long.L1_um.min():.2f}-{long.L1_um.max():.2f} um, "
          f"k values {sorted(long.k.round(4).unique())[:6]}, freq {long.freq_GHz.min():.1f}-"
          f"{long.freq_GHz.max():.1f} GHz")

    bands = to_bands(long, args.corner)
    edges = to_edges(bands)
    os.makedirs(OUT_DIR, exist_ok=True)
    bands.to_csv(os.path.join(OUT_DIR, "bands_new.csv"), index=False)
    edges.to_csv(os.path.join(OUT_DIR, "band_edges_new.csv"), index=False)
    print(f"{len(edges)} geometries, {int((edges.gap > 0.1).sum())} with gap > 0.1 GHz "
          f"-> {OUT_DIR}/band_edges_new.csv")
    compare_with_old(edges)


if __name__ == "__main__":
    main()
