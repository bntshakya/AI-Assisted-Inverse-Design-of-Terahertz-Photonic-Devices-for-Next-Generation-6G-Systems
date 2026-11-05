"""Reduce the COMSOL per-mesh-node exports to small band tables (both corner types).

Run from the Project folder:  python src/utils/process_bands.py

Writes to data/comsol/:
  bands.csv       corner, L1_um, L2_um, k, f1_GHz, f2_GHz        (one row per geometry and k)
  band_edges.csv  corner, L1_um, L2_um, f_lower, f_upper, gap    (one row per geometry)
corner = 1 for rounded corners (r = 2L/(3*sqrt(3))), 0 for sharp corners (r = 0).
L is stored in metres in the exports; it is converted to um here, once.
"""
import os

import pandas as pd

DATA_DIR = "data/comsol"
SOURCES = {
    1: "parameter_sweep_values (Varied L1, L2 and r = 2Lby 3sqrt(3)).csv",
    0: "r1=r2=0 ; varied L1 & L2 (Parametric sweep).csv",
}
MIN_GAP_GHZ = 0.1  # below this the "gap" is numerical noise (e.g. sharp corners with L1 == L2)


def count_header_lines(path):
    """COMSOL comment lines start with '%'; the two exports have 9 and 2 of them."""
    n = 0
    with open(path, encoding="utf8", errors="replace") as f:
        for line in f:
            if not line.startswith("%"):
                return n
            n += 1
    return n


def read_states(path, chunksize=2_000_000):
    """Unique (L1, L2, k, freq) states; the X, Y node columns carry no information."""
    skip = count_header_lines(path)
    parts = []
    for i, chunk in enumerate(pd.read_csv(path, skiprows=skip, header=None, usecols=[2, 3, 4, 6],
                                          names=["L1", "L2", "k", "freq"], chunksize=chunksize)):
        chunk["freq"] = pd.to_numeric(chunk["freq"], errors="coerce")
        chunk = chunk.dropna()
        # the same eigenfrequency is written with last-digit float noise at different nodes
        chunk = chunk.round({"L1": 9, "L2": 9, "k": 4, "freq": 6}).drop_duplicates()
        parts.append(chunk)
        print(f"  chunk {i + 1}", flush=True)
    return pd.concat(parts).drop_duplicates()


def to_bands(states, corner):
    g = states.groupby(["L1", "L2", "k"])["freq"]
    n = g.size()
    if (n != 2).any():
        raise ValueError(f"expected 2 eigenfrequencies per (L1, L2, k), got {n.value_counts().to_dict()}")
    bands = g.agg(f1_GHz="min", f2_GHz="max").reset_index()
    bands["L1_um"] = (bands.pop("L1") * 1e6).round(2)
    bands["L2_um"] = (bands.pop("L2") * 1e6).round(2)
    bands.insert(0, "corner", corner)
    return bands[["corner", "L1_um", "L2_um", "k", "f1_GHz", "f2_GHz"]]


def to_edges(bands):
    edges = bands.groupby(["corner", "L1_um", "L2_um"]).agg(
        f_lower=("f1_GHz", "max"), f_upper=("f2_GHz", "min")).reset_index()
    edges["gap"] = edges.f_upper - edges.f_lower
    return edges


def main():
    all_bands = []
    for corner, name in SOURCES.items():
        path = os.path.join(DATA_DIR, name)
        print(f"{name}: {count_header_lines(path)} header lines", flush=True)
        all_bands.append(to_bands(read_states(path), corner))
    bands = pd.concat(all_bands, ignore_index=True)
    edges = to_edges(bands)
    bands.to_csv(os.path.join(DATA_DIR, "bands.csv"), index=False)
    edges.to_csv(os.path.join(DATA_DIR, "band_edges.csv"), index=False)

    for corner, e in edges.groupby("corner"):
        ok = e[e.gap > MIN_GAP_GHZ]
        print(f"corner={corner}: {len(e)} geometries, {len(ok)} with gap > {MIN_GAP_GHZ} GHz, "
              f"gap {ok.gap.min():.2f}-{ok.gap.max():.2f} GHz")
    print(f"wrote bands.csv ({len(bands)} rows) and band_edges.csv ({len(edges)} rows)")


if __name__ == "__main__":
    main()
