"""Shared reduction: (geometry, k, eigenfrequencies) rows -> bands and band edges.

Used by run_mph.py and parse_table.py so both routes produce the same files as
src/utils/process_bands.py (same columns, L in um, f in GHz).
"""
import pandas as pd


def to_bands(long, corner):
    """long: columns L1_um, L2_um, k, freq_GHz (one row per eigenfrequency).

    Bands 1 and 2 are the two lowest eigenfrequencies at each (geometry, k), as in the
    original export, which returned exactly two per k.
    """
    long = long.round({"L1_um": 3, "L2_um": 3, "k": 4, "freq_GHz": 6}).drop_duplicates()
    g = long.groupby(["L1_um", "L2_um", "k"])["freq_GHz"]
    n = g.size()
    if (n < 2).any():
        bad = n[n < 2].reset_index().head().to_string(index=False)
        raise ValueError(f"fewer than 2 eigenfrequencies for some (geometry, k):\n{bad}")
    two = g.nsmallest(2).reset_index(level=-1, drop=True).reset_index()
    bands = two.groupby(["L1_um", "L2_um", "k"])["freq_GHz"].agg(f1_GHz="min", f2_GHz="max").reset_index()
    bands.insert(0, "corner", corner)
    return bands


def to_edges(bands):
    """f_lower = max over k of band 1, f_upper = min over k of band 2 (as process_bands.py)."""
    edges = bands.groupby(["corner", "L1_um", "L2_um"]).agg(
        f_lower=("f1_GHz", "max"), f_upper=("f2_GHz", "min"), n_k=("k", "size")).reset_index()
    edges["gap"] = edges.f_upper - edges.f_lower
    return edges


def compare_with_old(edges, old_path="data/comsol/band_edges.csv", tol_um=0.05):
    """Print new vs old band edges for geometries that sit on the old 8x8 grid."""
    old = pd.read_csv(old_path)
    rows = []
    for _, r in edges.iterrows():
        m = old[(old.corner == r.corner) & ((old.L1_um - r.L1_um).abs() < tol_um)
                & ((old.L2_um - r.L2_um).abs() < tol_um)]
        if len(m):
            rows.append((r.L1_um, r.L2_um, r.f_lower - m.f_lower.iloc[0], r.f_upper - m.f_upper.iloc[0]))
    if not rows:
        print("no geometries on the old grid to compare with")
        return
    d = pd.DataFrame(rows, columns=["L1_um", "L2_um", "d_f_lower_GHz", "d_f_upper_GHz"])
    print("new minus old band edges (should be ~0 if the model and mesh are unchanged):")
    print(d.round(4).to_string(index=False))
