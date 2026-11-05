import numpy as np
import pandas as pd

EDGES_CSV = "data/comsol/band_edges.csv"  # written by src/utils/process_bands.py
MIN_GAP_GHZ = 0.1
CORNER = {"sharp": 0, "rounded": 1}


def load_corner(corner, path=EDGES_CSV):
    """Gapped geometries of one corner type: x = (L1, L2) in um, y = (f_lower, f_upper) in GHz."""
    e = pd.read_csv(path)
    e = e[(e.corner == CORNER[corner]) & (e.gap > MIN_GAP_GHZ)]
    return e[["L1_um", "L2_um"]].to_numpy(float), e[["f_lower", "f_upper"]].to_numpy(float)


def with_corner(x, corner):
    """Append the corner-type input column (1 = rounded, 0 = sharp)."""
    return np.c_[x, np.full(len(x), CORNER[corner], float)]


def random_split(n, frac_train, seed):
    perm = np.random.default_rng(seed).permutation(n)
    k = int(round(frac_train * n))
    return perm[:k], perm[k:]


def gap_of(y):
    return (y[:, 1] - y[:, 0])[:, None]
