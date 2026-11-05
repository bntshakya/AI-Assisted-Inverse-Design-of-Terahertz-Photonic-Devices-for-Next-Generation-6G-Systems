"""One table for every dataset, whatever the hole shapes and whichever simulator produced them.

Each row is one unit cell: famA, famB, paramsA, paramsB (JSON), the 18 C3-Fourier features
(geometry.FEATURE_NAMES), f_lower, f_upper, gap (GHz), and
  source   "comsol" (high fidelity) or "pwe" (fast plane-wave solver, low fidelity)
  dataset  e.g. comsol_rounded, comsol_sharp, comsol_circle, pwe_fourier, pwe_library

COMSOL files read:
  data/comsol/band_edges.csv     the two original sweeps (rounded and sharp triangles)
  data/comsol/extra_*.csv        anything added with add_comsol_dataset.py (e.g. the circle cells)
PWE files read: data/pwe/fourier.csv, data/pwe/library.csv (gen_dataset.py).
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from geometry import FEATURE_NAMES, cell_features

COMSOL_EDGES = "data/comsol/band_edges.csv"
COMSOL_EXTRA = "data/comsol/extra_*.csv"
PWE_FILES = {"pwe_fourier": "data/pwe/fourier.csv", "pwe_library": "data/pwe/library.csv"}
MIN_GAP_GHZ = 0.1
FAMILY = {1: "rounded", 0: "sharp"}


def holes(row):
    """(holeA, holeB) as (family, params) tuples from a table row."""
    return ((row["famA"], json.loads(row["paramsA"])), (row["famB"], json.loads(row["paramsB"])))


def _with_features(df):
    feats = np.array([cell_features(*holes(r)) for _, r in df.iterrows()])
    for j, name in enumerate(FEATURE_NAMES):
        df[name] = feats[:, j]
    return df


def load_comsol():
    e = pd.read_csv(COMSOL_EDGES)
    df = pd.DataFrame({
        "id": [f"comsol_{FAMILY[c]}_{a:.2f}_{b:.2f}" for c, a, b in zip(e.corner, e.L1_um, e.L2_um)],
        "dataset": ["comsol_" + FAMILY[c] for c in e.corner],
        "famA": [FAMILY[c] for c in e.corner], "famB": [FAMILY[c] for c in e.corner],
        "paramsA": [json.dumps({"L": float(v)}) for v in e.L1_um],
        "paramsB": [json.dumps({"L": float(v)}) for v in e.L2_um],
        "f_lower": e.f_lower, "f_upper": e.f_upper, "gap": e.gap})
    parts = [df]
    for path in sorted(glob.glob(COMSOL_EXTRA)):
        x = pd.read_csv(path)
        parts.append(x[["id", "dataset", "famA", "famB", "paramsA", "paramsB", "f_lower", "f_upper", "gap"]])
    df = pd.concat(parts, ignore_index=True)
    df["source"] = "comsol"
    return _with_features(df)


def load_pwe(names=("pwe_fourier", "pwe_library")):
    parts = []
    for name in names:
        if os.path.exists(PWE_FILES[name]):
            x = pd.read_csv(PWE_FILES[name])
            x["dataset"], x["source"] = name, "pwe"
            parts.append(x)
    if not parts:
        raise FileNotFoundError("no PWE data: run src/shapes/gen_dataset.py first")
    return pd.concat(parts, ignore_index=True)


def xy(df, gapped=True):
    """Feature matrix (n, 18) in um and targets (n, 2) = (f_lower, f_upper) in GHz."""
    if gapped:
        df = df[df.gap > MIN_GAP_GHZ]
    return df[FEATURE_NAMES].to_numpy(float), df[["f_lower", "f_upper"]].to_numpy(float), df
