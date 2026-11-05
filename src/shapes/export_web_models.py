"""Export small trained models + demo data for the interactive learning site.

Run from the Project folder:  python src/shapes/export_web_models.py
Writes learn/site/web_data.json (weights as base64 float32, plus precomputed demo data).
Trains the same models as run_optionb.py (seed 0), but with a 2-member forward ensemble to keep the file small.
"""
import base64
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_pwe, holes  # noqa: E402
from geometry import FEATURE_NAMES, radius, check_cell  # noqa: E402
from models import CVAE, Ensemble, Tandem, p_to_u, R0, A3MAX, A6MAX  # noqa: E402
from pwe import Solver  # noqa: E402
from gen_dataset import eps_from_validation  # noqa: E402
from run_optionb import split, P_IDX  # noqa: E402

OUT = "learn/site/web_data.json"
TH = np.linspace(0, 2 * np.pi, 120, endpoint=False)


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()


def net_layers(net):
    """nn.Sequential of Linear/SiLU -> list of {W: base64 (out x in), b, out, in}."""
    out = []
    for m in net:
        if isinstance(m, torch.nn.Linear):
            W = m.weight.detach().cpu().numpy()
            out.append({"W": b64(W), "b": b64(m.bias.detach().cpu().numpy()), "out": W.shape[0], "in": W.shape[1]})
    return out


def rr(fam, par):
    return np.round(radius(fam, par, TH), 2).tolist()


def main():
    t0 = time.time()
    torch.manual_seed(0)
    pwe = load_pwe()
    X, Y = pwe[FEATURE_NAMES].to_numpy(float), pwe[["f_lower", "f_upper"]].to_numpy(float)
    tr, va, te = split(len(pwe), seed=0)
    trv = np.r_[tr, va]
    fwd = Ensemble(n_members=2, epochs=400).fit(X[trv], Y[trv], seed=0)
    mu = fwd.predict(X[te])[0]
    fwd_mae = np.abs(mu - Y[te]).mean(0).round(3).tolist()
    print("forward (2 members) test MAE", fwd_mae, flush=True)

    is4 = (pwe.dataset == "pwe_fourier").to_numpy()
    gap = pwe.gap.to_numpy()
    tandem = Tandem().fit(fwd, Y[tr[gap[tr] > 0.5]], seed=0)
    cvae = CVAE().fit(p_to_u(X[tr[is4[tr]]][:, P_IDX]), Y[tr[is4[tr]]], seed=0)
    print(f"models trained ({time.time() - t0:.0f} s)", flush=True)

    data = {"models": {
        "fwd": {"members": [net_layers(n) for n in fwd.nets], "xm": fwd.xm.tolist(), "xs": fwd.xs.tolist(),
                "ym": fwd.ym.tolist(), "ys": fwd.ys.tolist(), "test_mae": fwd_mae, "n_test": int(len(te))},
        "tandem": {"layers": net_layers(tandem.net), "tm": tandem.tm.tolist(), "ts": tandem.ts.tolist()},
        "cvae": {"layers": net_layers(cvae.dec), "tm": cvae.tm.tolist(), "ts": cvae.ts.tolist(), "latent": cvae.latent},
        "bounds": {"R0": list(R0), "A3MAX": A3MAX, "A6MAX": A6MAX},
        "feature_names": FEATURE_NAMES}}

    # band-gap explorer: circle pairs, full band diagram on the k-path
    s = Solver(eps_si=eps_from_validation())
    rs = list(range(30, 75, 5))
    ks = np.linspace(0, 3, 31)
    grid = {}
    for r1 in rs:
        for r2 in rs:
            hA, hB = ("circle", {"R": r1}), ("circle", {"R": r2})
            ok, info = check_cell(hA, hB)
            if not ok:
                continue
            b = s.bands(hA, hB, ks)
            grid[f"{r1}_{r2}"] = {"b1": np.round(b[:, 0], 2).tolist(), "b2": np.round(b[:, 1], 2).tolist()}
    data["circle_bands"] = {"r": rs, "k": ks.round(3).tolist(), "cells": grid}
    print(f"circle bands: {len(grid)} cells ({time.time() - t0:.0f} s)", flush=True)

    # PWE demo: pixel image and Fourier coefficients of one rounded-triangle cell
    hA, hB = ("rounded", {"L": 168.0}), ("rounded", {"L": 119.0})
    eps = s.eps_grid(hA, hB)
    img = (255 * (eps - eps.min()) / (eps.max() - eps.min())).astype(np.uint8)
    F = np.abs(np.fft.fft2(eps)) / eps.size
    F = np.fft.fftshift(F)
    c = F.shape[0] // 2
    Fc = np.log10(F[c - 12:c + 13, c - 12:c + 13] + 1e-4)
    bands = s.bands(hA, hB, ks)
    data["pwe_demo"] = {"n": int(img.shape[0]), "img": base64.b64encode(img.tobytes()).decode(),
                        "coef": np.round(Fc, 3).tolist(), "b1": np.round(bands[:, 0], 2).tolist(),
                        "b2": np.round(bands[:, 1], 2).tolist(), "n_pw": int(s.n_pw), "eps": float(s.eps_si)
                        if hasattr(s, "eps_si") else 8.914}

    # dataset examples (exact hole outlines as r(theta))
    ex = []
    for name, path, n in (("PWE Fourier", "data/pwe/fourier.csv", 8), ("PWE library", "data/pwe/library.csv", 8)):
        df = pd.read_csv(path).sample(n, random_state=3)
        for _, r in df.iterrows():
            pa, pb = json.loads(r.paramsA), json.loads(r.paramsB)
            ex.append({"set": name, "famA": r.famA, "famB": r.famB, "pA": pa, "pB": pb, "rA": rr(r.famA, pa),
                       "rB": rr(r.famB, pb), "f": [round(r.f_lower, 2), round(r.f_upper, 2)]})
    be = pd.read_csv("data/comsol/band_edges.csv")
    for corner, fam in ((1, "rounded"), (0, "sharp")):
        for _, r in be[be.corner == corner].sample(4, random_state=1).iterrows():
            pa, pb = {"L": round(r.L1_um, 2)}, {"L": round(r.L2_um, 2)}
            ex.append({"set": f"COMSOL {fam}", "famA": fam, "famB": fam, "pA": pa, "pB": pb, "rA": rr(fam, pa),
                       "rB": rr(fam, pb), "f": [round(r.f_lower, 2), round(r.f_upper, 2)]})
    for path, name in (("data/comsol/extra_circle.csv", "COMSOL circle"), ("data/comsol/extra_fourier.csv", "COMSOL freeform")):
        df = pd.read_csv(path).sample(4, random_state=2)
        for _, r in df.iterrows():
            pa, pb = json.loads(r.paramsA), json.loads(r.paramsB)
            ex.append({"set": name, "famA": r.famA, "famB": r.famB, "pA": pa, "pB": pb, "rA": rr(r.famA, pa),
                       "rB": rr(r.famB, pb), "f": [round(r.f_lower, 2), round(r.f_upper, 2)]})
    data["examples"] = ex

    # verified design gallery for 300/330 GHz
    sc = pd.read_csv("data/pwe/showcase_designs.csv") if os.path.exists("data/pwe/showcase_designs.csv") else None
    res = json.load(open("results/optionb/results.json"))
    gal = []
    for i, d in enumerate(res["showcase"]["designs"]):
        p = d["p"]
        k = ["r0", "c3", "s3", "c6", "s6"]
        gal.append({"p": [round(v, 2) for v in p], "pwe": d["pwe"], "rA": rr("fourier", dict(zip(k, p[:5]))),
                    "rB": rr("fourier", dict(zip(k, p[5:]))),
                    "comsol": [[299.10, 330.25], [299.11, 330.06]][i] if i < 2 else None})
    data["gallery"] = {"target": res["showcase"]["target"], "designs": gal}
    data["results"] = {"seed_summary": json.load(open("results/optionb/seed_summary.json")),
                       "comsol_vs_pwe": json.load(open("results/optionb/comsol_vs_pwe.json")),
                       "forward_test": res["forward_test"], "multifidelity": res["multifidelity"]}
    sys.path.insert(0, "learn")
    from patch_site_numbers import patch  # final numbers, verified designs, final COMSOL study
    data = patch(data)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
