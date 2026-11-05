"""Train and save every network the design tools need (same data, settings and seed as run_optionb.py):
  - forward ensemble on the PWE cells (5 networks)
  - tandem inverse network and conditional VAE
  - COMSOL correction: Gaussian process on (COMSOL - network) over all COMSOL cells (kernel fitted on a subset of
    700 cells, then fixed and refitted on all cells; saved in a compact mean-only form)
  - COMSOL-trained forward ensemble (5 networks) on all COMSOL cells

After this, design.py and the Streamlit app work in seconds. Rerun it whenever the data change (update_all.py does).
Run from the Project folder:  python src/shapes/save_models.py      (about 10-12 minutes on the laptop GPU)
Writes models/optionb/bundle.pt and models/optionb/correction.pkl
"""
import datetime
import os
import sys
import time
import warnings

import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_comsol, load_pwe  # noqa: E402
from designer import LightGP, save_bundle  # noqa: E402
from geometry import FEATURE_NAMES  # noqa: E402
from models import CVAE, Ensemble, Tandem, p_to_u  # noqa: E402
from run_optionb import P_IDX, split  # noqa: E402


def light_gp(X, y, n_sub=700, seed=0):
    xm, xs = X.mean(0), X.std(0) + 1e-3
    Z = (X - xm) / xs
    idx = np.random.default_rng(seed).choice(len(X), min(n_sub, len(X)), replace=False)
    k = ConstantKernel(1.0) * RBF(np.ones(X.shape[1]), (1e-2, 1e3)) + WhiteKernel(1e-3, (1e-8, 1e1))
    kern = GaussianProcessRegressor(k, normalize_y=True, random_state=0).fit(Z[idx], y[idx]).kernel_
    gpr = GaussianProcessRegressor(kern, optimizer=None, normalize_y=True).fit(Z, y)
    return LightGP(gpr, xm, xs)


def main():
    t0 = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    pwe, com = load_pwe(), load_comsol()
    X, Y = pwe[FEATURE_NAMES].to_numpy(float), pwe[["f_lower", "f_upper"]].to_numpy(float)
    tr, va, te = split(len(pwe), seed=0)
    trv = np.r_[tr, va]
    fwd = Ensemble(n_members=5, epochs=400).fit(X[trv], Y[trv], seed=0)
    mae = np.abs(fwd.predict(X[te])[0] - Y[te]).mean(0)
    print(f"forward ensemble: test MAE {mae.round(3)} GHz ({time.time() - t0:.0f} s)", flush=True)
    gap = pwe.gap.to_numpy()
    tandem = Tandem().fit(fwd, Y[trv[gap[trv] > 0.5]], seed=0)
    is4 = (pwe.dataset == "pwe_fourier").to_numpy()
    cvae = CVAE().fit(p_to_u(X[trv[is4[trv]]][:, P_IDX]), Y[trv[is4[trv]]], seed=0)
    print(f"tandem + CVAE trained ({time.time() - t0:.0f} s)", flush=True)

    Xc, Yc = com[FEATURE_NAMES].to_numpy(float), com[["f_lower", "f_upper"]].to_numpy(float)
    resid = Yc - fwd.predict(Xc)[0]
    gps = [light_gp(Xc, resid[:, j]) for j in range(2)]
    print(f"COMSOL correction fitted on {len(com)} cells ({time.time() - t0:.0f} s)", flush=True)
    spe = int(np.ceil(len(Xc) / 128))
    fwd_c = Ensemble(n_members=5, epochs=int(np.clip(8000 / spe, 150, 1500))).fit(Xc, Yc, seed=0)
    print(f"COMSOL-trained ensemble fitted ({time.time() - t0:.0f} s)", flush=True)

    info = {"trained": datetime.datetime.now().isoformat(timespec="minutes"), "n_pwe_train": int(len(trv)),
            "forward_test_mae_GHz": mae.round(3).tolist(), "n_comsol": int(len(com)),
            "comsol_sets": com.dataset.value_counts().to_dict()}
    save_bundle(fwd, tandem, cvae, {"gps": gps}, info, fwd_comsol=fwd_c)
    print(f"saved models/optionb/ in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
