"""Architecture test (ablation): why 3 hidden layers of 256, SiLU, 5 networks, 400 epochs, and the inverse settings.

Uses the SAME data split as the main run (run_optionb.py, seed 0). Every network is trained on the TRAIN split and
scored on the VALIDATION split only; the TEST split behind the reported numbers is never touched.

Run from the Project folder:  python src/shapes/ablation_architecture.py      (about 1.5-2 h on a laptop GPU)
Writes results/optionb/ablation.json (updated after every part) and results/optionb/figs/ablation.png.

Parts
  A  forward network: depth 1-5 x width 64-512 (single network, 3 seeds)
  B  forward network: activation function at 3 x 256
  C  forward network: ensemble size 1-10 (and whether the spread tracks the error)
  D  forward network: training epochs
  E  tandem inverse network: size, and the fabrication-penalty weight
  F  CVAE: latent size and beta
Inverse parts (E, F) are scored with the forward ensemble (surrogate error) and the exact fabrication rules on
validation targets; they are NOT re-simulated with PWE (the main run does that for the final numbers).
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_pwe  # noqa: E402
from geometry import FEATURE_NAMES, check_cell  # noqa: E402
import models as M  # noqa: E402
from run_optionb import P_IDX, select, split  # noqa: E402

OUT = "results/optionb/ablation.json"
FIG = "results/optionb/figs/ablation.png"
DEV = M.DEV
SEEDS = (0, 1, 2)
R = {"note": "trained on the train split, scored on the validation split (seed-0 split of run_optionb.py); "
             "MAE in GHz, mean over seeds; inverse parts scored by the forward ensemble + exact fabrication rules"}


def save():
    with open(OUT, "w") as f:
        json.dump(R, f, indent=1)


def mlp(i, o, h, depth, act=nn.SiLU):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, h), act()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


def n_params(net):
    return int(sum(p.numel() for p in net.parameters()))


class Data:
    def __init__(self):
        pwe = load_pwe()
        self.pwe = pwe
        X, Y = pwe[FEATURE_NAMES].to_numpy(float), pwe[["f_lower", "f_upper"]].to_numpy(float)
        self.tr, self.va, self.te = split(len(pwe), seed=0)
        self.X, self.Y = X, Y
        self.xm, self.xs = X[self.tr].mean(0), X[self.tr].std(0) + 1e-3
        self.ym, self.ys = Y[self.tr].mean(0), Y[self.tr].std(0)
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
        self.Xtr, self.Ytr = t((X[self.tr] - self.xm) / self.xs), t((Y[self.tr] - self.ym) / self.ys)
        self.Xva = t((X[self.va] - self.xm) / self.xs)


def train_forward(D, depth=3, width=256, act=nn.SiLU, epochs=400, seed=0):
    torch.manual_seed(seed)
    net = mlp(D.Xtr.shape[1], 2, width, depth, act).to(DEV)
    t0 = time.time()
    M._train(net, net.parameters(), lambda idx: ((net(D.Xtr[idx]) - D.Ytr[idx]) ** 2).mean(),
             len(D.Xtr), epochs, 128, 2e-3, seed)
    net.eval()
    with torch.no_grad():
        pred = net(D.Xva).cpu().numpy() * D.ys + D.ym
    return net, pred, time.time() - t0


def mae(pred, D):
    e = np.abs(pred - D.Y[D.va]).mean(0)
    return [round(float(e[0]), 4), round(float(e[1]), 4)]


def part_a(D):
    rows = []
    for depth in (1, 2, 3, 4, 5):
        for width in (64, 128, 256, 512):
            m, secs = [], []
            for s in SEEDS:
                net, pred, t = train_forward(D, depth, width, seed=s)
                m.append(mae(pred, D))
                secs.append(t)
            m = np.array(m)
            row = {"depth": depth, "width": width, "params": n_params(net), "mae": m.mean(0).round(4).tolist(),
                   "mae_mean": round(float(m.mean()), 4), "mae_seed_std": round(float(m.mean(1).std()), 4),
                   "train_s": round(float(np.mean(secs)), 1)}
            rows.append(row)
            print("  A", row, flush=True)
            R["A_depth_width"] = rows
            save()


def part_b(D):
    rows = []
    for name, act in (("SiLU", nn.SiLU), ("ReLU", nn.ReLU), ("GELU", nn.GELU), ("Tanh", nn.Tanh)):
        m = np.array([mae(train_forward(D, act=act, seed=s)[1], D) for s in SEEDS])
        row = {"activation": name, "mae": m.mean(0).round(4).tolist(), "mae_mean": round(float(m.mean()), 4),
               "mae_seed_std": round(float(m.mean(1).std()), 4)}
        rows.append(row)
        print("  B", row, flush=True)
    R["B_activation"] = rows
    save()


def part_c(D):
    preds = [train_forward(D, seed=10 + s)[1] for s in range(10)]
    P = np.stack(preds)
    rows = []
    for k in (1, 2, 3, 5, 10):
        mu = P[:k].mean(0)
        row = {"members": k, "mae": mae(mu, D), "mae_mean": round(float(np.mean(mae(mu, D))), 4)}
        if k >= 3:
            sd = P[:k].std(0).mean(1)
            err = np.abs(mu - D.Y[D.va]).mean(1)
            row["corr_spread_error"] = round(float(np.corrcoef(sd, err)[0, 1]), 3)
            q = np.quantile(sd, 0.9)
            row["err_top10pct_spread"] = round(float(err[sd >= q].mean()), 3)
            row["err_rest"] = round(float(err[sd < q].mean()), 3)
        rows.append(row)
        print("  C", row, flush=True)
    R["C_ensemble"] = rows
    save()


def part_d(D):
    rows = []
    for ep in (100, 200, 400, 800):
        out = [train_forward(D, epochs=ep, seed=s) for s in SEEDS]
        m = np.array([mae(o[1], D) for o in out])
        row = {"epochs": ep, "mae": m.mean(0).round(4).tolist(), "mae_mean": round(float(m.mean()), 4),
               "train_s": round(float(np.mean([o[2] for o in out])), 1)}
        rows.append(row)
        print("  D", row, flush=True)
    R["D_epochs"] = rows
    save()


# ------------------------------------------------------------------ inverse parts
def val_targets(D, n=200):
    is4 = (D.pwe.dataset == "pwe_fourier").to_numpy()
    gap = D.pwe.gap.to_numpy()
    idx = D.va[is4[D.va] & (gap[D.va] > 2.0)]
    idx = np.random.default_rng(5).choice(idx, size=min(n, len(idx)), replace=False)
    return D.Y[idx]


def score_designs(fwd, cands, T):
    """cands (n_t, n_s, 10): best valid distinct design per target (surrogate-ranked) -> surrogate worst-edge error."""
    picks = select(cands, T, fwd, k=1)
    errs = [float(np.abs(g[0][1] - t).max()) if g else np.nan for g, t in zip(picks, T)]
    errs = np.array(errs)
    found = ~np.isnan(errs)
    return {"valid_found": round(float(found.mean()), 3),
            "worst_edge_mean": round(float(np.nanmean(errs)), 3),
            "within_1GHz": round(float((errs[found] < 1).sum() / len(T)), 3)}


class TandemSized(M.Tandem):
    def __init__(self, depth=3, width=256, **kw):
        super().__init__(**kw)
        self.depth, self.width = depth, width

    def fit(self, fwd, T, seed=0):
        orig = M.mlp
        M.mlp = lambda i, o, h=256, depth=3: mlp(i, o, self.width, self.depth)
        try:
            return super().fit(fwd, T, seed)
        finally:
            M.mlp = orig


def part_e(D, fwd, T):
    gap = D.pwe.gap.to_numpy()
    tr_g = D.tr[gap[D.tr] > 0.5]
    rows = []
    for depth, width, lam in ((2, 128, 1e-3), (3, 256, 1e-3), (4, 256, 1e-3), (3, 512, 1e-3), (3, 256, 0.0), (3, 256, 1e-2)):
        t0 = time.time()
        tan = TandemSized(depth=depth, width=width, lam=lam).fit(fwd, D.Y[tr_g], seed=0)
        cands = tan.propose(T)
        raw_valid = float(np.mean([check_cell(*M.p_to_holes(p))[0] for p in cands[:, 0]]))
        pred = fwd.predict(M.p_to_feat(cands[:, 0]))[0]
        err = np.abs(pred - T).max(1)
        row = {"depth": depth, "width": width, "penalty_weight": lam, "valid_share": round(raw_valid, 3),
               "worst_edge_mean": round(float(err.mean()), 3), "within_1GHz": round(float((err < 1).mean()), 3),
               "within_1GHz_and_valid": round(float(((err < 1) & np.array([check_cell(*M.p_to_holes(p))[0]
                                                                            for p in cands[:, 0]])).mean()), 3),
               "train_s": round(time.time() - t0, 1)}
        rows.append(row)
        print("  E", row, flush=True)
        R["E_tandem"] = rows
        save()


def part_f(D, fwd, T):
    is4 = (D.pwe.dataset == "pwe_fourier").to_numpy()
    tr4 = D.tr[is4[D.tr]]
    U = M.p_to_u(D.X[tr4][:, P_IDX])
    rows = []
    for latent, beta in ((2, 0.5), (4, 0.5), (8, 0.5), (4, 0.1), (4, 1.0)):
        t0 = time.time()
        cv = M.CVAE(latent=latent, beta=beta).fit(U, D.Y[tr4], seed=0)
        cands = cv.propose(T[:100], 64, seed=0)
        n_t, n_s, _ = cands.shape
        pred = fwd.predict(M.p_to_feat(cands.reshape(-1, 10)))[0].reshape(n_t, n_s, 2)
        err = np.abs(pred - T[:100, None]).max(-1)
        picks = select(cands, T[:100], fwd, k=8)
        distinct = [sum(1 for p, pr in g if np.abs(pr - t).max() < 1) for g, t in zip(picks, T[:100])]
        row = {"latent": latent, "beta": beta, "samples_within_1GHz": round(float((err < 1).mean()), 3),
               "distinct_within_1GHz_of_8": round(float(np.mean(distinct)), 2),
               "best_worst_edge_mean": round(float(err.min(1).mean()), 3), "train_s": round(time.time() - t0, 1)}
        rows.append(row)
        print("  F", row, flush=True)
        R["F_cvae"] = rows
        save()


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    A = R.get("A_depth_width", [])
    for w, c in zip((64, 128, 256, 512), ("#9aa5ab", "#7fb7be", "#0d7a84", "#0b3c49")):
        rows = [r for r in A if r["width"] == w]
        if rows:
            ax[0].errorbar([r["depth"] for r in rows], [r["mae_mean"] for r in rows],
                           yerr=[r["mae_seed_std"] for r in rows], marker="o", color=c, label=f"width {w}", capsize=3)
    used = [r for r in A if r["depth"] == 3 and r["width"] == 256]
    if used:
        ax[0].plot(3, used[0]["mae_mean"], "o", ms=15, mfc="none", mec="#c8553d", mew=2)
        ax[0].annotate("used", (3, used[0]["mae_mean"]), xytext=(3.25, used[0]["mae_mean"] + 0.12), color="#c8553d",
                       fontsize=9, arrowprops=dict(arrowstyle="-", color="#c8553d"))
    ax[0].set_xlabel("hidden layers")
    ax[0].set_ylabel("validation MAE, mean of both edges (GHz)")
    ax[0].set_title("(a) depth and width")
    ax[0].set_xticks([1, 2, 3, 4, 5])
    ax[0].legend(fontsize=8)
    Cc = R.get("C_ensemble", [])
    if Cc:
        ax[1].plot([r["members"] for r in Cc], [r["mae_mean"] for r in Cc], "o-", color="#0d7a84")
        ax[1].set_xlabel("networks in the ensemble")
        ax[1].set_ylabel("validation MAE (GHz)")
        ax[1].set_title("(b) ensemble size")
        ax[1].set_xticks([1, 2, 3, 5, 10])
        u5 = [r for r in Cc if r["members"] == 5]
        if u5:
            ax[1].plot(5, u5[0]["mae_mean"], "o", ms=15, mfc="none", mec="#c8553d", mew=2)
            ax[1].annotate("used", (5, u5[0]["mae_mean"]), xytext=(6.2, u5[0]["mae_mean"] + 0.006), color="#c8553d", fontsize=9,
                           arrowprops=dict(arrowstyle="-", color="#c8553d"))
    Dd = R.get("D_epochs", [])
    if Dd:
        ax[2].plot([r["epochs"] for r in Dd], [r["mae_mean"] for r in Dd], "o-", color="#e89b2d")
        ax[2].set_xscale("log", base=2)
        ax[2].set_xticks([100, 200, 400, 800])
        ax[2].set_xticklabels(["100", "200", "400", "800"])
        u4 = [r for r in Dd if r["epochs"] == 400]
        if u4:
            ax[2].plot(400, u4[0]["mae_mean"], "o", ms=15, mfc="none", mec="#c8553d", mew=2)
            ax[2].annotate("used", (400, u4[0]["mae_mean"]), xytext=(470, u4[0]["mae_mean"] + 0.03), color="#c8553d", fontsize=9,
                           arrowprops=dict(arrowstyle="-", color="#c8553d"))
        ax[2].set_xlabel("training epochs")
        ax[2].set_ylabel("validation MAE (GHz)")
        ax[2].set_title("(c) training length")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG, dpi=170)
    print("wrote", FIG)


def main():
    t0 = time.time()
    D = Data()
    R["n_train"], R["n_val"] = int(len(D.tr)), int(len(D.va))
    print(f"train {len(D.tr)}, validation {len(D.va)} cells; device {DEV}", flush=True)
    part_a(D); print(f"A done {time.time() - t0:.0f} s", flush=True)
    part_b(D); print(f"B done {time.time() - t0:.0f} s", flush=True)
    part_c(D); print(f"C done {time.time() - t0:.0f} s", flush=True)
    part_d(D); print(f"D done {time.time() - t0:.0f} s", flush=True)
    figure()
    fwd = M.Ensemble(n_members=5, epochs=400).fit(D.X[D.tr], D.Y[D.tr], seed=0)
    T = val_targets(D)
    R["inverse_val_targets"] = int(len(T))
    part_e(D, fwd, T); print(f"E done {time.time() - t0:.0f} s", flush=True)
    part_f(D, fwd, T); print(f"F done {time.time() - t0:.0f} s", flush=True)
    R["minutes"] = round((time.time() - t0) / 60, 1)
    save()
    print(f"all done in {R['minutes']} min -> {OUT}")


if __name__ == "__main__":
    main()
