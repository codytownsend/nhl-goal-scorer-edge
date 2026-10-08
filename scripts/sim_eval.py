"""Decisive test of variance-source (b): does conditioning on tonight's linemate
configuration beat the rate model, and does it beat it WHERE config deviates?

Baseline (rate model, physical form):   P0  = 1 - exp(-lambda0),  lambda0 = c*ixg_pg
Sim    (adds oracle linemate support):  Psim = 1 - exp(-lambda0 * clip(m))

c is fit on DEV seasons (so lambda0 is calibrated expected goals, no peeking).
We report overall top-1/top-3/logloss, then the deviation subset (player-games
where tonight's linemate support is most unlike the player's season norm) -- the
place the sim is supposed to differ. The clean signal test is whether log(m)
correlates with the rate model's residual (scored - P0).
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss, roc_auc_score

from nhlsit import simulate as S

DEV = [20202021, 20212022, 20222023, 20232024]
TEST = [20242025, 20252026]
M_CLIP = (0.5, 2.0)
SUPPORT_CACHE = "data/sim_support.parquet"


def topk_hit(df, col, k):
    hits = []
    for _, g in df.groupby("game_id"):
        g = g.sort_values(col, ascending=False).head(k)
        hits.append(int(g["scored"].max() >= 1))
    return float(np.mean(hits))


def fit_c(dev):
    x, y = dev["ixg_pg"].values, dev["scored"].values
    def nll(c):
        p = 1.0 - np.exp(-np.clip(c, 1e-6, None) * x)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return log_loss(y, p, labels=[0, 1])
    return minimize_scalar(nll, bounds=(0.1, 5.0), method="bounded").x


def block(name, d):
    y = d["scored"].values
    print(f"\n--- {name} (n={len(d)}, base {y.mean():.3f}) ---")
    for lbl, col in [("rate P0", "p0"), ("sim Psim", "psim")]:
        ll = log_loss(y, d[col].clip(1e-6, 1 - 1e-6), labels=[0, 1])
        au = roc_auc_score(y, d[col]) if y.min() != y.max() else float("nan")
        print(f"  {lbl:<10} logloss {ll:.4f}   auc {au:.3f}")


def main():
    if "--rebuild" in sys.argv or not _exists(SUPPORT_CACHE):
        print("building linemate support (oracle) for test seasons...", flush=True)
        sup = S.build_support(TEST, DEV)
        sup.to_parquet(SUPPORT_CACHE)
    else:
        sup = pd.read_parquet(SUPPORT_CACHE)
    print(f"support rows {len(sup)}, with m!=1: {(sup.m != 1.0).sum()}")

    df = pd.read_parquet("data/scorer_playergames.parquet")
    df["season"] = df["season"].astype(int)
    dev = df[df.season.isin(DEV)]
    test = df[df.season.isin(TEST)].merge(
        sup[["game_id", "player_id", "m", "support", "support_avg", "n_prior"]],
        on=["game_id", "player_id"], how="left")
    test["m"] = test["m"].fillna(1.0)

    c = fit_c(dev)
    print(f"\nfitted lambda0 = {c:.3f} * ixg_pg  (on dev)")
    lam0 = c * test["ixg_pg"].values
    mc = test["m"].clip(*M_CLIP).values
    test["p0"] = 1.0 - np.exp(-lam0)
    test["psim"] = 1.0 - np.exp(-lam0 * mc)

    # ---- overall per-game ranking + per-row logloss ----
    print("\n===== OVERALL (all test games) =====")
    for col, lbl in [("p0", "rate P0"), ("psim", "sim Psim")]:
        h1 = topk_hit(test, col, 1)
        h3 = topk_hit(test, col, 3)
        ll = log_loss(test["scored"], test[col].clip(1e-6, 1 - 1e-6), labels=[0, 1])
        print(f"  {lbl:<10} top1 {h1:.3f}  top3 {h3:.3f}  logloss {ll:.4f}")

    # ---- the clean signal test: does log(m) explain the rate model's miss? ----
    sig = test[(test.m != 1.0) & (test.m > 0)].copy()
    sig["logm"] = np.log(sig["m"].clip(*M_CLIP))
    sig["resid"] = sig["scored"] - sig["p0"]
    r = np.corrcoef(sig["logm"], sig["resid"])[0, 1]
    print(f"\n===== SIGNAL TEST (n={len(sig)} player-games with a real m) =====")
    print(f"corr( log(m), scored - P0 ) = {r:+.4f}   "
          f"(positive => better linemates tonight => scores more than rate says)")
    # bucketed dose-response
    sig["bucket"] = pd.qcut(sig["logm"], 5, duplicates="drop")
    tab = sig.groupby("bucket", observed=True).agg(
        n=("scored", "size"), mean_logm=("logm", "mean"),
        obs=("scored", "mean"), rate_pred=("p0", "mean"),
        sim_pred=("psim", "mean")).round(4)
    print(tab.to_string())

    # ---- deviation subset: top quartile |log m| ----
    thr = sig["logm"].abs().quantile(0.75)
    dev_sub = sig[sig["logm"].abs() >= thr]
    print(f"\n===== DEVIATION SUBSET (|log m| >= {thr:.3f}, top quartile) =====")
    block("deviation games", dev_sub)
    block("all-with-real-m", sig)


def _exists(p):
    from pathlib import Path
    return Path(p).exists()


if __name__ == "__main__":
    main()
