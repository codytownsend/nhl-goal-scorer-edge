"""Seed-robustness of the Poisson-GRU top-1 gain.

A single GRU run can win by luck of initialization. Re-train across several seeds
and report mean +/- std top-1/top-3 for: rate alone, rate+binary GRU, rate+Poisson
GRU, both season directions. If the Poisson pick-gain survives averaging, it's real.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

import form_poisson as FP   # reuse GRUNet, train_gru, topk, logit, SEASON_AVG

torch.set_num_threads(1)
SEEDS = [0, 1, 2, 3, 4]


def load():
    sq = np.load("data/player_sequences.npz", allow_pickle=True)
    X = sq["X"]
    key = pd.DataFrame({"game_id": sq["game_id"], "player_id": sq["player_id"],
                        "season": sq["season"].astype(int), "scored": sq["y"]})
    sc = pd.read_parquet("data/scorer_playergames.parquet")[
        ["game_id", "player_id", "goals", "rate_shrunk"]]
    beh = pd.read_parquet("data/recent_behavior.parquet").drop(columns=["scored"])
    key = key.merge(beh, on=["game_id", "player_id", "season"], how="left")
    key = key.merge(sc, on=["game_id", "player_id"], how="left")
    key["rate_shrunk"] = key["rate_shrunk"].fillna(key["rate_shrunk"].mean())
    key["goals"] = key["goals"].fillna(0.0)
    return X, key


def main():
    X, key = load()
    for a, b in [(20242025, 20252026), (20252026, 20242025)]:
        tr, te = (key.season == a).values, (key.season == b).values
        ktr, kte = key[tr], key[te]
        yte, g = kte.scored.values, kte.game_id.values

        rate = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0,
            min_samples_leaf=200, random_state=0).fit(ktr[FP.SEASON_AVG].astype(float),
                                                       ktr.scored)
        p_rate = rate.predict_proba(kte[FP.SEASON_AVG].astype(float))[:, 1]
        r1, r3 = FP.topk(g, p_rate, yte, 1), FP.topk(g, p_rate, yte, 3)

        res = {"rate+binary": [], "rate+poisson": []}
        for s in SEEDS:
            torch.manual_seed(s); np.random.seed(s)
            p_bin = FP.train_gru(X[tr], ktr.scored.values, X[te], "binary")
            torch.manual_seed(s); np.random.seed(s)
            p_poi = FP.train_gru(X[tr], ktr.goals.values, X[te], "poisson")
            eb = 1 / (1 + np.exp(-(0.5 * FP.logit(p_rate) + 0.5 * FP.logit(p_bin))))
            ep = 1 / (1 + np.exp(-(0.5 * FP.logit(p_rate) + 0.5 * FP.logit(p_poi))))
            res["rate+binary"].append((FP.topk(g, eb, yte, 1), FP.topk(g, eb, yte, 3)))
            res["rate+poisson"].append((FP.topk(g, ep, yte, 1), FP.topk(g, ep, yte, 3)))

        print(f"\n===== train {a} -> test {b}  (over {len(SEEDS)} seeds) =====")
        print(f"  rate alone            top1 {r1:.3f}         top3 {r3:.3f}")
        for k, v in res.items():
            t1 = np.array([x[0] for x in v]); t3 = np.array([x[1] for x in v])
            print(f"  {k:<22}top1 {t1.mean():.3f}+-{t1.std():.3f}   "
                  f"top3 {t3.mean():.3f}+-{t3.std():.3f}")


if __name__ == "__main__":
    main()
