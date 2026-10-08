"""Poisson goal-COUNT GRU vs the binary scored GRU.

Same raw play sequences, but the target is goal COUNT (0,1,2,3...) with a Poisson
head, so multi-goal nights supervise the model more strongly than one-goal nights.
P(scores >=1) = 1 - exp(-lambda). We ask whether richer count supervision sharpens
the TOP of the ranking (the pick), not just the odds -- tested out-of-season both
directions, alongside the binary GRU and the rate model, plus ensembles.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

torch.set_num_threads(1)
torch.manual_seed(0)
np.random.seed(0)

SEASON_AVG = ["ixg_pg", "sog_pg", "satt_pg", "hd_pg", "md_pg", "rush_pg", "reb_pg",
              "pp_ixg_pg", "toi_pg", "assist_pg", "onxgf_pg", "goals_pg",
              "sogpct_s", "xgpsh_s", "hdshare_s", "rate_shrunk"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def topk(gids, p, y, k):
    d = pd.DataFrame({"g": gids, "p": p, "y": y})
    return float(np.mean([grp.sort_values("p", ascending=False).head(k).y.max() >= 1
                          for _, grp in d.groupby("g")]))


class GRUNet(nn.Module):
    def __init__(self, f, h=32):
        super().__init__()
        self.gru = nn.GRU(f, h, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(h, 1))

    def forward(self, x):
        _, hn = self.gru(x)
        return self.head(hn[-1]).squeeze(-1)


def fit_gru(Xtr, ttr, mode, epochs=25, bs=512, seed=None):
    """Train a GRU and return (net, mu, sd). seed=0 for a reproducible model."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    mu = np.asarray(Xtr.mean((0, 1)))
    sd = np.asarray(Xtr.std((0, 1))) + 1e-6
    Xn = (Xtr - mu) / sd
    net = GRUNet(Xtr.shape[2])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss() if mode == "binary" else \
        nn.PoissonNLLLoss(log_input=True, full=False)
    Xt = torch.tensor(Xn, dtype=torch.float32)
    tt = torch.tensor(ttr.astype(np.float32))
    n = len(Xt)
    for _ in range(epochs):
        net.train()
        for i in range(0, n, bs):
            idx = slice(i, i + bs)
            opt.zero_grad()
            loss = lossf(net(Xt[idx]), tt[idx])
            loss.backward(); opt.step()
    net.eval()
    return (net, torch.tensor(mu, dtype=torch.float32),
            torch.tensor(sd, dtype=torch.float32))


def predict_gru(net, mu, sd, Xte, mode="binary"):
    """P(scores>=1) for new sequences from a trained GRU."""
    Xn = (torch.tensor(np.asarray(Xte), dtype=torch.float32) - mu) / sd
    with torch.no_grad():
        out = net(Xn).numpy()
    if mode == "binary":
        return 1 / (1 + np.exp(-out))
    return 1 - np.exp(-np.exp(out))                 # poisson -> P(>=1)


def save_gru(net, mu, sd, path):
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd}, path)


def load_gru(path):
    try:
        d = torch.load(path, weights_only=True)
    except TypeError:                               # older torch w/o weights_only
        d = torch.load(path)
    net = GRUNet(int(d["mu"].shape[0]))
    net.load_state_dict(d["state"])
    net.eval()
    return net, d["mu"], d["sd"]


def train_gru(Xtr, ttr, Xte, mode, epochs=25, bs=512):
    """Train-then-predict in one call (used by the eval scripts)."""
    net, mu, sd = fit_gru(Xtr, ttr, mode, epochs, bs)
    return predict_gru(net, mu, sd, Xte, mode)


def main():
    sq = np.load("data/player_sequences.npz", allow_pickle=True)
    X = sq["X"]
    key = pd.DataFrame({"game_id": sq["game_id"], "player_id": sq["player_id"],
                        "season": sq["season"].astype(int), "scored": sq["y"]})
    sc = pd.read_parquet("data/scorer_playergames.parquet")[
        ["game_id", "player_id", "goals", "rate_shrunk"]]
    beh = pd.read_parquet("data/recent_behavior.parquet")
    key = key.merge(beh.drop(columns=["scored"]), on=["game_id", "player_id", "season"],
                    how="left")
    key = key.merge(sc, on=["game_id", "player_id"], how="left")
    key["rate_shrunk"] = key["rate_shrunk"].fillna(key["rate_shrunk"].mean())
    key["goals"] = key["goals"].fillna(0.0)
    assert len(key) == len(X)
    print("multi-goal share of scorers: "
          f"{(key.goals >= 2).sum() / max((key.goals >= 1).sum(), 1):.3f}")

    seasons = [20242025, 20252026]
    for a, b in [(seasons[0], seasons[1]), (seasons[1], seasons[0])]:
        tr = (key.season == a).values
        te = (key.season == b).values
        ktr, kte = key[tr], key[te]
        yte, g = kte.scored.values, kte.game_id.values
        print(f"\n===== train {a} -> test {b} =====")

        rate = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0,
            min_samples_leaf=200, random_state=0).fit(ktr[SEASON_AVG].astype(float),
                                                       ktr.scored)
        p_rate = rate.predict_proba(kte[SEASON_AVG].astype(float))[:, 1]
        p_bin = train_gru(X[tr], ktr.scored.values, X[te], "binary")
        p_poi = train_gru(X[tr], ktr.goals.values, X[te], "poisson")
        ens_bin = 1 / (1 + np.exp(-(0.5 * logit(p_rate) + 0.5 * logit(p_bin))))
        ens_poi = 1 / (1 + np.exp(-(0.5 * logit(p_rate) + 0.5 * logit(p_poi))))

        print(f"  {'model':<22}{'logloss':>9}{'top1':>8}{'top3':>8}")
        for lbl, p in [("rate (avgs)", p_rate), ("GRU binary", p_bin),
                       ("GRU poisson", p_poi), ("rate + GRU binary", ens_bin),
                       ("rate + GRU poisson", ens_poi)]:
            ll = log_loss(yte, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])
            print(f"  {lbl:<22}{ll:>9.4f}{topk(g, p, yte, 1):>8.3f}{topk(g, p, yte, 3):>8.3f}")


if __name__ == "__main__":
    main()
