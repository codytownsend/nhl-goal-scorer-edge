"""A model built NATIVELY on the play data, tested for ORTHOGONAL signal.

Three models, trained on one season, evaluated on the other (both directions):
  * rate   : GBM on season averages          (the existing-style model)
  * form   : GRU on raw game-by-game sequences (built only on how he's playing)
  * formGB : GBM on summarized play features   (play data, but order discarded)

The decisive question (same test that validated 'finishing' and rejected 'xEV'):
does ENSEMBLING the play model with the rate model beat the rate model alone?
If yes, the play data carries signal the rate doesn't -> the user is right and
we found something. If no, the play data is redundant even with its own model.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(1)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

torch.manual_seed(0)
np.random.seed(0)

SEASON_AVG = ["ixg_pg", "sog_pg", "satt_pg", "hd_pg", "md_pg", "rush_pg", "reb_pg",
              "pp_ixg_pg", "toi_pg", "assist_pg", "onxgf_pg", "goals_pg",
              "sogpct_s", "xgpsh_s", "hdshare_s", "rate_shrunk"]
FORM_GB = (["sog_l5", "satt_l5", "ixg_l5", "hd_l5", "md_l5", "rush_l5", "reb_l5",
            "pp_ixg_l5", "toi_l5", "assist_l5", "onxgf_l5", "sogpct_l5", "xgpsh_l5",
            "dist_l5"] +
           ["sog_dev", "satt_dev", "ixg_dev", "hd_dev", "md_dev", "rush_dev",
            "reb_dev", "pp_ixg_dev", "toi_dev", "assist_dev", "onxgf_dev",
            "sogpct_dev", "xgpsh_dev"] +
           ["ixg_std5", "games_w_shot5", "dry5", "since_goal"])


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


def train_gru(Xtr, ytr, Xte, epochs=25, bs=512):
    mu, sd = Xtr.mean((0, 1)), Xtr.std((0, 1)) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    net = GRUNet(Xtr.shape[2])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    n = len(Xt)
    for ep in range(epochs):
        net.train()
        for i in range(0, n, bs):
            idx = slice(i, i + bs)
            opt.zero_grad()
            loss = lossf(net(Xt[idx]), yt[idx])
            loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(Xte))).numpy()


def main():
    sq = np.load("data/player_sequences.npz", allow_pickle=True)
    X, y = sq["X"], sq["y"]
    key = pd.DataFrame({"game_id": sq["game_id"], "player_id": sq["player_id"],
                        "season": sq["season"].astype(int), "y": y})
    beh = pd.read_parquet("data/recent_behavior.parquet")
    sc = pd.read_parquet("data/scorer_playergames.parquet")[
        ["game_id", "player_id", "rate_shrunk"]]
    feat = beh.merge(sc, on=["game_id", "player_id"], how="left")
    feat["rate_shrunk"] = feat["rate_shrunk"].fillna(feat["rate_shrunk"].mean())
    key = key.merge(feat, on=["game_id", "player_id", "season"], how="left")
    assert len(key) == len(X)

    seasons = [20242025, 20252026]
    for a, b in [(seasons[0], seasons[1]), (seasons[1], seasons[0])]:
        tr = (key.season == a).values
        te = (key.season == b).values
        ktr, kte = key[tr], key[te]
        print(f"\n===== train {a} -> test {b}  (n_test={te.sum()}) =====")

        rate = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0,
            min_samples_leaf=200, random_state=0).fit(ktr[SEASON_AVG].astype(float), ktr.y)
        p_rate = rate.predict_proba(kte[SEASON_AVG].astype(float))[:, 1]

        fgb = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0,
            min_samples_leaf=200, random_state=0).fit(ktr[FORM_GB].astype(float), ktr.y)
        p_fgb = fgb.predict_proba(kte[FORM_GB].astype(float))[:, 1]

        p_seq = train_gru(X[tr], y[tr], X[te])

        yte = kte.y.values
        g = kte.game_id.values
        # equal-weight logit-average ensembles (no fitting -> leak-free)
        ens_seq = 1 / (1 + np.exp(-(0.5 * logit(p_rate) + 0.5 * logit(p_seq))))
        ens_gb = 1 / (1 + np.exp(-(0.5 * logit(p_rate) + 0.5 * logit(p_fgb))))

        print(f"  {'model':<22}{'logloss':>9}{'top1':>8}{'top3':>8}")
        for lbl, p in [("rate (avgs)", p_rate), ("form GRU (seq)", p_seq),
                       ("form GBM (summary)", p_fgb),
                       ("rate + GRU", ens_seq), ("rate + formGBM", ens_gb)]:
            ll = log_loss(yte, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])
            print(f"  {lbl:<22}{ll:>9.4f}{topk(g, p, yte, 1):>8.3f}{topk(g, p, yte, 3):>8.3f}")


if __name__ == "__main__":
    main()
