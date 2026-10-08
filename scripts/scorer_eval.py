"""First end-to-end evaluation of the anytime-goal-scorer model.

Trains on dev seasons, tests OUT-OF-SAMPLE on later seasons. Reports the
metrics that match the actual question ("most likely goal scorer"):
  - top-1 / top-3 per-game hit rate (did our top-ranked skater(s) score?)
  - log-loss / Brier / AUC on P(scored >=1)
  - calibration deciles
and compares the model against naive baselines (rank by season G/GP, by
individual xG/game, by last-5 recency).
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

FEATURES = ["g_pg", "sh_pg", "satt_pg", "ixg_pg", "hd_pg", "pp_ixg_pg",
            "finish", "toi_pg", "gp", "g_l5", "ixg_l5", "sh_l5", "g_l10",
            "ixg_l10", "is_home", "is_def", "opp_goalie", "opp_def",
            # added: shrink-to-prior rate + role/usage trends
            "rate_shrunk", "g_rate_shrunk", "toi_l5", "toi_trend",
            "pp_ixg_l5", "pp_trend"]

DEV = [20202021, 20212022, 20222023, 20232024]
TEST = [20242025, 20252026]


def topk_hit(df, score_col, k):
    """Per game: fraction of games where >=1 of the top-k ranked skaters scored.
    Also returns precision@k = fraction of top-k picks that scored."""
    hits, prec = [], []
    for _, g in df.groupby("game_id"):
        g = g.sort_values(score_col, ascending=False).head(k)
        hits.append(int(g["scored"].max() >= 1))
        prec.append(g["scored"].mean())
    return np.mean(hits), np.mean(prec)


def calib(y, p, bins=10):
    q = pd.qcut(p, bins, duplicates="drop")
    t = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q", observed=True).agg(
        pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    return t


def main():
    df = pd.read_parquet("data/scorer_playergames.parquet")
    df["season"] = df["season"].astype(int)
    dev = df[df.season.isin(DEV)].copy()
    test = df[df.season.isin(TEST)].copy()

    Xd, yd = dev[FEATURES].astype(float), dev["scored"].values
    gbm = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4,
        l2_regularization=1.0, min_samples_leaf=200, random_state=0).fit(Xd, yd)
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(
        (Xd - Xd.mean()) / Xd.std(), yd)

    for name, season in [("2024-25", 20242025), ("2025-26", 20252026),
                         ("OOS both", None)]:
        t = test if season is None else test[test.season == season]
        Xt, yt = t[FEATURES].astype(float), t["scored"].values
        pm = gbm.predict_proba(Xt)[:, 1]
        pl = lr.predict_proba((Xt - Xd.mean()) / Xd.std())[:, 1]
        t = t.assign(p_gbm=pm, p_lr=pl)
        print(f"\n===== {name}  (n={len(t)} player-games, "
              f"{yt.mean():.3f} base scored) =====")
        base = np.full_like(yt, yt.mean(), dtype=float)
        print(f"{'model':<16}{'logloss':>9}{'brier':>8}{'auc':>7}"
              f"{'top1':>8}{'top3':>8}{'prec@3':>8}")
        rows = [
            ("GBM", pm, "p_gbm"),
            ("logistic", pl, "p_lr"),
            ("base rate", base, None),
        ]
        for lbl, p, col in rows:
            ll = log_loss(yt, p, labels=[0, 1])
            br = brier_score_loss(yt, p)
            try:
                au = roc_auc_score(yt, p)
            except ValueError:
                au = float("nan")
            if col:
                h1, _ = topk_hit(t, col, 1)
                h3, pr3 = topk_hit(t, col, 3)
                print(f"{lbl:<16}{ll:>9.4f}{br:>8.4f}{au:>7.3f}"
                      f"{h1:>8.3f}{h3:>8.3f}{pr3:>8.3f}")
            else:
                print(f"{lbl:<16}{ll:>9.4f}{br:>8.4f}{au:>7.3f}"
                      f"{'-':>8}{'-':>8}{'-':>8}")
        # naive ranking baselines (no training) for top-k comparison
        for lbl, col in [("rank g_pg", "g_pg"), ("rank ixg_pg", "ixg_pg"),
                         ("rank g_l5", "g_l5"), ("rank sh_pg", "sh_pg")]:
            h1, _ = topk_hit(t, col, 1)
            h3, pr3 = topk_hit(t, col, 3)
            print(f"{lbl:<16}{'-':>9}{'-':>8}{'-':>7}"
                  f"{h1:>8.3f}{h3:>8.3f}{pr3:>8.3f}")

    # calibration + feature importance on OOS both
    Xt = test[FEATURES].astype(float)
    p = gbm.predict_proba(Xt)[:, 1]
    print("\nGBM calibration (OOS both):")
    print(calib(test["scored"].values, p).round(3).to_string())

    # permutation importance (quick, on a sample)
    from sklearn.inspection import permutation_importance
    samp = test.sample(min(20000, len(test)), random_state=0)
    imp = permutation_importance(
        gbm, samp[FEATURES].astype(float), samp["scored"].values,
        scoring="neg_log_loss", n_repeats=3, random_state=0)
    order = np.argsort(imp.importances_mean)[::-1]
    print("\nPermutation importance (neg-logloss drop, OOS):")
    for i in order:
        print(f"  {FEATURES[i]:<12}{imp.importances_mean[i]:+.5f}")


if __name__ == "__main__":
    main()
