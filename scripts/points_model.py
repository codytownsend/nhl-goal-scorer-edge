"""Points-trained model: predict P(player records >=1 POINT) per game.

The goal model already hits ~67% top-1 / ~91% top-3 on the POINTS target just by
reusing its goal ranking (points are far more predictable than who-finishes).
This trains directly on the point label and adds a shrunk POINT-rate feature
(the direct analog of rate_shrunk) to see how much higher a points-native model
ranks.

Compares on the POINTS target, both test seasons:
  base rate | goal-model ranking (the floor) | points-trained model
"""
import glob
import numpy as np
import pandas as pd

from style_vs_working import FEATURES, gbm, DEV, TEST
from sklearn.metrics import log_loss

K = 30.0


def topk(df, score_col, label_col, k):
    hits = []
    for _, g in df.groupby("game_id"):
        g = g.sort_values(score_col, ascending=False).head(k)
        hits.append(int(g[label_col].max() >= 1))
    return np.mean(hits)


def main():
    sp = pd.read_parquet("data/scorer_playergames.parquet")
    sp["season"] = sp["season"].astype(int)
    sp["date"] = pd.to_datetime(sp["date"])

    # points label from lineups (all seasons)
    lp = pd.concat([pd.read_parquet(f) for f in
                    sorted(glob.glob("data/lineups_2*.parquet"))],
                   ignore_index=True)[["game_id", "player_id", "points"]]
    lp = lp.drop_duplicates(["game_id", "player_id"])
    df = sp.merge(lp, on=["game_id", "player_id"], how="inner")
    df["point"] = (df["points"] >= 1).astype(int)
    print(f"merged {len(df)} player-games "
          f"(base point rate {df['point'].mean():.3f} vs goal {df['scored'].mean():.3f})")

    # shrunk POINT-rate feature (walk-forward, analog of rate_shrunk)
    df = df.sort_values(["player_id", "season", "date"]).reset_index(drop=True)
    grp = df.groupby(["player_id", "season"])
    df["_cp"] = grp["point"].cumsum() - df["point"]
    df["_n"] = grp.cumcount()
    pr = grp["point"].mean().rename("_pr").reset_index()
    seasons = sorted(df["season"].unique())
    nxt = {seasons[i]: seasons[i + 1] for i in range(len(seasons) - 1)}
    pr["season"] = pr["season"].map(nxt)
    pr = pr.dropna(subset=["season"])
    df = df.merge(pr, on=["player_id", "season"], how="left")
    df["_pr"] = df["_pr"].fillna(df["point"].mean())
    df["point_rate_shrunk"] = (df["_pr"] * K + df["_cp"]) / (K + df["_n"])
    df["point_l5"] = grp["point"].transform(
        lambda s: s.shift().rolling(5, min_periods=1).mean()).fillna(df["_pr"])

    dev, test = df[df.season.isin(DEV)], df[df.season.isin(TEST)].copy()

    # (1) goal model -> rank, score against POINT label (the floor)
    gm = gbm().fit(dev[FEATURES].astype(float), dev["scored"].values)
    test["p_goal"] = gm.predict_proba(test[FEATURES].astype(float))[:, 1]

    # (2a) points-trained on FEATURES only = what the DASHBOARD already does
    pm0 = gbm().fit(dev[FEATURES].astype(float), dev["point"].values)
    test["p_point_dash"] = pm0.predict_proba(test[FEATURES].astype(float))[:, 1]

    # (2b) points-trained + the NEW point-rate features (my addition)
    pfeat = FEATURES + ["point_rate_shrunk", "point_l5"]
    pm = gbm().fit(dev[pfeat].astype(float), dev["point"].values)
    test["p_point"] = pm.predict_proba(test[pfeat].astype(float))[:, 1]

    print(f"\n{'model (ranking)':<26}{'top1':>8}{'top3':>8}{'logloss(pt)':>12}")
    base = np.full(len(test), test["point"].mean())
    print(f"{'base point rate':<26}{'-':>8}{'-':>8}"
          f"{log_loss(test['point'], base, labels=[0,1]):>12.4f}")
    for lbl, col in [("goal-model ranking", "p_goal"),
                     ("points-trained (dashboard)", "p_point_dash"),
                     ("points-trained + pt-rate", "p_point")]:
        t1 = topk(test, col, "point", 1)
        t3 = topk(test, col, "point", 3)
        ll = log_loss(test["point"], test[col], labels=[0, 1])
        print(f"{lbl:<26}{t1:>8.3f}{t3:>8.3f}{ll:>12.4f}")

    print("\nper-season top1 / top3 (dashboard vs +pt-rate):")
    for sv in TEST:
        s = test[test.season == sv]
        d1, d3 = topk(s, "p_point_dash", "point", 1), topk(s, "p_point_dash", "point", 3)
        p1, p3 = topk(s, "p_point", "point", 1), topk(s, "p_point", "point", 3)
        print(f"  {str(sv)[:4]}: dashboard {d1:.3f}/{d3:.3f}   "
              f"+pt-rate {p1:.3f}/{p3:.3f}   (d top1 {p1-d1:+.3f})")


if __name__ == "__main__":
    main()
