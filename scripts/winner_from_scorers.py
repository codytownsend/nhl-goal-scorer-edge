"""Can the goal-scorer rankings alone predict the GAME WINNER?

Idea (user): if more of team A's skaters are likely to score than team B's,
team A should win more often. Aggregate each team's per-skater P(goal) into a
team score, pick the higher team, and measure accuracy. Bucket by the gap
between the two team scores to see if confidence tracks the margin.

Leak-free: train the scorer GBM on DEV seasons, predict P(goal) walk-forward
on TEST player-games (identical setup to scorer_eval.py), then aggregate.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

FEATURES = ["g_pg", "sh_pg", "satt_pg", "ixg_pg", "hd_pg", "pp_ixg_pg",
            "finish", "toi_pg", "gp", "g_l5", "ixg_l5", "sh_l5", "g_l10",
            "ixg_l10", "is_home", "is_def", "opp_goalie", "opp_def",
            "rate_shrunk", "g_rate_shrunk", "toi_l5", "toi_trend",
            "pp_ixg_l5", "pp_trend"]
DEV = [20202021, 20212022, 20222023, 20232024]
TEST = [20242025, 20252026]


def team_scores(df):
    """One row per game: home/away aggregated skater P(goal) + expected goals."""
    # expected distinct scorers (sum of P) and expected goals (lambda = -ln(1-p))
    df = df.assign(lam=-np.log(np.clip(1 - df["p"], 1e-9, 1.0)))
    agg = (df.groupby(["game_id", "team"])
             .agg(sum_p=("p", "sum"), xg=("lam", "sum"), n=("p", "size"),
                  is_home=("is_home", "max"))
             .reset_index())
    home = agg[agg.is_home == 1].set_index("game_id")
    away = agg[agg.is_home == 0].set_index("game_id")
    out = pd.DataFrame({
        "home_sum_p": home["sum_p"], "away_sum_p": away["sum_p"],
        "home_xg": home["xg"], "away_xg": away["xg"],
        "home_n": home["n"], "away_n": away["n"],
    }).dropna()
    return out


def report(out, score="sum_p"):
    h, a = out[f"home_{score}"], out[f"away_{score}"]
    gap = h - a                      # signed: >0 => home favored
    pred_home = (gap > 0).astype(int)
    acc = (pred_home == out["home_win"]).mean()
    n = len(out)
    print(f"\n--- aggregate = {score}  (n={n} games) ---")
    print(f"overall accuracy picking higher team : {acc:.3f}")
    print(f"  (baselines: always-home {out['home_win'].mean():.3f}, "
          f"coin flip 0.500)")

    # confidence buckets by |gap|
    absgap = gap.abs()
    qs = pd.qcut(absgap, 5, labels=["Q1 closest", "Q2", "Q3", "Q4",
                                    "Q5 widest"])
    print(f"\n  confidence by |team score gap| quintile:")
    print(f"  {'bucket':<12}{'n':>6}{'gap_lo':>8}{'gap_hi':>8}{'acc':>8}")
    tmp = pd.DataFrame({"q": qs, "gap": absgap,
                        "correct": (pred_home == out["home_win"]).astype(int)})
    for name, grp in tmp.groupby("q", observed=True):
        print(f"  {name:<12}{len(grp):>6}{grp.gap.min():>8.2f}"
              f"{grp.gap.max():>8.2f}{grp.correct.mean():>8.3f}")

    # fixed thresholds too
    print(f"\n  accuracy at |gap| thresholds:")
    for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
        m = absgap >= t
        if m.sum() > 30:
            print(f"    |gap|>={t:<4} n={m.sum():>5}  "
                  f"acc={(pred_home[m] == out['home_win'][m]).mean():.3f}  "
                  f"({100*m.mean():.0f}% of games)")


def main():
    df = pd.read_parquet("data/scorer_playergames.parquet")
    df["season"] = df["season"].astype(int)
    dev = df[df.season.isin(DEV)]
    test = df[df.season.isin(TEST)].copy()

    gbm = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0,
        min_samples_leaf=200, random_state=0
    ).fit(dev[FEATURES].astype(float), dev["scored"].values)
    test["p"] = gbm.predict_proba(test[FEATURES].astype(float))[:, 1]

    games = pd.concat([pd.read_parquet(f"data/games_{s}.parquet")
                       for s in ["20242025", "20252026"]])
    games = games.set_index("game_id")["home_win"]

    for season, label in [(None, "OOS both"), (20242025, "2024-25"),
                          (20252026, "2025-26")]:
        t = test if season is None else test[test.season == season]
        out = team_scores(t)
        out = out.join(games, how="inner")
        print(f"\n===== {label} =====")
        report(out, "sum_p")
        report(out, "xg")


if __name__ == "__main__":
    main()
