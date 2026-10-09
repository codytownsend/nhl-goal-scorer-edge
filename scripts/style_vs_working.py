"""Head-to-head: STYLE-aware pick vs the current WORKING model's pick.

Three predictors, all producing P(goal) per player-game, ranked within each
game:
  BASELINE  = current working scorer GBM (FEATURES only, trained on DEV) -- the
              deployed model's pick.
  GLOBAL+R  = one GBM on FEATURES + recent-play features (no styling) -- isolates
              whether the RECENT features help at all.
  STYLE+R   = per-style GBMs on FEATURES + recent play, each player routed to his
              style's model -- the new approach.

Because the recent-play features only exist for the two test seasons, GLOBAL+R
and STYLE+R are trained with a both-direction split (fit 24-25 -> predict 25-26
and vice-versa) so every test row is predicted out-of-fold. BASELINE is the
DEV-trained working model. All three are compared on the SAME pool (player-games
that have a style), ranked within each game.

Reports top-1/top-3 for each, how often the top-1 pick differs between BASELINE
and STYLE+R, and accuracy when they agree vs disagree.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

FEATURES = ["g_pg", "sh_pg", "satt_pg", "ixg_pg", "hd_pg", "pp_ixg_pg",
            "finish", "toi_pg", "gp", "g_l5", "ixg_l5", "sh_l5", "g_l10",
            "ixg_l10", "is_home", "is_def", "opp_goalie", "opp_def",
            "rate_shrunk", "g_rate_shrunk", "toi_l5", "toi_trend",
            "pp_ixg_l5", "pp_trend"]
RECENT = ["ixg_dev", "sog_dev", "hd_dev", "md_dev", "satt_dev", "rush_dev",
          "reb_dev", "pp_ixg_dev", "onxgf_dev", "assist_dev", "toi_dev",
          "sog_l5", "hd_l5", "satt_l5", "onxgf_l5", "dist_l5",
          "dry5", "since_goal", "games_w_shot5", "ixg_std5"]
DEV = [20202021, 20212022, 20222023, 20232024]
TEST = [20242025, 20252026]


def gbm():
    return HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.05, max_depth=4, l2_regularization=1.0,
        min_samples_leaf=150, random_state=0)


def topk(df, col, k):
    """per game: did >=1 of top-k ranked players score (hit), and top-1 player."""
    hits, picks = [], {}
    for gid, g in df.groupby("game_id"):
        g = g.sort_values(col, ascending=False)
        hits.append(int(g["scored"].head(k).max() >= 1))
        picks[gid] = g["player_id"].iloc[0]
    return np.mean(hits), picks


def build_predictions():
    """Build the shared pool with p_base / p_glob / p_style for every row."""
    sp = pd.read_parquet("data/scorer_playergames.parquet")
    sp["season"] = sp["season"].astype(int)
    beh = pd.read_parquet("data/recent_behavior.parquet")
    sty = pd.read_parquet("data/player_styles.parquet")

    # pool = test player-games that have recent features AND a style
    keep_beh = ["game_id", "player_id", "season"] + \
        [c for c in RECENT if c not in sp.columns]
    pool = (sp[sp.season.isin(TEST)]
            .merge(beh[keep_beh], on=["game_id", "player_id", "season"])
            .merge(sty[["shooter_id", "season", "style"]],
                   left_on=["player_id", "season"],
                   right_on=["shooter_id", "season"]))
    allfeat = FEATURES + [c for c in RECENT if c not in FEATURES]
    pool = pool.dropna(subset=allfeat).copy()

    # BASELINE: working model trained on DEV, FEATURES only
    dev = sp[sp.season.isin(DEV)]
    base = gbm().fit(dev[FEATURES].astype(float), dev["scored"].values)
    pool["p_base"] = base.predict_proba(pool[FEATURES].astype(float))[:, 1]

    # GLOBAL+R and STYLE+R via both-direction split
    pool["p_glob"] = np.nan
    pool["p_style"] = np.nan
    for tr_s, te_s in [(TEST[0], TEST[1]), (TEST[1], TEST[0])]:
        tr = pool[pool.season == tr_s]
        te_mask = pool.season == te_s
        te = pool[te_mask]
        # global + recent
        g = gbm().fit(tr[allfeat].astype(float), tr["scored"].values)
        pool.loc[te_mask, "p_glob"] = g.predict_proba(
            te[allfeat].astype(float))[:, 1]
        # per-style
        ps = np.zeros(len(te))
        for s in sorted(tr["style"].unique()):
            trs = tr[tr["style"] == s]
            if len(trs) < 1500:
                continue
            ms = gbm().fit(trs[allfeat].astype(float), trs["scored"].values)
            idx = (te["style"] == s).values
            if idx.sum():
                ps[idx] = ms.predict_proba(
                    te[idx][allfeat].astype(float))[:, 1]
        # fallback for thin styles (never got a per-style model): use global
        miss = ps == 0
        if miss.any():
            ps[miss] = g.predict_proba(
                te[miss][allfeat].astype(float))[:, 1]
        pool.loc[te_mask, "p_style"] = ps
    return pool


def main():
    pool = build_predictions()
    print(f"pool: {len(pool)} player-games, "
          f"{pool.game_id.nunique()} games, base scored {pool.scored.mean():.3f}")

    # ---- per-game metrics on identical pool ----
    print(f"\n{'model':<14}{'top1':>8}{'top3':>8}")
    picks = {}
    for lbl, col in [("BASELINE", "p_base"), ("GLOBAL+R", "p_glob"),
                     ("STYLE+R", "p_style")]:
        t1, pk = topk(pool, col, 1)
        t3, _ = topk(pool, col, 3)
        picks[lbl] = pk
        print(f"{lbl:<14}{t1:>8.3f}{t3:>8.3f}")

    # ---- BASELINE vs STYLE+R agreement analysis ----
    scored_top = {}
    for gid, g in pool.groupby("game_id"):
        scored_top[gid] = dict(zip(g["player_id"], g["scored"]))
    games = list(picks["BASELINE"].keys())
    same = [gid for gid in games
            if picks["BASELINE"][gid] == picks["STYLE+R"][gid]]
    diff = [gid for gid in games
            if picks["BASELINE"][gid] != picks["STYLE+R"][gid]]

    def acc(gids, model):
        return np.mean([scored_top[g][picks[model][g]] for g in gids])

    print(f"\n=== BASELINE vs STYLE+R top-1 pick ===")
    print(f"total games: {len(games)}")
    print(f"SAME pick:   {len(same)} ({len(same)/len(games):.1%})  "
          f"-> top-1 acc {acc(same,'BASELINE'):.3f}")
    print(f"DIFFERENT:   {len(diff)} ({len(diff)/len(games):.1%})")
    if diff:
        print(f"   when different, BASELINE pick scores: "
              f"{acc(diff,'BASELINE'):.3f}")
        print(f"   when different, STYLE+R  pick scores: "
              f"{acc(diff,'STYLE+R'):.3f}")


if __name__ == "__main__":
    main()
