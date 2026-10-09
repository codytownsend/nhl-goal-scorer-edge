"""SLEEPER test: among NON-STARS, which behavior signal flags a player who
beats his own baseline rate tonight?

The whole project predicts a player's RATE and applies it to every game. This
asks the opposite (the "residual"): hold the rate fixed, restrict to players
who are NOT their team's leading scorers, and find a signal that says THIS guy
is more likely than usual TONIGHT.

Metric is PRECISION/LIFT on a flag (not overall top-1): when we flag a
non-star, does he score more than his baseline P0 implies -- and does it
REPLICATE in a second season (same sign). A rare high-precision flag is the
goal, even if it never moves aggregate accuracy.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

# rate/talent model features (season-average + role), identical to scorer_eval
RATE_FEATURES = ["g_pg", "sh_pg", "satt_pg", "ixg_pg", "hd_pg", "pp_ixg_pg",
                 "finish", "toi_pg", "gp", "g_l5", "ixg_l5", "sh_l5", "g_l10",
                 "ixg_l10", "is_home", "is_def", "opp_goalie", "opp_def",
                 "rate_shrunk", "g_rate_shrunk", "toi_l5", "toi_trend",
                 "pp_ixg_l5", "pp_trend"]
DEV = [20202021, 20212022, 20222023, 20232024]
TEST = [20242025, 20252026]

# "behavior" signals = how he's been playing vs his OWN baseline (deviations)
# + a couple of matchup/opportunity knobs that could specifically help a sleeper
SIGNALS = ["toi_dev", "ixg_dev", "satt_dev", "sog_dev", "hd_dev", "md_dev",
           "pp_ixg_dev", "onxgf_dev", "rush_dev", "reb_dev", "assist_dev",
           "sogpct_dev", "xgpsh_dev", "games_w_shot5", "dry5", "since_goal",
           "ixg_std5"]


def build():
    sp = pd.read_parquet("data/scorer_playergames.parquet")
    sp["season"] = sp["season"].astype(int)
    dev = sp[sp.season.isin(DEV)]
    gbm = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0,
        min_samples_leaf=200, random_state=0
    ).fit(dev[RATE_FEATURES].astype(float), dev["scored"].values)
    sp = sp[sp.season.isin(TEST)].copy()
    sp["P0"] = gbm.predict_proba(sp[RATE_FEATURES].astype(float))[:, 1]

    beh = pd.read_parquet("data/recent_behavior.parquet")
    keep = ["game_id", "player_id", "season"] + SIGNALS
    df = sp.merge(beh[keep], on=["game_id", "player_id", "season"], how="inner")

    # non-star = NOT top-2 on his team tonight by baseline rate P0
    df["team_rank"] = (df.groupby(["game_id", "team"])["P0"]
                         .rank(ascending=False, method="first"))
    df["is_nonstar"] = df["team_rank"] > 2
    df["resid"] = df["scored"] - df["P0"]
    return df


def signal_table(df, pool_name):
    print(f"\n{'='*66}\nPOOL: {pool_name}  (n={len(df)}, "
          f"base scored {df['scored'].mean():.3f}, mean P0 {df['P0'].mean():.3f})"
          f"\n{'='*66}")
    rows = []
    for s in SIGNALS:
        # correlation of signal with residual, PER SEASON (replication gate)
        corrs = {}
        for seas in TEST:
            sub = df[df.season == seas]
            x = sub[s].values
            if np.nanstd(x) == 0:
                corrs[seas] = np.nan
            else:
                corrs[seas] = np.corrcoef(x, sub["resid"].values)[0, 1]
        c24, c25 = corrs[20242025], corrs[20252026]
        stable = (np.sign(c24) == np.sign(c25)) and abs(c24) > 0.01 and abs(c25) > 0.01

        # FLAG test: top decile of signal (within season) -> scored vs baseline
        lifts = {}
        for seas in TEST:
            sub = df[df.season == seas]
            thr = sub[s].quantile(0.90)
            fl = sub[sub[s] >= thr]
            lifts[seas] = (fl["scored"].mean(), fl["P0"].mean(), len(fl))
        rows.append((s, c24, c25, stable, lifts))

    rows.sort(key=lambda r: -(abs(r[1]) + abs(r[2])) if r[3] else 0)
    print(f"{'signal':<14}{'corr24':>8}{'corr25':>8}{'stable':>7}"
          f"   top-decile flag: scored vs baseline (lift)")
    for s, c24, c25, stable, lifts in rows:
        (sc24, b24, n24), (sc25, b25, n25) = lifts[20242025], lifts[20252026]
        flag = "  <<" if stable else ""
        print(f"{s:<14}{c24:>+8.3f}{c25:>+8.3f}{str(stable):>7}   "
              f"24:{sc24:.3f}/{b24:.3f}({sc24-b24:+.3f})  "
              f"25:{sc25:.3f}/{b25:.3f}({sc25-b25:+.3f}){flag}")


def main():
    df = build()
    signal_table(df[df.is_nonstar], "NON-STARS (rank>2 on team by rate)")
    signal_table(df[~df.is_nonstar], "STARS (top-2 on team, for contrast)")


if __name__ == "__main__":
    main()
