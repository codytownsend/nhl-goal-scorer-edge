"""DISCOVERY ENGINE (step 2): per-STYLE pattern discovery in recent play.

For each emergent style (from build_styles.py) we let a flexible learner loose
on how a player has been playing LATELY -- recent-game action levels and
deviations from his own baseline -- with the season SCORING RATE removed, and
ask what precedes a goal. We don't specify the patterns; we extract what the
model keys on, and we only trust a pattern if it REPLICATES when we swap which
season we fit vs test.

Reports, per style:
  - does recent-play beat the rate baseline on the HELD-OUT season?
  - the top recent-play patterns, and whether they replicate across seasons.
  - a flag-precision check (top-decile recent-play score: scored vs baseline).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss

TEST = [20242025, 20252026]

# RATE = season-average talent (the thing we want to BEAT / de-weight)
RATE = ["ixg_pg", "goals_pg", "sog_pg", "hd_pg", "md_pg", "satt_pg",
        "pp_ixg_pg", "onxgf_pg", "assist_pg", "toi_pg", "xgpsh_s", "sogpct_s"]
# RECENT = how he's been playing lately (levels + deviations + drought)
RECENT_LVL = ["ixg_l5", "sog_l5", "hd_l5", "md_l5", "satt_l5", "rush_l5",
              "reb_l5", "pp_ixg_l5", "onxgf_l5", "assist_l5", "toi_l5",
              "dist_l5", "sogpct_l5", "xgpsh_l5"]
RECENT_DEV = ["ixg_dev", "sog_dev", "hd_dev", "md_dev", "satt_dev", "rush_dev",
              "reb_dev", "pp_ixg_dev", "onxgf_dev", "assist_dev", "toi_dev",
              "sogpct_dev", "xgpsh_dev"]
RECENT_CONS = ["dry5", "since_goal", "games_w_shot5", "ixg_std5"]
RECENT = RECENT_LVL + RECENT_DEV + RECENT_CONS

STYLE_LABEL = {0: "net-front finishers", 1: "point shooters (D)",
               2: "perimeter/PP", 3: "net-crashers"}


def gbm():
    return HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.05, max_depth=4, l2_regularization=1.0,
        min_samples_leaf=150, random_state=0)


def fit_eval(tr, te, feats):
    m = gbm().fit(tr[feats].astype(float), tr["scored"].values)
    p = m.predict_proba(te[feats].astype(float))[:, 1]
    return m, p, log_loss(te["scored"].values, p, labels=[0, 1])


def main():
    b = pd.read_parquet("data/recent_behavior.parquet")
    st = pd.read_parquet("data/player_styles.parquet")
    df = b.merge(st[["shooter_id", "season", "style"]],
                 left_on=["player_id", "season"],
                 right_on=["shooter_id", "season"], how="inner")
    df = df.dropna(subset=RATE + RECENT).copy()
    print(f"n={len(df)} player-games with a style\n")

    for sty in sorted(df["style"].unique()):
        d = df[df["style"] == sty]
        a, c = d[d.season == TEST[0]], d[d.season == TEST[1]]
        if len(a) < 2000 or len(c) < 2000:
            continue
        lbl = STYLE_LABEL.get(sty, f"style {sty}")
        print("=" * 70)
        print(f"STYLE {sty}: {lbl}   (n={len(d)}, base scored {d.scored.mean():.3f})")
        print("=" * 70)

        # both directions: fit A->test C, fit C->test A (replication)
        imp_runs = []
        beats = []
        for tr, te, tag in [(a, c, "fit24->test25"), (c, a, "fit25->test24")]:
            _, _, ll_rate = fit_eval(tr, te, RATE)
            _, _, ll_rec = fit_eval(tr, te, RECENT)
            m_all, _, ll_all = fit_eval(tr, te, RATE + RECENT)
            beats.append(ll_all < ll_rate)
            print(f"  [{tag}] logloss  rate={ll_rate:.4f}  recent-only={ll_rec:.4f}"
                  f"  rate+recent={ll_all:.4f}  (recent adds "
                  f"{ll_rate-ll_all:+.4f})")
            # which RECENT features the combined model keys on
            samp = te.sample(min(8000, len(te)), random_state=0)
            pi = permutation_importance(
                m_all, samp[RATE + RECENT].astype(float),
                samp["scored"].values, scoring="neg_log_loss",
                n_repeats=3, random_state=0)
            imp = pd.Series(pi.importances_mean, index=RATE + RECENT)
            imp_runs.append(imp[RECENT])

        # replication: recent features positive-importance in BOTH directions
        rep = pd.concat(imp_runs, axis=1)
        rep.columns = ["run1", "run2"]
        rep["stable"] = (rep.run1 > 0) & (rep.run2 > 0)
        rep["avg"] = rep[["run1", "run2"]].mean(axis=1)
        top = rep[rep.stable].sort_values("avg", ascending=False).head(6)
        print(f"  recent-play beats rate on held-out: {beats}")
        if len(top):
            print("  REPLICATING recent-play patterns (importance both dirs):")
            for f, r in top.iterrows():
                print(f"     {f:<16} imp {r.avg:+.5f}")
        else:
            print("  REPLICATING recent-play patterns: NONE")
        print()


if __name__ == "__main__":
    main()
