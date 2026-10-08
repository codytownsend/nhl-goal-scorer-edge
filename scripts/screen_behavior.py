"""Let the data find patterns in HOW players play before they score.

The honest question: given a model that already knows all of a player's SEASON
AVERAGES, does adding his RECENT PLAY (last-5 levels, deviations from his own
baseline, consistency/drought) improve prediction of whether he scores the next
game? We let a gradient-boosted model search for patterns (nonlinear,
interactions) and judge it OUT OF SEASON (train one season, test the other, both
directions) -- so any 'pattern' must generalize, not fit noise.

Then we show WHICH behaviors the data leaned on (permutation importance) and a
per-feature stability table (same sign both seasons = real, else noise).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss

TEST = [20242025, 20252026]

SEASON_AVG = ["ixg_pg", "sog_pg", "satt_pg", "hd_pg", "md_pg", "rush_pg",
              "reb_pg", "pp_ixg_pg", "toi_pg", "assist_pg", "onxgf_pg",
              "goals_pg", "sogpct_s", "xgpsh_s", "hdshare_s", "rate_shrunk"]
RECENT = ["sog_l5", "satt_l5", "ixg_l5", "hd_l5", "md_l5", "rush_l5", "reb_l5",
          "pp_ixg_l5", "toi_l5", "assist_l5", "onxgf_l5", "sogpct_l5",
          "xgpsh_l5", "dist_l5"]
DEV = ["sog_dev", "satt_dev", "ixg_dev", "hd_dev", "md_dev", "rush_dev",
       "reb_dev", "pp_ixg_dev", "toi_dev", "assist_dev", "onxgf_dev",
       "sogpct_dev", "xgpsh_dev"]
CONSIST = ["ixg_std5", "games_w_shot5", "dry5", "since_goal"]
RECENT_ALL = RECENT + DEV + CONSIST


def topk(df, col, k):
    return float(np.mean([g.sort_values(col, ascending=False).head(k).scored.max() >= 1
                          for _, g in df.groupby("game_id")]))


def gbm():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0,
        min_samples_leaf=200, random_state=0)


def fit_eval(tr, te, feats):
    m = gbm().fit(tr[feats].astype(float), tr.scored)
    te = te.copy()
    te["p"] = m.predict_proba(te[feats].astype(float))[:, 1]
    ll = log_loss(te.scored, te.p.clip(1e-6, 1 - 1e-6), labels=[0, 1])
    return m, te, ll, topk(te, "p", 1), topk(te, "p", 3)


def partial_corr(d, feat, rate="ixg_pg", bins=10):
    d = d.copy()
    d["b"] = pd.qcut(d[rate], bins, duplicates="drop")
    fc = d[feat] - d.groupby("b", observed=True)[feat].transform("mean")
    yc = d.scored - d.groupby("b", observed=True)["scored"].transform("mean")
    return np.corrcoef(fc, yc)[0, 1] if fc.std() else 0.0


def main():
    beh = pd.read_parquet("data/recent_behavior.parquet")
    sc = pd.read_parquet("data/scorer_playergames.parquet")[
        ["game_id", "player_id", "rate_shrunk"]]
    d = beh.merge(sc, on=["game_id", "player_id"], how="left")
    d["rate_shrunk"] = d["rate_shrunk"].fillna(d["rate_shrunk"].mean())
    d = d[d.gp >= 10].copy()
    s = {k: d[d.season == k].copy() for k in TEST}

    # ---- main test: season-averages vs averages + recent play ----
    print("===== Does RECENT PLAY add to SEASON AVERAGES? (out-of-season GBM) =====")
    print(f"  {'train->test':<16}{'model':<22}{'logloss':>9}{'top1':>8}{'top3':>8}")
    agg = {}
    for a, b, nm in [(TEST[0], TEST[1], "fit24->test25"),
                     (TEST[1], TEST[0], "fit25->test24")]:
        _, _, llb, t1b, t3b = fit_eval(s[a], s[b], SEASON_AVG)
        mE, teE, llE, t1E, t3E = fit_eval(s[a], s[b], SEASON_AVG + RECENT_ALL)
        print(f"  {nm:<16}{'season-avg only':<22}{llb:>9.4f}{t1b:>8.3f}{t3b:>8.3f}")
        print(f"  {nm:<16}{'avg + recent play':<22}{llE:>9.4f}{t1E:>8.3f}{t3E:>8.3f}")
        print(f"  {'':<16}{'-> delta':<22}{llE-llb:>+9.4f}{t1E-t1b:>+8.3f}{t3E-t3b:>+8.3f}\n")
        agg[nm] = (mE, s[a], s[b])

    # ---- which behaviors did the data lean on (permutation importance) ----
    a, b, nm = TEST[1], TEST[0], "fit25->test24"
    mE, tr, te = agg[nm]
    feats = SEASON_AVG + RECENT_ALL
    samp = te.sample(min(20000, len(te)), random_state=0)
    imp = permutation_importance(mE, samp[feats].astype(float), samp.scored,
                                 scoring="neg_log_loss", n_repeats=3, random_state=0)
    recent_set = set(RECENT_ALL)
    print(f"===== Top features the data used ({nm}); * = recent-play signal =====")
    for i in np.argsort(imp.importances_mean)[::-1][:18]:
        tag = " *" if feats[i] in recent_set else ""
        print(f"  {feats[i]:<14}{imp.importances_mean[i]:+.5f}{tag}")

    # ---- per-feature stability: which recent-play signals hold both seasons ----
    print("\n===== Recent-play features: partial corr w/ scoring (rate held fixed) =====")
    print(f"  {'feature':<14}{'2024-25':>10}{'2025-26':>10}   stable & sign")
    scored_rows = []
    for f in DEV + CONSIST:
        c1, c2 = partial_corr(s[TEST[0]], f), partial_corr(s[TEST[1]], f)
        stable = np.sign(c1) == np.sign(c2) and min(abs(c1), abs(c2)) > 0.01
        scored_rows.append((f, c1, c2, stable))
    for f, c1, c2, st in sorted(scored_rows, key=lambda x: -abs(x[1] + x[2]) / 2):
        flag = ("STABLE " + ("+" if c1 > 0 else "-")) if st else "noise"
        print(f"  {f:<14}{c1:>+10.4f}{c2:>+10.4f}   {flag}")


if __name__ == "__main__":
    main()
