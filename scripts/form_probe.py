"""Does HOW a player has been playing lately add signal beyond his average?

User's sharpened hypothesis: a player creating chances above his usual level but
not yet scoring (input up, output lagging) is more likely to score. Distinct
from naive goal-streaks because shots/xG are ~10x higher-sample than goals, so
recent chance volume may be a faster, cleaner read on a player's CURRENT level.

Three tests, each controlling for the player's underlying rate and -- the
project's non-negotiable -- requiring the effect to hold with the SAME SIGN in
both test seasons (else it's noise):

  1. the exact 'unlucky/due' test: bucket by recent (xG - goals), within rate
     bands, and read off next-game scoring.
  2. partial correlation of each form feature with scoring, rate held fixed.
  3. out-of-season gate: fit [rate] vs [rate + form] on one season, score the
     OTHER season's logloss. Form earns its keep only if it helps both ways.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

TEST = [20242025, 20252026]


def add_form(d):
    d = d[d.gp >= 10].copy()                      # established within-season only
    d["rc5"] = d["ixg_l5"] / 5.0                  # recent chance rate (xG/gm)
    d["rg5"] = d["g_l5"] / 5.0                    # recent goal rate
    d["rs5"] = d["sh_l5"] / 5.0                   # recent shot rate
    d["ixg_surge"] = d["rc5"] - d["ixg_pg"]       # chances above own baseline
    d["shot_surge"] = d["rs5"] - d["sh_pg"]       # shots above own baseline
    d["luck_gap10"] = d["ixg_l10"] - d["g_l10"]   # created but didn't score (>0 = unlucky)
    d["goal_hot5"] = d["g_l5"] - 5.0 * d["g_pg"]  # scoring above baseline (naive hot hand)
    return d


FORMS = ["ixg_surge", "shot_surge", "luck_gap10", "goal_hot5"]


def partial_corr(d, feat, rate="ixg_pg", bins=10):
    """corr(feat, scored) with rate held fixed (both mean-centered within rate band)."""
    d = d.copy()
    d["band"] = pd.qcut(d[rate], bins, duplicates="drop")
    fc = d[feat] - d.groupby("band", observed=True)[feat].transform("mean")
    yc = d["scored"] - d.groupby("band", observed=True)["scored"].transform("mean")
    return np.corrcoef(fc, yc)[0, 1]


def main():
    df = pd.read_parquet("data/scorer_playergames.parquet")
    df["season"] = df["season"].astype(int)
    d = add_form(df[df.season.isin(TEST)])
    s1, s2 = d[d.season == TEST[0]], d[d.season == TEST[1]]

    # ---- Test 1: the exact 'unlucky/due' table, within rate bands ----
    print("===== TEST 1: recent (xG - goals) vs next-game scoring, rate held fixed =====")
    print("(user's hypothesis => the 'unlucky' top bucket should score MORE)")
    for nm, s in [("2024-25", s1), ("2025-26", s2)]:
        s = s.copy()
        s["band"] = pd.qcut(s["ixg_pg"], 5, labels=False, duplicates="drop")
        s["luck_q"] = s.groupby("band")["luck_gap10"].transform(
            lambda x: pd.qcut(x, 4, labels=["lucky(scored>xG)", "q2", "q3",
                                            "unlucky(xG>scored)"], duplicates="drop"))
        tab = s.groupby("luck_q", observed=True).agg(
            n=("scored", "size"), next_scored=("scored", "mean"),
            rate=("ixg_pg", "mean")).round(4)
        print(f"\n  {nm}:")
        print(tab.to_string())

    # ---- Test 2: partial correlation, per season (sign must match) ----
    print("\n\n===== TEST 2: partial corr(form, scored | rate), per season =====")
    print(f"  {'feature':<14}{'2024-25':>10}{'2025-26':>10}   stable?")
    for f in FORMS:
        c1, c2 = partial_corr(s1, f), partial_corr(s2, f)
        stable = "YES" if np.sign(c1) == np.sign(c2) and min(abs(c1), abs(c2)) > 0.01 else "no"
        print(f"  {f:<14}{c1:>+10.4f}{c2:>+10.4f}   {stable}")

    # ---- Test 3: out-of-season logloss gate ----
    print("\n\n===== TEST 3: out-of-season gate (fit on A, score B's logloss) =====")
    print("  does [rate + form] beat [rate alone] on the OTHER season?")
    base = ["ixg_pg", "rate_shrunk"]
    def fit_eval(tr, te, feats):
        mu, sd = tr[feats].mean(), tr[feats].std().replace(0, 1)
        m = LogisticRegression(max_iter=2000).fit((tr[feats] - mu) / sd, tr["scored"])
        p = m.predict_proba((te[feats] - mu) / sd)[:, 1]
        return log_loss(te["scored"], p, labels=[0, 1]), dict(zip(feats, m.coef_[0]))
    for a, b, nm in [(s1, s2, "fit24->test25"), (s2, s1, "fit25->test24")]:
        llb, _ = fit_eval(a, b, base)
        print(f"\n  {nm}:  base logloss {llb:.5f}")
        for f in FORMS:
            llf, co = fit_eval(a, b, base + [f])
            print(f"    +{f:<12} logloss {llf:.5f}  delta {llf-llb:+.5f}  coef {co[f]:+.4f}"
                  f"  {'(worse)' if llf>llb else '(better)'}")


if __name__ == "__main__":
    main()
