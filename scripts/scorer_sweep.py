"""Wide feature sweep for the goal-scorer model: throw many candidate signals
at it, let the data say what sticks. Builds team/opponent context features
(record, goal-diff, offense, PK, rest) + a few player splits, then screens them:

  1. OOS permutation importance (what the model actually values)
  2. season-split stability of each feature's univariate signal (guards against
     the single-season mirages this project keeps catching)
  3. model with vs without the new batch (top-1 / top-3 / log-loss)
  4. stage-2 top-2 discrimination with the enriched pair features
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from scripts.scorer_eval import FEATURES as BASE, DEV, TEST

CARRY = 0.5  # cross-season carry for team totals (records partly reset each year)


def team_context(games_by_season, events, seasons):
    """Walk-forward per (game_id, team): record, goal-diff, offense, rest; and
    per (game_id, team) opponent-facing PK xGA. All pre-game, leak-free."""
    g = pd.concat([games_by_season[s] for s in seasons], ignore_index=True)
    g["date"] = pd.to_datetime(g["date"])
    g = g.sort_values(["date", "game_id"]).reset_index(drop=True)

    cum = defaultdict(lambda: dict(n=0.0, pts=0.0, gf=0.0, ga=0.0))
    lastd, lasts = {}, {}
    rows = []
    for r in g.itertuples(index=False):
        for team, gf, ga, won in ((r.home, r.home_goals, r.away_goals, r.home_win == 1),
                                  (r.away, r.away_goals, r.home_goals, r.home_win == 0)):
            c = cum[team]
            if team in lasts and lasts[team] != r.season:
                for k in c: c[k] *= CARRY
            n = c["n"]
            rest = (r.date - lastd[team]).days if team in lasts and lasts[team] == r.season else 3
            rows.append({"game_id": r.game_id, "team": team,
                         "team_winpct": c["pts"] / (2 * n) if n else 0.5,
                         "team_gd_pg": (c["gf"] - c["ga"]) / n if n else 0.0,
                         "team_gf_pg": c["gf"] / n if n else 2.9,
                         "rest_days": min(rest, 5), "b2b": int(rest <= 1)})
            # OT/SO loss = 1 pt
            lost_ot = (not won) and (r.extra_time == 1)
            c["pts"] += 2 if won else (1 if lost_ot else 0)
            c["gf"] += gf; c["ga"] += ga; c["n"] += 1
            lastd[team] = r.date; lasts[team] = r.season
    tc = pd.DataFrame(rows)

    # opponent PK: PP xG allowed per game, walk-forward (defending team = opponent col)
    pp = events[events["strength_bucket"] == "PP"]
    pk = pp.groupby(["game_id", "opponent"]).agg(pkxga=("xg", "sum")).reset_index()
    pk = pk.rename(columns={"opponent": "team"}).merge(
        g[["game_id", "date", "season"]], on="game_id").sort_values(["date", "game_id"])
    cpk = defaultdict(lambda: dict(x=0.0, n=0.0)); lp = {}
    prk = []
    for r in pk.itertuples(index=False):
        c = cpk[r.team]
        if r.team in lp and lp[r.team] != r.season:
            c["x"] *= CARRY; c["n"] *= CARRY
        prk.append({"game_id": r.game_id, "team": r.team,
                    "pk_xga_pg": c["x"] / c["n"] if c["n"] else None})
        c["x"] += r.pkxga; c["n"] += 1; lp[r.team] = r.season
    tc = tc.merge(pd.DataFrame(prk), on=["game_id", "team"], how="left")
    return tc


def enrich(df, games_by_season, events, seasons):
    tc = team_context(games_by_season, events, seasons)
    # player's own team context
    df = df.merge(tc, on=["game_id", "team"], how="left")
    # opponent context (same table, keyed by opp)
    opp = tc.rename(columns={c: "opp_" + c for c in tc.columns if c not in ("game_id", "team")})
    df = df.merge(opp.rename(columns={"team": "opp"}), on=["game_id", "opp"], how="left")
    # player splits
    df["shoot_pct"] = np.where(df.sh_pg > 0, df.g_pg / df.sh_pg, 0.0)
    df["es_ixg_pg"] = (df.ixg_pg - df.pp_ixg_pg).clip(lower=0)
    for c in ["pk_xga_pg", "opp_pk_xga_pg"]:
        df[c] = df[c].fillna(df[c].median())
    return df


NEW = ["team_winpct", "team_gd_pg", "team_gf_pg", "rest_days", "b2b", "pk_xga_pg",
       "opp_team_winpct", "opp_team_gd_pg", "opp_team_gf_pg", "opp_rest_days",
       "opp_b2b", "opp_pk_xga_pg", "shoot_pct", "es_ixg_pg"]
ALL = BASE + NEW


def top1(d, col):
    return np.mean([int(x.sort_values(col, ascending=False).head(1).scored.iloc[0])
                    for _, x in d.groupby("game_id")])


def topk(d, col, k):
    return np.mean([int(x.sort_values(col, ascending=False).head(k).scored.max() >= 1)
                    for _, x in d.groupby("game_id")])


def main():
    df = pd.read_parquet("data/scorer_playergames.parquet"); df["season"] = df.season.astype(int)
    gbs = {s: pd.read_parquet(f"data/games_{s}.parquet") for s in DEV + TEST}
    ev = pd.concat([pd.read_parquet(f"data/events_{s}.parquet") for s in DEV + TEST],
                   ignore_index=True)
    df = enrich(df, gbs, ev, DEV + TEST)
    dev = df[df.season.isin(DEV)].copy(); test = df[df.season.isin(TEST)].copy()

    def fit(feats):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
            l2_regularization=1.0, min_samples_leaf=200, random_state=0)
        return m.fit(dev[feats].astype(float), dev.scored.values)

    m_all = fit(ALL); m_base = fit(BASE)
    test["p_all"] = m_all.predict_proba(test[ALL].astype(float))[:, 1]
    test["p_base"] = m_base.predict_proba(test[BASE].astype(float))[:, 1]
    dev["p_all"] = m_all.predict_proba(dev[ALL].astype(float))[:, 1]

    print("=== MODEL: base vs base+new batch (OOS) ===")
    for lbl, col, feats in [("base", "p_base", BASE), ("base+new", "p_all", ALL)]:
        from sklearn.metrics import log_loss
        ll = log_loss(test.scored, test[col])
        print(f"  {lbl:<10} logloss {ll:.4f}  top1 {top1(test,col):.3f}  "
              f"top3 {topk(test,col,3):.3f}")

    print("\n=== SEASON-SPLIT STABILITY of NEW features (univariate corr w/ scored) ===")
    print(f"  {'feature':<18}{'2024-25':>9}{'2025-26':>9}   flag")
    for f in NEW:
        c1 = test[test.season == 20242025][[f, "scored"]].corr().iloc[0, 1]
        c2 = test[test.season == 20252026][[f, "scored"]].corr().iloc[0, 1]
        flag = "FLIPS (noise)" if c1 * c2 < 0 else ("stable" if abs(c1) > 0.02 else "weak")
        print(f"  {f:<18}{c1:>+9.3f}{c2:>+9.3f}   {flag}")

    print("\n=== OOS permutation importance (all features) ===")
    from sklearn.inspection import permutation_importance
    s = test.sample(min(20000, len(test)), random_state=0)
    imp = permutation_importance(m_all, s[ALL].astype(float), s.scored.values,
                                 scoring="neg_log_loss", n_repeats=3, random_state=0)
    for i in np.argsort(imp.importances_mean)[::-1][:16]:
        tag = "  <-- NEW" if ALL[i] in NEW else ""
        print(f"  {ALL[i]:<18}{imp.importances_mean[i]:+.5f}{tag}")

    # ---- stage-2 with enriched pair features (incl team-record diffs) ----
    print("\n=== STAGE-2 top-2 discrimination, enriched (esp. different-team pairs) ===")
    def pairs(d):
        return [(g.sort_values("p_all", ascending=False).iloc[0],
                 g.sort_values("p_all", ascending=False).iloc[1]) for _, g in d.groupby("game_id")]
    def build(rows):
        X = []; y = []; diff = []
        for a, b in rows:
            if a.scored == b.scored: continue
            fa = a[ALL].to_numpy(float); fb = b[ALL].to_numpy(float)
            X.append(np.concatenate([fa, fb, [a.p_all, b.p_all]]))
            y.append(int(a.scored == 1)); diff.append(a.team != b.team)
        return np.nan_to_num(np.array(X)), np.array(y), np.array(diff)
    Xtr, ytr, _ = build(pairs(dev)); Xte, yte, dte = build(pairs(test))
    m2 = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
        min_samples_leaf=100, random_state=0).fit(Xtr, ytr)
    pa = m2.predict_proba(Xte)[:, 1]
    print(f"  stage-1 picks scorer:  {yte.mean():.3f}")
    print(f"  stage-2 (enriched):    {np.mean((pa>=.5)==(yte==1)):.3f}")
    print(f"  stage-2 on DIFFERENT-team pairs only: "
          f"{np.mean((pa[dte]>=.5)==(yte[dte]==1)):.3f}  (stage-1 {yte[dte].mean():.3f}, n={dte.sum()})")


if __name__ == "__main__":
    main()
