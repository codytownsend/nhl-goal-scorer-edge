"""Matchup-adjusted xEV: scale each attacking event's value by the opponent's
(prior-season) defensive allowance in that event/zone, then test predictively.

    python -m scripts.test_matchup_xev
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nhlsit import event_value as EV, events_all as EA, games as G, predict, rating
from scripts.backtest_oos import build_preds

ROOT = Path(__file__).resolve().parent.parent
DEV = ["20202021", "20212022", "20222023", "20232024"]
OOS = ["20242025", "20252026"]
ALL = DEV + OOS
ATTACK = {"shot-on-goal", "missed-shot", "blocked-shot", "takeaway", "faceoff"}
SHRINK = 4000   # heavy: most slices are noisy


def main():
    ae = pd.concat([pd.read_parquet(ROOT / f"data/allevents_{s}.parquet") for s in ALL], ignore_index=True)
    ae["season"] = ae["season"].astype(str)
    gbs = {s: G.load_or_build(s) for s in ALL}
    gm = pd.concat([gbs[s] for s in ALL])[["game_id", "home", "away"]]
    ae = ae.merge(gm, on="game_id")
    ae["defender"] = np.where(ae.is_home, ae.away, ae.home)
    ae["slice"] = ae.event + "_" + ae.zone

    values = EV.learn_values(ae[ae.season.isin(DEV)])          # base value per slice
    aeg = EA.next_goal_value(ae, window=15)                    # realized gv
    lg = aeg[aeg.season.isin(DEV)].groupby("slice").gv.mean().to_dict()

    # prior-season defensive allowance factor per (team, slice): mean gv conceded
    # (opponent attacking events vs team) / league, shrunk toward 1.0
    seq = sorted(set(ALL))
    factor = {}   # (season, team, slice) -> multiplier
    att = aeg[aeg.event.isin(ATTACK)]
    for i, s in enumerate(seq):
        if i == 0:
            continue
        prev = seq[i - 1]
        d = att[att.season == prev]
        agg = d.groupby(["defender", "slice"]).gv.agg(["mean", "size"])
        for (team, sl), row in agg.iterrows():
            base = lg.get(sl, 0.0)
            n = row["size"]
            # shrunk conceded value toward league; factor = conceded/league
            adj = (row["mean"] * n + base * SHRINK) / (n + SHRINK)
            factor[(s, team, sl)] = np.clip(adj / base, 0.5, 2.0) if abs(base) > 1e-6 else 1.0

    # matchup-adjusted per-game team value
    ae["v"] = [values.get((e, z), 0.0) for e, z in zip(ae.event, ae.zone)]
    def fac(r):
        if r.event not in ATTACK:
            return 1.0
        return factor.get((r.season, r.defender, r.slice), 1.0)
    ae["vm"] = ae["v"] * ae.apply(fac, axis=1)
    tv = ae.groupby(["game_id", "team"]).agg(xev=("v", "sum"), xev_m=("vm", "sum")).reset_index()

    # build ratings (base xEV already redundant; here the matchup version)
    def rate(col):
        prior, out = {}, []
        for s in ALL:
            games = gbs[s]
            t = tv[tv.game_id.isin(games.game_id)][["game_id", "team", col]]
            g = games.merge(t.rename(columns={"team": "home", col: "h"}), on=["game_id", "home"], how="left")
            g = g.merge(t.rename(columns={"team": "away", col: "a"}), on=["game_id", "away"], how="left")
            g[["h", "a"]] = g[["h", "a"]].fillna(0.0); g["d"] = g.h - g.a
            teams = sorted(set(g.home) | set(g.away) | set(prior))
            solver = rating.RatingSolver(teams, prior=prior, lam=3.0)
            for _, r in g.iterrows():
                cur = solver.solve()
                out.append({"game_id": r.game_id, "season": s,
                            f"m_{col}": cur["home_adv"] + cur["ratings"].get(r.home, 0) - cur["ratings"].get(r.away, 0)})
                solver.add(r.home, r.away, r["d"])
            prior = {k: 0.75 * v for k, v in solver.solve()["ratings"].items()}
        return pd.DataFrame(out)

    xr = rate("xev_m").merge(rate("xev"), on=["game_id", "season"])
    preds = build_preds().merge(xr, on=["game_id", "season"], how="inner")
    allg = pd.concat([gbs[s] for s in ALL])[["game_id", "home_win", "home_margin"]]
    preds = preds.merge(allg, on="game_id", suffixes=("", "_y"))

    def oos(margin, feats):
        d = preds[preds.season.isin(DEV)]
        mm = predict.MarginMap().fit(d[margin], d.home_margin, {f: d[f].to_numpy() for f in feats} if feats else None)
        te = preds[preds.season.isin(OOS)]
        ex = {f: te[f].to_numpy() for f in feats} if feats else None
        p = np.clip(mm.p_home_win(te[margin], ex), 1e-6, 1 - 1e-6); y = te.home_win.to_numpy()
        return ((mm.margin(te[margin], ex) > 0).astype(int) == y).mean(), \
               -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    print("=== STANDALONE ratings, OOS (2024-26) ===")
    for name, col in [("xG rating", "pred_xg_margin"), ("base xEV", "m_xev"), ("matchup xEV", "m_xev_m")]:
        a, l = oos(col, []); print(f"  {name:14s} acc={a:.3f} logloss={l:.4f}")
    print(f"\ncorr(matchup xEV, xG rating): {preds.m_xev_m.corr(preds.pred_xg_margin):.3f}")
    print(f"corr(matchup xEV, base xEV): {preds.m_xev_m.corr(preds.m_xev):.3f}")

    print("\n=== add matchup xEV to full model, OOS ===")
    FULL = ["goalie_diff", "rest_diff", "st_diff", "linedev_diff", "finish_diff"]
    for c in FULL:
        preds[c] = preds.get(c, 0.0)
    preds[FULL] = preds[FULL].fillna(0.0)
    a, l = oos("pred_xg_margin", FULL); print(f"  full model:          acc={a:.3f} logloss={l:.4f}")
    a, l = oos("pred_xg_margin", FULL + ["m_xev_m"]); print(f"  full + matchup xEV:  acc={a:.3f} logloss={l:.4f}")


if __name__ == "__main__":
    main()
