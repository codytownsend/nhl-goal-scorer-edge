"""Walk-forward forward-test: predict goal scorers for every day of the
2026-27 regular season so far, grade against what actually happened.

Leak-free by construction: build_player_games sorts all player-games by date
and computes each row's features from strictly-earlier games only. We train on
2020-26 history, predict every 2026-27 player-game, and compare the per-game
ranking to the real `scored` label (actual goals used ONLY to grade, never as
a feature).

    python -m scripts.grade_season [YYYY-MM-DD cutoff]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from nhlsit import api, parse, xg, lineups as LU, scorer
from scripts.scorer_eval import FEATURES

ROOT = Path(__file__).resolve().parent.parent
HIST = [20202021, 20212022, 20222023, 20232024, 20242025, 20252026]
POS = {"C": "forwards", "L": "forwards", "R": "forwards", "D": "defense"}
SEASON = 20262027


def discover(cutoff: str):
    seen = {}
    for probe in ["2026-09-28", "2026-10-05", "2026-10-12"]:
        for day in api._get(f"/v1/schedule/{probe}").get("gameWeek", []):
            for g in day.get("games", []):
                if g.get("gameType") == 2 and g.get("gameState") in ("OFF", "FINAL") \
                        and day["date"] <= cutoff:
                    seen[g["id"]] = (day["date"], g["homeTeam"]["abbrev"],
                                     g["awayTeam"]["abbrev"])
    return seen


def build_new_season(cutoff: str):
    cache_ev = ROOT / "data" / f"events_{SEASON}.parquet"
    cache_lu = ROOT / "data" / f"lineups_{SEASON}.parquet"
    seen = discover(cutoff)
    ids = sorted(seen)
    print(f"{len(ids)} final reg-season games through {cutoff}")
    games = pd.DataFrame([{"game_id": gid, "date": dt, "season": SEASON,
                           "home": h, "away": a} for gid, (dt, h, a) in seen.items()])
    if cache_ev.exists() and cache_lu.exists():
        return games, pd.read_parquet(cache_ev), pd.read_parquet(cache_lu)
    hist_ev = pd.concat([pd.read_parquet(f"data/events_{s}.parquet") for s in HIST],
                        ignore_index=True)
    xgm = xg.XGModel().fit(hist_ev)
    ev = xg.add_xg(parse.parse_games(ids), model=xgm)
    ev["season"] = SEASON
    lu = LU.build_lineups(ids)
    ev.to_parquet(cache_ev, index=False); lu.to_parquet(cache_lu, index=False)
    return games, ev, lu


def main():
    cutoff = sys.argv[1] if len(sys.argv) > 1 else "2026-10-06"
    gnew, ev_new, lu_new = build_new_season(cutoff)

    events = pd.concat(
        [pd.read_parquet(f"data/events_{s}.parquet") for s in HIST] + [ev_new],
        ignore_index=True)
    lineups = pd.concat(
        [pd.read_parquet(f"data/lineups_{s}.parquet") for s in HIST] + [lu_new],
        ignore_index=True)
    gbs = {}
    for s in HIST:
        g = pd.read_parquet(f"data/games_{s}.parquet"); g["season"] = g["season"].astype(int)
        gbs[s] = g
    gbs[SEASON] = gnew

    feats = scorer.build_player_games(events, lineups, gbs, HIST + [SEASON])
    feats["season"] = feats["season"].astype(int)
    train = feats[feats.season != SEASON]
    slate = feats[feats.season == SEASON].copy()
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0,
        min_samples_leaf=200, random_state=0).fit(
        train[FEATURES].astype(float), train.scored.values)
    slate["p"] = model.predict_proba(slate[FEATURES].astype(float))[:, 1]
    slate["date"] = slate["date"].astype(str)

    def grade(d):
        t1 = []; t3 = []; base = []
        for _, g in d.groupby("game_id"):
            g = g.sort_values("p", ascending=False)
            t1.append(int(g.head(1).scored.iloc[0] == 1))
            t3.append(int(g.head(3).scored.max() >= 1))
            base.append(g.scored.mean())
        return np.mean(t1), np.mean(t3), len(t1), np.mean(base)

    print(f"\n{'date':<12}{'games':>6}{'top-1':>8}{'top-3':>8}{'(base/player)':>15}")
    for dt in sorted(slate.date.unique()):
        d = slate[slate.date == dt]
        a, b, n, br = grade(d)
        print(f"{dt:<12}{n:>6}{a:>8.0%}{b:>8.0%}{br:>14.1%}")
    a, b, n, br = grade(slate)
    print(f"{'OVERALL':<12}{n:>6}{a:>8.0%}{b:>8.0%}{br:>14.1%}")
    print(f"\n(top-1 = our #1 pick scored; top-3 = >=1 of our top-3 scored; "
          f"n={n} games, {len(slate)} player-games)")

    # a little color: show each day's actual top pick vs whether they scored
    print("\nSample — our #1 pick per game, did they score?")
    import collections
    hit = collections.Counter()
    for _, g in slate.groupby("game_id"):
        top = g.sort_values("p", ascending=False).iloc[0]
        hit[int(top.scored == 1)] += 1
    print(f"  #1 picks that scored: {hit[1]}/{hit[0]+hit[1]}")


if __name__ == "__main__":
    main()
