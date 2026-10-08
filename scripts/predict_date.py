"""Rank most-likely goal scorers for every game on a given date.

Leak-free: a player's rating comes ONLY from his prior-season-and-earlier
history (walk-forward, carried across the season boundary). The ONLY thing
taken from the target date is the identity of who dressed (known at puck drop)
and the home/away/opponent of each game — zero in-game performance.

Usage: python -m scripts.predict_date 2026-10-06
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from nhlsit import api, scorer
from scripts.scorer_eval import FEATURES

HIST = [20202021, 20212022, 20222023, 20232024, 20242025, 20252026]
POS = {"C": "forwards", "L": "forwards", "R": "forwards", "D": "defense"}


def fetch_slate(date: str):
    """Return (games_df, lineup_rows_df, name_map) for all games on `date`."""
    sched = api._get(f"/v1/schedule/{date}")
    day = next(d for d in sched["gameWeek"] if d["date"] == date)
    season = int(str(day["games"][0]["id"])[:4])
    season = int(f"{season}{season+1}")
    grows, lrows, names = [], [], {}
    for g in day["games"]:
        gid = g["id"]; home = g["homeTeam"]["abbrev"]; away = g["awayTeam"]["abbrev"]
        grows.append({"game_id": gid, "date": date, "season": season,
                      "home": home, "away": away})
        box = api.boxscore(gid)["playerByGameStats"]
        for side, team in (("homeTeam", home), ("awayTeam", away)):
            for grp in ("forwards", "defense"):
                for p in box[side][grp]:
                    pid = p["playerId"]
                    names[pid] = p["name"]["default"]
                    lrows.append({"game_id": gid, "team": team, "player_id": pid,
                                  "pos": POS.get(p["position"], "forwards"),
                                  "toi_sec": 0, "points": 0})
    return pd.DataFrame(grows), pd.DataFrame(lrows), names


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else "2026-10-06"
    gnew, lnew, names = fetch_slate(date)
    season_new = int(gnew.season.iloc[0])
    print(f"\n{date}: {len(gnew)} games, {len(lnew)} dressed skaters "
          f"(season {season_new}, rated from <= 2025-26 history)\n")

    events = pd.concat([pd.read_parquet(f"data/events_{s}.parquet") for s in HIST],
                       ignore_index=True)
    lineups = pd.concat([pd.read_parquet(f"data/lineups_{s}.parquet") for s in HIST]
                        + [lnew], ignore_index=True)
    gbs = {}
    for s in HIST:
        g = pd.read_parquet(f"data/games_{s}.parquet")
        g["season"] = g["season"].astype(int)
        gbs[s] = g
    gbs[season_new] = gnew
    seasons = HIST + [season_new]

    feats = scorer.build_player_games(events, lineups, gbs, seasons)
    feats["season"] = feats["season"].astype(int)
    train = feats[feats.season != season_new]
    slate = feats[feats.season == season_new].copy()

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0,
        min_samples_leaf=200, random_state=0).fit(
        train[FEATURES].astype(float), train.scored.values)
    slate["p"] = model.predict_proba(slate[FEATURES].astype(float))[:, 1]
    slate["name"] = slate.player_id.map(names)

    for g in gnew.itertuples(index=False):
        s = slate[slate.game_id == g.game_id].sort_values("p", ascending=False)
        print(f"=== {g.away} @ {g.home} ===")
        print(f"   {'player':<22}{'team':>5}{'P(goal)':>9}")
        for r in s.head(6).itertuples(index=False):
            star = " *" if r.is_def else ""
            print(f"   {str(r.name):<22}{r.team:>5}{r.p:>8.1%}{star}")
        print()


if __name__ == "__main__":
    main()
