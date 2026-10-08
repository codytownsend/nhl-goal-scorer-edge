"""Per-game top-3 goal% and point% (ensemble) for an upcoming slate.

Today's games haven't started, so no dressed lineups are posted. We rank each
team's full active roster instead (the top-3 are stars, never the scratches, so
this is robust for the top of the board -- a late scratch is the only risk).
Uses the rate+GRU ensemble from predict_ensemble_date.

Usage: python -m scripts.predict_slate_today 2026-10-08
"""
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd

from nhlsit import api
from scripts.predict_ensemble_date import compute_slate

POS = {"C": "forwards", "L": "forwards", "R": "forwards", "D": "defense"}


def fetch_slate_roster(date: str):
    """(games_df, roster-lineup_df, name_map) using current team rosters."""
    sched = api._get(f"/v1/schedule/{date}")
    day = next(d for d in sched["gameWeek"] if d["date"] == date)
    season = int(str(day["games"][0]["id"])[:4])
    season = int(f"{season}{season + 1}")
    grows, lrows, names = [], [], {}
    roster_cache = {}
    for g in day["games"]:
        gid, home, away = g["id"], g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]
        grows.append({"game_id": gid, "date": date, "season": season,
                      "home": home, "away": away})
        for team in (home, away):
            if team not in roster_cache:
                roster_cache[team] = api._get(f"/v1/roster/{team}/current")
            r = roster_cache[team]
            for grp, key in (("forwards", "forwards"), ("defensemen", "defense")):
                for p in r.get(grp, []):
                    pid = p["id"]
                    names[pid] = f"{p['firstName']['default'][0]}. {p['lastName']['default']}"
                    lrows.append({"game_id": gid, "team": team, "player_id": pid,
                                  "pos": key, "toi_sec": 0, "points": 0})
    return pd.DataFrame(grows), pd.DataFrame(lrows), names


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else "2026-10-08"
    gnew, lnew, names = fetch_slate_roster(date)
    print(f"\n{date}: {len(gnew)} games, roster-based candidates "
          f"(lineups not yet posted)\n")
    slate = compute_slate(gnew, lnew, names)

    order = [(r.game_id, r.away, r.home) for r in gnew.itertuples(index=False)]
    rows, games_json = [], []
    for gid, away, home in order:
        g = slate[slate.game_id == gid].sort_values("ens_g", ascending=False)
        print(f"=== {away} @ {home} ===")
        print(f"   {'player':<20}{'team':>4}{'P(goal)':>9}{'P(point)':>10}")
        for rank, r in enumerate(g.head(3).itertuples(index=False), 1):
            print(f"   {str(r.name):<20}{r.team:>4}{r.ens_g:>8.1%}{r.ens_p:>10.1%}")
            rows.append({"date": date, "game": f"{away}@{home}", "rank": rank,
                         "player": r.name, "team": r.team,
                         "p_goal": round(float(r.ens_g), 4),
                         "p_point": round(float(r.ens_p), 4)})
        print()
        # richer per-game board for the dashboard (top 15, odds slots to fill later)
        players = [{"name": r.name, "team": r.team,
                    "p_goal": round(float(r.ens_g), 4),
                    "p_point": round(float(r.ens_p), 4),
                    "vegas_odds": None, "vegas_prob": None,
                    "edge": None, "ev": None}
                   for r in g.head(15).itertuples(index=False)]
        games_json.append({"game": f"{away}@{home}", "away": away, "home": home,
                           "players": players})

    pd.DataFrame(rows).to_csv(f"predictions_{date}.csv", index=False)
    doc = {"date": date, "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
           "odds_source": None, "games": games_json}
    with open(f"predictions_{date}.json", "w") as fh:
        json.dump(doc, fh, indent=2)
    print(f"wrote predictions_{date}.csv and predictions_{date}.json "
          f"({len(games_json)} games)")


if __name__ == "__main__":
    main()
