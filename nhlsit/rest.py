"""Rest / back-to-back features from the game schedule.

For each game, days since each team's previous game (within season). The model
feature is a capped rest differential (home - away); a back-to-back = 1 day.
"""
from __future__ import annotations

import pandas as pd

REST_CAP = 3


def rest_features(games_by_season: dict, seasons: list) -> pd.DataFrame:
    rows = []
    for s in seasons:
        g = games_by_season[s].copy()
        g["date"] = pd.to_datetime(g["date"])
        g = g.sort_values(["date", "game_id"])
        last_played = {}
        for r in g.itertuples(index=False):
            hr = (r.date - last_played[r.home]).days if r.home in last_played else 5
            ar = (r.date - last_played[r.away]).days if r.away in last_played else 5
            rows.append({
                "game_id": r.game_id,
                "home_rest": hr, "away_rest": ar,
                "home_b2b": int(hr <= 1), "away_b2b": int(ar <= 1),
                "rest_diff": min(hr, REST_CAP) - min(ar, REST_CAP),
            })
            last_played[r.home] = r.date
            last_played[r.away] = r.date
    return pd.DataFrame(rows)
