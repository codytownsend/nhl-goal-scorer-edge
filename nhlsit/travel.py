"""Travel / road-trip / altitude context features.

Hockey fatigue isn't just back-to-backs: a team five games into a western road
trip, three time zones from home, is different from one playing at home. We
reconstruct each team's city-to-city itinerary from the schedule and build
home-perspective differentials (positive = tougher on the away side = helps home).
"""
from __future__ import annotations

import math

import pandas as pd

# (lat, lon, tz_offset_hours, altitude_ft) for each club's home arena.
ARENA = {
    "ANA": (33.81, -117.88, -8, 160), "BOS": (42.37, -71.06, -5, 20),
    "BUF": (42.87, -78.87, -5, 600), "CAR": (35.80, -78.72, -5, 315),
    "CBJ": (39.97, -83.00, -5, 700), "CGY": (51.04, -114.07, -7, 3428),
    "CHI": (41.88, -87.67, -6, 590), "COL": (39.75, -105.00, -7, 5280),
    "DAL": (32.79, -96.81, -6, 430), "DET": (42.32, -83.05, -5, 600),
    "EDM": (53.55, -113.50, -7, 2192), "FLA": (26.16, -80.33, -5, 10),
    "LAK": (34.04, -118.27, -8, 285), "MIN": (44.94, -93.10, -6, 830),
    "MTL": (45.50, -73.57, -5, 118), "NJD": (40.73, -74.17, -5, 30),
    "NSH": (36.16, -86.78, -6, 400), "NYI": (40.72, -73.59, -5, 50),
    "NYR": (40.75, -73.99, -5, 30), "OTT": (45.30, -75.93, -5, 200),
    "PHI": (39.90, -75.17, -5, 40), "PIT": (40.44, -79.99, -5, 750),
    "SEA": (47.62, -122.35, -8, 160), "SJS": (37.33, -121.90, -8, 82),
    "STL": (38.63, -90.20, -6, 465), "TBL": (27.94, -82.45, -5, 15),
    "TOR": (43.64, -79.38, -5, 250), "UTA": (40.77, -111.90, -7, 4226),
    "VAN": (49.28, -123.11, -8, 0), "VGK": (36.10, -115.18, -8, 2001),
    "WPG": (49.89, -97.14, -6, 764), "WSH": (38.90, -77.02, -5, 25),
    "ARI": (33.53, -112.26, -7, 1150),
}
HIGH_ALT = 3000   # ft; visiting teams feel COL / CGY / UTA / EDM


def _haversine(a, b):
    (la1, lo1), (la2, lo2) = a, b
    r = 3959.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = math.radians(la2 - la1), math.radians(lo2 - lo1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def travel_features(games_by_season: dict, seasons: list) -> pd.DataFrame:
    rows = []
    for s in seasons:
        g = games_by_season[s].copy()
        g["date"] = pd.to_datetime(g["date"])
        g = g.sort_values(["date", "game_id"])
        # per-team state: last (loc, date), current road-trip length & distance
        last_loc, last_date, trip_len, trip_dist = {}, {}, {}, {}

        def team_game(team, venue_city, date, is_home):
            loc = ARENA[venue_city][:2]
            dist = _haversine(last_loc[team], loc) if team in last_loc else 0.0
            tz = abs(ARENA.get(team, ARENA[venue_city])[2] - ARENA[venue_city][2])
            if is_home:
                trip_len[team] = 0
                trip_dist[team] = 0.0
            else:
                trip_len[team] = trip_len.get(team, 0) + 1
                trip_dist[team] = trip_dist.get(team, 0.0) + dist
            days = (date - last_date[team]).days if team in last_date else 5
            last_loc[team] = loc
            last_date[team] = date
            return {"dist": dist, "trip_len": trip_len[team],
                    "trip_dist": trip_dist[team], "tz": tz, "days": days}

        for r in g.itertuples(index=False):
            venue = r.home if r.home in ARENA else None
            if venue is None:
                continue
            h = team_game(r.home, venue, r.date, True)
            a = team_game(r.away, venue, r.date, False)
            alt = ARENA[venue][3]
            rows.append({
                "game_id": r.game_id,
                # positive = harder on the away team => helps home
                "travel_diff": a["dist"] - h["dist"],
                "roadtrip_len": a["trip_len"],          # away games into the trip
                "roadtrip_dist": a["trip_dist"] / 1000.0,
                "tz_diff": a["tz"] - h["tz"],
                "away_altitude": alt / 1000.0 if alt >= HIGH_ALT else 0.0,
            })
    return pd.DataFrame(rows)
