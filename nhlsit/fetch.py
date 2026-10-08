"""Discover the set of game IDs for a season via team schedules."""
from __future__ import annotations

from . import api

# 32 current NHL club abbreviations.
TEAMS = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET",
    "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT",
    "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK",
    "WPG", "WSH",
]


def season_game_ids(season: str, game_types=(2,), include_future=False):
    """Return sorted unique game IDs for a season (default: regular season).

    season is the 8-digit form, e.g. '20232024'. game_types: 2=regular, 3=playoffs.
    """
    ids = set()
    for team in TEAMS:
        try:
            sched = api.club_schedule_season(team, season)
        except Exception as e:  # noqa: BLE001
            print(f"  !! schedule {team} {season}: {e}")
            continue
        for gm in sched.get("games", []):
            if gm.get("gameType") not in game_types:
                continue
            if not include_future and gm.get("gameState") not in ("OFF", "FINAL"):
                continue
            ids.add(gm["id"])
    return sorted(ids)
