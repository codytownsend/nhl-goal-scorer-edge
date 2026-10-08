"""Upcoming-slate helpers for the live forward test."""
from __future__ import annotations

import pandas as pd

from . import api


def upcoming_games() -> pd.DataFrame:
    """Games in the current schedule week that haven't finished yet."""
    d = api._cached("/v1/schedule/now", "schedule_now", ttl=1800)
    rows = []
    for day in d.get("gameWeek", []):
        for g in day.get("games", []):
            if g.get("gameState") in ("OFF", "FINAL"):
                continue
            rows.append({
                "game_id": g["id"], "date": g.get("gameDate", day.get("date")),
                "home": g["homeTeam"]["abbrev"], "away": g["awayTeam"]["abbrev"],
                "start_utc": g.get("startTimeUTC"), "state": g.get("gameState"),
            })
    return pd.DataFrame(rows)


def results_for_date(date: str) -> pd.DataFrame:
    """Final scores for a date (to grade the ledger)."""
    d = api._cached(f"/v1/score/{date}", f"score_{date}", ttl=1800)
    rows = []
    for g in d.get("games", []):
        if g.get("gameState") not in ("OFF", "FINAL"):
            continue
        rows.append({
            "game_id": g["id"],
            "home": g["homeTeam"]["abbrev"], "away": g["awayTeam"]["abbrev"],
            "home_goals": g["homeTeam"].get("score"),
            "away_goals": g["awayTeam"].get("score"),
        })
    return pd.DataFrame(rows)
