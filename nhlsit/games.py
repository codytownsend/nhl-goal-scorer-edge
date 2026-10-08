"""Build a per-game table (date, teams, final score, winner) from schedules.

Used to order games chronologically for walk-forward prediction and to supply
the actual outcomes we train the margin map / evaluate against.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import api, fetch

ROOT = Path(__file__).resolve().parent.parent


def build_games(season: str, game_types=(2,)) -> pd.DataFrame:
    seen = {}
    for team in fetch.TEAMS:
        try:
            sched = api.club_schedule_season(team, season)
        except Exception:  # noqa: BLE001
            continue
        for g in sched.get("games", []):
            if g.get("gameType") not in game_types:
                continue
            if g.get("gameState") not in ("OFF", "FINAL"):
                continue
            gid = g["id"]
            if gid in seen:
                continue
            h, a = g["homeTeam"], g["awayTeam"]
            hs, as_ = h.get("score"), a.get("score")
            if hs is None or as_ is None:
                continue
            last = (g.get("gameOutcome") or {}).get("lastPeriodType", "REG")
            seen[gid] = {
                "game_id": gid,
                "date": g["gameDate"],
                "season": season,
                "home": h["abbrev"],
                "away": a["abbrev"],
                "home_goals": hs,
                "away_goals": as_,
                "home_margin": hs - as_,          # always +/-1 for OT/SO
                "home_win": int(hs > as_),
                "last_period": last,               # REG / OT / SO
                "extra_time": int(last in ("OT", "SO")),
            }
    df = pd.DataFrame(sorted(seen.values(), key=lambda r: (r["date"], r["game_id"])))
    return df.reset_index(drop=True)


def load_or_build(season: str) -> pd.DataFrame:
    fp = ROOT / "data" / f"games_{season}.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    df = build_games(season)
    df.to_parquet(fp, index=False)
    return df
