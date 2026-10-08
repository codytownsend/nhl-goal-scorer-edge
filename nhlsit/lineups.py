"""Per-game dressed lineups + ice time from the boxscore.

This is the raw material for the Track-B "who's actually playing tonight"
signal: for every game we get each skater who dressed and their TOI, so we can
detect when a team is missing key players (injury/scratch) relative to its norm.
"""
from __future__ import annotations

import pandas as pd

from . import api


def _toi_sec(s: str) -> int:
    try:
        m, sec = s.split(":")
        return int(m) * 60 + int(sec)
    except (ValueError, AttributeError):
        return 0


def parse_boxscore(game_id: int) -> pd.DataFrame:
    d = api.boxscore(game_id)
    pbg = d.get("playerByGameStats", {})
    rows = []
    for side, teamkey in (("homeTeam", "homeTeam"), ("awayTeam", "awayTeam")):
        abbr = d[teamkey]["abbrev"]
        grp = pbg.get(side, {})
        for pos_group in ("forwards", "defense"):        # skaters only
            for p in grp.get(pos_group, []):
                rows.append({
                    "game_id": game_id,
                    "team": abbr,
                    "player_id": p.get("playerId"),
                    "pos": pos_group,
                    "toi_sec": _toi_sec(p.get("toi", "0:00")),
                    "points": p.get("points", 0),
                })
    return pd.DataFrame(rows)


def build_lineups(game_ids, progress=True) -> pd.DataFrame:
    frames = []
    n = len(game_ids)
    for i, gid in enumerate(game_ids, 1):
        if progress and (i % 250 == 0 or i == n):
            print(f"  boxscore {i}/{n}", flush=True)
        try:
            frames.append(parse_boxscore(gid))
        except Exception as e:  # noqa: BLE001
            if progress:
                print(f"  !! skip {gid}: {e}", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
