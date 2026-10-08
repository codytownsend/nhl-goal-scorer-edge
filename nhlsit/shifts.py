"""Shift-chart fetch + on-ice reconstruction (foundation for player ratings).

For each shot event we need the five skaters on the ice for each team. The
basic play-by-play doesn't give that, but the stats-API shift charts do
(per-player start/end times). We cache raw shift JSON and, per game, build a
function that returns the on-ice skaters for each team at any (period, seconds).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

STATS = "https://api.nhle.com/stats/rest/en"
HEADERS = {"User-Agent": "Mozilla/5.0 (nhlsit research tool)"}
_ROOT = Path(__file__).resolve().parent.parent
CACHE = _ROOT / "data" / "raw"
_session = requests.Session()
_session.headers.update(HEADERS)


def _mmss(s: str) -> int:
    try:
        m, sec = s.split(":")
        return int(m) * 60 + int(sec)
    except (ValueError, AttributeError):
        return 0


def shiftcharts(game_id: int) -> list:
    fp = CACHE / f"shifts_{game_id}.json"
    if fp.exists():
        return json.loads(fp.read_text()).get("data", [])
    url = f"{STATS}/shiftcharts?cayenneExp=gameId={game_id}"
    for attempt in range(3):
        try:
            r = _session.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            fp.write_text(json.dumps(data))
            return data.get("data", [])
        except (requests.RequestException, ValueError):
            time.sleep(1.5 * (attempt + 1))
    return []


def on_ice_index(game_id: int) -> dict:
    """Return {(period, teamAbbrev): [(start_sec, end_sec, playerId), ...]}.

    Only real on-ice shifts (duration > 0); goalie shifts are included but
    filtered later using the boxscore skater set.
    """
    idx = {}
    for s in shiftcharts(game_id):
        if not s.get("duration") or s.get("startTime") is None or s.get("endTime") is None:
            continue
        per = s.get("period")
        team = s.get("teamAbbrev")
        st, en = _mmss(s["startTime"]), _mmss(s["endTime"])
        if en <= st:
            continue
        idx.setdefault((per, team), []).append((st, en, s["playerId"]))
    return idx


def on_ice_at(index: dict, period: int, sec_in_period: int, team: str) -> set:
    """Player ids on the ice for `team` at a moment (seconds into the period)."""
    out = set()
    for st, en, pid in index.get((period, team), []):
        if st <= sec_in_period < en:
            out.add(pid)
    return out
