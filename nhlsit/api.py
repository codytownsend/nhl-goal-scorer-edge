"""Thin HTTP client for the NHL web API with on-disk JSON caching.

Base: https://api-web.nhle.com  (ref: github.com/Zmalski/NHL-API-Reference)
The API 403s without a browser-ish User-Agent.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

BASE = "https://api-web.nhle.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (nhlsit research tool)"}

_ROOT = Path(__file__).resolve().parent.parent
CACHE = _ROOT / "data" / "raw"
CACHE.mkdir(parents=True, exist_ok=True)

_session = requests.Session()
_session.headers.update(HEADERS)


def _get(path: str, timeout: int = 30, retries: int = 3) -> dict:
    url = f"{BASE}{path}"
    last = None
    for attempt in range(retries):
        try:
            r = _session.get(url, timeout=timeout)
            if r.status_code == 404:
                raise FileNotFoundError(f"404: {url}")
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:  # noqa: PERF203
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url} ({last})")


def _cached(path: str, cache_name: str, ttl: float | None = None) -> dict:
    """Fetch `path`, caching under data/raw/<cache_name>.json.

    ttl=None means cache forever (good for completed games). Set a ttl
    (seconds) for volatile endpoints like schedules.
    """
    fp = CACHE / f"{cache_name}.json"
    if fp.exists() and (ttl is None or (time.time() - fp.stat().st_mtime) < ttl):
        return json.loads(fp.read_text())
    data = _get(path)
    fp.write_text(json.dumps(data))
    return data


# ---- endpoints -----------------------------------------------------------

def play_by_play(game_id: int) -> dict:
    return _cached(f"/v1/gamecenter/{game_id}/play-by-play", f"pbp_{game_id}")


def boxscore(game_id: int) -> dict:
    return _cached(f"/v1/gamecenter/{game_id}/boxscore", f"box_{game_id}")


def club_schedule_season(team: str, season: str) -> dict:
    # volatile if season in progress; cache 6h
    return _cached(
        f"/v1/club-schedule-season/{team}/{season}",
        f"sched_{team}_{season}",
        ttl=6 * 3600,
    )


def standings_now() -> dict:
    return _cached("/v1/standings/now", "standings_now", ttl=3600)
