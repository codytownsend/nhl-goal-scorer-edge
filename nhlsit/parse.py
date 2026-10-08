"""Flatten NHL play-by-play into a tidy shot-attempt event table.

One row per shot attempt (Corsi event: goal, shot-on-goal, missed-shot,
blocked-shot), tagged with the situational dimensions we compare teams on.
"""
from __future__ import annotations

import pandas as pd

from . import api, situations as S

# Corsi = all shot attempts.
SHOT_TYPES = {"goal", "shot-on-goal", "missed-shot", "blocked-shot"}
# Unblocked (Fenwick) = has a real shooter/location for xG.
UNBLOCKED = {"goal", "shot-on-goal", "missed-shot"}


def parse_game(game_id: int) -> pd.DataFrame:
    d = api.play_by_play(game_id)
    home = d["homeTeam"]
    away = d["awayTeam"]
    id2abbrev = {home["id"]: home["abbrev"], away["id"]: away["abbrev"]}
    home_id = home["id"]
    season = d.get("season")
    game_type = d.get("gameType")  # 2 = regular season, 3 = playoffs

    rows = []
    home_goals = away_goals = 0
    prev = {"secs": None, "type": None, "owner": None, "zone": None, "period": None}

    for p in d.get("plays", []):
        etype = p.get("typeDescKey")
        pd_ = p.get("periodDescriptor", {})
        pnum = pd_.get("number", 0)
        ptype = pd_.get("periodType", "REG")
        det = p.get("details", {}) or {}
        secs = S.time_to_seconds(p.get("timeInPeriod", "0:00"), pnum)

        # snapshot of the previous play (any type) BEFORE we overwrite it
        prev_snapshot = dict(prev)
        # update running "previous play" for the next iteration
        prev = {"secs": secs, "type": etype, "owner": det.get("eventOwnerTeamId"),
                "zone": det.get("zoneCode"), "period": pnum}

        # Track running score BEFORE recording the shot's score-state, then
        # apply this event if it's a (non-shootout) goal.
        if etype not in SHOT_TYPES:
            if etype == "goal" and ptype != "SO":
                owner = det.get("eventOwnerTeamId")
                if owner == home_id:
                    home_goals += 1
                else:
                    away_goals += 1
            continue

        # In this API `eventOwnerTeamId` is the SHOOTING (attacking) team for
        # every shot type, including blocked-shot (the block coordinate sits in
        # the attacker's offensive zone; `shootingPlayerId` confirms ownership).
        attacker_id = det.get("eventOwnerTeamId")
        if attacker_id is None:
            continue
        owner_is_home = attacker_id == home_id

        # Score differential from the attacking team's perspective (pre-event).
        diff = (home_goals - away_goals) if owner_is_home else (away_goals - home_goals)

        x_raw, y_raw = det.get("xCoord"), det.get("yCoord")
        x, y = S.orient(x_raw, y_raw, owner_is_home, p.get("homeTeamDefendingSide"))
        dist, angle = S.distance_angle(x, y)

        state = S.parse_situation(p.get("situationCode", ""), owner_is_home)

        # rebound / rush context from the previous play (same period only)
        gap = None
        is_rebound = is_rush = False
        if prev_snapshot["secs"] is not None and prev_snapshot["period"] == pnum:
            gap = secs - prev_snapshot["secs"]
            if 0 <= gap <= 3 and prev_snapshot["type"] in UNBLOCKED \
                    and prev_snapshot["owner"] == attacker_id:
                is_rebound = True          # quick 2nd shot by same team
            if 0 <= gap <= 4 and prev_snapshot["zone"] in ("N", "D"):
                is_rush = True             # shot soon after neutral/def-zone play

        rows.append({
            "game_id": game_id,
            "season": season,
            "game_type": game_type,
            "team": id2abbrev.get(attacker_id, str(attacker_id)),
            "opponent": id2abbrev.get(
                away["id"] if attacker_id == home_id else home_id),
            "is_home": owner_is_home,
            "event": etype,
            "is_shot_attempt": True,
            "is_unblocked": etype in UNBLOCKED,
            "is_goal": etype == "goal",
            "period": pnum,
            "period_label": S.period_label(pnum, ptype),
            "game_seconds": S.time_to_seconds(p.get("timeInPeriod", "0:00"), pnum),
            "strength": state,
            "strength_bucket": S.strength_bucket(state),
            "score_state": S.score_state(diff),
            "x": x, "y": y,
            "distance": dist, "angle": angle,
            "danger": S.danger_zone(x, y),
            "shot_type": det.get("shotType"),
            "shooter_id": det.get("shootingPlayerId") or det.get("scoringPlayerId"),
            "goalie_id": det.get("goalieInNetId"),
            "secs_since_last": gap,
            "is_rebound": is_rebound,
            "is_rush": is_rush,
        })

    return pd.DataFrame(rows)


def parse_games(game_ids, progress=True) -> pd.DataFrame:
    frames = []
    n = len(game_ids)
    for i, gid in enumerate(game_ids, 1):
        if progress and (i % 25 == 0 or i == n):
            print(f"  parsed {i}/{n} games")
        try:
            frames.append(parse_game(gid))
        except Exception as e:  # noqa: BLE001
            print(f"  !! skip game {gid}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
