"""Parse ALL play-by-play events (not just shots) for the expected-event-value
model. Each event gets an owner team, oriented coordinates, and a zone, so we
can learn each (event-type, zone) action's +/- goal value from history.
"""
from __future__ import annotations

import pandas as pd

from . import api, situations as S

# Events we assign value to (goal included as the terminal event).
VALUE_EVENTS = {"faceoff", "hit", "giveaway", "takeaway",
                "blocked-shot", "shot-on-goal", "missed-shot", "goal"}


def _zone(x):
    if x is None:
        return "U"
    if x >= 25:
        return "O"
    if x <= -25:
        return "D"
    return "N"


def parse_all_events(game_id: int) -> pd.DataFrame:
    d = api.play_by_play(game_id)
    home, away = d["homeTeam"], d["awayTeam"]
    id2 = {home["id"]: home["abbrev"], away["id"]: away["abbrev"]}
    home_id = home["id"]
    season = d.get("season")
    rows = []
    for p in d.get("plays", []):
        et = p.get("typeDescKey")
        if et not in VALUE_EVENTS:
            continue
        det = p.get("details", {}) or {}
        owner = det.get("eventOwnerTeamId")
        if owner is None:
            continue
        pd_ = p.get("periodDescriptor", {})
        pnum = pd_.get("number", 0)
        ptype = pd_.get("periodType", "REG")
        if ptype == "SO":
            continue
        owner_is_home = owner == home_id
        sip = _sec(p.get("timeInPeriod", "0:00"))
        x, y = S.orient(det.get("xCoord"), det.get("yCoord"),
                        owner_is_home, p.get("homeTeamDefendingSide"))
        state = S.parse_situation(p.get("situationCode", ""), owner_is_home)
        rows.append({
            "game_id": game_id, "season": season,
            "period": pnum, "sip": sip,
            "event": et, "team": id2.get(owner), "is_home": owner_is_home,
            "zone": _zone(x), "strength": S.strength_bucket(state),
            "is_goal": et == "goal",
        })
    return pd.DataFrame(rows)


def _sec(t):
    try:
        m, s = t.split(":")
        return int(m) * 60 + int(s)
    except (ValueError, AttributeError):
        return 0


def next_goal_value(df: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    """For each event, realized outcome = +1 if the owner's team scores the next
    goal within `window` seconds (same period), -1 if the opponent does, else 0.
    Computed per game/period in event order."""
    df = df.sort_values(["game_id", "period", "sip"]).reset_index(drop=True)
    out = []
    for (gid, per), g in df.groupby(["game_id", "period"], sort=False):
        g = g.reset_index()
        goals = g[g.is_goal][["sip", "is_home"]].to_numpy()
        gt = goals[:, 0]
        gh = goals[:, 1]
        for r in g.itertuples(index=False):
            val = 0
            # first goal strictly after this event within the window
            for k in range(len(gt)):
                if gt[k] > r.sip and gt[k] - r.sip <= window:
                    val = 1 if bool(gh[k]) == r.is_home else -1
                    break
                if gt[k] > r.sip + window:
                    break
            out.append(val)
    df = df.copy()
    df["gv"] = out
    return df
