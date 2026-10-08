"""Player on-ice ratings from shot-level xG (Direction B, Stage 1).

For each 5v5 shot we attribute its xG to the five skaters on the ice: the
shooting team's skaters get on-ice xG-for, the defending team's skaters get
on-ice xG-against. A player's rating is his on-ice xG share (xGF%), shrunk toward
league average. Raw (not yet teammate/opponent-adjusted — that's RAPM, a later
step); this Stage-1 version is to prove the pipeline and sanity-check names.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import shifts

SHRINK = 200.0   # pseudo shot-xG toward league (0.5 share)


def build_player_onice(game_ids, events: pd.DataFrame, lineups: pd.DataFrame,
                       strength="5v5", progress=True) -> pd.DataFrame:
    """Accumulate on-ice xGF / xGA per player across games."""
    skaters_by_game = {gid: set(g.player_id) for gid, g in lineups.groupby("game_id")}
    ev = events[(events.strength_bucket == strength) & events.is_unblocked].copy()
    ev["sip"] = ev["game_seconds"] - (ev["period"] - 1) * 1200
    xgf = {}  # player -> on-ice xG for
    xga = {}  # player -> on-ice xG against
    ids = list(game_ids)
    for i, gid in enumerate(ids, 1):
        if progress and (i % 250 == 0 or i == len(ids)):
            print(f"  onice {i}/{len(ids)}", flush=True)
        idx = shifts.on_ice_index(gid)
        if not idx:
            continue
        skset = skaters_by_game.get(gid, set())
        g = ev[ev.game_id == gid]
        for r in g.itertuples(index=False):
            off = shifts.on_ice_at(idx, r.period, int(r.sip), r.team) & skset
            deff = shifts.on_ice_at(idx, r.period, int(r.sip), r.opponent) & skset
            for p in off:
                xgf[p] = xgf.get(p, 0.0) + r.xg
            for p in deff:
                xga[p] = xga.get(p, 0.0) + r.xg
    players = set(xgf) | set(xga)
    rows = []
    for p in players:
        f, a = xgf.get(p, 0.0), xga.get(p, 0.0)
        rows.append({"player_id": p, "on_xgf": f, "on_xga": a,
                     "xg_share": (f + 0.5 * SHRINK) / (f + a + SHRINK)})
    return pd.DataFrame(rows)


def player_game_onice(game_ids, events: pd.DataFrame, lineups: pd.DataFrame,
                      strength="5v5", progress=True) -> pd.DataFrame:
    """Per (game, player): this game's on-ice xG-for / against. Building block
    for walk-forward player ratings (accumulate these chronologically)."""
    skaters_by_game = {gid: set(g.player_id) for gid, g in lineups.groupby("game_id")}
    ev = events[(events.strength_bucket == strength) & events.is_unblocked].copy()
    ev["sip"] = ev["game_seconds"] - (ev["period"] - 1) * 1200
    rows = []
    ids = set(game_ids)
    groups = [(gid, g) for gid, g in ev.groupby("game_id") if gid in ids]
    for i, (gid, g) in enumerate(groups, 1):
        if progress and (i % 500 == 0 or i == len(groups)):
            print(f"  onice {i}/{len(groups)}", flush=True)
        idx = shifts.on_ice_index(gid)
        if not idx:
            continue
        skset = skaters_by_game.get(gid, set())
        f, a = {}, {}
        for r in g.itertuples(index=False):
            for p in shifts.on_ice_at(idx, r.period, int(r.sip), r.team) & skset:
                f[p] = f.get(p, 0.0) + r.xg
            for p in shifts.on_ice_at(idx, r.period, int(r.sip), r.opponent) & skset:
                a[p] = a.get(p, 0.0) + r.xg
        for p in set(f) | set(a):
            rows.append({"game_id": gid, "player_id": p,
                         "g_xgf": f.get(p, 0.0), "g_xga": a.get(p, 0.0)})
    return pd.DataFrame(rows)


def walk_forward_quality(pg_onice: pd.DataFrame, order: pd.DataFrame,
                         shrink=200.0, carry=0.8) -> dict:
    """Pre-game on-ice xG-share per (game, player), walk-forward & leak-free.

    order: game_id, date, season (chronological). Returns {(game_id, player): share}.
    """
    pg = pg_onice.merge(order[["game_id", "date", "season"]], on="game_id")
    pg = pg.sort_values(["date", "game_id"])
    cf, ca, last_s = {}, {}, {}
    pre = {}
    for r in pg.itertuples(index=False):
        p = r.player_id
        if p in last_s and last_s[p] != r.season:
            cf[p] *= carry; ca[p] *= carry
        f, a = cf.get(p, 0.0), ca.get(p, 0.0)
        pre[(r.game_id, p)] = (f + 0.5 * shrink) / (f + a + shrink)
        cf[p] = f + r.g_xgf; ca[p] = a + r.g_xga; last_s[p] = r.season
    return pre


def id_to_name(game_ids) -> dict:
    """Player id -> name, harvested from cached shift charts."""
    names = {}
    for gid in game_ids:
        for s in shifts.shiftcharts(gid):
            pid = s.get("playerId")
            if pid and pid not in names:
                names[pid] = f"{s.get('firstName','')} {s.get('lastName','')}".strip()
    return names
