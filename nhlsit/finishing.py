"""Roster finishing / conversion talent (the genuinely-new component).

xG measures chance *quality*; it assumes league-average conversion. This layer
adds who is actually *shooting*: each player's historical goals-above-expected
per shot (walk-forward, heavily shrunk toward 0 since finishing is noisy),
aggregated over the dressed roster weighted by shot volume. Feature = home
roster finishing - away roster finishing.
"""
from __future__ import annotations

import pandas as pd

SHOT_PC = 300.0    # pseudo-shots shrinking a player's finishing toward league (0)
CARRY = 0.7        # cross-season decay of a player's finishing history
VOL_DEFAULT = 1.0  # shots/game prior for players with no history


def walk_forward_finishing(events: pd.DataFrame, lineups: pd.DataFrame,
                           games_by_season: dict, seasons: list) -> pd.DataFrame:
    allg = pd.concat([games_by_season[s] for s in seasons], ignore_index=True)
    allg = allg[["game_id", "date", "season", "home", "away"]].copy()
    allg["date"] = pd.to_datetime(allg["date"])

    ev = events[events["is_unblocked"] & events["shooter_id"].notna()].copy()
    ev["gax"] = ev["is_goal"].astype(float) - ev["xg"]
    pg = ev.groupby(["game_id", "shooter_id"]).agg(
        shots=("gax", "size"), gax=("gax", "sum")).reset_index()
    pg = pg.merge(allg[["game_id", "date", "season"]], on="game_id")
    pg = pg.sort_values(["date", "game_id"])

    # walk-forward per player: pre-game finishing rate + shots/game volume
    cum_gax, cum_sh, games, last_s = {}, {}, {}, {}
    finish, vol = {}, {}
    for r in pg.itertuples(index=False):
        p = r.shooter_id
        if p in last_s and last_s[p] != r.season:
            cum_gax[p] *= CARRY; cum_sh[p] *= CARRY; games[p] *= CARRY
        g = cum_gax.get(p, 0.0); s = cum_sh.get(p, 0.0); n = games.get(p, 0.0)
        finish[(r.game_id, p)] = g / (s + SHOT_PC)              # goals above x / shot, shrunk
        vol[(r.game_id, p)] = s / n if n > 0 else VOL_DEFAULT   # shots per game
        cum_gax[p] = g + r.gax; cum_sh[p] = s + r.shots; games[p] = n + 1; last_s[p] = r.season

    # aggregate over dressed skaters, volume-weighted
    lu = lineups[["game_id", "team", "player_id"]].copy()
    lu["f"] = lu.apply(lambda x: finish.get((x["game_id"], x["player_id"]), 0.0), axis=1)
    lu["v"] = lu.apply(lambda x: vol.get((x["game_id"], x["player_id"]), VOL_DEFAULT), axis=1)
    team = lu.assign(fw=lu.f * lu.v).groupby(["game_id", "team"]).agg(
        fw=("fw", "sum"), v=("v", "sum")).reset_index()
    team["team_finish"] = team.fw / team.v          # shot-volume-weighted finishing rate
    tf = dict(zip(zip(team.game_id, team.team), team.team_finish))

    return pd.DataFrame([{"game_id": r.game_id,
                          "finish_diff": tf.get((r.game_id, r.home), 0.0)
                          - tf.get((r.game_id, r.away), 0.0)}
                         for r in allg.itertuples(index=False)])
