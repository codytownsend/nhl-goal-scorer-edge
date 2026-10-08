"""Walk-forward goalie quality (GSAx) and per-game starter identification.

Goalie quality = Goals Saved Above Expected per shot faced, on shots ON GOAL
(the shots a goalie can actually save): GSAx = sum(xG_on_goal) - goals_allowed.
Rated per shot and shrunk toward league average (0) with a pseudo-count, carried
across seasons with decay. Everything is walk-forward: a starter's rating before
game g uses only shots they faced before g.

Starter = the goalie who faced the most on-goal shots for a team in that game
(historical proxy for the announced starter; in a live forward test you'd pull
the confirmed starter pregame instead).
"""
from __future__ import annotations

import pandas as pd

ON_GOAL = ("goal", "shot-on-goal")
SHRINK_SHOTS = 250      # pseudo-shots of league-average prior (~half a season)
SEASON_CARRY = 0.80     # keep this fraction of totals across a season boundary


def goalie_game_logs(events: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Per (game, defending team, goalie): shots faced, xG faced, goals allowed.

    A shot row has team=attacker, opponent=defender, goalie_id=defender's goalie.
    """
    og = events[events["event"].isin(ON_GOAL) & events["goalie_id"].notna()].copy()
    grp = og.groupby(["game_id", "opponent", "goalie_id"]).agg(
        shots=("event", "size"),
        xg_faced=("xg", "sum"),
        goals_allowed=("is_goal", "sum"),
    ).reset_index().rename(columns={"opponent": "team"})
    grp["gsax"] = grp["xg_faced"] - grp["goals_allowed"]
    dates = games[["game_id", "date", "home", "away"]]
    grp = grp.merge(dates, on="game_id", how="left")
    return grp


def starters(logs: pd.DataFrame) -> pd.DataFrame:
    """One starter per (game_id, team): the goalie who faced the most shots."""
    idx = logs.groupby(["game_id", "team"])["shots"].idxmax()
    return logs.loc[idx, ["game_id", "team", "goalie_id", "date"]].reset_index(drop=True)


def walk_forward_goalie(events: pd.DataFrame, games_by_season: dict,
                        seasons: list) -> pd.DataFrame:
    """Pre-game GSAx-per-shot rating for each game's starters.

    Returns per game_id: home_starter, away_starter, home_g_rate, away_g_rate,
    goalie_diff (home - away). Positive rate = saves above expected (good).
    """
    all_games = pd.concat([games_by_season[s] for s in seasons], ignore_index=True)
    ev = events[events["season"].astype(str).isin([str(s) for s in seasons])]
    logs = goalie_game_logs(ev, all_games)
    start = starters(logs)

    # chronological game log per goalie, joined to season for carry resets
    logs = logs.merge(all_games[["game_id", "season"]], on="game_id", how="left")
    logs = logs.sort_values(["date", "game_id"]).reset_index(drop=True)

    # accumulate walk-forward per goalie
    cum_gsax, cum_shots, last_season = {}, {}, {}
    pre_rate = {}   # (game_id, goalie_id) -> rate BEFORE this game
    for r in logs.itertuples(index=False):
        gid, go, s = r.game_id, r.goalie_id, r.season
        if go in last_season and last_season[go] != s:
            cum_gsax[go] *= SEASON_CARRY
            cum_shots[go] *= SEASON_CARRY
        g0 = cum_gsax.get(go, 0.0)
        n0 = cum_shots.get(go, 0.0)
        pre_rate[(gid, go)] = g0 / (n0 + SHRINK_SHOTS)     # shrunk toward 0
        cum_gsax[go] = g0 + r.gsax
        cum_shots[go] = n0 + r.shots
        last_season[go] = s

    start["g_rate"] = start.apply(
        lambda x: pre_rate.get((x["game_id"], x["goalie_id"]), 0.0), axis=1)

    home = start.merge(all_games[["game_id", "home"]], on="game_id")
    home = home[home["team"] == home["home"]][["game_id", "goalie_id", "g_rate"]]
    home = home.rename(columns={"goalie_id": "home_starter", "g_rate": "home_g_rate"})
    away = start.merge(all_games[["game_id", "away"]], on="game_id")
    away = away[away["team"] == away["away"]][["game_id", "goalie_id", "g_rate"]]
    away = away.rename(columns={"goalie_id": "away_starter", "g_rate": "away_g_rate"})

    out = home.merge(away, on="game_id", how="outer").fillna(
        {"home_g_rate": 0.0, "away_g_rate": 0.0})
    out["goalie_diff"] = out["home_g_rate"] - out["away_g_rate"]
    return out
