"""Special-teams (PP / PK) quality, walk-forward, with a reliability gate.

A team's power-play xG-for and penalty-kill xG-against (per game) are tracked
pre-game, shrunk toward league average and chained across seasons. The matchup
feature is a net special-teams differential (home - away). Per the spec, run
`reliability_gate` first: if the underlying rate doesn't repeat within season,
it's noise and shouldn't be modeled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SHRINK_GAMES = 20        # pseudo-games of league-average prior
SEASON_CARRY = 0.8


def _per_game_st(events: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Per (game, team): PP xG-for and PK xG-against (shorthanded xGA)."""
    pp = events[events["strength_bucket"] == "PP"]
    ppf = pp.groupby(["game_id", "team"])["xg"].sum().rename("pp_xgf").reset_index()
    # shots allowed while shorthanded = opponent's PP shots
    pka = pp.groupby(["game_id", "opponent"])["xg"].sum().rename("pk_xga").reset_index()
    pka = pka.rename(columns={"opponent": "team"})
    # every team that played each game
    base = pd.concat([
        games[["game_id", "home"]].rename(columns={"home": "team"}),
        games[["game_id", "away"]].rename(columns={"away": "team"}),
    ])
    out = base.merge(ppf, on=["game_id", "team"], how="left") \
              .merge(pka, on=["game_id", "team"], how="left") \
              .merge(games[["game_id", "date", "season"]], on="game_id", how="left")
    out[["pp_xgf", "pk_xga"]] = out[["pp_xgf", "pk_xga"]].fillna(0.0)
    return out.sort_values(["date", "game_id"]).reset_index(drop=True)


def walk_forward_st(events: pd.DataFrame, games_by_season: dict,
                    seasons: list) -> pd.DataFrame:
    """Pre-game net special-teams differential per game (home - away)."""
    all_games = pd.concat([games_by_season[s] for s in seasons], ignore_index=True)
    ev = events[events["season"].astype(str).isin([str(s) for s in seasons])]
    log = _per_game_st(ev, all_games)
    lg_pp = log["pp_xgf"].mean()
    lg_pk = log["pk_xga"].mean()

    cum_pp, cum_pk, cnt, last_season = {}, {}, {}, {}
    pre_st = {}
    for r in log.itertuples(index=False):
        t, s = r.team, r.season
        if t in last_season and last_season[t] != s:
            cum_pp[t] *= SEASON_CARRY
            cum_pk[t] *= SEASON_CARRY
            cnt[t] *= SEASON_CARRY
        n = cnt.get(t, 0.0)
        pp_rate = (cum_pp.get(t, 0.0) + lg_pp * SHRINK_GAMES) / (n + SHRINK_GAMES)
        pk_rate = (cum_pk.get(t, 0.0) + lg_pk * SHRINK_GAMES) / (n + SHRINK_GAMES)
        pre_st[(r.game_id, t)] = pp_rate - pk_rate          # net ST xG/game
        cum_pp[t] = cum_pp.get(t, 0.0) + r.pp_xgf
        cum_pk[t] = cum_pk.get(t, 0.0) + r.pk_xga
        cnt[t] = n + 1
        last_season[t] = s

    rows = []
    for r in all_games.itertuples(index=False):
        rows.append({"game_id": r.game_id,
                     "st_diff": pre_st.get((r.game_id, r.home), 0.0)
                     - pre_st.get((r.game_id, r.away), 0.0)})
    return pd.DataFrame(rows)


def reliability_gate(events: pd.DataFrame, games: pd.DataFrame,
                     metric: str = "pp_xgf") -> float:
    """Split-half repeatability of a per-game ST metric within team-seasons,
    Spearman-Brown adjusted to full length. High -> real skill; ~0 -> noise."""
    log = _per_game_st(events, games)
    log["half"] = log.groupby(["team", "season"]).cumcount() % 2
    piv = log.groupby(["team", "season", "half"])[metric].mean().unstack("half")
    piv = piv.dropna()
    if len(piv) < 10:
        return float("nan")
    r = piv[0].corr(piv[1], method="spearman")
    return float(2 * r / (1 + r)) if r > -1 else float("nan")
