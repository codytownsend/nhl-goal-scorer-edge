"""Expected event value (xEV): learn each action's +/- goal value from history,
then build opponent-adjusted team ratings on total event value — the same way
we rate teams on xG, but valuing every event, not just shots.
"""
from __future__ import annotations

import pandas as pd

from . import events_all as EA, rating


def learn_values(all_events: pd.DataFrame, window: int = 15,
                 min_n: int = 300) -> dict:
    """Learn value[(event, zone)] = mean realized next-goal outcome (P own goal
    next - P opp goal next) within `window`s. Sparse cells fall back to 0."""
    df = EA.next_goal_value(all_events, window=window)
    grp = df.groupby(["event", "zone"]).gv.agg(["mean", "size"])
    return {k: (float(v["mean"]) if v["size"] >= min_n else 0.0)
            for k, v in grp.iterrows()}


def assign_and_aggregate(all_events: pd.DataFrame, values: dict) -> pd.DataFrame:
    """Per (game_id, team): total expected event value generated."""
    v = all_events.copy()
    v["value"] = [values.get((e, z), 0.0) for e, z in zip(v.event, v.zone)]
    return v.groupby(["game_id", "team"])["value"].sum().rename("xev").reset_index()


def game_xev_table(team_xev: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    g = games.merge(team_xev.rename(columns={"team": "home", "xev": "home_xev"}),
                    on=["game_id", "home"], how="left")
    g = g.merge(team_xev.rename(columns={"team": "away", "xev": "away_xev"}),
                on=["game_id", "away"], how="left")
    g[["home_xev", "away_xev"]] = g[["home_xev", "away_xev"]].fillna(0.0)
    g["home_xev_diff"] = g["home_xev"] - g["away_xev"]
    return g


def walk_forward_xev(all_events: pd.DataFrame, games_by_season: dict, seasons: list,
                     values: dict, carry=0.75, lam=3.0) -> pd.DataFrame:
    """Opponent-adjusted, walk-forward team rating on expected event value.
    Mirrors rating.walk_forward but uses per-game xEV differential."""
    team_xev = assign_and_aggregate(all_events, values)
    prior, out = {}, []
    for season in seasons:
        games = games_by_season[season]
        gx = game_xev_table(team_xev[team_xev.game_id.isin(games.game_id)], games)
        teams = sorted(set(gx.home) | set(gx.away) | set(prior))
        solver = rating.RatingSolver(teams, prior=prior, lam=lam)
        for _, r in gx.iterrows():
            cur = solver.solve()
            rh = cur["ratings"].get(r["home"], 0.0)
            ra = cur["ratings"].get(r["away"], 0.0)
            out.append({"game_id": r["game_id"], "season": season,
                        "pred_xev_margin": cur["home_adv"] + rh - ra})
            solver.add(r["home"], r["away"], r["home_xev_diff"])
        prior = {t: carry * v for t, v in solver.solve()["ratings"].items()}
    return pd.DataFrame(out)
