"""Situational aggregation and team comparison.

Each event row is one shot ATTEMPT with team = attacker, opponent = defender.
So a team's offense = rows where team == X; its defense = rows where
opponent == X. Everything here is share-based or per-game, so it needs no
time-on-ice data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _games_played(events: pd.DataFrame, team: str) -> int:
    m = (events["team"] == team) | (events["opponent"] == team)
    return events.loc[m, "game_id"].nunique()


def _metrics(for_df: pd.DataFrame, against_df: pd.DataFrame, games: int) -> dict:
    cf, ca = len(for_df), len(against_df)
    xgf, xga = for_df["xg"].sum(), against_df["xg"].sum()
    gf, ga = int(for_df["is_goal"].sum()), int(against_df["is_goal"].sum())
    hf = (for_df["danger"] == "high").sum()
    ha = (against_df["danger"] == "high").sum()
    g = max(games, 1)
    return {
        "GP": games,
        "CF/g": round(cf / g, 1), "CA/g": round(ca / g, 1),
        "CF%": round(100 * cf / max(cf + ca, 1), 1),
        "xGF/g": round(xgf / g, 2), "xGA/g": round(xga / g, 2),
        "xGF%": round(100 * xgf / max(xgf + xga, 1e-9), 1),
        "GF/g": round(gf / g, 2), "GA/g": round(ga / g, 2),
        "HDCF/g": round(hf / g, 1), "HDCA/g": round(ha / g, 1),
        "HDCF%": round(100 * hf / max(hf + ha, 1), 1),
        "Sh%": round(100 * gf / max(len(for_df[for_df.is_unblocked]), 1), 1),
    }


def team_profile(events: pd.DataFrame, team: str, by: str = "strength_bucket",
                 situation: dict | None = None) -> pd.DataFrame:
    """For/against metrics for one team, split by dimension `by`.

    `situation` optionally filters events, e.g. {"strength_bucket": "5v5"}.
    """
    df = _filter(events, situation)
    games = _games_played(events, team)
    rows = []
    for key, _ in sorted(df.groupby(by)):
        fdf = df[(df["team"] == team) & (df[by] == key)]
        adf = df[(df["opponent"] == team) & (df[by] == key)]
        if len(fdf) + len(adf) == 0:
            continue
        rows.append({by: key, **_metrics(fdf, adf, games)})
    out = pd.DataFrame(rows).set_index(by)
    return out


def compare(events: pd.DataFrame, team_a: str, team_b: str,
            situation: dict | None = None,
            metrics=("CF%", "xGF%", "xGF/g", "xGA/g", "HDCF%", "GF/g", "GA/g")):
    """One-row-per-team headline comparison under a fixed situation filter."""
    df = _filter(events, situation)
    rows = {}
    for t in (team_a, team_b):
        fdf = df[df["team"] == t]
        adf = df[df["opponent"] == t]
        rows[t] = _metrics(fdf, adf, _games_played(events, t))
    tbl = pd.DataFrame(rows).T
    return tbl[list(metrics)] if metrics else tbl


def league_table(events: pd.DataFrame, metric: str = "xGF%",
                 situation: dict | None = None,
                 by: str | None = None) -> pd.DataFrame:
    """Rank every team on one metric (optionally within a situation)."""
    df = _filter(events, situation)
    teams = sorted(set(df["team"]) | set(df["opponent"]))
    rows = []
    for t in teams:
        fdf = df[df["team"] == t]
        adf = df[df["opponent"] == t]
        if len(fdf) + len(adf) == 0:
            continue
        rows.append({"team": t, **_metrics(fdf, adf, _games_played(events, t))})
    tbl = pd.DataFrame(rows).set_index("team")
    return tbl.sort_values(metric, ascending=False)


def danger_mix(events: pd.DataFrame, team: str, side: str = "offense",
               situation: dict | None = None) -> pd.Series:
    """Share of shot attempts by danger zone (where the team shoots / allows)."""
    df = _filter(events, situation)
    col = "team" if side == "offense" else "opponent"
    d = df[df[col] == team]["danger"].value_counts(normalize=True) * 100
    return d.reindex(["high", "mid", "low", "unknown"]).dropna().round(1)


def _filter(events: pd.DataFrame, situation: dict | None) -> pd.DataFrame:
    if not situation:
        return events
    m = pd.Series(True, index=events.index)
    for k, v in situation.items():
        m &= events[k].isin(v) if isinstance(v, (list, tuple, set)) else events[k] == v
    return events[m]
