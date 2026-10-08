"""Per-player goal-scorer dataset (walk-forward, leak-free).

Builds one row per (game, dressed skater) with:
  - label `scored` = 1 if the player scored >=1 goal in that game
  - `goals` (count, for a Poisson target if wanted)
  - a set of PRE-GAME candidate features computed only from games strictly
    earlier than this one (walk-forward), so the table can be used to train and
    evaluate an "anytime goal scorer" model without leakage.

Philosophy matches the rest of the project: compute many candidate signals,
then let a reliability gate + model pick which ones actually predict. Nothing
here decides a feature is good; it just makes them available honestly.

Player signals (season-to-date, cross-season carried with decay CARRY):
  g_pg, sh_pg, satt_pg, ixg_pg   per-game goals / shots-on-goal / attempts / ind.xG
  hd_pg                          high-danger unblocked shots per game
  pp_ixg_pg                      power-play individual xG per game (PP role)
  finish                         goals-above-xG per shot, shrunk (finishing talent)
  toi_pg                         average time on ice (minutes) -> role/minutes
  is_def                         1 if a defenseman
Recency (hot/cold):
  g_l5, ixg_l5, sh_l5, g_l10, ixg_l10   last-5 / last-10 game sums
Context (opponent / situation):
  is_home
  opp_goalie      opposing starter's pre-game GSAx per shot (+ = good goalie)
  opp_def         opponent team's pre-game xG-allowed per game (team defense)
"""
from __future__ import annotations

from collections import deque, defaultdict

import pandas as pd

CARRY = 0.7          # cross-season decay applied to a player's cumulative totals
FINISH_SHOT_PC = 300.0   # pseudo-shots shrinking finishing toward league (0)
ON_GOAL = ("goal", "shot-on-goal")


def _player_game_stats(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw per (game, shooter) counting stats from the event table."""
    ev = events[events["is_shot_attempt"] & events["shooter_id"].notna()].copy()
    ev["on_goal"] = ev["event"].isin(ON_GOAL)
    ev["is_hd"] = (ev["danger"] == "high") & ev["is_unblocked"]
    ev["pp_ixg"] = ev["xg"].where(ev["strength_bucket"] == "PP", 0.0)
    g = ev.groupby(["game_id", "shooter_id"]).agg(
        goals=("is_goal", "sum"),
        shots=("on_goal", "sum"),
        satt=("event", "size"),
        ixg=("xg", "sum"),
        hd=("is_hd", "sum"),
        pp_ixg=("pp_ixg", "sum"),
    ).reset_index().rename(columns={"shooter_id": "player_id"})
    return g


def build_player_games(events: pd.DataFrame, lineups: pd.DataFrame,
                       games_by_season: dict, seasons: list) -> pd.DataFrame:
    """Return one leak-free row per (game, dressed skater) with label+features."""
    allg = pd.concat([games_by_season[s] for s in seasons], ignore_index=True)
    allg = allg[["game_id", "date", "season", "home", "away"]].copy()
    allg["date"] = pd.to_datetime(allg["date"])
    order = allg.sort_values(["date", "game_id"]).reset_index(drop=True)
    team_of = {}  # (game_id, player) -> team ; filled from lineups below

    # --- candidate rows: every dressed skater (the pool we must rank) ---
    lu = lineups[["game_id", "team", "player_id", "pos", "toi_sec"]].copy()
    lu = lu.merge(order[["game_id", "date", "season", "home", "away"]], on="game_id")
    lu["is_home"] = (lu["team"] == lu["home"]).astype(int)
    lu["opp"] = lu["home"].where(lu["team"] == lu["away"], lu["away"])
    lu["is_def"] = (lu["pos"] == "defense").astype(int)

    # raw per-player-game production (for label + walk-forward accumulation)
    stats = _player_game_stats(events)
    stats = stats.merge(order[["game_id", "date", "season"]], on="game_id")

    df = lu.merge(stats.drop(columns=["date", "season"]),
                  on=["game_id", "player_id"], how="left")
    for c in ["goals", "shots", "satt", "ixg", "hd", "pp_ixg"]:
        df[c] = df[c].fillna(0.0)
    df["scored"] = (df["goals"] >= 1).astype(int)
    df = df.sort_values(["date", "game_id"]).reset_index(drop=True)

    # --- walk-forward accumulators (pre-game state per player) ---
    cum = defaultdict(lambda: dict(gp=0.0, g=0.0, sh=0.0, sa=0.0, ixg=0.0,
                                   hd=0.0, ppx=0.0, toi=0.0, gax=0.0, shf=0.0))
    last_season = {}
    recent = defaultdict(lambda: dict(g=deque(maxlen=10), ixg=deque(maxlen=10),
                                      sh=deque(maxlen=10), toi=deque(maxlen=10),
                                      pp=deque(maxlen=10)))

    feat_rows = []
    for r in df.itertuples(index=False):
        p = r.player_id
        c = cum[p]
        if p in last_season and last_season[p] != r.season:
            for k in ("gp", "g", "sh", "sa", "ixg", "hd", "ppx", "toi", "gax", "shf"):
                c[k] *= CARRY
            # recency deques intentionally NOT carried (hot/cold is within-season)
        gp = c["gp"]
        rc = recent[p]
        feat_rows.append({
            "game_id": r.game_id, "player_id": p, "team": r.team, "opp": r.opp,
            "date": r.date, "season": r.season,
            "scored": r.scored, "goals": r.goals,
            "is_home": r.is_home, "is_def": r.is_def,
            "gp": gp,
            "g_pg": c["g"] / gp if gp else 0.0,
            "sh_pg": c["sh"] / gp if gp else 0.0,
            "satt_pg": c["sa"] / gp if gp else 0.0,
            "ixg_pg": c["ixg"] / gp if gp else 0.0,
            "hd_pg": c["hd"] / gp if gp else 0.0,
            "pp_ixg_pg": c["ppx"] / gp if gp else 0.0,
            "toi_pg": (c["toi"] / gp / 60.0) if gp else 0.0,
            "finish": c["gax"] / (c["shf"] + FINISH_SHOT_PC),
            "g_l5": sum(list(rc["g"])[-5:]),
            "ixg_l5": sum(list(rc["ixg"])[-5:]),
            "sh_l5": sum(list(rc["sh"])[-5:]),
            "g_l10": sum(rc["g"]),
            "ixg_l10": sum(rc["ixg"]),
            # recent role/usage (averages, to compare against season baseline)
            "toi_l5": (sum(list(rc["toi"])[-5:]) / len(list(rc["toi"])[-5:]))
                      if rc["toi"] else 0.0,
            "pp_ixg_l5": (sum(list(rc["pp"])[-5:]) / len(list(rc["pp"])[-5:]))
                         if rc["pp"] else 0.0,
        })
        # update state AFTER emitting (walk-forward)
        c["gp"] += 1
        c["g"] += r.goals; c["sh"] += r.shots; c["sa"] += r.satt
        c["ixg"] += r.ixg; c["hd"] += r.hd; c["ppx"] += r.pp_ixg
        c["toi"] += (r.toi_sec or 0.0)
        c["gax"] += (r.goals - r.ixg); c["shf"] += r.shots
        rc["g"].append(r.goals); rc["ixg"].append(r.ixg); rc["sh"].append(r.shots)
        rc["toi"].append((r.toi_sec or 0.0) / 60.0); rc["pp"].append(r.pp_ixg)
        last_season[p] = r.season

    out = pd.DataFrame(feat_rows)
    out = _attach_context(out, events, order, games_by_season, seasons)
    return _add_rate_features(out)


def _attach_context(out: pd.DataFrame, events: pd.DataFrame, order: pd.DataFrame,
                    games_by_season: dict, seasons: list) -> pd.DataFrame:
    """Add opponent goalie (GSAx) and opponent team-defense (xGA/game), walk-forward."""
    from . import goalie as G

    # opponent team defense: pre-game xG allowed per game, cross-season carried
    xga = events.groupby(["game_id", "opponent"]).agg(xga=("xg", "sum")).reset_index()
    xga = xga.rename(columns={"opponent": "team"}).merge(
        order[["game_id", "date", "season"]], on="game_id").sort_values(["date", "game_id"])
    cum = defaultdict(lambda: dict(x=0.0, n=0.0))
    last = {}
    def_rows = []
    for r in xga.itertuples(index=False):
        c = cum[r.team]
        if r.team in last and last[r.team] != r.season:
            c["x"] *= CARRY; c["n"] *= CARRY
        def_rows.append({"game_id": r.game_id, "team": r.team,
                         "opp_def": c["x"] / c["n"] if c["n"] else None})
        c["x"] += r.xga; c["n"] += 1; last[r.team] = r.season
    defdf = pd.DataFrame(def_rows)  # keyed by DEFENDING team
    out = out.merge(defdf.rename(columns={"team": "opp"}), on=["game_id", "opp"], how="left")

    # opponent starting goalie GSAx/shot (home/away split from goalie module)
    gl = G.walk_forward_goalie(events, games_by_season, seasons)
    gmap = order[["game_id", "home", "away"]].merge(gl, on="game_id", how="left")
    # a player's opponent goalie = other team's starter rate
    hr = dict(zip(gmap.game_id, gmap.get("home_g_rate")))
    ar = dict(zip(gmap.game_id, gmap.get("away_g_rate")))
    out["opp_goalie"] = [ (ar.get(g) if h else hr.get(g))
                          for g, h in zip(out.game_id, out.is_home) ]
    league_def = out["opp_def"].median()
    out["opp_def"] = out["opp_def"].fillna(league_def)
    out["opp_goalie"] = out["opp_goalie"].fillna(0.0)
    return out


SHRINK_K = 30.0   # games of prior-season weight anchoring a player's rate (tuned)


def _add_rate_features(out: pd.DataFrame, K: float = SHRINK_K) -> pd.DataFrame:
    """Explicit shrink-to-prior rate estimates + role/usage trends.

    `rate_shrunk`/`g_rate_shrunk`: blend prior-season rate (worth K games) with
    this season's results so far -> anchored, doesn't chase hot starts (the
    tested-best rate estimator). `toi_trend`/`pp_trend`: recent usage minus
    season baseline -> detects line/PP promotions (a real rate change).
    """
    out = out.sort_values(["player_id", "season", "date"]).reset_index(drop=True)
    grp = out.groupby(["player_id", "season"])
    out["_cs"] = grp["scored"].cumsum() - out["scored"]          # within-season, pre-game
    out["_cg"] = grp["goals"].cumsum() - out["goals"]
    out["_n"] = grp.cumcount()

    # prior-season rate mapped forward to the next season
    sr = grp.agg(_r=("scored", "mean"), _gr=("goals", "mean")).reset_index()
    seasons = sorted(out["season"].unique())
    nxt = {seasons[i]: seasons[i + 1] for i in range(len(seasons) - 1)}
    sr["season"] = sr["season"].map(nxt)
    sr = sr.dropna(subset=["season"])
    out = out.merge(sr, on=["player_id", "season"], how="left")
    out["_r"] = out["_r"].fillna(out["scored"].mean())
    out["_gr"] = out["_gr"].fillna(out["goals"].mean())

    out["rate_shrunk"] = (out["_r"] * K + out["_cs"]) / (K + out["_n"])
    out["g_rate_shrunk"] = (out["_gr"] * K + out["_cg"]) / (K + out["_n"])
    out["toi_trend"] = out["toi_l5"] - out["toi_pg"]
    out["pp_trend"] = out["pp_ixg_l5"] - out["pp_ixg_pg"]
    return out.drop(columns=[c for c in out.columns if c.startswith("_")])
