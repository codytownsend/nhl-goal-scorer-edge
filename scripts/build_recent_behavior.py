"""Wide behavioral history per (game, dressed skater), walk-forward & leak-free.

Goal: capture HOW a player has actually been playing in his recent games -- not
just one metric -- so we can screen which patterns of play precede a goal.

For each prior game we record a player's shot volume, shot QUALITY (xG/shot),
shot LOCATION (high/mid danger), chance TYPE (rush, rebound), special-teams
involvement (PP), on-ice chance share, playmaking (assists), ice time, and
shooting accuracy. We then emit, per upcoming game (features computed strictly
from earlier games):
  * season-to-date rate of each    (the 'average' baseline)
  * recent last-5-game rate         (how he's playing lately)
  * the DEVIATION recent-minus-season (is he trending above/below himself)
  * consistency signals (variance, goal drought, shot-less games)

Everything is within-season recency (resets each season), matching the model's
existing recency treatment.
"""
from collections import deque, defaultdict

import numpy as np
import pandas as pd

SEASONS = [20242025, 20252026]
ON_GOAL = ("goal", "shot-on-goal")

# per-game behavioral quantities we track in the rolling window
KEYS = ["sog", "satt", "goals", "ixg", "hd", "md", "rush", "reb",
        "pp_ixg", "pp_satt", "toi", "assist", "onxgf", "onxga", "dist"]


def raw_stats(ev: pd.DataFrame) -> pd.DataFrame:
    e = ev[ev.is_shot_attempt & ev.shooter_id.notna()].copy()
    e["on_goal"] = e.event.isin(ON_GOAL)
    e["hd"] = (e.danger == "high") & e.is_unblocked
    e["md"] = (e.danger == "mid") & e.is_unblocked
    e["pp"] = e.strength_bucket == "PP"
    e["pp_ixg"] = e.xg.where(e.pp, 0.0)
    e["dist_ub"] = e.distance.where(e.is_unblocked)
    g = e.groupby(["game_id", "shooter_id"]).agg(
        satt=("event", "size"), sog=("on_goal", "sum"), goals=("is_goal", "sum"),
        ixg=("xg", "sum"), hd=("hd", "sum"), md=("md", "sum"),
        rush=("is_rush", "sum"), reb=("is_rebound", "sum"),
        pp_satt=("pp", "sum"), pp_ixg=("pp_ixg", "sum"),
        dist_sum=("dist_ub", "sum"), ub=("is_unblocked", "sum"),
    ).reset_index().rename(columns={"shooter_id": "player_id"})
    g["dist"] = np.where(g.ub > 0, g.dist_sum / g.ub.replace(0, 1), np.nan)
    return g


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else np.nan


def build_season(season: int) -> pd.DataFrame:
    ev = pd.read_parquet(f"data/events_{season}.parquet")
    lu = pd.read_parquet(f"data/lineups_{season}.parquet")
    games = pd.read_parquet(f"data/games_{season}.parquet")[["game_id", "date"]]
    on = pd.read_parquet("data/onice_playergames.parquet")
    on = on[on.season == season][["game_id", "player_id", "g_xgf", "g_xga"]]

    base = lu[["game_id", "team", "player_id", "toi_sec", "points"]].merge(
        games, on="game_id")
    df = base.merge(raw_stats(ev), on=["game_id", "player_id"], how="left")
    df = df.merge(on, on=["game_id", "player_id"], how="left")
    statcols = ["satt", "sog", "goals", "ixg", "hd", "md", "rush", "reb",
                "pp_satt", "pp_ixg", "g_xgf", "g_xga"]
    df[statcols] = df[statcols].fillna(0.0)
    df["toi"] = df["toi_sec"].fillna(0.0) / 60.0
    df["assist"] = (df["points"].fillna(0) - df["goals"]).clip(lower=0)
    df["onxgf"] = df["g_xgf"]; df["onxga"] = df["g_xga"]
    df["scored"] = (df["goals"] >= 1).astype(int)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "game_id"]).reset_index(drop=True)

    cum = defaultdict(lambda: defaultdict(float))
    gpc = defaultdict(float)
    rec = defaultdict(lambda: {k: deque(maxlen=10) for k in KEYS})
    goaldq = defaultdict(lambda: deque(maxlen=10))
    rows = []
    for r in df.itertuples(index=False):
        p = r.player_id
        gp = gpc[p]
        c = cum[p]
        rc = rec[p]

        def srate(k):                       # season per-game rate
            return c[k] / gp if gp else 0.0
        def l5(k):                          # recent last-5 per-game rate
            return _mean(list(rc[k])[-5:])

        row = {"game_id": r.game_id, "player_id": p, "season": season,
               "date": r.date, "scored": r.scored, "gp": gp}
        # season baselines
        for k in ["sog", "satt", "ixg", "hd", "md", "rush", "reb", "pp_ixg",
                  "toi", "assist", "onxgf", "goals"]:
            row[f"{k}_pg"] = srate(k)
        row["sogpct_s"] = c["sog"] / c["satt"] if c["satt"] else 0.0
        row["xgpsh_s"] = c["ixg"] / c["satt"] if c["satt"] else 0.0
        row["hdshare_s"] = c["hd"] / c["satt"] if c["satt"] else 0.0
        # recent levels + deviations (recent minus season)
        for k in ["sog", "satt", "ixg", "hd", "md", "rush", "reb", "pp_ixg",
                  "toi", "assist", "onxgf"]:
            rv = l5(k)
            row[f"{k}_l5"] = rv
            row[f"{k}_dev"] = (rv - srate(k)) if not np.isnan(rv) else 0.0
        last5 = list(rc["satt"])[-5:]
        last5_sog = list(rc["sog"])[-5:]
        last5_ixg = list(rc["ixg"])[-5:]
        rsatt = sum(last5); rsog = sum(last5_sog)
        row["sogpct_l5"] = rsog / rsatt if rsatt else 0.0
        row["xgpsh_l5"] = sum(last5_ixg) / rsatt if rsatt else 0.0
        row["sogpct_dev"] = row["sogpct_l5"] - row["sogpct_s"]
        row["xgpsh_dev"] = row["xgpsh_l5"] - row["xgpsh_s"]
        row["dist_l5"] = l5("dist")
        # consistency / drought signals
        row["ixg_std5"] = float(np.std(last5_ixg)) if last5_ixg else 0.0
        row["games_w_shot5"] = float(sum(1 for x in last5_sog if x > 0))
        g5 = list(goaldq[p])[-5:]
        row["dry5"] = float(sum(1 for x in g5 if x == 0))        # 0-goal games in last 5
        row["since_goal"] = _games_since(goaldq[p])              # games since last goal
        rows.append(row)

        # update AFTER emit (walk-forward)
        for k in KEYS:
            cum[p][k] += getattr(r, k) if not (isinstance(getattr(r, k), float)
                                               and np.isnan(getattr(r, k))) else 0.0
            rc[k].append(getattr(r, k))
        goaldq[p].append(r.goals)
        gpc[p] += 1
    return pd.DataFrame(rows)


def _games_since(dq):
    n = 0
    for x in reversed(dq):
        if x >= 1:
            return float(n)
        n += 1
    return float(n)


def main():
    out = pd.concat([build_season(s) for s in SEASONS], ignore_index=True)
    out.to_parquet("data/recent_behavior.parquet")
    print("wrote data/recent_behavior.parquet", out.shape)
    print("features:", [c for c in out.columns if c not in
                        ("game_id", "player_id", "season", "date", "scored", "gp")])


if __name__ == "__main__":
    main()
