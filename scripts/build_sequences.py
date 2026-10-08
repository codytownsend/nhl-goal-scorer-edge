"""Raw game-by-game play SEQUENCES per (game, player) -- the native play data.

Unlike the summary features (last-5 means), this keeps the ORDER and SHAPE of a
player's recent games: a length-10 sequence of per-game play vectors feeding the
upcoming game's label. This is what a sequence model needs to find trajectory
patterns (ramping up, spiking, fading) that means throw away.
"""
import numpy as np
import pandas as pd

SEASONS = [20242025, 20252026]
ON_GOAL = ("goal", "shot-on-goal")
FEATS = ["sog", "satt", "ixg", "hd", "md", "rush", "reb", "pp_ixg",
         "toi", "assist", "onxgf", "onxga", "dist", "goals"]
L = 10


def per_game(season):
    ev = pd.read_parquet(f"data/events_{season}.parquet")
    lu = pd.read_parquet(f"data/lineups_{season}.parquet")
    games = pd.read_parquet(f"data/games_{season}.parquet")[["game_id", "date"]]
    on = pd.read_parquet("data/onice_playergames.parquet")
    on = on[on.season == season][["game_id", "player_id", "g_xgf", "g_xga"]]

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
        pp_ixg=("pp_ixg", "sum"), dist_sum=("dist_ub", "sum"),
        ub=("is_unblocked", "sum"),
    ).reset_index().rename(columns={"shooter_id": "player_id"})
    g["dist"] = np.where(g.ub > 0, g.dist_sum / g.ub.replace(0, 1), 40.0)

    base = lu[["game_id", "player_id", "toi_sec", "points"]].merge(games, on="game_id")
    df = base.merge(g, on=["game_id", "player_id"], how="left").merge(
        on, on=["game_id", "player_id"], how="left")
    for c in ["satt", "sog", "goals", "ixg", "hd", "md", "rush", "reb", "pp_ixg",
              "g_xgf", "g_xga"]:
        df[c] = df[c].fillna(0.0)
    df["dist"] = df["dist"].fillna(40.0)
    df["toi"] = df["toi_sec"].fillna(0.0) / 60.0
    df["assist"] = (df["points"].fillna(0) - df["goals"]).clip(lower=0)
    df["onxgf"] = df["g_xgf"]; df["onxga"] = df["g_xga"]
    df["scored"] = (df["goals"] >= 1).astype(int)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "game_id"]).reset_index(drop=True)


def main():
    from collections import defaultdict, deque
    X, y, gid, pid, ssn = [], [], [], [], []
    for s in SEASONS:
        df = per_game(s)
        hist = defaultdict(lambda: deque(maxlen=L))
        for r in df.itertuples(index=False):
            p = r.player_id
            h = hist[p]
            if len(h) == L:                      # full 10-game window (gp>=10)
                X.append(np.array(h, dtype=np.float32))
                y.append(r.scored); gid.append(r.game_id); pid.append(p); ssn.append(s)
            hist[p].append([getattr(r, f) for f in FEATS])
    X = np.stack(X)
    np.savez("data/player_sequences.npz", X=X, y=np.array(y, dtype=np.float32),
             game_id=np.array(gid), player_id=np.array(pid), season=np.array(ssn),
             feats=np.array(FEATS))
    print("wrote data/player_sequences.npz", X.shape, "pos rate", np.mean(y).round(3))


if __name__ == "__main__":
    main()
