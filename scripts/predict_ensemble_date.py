"""Ensemble (rate + GRU) goal-scorer board for a date, vs the previous rate model,
plus a parallel POINT-likelihood board. Same top-10 players in both tables.

previous  = rate GBM on season-average features (what predict_date.py gives)
ensemble  = logit-average(rate GBM, GRU trained on raw play sequences)
points    = same two-model ensemble, retrained on the point label (>=1 point)

Leak-free: only the slate's dressed identities + home/away come from the date;
all ratings/sequences are built from cached history (<= game 52 of 2026-27).
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from nhlsit import scorer
from scripts.scorer_eval import FEATURES
from scripts.predict_date import fetch_slate
import scripts.form_poisson as FP

torch.set_num_threads(1)
HIST = [20202021, 20212022, 20222023, 20232024, 20242025, 20252026]
SEQ_SEASONS = [20242025, 20252026, 20262027]     # for building last-10 windows
FEAT_ORDER = ["sog", "satt", "ixg", "hd", "md", "rush", "reb", "pp_ixg",
              "toi", "assist", "dist", "goals"]
ON_GOAL = ("goal", "shot-on-goal")
L = 10


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _gru_probs(path, Xtr, y, Xsl):
    """Load the saved GRU and predict; if weights are missing, train+save once."""
    if os.path.exists(path):
        net, mu, sd = FP.load_gru(path)
    else:
        net, mu, sd = FP.fit_gru(Xtr, y, "binary", seed=0)
        FP.save_gru(net, mu, sd, path)
    return FP.predict_gru(net, mu, sd, Xsl, "binary")


def per_game_vectors(season):
    """Per (game, player) 12-feature play vector, ordered chronologically."""
    ev = pd.read_parquet(f"data/events_{season}.parquet")
    lu = pd.read_parquet(f"data/lineups_{season}.parquet")
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
        rush=("is_rush", "sum"), reb=("is_rebound", "sum"), pp_ixg=("pp_ixg", "sum"),
        dist_sum=("dist_ub", "sum"), ub=("is_unblocked", "sum"),
    ).reset_index().rename(columns={"shooter_id": "player_id"})
    g["dist"] = np.where(g.ub > 0, g.dist_sum / g.ub.replace(0, 1), 40.0)
    base = lu[["game_id", "player_id", "toi_sec", "points"]]
    df = base.merge(g, on=["game_id", "player_id"], how="left")
    for c in ["satt", "sog", "goals", "ixg", "hd", "md", "rush", "reb", "pp_ixg"]:
        df[c] = df[c].fillna(0.0)
    df["dist"] = df["dist"].fillna(40.0)
    df["toi"] = df["toi_sec"].fillna(0.0) / 60.0
    df["assist"] = (df["points"].fillna(0) - df["goals"]).clip(lower=0)
    if season == 20262027:
        df["ord"] = df["game_id"]
    else:
        gm = pd.read_parquet(f"data/games_{season}.parquet")[["game_id", "date"]]
        gm["date"] = pd.to_datetime(gm["date"])
        df = df.merge(gm, on="game_id")
        df["ord"] = df["date"].rank(method="dense")
    return df[["game_id", "player_id", "ord"] + FEAT_ORDER]


def slate_sequences(player_ids):
    """Last-10-game play sequence for each slate player (None if <10 games)."""
    frames = []
    for i, s in enumerate(SEQ_SEASONS):
        d = per_game_vectors(s)
        d["gord"] = i * 1e9 + d["ord"]
        frames.append(d)
    allpg = pd.concat(frames, ignore_index=True).sort_values("gord")
    seqs, have = {}, []
    for pid, grp in allpg[allpg.player_id.isin(player_ids)].groupby("player_id"):
        if len(grp) >= L:
            seqs[pid] = grp.tail(L)[FEAT_ORDER].values.astype(np.float32)
            have.append(pid)
    return seqs, have


def compute_slate(gnew, lnew, names):
    """Return slate df with rate / GRU / ensemble probs for goals and points."""
    season_new = int(gnew.season.iloc[0])
    gmap = {r.game_id: f"{r.away}@{r.home}" for r in gnew.itertuples(index=False)}

    # ---------- rate models (goal + point), the 'previous' number ----------
    events = pd.concat([pd.read_parquet(f"data/events_{s}.parquet") for s in HIST],
                       ignore_index=True)
    lineups = pd.concat([pd.read_parquet(f"data/lineups_{s}.parquet") for s in HIST]
                        + [lnew], ignore_index=True)
    gbs = {}
    for s in HIST:
        gg = pd.read_parquet(f"data/games_{s}.parquet"); gg["season"] = gg["season"].astype(int)
        gbs[s] = gg
    gbs[season_new] = gnew
    feats = scorer.build_player_games(events, lineups, gbs, HIST + [season_new])
    feats["season"] = feats["season"].astype(int)
    pts = pd.concat([pd.read_parquet(f"data/lineups_{s}.parquet")[
        ["game_id", "player_id", "points"]] for s in HIST], ignore_index=True)
    feats = feats.merge(pts, on=["game_id", "player_id"], how="left")
    feats["point"] = (feats["points"].fillna(0) >= 1).astype(int)
    train = feats[feats.season != season_new]
    slate = feats[feats.season == season_new].copy()

    def gbm(y):
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0,
            min_samples_leaf=200, random_state=0).fit(train[FEATURES].astype(float), y)
    slate["p_rate_g"] = gbm(train.scored.values).predict_proba(
        slate[FEATURES].astype(float))[:, 1]
    slate["p_rate_p"] = gbm(train.point.values).predict_proba(
        slate[FEATURES].astype(float))[:, 1]

    # ---------- GRU models on raw play sequences (goal + point) ----------
    sq = np.load("data/player_sequences.npz", allow_pickle=True)
    keep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13]          # drop on-ice feats -> 12
    Xtr = sq["X"][:, :, keep]
    ktr = pd.DataFrame({"game_id": sq["game_id"], "player_id": sq["player_id"],
                        "scored": sq["y"]})
    ktr = ktr.merge(pts, on=["game_id", "player_id"], how="left")
    ytr_g = ktr["scored"].values
    ytr_p = (ktr["points"].fillna(0) >= 1).astype(float).values

    seqs, have = slate_sequences(set(slate.player_id))
    Xsl = np.stack([seqs[p] for p in have]) if have else np.zeros((0, L, 12), np.float32)
    pg = _gru_probs("data/gru_goal.pt", Xtr, ytr_g, Xsl) if have else np.array([])
    pp = _gru_probs("data/gru_point.pt", Xtr, ytr_p, Xsl) if have else np.array([])
    gru_g = dict(zip(have, pg)); gru_p = dict(zip(have, pp))

    # players w/o a 10-game window fall back to the rate number (no GRU tilt)
    slate["p_gru_g"] = [gru_g.get(p, r) for p, r in zip(slate.player_id, slate.p_rate_g)]
    slate["p_gru_p"] = [gru_p.get(p, r) for p, r in zip(slate.player_id, slate.p_rate_p)]
    slate["ens_g"] = 1 / (1 + np.exp(-(0.5 * logit(slate.p_rate_g) + 0.5 * logit(slate.p_gru_g))))
    slate["ens_p"] = 1 / (1 + np.exp(-(0.5 * logit(slate.p_rate_p) + 0.5 * logit(slate.p_gru_p))))
    slate["name"] = slate.player_id.map(names)
    slate["g"] = slate.game_id.map(gmap)
    slate["seq"] = slate.player_id.isin(have)
    return slate


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else "2026-10-07"
    gnew, lnew, names = fetch_slate(date)
    print(f"\n{date}: {len(gnew)} games, {len(lnew)} dressed skaters\n")
    slate = compute_slate(gnew, lnew, names)

    top = slate.sort_values("ens_g", ascending=False).head(10)
    print("===== TOP 10 ANYTIME GOAL SCORERS (Oct 7 2026, all 3 games) =====")
    print(f"{'#':>2} {'player':<20}{'team':>4}{'game':>10}{'prev(rate)':>12}"
          f"{'ensemble':>10}{'Δ':>8}")
    for i, r in enumerate(top.itertuples(index=False), 1):
        flag = "" if r.seq else "  (no GRU: <10 gm)"
        print(f"{i:>2} {str(r.name):<20}{r.team:>4}{r.g:>10}{r.p_rate_g:>11.1%}"
              f"{r.ens_g:>10.1%}{(r.ens_g - r.p_rate_g) * 100:>+7.1f}{flag}")

    print("\n===== POINT LIKELIHOOD for the same top 10 (>=1 point) =====")
    print(f"{'#':>2} {'player':<20}{'team':>4}{'prev(rate)':>12}{'ensemble':>10}{'Δ':>8}")
    for i, r in enumerate(top.itertuples(index=False), 1):
        print(f"{i:>2} {str(r.name):<20}{r.team:>4}{r.p_rate_p:>11.1%}"
              f"{r.ens_p:>10.1%}{(r.ens_p - r.p_rate_p) * 100:>+7.1f}")


if __name__ == "__main__":
    main()
