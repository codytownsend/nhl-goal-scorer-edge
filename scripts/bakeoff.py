"""Track-A bake-off: test each signal/feature improvement via LOSO.

Stage 1 finds the best RATING signal (5v5 vs all-situations, raw vs
score-adjusted, +/- recency). Stage 2 layers feature groups (goalie, rest,
special teams, travel/road-trip) on the best rating. Everything is scored by
leave-one-season-out log-loss + accuracy on the 3 test seasons (prior season
seeds ratings and joins map-training only). No betting data needed.

    python -m scripts.bakeoff
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nhlsit import games as G
from nhlsit import goalie, predict, rating, rest, special, travel

ROOT = Path(__file__).resolve().parent.parent
PRIOR = "20202021"
TEST = ["20212022", "20222023", "20232024"]

TRAVEL_FEATS = ["travel_diff", "roadtrip_len", "roadtrip_dist", "tz_diff", "away_altitude"]


def load_events(seasons):
    out = []
    for s in seasons:
        fp = ROOT / "data" / f"events_{s}.parquet"
        if fp.exists():
            df = pd.read_parquet(fp); df["season"] = str(s); out.append(df)
    return pd.concat(out, ignore_index=True)


def loso(preds, feats):
    held = []
    for s in TEST:
        tr, te = preds[preds.season != s], preds[preds.season == s].copy()
        ex_tr = {f: tr[f].to_numpy() for f in feats} if feats else None
        ex_te = {f: te[f].to_numpy() for f in feats} if feats else None
        mm = predict.MarginMap().fit(tr.pred_xg_margin, tr.home_margin, ex_tr)
        p = np.clip(mm.p_home_win(te.pred_xg_margin, ex_te), 1e-6, 1 - 1e-6)
        y = te.home_win.to_numpy()
        te["ll"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        te["ok"] = ((mm.margin(te.pred_xg_margin, ex_te) > 0).astype(int) == y).astype(int)
        held.append(te)
    R = pd.concat(held)
    return R.ll.mean(), R.ok.mean()


def main():
    seasons = ([PRIOR] if (ROOT / "data" / f"events_{PRIOR}.parquet").exists() else []) + TEST
    events = load_events(seasons)
    gbs = {s: G.load_or_build(s) for s in seasons}

    # feature frames (independent of the rating config) — compute once
    feat = goalie.walk_forward_goalie(events, gbs, seasons)[["game_id", "goalie_diff"]]
    feat = feat.merge(rest.rest_features(gbs, seasons)[["game_id", "rest_diff"]], on="game_id")
    feat = feat.merge(special.walk_forward_st(events, gbs, seasons), on="game_id")
    feat = feat.merge(travel.travel_features(gbs, seasons), on="game_id", how="left")

    def build(**kw):
        preds, _ = rating.walk_forward(events, gbs, seasons, **kw)
        preds = preds.merge(feat, on="game_id", how="left")
        preds[feat.columns.drop("game_id")] = preds[feat.columns.drop("game_id")].fillna(0.0)
        return preds

    print("=== STAGE 1: rating signal (no extra features) ===")
    print(f"{'config':38} {'logloss':>9} {'acc':>7}")
    configs = {
        "5v5 raw (current baseline)": dict(situation="5v5"),
        "5v5 + score-adjusted": dict(situation="5v5", score_adjust=True),
        "all-situations raw": dict(situation="all"),
        "all-situations + score-adj": dict(situation="all", score_adjust=True),
        "nonempty + score-adj": dict(situation="nonempty", score_adjust=True),
        "all + score-adj + recency(H40)": dict(situation="all", score_adjust=True, recency=0.5 ** (1 / 40)),
        "all + score-adj + recency(H20)": dict(situation="all", score_adjust=True, recency=0.5 ** (1 / 20)),
    }
    best, best_kw, cache = None, None, {}
    for name, kw in configs.items():
        preds = build(**kw); cache[name] = preds
        ll, acc = loso(preds, [])
        tag = ""
        if best is None or ll < best[0]:
            best, best_kw = (ll, acc), kw; tag = "  <-- best"
        print(f"{name:38} {ll:9.4f} {acc:7.3f}{tag}")

    # rebuild best rating for stage 2
    preds = build(**best_kw)
    print(f"\n=== STAGE 2: feature groups on best rating ({best_kw}) ===")
    print(f"{'features':38} {'logloss':>9} {'acc':>7}")
    groups = {
        "none": [],
        "+goalie": ["goalie_diff"],
        "+rest": ["rest_diff"],
        "+special teams": ["st_diff"],
        "+travel/roadtrip": TRAVEL_FEATS,
        "+goalie+rest+st": ["goalie_diff", "rest_diff", "st_diff"],
        "ALL (incl travel)": ["goalie_diff", "rest_diff", "st_diff"] + TRAVEL_FEATS,
    }
    for name, feats in groups.items():
        ll, acc = loso(preds, feats)
        print(f"{name:38} {ll:9.4f} {acc:7.3f}")


if __name__ == "__main__":
    main()
