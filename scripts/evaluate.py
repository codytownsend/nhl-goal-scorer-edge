"""Leave-one-season-out evaluation of the walk-forward predictor.

Levers (all walk-forward / leak-free): opponent-adjusted 5v5 xG rating, starting
goalie (GSAx), rest / back-to-back, special teams (PP/PK). An optional earlier
PRIOR season seeds the ratings (fixes cold start) and adds map-training data but
is never itself a held-out test season. Win prob via Normal-CDF or Skellam.

    python -m scripts.evaluate                 # default seasons + prior
    python -m scripts.evaluate --winprob skellam
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nhlsit import availability, finishing, games as G
from nhlsit import goalie, predict, rating, rest, special

ROOT = Path(__file__).resolve().parent.parent
PRIOR = "20202021"
TEST = ["20212022", "20222023", "20232024"]

VARIANTS = {
    "base": [],
    "+goalie": ["goalie_diff"],
    "+rest": ["rest_diff"],
    "+st": ["st_diff"],
    "+availability": ["linedev_diff"],
    "+finishing": ["finish_diff"],
    "full": ["goalie_diff", "rest_diff", "st_diff", "linedev_diff", "finish_diff"],
}


def load_events(seasons):
    frames = []
    for s in seasons:
        fp = ROOT / "data" / f"events_{s}.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        df["season"] = str(s)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _extras(df, feats):
    return {f: df[f].to_numpy() for f in feats} if feats else None


def _winprob(mm, xg, extras, method):
    if method == "skellam":
        return predict.skellam_win_prob(mm.margin(xg, extras))
    return mm.p_home_win(xg, extras)


def _metrics(df):
    p = np.clip(df["p_home"].to_numpy(), 1e-6, 1 - 1e-6)
    y = df["home_win"].to_numpy()
    acc = (df["pred_home_win"] == df["home_win"]).mean()
    ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    brier = np.mean((p - y) ** 2)
    return acc, ll, brier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--winprob", choices=["normal", "skellam"], default="normal")
    ap.add_argument("--lam", type=float, default=3.0)
    ap.add_argument("--carry", type=float, default=0.75)
    args = ap.parse_args()

    all_seasons = ([PRIOR] if (ROOT / "data" / f"events_{PRIOR}.parquet").exists()
                   else []) + TEST
    print(f"seasons: {all_seasons}  (prior={'yes' if PRIOR in all_seasons else 'no'})")
    events = load_events(all_seasons)
    games_by_season = {s: G.load_or_build(s) for s in all_seasons}

    # all-situations rating (Track A: beats 5v5-only for the base signal)
    preds, _ = rating.walk_forward(events, games_by_season, all_seasons,
                                   carry=args.carry, lam=args.lam, situation="all")
    gdf = goalie.walk_forward_goalie(events, games_by_season, all_seasons)
    rdf = rest.rest_features(games_by_season, all_seasons)
    sdf = special.walk_forward_st(events, games_by_season, all_seasons)
    feat_cols = ["goalie_diff", "rest_diff", "st_diff"]
    for extra in (gdf[["game_id", "goalie_diff"]], rdf[["game_id", "rest_diff"]],
                  sdf[["game_id", "st_diff"]]):
        preds = preds.merge(extra, on="game_id", how="left")
    # Track-B lineup availability, if boxscore lineups have been pulled
    lu = [pd.read_parquet(ROOT / "data" / f"lineups_{s}.parquet")
          for s in all_seasons if (ROOT / "data" / f"lineups_{s}.parquet").exists()]
    if lu:
        lineups = pd.concat(lu)
        adf = availability.walk_forward_availability(lineups, games_by_season, all_seasons)
        preds = preds.merge(adf, on="game_id", how="left")
        feat_cols.append("linedev_diff")
        # roster finishing / conversion talent (orthogonal to team xG)
        fdf = finishing.walk_forward_finishing(events, lineups, games_by_season, all_seasons)
        preds = preds.merge(fdf, on="game_id", how="left")
        feat_cols.append("finish_diff")
    else:
        preds["linedev_diff"] = 0.0
        preds["finish_diff"] = 0.0
    preds[feat_cols] = preds[feat_cols].fillna(0.0)

    # evaluate only on TEST seasons; map trains on all OTHER seasons (incl prior)
    test_preds = preds[preds["season"].isin(TEST)]
    print(f"walk-forward predictions: {len(preds):,} ({len(test_preds):,} in test)\n")

    results = {}
    for name, feats in VARIANTS.items():
        held = []
        for s in TEST:
            train = preds[preds["season"] != s]
            test = preds[preds["season"] == s].copy()
            mm = predict.MarginMap().fit(
                train["pred_xg_margin"], train["home_margin"], _extras(train, feats))
            test["p_home"] = _winprob(mm, test["pred_xg_margin"],
                                      _extras(test, feats), args.winprob)
            test["pred_home_win"] = (
                mm.margin(test["pred_xg_margin"], _extras(test, feats)) > 0).astype(int)
            held.append(test)
        R = pd.concat(held, ignore_index=True)
        results[name] = R
        acc, ll, brier = _metrics(R)
        print(f"{name:9s}  acc={acc:.3f}  logloss={ll:.4f}  brier={brier:.4f}")

    print(f"\nbaseline (always home): acc={test_preds['home_win'].mean():.3f}"
          f"   [winprob={args.winprob}, lam={args.lam}, carry={args.carry}]")

    R = results["full"]
    R["bucket"] = pd.cut(R["p_home"], bins=np.linspace(0, 1, 11))
    cal = R.groupby("bucket", observed=True).agg(
        n=("home_win", "size"), pred=("p_home", "mean"), actual=("home_win", "mean"))
    print("\n=== calibration (full) ===")
    print(cal.round(3).to_string())

    out = ROOT / "data" / "loso_predictions.parquet"
    R.drop(columns=["bucket"]).to_parquet(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
