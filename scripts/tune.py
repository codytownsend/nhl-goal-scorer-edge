"""Coarse hyperparameter search (#3) for the rating solver.

Grids ridge lambda and cross-season carry, scoring LOSO log-loss of the full
feature model. Ratings are the expensive part, so we re-run walk_forward per
(lam, carry) and reuse the (cheap) goalie/rest/st features.

    python -m scripts.tune
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nhlsit import games as G
from nhlsit import goalie, predict, rating, rest, special

ROOT = Path(__file__).resolve().parent.parent
PRIOR = "20202021"
TEST = ["20212022", "20222023", "20232024"]
FEATS = ["goalie_diff", "rest_diff", "st_diff"]

LAMS = [2.0, 3.0, 5.0, 8.0]
CARRIES = [0.6, 0.75, 0.85]


def load_events(seasons):
    out = []
    for s in seasons:
        fp = ROOT / "data" / f"events_{s}.parquet"
        if fp.exists():
            df = pd.read_parquet(fp)
            df["season"] = str(s)
            out.append(df)
    return pd.concat(out, ignore_index=True)


def score(preds):
    held = []
    for s in TEST:
        train = preds[preds["season"] != s]
        test = preds[preds["season"] == s].copy()
        ex_tr = {f: train[f].to_numpy() for f in FEATS}
        ex_te = {f: test[f].to_numpy() for f in FEATS}
        mm = predict.MarginMap().fit(train["pred_xg_margin"], train["home_margin"], ex_tr)
        p = np.clip(mm.p_home_win(test["pred_xg_margin"], ex_te), 1e-6, 1 - 1e-6)
        y = test["home_win"].to_numpy()
        test["ll"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        test["correct"] = ((mm.margin(test["pred_xg_margin"], ex_te) > 0).astype(int)
                           == test["home_win"]).astype(int)
        held.append(test)
    R = pd.concat(held)
    return R["ll"].mean(), R["correct"].mean()


def main():
    seasons = ([PRIOR] if (ROOT / "data" / f"events_{PRIOR}.parquet").exists() else []) + TEST
    events = load_events(seasons)
    gbs = {s: G.load_or_build(s) for s in seasons}
    # features that don't depend on (lam, carry) — compute once
    gdf = goalie.walk_forward_goalie(events, gbs, seasons)
    rdf = rest.rest_features(gbs, seasons)
    sdf = special.walk_forward_st(events, gbs, seasons)

    print(f"{'lam':>5} {'carry':>6} {'logloss':>9} {'acc':>7}")
    best = None
    for lam in LAMS:
        for carry in CARRIES:
            preds, _ = rating.walk_forward(events, gbs, seasons, carry=carry, lam=lam)
            for extra in (gdf[["game_id", "goalie_diff"]], rdf[["game_id", "rest_diff"]],
                          sdf[["game_id", "st_diff"]]):
                preds = preds.merge(extra, on="game_id", how="left")
            preds[FEATS] = preds[FEATS].fillna(0.0)
            ll, acc = score(preds)
            flag = ""
            if best is None or ll < best[0]:
                best = (ll, acc, lam, carry)
                flag = "  <-- best"
            print(f"{lam:5.1f} {carry:6.2f} {ll:9.4f} {acc:7.3f}{flag}")
    print(f"\nBEST: lam={best[2]} carry={best[3]}  logloss={best[0]:.4f} acc={best[1]:.3f}")


if __name__ == "__main__":
    main()
