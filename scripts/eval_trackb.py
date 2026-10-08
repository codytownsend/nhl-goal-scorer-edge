"""Evaluate the Track-B lineup-availability signal.

Does knowing who's actually dressed (injury/scratch deviation) improve
prediction — and specifically, does it help the TOSS-UP games where our whole
deficit to the market lives?

    python -m scripts.eval_trackb
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nhlsit import availability, games as G, goalie, predict, rating, rest, special

ROOT = Path(__file__).resolve().parent.parent
PRIOR = "20202021"
TEST = ["20212022", "20222023", "20232024"]
BASE = ["goalie_diff", "rest_diff", "st_diff"]


def load_events(seasons):
    out = []
    for s in seasons:
        fp = ROOT / "data" / f"events_{s}.parquet"
        if fp.exists():
            out.append(pd.read_parquet(fp).assign(season=str(s)))
    return pd.concat(out, ignore_index=True)


def load_lineups(seasons):
    out = []
    for s in seasons:
        fp = ROOT / "data" / f"lineups_{s}.parquet"
        if fp.exists():
            out.append(pd.read_parquet(fp))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def loso(preds, feats):
    held = []
    for s in TEST:
        tr, te = preds[preds.season != s], preds[preds.season == s].copy()
        etr = {f: tr[f].to_numpy() for f in feats} if feats else None
        ete = {f: te[f].to_numpy() for f in feats} if feats else None
        mm = predict.MarginMap().fit(tr.pred_xg_margin, tr.home_margin, etr)
        te["p_home"] = mm.p_home_win(te.pred_xg_margin, ete)
        te["ok"] = ((mm.margin(te.pred_xg_margin, ete) > 0).astype(int) == te.home_win).astype(int)
        held.append(te)
    return pd.concat(held, ignore_index=True)


def summary(R, label):
    p = np.clip(R.p_home, 1e-6, 1 - 1e-6); y = R.home_win.to_numpy()
    ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    print(f"  {label:26s} acc={R.ok.mean():.3f}  logloss={ll:.4f}")


def main():
    seasons = ([PRIOR] if (ROOT / "data" / f"events_{PRIOR}.parquet").exists() else []) + TEST
    events = load_events(seasons)
    lu = load_lineups(seasons)
    if lu.empty:
        raise SystemExit("no lineups_*.parquet yet; run scripts.build_lineups first")
    gbs = {s: G.load_or_build(s) for s in seasons}

    preds, _ = rating.walk_forward(events, gbs, seasons, situation="all")
    feat = goalie.walk_forward_goalie(events, gbs, seasons)[["game_id", "goalie_diff"]]
    feat = feat.merge(rest.rest_features(gbs, seasons)[["game_id", "rest_diff"]], on="game_id")
    feat = feat.merge(special.walk_forward_st(events, gbs, seasons), on="game_id")
    feat = feat.merge(availability.walk_forward_availability(lu, gbs, seasons), on="game_id", how="left")
    preds = preds.merge(feat, on="game_id", how="left")
    preds[BASE + ["linedev_diff"]] = preds[BASE + ["linedev_diff"]].fillna(0.0)

    print("=== OVERALL (LOSO, 3 test seasons) ===")
    R0 = loso(preds, BASE)
    R1 = loso(preds, BASE + ["linedev_diff"])
    summary(R0, "best (no availability)")
    summary(R1, "best + availability")

    print("\n=== BY CONFIDENCE — does it move the toss-ups? ===")
    print(f"{'band':>10} {'n':>5} {'acc no-avail':>13} {'acc +avail':>11}")
    for lo, hi in [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 1.01)]:
        c0 = np.maximum(R0.p_home, 1 - R0.p_home)
        c1 = np.maximum(R1.p_home, 1 - R1.p_home)
        m0 = R0[(c0 >= lo) & (c0 < hi)]; m1 = R1[(c1 >= lo) & (c1 < hi)]
        print(f"{lo:.2f}-{hi if hi<1 else 1.0:.2f} {len(m1):5d} {m0.ok.mean():13.3f} {m1.ok.mean():11.3f}")

    # coefficient sign check
    mm = predict.MarginMap().fit(preds.pred_xg_margin, preds.home_margin,
                                 {f: preds[f].to_numpy() for f in BASE + ["linedev_diff"]})
    print("\nfitted feature coefficients:", dict(zip(mm.features, np.round(mm.betas, 3))))


if __name__ == "__main__":
    main()
