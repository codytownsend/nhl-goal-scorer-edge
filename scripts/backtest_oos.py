"""Out-of-sample market backtest on 2024-25 & 2025-26 closing lines.

The model is built on 2020-21..2023-24; these two seasons are fully OUT OF
SAMPLE. Ratings are walk-forward (leak-free); the margin map + feature betas are
fit ONLY on the development seasons and applied forward. We then compare to the
ESPN BET closing lines: straight-up accuracy, calibration, and flat-stake
moneyline ROI (two-sided, real prices, correct juiced break-even) — overall and
for targeted selection strategies. ROI is reported with a t-stat: assume small
edges are noise until proven otherwise.

    python -m scripts.backtest_oos
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nhlsit import availability, finishing, goalie, games as G, market, predict, rating, rest, special

ROOT = Path(__file__).resolve().parent.parent
DEV = ["20202021", "20212022", "20222023", "20232024"]   # model development
OOS = ["20242025", "20252026"]                            # out-of-sample test
ALL = DEV + OOS
FEATS = ["goalie_diff", "rest_diff", "st_diff", "linedev_diff", "finish_diff"]
ODDS = {"20242025": "NHL 24-25.xlsx", "20252026": "NHL 25-26.xlsx"}


def load_events(seasons):
    out = []
    for s in seasons:
        fp = ROOT / "data" / f"events_{s}.parquet"
        if fp.exists():
            out.append(pd.read_parquet(fp).assign(season=str(s)))
    return pd.concat(out, ignore_index=True)


def build_preds():
    events = load_events(ALL)
    gbs = {s: G.load_or_build(s) for s in ALL}
    preds, _ = rating.walk_forward(events, gbs, ALL, situation="all")
    preds = preds.merge(goalie.walk_forward_goalie(events, gbs, ALL)[["game_id", "goalie_diff"]], on="game_id", how="left")
    preds = preds.merge(rest.rest_features(gbs, ALL)[["game_id", "rest_diff"]], on="game_id", how="left")
    preds = preds.merge(special.walk_forward_st(events, gbs, ALL), on="game_id", how="left")
    lu = [pd.read_parquet(ROOT / "data" / f"lineups_{s}.parquet") for s in ALL
          if (ROOT / "data" / f"lineups_{s}.parquet").exists()]
    if lu:
        lineups = pd.concat(lu)
        preds = preds.merge(availability.walk_forward_availability(lineups, gbs, ALL), on="game_id", how="left")
        preds = preds.merge(finishing.walk_forward_finishing(events, lineups, gbs, ALL), on="game_id", how="left")
    else:
        preds["linedev_diff"] = 0.0
        preds["finish_diff"] = 0.0
    preds[FEATS] = preds[FEATS].fillna(0.0)
    return preds


def roi(df, label):
    """Flat-stake ROI betting the model's +EV side(s) at real prices."""
    rows = []
    for r in df.itertuples(index=False):
        for side, pm, praw, ml in (("h", r.p_home, r.mkt_home_raw, r.home_ml),
                                   ("a", 1 - r.p_home, r.mkt_away_raw, r.away_ml)):
            if pm > praw:                                  # model sees value
                won = (r.hw == 1) if side == "h" else (r.hw == 0)
                dec = market.american_to_decimal(ml)
                rows.append((dec - 1) if won else -1.0)
    if not rows:
        print(f"  {label:34s} no +EV bets"); return
    a = np.array(rows); se = a.std() / np.sqrt(len(a))
    print(f"  {label:34s} n={len(a):4d} hit={ (a>0).mean():.3f} ROI={a.mean():+.2%} "
          f"t={a.mean()/se:+.2f} units={a.sum():+.1f}")


def main():
    preds = build_preds()

    # fit the margin map + feature betas on DEVELOPMENT seasons only, then apply
    # forward to the out-of-sample seasons (truly held out).
    dev = preds[preds.season.isin(DEV)]
    mm = predict.MarginMap().fit(dev.pred_xg_margin, dev.home_margin,
                                 {f: dev[f].to_numpy() for f in FEATS})
    print("margin map (fit on 2020-24):", mm.describe())
    ex = {f: preds[f].to_numpy() for f in FEATS}
    preds["p_home"] = mm.p_home_win(preds.pred_xg_margin, ex)

    odds = pd.concat([market.load_twosided_xlsx(ROOT / p) for p in ODDS.values()])
    m = preds.merge(odds, on=["date", "home", "away"], how="inner")   # preds keeps `season`
    m["hw"] = m["home_win"]
    m["conf"] = np.maximum(m.p_home, 1 - m.p_home)
    m["model_home"] = (m.p_home >= 0.5).astype(int)
    m["mkt_home"] = (m.mkt_home_devig >= 0.5).astype(int)
    print(f"matched {len(m)} of {len(odds)} odds games\n")

    for s in OOS:
        g = m[m.season == s]
        acc = (g.model_home == g.hw).mean(); mkt = (g.mkt_home == g.hw).mean()
        print(f"{s}: n={len(g):4d}  model acc={acc:.3f}  market acc={mkt:.3f}")
    acc = (m.model_home == m.hw).mean(); mkt = (m.mkt_home == m.hw).mean()
    p = np.clip(m.p_home, 1e-6, 1 - 1e-6); y = m.hw.to_numpy()
    ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    print(f"\nOVERALL OOS: model acc={acc:.3f}  market acc={mkt:.3f}  model logloss={ll:.4f}")

    print("\n=== accuracy by our confidence, model vs market ===")
    for lo, hi in [(.5, .55), (.55, .6), (.6, .65), (.65, 1.01)]:
        g = m[(m.conf >= lo) & (m.conf < hi)]
        if len(g):
            print(f"  {lo:.2f}-{min(hi,1):.2f}: n={len(g):4d} model={(g.model_home==g.hw).mean():.3f} "
                  f"market={(g.mkt_home==g.hw).mean():.3f}")

    print("\n=== flat-stake moneyline ROI (real prices, +EV side) ===")
    roi(m, "all +EV bets")
    roi(m[m.conf >= 0.60], "confident (>=60%)")
    roi(m[m.model_home != m.mkt_home], "disagreements with market")
    roi(m[(m.model_home != m.mkt_home) & (m.conf >= 0.60)], "confident disagreements")
    roi(m[m.linedev_diff.abs() >= m.linedev_diff.abs().quantile(0.80)], "big availability edge (top 20%)")
    roi(m[m.goalie_diff.abs() >= m.goalie_diff.abs().quantile(0.80)], "big goalie edge (top 20%)")


if __name__ == "__main__":
    main()
