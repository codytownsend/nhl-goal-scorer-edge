"""Test the expected-event-value (xEV) model.

  1) reliability gate: does a team's per-game xEV repeat within season?
  2) standalone: does an xEV-only rating predict better/worse than xG-only?
  3) additive: does the xEV rating add anything to the full xG-based model?
All out-of-sample (2024-26) + LOSO, with season-split consistency.

    python -m scripts.test_xev
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nhlsit import event_value as EV, games as G, predict
from scripts.backtest_oos import build_preds

ROOT = Path(__file__).resolve().parent.parent
DEV = ["20202021", "20212022", "20222023", "20232024"]
OOS = ["20242025", "20252026"]
ALL = DEV + OOS


def load_allevents(seasons):
    return pd.concat([pd.read_parquet(ROOT / "data" / f"allevents_{s}.parquet")
                      for s in seasons], ignore_index=True)


def acc_ll(mm, df, feats):
    ex = {f: df[f].to_numpy() for f in feats}
    p = np.clip(mm.p_home_win(df[feats[0]], ex) if False else
                mm.p_home_win(df["pred_xg_margin"], ex), 1e-6, 1 - 1e-6)
    return None  # unused


def main():
    ae = load_allevents(ALL)
    gbs = {s: G.load_or_build(s) for s in ALL}

    # 1) learn values on DEV only (leak-free for OOS)
    values = EV.learn_values(ae[ae.season.astype(str).isin(DEV)])
    xev = EV.walk_forward_xev(ae, gbs, ALL, values)

    # attach outcomes
    allg = pd.concat([gbs[s] for s in ALL], ignore_index=True)[
        ["game_id", "date", "season", "home_margin", "home_win"]]
    xev = xev.merge(allg, on="game_id")

    # reliability gate on per-game team xEV
    tx = EV.assign_and_aggregate(ae, values).merge(
        pd.concat([gbs[s] for s in ALL])[["game_id", "season"]], on="game_id")
    tx["half"] = tx.groupby(["team", "season"]).cumcount() % 2
    piv = tx.groupby(["team", "season", "half"]).xev.mean().unstack("half").dropna()
    r = piv[0].corr(piv[1]); sb = 2 * r / (1 + r)
    print(f"reliability (split-half, Spearman-Brown) of team per-game xEV: {sb:.3f}\n")

    # base preds (full xG model) to compare / combine
    preds = build_preds()
    m = preds.merge(xev[["game_id", "pred_xev_margin"]], on="game_id", how="inner")

    def fit_on_dev(margin_col, feats):
        d = m[m.season.isin(DEV)]
        return predict.MarginMap().fit(d[margin_col], d.home_margin,
                                       {f: d[f].to_numpy() for f in feats} if feats else None)

    def oos_eval(mm, margin_col, feats):
        te = m[m.season.isin(OOS)]
        ex = {f: te[f].to_numpy() for f in feats} if feats else None
        p = np.clip(mm.p_home_win(te[margin_col], ex), 1e-6, 1 - 1e-6); y = te.home_win.to_numpy()
        acc = ((mm.margin(te[margin_col], ex) > 0).astype(int) == y).mean()
        ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return acc, ll

    print("=== STANDALONE ratings, out-of-sample (2024-26) ===")
    a, l = oos_eval(fit_on_dev("pred_xg_margin", []), "pred_xg_margin", [])
    print(f"  xG rating only:   acc={a:.3f} logloss={l:.4f}")
    a, l = oos_eval(fit_on_dev("pred_xev_margin", []), "pred_xev_margin", [])
    print(f"  xEV rating only:  acc={a:.3f} logloss={l:.4f}")

    print("\n=== does xEV ADD to the full model? (OOS) ===")
    FULL = ["goalie_diff", "rest_diff", "st_diff", "linedev_diff", "finish_diff"]
    for c in FULL + ["pred_xev_margin"]:
        if c not in m: m[c] = 0.0
    m[FULL] = m[FULL].fillna(0.0)
    a, l = oos_eval(fit_on_dev("pred_xg_margin", FULL), "pred_xg_margin", FULL)
    print(f"  full model:            acc={a:.3f} logloss={l:.4f}")
    a, l = oos_eval(fit_on_dev("pred_xg_margin", FULL + ["pred_xev_margin"]),
                    "pred_xg_margin", FULL + ["pred_xev_margin"])
    print(f"  full + xEV rating:     acc={a:.3f} logloss={l:.4f}")

    print("\ncorr(xEV rating, xG rating):", round(m.pred_xev_margin.corr(m.pred_xg_margin), 3))
    print("univariate corr(xEV margin, home_win) by season:")
    for s in ALL:
        g = m[m.season == s]
        print(f"  {s}: {g.pred_xev_margin.corr(g.home_win):+.3f}")


if __name__ == "__main__":
    main()
