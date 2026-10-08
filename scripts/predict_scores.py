"""Expected game scores + confidence, with out-of-sample confidence validation.

Fits the model + confidence table on 2020-24, then for the out-of-sample seasons
(2024-26): prints sample expected scores, and checks that each confidence tier
actually wins at its stated rate on games the model never saw.

    python -m scripts.predict_scores
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nhlsit import games as G, predict, scoreline as SL
from scripts.backtest_oos import build_preds

ROOT = Path(__file__).resolve().parent.parent
DEV = ["20202021", "20212022", "20222023", "20232024"]
OOS = ["20242025", "20252026"]
FEATS = ["goalie_diff", "rest_diff", "st_diff", "linedev_diff", "finish_diff"]


def main():
    preds = build_preds()
    # attach names + actual scores for display
    allg = pd.concat([G.load_or_build(s) for s in DEV + OOS])[
        ["game_id", "date", "home", "away", "home_goals", "away_goals", "home_win", "home_margin"]]
    total = (allg.home_goals + allg.away_goals).mean()

    dev = preds[preds.season.isin(DEV)]
    mm = predict.MarginMap().fit(dev.pred_xg_margin, dev.home_margin, {f: dev[f].to_numpy() for f in FEATS})
    ex = {f: preds[f].to_numpy() for f in FEATS}
    preds["p_home"] = mm.p_home_win(preds.pred_xg_margin, ex)
    preds["margin"] = mm.margin(preds.pred_xg_margin, ex)

    # confidence table learned on DEV
    ctab = SL.fit_confidence(preds[preds.season.isin(DEV)])
    print(f"expected total used: {total:.2f} goals/game\n")
    print("Confidence table (learned on 2020-24):")
    print(ctab.assign(hit_rate=ctab.hit_rate.round(3), model_conf=ctab.model_conf.round(3)).to_string(index=False))

    oos = preds[preds.season.isin(OOS)].merge(allg[["game_id", "home_goals", "away_goals"]], on="game_id")
    print("\nSample out-of-sample predictions (with actual result):")
    for r in oos.sort_values("date").head(10).itertuples(index=False):
        line = SL.describe(r.home, r.away, r.p_home, r.margin, ctab, total)
        print(f"  {line}   [actual {r.away} {int(r.away_goals)}-{int(r.home_goals)} {r.home}]")

    # --- validate confidence OUT OF SAMPLE ---
    oos = oos.assign(pick_p=np.maximum(oos.p_home, 1 - oos.p_home),
                     correct=((oos.p_home >= 0.5).astype(int) == oos.home_win).astype(int))
    oos["tier"] = oos.pick_p.apply(lambda p: SL.confidence(p)["tier"])
    print("\n=== confidence validation (stated vs ACTUAL out-of-sample) ===")
    print(f"{'tier':>11} {'n':>5} {'stated (DEV)':>13} {'actual (OOS)':>13}")
    stated = {}
    for r in ctab.itertuples(index=False):
        pass
    for tier in ["Very high", "High", "Medium", "Lean", "Coin-flip"]:
        g = oos[oos.tier == tier]
        if len(g) == 0:
            continue
        # stated = mean DEV hit-rate for that tier
        d = preds[preds.season.isin(DEV)].assign(
            pick_p=lambda x: np.maximum(x.p_home, 1 - x.p_home),
            correct=lambda x: ((x.p_home >= 0.5).astype(int) == x.home_win).astype(int))
        d["tier"] = d.pick_p.apply(lambda p: SL.confidence(p)["tier"])
        st = d[d.tier == tier].correct.mean()
        print(f"{tier:>11} {len(g):5d} {st:13.3f} {g.correct.mean():13.3f}")

    # score quality
    oos["pred_total"] = total
    print(f"\nexpected-score check: predicted margin corr with actual margin = "
          f"{oos.margin.corr(oos.home_margin):.3f}")


if __name__ == "__main__":
    main()
