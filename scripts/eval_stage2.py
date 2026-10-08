"""Stage-2: does QUALITY-weighted availability beat TOI-weighted?

Builds walk-forward player on-ice ratings (from shifts) and a quality-weighted
lineup-deviation feature, then compares it to the current TOI-based availability
in LOSO. If knowing *who* is missing (star vs depth) beats knowing *how many
minutes* are missing, quality-availability improves log-loss / accuracy.

    python -m scripts.eval_stage2
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nhlsit import availability, fetch, games as G, goalie, player_ratings as PR, predict, rating, rest, special

ROOT = Path(__file__).resolve().parent.parent
PRIOR = "20202021"
TEST = ["20212022", "20222023", "20232024"]
SEASONS = [PRIOR] + TEST


def load_events(seasons):
    return pd.concat([pd.read_parquet(ROOT / "data" / f"events_{s}.parquet").assign(season=str(s))
                      for s in seasons], ignore_index=True)


def main():
    events = load_events(SEASONS)
    lineups = pd.concat([pd.read_parquet(ROOT / "data" / f"lineups_{s}.parquet") for s in SEASONS])
    gbs = {s: G.load_or_build(s) for s in SEASONS}
    allg = pd.concat([gbs[s] for s in SEASONS], ignore_index=True)

    preds, _ = rating.walk_forward(events, gbs, SEASONS, situation="all")
    preds = preds.merge(goalie.walk_forward_goalie(events, gbs, SEASONS)[["game_id", "goalie_diff"]], on="game_id", how="left")
    preds = preds.merge(rest.rest_features(gbs, SEASONS)[["game_id", "rest_diff"]], on="game_id", how="left")
    preds = preds.merge(special.walk_forward_st(events, gbs, SEASONS), on="game_id", how="left")
    preds = preds.merge(availability.walk_forward_availability(lineups, gbs, SEASONS), on="game_id", how="left")

    # ---- Stage-2 quality-weighted availability ----
    all_ids = list(allg.game_id.unique())
    print("computing on-ice player contributions (needs shifts)...", flush=True)
    pg = PR.player_game_onice(all_ids, events, lineups)
    pq = PR.walk_forward_quality(pg, allg[["game_id", "date", "season"]].assign(
        season=allg["season"].astype(str)))
    qav = availability.walk_forward_quality_availability(lineups, pq, gbs, SEASONS)
    preds = preds.merge(qav, on="game_id", how="left")

    for c in ["goalie_diff", "rest_diff", "st_diff", "linedev_diff", "qavail_diff"]:
        preds[c] = preds[c].fillna(0.0)

    def loso(feats):
        H = []
        for s in TEST:
            tr, te = preds[preds.season != s], preds[preds.season == s].copy()
            etr = {f: tr[f].to_numpy() for f in feats}; ete = {f: te[f].to_numpy() for f in feats}
            mm = predict.MarginMap().fit(tr.pred_xg_margin, tr.home_margin, etr)
            p = np.clip(mm.p_home_win(te.pred_xg_margin, ete), 1e-6, 1 - 1e-6); y = te.home_win.to_numpy()
            te["ll"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
            te["ok"] = ((mm.margin(te.pred_xg_margin, ete) > 0).astype(int) == y).astype(int)
            H.append(te)
        R = pd.concat(H); return R.ll.mean(), R.ok.mean()

    base = ["goalie_diff", "rest_diff", "st_diff"]
    print("\n=== LOSO (2021-24) ===")
    for name, feats in [("no availability", base),
                        ("+ TOI availability (current)", base + ["linedev_diff"]),
                        ("+ QUALITY availability", base + ["qavail_diff"]),
                        ("+ both", base + ["linedev_diff", "qavail_diff"])]:
        ll, acc = loso(feats)
        print(f"  {name:30s} logloss={ll:.4f} acc={acc:.3f}")

    # coefficient + correlation sanity
    print(f"\ncorr(TOI-avail, quality-avail): {preds.linedev_diff.corr(preds.qavail_diff):.3f}")


if __name__ == "__main__":
    main()
