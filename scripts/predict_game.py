"""Predict a single matchup BEFORE it happens, from current team ratings.

Uses the latest season's opponent-adjusted 5v5 xG ratings (seeded across
seasons) and the fitted margin->win-probability map. This is team-quality only:
no starting goalie or rest yet (those are the next levers to add).

    python -m scripts.predict_game --home EDM --away CGY
    python -m scripts.predict_game --home BOS --away TOR --neutral
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nhlsit import games as G
from nhlsit import predict, rating

ROOT = Path(__file__).resolve().parent.parent
SEASONS = ["20212022", "20222023", "20232024"]


def load_events(seasons):
    frames = []
    for s in seasons:
        df = pd.read_parquet(ROOT / "data" / f"events_{s}.parquet")
        df["season"] = str(s)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--neutral", action="store_true", help="drop home-ice edge")
    ap.add_argument("--seasons", nargs="*", default=SEASONS)
    args = ap.parse_args()

    events = load_events(args.seasons)
    games_by_season = {s: G.load_or_build(s) for s in args.seasons}

    state = rating.current_state(events, games_by_season, args.seasons)
    ratings, home_adv = state["ratings"], state["home_adv"]

    # fit the deployment margin map on all walk-forward predictions
    preds, _ = rating.walk_forward(events, games_by_season, args.seasons)
    mm = predict.MarginMap().fit(preds["pred_xg_margin"], preds["home_margin"])

    rh = ratings.get(args.home)
    ra = ratings.get(args.away)
    if rh is None or ra is None:
        raise SystemExit(f"unknown team(s); known: {sorted(ratings)}")

    edge = 0.0 if args.neutral else home_adv
    xg_margin = edge + rh - ra
    goal_margin = float(mm.margin(xg_margin))
    p_home = float(mm.p_home_win(xg_margin))

    fav, p = (args.home, p_home) if p_home >= 0.5 else (args.away, 1 - p_home)
    print(f"\n{args.away} @ {args.home}"
          + ("  (neutral)" if args.neutral else "") + "\n")
    print(f"  rating  {args.home}: {rh:+.2f}   {args.away}: {ra:+.2f}"
          + ("" if args.neutral else f"   home-ice: {home_adv:+.2f}"))
    print(f"  predicted xG margin (home): {xg_margin:+.2f}")
    print(f"  predicted goal margin (home): {goal_margin:+.2f}")
    print(f"  P({args.home} win): {p_home:.1%}   P({args.away} win): {1-p_home:.1%}")
    print(f"  --> pick {fav} ({p:.1%})\n")


if __name__ == "__main__":
    main()
