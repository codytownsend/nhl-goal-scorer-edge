"""Build the tidy situational event table for a season.

Usage:
    python -m scripts.build_events 20232024 [--playoffs] [--limit N]

Writes data/events_<season>.parquet with an `xg` column attached.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from nhlsit import fetch, parse, xg

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("season", help="8-digit season, e.g. 20232024")
    ap.add_argument("--playoffs", action="store_true", help="include playoff games")
    ap.add_argument("--limit", type=int, default=0, help="cap #games (dev)")
    args = ap.parse_args()

    gtypes = (2, 3) if args.playoffs else (2,)
    print(f"Discovering game IDs for {args.season} ...")
    ids = fetch.season_game_ids(args.season, game_types=gtypes)
    if args.limit:
        ids = ids[: args.limit]
    print(f"  {len(ids)} games")

    print("Fetching + parsing play-by-play (cached) ...")
    events = parse.parse_games(ids)
    print(f"  {len(events):,} shot-attempt events")

    print("Fitting simple xG model + attaching xg ...")
    events = xg.add_xg(events)

    out = ROOT / "data" / f"events_{args.season}.parquet"
    events.to_parquet(out, index=False)
    print(f"Wrote {out}  ({len(events):,} rows)")


if __name__ == "__main__":
    main()
