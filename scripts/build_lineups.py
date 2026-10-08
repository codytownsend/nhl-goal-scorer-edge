"""Pull boxscore lineups for a season -> data/lineups_<season>.parquet.

    python -m scripts.build_lineups 20232024
"""
from __future__ import annotations

import sys
from pathlib import Path

from nhlsit import fetch, lineups

ROOT = Path(__file__).resolve().parent.parent


def main():
    season = sys.argv[1]
    ids = fetch.season_game_ids(season, game_types=(2,))
    print(f"{season}: {len(ids)} games", flush=True)
    df = lineups.build_lineups(ids)
    out = ROOT / "data" / f"lineups_{season}.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out}  ({len(df):,} player-game rows)", flush=True)


if __name__ == "__main__":
    main()
