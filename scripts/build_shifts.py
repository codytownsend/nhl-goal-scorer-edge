"""Cache shift charts for a season (foundation for player on-ice ratings).

    python -m scripts.build_shifts 20232024
"""
from __future__ import annotations

import sys

from nhlsit import fetch, shifts


def main():
    season = sys.argv[1]
    ids = fetch.season_game_ids(season, game_types=(2,))
    print(f"{season}: {len(ids)} games", flush=True)
    ok = 0
    for i, gid in enumerate(ids, 1):
        rows = shifts.shiftcharts(gid)
        ok += 1 if rows else 0
        if i % 250 == 0 or i == len(ids):
            print(f"  shifts {i}/{len(ids)} (non-empty {ok})", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
