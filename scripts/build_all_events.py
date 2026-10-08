"""Parse ALL play-by-play events for a season -> data/allevents_<season>.parquet
(from cached pbp; no network). Foundation for the expected-event-value model.

    python -m scripts.build_all_events 20232024
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from nhlsit import events_all as EA, fetch

ROOT = Path(__file__).resolve().parent.parent


def main():
    season = sys.argv[1]
    ids = fetch.season_game_ids(season, game_types=(2,))
    print(f"{season}: {len(ids)} games", flush=True)
    frames = []
    for i, gid in enumerate(ids, 1):
        if i % 300 == 0 or i == len(ids):
            print(f"  {i}/{len(ids)}", flush=True)
        try:
            frames.append(EA.parse_all_events(gid))
        except Exception as e:  # noqa: BLE001
            print(f"  !! {gid}: {e}", flush=True)
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    out = ROOT / "data" / f"allevents_{season}.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df):,} events)", flush=True)


if __name__ == "__main__":
    main()
