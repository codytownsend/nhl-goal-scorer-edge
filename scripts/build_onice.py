"""Cache per-(game, player) 5v5 on-ice xG-for / against for all seasons.

This is the raw material the simulator needs: a player's chance-generation
level = on-ice xGF per minute, built walk-forward from these per-game totals.
Heavy (shot-by-shot shift reconstruction over ~7800 games) so we cache it once.
"""
import pandas as pd

from nhlsit import player_ratings as PR

SEASONS = [20202021, 20212022, 20222023, 20232024, 20242025, 20252026]


def main():
    frames = []
    for s in SEASONS:
        ev = pd.read_parquet(f"data/events_{s}.parquet")
        lu = pd.read_parquet(f"data/lineups_{s}.parquet")
        gids = pd.read_parquet(f"data/games_{s}.parquet")["game_id"].tolist()
        print(f"season {s}: {len(gids)} games", flush=True)
        pg = PR.player_game_onice(gids, ev, lu, strength="5v5", progress=True)
        pg["season"] = s
        frames.append(pg)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet("data/onice_playergames.parquet")
    print("wrote data/onice_playergames.parquet", out.shape, flush=True)


if __name__ == "__main__":
    main()
