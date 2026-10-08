"""Direction B, Stage 1 validation: do the player on-ice ratings make sense?

Builds 2023-24 5v5 on-ice xG-share per player and prints the top/bottom regulars.
If the pipeline is right, the top of the list should be recognizable stars.
Also checks that a bottom-up (player-based) team rating tracks the team's actual
5v5 xGF% — confirming the aggregation is coherent before we go walk-forward.

    python -m scripts.player_stage1
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nhlsit import fetch, player_ratings as PR

ROOT = Path(__file__).resolve().parent.parent
SEASON = "20232024"


def main():
    events = pd.read_parquet(ROOT / "data" / f"events_{SEASON}.parquet")
    lineups = pd.read_parquet(ROOT / "data" / f"lineups_{SEASON}.parquet")
    ids = fetch.season_game_ids(SEASON, game_types=(2,))

    pr = PR.build_player_onice(ids, events, lineups)
    names = PR.id_to_name(ids)
    pr["name"] = pr.player_id.map(names)
    pr["exposure"] = pr.on_xgf + pr.on_xga           # on-ice xG involvement
    reg = pr[pr.exposure >= pr.exposure.quantile(0.5)].copy()   # regulars

    print(f"players: {len(pr)} | regulars: {len(reg)}\n")
    print("=== TOP 15 regulars by 5v5 on-ice xG share ===")
    print(reg.sort_values("xg_share", ascending=False)
          .head(15)[["name", "xg_share", "on_xgf", "on_xga"]].round(3).to_string(index=False))
    print("\n=== BOTTOM 10 regulars ===")
    print(reg.sort_values("xg_share")
          .head(10)[["name", "xg_share", "on_xgf", "on_xga"]].round(3).to_string(index=False))

    # bottom-up team rating (TOI-weighted player share) vs actual team 5v5 xGF%
    pv = pr.set_index("player_id")["xg_share"]
    lu = lineups.copy()
    lu["pr"] = lu.player_id.map(pv)
    lu = lu.dropna(subset=["pr"])
    team_bottom = (lu.assign(w=lu.pr * lu.toi_sec).groupby("team")
                   .apply(lambda d: d.w.sum() / d.toi_sec.sum())).rename("bottom_up")

    ev5 = events[events.strength_bucket == "5v5"]
    tf = ev5.groupby("team").xg.sum(); ta = ev5.groupby("opponent").xg.sum()
    team_top = (tf / (tf + ta)).rename("team_xgf_pct")
    cmp = pd.concat([team_bottom, team_top], axis=1).dropna()
    print(f"\ncorr(bottom-up player rating, team 5v5 xGF%): {cmp.bottom_up.corr(cmp.team_xgf_pct):.3f}")
    print(cmp.sort_values("bottom_up", ascending=False).head(6).round(3).to_string())


if __name__ == "__main__":
    main()
