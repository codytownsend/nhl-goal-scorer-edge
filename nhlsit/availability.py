"""Track-B: per-game lineup-availability signal ("who's playing tonight").

Idea: a team missing key players is weaker than its season rating implies. We
estimate each skater's ESTABLISHED role by their walk-forward average TOI (coaches
give minutes to their best players), then for each game sum the established value
of the skaters who actually dressed. The orthogonal injury/scratch signal is the
DEVIATION of tonight's dressed value from the team's own running norm — negative
means key minutes are missing.

Leak-free: player values use only prior games; the dressed set is known at puck
drop; the team baseline uses only prior games.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# positional priors (seconds) for players with little/no history (call-ups)
POS_DEFAULT = {"forwards": 12 * 60, "defense": 17 * 60}
PLAYER_PC = 5      # pseudo-games shrinking a player's value toward the prior
CARRY = 0.8        # keep this fraction of a player's history across seasons


def walk_forward_availability(lineups: pd.DataFrame, games_by_season: dict,
                              seasons: list) -> pd.DataFrame:
    allg = pd.concat([games_by_season[s] for s in seasons], ignore_index=True)
    allg = allg[["game_id", "date", "season", "home", "away"]].copy()
    allg["date"] = pd.to_datetime(allg["date"])

    lu = lineups.merge(allg[["game_id", "date", "season"]], on="game_id", how="inner")
    lu = lu.sort_values(["date", "game_id"]).reset_index(drop=True)

    # 1) walk-forward player value (established role, in seconds)
    cum, cnt, last_season = {}, {}, {}
    pval = []
    for r in lu.itertuples(index=False):
        pid = r.player_id
        if pid in last_season and last_season[pid] != r.season:
            cum[pid] *= CARRY
            cnt[pid] *= CARRY
        n = cnt.get(pid, 0.0)
        prior = POS_DEFAULT.get(r.pos, 12 * 60)
        pval.append((cum.get(pid, 0.0) + prior * PLAYER_PC) / (n + PLAYER_PC))
        cum[pid] = cum.get(pid, 0.0) + r.toi_sec
        cnt[pid] = n + 1
        last_season[pid] = r.season
    lu["pval"] = pval

    # 2) dressed established value per team-game (minutes)
    linev = lu.groupby(["game_id", "team"])["pval"].sum().div(60.0).rename("linev").reset_index()
    linev = linev.merge(allg[["game_id", "date", "season"]], on="game_id")
    linev = linev.sort_values(["date", "game_id"])

    # 3) deviation from the team's running (walk-forward) baseline
    run_sum, run_cnt, last_s = {}, {}, {}
    dev = {}
    for r in linev.itertuples(index=False):
        t = r.team
        if t in last_s and last_s[t] != r.season:
            run_sum[t] = run_cnt[t] = 0.0
        n = run_cnt.get(t, 0.0)
        baseline = run_sum.get(t, 0.0) / n if n > 0 else r.linev
        dev[(r.game_id, t)] = r.linev - baseline
        run_sum[t] = run_sum.get(t, 0.0) + r.linev
        run_cnt[t] = n + 1
        last_s[t] = r.season

    rows = []
    for r in allg.itertuples(index=False):
        rows.append({"game_id": r.game_id,
                     "linedev_diff": dev.get((r.game_id, r.home), 0.0)
                     - dev.get((r.game_id, r.away), 0.0)})
    return pd.DataFrame(rows)


def walk_forward_quality_availability(lineups: pd.DataFrame, player_quality: dict,
                                      games_by_season: dict, seasons: list) -> pd.DataFrame:
    """Stage-2: quality-weighted lineup deviation.

    Each dressed skater contributes (pre-game on-ice xG share - 0.5) * established
    TOI(min): missing a high-quality, high-minutes player hurts much more than
    missing a depth player. Feature = home dressed-quality deviation - away's,
    each relative to the team's running (walk-forward) baseline. `player_quality`
    is {(game_id, player_id): pre-game xg_share} from player_ratings.
    """
    allg = pd.concat([games_by_season[s] for s in seasons], ignore_index=True)
    allg = allg[["game_id", "date", "season", "home", "away"]].copy()
    allg["date"] = pd.to_datetime(allg["date"])
    lu = lineups.merge(allg[["game_id", "date", "season"]], on="game_id", how="inner")
    lu = lu.sort_values(["date", "game_id"]).reset_index(drop=True)

    # walk-forward established TOI (role), same as the TOI version
    cum, cnt, last_season, toi_val = {}, {}, {}, []
    for r in lu.itertuples(index=False):
        pid = r.player_id
        if pid in last_season and last_season[pid] != r.season:
            cum[pid] *= CARRY; cnt[pid] *= CARRY
        n = cnt.get(pid, 0.0)
        prior = POS_DEFAULT.get(r.pos, 12 * 60)
        toi_val.append((cum.get(pid, 0.0) + prior * PLAYER_PC) / (n + PLAYER_PC))
        cum[pid] = cum.get(pid, 0.0) + r.toi_sec; cnt[pid] = n + 1; last_season[pid] = r.season
    lu["toi_min"] = np.array(toi_val) / 60.0
    # quality contribution = (pre-game share - 0.5) * established minutes
    lu["q"] = lu.apply(lambda x: (player_quality.get((x["game_id"], x["player_id"]), 0.5) - 0.5)
                       * x["toi_min"], axis=1)

    qual = lu.groupby(["game_id", "team"])["q"].sum().rename("qual").reset_index()
    qual = qual.merge(allg[["game_id", "date", "season"]], on="game_id").sort_values(["date", "game_id"])
    run_sum, run_cnt, last_s, dev = {}, {}, {}, {}
    for r in qual.itertuples(index=False):
        t = r.team
        if t in last_s and last_s[t] != r.season:
            run_sum[t] = run_cnt[t] = 0.0
        n = run_cnt.get(t, 0.0)
        baseline = run_sum.get(t, 0.0) / n if n > 0 else r.qual
        dev[(r.game_id, t)] = r.qual - baseline
        run_sum[t] = run_sum.get(t, 0.0) + r.qual; run_cnt[t] = n + 1; last_s[t] = r.season

    return pd.DataFrame([{"game_id": r.game_id,
                          "qavail_diff": dev.get((r.game_id, r.home), 0.0)
                          - dev.get((r.game_id, r.away), 0.0)}
                         for r in allg.itertuples(index=False)])
