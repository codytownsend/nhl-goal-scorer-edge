"""Game-conditioned goal-scorer model (the generative/structured-lambda approach).

The rate model predicts a player's AVERAGE game: a season-long expected-goals
rate applied to every game regardless of that night's configuration. This module
tests the unexplored variance source (b): tonight's deployment differs from the
player's season norm (linemates injured/promoted, line juggling), and goals live
in those deviations.

Mechanism, per (game, player i):
    P(scores >=1) = 1 - exp(-lambda_i)
    lambda_i      = lambda0_i * m_i(game)
where
    lambda0_i   = player's walk-forward expected goals in an AVERAGE game
                  (so the model reduces to the rate model when m == 1), and
    m_i(game)   = linemate-support multiplier: how good tonight's actual
                  on-ice linemates are at generating chances, relative to the
                  player's own running-average linemate support this season.

Linemate QUALITY (q) is a walk-forward pre-game on-ice xGF/min rating (leak-free).
Linemate IDENTITY and time-on-ice-together come from THIS game's shift charts --
the "oracle deployment" upper bound (in a live forward test you'd get the same
from projected lines). If this upper bound can't beat the rate model on deviation
games, variance source (b) is not harvestable and we stop.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from . import shifts

Q_SHRINK_MIN = 100.0   # pseudo-minutes shrinking a player's on-ice rate to league
Q_CARRY = 0.8          # cross-season decay of on-ice-rate history
MIN_PRIOR_GAMES = 5    # need this many prior support obs before trusting m
SUPP_CARRY = 0.7       # cross-season decay of a player's running support average


def load_order(seasons: list) -> pd.DataFrame:
    g = pd.concat([pd.read_parquet(f"data/games_{s}.parquet") for s in seasons],
                  ignore_index=True)
    g = g[["game_id", "date", "season", "home", "away"]].copy()
    g["season"] = g["season"].astype(int)
    g["date"] = pd.to_datetime(g["date"])
    return g.sort_values(["date", "game_id"]).reset_index(drop=True)


def walk_forward_onice_rate(onice: pd.DataFrame, lineups: pd.DataFrame,
                            order: pd.DataFrame) -> dict:
    """Pre-game on-ice xGF per minute per (game, player), walk-forward.

    This is the 'chance-generation level' used to score a player's linemates.
    """
    pg = onice[["game_id", "player_id", "g_xgf"]].merge(
        lineups[["game_id", "player_id", "toi_sec"]],
        on=["game_id", "player_id"], how="left")
    pg = pg.merge(order[["game_id", "date", "season"]], on="game_id")
    pg["toi_min"] = pg["toi_sec"].fillna(0.0) / 60.0
    pg = pg.sort_values(["date", "game_id"])
    league = pg["g_xgf"].sum() / max(pg["toi_min"].sum(), 1.0)

    cumx, cummin, last = defaultdict(float), defaultdict(float), {}
    q = {}
    for r in pg.itertuples(index=False):
        p = r.player_id
        if p in last and last[p] != r.season:
            cumx[p] *= Q_CARRY
            cummin[p] *= Q_CARRY
        x, m = cumx[p], cummin[p]
        q[(r.game_id, p)] = (x + league * Q_SHRINK_MIN) / (m + Q_SHRINK_MIN)
        cumx[p] = x + (r.g_xgf or 0.0)
        cummin[p] = m + (r.toi_min or 0.0)
        last[p] = r.season
    return q, league


def _pairwise_together(gid: int, skset: set) -> dict:
    """{player: {teammate: seconds_on_ice_together}} from one game's shifts.

    Overlap is computed within (period, team), restricted to dressed skaters.
    """
    idx = shifts.on_ice_index(gid)
    by_team_per = defaultdict(list)        # (team, period) -> [(st, en, pid)]
    for (per, team), lst in idx.items():
        for st, en, pid in lst:
            if pid in skset:
                by_team_per[(team, per)].append((st, en, pid))

    together = defaultdict(lambda: defaultdict(float))
    for lst in by_team_per.values():
        n = len(lst)
        for a in range(n):
            sa, ea, pa = lst[a]
            for b in range(a + 1, n):
                sb, eb, pb = lst[b]
                if pa == pb:
                    continue
                ov = min(ea, eb) - max(sa, sb)
                if ov > 0:
                    together[pa][pb] += ov
                    together[pb][pa] += ov
    return together


def build_support(test_seasons: list, hist_seasons: list) -> pd.DataFrame:
    """Per (game, player) linemate-support and the deviation multiplier m.

    Returns columns: game_id, player_id, season, support, support_avg, m,
    n_prior (prior support obs). m == 1.0 until a player has MIN_PRIOR_GAMES.
    """
    all_seasons = sorted(set(hist_seasons) | set(test_seasons))
    order = load_order(all_seasons)
    onice = pd.read_parquet("data/onice_playergames.parquet")
    lineups = pd.concat([pd.read_parquet(f"data/lineups_{s}.parquet")
                         for s in all_seasons], ignore_index=True)

    q, _ = walk_forward_onice_rate(onice, lineups, order)

    # team + dressed-skater set per game (for same-team overlap + goalie filter)
    lu = lineups[["game_id", "team", "player_id"]]
    skset_by_game = {g: set(d.player_id) for g, d in lu.groupby("game_id")}

    test_games = order[order.season.isin(test_seasons)]
    run_sum, run_n, last_season = defaultdict(float), defaultdict(float), {}
    rows = []
    games = list(test_games.itertuples(index=False))
    for i, gr in enumerate(games, 1):
        if i % 250 == 0 or i == len(games):
            print(f"  support {i}/{len(games)}", flush=True)
        gid = gr.game_id
        skset = skset_by_game.get(gid, set())
        together = _pairwise_together(gid, skset)
        for p in skset:
            if p in last_season and last_season[p] != gr.season:
                run_sum[p] *= SUPP_CARRY
                run_n[p] *= SUPP_CARRY
            tmates = together.get(p, {})
            num = den = 0.0
            for j, secs in tmates.items():
                qj = q.get((gid, j))
                if qj is None:
                    continue
                num += secs * qj
                den += secs
            support = num / den if den > 0 else None

            n_prior = run_n[p]
            avg = run_sum[p] / n_prior if n_prior > 0 else None
            if support is not None and avg and n_prior >= MIN_PRIOR_GAMES:
                m = support / avg
            else:
                m = 1.0
            rows.append({"game_id": gid, "player_id": p, "season": gr.season,
                         "support": support, "support_avg": avg,
                         "n_prior": n_prior, "m": m})
            if support is not None:
                run_sum[p] += support
                run_n[p] += 1.0
            last_season[p] = gr.season
    return pd.DataFrame(rows)
