"""Opponent-adjusted, walk-forward team ratings from 5v5 xG differential.

Model (per game, home perspective):
    home_xg_diff  ~  home_adv + rating[home] - rating[away]

Ratings are solved by ridge-regularized least squares (a Massey-style system).
The ridge pulls each team toward a *prior* (last season's final rating x carry);
because the data terms grow as a team plays more games, the prior dominates
early and fades naturally — this is the decaying cross-season prior from the
spec, implemented as shrinkage. Everything is walk-forward: the rating used to
predict game g is solved from only the games played before g.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import fetch

TEAMS = fetch.TEAMS  # canonical 32; extra teams (e.g. ARI) handled dynamically


def _select(events: pd.DataFrame, situation: str) -> pd.DataFrame:
    """Filter events to the situation set used for the rating metric."""
    if situation == "5v5":
        return events[events["strength_bucket"] == "5v5"]
    if situation == "nonempty":               # everything except empty-net noise
        return events[events["strength_bucket"] != "empty_net"]
    return events                              # "all": 5v5 + PP + PK + 4v4/3v3 + EN


def score_weights(events: pd.DataFrame, situation: str = "5v5") -> dict:
    """Score-adjustment weights: neutralize the fact that trailing teams
    generate more xG (score effects) regardless of quality. weight[state] =
    (mean xGF across states) / (mean xGF in that state), so inflated (trailing)
    states are down-weighted and suppressed (leading) states up-weighted."""
    ev = _select(events, situation)
    per = ev.groupby(["score_state"])["xg"].mean()
    base = per.mean()
    w = (base / per).to_dict()
    return {s: float(v) for s, v in w.items()}


def game_metric_table(events: pd.DataFrame, games: pd.DataFrame,
                      situation: str = "5v5", score_adjust: bool = False,
                      score_wts: dict | None = None) -> pd.DataFrame:
    """Per game: home/away xG (optionally all-situations & score-adjusted) and
    the home xG differential used to fit ratings."""
    ev = _select(events, situation).copy()
    if score_adjust:
        w = score_wts or score_weights(events, situation)
        ev["xg"] = ev["xg"] * ev["score_state"].map(w).fillna(1.0)
    xg_for = ev.groupby(["game_id", "team"])["xg"].sum().rename("xgf").reset_index()
    g = games.merge(
        xg_for.rename(columns={"team": "home", "xgf": "home_xgf"}),
        on=["game_id", "home"], how="left")
    g = g.merge(
        xg_for.rename(columns={"team": "away", "xgf": "away_xgf"}),
        on=["game_id", "away"], how="left")
    g[["home_xgf", "away_xgf"]] = g[["home_xgf", "away_xgf"]].fillna(0.0)
    g["home_xg_diff"] = g["home_xgf"] - g["away_xgf"]
    return g


# backwards-compatible alias
def game_xg_table(events, games, strength="5v5"):
    return game_metric_table(events, games, situation=strength)


class RatingSolver:
    """Incremental ridge solver for the Massey-style system.

    Accumulate games with `add`, read current ratings with `solve`. Reset per
    season and seed the prior from the previous season's final ratings.
    """

    def __init__(self, teams, prior=None, lam=3.0, lam_home=5.0, home_prior=0.15):
        self.teams = list(teams)
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.T = len(self.teams)
        self.p = self.T + 1                      # +1 for home_adv
        self.AtA = np.zeros((self.p, self.p))
        self.Aty = np.zeros(self.p)
        self.lam = lam
        self.lam_home = lam_home
        self.home_prior = home_prior
        self.prior = np.zeros(self.T)
        if prior:
            for t, v in prior.items():
                if t in self.idx:
                    self.prior[self.idx[t]] = v

    def _f(self, home, away):
        f = np.zeros(self.p)
        f[0] = 1.0
        f[self.idx[home] + 1] = 1.0
        f[self.idx[away] + 1] = -1.0
        return f

    def decay(self, d):
        """Exponentially down-weight all accumulated games (recency)."""
        if d < 1.0:
            self.AtA *= d
            self.Aty *= d

    def add(self, home, away, y):
        if home not in self.idx or away not in self.idx:
            return
        f = self._f(home, away)
        self.AtA += np.outer(f, f)
        self.Aty += f * y

    def solve(self) -> dict:
        R = np.zeros((self.p, self.p))
        R[0, 0] = self.lam_home
        for i in range(1, self.p):
            R[i, i] = self.lam
        rhs = self.Aty.copy()
        rhs[0] += self.lam_home * self.home_prior
        rhs[1:] += self.lam * self.prior
        x = np.linalg.solve(self.AtA + R, rhs)
        # center team ratings so home_adv absorbs the mean
        team = x[1:] - x[1:].mean()
        return {"home_adv": float(x[0]),
                "ratings": {t: float(team[i]) for t, i in self.idx.items()}}


def walk_forward(events: pd.DataFrame, games_by_season: dict,
                 seasons: list, carry=0.75, lam=3.0,
                 situation="5v5", score_adjust=False, recency=1.0) -> pd.DataFrame:
    """Leak-free predictions for every game across `seasons` (chronological).

    situation: '5v5' | 'nonempty' | 'all'.  score_adjust: neutralize score
    effects.  recency: per-game decay (<1 weights recent games more).
    Returns predictions + end-of-season ratings for seeding.
    """
    # score-adjustment weights are fit on ALL supplied events (leak-free enough:
    # they're a stable league-wide constant, not team- or game-specific).
    swts = score_weights(events, situation) if score_adjust else None
    prior = {}
    final_ratings = {}
    out = []
    for season in seasons:
        games = games_by_season[season]
        gx = game_metric_table(events[events["season"].astype(str) == season], games,
                               situation=situation, score_adjust=score_adjust,
                               score_wts=swts)
        teams = sorted(set(gx["home"]) | set(gx["away"]) | set(prior))
        solver = RatingSolver(teams, prior=prior, lam=lam)
        for _, r in gx.iterrows():
            cur = solver.solve()
            rh = cur["ratings"].get(r["home"], 0.0)
            ra = cur["ratings"].get(r["away"], 0.0)
            out.append({
                "game_id": r["game_id"], "date": r["date"], "season": season,
                "home": r["home"], "away": r["away"],
                "pred_xg_margin": cur["home_adv"] + rh - ra,
                "home_adv": cur["home_adv"],
                "home_margin": r["home_margin"], "home_win": r["home_win"],
                "extra_time": r["extra_time"],
            })
            solver.decay(recency)
            solver.add(r["home"], r["away"], r["home_xg_diff"])
        final = solver.solve()
        final_ratings[season] = final          # {"home_adv":..., "ratings":{...}}
        prior = {t: carry * v for t, v in final["ratings"].items()}
    return pd.DataFrame(out), final_ratings


def current_state(events: pd.DataFrame, games_by_season: dict, seasons: list,
                  carry=0.75, lam=3.0, **kw):
    """Full-season-solved ratings + home_adv for the latest season, seeded by
    the prior-season chain. Use these to rate an upcoming matchup."""
    _, final = walk_forward(events, games_by_season, seasons, carry, lam, **kw)
    return final[seasons[-1]]
