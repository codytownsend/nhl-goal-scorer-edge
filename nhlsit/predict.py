"""Map predicted xG margin (+ optional extra features) -> goal margin -> P(win).

    goal_margin ~ A*tanh(xg_margin/s) + B·extras + c      (fit on TRAIN only)
    P(home win) = Phi(goal_margin_hat / sigma)

`extras` are extra per-game linear features in home perspective, e.g. goalie
GSAx differential and rest differential. Saturating tanh keeps thin-data
blowout ratings from producing absurd lines. The market is never used here.
"""
from __future__ import annotations

import math

import numpy as np


class MarginMap:
    def __init__(self):
        self.s = 1.0
        self.A = 1.0
        self.c = 0.0
        self.betas = np.zeros(0)
        self.features = []
        self.sigma = 2.0

    def _extra_matrix(self, extras, n):
        if not self.features:
            return np.zeros((n, 0))
        return np.column_stack([np.asarray(extras[f], float) for f in self.features])

    def fit(self, xg_margin, goal_margin, extras=None):
        x = np.asarray(xg_margin, float)
        y = np.asarray(goal_margin, float)
        self.features = list(extras.keys()) if extras else []
        E = self._extra_matrix(extras or {}, len(x))
        best = None
        for s in np.linspace(0.3, 6.0, 58):
            f = np.tanh(x / s).reshape(-1, 1)
            X = np.column_stack([f, E, np.ones_like(x)])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ coef
            rmse = float(np.sqrt(np.mean(resid ** 2)))
            if best is None or rmse < best[0]:
                best = (rmse, s, coef)
        rmse, s, coef = best
        self.s = s
        self.A = float(coef[0])
        self.betas = coef[1:1 + E.shape[1]]
        self.c = float(coef[-1])
        self.sigma = rmse
        return self

    def margin(self, xg_margin, extras=None):
        x = np.asarray(xg_margin, float)
        m = self.A * np.tanh(x / self.s) + self.c
        if self.features:
            E = self._extra_matrix(extras or {}, len(x))
            m = m + E @ self.betas
        return m

    def p_home_win(self, xg_margin, extras=None):
        z = self.margin(xg_margin, extras) / self.sigma
        return 0.5 * (1.0 + _erf_vec(z / math.sqrt(2.0)))

    def describe(self):
        parts = [f"A={self.A:.2f}", f"s={self.s:.2f}", f"c={self.c:+.2f}"]
        for f, b in zip(self.features, self.betas):
            parts.append(f"{f}={b:+.2f}")
        parts.append(f"sigma={self.sigma:.2f}")
        return "  ".join(parts)


def _erf_vec(z):
    return np.vectorize(math.erf)(np.asarray(z, float))


# --- Skellam / two-Poisson win probability (#6) --------------------------
# Model home goals ~ Poisson(lam_h), away ~ Poisson(lam_a) with
# lam_h - lam_a = expected margin (mu) and lam_h + lam_a = expected total.
# P(home win) includes a coin-flip on regulation ties (OT/SO). Gives totals free.

_MAXG = 16
_KGRID = np.arange(_MAXG)


def _pois_pmf(lam):
    lam = max(float(lam), 1e-3)
    logp = _KGRID * math.log(lam) - lam - np.array([math.lgamma(k + 1) for k in _KGRID])
    return np.exp(logp)


def skellam_win_prob(mu, total=6.0):
    """Vectorized P(home win incl. OT/SO) from expected margin `mu`."""
    mu = np.atleast_1d(np.asarray(mu, float))
    out = np.empty(len(mu))
    for i, m in enumerate(mu):
        lam_h = max((total + m) / 2.0, 0.05)
        lam_a = max((total - m) / 2.0, 0.05)
        ph, pa = _pois_pmf(lam_h), _pois_pmf(lam_a)
        joint = np.outer(ph, pa)
        p_home = np.triu(joint, 1).sum()          # home goals > away goals
        p_tie = np.trace(joint)                    # equal -> OT coin flip
        out[i] = p_home + 0.5 * p_tie
    return out

