"""A deliberately simple expected-goals model.

Logistic regression of goal ~ distance, angle, and shot type on UNBLOCKED
shot attempts (blocked shots have no true release point). Fit on the data you
pass in (train seasons only, if you care about leakage). This is a sanity-grade
xG for situational comparison, not a state-of-the-art model — the spec says
start simple and only upgrade if it earns it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_FEATURES = ["distance", "angle", "dist_angle"]


class XGModel:
    def __init__(self):
        self.coef_ = None
        self.mean_ = None
        self.std_ = None
        self.shot_type_rate_ = {}
        self.base_rate_ = 0.06

    @staticmethod
    def _design(df: pd.DataFrame) -> np.ndarray:
        d = df["distance"].to_numpy(float)
        a = df["angle"].to_numpy(float)
        # rebound / rush flags sharpen xG (higher goal prob); absent in old
        # data -> treated as 0 so the model degrades gracefully.
        reb = df["is_rebound"].fillna(False).to_numpy(float) \
            if "is_rebound" in df.columns else np.zeros(len(df))
        rush = df["is_rush"].fillna(False).to_numpy(float) \
            if "is_rush" in df.columns else np.zeros(len(df))
        return np.column_stack([d, a, d * a / 90.0, reb, rush])

    def fit(self, events: pd.DataFrame) -> "XGModel":
        df = events[events["is_unblocked"] & events["distance"].notna()].copy()
        df = df[df["distance"] <= 100]  # drop junk/empty-net-from-afar
        y = df["is_goal"].to_numpy(float)
        X = self._design(df)
        self.mean_ = X.mean(0)
        self.std_ = X.std(0)
        self.std_[self.std_ == 0] = 1.0
        Xs = (X - self.mean_) / self.std_
        Xs = np.column_stack([np.ones(len(Xs)), Xs])
        self.coef_ = _irls_logistic(Xs, y)
        self.base_rate_ = float(y.mean())
        # shot-type multiplier (shrunk toward 1)
        for st, g in df.groupby("shot_type"):
            n = len(g)
            rate = g["is_goal"].mean()
            self.shot_type_rate_[st] = (rate * n + self.base_rate_ * 20) / (n + 20)
        return self

    def predict(self, events: pd.DataFrame) -> np.ndarray:
        X = self._design(events.fillna({"distance": 100, "angle": 0}))
        Xs = (X - self.mean_) / self.std_
        Xs = np.column_stack([np.ones(len(Xs)), Xs])
        p = _sigmoid(Xs @ self.coef_)
        # shot-type adjustment
        mult = events["shot_type"].map(
            lambda s: (self.shot_type_rate_.get(s, self.base_rate_) / self.base_rate_)
            if self.base_rate_ > 0 else 1.0
        ).fillna(1.0).to_numpy()
        p = np.clip(p * mult, 1e-4, 0.99)
        # blocked shots don't reach the net: no xG credit
        p[~events["is_unblocked"].to_numpy()] = 0.0
        return p


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _irls_logistic(X, y, iters=25, ridge=1e-4):
    n, k = X.shape
    w = np.zeros(k)
    for _ in range(iters):
        p = _sigmoid(X @ w)
        W = p * (1 - p)
        W = np.clip(W, 1e-6, None)
        z = X @ w + (y - p) / W
        A = X.T @ (W[:, None] * X) + ridge * np.eye(k)
        w_new = np.linalg.solve(A, X.T @ (W * z))
        if np.max(np.abs(w_new - w)) < 1e-6:
            w = w_new
            break
        w = w_new
    return w


def add_xg(events: pd.DataFrame, model: XGModel | None = None) -> pd.DataFrame:
    """Attach an `xg` column; fits a model on `events` if none supplied."""
    model = model or XGModel().fit(events)
    out = events.copy()
    out["xg"] = model.predict(out)
    return out
